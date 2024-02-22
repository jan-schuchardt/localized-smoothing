"""This script is for training a segmentation model, which is then certfied in cert.py."""

import albumentations as albu
import torch
from sacred import Experiment

import seml
from localized_smoothing.segmentation.datasets import get_datasets
from localized_smoothing.segmentation.models import get_model
from localized_smoothing.segmentation.training import train
from localized_smoothing.segmentation.transformations import (
    get_input_transformation_train, get_joint_transformation_train,
    get_joint_transformation_val, get_smoothing_transformation)
from localized_smoothing.utils import set_seed

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite))

    dataset = {
        'dataset_name':
            'pascal',
        'data_folder':
            '/nfs/homedirs/schuchaj/programming/localized_smoothing/data',
        'scaling':
            1,  # Factor by which all images are scaled (smaller=lower res)
        'image_interpolation_method':
            'area',  # openCV2 flag for interpolation of input images
        'target_interpolation_method':
            'nearest'  # openCV2 flag for interpolation of ground truth
    }

    model = {
        # from [unet, unetplusplus, manet, linknet, fpn, pan, deeplabv3, deeplabv3plus, pspnet]
        'model_type': 'fpn',
        'model_params': {
            # See https://segmentation-modelspytorch.readthedocs.io/en/latest/#models
            'encoder_name': 'mobilenet_v2',
            'encoder_weights': 'imagenet',
            'in_channels': 3,
            'classes': 21
        }
    }

    transformations = {
        'random_brightness_params': {
            # Parameters from
            # https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomBrightness
            'p': 0.5
        },
        'random_contrast_params': {
            # Parameters from
            # https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomContrast
            'p': 0.5  # Probability of randomly changing contrast
        },
        'blur_params': {
            # Parameters from
            # https://albumentations.ai/docs/api_reference/augmentations/blur/transforms/#albumentations.augmentations.blur.transforms.Blur
            'p': 0.5,
            'blur_limit': 3
        },
        'smoothing_params': {  # See also LocalizedSmoothingTransform in localized_smoothing.segmentation.transformations
            'p': 1,  # Probability of randomly perturbing training sample
            'localized':
                False,  # Whether to use localized distribution (True) or isotropic noise
            'distribution_params': {
                'std': 0.1  # Isotropic smoothing standard deviation
            },
            'localized_params': {
                'mask_value': 255,  # Ignore label, in case label_mask_distance
                'label_mask_distance':
                    1,  # If not None: When sampling localized noise for cell (i, j), set targets
                # for cells (k, l) with larger distance to mask_value
                'distribution_params': {
                    'std_min': 0.1,  # Minimum smoothing standard deviation
                    'std_max': 1,  # Maximum smoothing standard deviation
                    'grid_shape': [
                        2, 4
                    ],  # Number of vertical and horizontal grid cells (see Fig. 1)
                    'metric': 'l_0',  # Distance function for grid cells
                    'interpolate_variance':
                        False,  # Whether to linearly interpolate stds or variances
                    'mask_distance':
                        None,  # If not None: When sampling localized noise for cell (i, j), set inputs
                    # for cells (k, l) with larger distance to 0
                    'max_std_at_boundary':
                        False  # If True: std(k,l)=max_std when (k, l) is farthest away cell from (i, j)
                    # If False: std(k,l)=max_std when distance to (k, l) is the largest
                    #     possible distance between two grid cells.
                }
            }
        },
        'flip_horizontally':
            True,  # Randomly flip image horizontally with 50% chance
        'shift_scale_rotate_params': {
            # Parameters for
            # https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.ShiftScaleRotate
            'shift_limit': 0.1,
            'scale_limit': [-0.5, 1],
            'rotate_limit': 10,
            'p': 1
        },
        'padding_params_train': {
            # Parameters for
            # https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.PadIfNeeded
            # Applied to training data
            'border_mode': 0,
            'value': 0,
            'mask_value': 255
        },
        'padding_params_val': {
            # Parameters for
            # https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.PadIfNeeded
            # Applied to validation data.
            'border_mode': 0,
            'value': 0,
            'mask_value': 255,
            'pad_height_divisor': 32,
            'pad_width_divisor': 32,
        },
        'random_cropping_params': {
            # Parameters for
            # https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomCrop
            # Applied to training data.
            'height': 256,
            'width': 320
        }
    }

    training_params = {
        'num_epochs': 256,
        'lr': 0.001,
        'batch_size': 64,
        'max_batch_size': None,
        'train_encoder': False,  # Also train model backbone.
        'epochs_val': 8,  # Interval at which validation is performed
        'stitch_smooth_predictions':
            False,  # Whether to stitch predictions together from n_cell_rows x n_cell_coll different
        # Monte Carlo samples (with a single sample), as in Fig. 1.
        'accumulate_gradients':
            False  # Whether to use virtual minibatching when stitch_smooth_predictions is True
        # Because we need to do n_cell_rows x n_cell_cols forward passes per prediction
    }

    seed = 0
    save_dir = '/home/jan/tmp/save'


get_model = ex.capture(get_model, prefix='model')

get_input_transformation_train = ex.capture(get_input_transformation_train,
                                            prefix='transformations')

get_smoothing_transformation = ex.capture(get_smoothing_transformation,
                                          prefix='transformations')

get_joint_transformation_train = ex.capture(get_joint_transformation_train,
                                            prefix='transformations')

get_joint_transformation_val = ex.capture(get_joint_transformation_val,
                                          prefix='transformations')

get_datasets = ex.capture(get_datasets, prefix='dataset')


@ex.automain
def main(_config, seed, save_dir, training_params, transformations):

    set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model, input_mean, input_std = get_model()
    model.to(device)

    trans_input = get_input_transformation_train()
    trans_smooth = get_smoothing_transformation()
    trans_joint_train = get_joint_transformation_train()
    trans_joint_val = get_joint_transformation_val()
    normalization = albu.Normalize(mean=input_mean,
                                   std=input_std,
                                   max_pixel_value=1.0)

    if training_params['stitch_smooth_predictions']:
        trans_smooth = None
        normalization = None
        smoothing_params = transformations['smoothing_params']
        normalization_params = {
            'mean': torch.Tensor(input_mean),
            'std': torch.Tensor(input_std)
        }
    else:
        smoothing_params = None
        normalization_params = None

    data_train, data_val = get_datasets(trans_input=trans_input,
                                        trans_smooth=trans_smooth,
                                        trans_joint_train=trans_joint_train,
                                        trans_joint_val=trans_joint_val,
                                        normalization=normalization)

    best_state_loss, best_state_iou, losses_train, losses_val, ious_val = train(
        model,
        data_train,
        data_val,
        device=device,
        smoothing_params=smoothing_params,
        normalization_params=normalization_params,
        **training_params)

    run_id = _config['overwrite']
    db_collection = _config['db_collection']

    dict_to_save = {
        'losses_train': losses_train,
        'losses_val': losses_val,
        'ious_val': ious_val,
        'state_dict_best_loss': best_state_loss,
        'state_dict_best_iou': best_state_iou,
        'config': _config
    }

    torch.save(dict_to_save, f'{save_dir}/{db_collection}_{run_id}')

    return {
        'best_loss_train': losses_train.mean(axis=1).min(),
        'best_loss_val': losses_val.min(),
        'best_iou_val': ious_val.mean(axis=1).max(),
    }
