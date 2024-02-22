"""This script is for certifying a model that has been trained in train.py"""

import numpy as np
import torch
from sacred import Experiment

import seml
from localized_smoothing.segmentation.certification import certify_dataset
from localized_smoothing.segmentation.datasets import (
    AugmentedCityscapesSegmentation, AugmentedVOCSegmentation, get_datasets)
from localized_smoothing.segmentation.models import get_model
from localized_smoothing.segmentation.transformations import \
    get_joint_transformation_val
from localized_smoothing.utils import dict_to_dot, set_seed

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

    cert_dir = '/home/jan/tmp/save'
    seed = 0
    calc_confusion = True

    dataset = {
        'dataset_name':
            'pascal',
        'data_folder':
            '/nfs/homedirs/schuchaj/programming/localized_smoothing/data',
        'scaling':
            1,
        'image_interpolation_method':
            'area',
        'target_interpolation_method':
            'nearest'
    }

    train_loading = {
        'collection': None,  # Database for train experiments
        'exp_id':
            None,  # Experiment id to load. Otherwise, load based on restrictions.
        'restrictions':
            None,  # Restrictions (mongodb query) for loading when exp_id not specified
        'load_best_iou':
            True,  # Load model with highest val iou (true) or val accuracy (false)
        'find_std_min':
            False,  # Add distribution_params[std_min] to restrictions
        'find_std_max':
            False,  # Add distribution_params[std_max] to restrictions
        'find_localized_distribution':
            False  # Add all of distribution_params to restrictions
    }

    padding_params = {
        # Parameters for
        # https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.PadIfNeeded
        'border_mode': 0,
        'value': 0,
        'mask_value': 255,
        'pad_height_divisor': 32,
        'pad_width_divisor': 32
    }

    sample_params = {
        'batch_size': 64,  # Number of MC samples to segment in a single pass.
        'n_samples_pred':
            128,  # Number of samples for making smoothed prediction.
        'n_samples_cert': 256,  # Number of samples for certification.
        'upsampling_factor_pred':
            1,  # For debugging. If N, take 1/Nth of n_samples_pred, act is if we had n_samples_pred.
        'upsampling_factor_cert':
            1  # For debugging. If N, take 1/Nth of n_samples_cert, act is if we had n_samples_cert.
    }

    distribution_params = {
        'std_min': 0.1,  # Minimum smoothing standard deviation
        'std_max': 1,  # Maximum smoothing standard deviation
        'grid_shape':
            [2, 4],  # Number of vertical and horizontal grid cells (see Fig. 1)
        'metric': 'l_0',  # Distance function for grid cells
        'interpolate_variance':
            False,  # Whether to linearly interpolate stds or variances
        'mask_distance':
            None,  # If not None: When sampling localized noise for cell (i, j), set inputs
        # for cells (k, l) with larger distance to 0
        'max_std_at_boundary':
            False  # If True: std(k,l)=max_std when (k, l) is farthest away cell from (i, j)
        # If False: std(k,l)=max_std when distance to (k, l) is the largest
        # possible distance between two grid cells.
    }

    certification_params = {
        'n_images':
            50,  # Number of images from validation set to verify (always take first N)
        'budget_min': 0,  # Minimum adversarial l2 budget
        'budget_max': 5,  # Maximum adversarial l2 budget
        'budget_steps': 10,  # Number of linspace steps between them
        'alpha':
            0.01,  # Significance, i.e. 1-alpha is probability that certificate holds.
        'eps': None,
        'delta':
            0.05,  # Hyperparameter from center smoothing, see https://arxiv.org/abs/2102.09701
        'naive_certs': [
            'argmax_bonferroni', 'argmax_holm'
        ],  # Base certificates to use for computing naive collective certificate.
        # ['argmax_holm'], ['argmax_bonferroni']
        # Can also be set to
        # ['center_independent'], ['center_bonferroni'] for center smoothing.
        # we did not rename this argument for backwards compatbility.
        'base_certs': [
            'argmax_bonferroni', 'argmax_holm'
        ],  # # Base certificates to use for computing the LP-based collective certificate. 
        # ['argmax_holm'], ['argmax_bonferroni']
        'n_max_rad_bins':
            None  # Number of quantization bins for maximum certifiable radii, see Appendix E.2.
    }


@ex.capture(prefix='train_loading')
def load_train_data(collection,
                    exp_id,
                    restrictions,
                    find_std_min=False,
                    find_std_max=False,
                    find_localized_distribution=False,
                    distribution_params=None):

    if exp_id is None and restrictions is None:
        raise ValueError(
            'You must provide either an exp-id or a restriction dict')
    if collection is None:
        raise ValueError('You must a collection to load trained model from')

    mongodb_config = seml.database.get_mongodb_config()
    coll = seml.database.get_collection(collection, mongodb_config)

    if exp_id is not None:
        train_config = coll.find_one({'_id': exp_id}, ['config'])['config']
    else:
        coll_filter = restrictions.copy()
        coll_filter = {'config.' + k: v for k, v in dict_to_dot(coll_filter)}

        if int(find_std_min) + int(find_std_max) + int(
                find_localized_distribution) > 1:
            raise ValueError(
                'Can only search by std_min, std_max or localized distribution')

        smoothing_prefix = 'config.transformations.smoothing_params'

        # change to find some higher value
        if find_std_min:
            coll_filter[f'{smoothing_prefix}.localized'] = False
            coll_filter[
                f'{smoothing_prefix}.distribution_params.std'] = distribution_params[
                    'std_min']

        elif find_std_max:
            coll_filter[f'{smoothing_prefix}.localized'] = False
            coll_filter[
                'config.transformations.smoothing_params.std'] = distribution_params[
                    'std_max']

        elif find_localized_distribution:
            coll_filter[f'{smoothing_prefix}.localized'] = True
            for k, v in dict_to_dot(distribution_params):
                coll_filter[
                    f'{smoothing_prefix}.localized_params.distribution_params.{k}'] = v

        exps = list(coll.find(coll_filter, ['config']))
        if len(exps) == 0:
            raise ValueError(f"Find yielded no results.")
        elif len(exps) > 1:
            raise ValueError(f"Find yielded more than one result: {exps}")
        else:
            exp_id = exps[0]['_id']
            train_config = exps[0]['config']

    return train_config, exp_id


get_datasets = ex.capture(get_datasets, prefix='dataset')


def get_cert_img_idx(dataset):
    img_idx = []

    if isinstance(dataset, AugmentedVOCSegmentation):
        for i, x in enumerate(dataset):
            if np.all(x[2].numpy() == np.array([166, 250])):
                img_idx.append(i)

    elif isinstance(dataset, AugmentedCityscapesSegmentation):
        for i, x in enumerate(dataset):
            assert np.all(x[2].numpy() == np.array([512, 1024]))

        img_idx = list(range(0, 500, 5))

    else:
        raise ValueError('Unsupported dataset')

    return img_idx


def get_cert_original_shape(dataset):
    if isinstance(dataset, AugmentedVOCSegmentation):
        original_shape = np.array([166, 250])

    elif isinstance(dataset, AugmentedCityscapesSegmentation):
        original_shape = np.array([512, 1024])

    else:
        raise ValueError('Unsupported dataset')

    return original_shape


@ex.automain
def main(_config, seed, padding_params, train_loading, cert_dir, sample_params,
         distribution_params, certification_params):
    set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train_config, train_exp_id = load_train_data(
        distribution_params=distribution_params)

    model_config = train_config['model']

    train_file = torch.load(f'{train_config["save_dir"]}'
                            f'/{train_loading["collection"]}_{train_exp_id}')

    model, input_mean, input_std = get_model(model_config['model_type'],
                                             model_config['model_params'])

    model.to(device)
    if train_loading['load_best_iou']:
        model.load_state_dict(train_file['state_dict_best_iou'])
    else:
        model.load_state_dict(train_file['state_dict_best_loss'])

    trans_joint = get_joint_transformation_val(padding_params)
    # TODO: Test set
    _, data_val = get_datasets(trans_joint_val=trans_joint)

    # We don't normalize data directly, only after smoothing
    normalization_params = {
        'mean': torch.Tensor(input_mean),
        'std': torch.Tensor(input_std)
    }

    run_id = _config['overwrite']
    cert_file = f'{cert_dir}/{train_loading["collection"]}_{train_exp_id}_{run_id}'

    img_idx = get_cert_img_idx(data_val)
    original_shape = get_cert_original_shape(data_val)

    cert_dict = certify_dataset(data_val, img_idx, original_shape, model,
                                normalization_params, sample_params,
                                distribution_params, certification_params,
                                device, cert_file)

    cert_dict['config'] = _config

    torch.save(cert_dict, cert_file)

    return {'cert_file': cert_file}
