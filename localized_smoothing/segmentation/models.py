"""This module contains a method for obtaining a segmentation model."""

from operator import itemgetter

import segmentation_models_pytorch as smp


def get_model(
    model_type: str, model_params: dict
) -> tuple[smp.base.model.SegmentationModel, float, float]:
    """Creates pytorch SMP model of specified type.

    Args:
        model_type: Model name from
                    [unet, unetplusplus, manet, linknet, fpn, pan, deeplabv3, deeplabv3plus, pspnet]
        y: Parameters for chosen model type,
           see https://segmentation-modelspytorch.readthedocs.io/en/latest/#models

    Returns:
        Model, as well as mean and standard deviation used for z-normalization during pretraining.
    """
    model_type_dict = {
        'unet': smp.Unet,
        'unetplusplus': smp.UnetPlusPlus,
        'manet': smp.MAnet,
        'linknet': smp.Linknet,
        'fpn': smp.FPN,
        'pan': smp.PAN,
        'deeplabv3': smp.DeepLabV3,
        'deeplabv3plus': smp.DeepLabV3Plus,
        'pspnet': smp.PSPNet
    }

    model_type = model_type.lower()
    if model_type.lower() not in model_type_dict:
        raise NotImplementedError(f'Model type \"{model_type}\" not supported')
    model_class = model_type_dict[model_type]
    model = model_class(**model_params)
    model.classes = model_params['classes']

    if 'encoder_weights' in model_params:
        preprocessing_params = smp.encoders.get_preprocessing_params(
            model_params['encoder_name'], model_params['encoder_weights'])

        assert preprocessing_params['input_space'] == 'RGB'
        assert preprocessing_params['input_range'] == [0, 1]

        input_mean, input_std = itemgetter('mean', 'std')(preprocessing_params)

    else:
        input_mean, input_std = 0, 1

    return model, input_mean, input_std
