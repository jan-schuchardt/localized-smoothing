"""This module provides albumentations transformations for processing of images and
segmentation masks at training and segmentation time.


LocalizedSmoothingTransform: Applies localized noise to a random grid cell of an image, target pair.

get_input_transformation_train: Yields image transformations for training images.
get_joint_transformation_train: Yields geometric transformations for training images and targets.

get_smoothing_transformation: Yields transformation that samples images from smoothing distribution.

get_joint_transformation_val: Yields geometric transformations for validation images and targets. 
"""

from typing import Any

import albumentations as albu
import numpy as np
from albumentations import DualTransform
from numpy.typing import NDArray

from localized_smoothing.segmentation.utils import (generate_distance_mask,
                                                    generate_grid,
                                                    generate_smoothing_stds)


class LocalizedSmoothingTransform(DualTransform):
    """Transformation applying localized noise to a random grid cell of an image, target pair.

    The sampling procedure is as follows:
    1.) Pick grid (i, j) cell uniformly at random.
    2.) For pixels in grid_cell (k, l), add random Gaussian noise with distance-dependent
        standard deviation between std_min and std_max.
    3.) Clip pixel values between 0 and 1.
    4.) Optionally: For all grid cells (k, l) that have large distance from (i, j),
        set pixel values to 0.
    5.) Optionally: For all grid cells (k, l) that have large distance from (i, j),
        set target values to some mask value.

    Attributes:
        grid_shape: An array-like (I, J) with the vertical and horizontal grid resolution.
        std_min: The minimum smoothing standard deviation, which is applied at (i, j).
        std_max: The maximum smoothing standard deviation.
        metric: The notion of distance between grid cells.
            Currently only supports `l_0`, which is actually l_infty, i.e. max( |k - i|, |l - j|)
        interpolate_variance: Whether to linearly interpolate variance or standard deviation.
            If False: Linearly interpolate between std_min and std_max.
            If True: Linearly interpolate between std_min**2 and std_max**2, then take square root.
        mask_distance: The minimum grid cell distance at which the image pixels are masked.
            If it is None, no image masking is performed.
        label_mask_distance: The minimum grid cell distance at which the target pixels are masked.
            If it is None, no target masking is performed.
        max_std_at_boundary: Determines at which distance interpolation yields std_max.
            If True: Yields std_max when d((i, j), (k, l)) = max_{0<=m<=I, 0<=n<=J} d((i, j),(m, n))
            If False: Yields std_max when d((i, j), (k, l))
        mask_value: Value to use for masking for-away target cells.
        always_apply: Whether (image, target) pair is always transformed.
        p: Probability of transforming (image, target) pair when always_apply=False
    """

    def __init__(self,
                 grid_shape: np.ndarray,
                 std_min: float,
                 std_max: float,
                 metric: str = 'l_0',
                 interpolate_variance: bool = False,
                 mask_distance: int | None = None,
                 label_mask_distance: int | None = 1,
                 max_std_at_boundary: bool = False,
                 mask_value: int = 255,
                 always_apply: bool = False,
                 p: bool = 0.5) -> None:
        """Stores all the passed attributes as self.
        """

        super().__init__(always_apply, p)

        self.grid_shape = np.array(grid_shape)
        self.std_min = std_min
        self.std_max = std_max

        self.metric = metric
        self.interpolate_variance = interpolate_variance
        self.mask_distance = mask_distance
        self.label_mask_distance = label_mask_distance
        self.max_std_at_boundary = max_std_at_boundary
        self.mask_value = mask_value

    def apply(self, img: np.ndarray, **params: dict[str, Any]) -> np.ndarray:
        """Applies sampling procedure describe above to an input image.

        Grid_cell (i, j) is not sampled here but received through params argument
        to ensure consistency with the transformation applied to the segmentation targets
        in apply_to_mask method.

        Args:
            img: Input image of shape HxWxC
            params: Dictionary of parameters, automatically genererated
                by get_params and get_params_dependent_on_targets below.

        Returns:
            Transformed image of shape HxWxC.
        """
        assert np.all((img >= 0) & (img <= 1))

        stds = generate_smoothing_stds(params['i'], params['j'], img.shape[:2],
                                       params['grid_vertical'],
                                       params['grid_horizontal'], self.std_min,
                                       self.std_max, self.metric,
                                       self.interpolate_variance,
                                       self.max_std_at_boundary)

        if self.mask_distance is not None:
            distance_mask = generate_distance_mask(params['i'],
                                                   params['j'],
                                                   self.mask_distance,
                                                   img.shape[:2],
                                                   params['grid_vertical'],
                                                   params['grid_horizontal'],
                                                   metric=self.metric)
        else:
            distance_mask = None

        return LocalizedSmoothingTransform._sample(img, stds, distance_mask)

    def apply_to_mask(self, img: np.ndarray, **params: dict[str,
                                                            Any]) -> np.ndarray:
        """Applies sampling procedure describe above to targets, i.e. ground-truth segmentation.

        Grid_cell (i, j) is not sampled here but received through params argument
        to ensure consistency with the transformation applied to the input image in
        "apply" method.

        Args:
            img: Targets of shape HxW.
            params: Dictionary of parameters, automatically genererated
                by get_params and get_params_dependent_on_targets below.

        Returns:
            Transformed targets of shape HxW.
        """
        if self.label_mask_distance is not None:
            distance_mask = generate_distance_mask(params['i'],
                                                   params['j'],
                                                   self.label_mask_distance,
                                                   img.shape[:2],
                                                   params['grid_vertical'],
                                                   params['grid_horizontal'],
                                                   metric=self.metric)

            img = img.copy()
            img[~distance_mask] = self.mask_value

        return img

    def get_params(self) -> dict[str, int]:
        """Randomly samples grid cell from [I] x [J], which is automatically used in apply function.

        This overrides a default albumentations.Transformation method.

        Returns:
            A dict, with key 'i' containing vertical grid cell coordinate
            and 'j' containing horizontal grid cell coordinate.
        """
        return {
            'i': np.random.choice(np.arange(self.grid_shape[0])),
            'j': np.random.choice(np.arange(self.grid_shape[1]))
        }

    def get_params_dependent_on_targets(
            self, params: dict[str, Any]) -> dict[str, np.ndarray]:
        """Generates grid cell boundary coordiantes that are automatically used in apply function.

        This overrides a default albumentations.Transformation method.

        Returns:
            A dict with two key-value pairs:
                grid_vertical`: An array A with shape 2xI, where A[0,i] is
                    vertical starting coordinate of grid row i
                    and A[1, i] is vertical ending coordinate of grid row i.
                grid_horizontal`: An array B with shape 2xJ, where A[0,j] is
                    horizontal starting coordinate of grid column j
                    and A[1, j] is horizontal ending coordinate of grid column j.
        """
        img = params['image']

        grid_vertical, grid_horizontal = generate_grid(img.shape[:2],
                                                       self.grid_shape)

        return {
            'grid_vertical': grid_vertical,
            'grid_horizontal': grid_horizontal
        }

    @staticmethod
    def _sample(img: np.ndarray,
                stds: np.ndarray,
                distance_mask: NDArray[np.bool_] | None = None) -> np.ndarray:
        """Adds Gaussian noise to input image, clips to [0, 1] and finally masks specified pixels.

        Args:
            img: Input image of shape HxWxC.
            stds: Gaussian standard deviations of shape HxW.
            distance_mask: Boolean array. Pixels where False are set to 0.

        Returns:
            Modified image of shape HxWxC.
        """
        assert img.ndim == 3  # Last dim = channel

        # Sample and transpose to 3 x height x width
        noisy_image = np.random.normal(img.transpose((2, 0, 1)), stds)

        if distance_mask is not None:
            noisy_image[:, ~distance_mask] = 0

        return np.clip(noisy_image, 0, 1).transpose((1, 2, 0))

    @property
    def targets_as_params(self):
        return ["image"]


def get_input_transformation_train(random_brightness_params: dict[str, Any],
                                   random_contrast_params: dict[str, Any],
                                   blur_params: dict[str, Any]) -> albu.Compose:
    """Creates image transformation consisting of brightness changes, contrast changes and blur.

    Args:
        model_type: Parameters for RandomBrightess, see

        random_brightness_params: Parameters to be passed to albu.RandomBrightness
            see https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomBrightness
        random_contrast_params: Parameters to be passed to albu.RandomContrast
            see https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomContrast
        blur_params: Parameters to be passed to albu.Blur
            see https://albumentations.ai/docs/api_reference/augmentations/blur/transforms/#albumentations.augmentations.blur.transforms.Blur

    Returns:
        Composition of RandomBrightness, RandomContrast and Blur transformations.
    """

    trans_input = [
        albu.RandomBrightness(**random_brightness_params),
        albu.RandomContrast(**random_contrast_params),
        albu.Blur(**blur_params)
    ]

    return albu.Compose(trans_input)


def get_joint_transformation_train(
        flip_horizontally: bool, shift_scale_rotate_params: dict[str, Any],
        padding_params_train: dict[str, int],
        random_cropping_params: dict[str, Any]) -> albu.Compose:
    """Creates geometric transformations that are jointly applied to training images and targets.

    We
    1.) shift scale and rotate
    2.) Pad so that image and target are always large enough for subsequent cropping
    3.) Perform random cropping to some fixed size with random center.
    4.) Randomly flip horizontally.

    Args:
        flip_horizontally: If True, randomly flip image and target with 50% probability.
        shift_scale_rotate_params: Parameters to be passed to albu.ShiftScaleRotate
            see https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.ShiftScaleRotate
        padding_params_train: Dict specifying "border_mode", "value" and "mask_value" for albumentations padding.
            see https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.PadIfNeeded
        random_cropping_params: Parameters to be passed to albu.RandomCrop
            see https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomCrop

    Returns:
        Composition of ShiftScaleRotate, Padding, Random Cropping and horizontal Flipping.
    """
    trans_joint = [
        albu.ShiftScaleRotate(**shift_scale_rotate_params,
                              **padding_params_train),
        albu.PadIfNeeded(min_height=random_cropping_params['height'],
                         min_width=random_cropping_params['width'],
                         **padding_params_train,
                         always_apply=True),
        albu.RandomCrop(**random_cropping_params, always_apply=True)
    ]

    if flip_horizontally:
        trans_joint.insert(0, albu.HorizontalFlip(p=0.5))

    return albu.Compose(trans_joint)


def get_joint_transformation_val(
        padding_params_val: dict[str, int]) -> DualTransform:
    """Creates geometric transformations that are jointly applied to validation images and targets.

    Currently, this is just padding so that height and width are divisible by some number.

    Args:
        padding_params_val: Dict specifying
            "border_mode", "value",  "mask_value", "pad_height_divisor" and "pad_width_divisor"
            see https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.PadIfNeeded

    Returns:
        PadIfNeeded transformation.
    """
    return albu.PadIfNeeded(**padding_params_val,
                            min_height=None,
                            min_width=None,
                            always_apply=True)


def get_smoothing_transformation(
        smoothing_params: dict
) -> LocalizedSmoothingTransform | albu.GaussNoise:
    """Prepare albumentations transformation that perturbs data using (localized) Gaussian noise.

    For an explanation of the parameters, see the comments in the seml/scripts/segmentation/*.py files.

    Args:
        model_type: Parameters to be passed to LocalizedSmoothingTransform or GaussNoise

    Returns:
        The transformation.
    """
    if not smoothing_params['p'] > 0:
        return None

    if smoothing_params['localized']:
        return LocalizedSmoothingTransform(
            p=smoothing_params['p'],
            mask_value=smoothing_params['localized_params']['mask_value'],
            label_mask_distance=smoothing_params['localized_params']
            ['label_mask_distance'],
            **smoothing_params['localized_params']['distribution_params'])

    else:
        var_limit = list(
            np.power(2 * [smoothing_params['distribution_params']['std']], 2))
        return albu.GaussNoise(var_limit=var_limit, p=smoothing_params['p'])
