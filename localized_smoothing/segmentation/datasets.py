"""This module contains datasets and modifications thereof that apply image augmentations.

The Augmented datasets are designed be used with the transformations defined in .transformations.py.

_VOCBase: Copy of old torchvision PasCalVOC code that can also load trainaug data.
VOCSegmentation: Copy of old torchvision VOCSegmentation code that can also load trainaug data.

AugmentedVOCSegmentation: VOC dataset that applies augmentations when accessing dataset element.
AugmentedCityscapesSegmentation: Cityscapes dataset that applies augmentations
    when accessing dataset element.

get_datasets: Creates training and validation datasets with specified transformations.
"""

import os
import warnings
from typing import Any, Callable, List, Optional, Tuple

import albumentations as albu
import cv2
import numpy as np
import torch
from albumentations import DualTransform, ImageOnlyTransform
from albumentations.augmentations.geometric.functional import scale
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
from torchvision.datasets.utils import (download_and_extract_archive,
                                        verify_str_arg)
from torchvision.datasets.vision import VisionDataset

from localized_smoothing.segmentation.transformations import \
    LocalizedSmoothingTransform


def get_datasets(
        dataset_name: str,
        data_folder: str,
        trans_input: ImageOnlyTransform | albu.Compose | None = None,
        trans_smooth: LocalizedSmoothingTransform | albu.GaussNoise |
    None = None,
        trans_joint_train: DualTransform | albu.Compose | None = None,
        trans_joint_val: DualTransform | albu.Compose | None = None,
        normalization: albu.Normalize | None = None,
        train_set: str = 'trainaug',
        val_set: str = 'val',
        download: bool = False,
        scaling: float = 1,
        image_interpolation_method: str = 'linear',
        target_interpolation_method: str = 'nearest'
) -> tuple[Dataset, Dataset]:
    """
    Creates training and validation datasets with specified transformations.

    trans_input and trans_joint_train are only applied to the training dataset.
    trans_joint_val is only aplied to the validation dataset.

    Args:
        dataset_name: Name of the dataset, must be in ["pascal", "cityscapes"]
        data_folder: Folder to store the dataset / load the dataset from
        trans_input: Albumentations transformations that are applied to training input images.
        trans_smooth: Noising transformation applied to training and validation data.
        trans_joint_train: Joint transformation of training images and targets.
        trans_joint_val: Joint transformation of validation images and targets.
        normalization: Normalization transformation applied to training and validation input images.
        train_set: Which subset to use as training data,
            see AugmentedVOCSegmentation and AugmentedCityscapesSegmentation.
        val_set: Which subset to use as validation data,
            see AugmentedVOCSegmentation and AugmentedCityscapesSegmentation.
        download: Whether to download dataset if it is not stored in data_folder yet.
        scaling: Factor by which images should be scaled, larger means more pixels.
        image_interpolation_method: OpenCV2 image interpolation flag for input image.
        target_interpolation_method: OpenCV2 image interpolation flag for targets.

    Returns:
        Tuple, with first element being training dataset and second element being training dataset.
    """

    if dataset_name.lower() not in ['pascal', 'cityscapes']:
        raise NotImplementedError(f'Dataset \"{dataset_name}\" not supported.')

    if dataset_name.lower() == 'pascal':
        data_folder = os.path.join(
            data_folder, 'pascal_voc')  # TODO: Just directly specify folder

        data_train = AugmentedVOCSegmentation(
            data_folder,
            year='2012',
            image_set=train_set,
            scaling=scaling,
            input_transform=trans_input,
            joint_transform=trans_joint_train,
            smoothing_transform=trans_smooth,
            normalization=normalization,
            download=download,
            image_interpolation_method=image_interpolation_method,
            target_interpolation_method=target_interpolation_method)

        data_val = AugmentedVOCSegmentation(
            data_folder,
            year='2012',
            image_set=val_set,
            scaling=scaling,
            joint_transform=trans_joint_val,
            smoothing_transform=trans_smooth,
            normalization=normalization,
            return_original_shape=True,
            download=download,
            image_interpolation_method=image_interpolation_method,
            target_interpolation_method=target_interpolation_method)

    if dataset_name.lower() == 'cityscapes':

        data_train = AugmentedCityscapesSegmentation(
            data_folder,
            split=train_set,
            mode='fine',
            scaling=scaling,
            input_transform=trans_input,
            joint_transform=trans_joint_train,
            smoothing_transform=trans_smooth,
            normalization=normalization,
            return_original_shape=False,
            image_interpolation_method=image_interpolation_method,
            target_interpolation_method=target_interpolation_method)

        data_val = AugmentedCityscapesSegmentation(
            data_folder,
            split=val_set,
            mode='fine',
            scaling=scaling,
            joint_transform=trans_joint_val,
            smoothing_transform=trans_smooth,
            normalization=normalization,
            return_original_shape=True,
            image_interpolation_method=image_interpolation_method,
            target_interpolation_method=target_interpolation_method)

    return data_train, data_val


class _VOCBase(VisionDataset):
    """This is a copy of old torchvision PasCalVOC code that can also load trainaug data.

    There was some reason why we could not simply inherit from the torchvision VOCBaseclass,
    but I do not remember.
    """
    DATASET_YEAR_DICT = {
        '2012': {
            'url':
                'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
            'filename':
                'VOCtrainval_11-May-2012.tar',
            'md5':
                '6cd6e144f989b92b3379bac3b3de84fd',
            'base_dir':
                os.path.join('VOCdevkit', 'VOC2012')
        },
        '2011': {
            'url':
                'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
            'filename':
                'VOCtrainval_25-May-2011.tar',
            'md5':
                '6c3384ef61512963050cb5d687e5bf1e',
            'base_dir':
                os.path.join('TrainVal', 'VOCdevkit', 'VOC2011')
        },
        '2010': {
            'url':
                'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
            'filename':
                'VOCtrainval_03-May-2010.tar',
            'md5':
                'da459979d0c395079b5c75ee67908abb',
            'base_dir':
                os.path.join('VOCdevkit', 'VOC2010')
        },
        '2009': {
            'url':
                'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
            'filename':
                'VOCtrainval_11-May-2009.tar',
            'md5':
                '59065e4b188729180974ef6572f6a212',
            'base_dir':
                os.path.join('VOCdevkit', 'VOC2009')
        },
        '2008': {
            'url':
                'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
            'filename':
                'VOCtrainval_11-May-2012.tar',
            'md5':
                '2629fa636546599198acfcfbfcf1904a',
            'base_dir':
                os.path.join('VOCdevkit', 'VOC2008')
        },
        '2007': {
            'url':
                'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
            'filename':
                'VOCtrainval_06-Nov-2007.tar',
            'md5':
                'c52e279531787c972589f7e41ab4ae64',
            'base_dir':
                os.path.join('VOCdevkit', 'VOC2007')
        },
        '2007-test': {
            'url':
                'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
            'filename':
                'VOCtest_06-Nov-2007.tar',
            'md5':
                'b6e924de25625d8de591ea690078ad9f',
            'base_dir':
                os.path.join('VOCdevkit', 'VOC2007')
        }
    }

    _SPLITS_DIR: str
    _TARGET_DIR: str
    _TARGET_FILE_EXT: str

    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        if year == "2007-test":
            if image_set == "test":
                warnings.warn(
                    "Acessing the test image set of the year 2007 with year='2007-test' is deprecated. "
                    "Please use the combination year='2007' and image_set='test' instead."
                )
                year = "2007"
            else:
                raise ValueError(
                    "In the test image set of the year 2007 only image_set='test' is allowed. "
                    "For all other image sets use year='2007' instead.")
        self.year = year

        valid_image_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_image_sets.append("test")
        if year == "2012":
            valid_image_sets.extend(["trainaug", "test"])
        self.image_set = verify_str_arg(image_set, "image_set",
                                        valid_image_sets)

        key = "2007-test" if year == "2007" and image_set == "test" else year
        dataset_year_dict = self.DATASET_YEAR_DICT[key]

        self.url = dataset_year_dict["url"]
        self.filename = dataset_year_dict["filename"]
        self.md5 = dataset_year_dict["md5"]

        base_dir = dataset_year_dict["base_dir"]
        voc_root = os.path.join(self.root, base_dir)

        if download:
            download_and_extract_archive(self.url,
                                         self.root,
                                         filename=self.filename,
                                         md5=self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        if not (self.year == '2012' and self.image_set == 'test'):
            if self.image_set == 'trainaug':
                target_dir = os.path.join(voc_root, "SegmentationClassAug")
            else:
                target_dir = os.path.join(voc_root, self._TARGET_DIR)
            self.targets = [
                os.path.join(target_dir, x + self._TARGET_FILE_EXT)
                for x in file_names
            ]

            assert len(self.images) == len(self.targets)

    def __len__(self) -> int:
        return len(self.images)


class VOCSegmentation(_VOCBase):
    """This is a copy of old torchvision VOCSegmentation code that can also load trainaug data.

    There was some reason why we could not simply inherit from the torchvision VOCSegmentation class,
    but I forgor.
    """

    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"

    _PALETTE = [
        0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0,
        128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0,
        64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64,
        0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128,
        192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128,
        192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128,
        64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192,
        64, 0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0,
        192, 64, 128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64,
        128, 192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64,
        64, 64, 192, 64, 64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64,
        192, 64, 192, 192, 192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160,
        128, 0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0,
        224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128,
        128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32,
        64, 128, 160, 64, 128, 32, 192, 128, 160, 192, 128, 96, 64, 0, 224, 64,
        0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128, 96, 192, 128,
        224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0,
        192, 160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64,
        96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128, 192, 224,
        128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64,
        192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64, 64,
        96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192, 96, 192, 192, 224,
        192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128, 128,
        32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0,
        192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160, 128, 0,
        96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0,
        224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0,
        64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128,
        32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192,
        128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64,
        32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96,
        64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192,
        128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64,
        96, 192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32,
        0, 32, 160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128,
        160, 160, 128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32,
        128, 224, 32, 128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0,
        32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160,
        224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128,
        224, 96, 128, 96, 224, 128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32,
        160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192, 160,
        160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32,
        192, 224, 32, 192, 96, 160, 192, 224, 160, 192, 32, 96, 64, 160, 96, 64,
        32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192, 160,
        224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96,
        192, 224, 96, 192, 96, 224, 192, 224, 224, 192
    ]

    @property
    def masks(self) -> List[str]:
        return self.targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation. Only image for 2012 test set
        """
        img = Image.open(self.images[index]).convert("RGB")

        if self.image_set == 'trainaug':
            target = Image.open(self.masks[index])
            target = target.convert('P')
            target.putpalette(self._PALETTE)
        elif self.year == '2012' and self.image_set == 'test':
            target = None
        else:
            target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if target is None:
            return img
        else:
            return img, target


class AugmentedVOCSegmentation(VOCSegmentation):
    """VOCSegmentation dataset that applies different augmentations to the images and/or targets.

    Transformations are applied in the following order:
    1.) Scaling
    2.) Non-geometric transformations of the input image.
    3.) Geometric transformations that affect the image and target.
    4.) Adding sampled noise from smoothing distribution
    5.) Normalization of the input color values.

    All transformations can be skipped by setting the respective parameter to "None".
    """

    image_interpolation_flags = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'linear_exact': cv2.INTER_LINEAR_EXACT,
        'nearest_exact': cv2.INTER_NEAREST_EXACT
    }

    target_interpolation_flags = {
        'nearest': cv2.INTER_NEAREST,
        'nearest_exact': cv2.INTER_NEAREST_EXACT
    }

    def __init__(self,
                 root: str,
                 year: str = '2012',
                 image_set: str = 'train',
                 download: bool = False,
                 scaling: float = 1,
                 input_transform: albu.ImageOnlyTransform | albu.Compose = None,
                 joint_transform: albu.DualTransform | albu.Compose = None,
                 smoothing_transform: albu.ImageOnlyTransform = None,
                 normalization: albu.Normalize = None,
                 return_original_shape: bool = False,
                 image_interpolation_method: str = 'area',
                 target_interpolation_method: str = 'nearest') -> None:
        """Calls super constructor. Stores additional parameters as attributes.

        Args:
            root: Root directory of the VOC Dataset.
            year: The dataset year in ["2007", "2012"]
            image_set: The data subset to use.
                "train", "trainval", "val" or "trainaug".
            download: If true, downloads the dataset (does not work for trainaug)
            scaling: Scaling factor (1 = original shape)
            input_transform: Non-geometric transformation of input image.
            joint_transform: Geometric transformations that affect the image and target.
            smoothing_transform: Adding image noise from smoothing distribution.
            normalization: Normalization of the input color values.
            return_original_shape: If True, each datapoint includes the original image shape
                (after scaling, before the remaining transformations).
            image_interpolation_method: openCV flag for interpolation of color values for
                geometric transformations (see image_interpolation_flags)
            target_interpolation_method: openCV flag for interpolation of target classes for
                geometric transformations (see image_interpolation_flags)
        """

        super().__init__(root,
                         year=year,
                         image_set=image_set,
                         download=download)

        self.joint_transform = joint_transform
        self.input_transform = input_transform
        self.smoothing_transform = smoothing_transform
        self.normalization = normalization
        self.return_original_shape = return_original_shape

        self.scaling = scaling
        if image_interpolation_method.lower(
        ) not in self.image_interpolation_flags.keys():
            raise ValueError(
                'Img interpolation method {image_interpolation_method} not supported'
            )
        if target_interpolation_method.lower(
        ) not in self.target_interpolation_flags.keys():
            raise ValueError(
                'Img interpolation method {image_interpolation_method} not supported'
            )
        self.image_interpolation_flag = self.image_interpolation_flags[
            image_interpolation_method.lower()]
        self.target_interpolation_flag = self.target_interpolation_flags[
            target_interpolation_method.lower()]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """Args:
            index: Index in dataset

        Returns:
            Tuple of Tensors of length 1, 2 or 3.
                First element is input image (CxHxW)).
                If image_set is not "test", then the next element is the target segmentation mask.
                If return_original_shape, then the next element is the shape [H, W] after scaling.
        """

        if self.year == '2012' and self.image_set == 'test':
            return self._getitem_image_only(index)
        else:
            return self._getitem_image_and_target(index)

    def _getitem_image_only(self, index: int) -> tuple[torch.Tensor, ...]:
        """Args:
            index: Index in dataset

        Returns:
            Tuple of Tensors of length 1 or 2.
                First element is input image (CxHxW).
                If return_original_shape, then the next element is the shape [H, W] after scaling.
        """
        img = super().__getitem__(index)
        img = np.array(img)
        assert img.dtype == 'uint8'
        img = img.astype('float32') / 255

        if self.scaling != 1:
            img = scale(img,
                        self.scaling,
                        interpolation=self.image_interpolation_flag)

        if self.return_original_shape:
            assert img.ndim == 3 and img.shape[2] == 3
            original_shape = img.shape[:2]

        if self.input_transform is not None:
            img = self.input_transform(image=img)['image']

        if self.joint_transform is not None:
            transformed = self.joint_transform(image=img)
            img = transformed['image']

        if self.smoothing_transform is not None:
            img = self.smoothing_transform(image=img)['image']

        if self.normalization is not None:
            img = self.normalization(image=img)['image']

        img = img.transpose([2, 0, 1])

        if self.return_original_shape:
            return (torch.FloatTensor(img), torch.LongTensor(original_shape))
        else:
            return torch.FloatTensor(img)

    def _getitem_image_and_target(self, index: int) -> tuple[torch.Tensor, ...]:
        """Args:
            index: Index in dataset

        Returns:
            Tuple of Tensors of length 2 or 3.
                First element is input image (CxHxW)).
                The next element is the target segmentation mask.
                If return_original_shape, then the next element is the shape [H, W] after scaling.
        """
        img, target = super().__getitem__(index)

        img = np.array(img)
        assert img.dtype == 'uint8'
        img = img.astype('float32') / 255
        target = np.array(target)

        if self.scaling != 1:
            img = scale(img,
                        self.scaling,
                        interpolation=self.image_interpolation_flag)
            target = scale(target,
                           self.scaling,
                           interpolation=self.target_interpolation_flag)

        if self.return_original_shape:
            original_shape = target.shape

        if self.input_transform is not None:
            img = self.input_transform(image=img)['image']

        if self.joint_transform is not None:
            transformed = self.joint_transform(image=img, mask=target)
            img, target = transformed['image'], transformed['mask']

        if self.smoothing_transform is not None:
            if isinstance(self.smoothing_transform, DualTransform):
                transformed = self.smoothing_transform(image=img, mask=target)
                img, target = transformed['image'], transformed['mask']
            else:
                img = self.smoothing_transform(image=img)['image']

        if self.normalization is not None:
            img = self.normalization(image=img)['image']

        img = img.transpose([2, 0, 1])

        if self.return_original_shape:
            return (torch.FloatTensor(img), torch.LongTensor(target),
                    torch.LongTensor(original_shape))
        else:
            return torch.FloatTensor(img), torch.LongTensor(target)


class AugmentedCityscapesSegmentation(Cityscapes):
    """Cityscapes dataset that applies different augmentations to the images and/or targets.

    Transformations are applied in the following order:
    1.) Scaling
    2.) Non-geometric transformations of the input image.
    3.) Geometric transformations that affect the image and target.
    4.) Adding sampled noise from smoothing distribution
    5.) Normalization of the input color values.

    All transformations can be skipped by setting the respective parameter to "None".
    """

    image_interpolation_flags = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'linear_exact': cv2.INTER_LINEAR_EXACT,
        'nearest_exact': cv2.INTER_NEAREST_EXACT
    }

    target_interpolation_flags = {
        'nearest': cv2.INTER_NEAREST,
        'nearest_exact': cv2.INTER_NEAREST_EXACT
    }

    def __init__(self,
                 root: str,
                 split: str = "train",
                 mode: str = "fine",
                 scaling: float = 1,
                 input_transform: albu.ImageOnlyTransform | albu.Compose = None,
                 joint_transform: albu.DualTransform | albu.Compose = None,
                 smoothing_transform: albu.ImageOnlyTransform = None,
                 normalization: albu.Normalize = None,
                 return_original_shape: bool = False,
                 image_interpolation_method: str = 'area',
                 target_interpolation_method: str = 'nearest') -> None:
        """Calls super constructor. Stores additional parameters as attributes.

        Args:
            root (string): Root directory of dataset where directory leftImg8bit
                and "gtFine" or "gtCoarse" are located.
            split (string, optional): The image split to use,
                "train", "test" or "val" (if mode="fine")
                otherwise "train", "train_extra" or "val"
            mode (string, optional): The quality mode to use, "fine" or "coarse"
            download: If true, downloads the dataset (does not work for trainaug)
            scaling: Scaling factor (1 = original shape)
            input_transform: Non-geometric transformation of input image.
            joint_transform: Geometric transformations that affect the image and target.
            smoothing_transform: Adding image noise from smoothing distribution.
            normalization: Normalization of the input color values.
            return_original_shape: If True, each datapoint includes the original image shape
                (after scaling, before the remaining transformations).
            image_interpolation_method: openCV flag for interpolation of color values for
                geometric transformations (see image_interpolation_flags)
            target_interpolation_method: openCV flag for interpolation of target classes for
                geometric transformations (see image_interpolation_flags)
        """

        super().__init__(root, split=split, mode=mode, target_type='semantic')

        self.input_transform = input_transform
        self.smoothing_transform = smoothing_transform
        self.joint_transform = joint_transform
        self.normalization = normalization
        self.return_original_shape = return_original_shape

        self.scaling = scaling
        if image_interpolation_method.lower(
        ) not in self.image_interpolation_flags.keys():
            raise ValueError(
                'Img interpolation method {image_interpolation_method} not supported'
            )
        if target_interpolation_method.lower(
        ) not in self.target_interpolation_flags.keys():
            raise ValueError(
                'Img interpolation method {image_interpolation_method} not supported'
            )
        self.image_interpolation_flag = self.image_interpolation_flags[
            image_interpolation_method.lower()]
        self.target_interpolation_flag = self.target_interpolation_flags[
            target_interpolation_method.lower()]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        """Args:
            index: Index in dataset

        Returns:
            Tuple of Tensors of length 2 or 3.
                First element is input image (CxHxW)).
                The next element is the target segmentation mask.
                If return_original_shape, then the next element is the shape [H, W] after scaling.
        """

        assert self.target_type == ['semantic']

        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.targets[index][0])

        img = np.array(img)
        assert img.dtype == 'uint8'
        img = img.astype('float32') / 255
        target = np.array(target)
        target = self._set_train_ids(target)

        if self.scaling != 1:
            img = scale(img,
                        self.scaling,
                        interpolation=self.image_interpolation_flag)
            target = scale(target,
                           self.scaling,
                           interpolation=self.target_interpolation_flag)

        if self.return_original_shape:
            original_shape = target.shape

        if self.input_transform is not None:
            img = self.input_transform(image=img)['image']

        if self.joint_transform is not None:
            transformed = self.joint_transform(image=img, mask=target)
            img, target = transformed['image'], transformed['mask']

        if self.smoothing_transform is not None:
            if isinstance(self.smoothing_transform, DualTransform):
                transformed = self.smoothing_transform(image=img, mask=target)
                img, target = transformed['image'], transformed['mask']
            else:
                img = self.smoothing_transform(image=img)['image']

        if self.normalization is not None:
            img = self.normalization(image=img)['image']

        img = img.transpose(([2, 0, 1]))

        if self.return_original_shape:
            return (torch.FloatTensor(img), torch.LongTensor(target),
                    torch.LongTensor(original_shape))
        else:
            return torch.FloatTensor(img), torch.LongTensor(target)

    def _set_train_ids(self, target: np.array) -> np.array:
        """Replace target classes with proper training ids.

        Cityscapes superclass has 35 classes in total, but many of them
        are part of the 255 ignore class.
        Cityscapes normally returns the "id"s, i.e. class labels in [-1,...,33] as
        the target mask
        For proper training, we need to replace them with the "train_id"s, which are
        255 or in [0, 20].
        """
        relabel_masks = {
        }  # stores, per train_id, which target pixels should be assigned to it

        target = target.copy()

        for cls in self.classes:
            train_id = cls.train_id
            if train_id == -1:
                train_id = 255

            if train_id in relabel_masks:
                # Need to OR, because there are multiple class "id"s with the same "train_id".
                relabel_masks[train_id] |= (target == cls.id)
            else:
                relabel_masks[train_id] = (target == cls.id)

        for train_id, mask in relabel_masks.items():
            target[mask] = train_id

        return target
