"""This module implements auxiliary functionality, in particular the processing of model
predictions and the preparation of localized smoothing distribution parameters.

confusion_matrix: Calculates confusion matrix between ground-truth and predicted array.
iou_from_confusion: Calculates class-wise intersection over union from confusion matrix.

crop_grid_predictions: Extract pixel values of a single grid cell.
remove_padding: Yields center of symmetrically padded ground truth and prediction arrays.

normalize_batch: Z-normalization with channel-wise mean and standard deviation.

generate_distribution: Generates Gaussian localized smoothing distribution.
generate_smoothing_stds: Generates pixelwise smoothing standard deviations for localized smoothing.
generate_distance_mask: Indicates pixels in cells with distance to cell (i,j) above some threshold.
generate_grid: Generates horizontal and vertical start and end coordinates for image grid cells.
"""

import numpy as np
import sklearn
import sklearn.metrics
import torch
from albumentations import get_center_crop_coords
from numpy.typing import NDArray


def confusion_matrix(y_target: NDArray[np.int_],
                     y_pred: NDArray[np.int_],
                     labels: NDArray[np.int_] = np.arange(21),
                     ignore_label: int = 255) -> NDArray[np.int_]:
    """Calculates confusion matrix between ground-truth and predicted array.

    Args:
        y_target: Integer ground truth labels of arbitrary shape
        y_pred: Predicted labels of same shape as y_target
        labels: Subset of labels for which confusion matrix should be created
        ignore_label: Additional label that only appears in y_target.

    Returns:
        Confusion matrix of shape (len(labels) + 1) ** 2,
        where entry (i, j) is number of objects that have ground truth class i
        and are classified as j.
        The last row contains the predicted class counts for predictions
        where y_target=ignore_label.
        The last column (if ignore_label not in y_pred) is zero everywhere.
    """

    # Relabeling is just to ensure that we don't store a lot of zero columns/rows
    new_ignore_label = labels.max() + 1
    y_target = y_target.copy()
    y_target[y_target == ignore_label] = new_ignore_label

    return sklearn.metrics.confusion_matrix(y_target.flatten(),
                                            y_pred.flatten(),
                                            labels=np.append(
                                                labels, new_ignore_label))


def iou_from_confusion(conf_matrix: NDArray[np.int_],
                       eps: float = 1e-6) -> np.ndarray:
    """Calculates class-wise intersection over union from confusion matrix.

    Args:
        y_target: Confusion matrix of shape CxC,
                  where entry (i,j) is number of class i instances classified as j
        eps: Small number to ensure that IOU=1 when class not in ground truth and not predicted.

    Returns:
        C per-class IOU values in [0, 1]
    """
    tps = np.diag(conf_matrix)
    fps = np.sum(conf_matrix, axis=0) - tps
    fns = np.sum(conf_matrix, axis=1) - tps

    iou = (tps + eps) / (tps + fps + fns + eps)
    return iou


def crop_grid_predictions(pred: torch.Tensor | np.ndarray, i: int, j: int,
                          grid_vertical: NDArray[np.int_],
                          grid_horizontal: NDArray[np.int_],
                          original_shape: NDArray[np.int_]):
    """Extract pixel values of a single grid cell from a batch of padded predicted logits.

    We assume that the logits are padded symmetrically and that the shape without
    padding is known.
    Padding is due to segmentation models require input shapes that are multiple of some
    integer.

    Args:
        pred: Predicted logits of shape BxCx(H+pad_h)x(W+pad_w).
        i: Vertical grid cell coordinate.
        j: Horizontal grid cell coordinate.
        grid_vertical: Array A of shape 2xI. A[0,i] is vertical starting coordinate of grid row i
            and A[1, i] is vertical ending coordinate of grid row i.
        grid_horizontal: Array B of shape 2xJ.
            B[0,j] is horizontal starting coordinate of grid column j
            and B[1, j] is horizontal ending coordinate of grid column j.
        original_shape: Shape (H, W) of the prediction without padding

    Returns:
        Logits of shape B x C x cell_height(i, j) x cell_width(i, j).
    """
    new_shape = np.array(pred.shape[2:])

    x1, y1, x2, y2 = get_center_crop_coords(*new_shape, *original_shape)
    pred = pred[:, :, y1:y2, x1:x2]

    pred = pred[:, :, grid_vertical[i][0]:grid_vertical[i][1],
                grid_horizontal[j][0]:grid_horizontal[j][1]]

    return pred


def remove_padding(
        pred: np.ndarray, y: np.ndarray,
        original_shape: NDArray[np.int_]) -> tuple[np.ndarray, np.ndarray]:
    """Yields center of symmetrically padded ground truth and prediction arrays.

    Args:
        pred: Symmetrically padded scalar predictions of shape (H+pad_h) x (W+pad_w)
        y: Symmetrically padded ground truth scalars of shape (H+pad_h) x (W+pad_w)
        original_shape: Array-like of length 2 containing (H, W)

    Returns:
        Two arrays of shape HxW, corresponding to predicted and ground truth labels without padding
    """
    assert y.shape[1:] == pred.shape[2:]

    original_shape = original_shape[0]

    #  TODO: Always is True, because compare tensor to shape. Fix this
    if pred.shape[2:] != original_shape:
        x1, y1, x2, y2 = get_center_crop_coords(int(pred.shape[2]),
                                                int(pred.shape[3]),
                                                int(original_shape[0]),
                                                int(original_shape[1]))

        pred = pred[:, :, y1:y2, x1:x2]
        y = y[:, y1:y2, x1:x2]

    return pred, y


def normalize_batch(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                    device) -> torch.Tensor:
    """Z-normalization with channel-wise mean and standard deviation.

    Args:
        x: Input batch of shape ...xCxHxW
        mean: Means of shape C
        std: Standard deviations of shape C
        device: Device to move mean and std to before normalization.
            Should match device of x.

    Returns:
        Z-normalized tensor on specified device.
    """

    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)

    mean = mean[:, None, None].to(device)
    std = std[:, None, None].to(device)
    x = (x - mean) / std
    return x


def generate_distribution(i, j, original_shape, grid_vertical, grid_horizontal,
                          std_min, std_max, metric, interpolate_variance,
                          max_std_at_boundary, device):
    """Generates Gaussian localized smoothing distribution.

    Given a shape HxW, grid coordinates and a central grid cell (i, j), the
    smoothing standard deviation within cell (k, l) is interpolated between std_min
    and std_max, based on the distance between (i, j) and (k, l).

    Args:
        i: Vertical grid cell coordinate
        j: Horizontal grid cell coordinate
        original_shape: Image shape(H, W)
        grid_vertical: Array A of shape 2xI. A[0,i] is vertical starting coordinate of grid row i
            and A[1, i] is vertical ending coordinate of grid row i.
        grid_horizontal: Array B of shape 2xJ. B[0,j] is horizontal starting coordinate of grid column j
            and B[1, j] is horizontal ending coordinate of grid column j.
        std_min: The minimum smoothing standard deviation, which is applied at (i, j).
        std_max: The maximum smoothing standard deviation.
        metric: The notion of distance between grid cells.
            Currently only supports `l_0`, which is actually l_infty, i.e. max( |k - i|, |l - j|)
        interpolate_variance: Whether to linearly interpolate variance or standard deviation.
            If False: Linearly interpolate between std_min and std_max.
            If True: Linearly interpolate between std_min**2 and std_max**2, then take square root.
        max_std_at_boundary: Determines at which distance interpolation yields std_max.
            If True: Yields std_max when d((i, j), (k, l)) = max_{0<=m<=I, 0<=n<=J} d((i, j),(m, n))
            If False: Yields std_max when d((i, j), (k, l))

    Returns:
        Smoothing distribution with mean 0 and standard deviations as described above.
    """

    stds = generate_smoothing_stds(i, j, original_shape, grid_vertical,
                                   grid_horizontal, std_min, std_max, metric,
                                   interpolate_variance, max_std_at_boundary)

    dist = torch.distributions.Normal(0, torch.Tensor(stds).to(device))
    return dist


def generate_smoothing_stds(i: int,
                            j: int,
                            original_shape: tuple[int, int] | NDArray[np.int_],
                            grid_vertical: NDArray[np.int_],
                            grid_horizontal: NDArray[np.int_],
                            std_min: float,
                            std_max: float,
                            metric='l_0',
                            interpolate_variance=False,
                            max_std_at_boundary=False):
    """Generates pixelwise smoothing standard deviations for localized smoothing.

    Given a shape HxW, grid coordinates and a central grid cell (i, j), the
    smoothing standard deviation within cell (k, l) is interpolated between std_min
    and std_max, based on the distance between (i, j) and (k, l).

    Args:
        i: Vertical grid cell coordinate
        j: Horizontal grid cell coordinate
        original_shape: Image shape(H, W)
        grid_vertical: Array A of shape 2xI. A[0,i] is vertical starting coordinate of grid row i
            and A[1, i] is vertical ending coordinate of grid row i.
        grid_horizontal: Array B of shape 2xJ. B[0,j] is horizontal starting coordinate of grid column j
            and B[1, j] is horizontal ending coordinate of grid column j.
        std_min: The minimum smoothing standard deviation, which is applied at (i, j).
        std_max: The maximum smoothing standard deviation.
        metric: The notion of distance between grid cells.
            Currently only supports `l_0`, which is actually l_infty, i.e. max( |k - i|, |l - j|)
        interpolate_variance: Whether to linearly interpolate variance or standard deviation.
            If False: Linearly interpolate between std_min and std_max.
            If True: Linearly interpolate between std_min**2 and std_max**2, then take square root.
        max_std_at_boundary: Determines at which distance interpolation yields std_max.
            If True: Yields std_max when d((i, j), (k, l)) = max_{0<=m<=I, 0<=n<=J} d((i, j),(m, n))
            If False: Yields std_max when d((i, j), (k, l))

    Returns:
        Array of shape HxW, with entry (h, w) being smoothing standard deviation for pixel (h, w)
    """

    if len(grid_vertical) == 1 and len(grid_horizontal) == 1:
        return np.ones(original_shape) * std_min

    stds = np.zeros(original_shape)
    grid_shape = np.array([len(grid_vertical), len(grid_horizontal)])
    center = np.array([i, j])

    # Determine distance function and maximum distance, i.e. distance at which max_std is reached
    if metric == 'l_0':
        if max_std_at_boundary:
            # Ensures that we reach max_std at one of the edges of the image
            max_distance = max([
                i, j, (len(grid_vertical) - 1) - i,
                (len(grid_horizontal) - 1) - j
            ])
        else:
            max_distance = grid_shape.max() - 1
        distance_fn = (lambda x, y: np.linalg.norm(x - y, np.inf))
    else:
        raise NotImplementedError(f'Metric \"{metric}\" not implemented')

    # Iterate over grid. For each cell, calculate std of pixels
    for pos in np.ndindex(len(grid_vertical), len(grid_horizontal)):
        pos = np.array(pos)

        distance = distance_fn(center, pos)
        # Either interpolate variances, then take square root
        if interpolate_variance:
            var = (std_min**2) + ((std_max**2) -
                                  (std_min**2)) * (distance / max_distance)
            std = np.sqrt(var)
        # Or do linear interpolation between standard deviations
        else:
            std = std_min + (std_max - std_min) * (distance / max_distance)
        k, l = pos[0], pos[1]

        # Store the std of this grid cell for all corresponding entries in stds array
        stds[grid_vertical[k][0]:grid_vertical[k][1],
             grid_horizontal[l][0]:grid_horizontal[l][1]] = std

    return stds


def generate_distance_mask(i: int,
                           j: int,
                           mask_distance: float,
                           original_shape: tuple[int, int] | NDArray[np.int_],
                           grid_vertical: NDArray[np.int_],
                           grid_horizontal: NDArray[np.int_],
                           metric: str = 'l_0') -> NDArray[np.bool_]:
    """Generates mask indicating pixels in cells with distance to (i,j) above some threshold.

    Args:
        i: Vertical grid cell coordinate
        j: Horizontal grid cell coordinate
        mask_distance: Distance above which (inclusive) grid cells are to be masked.
        original_shape: Image shape (H, W)
        grid_vertical: Array A of shape 2xI. A[0,i] is vertical starting coordinate of grid row i
            and A[1, i] is vertical ending coordinate of grid row i.
        grid_horizontal: Array B of shape 2xJ. B[0,j] is horizontal starting coordinate of grid column j
            and B[1, j] is horizontal ending coordinate of grid column j.
        metric: The notion of distance between grid cells.
            Currently only supports `l_0`, which is actually l_infty, i.e. max( |k - i|, |l - j|)

    Returns:
        Boolean array of shape HxW. Entry (h, w) is True iff (h, w) is in a grid cell (k, l)
        such that distance((i, j), (k, l)) < mask_distance w.r.t. specified metric.
    """

    mask = np.zeros(original_shape, dtype='bool')
    center = np.array([i, j])

    if metric == 'l_0':
        distance_fn = (lambda x, y: np.linalg.norm(x - y, np.inf))
    else:
        raise NotImplementedError(f'Metric \"{metric}\" not implemented')

    # Iterate over grid. For each cell, calculate distance from (i, j) and mask accordingly
    for pos in np.ndindex(len(grid_vertical), len(grid_horizontal)):
        pos = np.array(pos)
        distance = distance_fn(center, pos)
        k, l = pos[0], pos[1]

        if distance >= mask_distance:
            mask[grid_vertical[k][0]:grid_vertical[k][1],
                 grid_horizontal[l][0]:grid_horizontal[l][1]] = False
        else:
            mask[grid_vertical[k][0]:grid_vertical[k][1],
                 grid_horizontal[l][0]:grid_horizontal[l][1]] = True

    return mask


def generate_grid(
    original_shape: list[int, int] | NDArray[np.int_],
    grid_shape: list[int, int] | NDArray[np.int_]
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Generates horizontal and vertical start and end coordinates for image grid cells.

    Args:
        original_shape: Shape (H, W) of the image to be split up into grid cells.
        grid_shape: Number of vertical and horizontal grid cell numbers (I, J)

    Returns:
        Tuple of two arrays (A, B):
            A has shape 2xI. A[0,i] is vertical starting coordinate of grid row i
                and A[1, i] is vertical ending coordinate of grid row i.
            B has shape 2xJ. B[0,j] is horizontal starting coordinate of grid column j
                and B[1, j] is horizontal ending coordinate of grid column j.
    """
    grid_vertical = np.linspace(0,
                                original_shape[0],
                                grid_shape[0] + 1,
                                dtype='int')
    grid_vertical = np.vstack((grid_vertical[:-1], grid_vertical[1:]))

    grid_horizontal = np.linspace(0,
                                  original_shape[1],
                                  grid_shape[1] + 1,
                                  dtype='int')
    grid_horizontal = np.vstack((grid_horizontal[:-1], grid_horizontal[1:]))

    return grid_vertical.T, grid_horizontal.T
