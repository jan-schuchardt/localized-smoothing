"""Methods for sampling predictions with localized input noise, or statistics thereof.

Methods for certification:

smooth_prediction: Generates different types of smoothed predictions for a single image.
sample_base_classifier_statistics: Samples statistics for certifying
    different types of smoothed predictions.
sample_grid_cell_scores: Samples softmax scores for base classifier within a specified grid cell.
sample_padded: Add noise sampled from distribution to central patch of padded image.
get_localized_distribution_cached: Yields distirbution for localized smoothing of a specific  cell.
get_distance_mask_cached: Yields distance mask for localized smoothing of a specific grid cell.

Methods for training:

sample_stitched_base_model_logits_batched: Samples logits stitched together from per-cell logits
    under localized noise applied to batch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations import get_center_crop_coords
#from time import time
from numpy.typing import NDArray

from localized_smoothing.segmentation.utils import (crop_grid_predictions,
                                                    generate_distance_mask,
                                                    generate_grid,
                                                    generate_smoothing_stds,
                                                    normalize_batch)


class SmoothedModel(nn.Module):
    """A base model combined with a smoothing distribution.

    Offers the following functionality:

    smooth_prediction: Generates different types of smoothed predictions for a single image.
    sample_base_classifier_statistics: Samples statistics for certifying
        different types of smoothed predictions.
    sample_stitched_base_model_logits_batched: Samples logits stitched together from per-cell logits
        under localized noise applied to batch (for training).

    For efficiency reasons, the smoothing standard deviations are only computed once and reused,
    until the shape of the input images changes. 
    Don't modify the attributes for caching thesm during execution.
    """

    def __init__(self, base_model: nn.Module, sample_params: dict,
                 distribution_params: dict, normalization_params: dict,
                 device: torch.device):
        """

        Args:
            base_model: The base segmentation model that is to be smoothed.
            sample_params: Dictionary containing number of samples etc.,
                see "seml/scripts/segmentation/cert.py"
            distribution_params: Dictionary containing parameters of the smoothing distribution
                see "seml/scripts/segmentation/cert.py"
            normalization_params: Dictionary with two keys "mean" and "std", each corresponding
                to Tensor of shape [3] indicating channelwise mean and std for z-normalization.
            device: The device the base model is on.
        """

        super().__init__()

        self.base_model = base_model
        self.sample_params = sample_params
        self.distribution_params = distribution_params
        self.normalization_params = normalization_params
        self.device = device

        self._cached_original_shape = None
        self._cached_stds = {}
        self._cached_distance_masks = {}

        self._cached_original_shape_batched = None
        self._cached_stds_batched = {}
        self._cached_distance_masks_batched = {}

    def smooth_prediction(
            self,
            img: torch.Tensor,
            original_shape: NDArray[np.float_],
            device: torch.device,
            continuous: bool = False,
            argmax: bool = False,
            center: bool = False,
            n_center_candidates: int = 64
    ) -> dict[str, torch.LongTensor | float]:
        """Generates different types of smoothed predictions for a single image.

        Supports three types of predictions:
            - "argmax", i.e. standard randomized smoothing with majority vote
            - "continuous", i.e. expected value smoothing (https://arxiv.org/abs/2009.08061)
            - "center", i.e. center smoothing (https://arxiv.org/abs/2102.09701)

        args:
            img: A padded input image of shape Cx(H_orig+H_pad)x(W_orig+W_pad)
            original_shape: Array containing [H_orig, W_orig]
            device: The device the model is on.
            continuous: Whether to compute a prediction using expected value smoothing.
            argmax: Whether to compute a prediction using majority voting.
            center: Whether to compute a prediction using center smoothing.

        Returns:
            Dictionary with 1 to 3 keys:
                - "continuous": Segmentation mask LongTensor consisting of classes with highest
                    expected softmax score,
                    shape H_orig x W_orig.
                - "argmax": Segmentation mask LongTensor consisting of most likely classes,
                        shape H_orig x W_orig.
                - "center": Segmentation mask LongTensor consisting of approximation of
                    center of smallest l_0 ball that contains at least 50% of probability mass,
                    shape H_orig x W_orig.
                - "-center_radius": Float radius of this ball.
        """

        # Extract some parameters from argument dictionaries
        if not (continuous or argmax or center):
            raise ValueError('Must specify at least one prediction type.')

        if center:
            assert n_center_candidates <= self.sample_params['batch_size']
            if np.any(self.distribution_params['grid_shape'] != [1, 1]):
                raise NotImplementedError(
                    'Center smoothing only supports 1x1 grid')

        n_samples = self.sample_params['n_samples_pred']

        upsampling_factor = self.sample_params.get('upsampling_factor_pred', 1)
        if upsampling_factor != 1:
            assert not (continuous or center)
        assert n_samples % upsampling_factor == 0
        n_samples = n_samples // upsampling_factor

        batch_size = self.sample_params['batch_size']
        # Uncomment the next line if cityscapes experiments crash due to batch size issues (peak code quality)
        # batch_size=205
        assert n_samples % batch_size == 0
        n_batches = n_samples // batch_size

        grid_shape = np.array(self.distribution_params['grid_shape'])
        grid_vertical, grid_horizontal = generate_grid(original_shape,
                                                       grid_shape)

        # Create tensors for storing final results
        pred_continuous = torch.zeros(
            (self.base_model.classes, *original_shape))

        pred_argmax_counts = torch.zeros(
            (self.base_model.classes, *original_shape), dtype=torch.long)

        pred_center = torch.zeros(*original_shape, dtype=torch.long)
        pred_center_radius = None

        # Create tensors for storing intermediate results
        center_candidates = None
        distances = torch.empty((n_center_candidates, 0), dtype=torch.long)

        img = img.to(device)

        # Iterate over all grid cells, compute per-grid-cell predictions
        # (Center smoothign only ever has one grid cell)
        for i, j in np.ndindex(*grid_shape):

            for _ in range(n_batches):

                grid_scores = self.sample_grid_cell_scores(
                    i, j, img, batch_size, grid_vertical, grid_horizontal,
                    original_shape, device)

                if continuous:
                    # Update average softmax scores for this grid cell
                    batch_pred_continuous = grid_scores.sum(dim=0) / n_samples
                    batch_pred_continuous = batch_pred_continuous.detach().cpu()

                    pred_continuous[
                        :,
                        grid_vertical[i][0]:grid_vertical[i][1],
                        grid_horizontal[j][0]:grid_horizontal[j][1],
                    ] += batch_pred_continuous

                if argmax:
                    # Update class counts for this grid cell
                    batch_pred_argmax = grid_scores.argmax(dim=1).detach().cpu()
                    batch_pred_counts = F.one_hot(
                        batch_pred_argmax,
                        self.base_model.classes).sum(dim=0).permute([2, 0, 1])

                    pred_argmax_counts[
                        :,
                        grid_vertical[i][0]:grid_vertical[i][1],
                        grid_horizontal[j][0]:grid_horizontal[j][1],
                    ] += batch_pred_counts

                if center:
                    batch_pred_argmax = grid_scores.argmax(dim=1).detach().cpu()

                    # In first batch: Pick first N as center candidates
                    if center_candidates is None:
                        center_candidates = batch_pred_argmax[:
                                                              n_center_candidates]
                        if n_center_candidates < self.sample_params[
                                'batch_size']:
                            batch_pred_argmax = batch_pred_argmax[
                                n_center_candidates:]
                        else:
                            continue

                    candidates_flattened = center_candidates.flatten(
                        start_dim=1)
                    batch_pred_flattened = batch_pred_argmax.flatten(
                        start_dim=1)

                    # Compute distance between candidates and predictions in batch
                    batch_distances = torch.linalg.norm(
                        (candidates_flattened[:, None].float() -
                         batch_pred_flattened[None]),
                        ord=0,  # l_0 norm
                        dim=2)

                    distances = torch.cat((distances, batch_distances), dim=1)

            if center:
                median_distances = distances.median(dim=1).values
                pred_center += center_candidates[median_distances.argmin()]
                pred_center_radius = median_distances.min().item()

        ret = {}
        if continuous:
            ret['continuous'] = pred_continuous.argmax(dim=0)
        if argmax:
            ret['argmax'] = pred_argmax_counts.argmax(dim=0)
        if center:
            ret['center'] = pred_center
            ret['center_radius'] = pred_center_radius

        return ret

    def sample_base_classifier_statistics(
            self,
            img: torch.Tensor,
            original_shape: NDArray[np.int_],
            device: torch.device,
            sample_histogram: bool = False,
            sample_consistency: bool = False,
            sample_distances: bool = False,
            predictions_continuous: torch.LongTensor = None,
            predictions_argmax: torch.LongTensor = None,
            predictions_center: torch.LongTensor = None,
            n_thresholds: int = 1000) -> dict[str, torch.Tensor]:
        """Samples statistics for certifying different types of smoothed predictions.

        Supports three types of statistics:
            - consistency, i.e. how often the smoothly predicted labels are predicted
                by the base model under random input noise.
            - histogram, i.e. how often the base model's softmax scores for the
                smoothly predicted labels are within quantization bins of [0, 1]
                under random input noise
                (https://arxiv.org/abs/2009.08061)
            - distances: i.e. l_0 distances of the center smoothed prediction to the
                predictions of the base model under random input noise
                (https://arxiv.org/abs/2102.09701).

        Computing these statistics requires providing the smoothed predictions via the
        "predictions_..." arguments. These can be computed via "smooth_prediction"
        above.

        args:
            img: A padded input image of shape Cx(H_orig+H_pad)x(W_orig+W_pad)
            original_shape: Array containing [H_orig, W_orig]
            device: The device the model is on.
            sample_histogram: Whether to sample histograms for expected value smoothing, see above.
            sample_consistency: Whether to sample consistency for majority voting, see above.
            sample_distances: Whether to sample distances for center smoothing, see above.
            predictions_continuous: Smoothed prediction for expected value smoothing,
                relative to which we compute histograms.
            predictions_argmax: Smoothed prediction for majority voting,
                relative to which we compute consistency.
            predictions_center: Smoothed prediction for center smoothing,
                relative to which we compute distances.
            n_thresholds: Number of quantization thresholds to use for splitting up [0, 1]
                into uniformly sized quantization bins.

        Returns:
            Dictionary with 1 to 3 keys:
                - "histogram": LongTensor of shape (n_thresholds+1) x H_orig x W_orig,
                    where entry (b, i, j) indicates how often the base classifier's softmax
                    score for pixel (i, j) and label predictions_continuous[i, j]
                    was within the bth quantization bin.
                - "consistency": LongTensor of shape H_orig x W_orig,
                    where entry (i, j) indicates how often the base classifier's prediction
                    for pixel (i, j) was identical to predictions_argmax[i, j]
                - "distances": FloatTensor of length n_samples,
                    where each entry is the l_0 distance of predictions_center to
                    a base model prediction sampled under random input noise.
        """

        # Extract some parameters from argument dictionaries
        if not (sample_histogram or sample_consistency or sample_distances):
            raise ValueError('Must specify at least one statistic to sample.')
        if sample_histogram and (predictions_continuous is None):
            raise ValueError(
                'Must provide predictions_continuous for sample_histogram')
        if sample_consistency and (predictions_argmax is None):
            raise ValueError(
                'Must provide predictions_argmax for sample_consistency')
        if sample_distances and (predictions_center is None):
            if np.any(self.distribution_params['grid_shape'] != [1, 1]):
                raise NotImplementedError(
                    'Center smoothing only supports 1x1 grid')
            if predictions_center is None:
                raise ValueError(
                    'Must provide predictions_center for sample_distances')

        n_samples = self.sample_params['n_samples_cert']

        upsampling_factor = self.sample_params.get('upsampling_factor_cert', 1)
        if upsampling_factor != 1:
            assert not (sample_histogram or sample_distances or
                        predictions_continuous or predictions_center)
        assert n_samples % upsampling_factor == 0
        n_samples = n_samples // upsampling_factor

        batch_size = self.sample_params['batch_size']
        if (n_samples % batch_size) != 0:
            raise ValueError('n_samples_cert must be divisible by batch size')
        n_batches = n_samples // batch_size

        # Determine start/end coordinates of grid cells
        grid_shape = np.array(self.distribution_params['grid_shape'])
        grid_vertical, grid_horizontal = generate_grid(original_shape,
                                                       grid_shape)

        img = img.to(device)

        # Create tensors for storing statistics
        thresholds = torch.arange(0, 1, 1 / (n_thresholds + 1))[1:].to(device)
        histogram = torch.zeros((n_thresholds + 1, *original_shape),
                                dtype=torch.long)
        consistency = torch.zeros(*original_shape, dtype=torch.long)
        distances = torch.empty(0, dtype=torch.long)

        for i, j in np.ndindex(*grid_shape):
            """For each statistic we want to compute,
            we first extract the part of the corresponding prediction that
            is relevant for this specific grid cell (i, j)
            """
            if sample_histogram:
                grid_predictions_continuous = predictions_continuous[
                    grid_vertical[i][0]:grid_vertical[i][1],
                    grid_horizontal[j][0]:grid_horizontal[j][1]].to(device)

                # Histogram for grid section (i, j). Shape (n_tresholds+1) x width_grid x height_grid
                grid_histogram = torch.zeros(
                    (n_thresholds + 1, *grid_predictions_continuous.shape),
                    dtype=torch.long)

            if sample_consistency:
                grid_predictions_argmax = predictions_argmax[
                    grid_vertical[i][0]:grid_vertical[i][1],
                    grid_horizontal[j][0]:grid_horizontal[j][1]].to(device)

                grid_consistency = torch.zeros(*grid_predictions_argmax.shape,
                                               dtype=torch.long).to(device)

            if sample_distances:
                grid_predictions_center = predictions_center.to(device)

            # Next, we aggregate per-cell statistics over multiple batches.
            for _ in range(n_batches):

                grid_scores = self.sample_grid_cell_scores(
                    i, j, img, batch_size, grid_vertical, grid_horizontal,
                    original_shape, device)

                if sample_histogram:
                    # Select probabilities corresponding to our previously predicted classes
                    prediction_indices = grid_predictions_continuous[
                        None,
                        None,
                        :,
                    ].repeat_interleave(batch_size, dim=0)

                    grid_scores_pred = torch.gather(
                        grid_scores, 1, prediction_indices).squeeze(1)

                    # Quantize based on bins. Gives batch x width_grid x height_grid tensor.
                    grid_scores_pred = torch.bucketize(grid_scores_pred,
                                                       thresholds).cpu()

                    # Count, for each pixel, how many samples fell into which bin
                    # 0 = bin dimension of target and batch dimension of source
                    grid_histogram.scatter_add_(
                        0, grid_scores_pred, torch.ones_like(grid_scores_pred))

                if sample_consistency:
                    batch_predictions = grid_scores.argmax(dim=1)
                    grid_consistency += (
                        batch_predictions == grid_predictions_argmax).sum(dim=0)

                if sample_distances:
                    center_flattened = grid_predictions_center.flatten()
                    batch_pred_flattened = grid_scores.argmax(dim=1).flatten(
                        start_dim=1)

                    batch_distances = torch.linalg.norm(
                        batch_pred_flattened - center_flattened.float(),
                        ord=0,  # l_0 norm
                        dim=1)

                    distances = torch.cat((distances, batch_distances.cpu()))

            # After we are done with all batches for this cell, we update the final results tensors.
            if sample_histogram:
                histogram[:, grid_vertical[i][0]:grid_vertical[i][1],
                          grid_horizontal[j][0]:grid_horizontal[j]
                          [1]] = grid_histogram

            if sample_consistency:
                consistency[grid_vertical[i][0]:grid_vertical[i][1],
                            grid_horizontal[j][0]:grid_horizontal[j]
                            [1]] = grid_consistency.cpu()

        ret = {}
        if sample_histogram:
            ret['histogram'] = histogram
        if sample_consistency:
            ret['consistency'] = consistency * upsampling_factor
        if sample_distances:
            ret['distances'] = distances

        return ret

    def sample_grid_cell_scores(self, i, j, img, n_samples, grid_vertical,
                                grid_horizontal, original_shape, device):
        """Samples softmax scores for base classifier within a specified grid cell.

        Args:
            i: Vertical grid cell coordinate.
            j: Horizontal grid cell coordinate.
            img: img: A padded input image of shape Cx(H_orig+H_pad)x(W_orig+W_pad).
            n_samples: The number of samples to take.
            grid_vertical: Array A of shape 2xI. A[0,i] is vertical starting coordinate of grid row i
                and A[1, i] is vertical ending coordinate of grid row i.
            grid_horizontal: Array B of shape 2xJ.
                B[0,j] is horizontal starting coordinate of grid column j
                and B[1, j] is horizontal ending coordinate of grid column j.
            original_shape: Array containing [H_orig, W_orig].
            device: The device the model is on.

        Returns:
            Sampled softmax scores of shape n_samples x K x H_{cell_i} x W_{cell_j}.
        """

        if ((self._cached_original_shape is None) or
            (np.any(original_shape != self._cached_original_shape))):

            self._recache_stds(original_shape, grid_vertical, grid_horizontal)

            self._recache_distance_masks(original_shape, grid_vertical,
                                         grid_horizontal)

            self._cached_original_shape = original_shape

        dist = torch.distributions.Normal(
            0,
            torch.Tensor(self._cached_stds[(i, j)]).to(device))

        distance_mask = self._cached_distance_masks[(i, j)]

        x = self.sample_padded(img, dist, original_shape, n_samples,
                               distance_mask)
        x = normalize_batch(x, **self.normalization_params, device=device)

        cell_scores = F.softmax(self.base_model(x), dim=1)

        cell_scores = crop_grid_predictions(cell_scores, i, j, grid_vertical,
                                            grid_horizontal, original_shape)

        return cell_scores.detach()

    @staticmethod
    def sample_padded(
            img: torch.FloatTensor,
            dist: torch.distributions.Distribution,
            original_shape: torch.LongTensor,
            n_samples: int,
            mask: NDArray[np.bool_] | None = None) -> torch.FloatTensor:
        """Add noise sampled from distribution to central patch of padded image.

        After sampling, all pixel values are clipped into [0, 1].
        Optionally allows masking of central patch.

        Args:
            img: Input image of shape Cx(H_orig+H_pad)x(W_orig+W_pad) with pixel values
                in [0, 1].
            dist: Distribution that yields samples of shape H_origxW_orig
            original_shape: 1D Tensor of length 2 that contains [H_orig, W_orig]
            n_samples: Number of samples to take
            mask: Boolean array of shape H_orig x W_orig that indicates all non-padded
                pixels that are to be set to 0

        Returns:
            Randomly perturbed images of shape n_samples x C x (H_orig+H_pad)x(W_orig+W_pad),
                with noise only added to the center part of shape H_orig x W_orig.
        """
        n_channels = img.shape[0]
        new_shape = np.array(img.shape[1:])

        img = img.unsqueeze(0).repeat_interleave(n_samples, dim=0)

        x1, y1, x2, y2 = get_center_crop_coords(*new_shape, *original_shape)

        unpadded_noise = dist.sample([n_samples, n_channels])

        img[:, :, y1:y2, x1:x2] += unpadded_noise

        if mask is not None:
            img[:, :, y1:y2, x1:x2][:, :, ~mask] = 0

        img = torch.clamp(img, 0, 1)

        return img

    def _recache_stds(self, original_shape, grid_vertical, grid_horizontal):
        grid_shape = np.array(self.distribution_params['grid_shape'])

        for i, j in np.ndindex(*grid_shape):
            stds = generate_smoothing_stds(
                i, j, original_shape, grid_vertical, grid_horizontal,
                self.distribution_params['std_min'],
                self.distribution_params['std_max'],
                self.distribution_params['metric'],
                self.distribution_params['interpolate_variance'],
                self.distribution_params['max_std_at_boundary'])

            self._cached_stds[(i, j)] = stds

    def _recache_distance_masks(self, original_shape, grid_vertical,
                                grid_horizontal):
        grid_shape = np.array(self.distribution_params['grid_shape'])

        for i, j in np.ndindex(*grid_shape):

            if self.distribution_params['mask_distance'] is None:
                self._cached_distance_masks[(i, j)] = None

            else:
                distance_mask = generate_distance_mask(
                    i,
                    j,
                    self.distribution_params['mask_distance'],
                    original_shape,
                    grid_vertical,
                    grid_horizontal,
                    metric=self.distribution_params['metric'])

                self._cached_distance_masks[(i, j)] = distance_mask

    def sample_stitched_base_model_logits_batched(
            self,
            imgs: torch.Tensor,
            n_samples: int,
            device: torch.device,
            original_shape: NDArray[np.int_] = None):
        """Samples logits stitched together from per-cell logits under localized noise applied to batch.

        This method does the following (simiilar to Fig. 1, but without majority voting.)
            for each image:
            1) Split the image (without padding) into grid cells.
            2) For each grid cell (i, j):
                2.1) Take n_samples from a localized smoothing distribution with center (i, j)
                2.2) Apply base model to these samples.
                2.3) Extract the logits for cell (i, j)
            3.) Stitch the per-cell logits (without padding) together into a single array of logits.

        Currently only supports n_samples=1.

        Unlike the methods above, this method does not .detach(), does not move tensors to .cpu(),
        so that we can use it during training.

        For hacky efficiency reasons that were needed to make a deadline,
        we cache some parameters in global variables.
        Don't use the method in the same script as smooth_prediction or
        sample_base_classifier_statistics.

        Args:
            imgs: A batch of input images of shape BxCx(H_orig+H_pad)x(W_orig+W_pad).
            n_samples: The number of samples to take (must currently be 1).
            device: The device the model is on.
            original_shape: Array containing [H_orig, W_orig].
                If None, we assume H_orig=W_orig=0.

        Returns:
            A batch of logits of shape BxKxH_origxW_orig.
        """

        if (n_samples != 1):
            raise NotImplementedError()

        assert (imgs.ndim == 4) and (imgs.shape[1] == 3)

        n_images = imgs.shape[0]
        n_classes = self.base_model.classes
        height, width = imgs.shape[2:]
        n_channels = imgs.shape[1]

        if original_shape is None:
            original_shape = np.array([height, width])

        grid_shape = np.array(self.distribution_params['grid_shape'])
        grid_vertical, grid_horizontal = generate_grid(original_shape,
                                                       grid_shape)

        # Prepare smoothing stds and distance masks for the different grid cells
        if (self._cached_original_shape_batched is None or
                np.any(self._cached_original_shape_batched != original_shape)):

            self._cached_original_shape_batched = original_shape
            self._cached_stds_batched = []
            self._cached_distance_masks_batched = []

            for i, j in np.ndindex(*grid_shape):

                stds = generate_smoothing_stds(
                    i, j, original_shape, grid_vertical, grid_horizontal,
                    self.distribution_params['std_min'],
                    self.distribution_params['std_max'],
                    self.distribution_params['metric'],
                    self.distribution_params['interpolate_variance'],
                    self.distribution_params['max_std_at_boundary'])

                self._cached_stds_batched.append(torch.Tensor(stds))

                if self.distribution_params['mask_distance'] is not None:
                    distance_mask = generate_distance_mask(
                        i,
                        j,
                        self.distribution_params['mask_distance'],
                        original_shape,
                        grid_vertical,
                        grid_horizontal,
                        metric=self.distribution_params['metric'])

                    self._cached_distance_masks_batched.append(
                        torch.BoolTensor(distance_mask))

                else:
                    self._cached_distance_masks_batched = None

            self._cached_stds_batched = torch.stack(self._cached_stds_batched,
                                                    dim=0).to(device).view(
                                                        np.prod(grid_shape), 1,
                                                        1, *original_shape)

            if self.distribution_params['mask_distance'] is not None:
                self._cached_distance_masks_batched = torch.stack(
                    self._cached_distance_masks_batched,
                    dim=0).view(np.prod(grid_shape), 1, 1, *original_shape)

        # Add noise to unpadded part of the image, then clip and normalize
        x1, y1, x2, y2 = get_center_crop_coords(height, width, *original_shape)
        imgs = imgs.to(device)
        imgs = imgs.view(1, n_images, n_channels, height, width)
        imgs = torch.repeat_interleave(imgs, np.prod(grid_shape), dim=0)

        imgs[:, :, :, y1:y2, x1:x2] = torch.normal(imgs[:, :, :, y1:y2, x1:x2],
                                                   self._cached_stds_batched)

        if self.distribution_params['mask_distance'] is not None:
            imgs[:, :, :, y1:y2,
                 x1:x2][~self._cached_distance_masks_batched.expand(
                     np.prod(grid_shape), n_images, n_channels, *original_shape
                 )] = 0

        imgs = torch.clamp(imgs, 0, 1)
        imgs = imgs.view(-1, n_channels, height, width)

        imgs = normalize_batch(imgs, **self.normalization_params,
                               device=device).detach()

        # Apply model to all samples, then stitch together the prediction
        all_preds = self.base_model(imgs)
        all_preds = all_preds.view(grid_shape[0], grid_shape[1], n_images,
                                   n_classes, height, width)

        del imgs

        localized_pred = torch.zeros(
            (n_images, n_classes, height, width)).to(device)

        for i, j in np.ndindex(*grid_shape):
            grid_pred = all_preds[i, j]

            grid_pred = crop_grid_predictions(grid_pred, i, j, grid_vertical,
                                              grid_horizontal, original_shape)

            localized_pred[:, :, y1:y2,
                           x1:x2][:, :, grid_vertical[i][0]:grid_vertical[i][1],
                                  grid_horizontal[j][0]:grid_horizontal[j]
                                  [1]] = grid_pred

        return localized_pred


def forward(self, *args, **kwargs):
    return self.base_model.forward(*args, **kwargs)
