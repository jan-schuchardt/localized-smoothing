"""This module contains the main training loop and its subroutines.

train: The main training loop, including validation.
train_epoch: Training for a single epoch.

update: A single update step based on applying the base model to a single minibatch.
update_stitch_smooth_predictions: A single update step based on applying
    a locally smoothed model (with a single MC sample) to a single minibatch.
prepare_stitching_params: Prepares additional parameters for update_stitch_smooth_predictions

validate: Evaluation of the model on the entire validation dataset.
evaluate_unpadded_prediction: Evaluation of a single batch of predictions, after removing padding.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from segmentation_models_pytorch.losses import DiceLoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from localized_smoothing.segmentation.smoothing import SmoothedModel

from .utils import confusion_matrix, iou_from_confusion, remove_padding


def train(
    model: nn.Module,
    data_train: Dataset,
    data_val: Dataset,
    num_epochs: int,
    lr: float,
    batch_size: int,
    train_encoder: bool,
    epochs_val: int,
    stitch_smooth_predictions: bool,
    accumulate_gradients: bool,
    device: torch.device,
    max_batch_size: int | None = None,
    smoothing_params: dict | None = None,
    normalization_params: Dict[str, torch.Tensor] | None = None,
    ignore_index: int | None = 255,
    num_workers: int = 5
) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
    """"Trains and validates a model for multiple epochs.

    There are two training modes for locally smoothed models, which
    are chosen by the stitch_smooth_predictions.
    1.) We train the base model on a normal batch of image, target pairs
        (which are potentially preprocessed, e.g. using the LocalizedSmoothingTransform)
        with the usual DiceLoss.
    2.) We train the locally smoothed model using a single Monte Carlo sample,
        i.e. per grid cell we take one sample from the localized smoothign distribution
        and then stitch together the predictions (see Fig. 1).

    For the sake of memory efficiency, the locally smoothed model can also be trained
    via virtual minibatching, i.e. splitting up each batch into microbatches
    whose gradients are accumulated before optimizer.step().

    Args:
        model: The pytorch module we want to train.
        data_train: The training data.
        data_val: The validation data.
        num_epochs: The number of epochs (i.e. passes over the entire dataset) to train for.
        lr: Learning rate for the Adam optimizer.
        batch_size: The batch size for each update step.
        train_encoder: Whether to also train the segmentation model backbone.
        epochs_val: The periodic time for evaluating the model on the validation dataset.
        stitch_smooth_predictions: If True, train locally smoothed model (see above).
        accumulate_gradients: If True, use virtual minibatching for training locally smoothed model.
        device: The device to move models and data to.
        max_batch_size: The maximum physical batch size to use (this controls into how many
            computational microbatches the logical minibatches are split for virtual minibatching).
            Must be passed when stitch_smooth_predictions==True.
        smoothing_params: Dict of smoothing parameters, see seml/scripts/segmentation/train.py.
            Must be passed when stitch_smooth_predictions==True.
        normalization_params: Dictionary with two keys "mean" and "std", each corresponding
            to Tensor of shape [3] indicating channelwise mean and std for z-normalization.
        ignore_index: Optional ignore index for Diceless. Target pixels with this label
            are ignored by loss computation.
        num_workers: Number of workers to use in DataLoader.
            0 means loading is performed in the main process.

    Returns:
        A tuple of length 5, containing
            - Model state dict with the model parameters that achieved the lowest validation loss.
            - Model state dict with the model parameters that achieved the highest validation mIOU.
            - 2D array containing training losses,
                with first dimension corresponding to epochs and second dimension to minibatches.
            - 1D array containing validation losses for epochs in which validation was performed.
            - 1D array containing validation mIOU for epochs in which validation was performed.
    """

    loader_train = DataLoader(data_train,
                              drop_last=True,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    loader_val = DataLoader(data_val,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers)

    if not train_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    loss_fn = DiceLoss('multiclass', ignore_index=ignore_index)
    optimizer = torch.optim.Adam(model.parameters())

    losses_train = []
    losses_val = []
    ious_val = []

    for epoch in tqdm(range(num_epochs)):
        epoch_losses_train = train_epoch(model, loss_fn, optimizer,
                                         loader_train, batch_size,
                                         stitch_smooth_predictions,
                                         accumulate_gradients, device,
                                         max_batch_size, smoothing_params,
                                         normalization_params)

        losses_train.append(epoch_losses_train)

        if (epoch == 0) or ((epoch + 1) % epochs_val == 0):

            loss_val, iou_val = validate(model, loss_fn, loader_val, batch_size,
                                         stitch_smooth_predictions,
                                         smoothing_params, normalization_params,
                                         device)

            if epoch == 0 or loss_val < min(losses_val):
                best_state_loss = {
                    key: value.cpu()
                    for key, value in model.state_dict().items()
                }

            if epoch == 0 or iou_val.mean() > max([i.mean() for i in ious_val]):
                best_state_iou = {
                    key: value.cpu()
                    for key, value in model.state_dict().items()
                }

            losses_val.append(loss_val)
            ious_val.append(iou_val)

    model.load_state_dict(best_state_loss)

    return (best_state_loss, best_state_iou, np.array(losses_train),
            np.array(losses_val), np.array(ious_val))


def train_epoch(
        model: nn.Module,
        loss_fn: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        loader_train: DataLoader,
        batch_size: int,
        stitch_smooth_predictions: bool,
        accumulate_gradients: bool,
        device: torch.device,
        max_batch_size: int | None = None,
        smoothing_params: dict | None = None,
        normalization_params: Dict[str, torch.Tensor] | None = None
) -> list[float]:
    """Trains a model for a single epoch.

    There are two training modes for locally smoothed models, which
    are chosen by the stitch_smooth_predictions.
    1.) We train the base model on a normal batch of image, target pairs
        (which are potentially preprocessed, e.g. using the LocalizedSmoothingTransform)
        with the usual DiceLoss.
    2.) We train the locally smoothed model using a single Monte Carlo sample,
        i.e. per grid cell we take one sample from the localized smoothign distribution
        and then stitch together the predictions (see Fig. 1).

    For the sake of memory efficiency, the locally smoothed model can also be trained
    via virtual minibatching, i.e. splitting up each batch into microbatches
    whose gradients are accumulated before optimizer.step().

    Args:
        model: The pytorch module we want to train.
        loss_fn: The loss function to use for training.
        optimizer: The optimizr for updating the model parameters.
        loader_train: Dataloader that yields training minibatches.
        batch_size: Batch size yielded by the Dataloader.
        stitch_smooth_predictions: If True, train locally smoothed model (see above).
        accumulate_gradients: If True, use virtual minibatching for training locally smoothed model.
        device: The device to move models and data to.
        max_batch_size: The maximum physical batch size to use (this controls into how many
            computational microbatches the logical minibatches are split for virtual minibatching).
            Must be passed when stitch_smooth_predictions==True.
        smoothing_params: Dict of smoothing parameters, see seml/scripts/segmentation/train.py.
            Must be passed when stitch_smooth_predictions==True.
        normalization_params: Dictionary with two keys "mean" and "std", each corresponding
            to Tensor of shape [3] indicating channelwise mean and std for z-normalization.

    Returns:
        A list of per-minibatch losses.
    """

    model.train()

    epoch_losses_train = []

    if stitch_smooth_predictions:
        microbatch_size, n_microbatches, distribution_params = prepare_stitching_params(
            smoothing_params, batch_size, max_batch_size)

        smoothed_model = SmoothedModel(model, None, distribution_params,
                                       normalization_params, device)

    for x, y in loader_train:
        optimizer.zero_grad()
        y = y.to(device)

        assert x.shape[0] == batch_size

        if stitch_smooth_predictions:
            loss = update_stitch_smooth_predictions(
                x, y, smoothed_model, loss_fn, optimizer, batch_size,
                accumulate_gradients, n_microbatches, microbatch_size, device)

        else:
            loss = update(x, y, model, loss_fn, optimizer, device)

        epoch_losses_train.append(loss)

    return epoch_losses_train


def update(x: torch.Tensor, y: torch.Tensor, model: nn.Module,
           loss_fn: torch.nn.modules.loss._Loss,
           optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """Updates model parameters using a single minibatch in normal training mode.

    Returns:
        The loss before updating model parameters.
    """
    x = x.to(device)
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().numpy()


def update_stitch_smooth_predictions(x: torch.Tensor, y: torch.Tensor,
                                     smoothed_model: SmoothedModel,
                                     loss_fn: torch.nn.modules.loss._Loss,
                                     optimizer: torch.optim.Optimizer,
                                     batch_size: int,
                                     accumulate_gradients: bool,
                                     n_microbatches: int, microbatch_size: int,
                                     device: torch.device) -> float:
    """Updates model parameters using locally smoothed prediction with single MC sample.

    As shown in Fig. 1, we sample from a different localized distribution for each grid cell,
    segment the samples separately and then stitch together the final prediction from
    the different grid cell segmentation masks.

    Args:
        x: Minibatch with B input images
        y: Corresponding B ground-truth segmentation masks
        smoothed_model: Model, whose base model we want to update.
        loss_fn: Loss function for comparing y and prediction.
        optimizer: The optimizer for the model's parameters.
        batch_size: Size of batches yielded by the DataLoader.
        accumulate_gradients: If True, use virtual minibatching for training locally smoothed model.
        n_microbatches: Number of microbatches to split the large minibatch into.
        microbatch_size: Size of the microbatches (last one may be smaller).
        device: The device to move inputs and models to.

    Returns:
        If accumulate_gradients: The minibatch loss before updating the model parameters.
        Else: The last microbatch loss before updating the model parameters.
    """

    input_remainder = x
    label_remainder = y
    for _ in range(n_microbatches):
        if not accumulate_gradients:
            # Without virtual minibatching, we need to reset the gradients each backward pass.
            optimizer.zero_grad()

        # Get next microbatch
        x_micro = input_remainder[:microbatch_size]
        input_remainder = input_remainder[microbatch_size:]
        y_micro = label_remainder[:microbatch_size]
        label_remainder = label_remainder[microbatch_size:]

        pred = smoothed_model.sample_stitched_base_model_logits_batched(
            x_micro, 1, device)

        loss = loss_fn(pred, y_micro)

        if accumulate_gradients:
            # With virtual minibatching, need to scale loss so that we are training on average
            # loss of overall minibatch.
            loss *= x_micro.shape[0] / batch_size

        loss.backward()

        if not accumulate_gradients:
            optimizer.step()

    if accumulate_gradients:
        # With virtual minibatching, we only update model parameters after accumulating gradients
        # for the entire minibatch.
        optimizer.step()

    return loss.detach().cpu().numpy()


def prepare_stitching_params(
        smoothing_params: dict,
        batch_size: int,
        max_batch_size: int | None = None) -> tuple[int, int, dict]:
    """Computes additional parameters for update_stitch_smooth_predictions.

    Args:
        smoothing_params: Dict of smoothing parameters, see seml/scripts/segmentation/train.py.
        batch_size: The batch_size yielded by the DataLoader.
        max_batch_size: The maximum physical batch size to use.

    Returns:
        Tuple with three elements:
            - The size of the microbatches (batch_size // max_batch_size)
            - The number of microbatches in a single minibatch
            - The localized smoothing distribution parameters from smoothing_params,
                see seml/scripts/segmentation/train.py.
    """

    if smoothing_params['localized_params']['label_mask_distance'] is not None:
        raise NotImplementedError(
            'stitch_smooth_predictions does not support label masking yet')
    if smoothing_params['localized']:
        distribution_params = smoothing_params['localized_params'][
            'distribution_params']
        n_grid_cells = np.prod(distribution_params['grid_shape'])
    else:
        raise ValueError(
            'stitch_smooth_predictions only meant for use with localized distribution'
        )

    if max_batch_size is None:
        max_batch_size = batch_size

    microbatch_size = int(min(batch_size, max_batch_size // n_grid_cells))
    n_microbatches = int(np.ceil(batch_size / microbatch_size))

    return microbatch_size, n_microbatches, distribution_params


def validate(model: nn.Module, loss_fn: torch.nn.modules.loss._Loss,
             loader_val: DataLoader, batch_size: int,
             stitch_smooth_predictions: bool, smoothing_params: dict,
             normalization_params: Dict[str, torch.Tensor],
             device: torch.device) -> tuple[float, float]:
    """Evaluates model predictions on validation data.

    There are two evaluation modes, which are chosen by the stitch_smooth_predictions.
    1.) We apply the model to normal batches of image, target pairs
        (which are potentially preprocessed, e.g. using the LocalizedSmoothingTransform).
    2.) We evaluate the locally smoothed model using a single Monte Carlo sample,
        i.e. per grid cell we take one sample from the localized smoothign distribution
        and then stitch together the predictions (see Fig. 1).

    Args:
        model: Model to evaluate.
        loss_fn: Loss function for comparing y and prediction.
        loader_val: The dataloader for validation data.
        batch_size: Size of batches yielded by the DataLoader.
        stitch_smooth_predictions: If True, evaluate locally smoothed model (see above).
        smoothing_params: Dict of smoothing parameters, see seml/scripts/segmentation/train.py.
            Must be passed when stitch_smooth_predictions==True.
        normalization_params: Dictionary with two keys "mean" and "std", each corresponding
            to Tensor of shape [3] indicating channelwise mean and std for z-normalization.
        device: The device to move inputs and models to.

    Returns:
        Tuple of average validation loss and average mIOU.
    """
    with torch.no_grad():
        model.eval()

        overall_loss = 0
        overall_conf_matrix = None

        if stitch_smooth_predictions:
            _, _, distribution_params = prepare_stitching_params(
                smoothing_params, batch_size)

            smoothed_model = SmoothedModel(model, None, distribution_params,
                                           normalization_params, device)

        for x, y, original_shape in loader_val:
            x = x.to(device)
            y = y.to(device)

            if stitch_smooth_predictions:
                pred = smoothed_model.sample_stitched_base_model_logits_batched(
                    x, 1, device, original_shape[0].numpy())
            else:
                pred = model(x)

            sample_loss, sample_conf_matrix = evaluate_unpadded_prediction(
                pred, y, model.classes, loss_fn, original_shape)

            overall_loss += sample_loss / len(loader_val)

            assert np.all(sample_conf_matrix.shape == np.array(
                [model.classes, model.classes]))

            if overall_conf_matrix is None:
                overall_conf_matrix = sample_conf_matrix
            else:
                overall_conf_matrix += sample_conf_matrix

        overall_iou = iou_from_confusion(overall_conf_matrix)

        return overall_loss, overall_iou


def evaluate_unpadded_prediction(
        pred: torch.Tensor, y: torch.Tensor, n_classes: int,
        loss_fn: torch.nn.modules.loss._Loss,
        original_shape: NDArray[np.int_]) -> tuple[float, NDArray[np.int_]]:
    """Removes padding from prediction and target before evaluating loss and mIOU.

    Pytorch segmentation models usually require some input padding, which means that predicted
    segmentation mask will also have padding, which we remove here.

    Args:
        pred: Prediction of shape [B, K, H, W]
        y: Target classes of shape [B, H, W]
        n_classes: Number of classes for segmentation masks.
        loss_fn: The loss function to use for evaluation.
        original_shape: Length 2 array [height, width] of image before padding.

    Returns:
        Tuple consisting of prediction loss and confusion matrix.
    """
    assert y.shape[1:] == pred.shape[2:]

    pred, y = remove_padding(pred, y, original_shape)
    pred = pred.contiguous()
    y = y.contiguous()
    loss = loss_fn(pred, y)

    pred = pred.detach().cpu().numpy()
    pred = pred[0].argmax(axis=0)
    y = y[0].cpu().numpy()

    conf_matrix = confusion_matrix(
        y, pred, labels=np.arange(n_classes))[:-1, :-1]  # Ignore label 255

    return loss.detach().cpu(), conf_matrix
