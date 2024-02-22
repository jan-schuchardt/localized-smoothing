"""
This module contains functionality for evaluating smoothed classifiers
and their corresponding robustness certificates.

This functionality uses the certificate dictionaries
generated by method certify_dataset() in certification.py.

A nested certificate dictionary for N images has N+1 keys:
- The first key is "budget" and contains an array of B adversarial budgets
- The other keys are integer image ids
    - Each of these contains one dictionary per certificate (e.g. "argmax_holm")
        Each entry has the following keys:
            - For certificates starting with 'argmax':
                - "raw_confusion": A (C+1)x(C+1) confusion matrix
                    for the smoothed prediction on this image,
                    ignoring abstention by the smoothed classifier.
                    "Ignore" class C is an additional class that is never predicted.
                - "abstained_confusion": A (C+2)x(C+2)confusion matrix
                    for the smoothed prediction on this image,
                    with abstention of the smoothed classifier as an additional class C.
                    "Ignore" class C+1 is an additional class that is never predicted.
                - "naive": An array of shape Bx(C*2+2)x(C*2+2).
                    The first dimension corresponds to the budgets in "budget" (see above).
                    The trailing dimensions are a confusion matrix where the classes are:
                        C certified classes, C uncertified classes, Abstain, Ignore label.
                    The ground truth targets are always certified classes {0,...,C-1}.
                    The uncertified classes and the abstain class are never in the ground truth.
                    The predicions are either certified classes {0,...,C-1}, uncertified classes
                    {C,...,2*C-1} or abstain class 2*C.
                    The "ignore" class 2*C+1 is a class that is never predicted.
                - "collective" (only for certificates from certification_params['base_certs']):
                    A dict with four entries:
                    - "all": Array of length B,
                        with each entry corresponding to number of collectively
                        certified predictions when applying collective certificate to all
                        predictions that are not already certified by base certificate.
                    - "correct_only": Array of length B,
                        with each entry corresponding to number of collectively
                        certified predictions when applying collective certificate to all
                        correct predictions that are not already certified by base certificate.
                    - "solver_fails_all": Boollean array of length B, indicating if
                        applying cvxpy solver to all predictions failed.
                        If True, then entry in "all" should be 0.
                    - "solver_fails_correct_only": Boolean array of length B, indicating if
                        applying cvxpy_solver to all correct predictions failed.
                        If True, then entry in "correct_only" should be 0.
            - For certificates starting with "center"
                - "abstain": Whether the smoothed model abstains (True) or not (False)
                    from the entire segmentation mask.
                - "n_perturbed": 1D array of certified l_0 output distances,
                    with one entry per adversarial budget.

Note: At the start of the project, this seemed like a great way of encoding certification
results while simultaneously enabling the extraction of detailed per-class statistics ... 🙃

compute_overall_confusion: Aggregates per-image confusion matrices into a single confusion matrix.
compute_overall_certificate_confusions: Aggregates into per-budget certified confusion matrices.

calc_pixel_accuracy: Computes average pixel accuracy across multiple images.
count_abstains: Counts the number of abstentions across multiple images.
calc_mean_iou: Computes the average mIOU across multiple iamges.

calc_certified_ratios_naive: Computes per-budget certified ratios of naive collective certificate.
calc_certified_pixel_accuracy_naive: Computes per-budget cert. accuracies of naive collective cert.
calc_certified_ratios_center: Computes per-budget certified ratios of center smoothing.
calc_certified_pixel_accuracy_center: Computes per-budget certified accuracies of center smoothing.
calc_certified_ratios_collective: Computes per-budget certified ratios of localized smoothing.
calc_certified_pixel_accuracy_collective: Computes per-budget certified accuracy of localized.
"""
import numpy as np
from numpy.typing import NDArray

from localized_smoothing.segmentation.utils import iou_from_confusion
from localized_smoothing.utils import swap_entries


def compute_overall_confusion(cert_dict: dict,
                              cert_type: str = 'argmax_holm',
                              center: bool = False,
                              abstain: bool = True,
                              n_images: int = 100,
                              n_classes: int = 21) -> NDArray[np.int_]:
    """Aggregates per-image confusion matrices into a single confusion matrix.

    Args:
        cert_dict: See module description above.
        cert_type: cert_type, see seml/scripts/segmentation/cert.py
        center: Whether to evaluate center smoothed model or majority-vote-smoothed model.
        abstain: Whether we should count abstentions by the smoothed model.
        n_images: Index up to which per-image confusion amtrices should be aggregated.
        n_classes: Number of segmentation mask classes, without ignore label and abstain label.

    Returns:
        Aggregate confusion matrix of shape (n_classes + 2) x (n_classes + 2).
            Classes 0,...,n_classes - 1 correspond to the normal segmentation mask classes.
            Class n_classes is the "abstain" class, which only appears in model predictions.
            Class n_classes+1 is the "ignore" class, which only appears in target masks.

            Entry (i, j) is the number of times a pixel of class i was classified as class j.

            If abstain=False, then the second-to-last column and row are always 0.
    """

    img_ids = list(cert_dict.keys())[1:(n_images+1)]
    if 'config' in img_ids:
        img_ids.remove('config')
    overall_confusion = np.zeros((n_classes + 2, n_classes + 2))

    if not center:
        if abstain:
            iou_key = 'abstained_confusion'
            overall_confusion[:, :] = np.sum(
                [cert_dict[k][cert_type][iou_key] for k in img_ids], axis=0)

        else:
            iou_key = 'raw_confusion'
            overall_confusion[:n_classes + 1, :n_classes + 1] = np.sum(
                [cert_dict[k][cert_type][iou_key] for k in img_ids], axis=0)

            overall_confusion = swap_entries(overall_confusion, n_classes,
                                             n_classes + 1)
    else:
        if not abstain:
            overall_confusion[:n_classes + 1, :n_classes + 1] = np.sum(
                [cert_dict[k][cert_type]['raw_confusion'] for k in img_ids],
                axis=0)

            overall_confusion = swap_entries(overall_confusion, n_classes,
                                             n_classes + 1)
        else:
            for k in img_ids:
                conf = cert_dict[k][cert_type]['raw_confusion']
                if cert_dict[k][cert_type]['abstain']:
                    raise NotImplementedError(
                        'Verify that abstain is handled correctly'
                        'for center smoothing')

                    overall_confusion[:, -1] += conf.sum(axis=1)
                else:
                    overall_confusion[:n_classes + 1, :n_classes + 1] += conf

            overall_confusion = swap_entries(overall_confusion, n_classes,
                                             n_classes + 1)

    assert np.sum(overall_confusion[-2]) == 0  # Nothing is labeled "abstain"
    assert np.sum(overall_confusion[:, -1]) == 0  # We never predict 255 label

    return overall_confusion


def compute_overall_certificate_confusions(
        cert_dict: dict,
        cert_type: str = 'argmax_holm',
        n_images: int = 100,
        n_classes: int = 21) -> NDArray[np.int_]:
    """Aggregates per-image-per-adverarial-budget certified confusion matrices.

    Args:
        cert_dict: See module description above.
        cert_type: cert_type, see seml/scripts/segmentation/cert.py
        n_images: Index up to which per-image confusion amtrices should be aggregated.
        n_classes: Number of segmentation mask classes, without ignore label and abstain label.

    Returns:
        Per-budget aggregate certified confusion matrices of shape
            #budgets x (2 * n_classes + 2) x (2 * n_classes + 2).
            where first dimension corresponds to adversarial budget step and trailing dimensions
            are confusion matrices.

            Classes 0,...,n_classes - 1 correspond to certified classes.
                All ground truth targets are treated as certified classes.
            Classes n_classes,...,2*n_classes-1 correspond to uncertified classes,
                which only appear in model prediction.
            Class n_classes is the "abstain" class, which only appears in model predictions.
            Class n_classes+1 is the "ignore" class, which only appears in target masks.

            Entry (i, j) is the number of times a pixel of class i was classified as class j.
    """
    assert cert_type.startswith('argmax') or cert_type == 'cdf'

    img_ids = list(cert_dict.keys())[1:(n_images+1)]
    if 'config' in img_ids:
        img_ids.remove('config')
    overall_confusions = np.sum([cert_dict[k][cert_type]['naive'] for k in img_ids], axis=0)

    # None of the uncertified/abstain classes are ground truth labels
    assert np.sum(overall_confusions[:, n_classes:-1]) == 0

    # Label 255 is never predicted
    assert np.sum(overall_confusions[:, :, -1]) == 0

    assert np.all(overall_confusions.shape[1:] == np.array(
        [2 * n_classes + 2, 2 * n_classes + 2]))
    return overall_confusions


def calc_pixel_accuracy(cert_dict: dict,
                        cert_type: str = 'argmax_holm',
                        center: bool = False,
                        abstain: bool = True,
                        n_images: int = 100,
                        n_classes: int = 21) -> float:
    """Computes average pixel accuracy across multiple images, ignoring the "ignore" class.

    Args:
        cert_dict: See module description above.
        cert_type: cert_type, see seml/scripts/segmentation/cert.py
        center: Whether to evaluate center smoothed model or majority-vote-smoothed model.
        abstain: Whether we should count abstentions by the smoothed model.
        n_images: Index up to which per-image confusion amtrices should be aggregated.
        n_classes: Number of segmentation mask classes, without ignore label and abstain label.

    Returns:
        The average accuracy.
    """

    overall_confusion = compute_overall_confusion(cert_dict,
                                                  cert_type=cert_type,
                                                  center=center,
                                                  abstain=abstain,
                                                  n_images=n_images,
                                                  n_classes=n_classes)

    overall_confusion = overall_confusion[:-1, :
                                          -1]  # Ignore 255 label in evaluation
    true_positives = np.sum(np.diagonal(overall_confusion))
    return true_positives / overall_confusion.sum()


def count_abstains(cert_dict: dict,
                   cert_type: str = 'argmax_holm',
                   center: bool = False,
                   n_images: int = 100,
                   n_classes: int = 21) -> int:
    """Counts the number of abstentions across multiple segmented images.

    Args:
        cert_dict: See module description above.
        cert_type: cert_type, see seml/scripts/segmentation/cert.py
        center: Whether to evaluate center smoothed model or majority-vote-smoothed model.
        n_images: Index up to which per-image confusion amtrices should be aggregated.
        n_classes: Number of segmentation mask classes, without ignore label and abstain label.

    Returns:
        The summed number of abstentions.
    """

    overall_confusion = compute_overall_confusion(cert_dict,
                                                  cert_type=cert_type,
                                                  center=center,
                                                  abstain=True,
                                                  n_images=n_images,
                                                  n_classes=n_classes)

    return np.sum(overall_confusion[:, -2])


def calc_mean_iou(cert_dict: dict,
                  cert_type: str = 'argmax_holm',
                  center: bool = False,
                  abstain: bool = True,
                  n_images: int = 100,
                  n_classes: int = 21) -> float:
    """Computes the average mIOU across multiple segmented iamges.

    Args:
        cert_dict: See module description above.
        cert_type: cert_type, see seml/scripts/segmentation/cert.py
        center: Whether to evaluate center smoothed model or majority-vote-smoothed model.
        abstain: Whether we should count abstentions by the smoothed model.
        n_images: Index up to which per-image confusion amtrices should be aggregated.
        n_classes: Number of segmentation mask classes, without ignore label and abstain label.

    Returns:
        The summed number of abstentions.
    """

    overall_confusion = compute_overall_confusion(cert_dict,
                                                  cert_type=cert_type,
                                                  center=center,
                                                  abstain=abstain,
                                                  n_images=n_images,
                                                  n_classes=n_classes)
    overall_confusion = overall_confusion[:-1, :
                                          -1]  # Ignore 255 label in evaluation

    # Ignore non-existing abstain ground truth class
    per_class_iou = iou_from_confusion(overall_confusion)[:-1]
    return per_class_iou.mean()


def calc_certified_ratios_naive(cert_dict: dict,
                                cert_type: str = 'argmax_holm',
                                n_images: int = 100,
                                n_classes: int = 21) -> NDArray[np.float_]:
    """Computes per-budget certified ratios of naive collective certificate across multiple images.

    The certified ratio is the percentage of per-pixel predictions that are provably robust.
    Abstentions count as non-robust.
    Pixels with the "ignore" label are not counted.

    Args:
        cert_dict: See module description above.
        cert_type: cert_type, see seml/scripts/segmentation/cert.py
        n_images: Index up to which per-image confusion amtrices should be aggregated.
        n_classes: Number of segmentation mask classes, without ignore label and abstain label.

    Returns:
        A 1D array containing per-adversarial-budget certified ratios,
         for budgets specified in cert_dict['budgets'].
    """

    assert cert_type.startswith('argmax') or cert_type == 'cdf'

    # Note that all pixels with class 255 are generally ignored
    overall_confusions = compute_overall_certificate_confusions(
        cert_dict, cert_type, n_images=n_images, n_classes=n_classes)

    assert np.all(np.sum(overall_confusions[:, n_classes:-1, :], axis=2) == 0)
    assert np.all(np.sum(overall_confusions[:, :, -1], axis=1) == 0)

    n_certified = (
        np.sum(overall_confusions[:, :n_classes, :n_classes], axis=(1, 2)) +
        np.sum(overall_confusions[:, -1, :n_classes], axis=1))

    n_pred = np.sum(overall_confusions, axis=(1, 2))
    return n_certified / n_pred


def calc_certified_pixel_accuracy_naive(
        cert_dict: dict,
        cert_type: str = 'argmax_holm',
        n_images: int = 100,
        n_classes: int = 21) -> NDArray[np.float_]:
    """Computes per-budget cert. accuracies of naive collective certificate across multiple images.

    The cert. accuracy is the percentage of pixel predictions that are correct and provably robust.
    Abstentions count as non-robust.
    Pixels with the "ignore" label are not counted.

    Args:
        cert_dict: See module description above.
        cert_type: cert_type, see seml/scripts/segmentation/cert.py
        n_images: Index up to which per-image confusion amtrices should be aggregated.
        n_classes: Number of segmentation mask classes, without ignore label and abstain label.

    Returns:
        A 1D array containing per-adversarial-budget certified accuracies,
            for budgets specified in cert_dict['budgets'].
    """
    assert cert_type.startswith('argmax') or cert_type == 'cdf'

    overall_confusions = compute_overall_certificate_confusions(
        cert_dict, cert_type, n_images=n_images, n_classes=n_classes)

    assert np.all(np.sum(overall_confusions[:, n_classes:-1, :], axis=2) == 0)
    assert np.all(np.sum(overall_confusions[:, :, -1], axis=1) == 0)

    n_certified_correct = np.sum(np.diagonal(overall_confusions,
                                             axis1=1,
                                             axis2=2),
                                 axis=1)
    n_predictions = np.sum(overall_confusions[:, :-1],
                           axis=(1, 2))  # Ignore 255 label in computation

    return n_certified_correct / n_predictions


def calc_certified_ratios_center(
    cert_dict: dict,
    cert_type: str = 'center_bonferroni',
    n_images: int = 100,
    n_pixels: int = (166 * 250)) -> NDArray[np.float_]:
    """Computes per-budget certified ratios of center smoothing across multiple images.

    The certified ratio is the percentage of per-pixel predictions that are provably robust,
    which in the case of center smoothing is the certified l_0 output distance.
    If center smoothing abstains for an image, all predictions for that image count as non-robust.

    Args:
        cert_dict: See module description above.
        cert_type: cert_type, see seml/scripts/segmentation/cert.py
        n_images: Index up to which per-image confusion amtrices should be aggregated.
        n_pixels: The number of pixels in an image, i.e. H*W.

    Returns:
        A 1D array containing per-adversarial-budget certified ratios,
         for budgets specified in cert_dict['budgets'].
    """

    n_certified = 0
    n_predictions = 0

    img_ids = list(cert_dict.keys())[1:(n_images+1)]
    if 'config' in img_ids:
        img_ids.remove('config')

    for k in img_ids:
        abstain = cert_dict[k][cert_type]['abstain']
        n_predictions += n_pixels
        if not abstain:
            n_certified += np.maximum(
                0, n_pixels - cert_dict[k][cert_type]['n_perturbed'])

    return n_certified / n_predictions


def calc_certified_pixel_accuracy_center(
        cert_dict: dict,
        cert_type: str = 'center_bonferroni',
        n_images: int = 100,
        n_classes: int = 21) -> NDArray[np.float_]:
    """Computes per-budget certified accuracies of center smoothing across multiple images.

    The cert. accuracy is the percentage of pixel predictions that are correct and provably robust.

    Center smoothing does not provide per-pixel guarantees but just a number of robust predictions.
    We thus have to make the worst-case assumption that robust predictions are incorrect,
    unless the number of robust productions is larger than the number of incorrect predictions.

    If center smoothing abstains for an image, all predictions for that image count as non-robust.

    Pixels with the "ignore" label are not counted.

    Args:
        cert_dict: See module description above.
        cert_type: cert_type, see seml/scripts/segmentation/cert.py
        n_images: Index up to which per-image confusion amtrices should be aggregated.
        n_classes: Number of segmentation mask classes, without ignore label and abstain label.

    Returns:
        A 1D array containing per-adversarial-budget certified accuracies,
         for budgets specified in cert_dict['budgets'].
    """

    n_certified_correct = 0
    n_predictions = 0

    img_ids = list(cert_dict.keys())[1:(n_images+1)]
    if 'config' in img_ids:
        img_ids.remove('config')

    for k in img_ids:
        conf = cert_dict[k][cert_type]['raw_confusion']
        abstain = cert_dict[k][cert_type]['abstain']
        assert np.all(conf.shape == np.array([n_classes + 1, n_classes + 1]))
        n_predictions += np.sum(conf[:-1])  # Ignore 255 label
        if not abstain:
            n_perturbed = cert_dict[k][cert_type]['n_perturbed']

            n_correct = np.sum(np.diagonal(conf))
            # Worst-case assumption, that we first certify incorrect predictions
            n_certified_correct += np.maximum(0, n_correct - n_perturbed)

    return n_certified_correct / n_predictions


def calc_certified_ratios_collective(cert_dict: dict,
                                     cert_type: str = 'argmax_holm',
                                     collective_all: bool = True,
                                     collective_correct: bool = False,
                                     n_images: int = 100,
                                     n_classes: int = 21) -> NDArray[np.float_]:
    """Computes per-budget certified ratios of localized smoothing across multiple images.

    The certified ratio is the percentage of per-pixel predictions that are provably robust.

    The number of provably robust predictions is the number of predictions that are
    robust according to the base certificates, plus the number of predictions that are
    additionally certified by the collective linear program.

    Args:
        cert_dict: See module description above.
        cert_type: Base certificate type, see seml/scripts/segmentation/cert.py
        collective_all: If True, evaluate collective LP applied to all predictions.
            This XOR collective_correct must be True.
        collective_correct: If False, evaluate collective LP applied to correct predictions.
            This XOR collective_all must be True.
        n_images: Index up to which per-image confusion amtrices should be aggregated.
        n_classes: Number of segmentation mask classes, without ignore label and abstain label.

    Returns:
        A 1D array containing per-adversarial-budget certified ratios,
         for budgets specified in cert_dict['budgets'].
    """

    assert cert_type.startswith('argmax') or cert_type == 'cdf'

    assert collective_all ^ collective_correct
    if collective_all:
        collective_key = 'all'
    if collective_correct:
        collective_key = 'correct_only'

    n_pred = 0
    n_certified = 0

    img_ids = list(cert_dict.keys())[1:(n_images+1)]
    if 'config' in img_ids:
        img_ids.remove('config')

    for k in img_ids:
        conf = cert_dict[k][cert_type]['naive']
        assert (conf[:, :, -1].sum() == 0)
        assert (conf[:, n_classes:-1, :].sum() == 0)
        assert np.all(conf.shape[1:] == np.array(2 * [2 * n_classes + 2]))

        n_pred_img = conf.sum(axis=(1, 2))
        n_certified_img = np.sum(conf[:, :, :n_classes], axis=(1, 2))

        x = cert_dict[k][cert_type]['collective'][collective_key].copy()

        # LP yields floats, we can round up to next integer
        # except for very small numbers.
        x[x < 0.0001] = 0
        n_certified_img = np.ceil(n_certified_img + x)

        assert np.all(n_certified_img <= n_pred_img)

        n_pred += n_pred_img
        n_certified += n_certified_img

    return n_certified / n_pred


def calc_certified_pixel_accuracy_collective(
        cert_dict: dict,
        cert_type: str = 'argmax_holm',
        collective_all: bool = True,
        collective_correct: bool = False,
        n_images=100,
        n_classes=21) -> NDArray[np.float_]:
    """Computes per-budget certified accuracy of localized smoothing across multiple images.

    The cert. accuracy is the percentage of pixel predictions that are correct and provably robust.

    The number of provably robust predictions is the number of predictions that are
    robust according to the base certificates, plus the number of predictions that are
    additionally certified by the collective linear program.

    Pixels with the "ignore" label are not counted.
    Abstentions count as incorrect.

    If we apply the collective LP to all predictions that are not certified by the base certificate
    (collective_all), instead of only correct predictions (collective_correct), we have
    to make the same worst-case assumption as center smoothing:
    The robust predictions are incorrect predictions, unless the number of robust predictions
    is larger than the number of incorrect predictions.

    Args:
        cert_dict: See module description above.
        cert_type: Base certificate type, see seml/scripts/segmentation/cert.py
        collective_all: If True, evaluate collective LP applied to all predictions.
            This XOR collective_correct must be True.
        collective_correct: If False, evaluate collective LP applied to correct predictions.
            This XOR collective_all must be True.
        n_images: Index up to which per-image confusion amtrices should be aggregated.
        n_classes: Number of segmentation mask classes, without ignore label and abstain label.

    Returns:
        A 1D array containing per-adversarial-budget certified accuracies,
         for budgets specified in cert_dict['budgets'].
    """

    assert collective_all ^ collective_correct

    n_pred = 0
    n_certified_correct = 0

    img_ids = list(cert_dict.keys())[1:(n_images+1)]
    if 'config' in img_ids:
        img_ids.remove('config')

    for k in img_ids:
        conf = cert_dict[k][cert_type]['naive']
        assert (conf[:, :, -1].sum() == 0)
        assert (conf[:, n_classes:-1, :].sum() == 0)
        assert np.all(conf.shape[1:] == np.array(2 * [2 * n_classes + 2]))

        n_pred_img = conf[:, :-1, :].sum(axis=(1, 2))  # Ignore label 255
        n_certified_correct_img = np.sum(np.diagonal(conf, axis1=1, axis2=2),
                                         axis=1)

        if collective_correct:
            x = cert_dict[k][cert_type]['collective']['correct_only'].copy()

            x[x < 0.0001] = 0

            # Certified correct predictions are those from base certificates
            # plus those from collective LP
            n_certified_correct_img = np.ceil(n_certified_correct_img + x)

        else:
            x = cert_dict[k][cert_type]['collective']['all'].copy()
            x[x < 0.0001] = 0
            n_certified_collective = np.ceil(x)

            # Incorrect and uncertified in labeled region
            # -2, because we never make abstaining predictions targets of the collective cert
            n_uncertified_incorrect_img = (
                conf[:, :n_classes, n_classes:-2].sum(axis=(1, 2)) -
                np.sum(np.diagonal(
                    conf[:, :n_classes, n_classes:-2], axis1=1, axis2=2),
                       axis=1))

            # uncertified in unlabeled region. TODO: Rename to something that better captures both
            # unlabeled + non-robust and labeled + incorrect + non-robust
            n_uncertified_incorrect_img += conf[:, -1, n_classes:-2].sum()

            mask = n_certified_collective > n_uncertified_incorrect_img

            n_certified_correct_img[mask] = (n_certified_correct_img[mask] +
                                             n_certified_collective[mask])

        n_pred += n_pred_img
        n_certified_correct += n_certified_correct_img

    return n_certified_correct / n_pred
