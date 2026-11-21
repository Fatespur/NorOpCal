"""Image-level anomaly metrics with explicit threshold semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


class MetricError(ValueError):
    """Report invalid labels, scores, shapes, or threshold inputs."""


NumericInput = ArrayLike | torch.Tensor


def to_numpy(values: NumericInput, *, dtype: np.dtype[Any]) -> NDArray[Any]:
    """Convert an array-like or tensor to a detached CPU NumPy array.

    Args:
        values: Numerical array-like values.
        dtype: Required NumPy dtype.

    Returns:
        A copied NumPy array.
    """

    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(dtype, copy=True)
    return np.asarray(values, dtype=dtype).copy()


def binary_vectors(
    labels: NumericInput,
    scores: NumericInput,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Validate aligned one-dimensional binary labels and finite scores.

    Args:
        labels: Binary ground-truth labels.
        scores: Continuous anomaly scores.

    Returns:
        Validated integer labels and float scores.

    Raises:
        MetricError: If shapes, labels, values, or class support are invalid.
    """

    label_array = to_numpy(labels, dtype=np.dtype(np.int64)).reshape(-1)
    score_array = to_numpy(scores, dtype=np.dtype(np.float64)).reshape(-1)
    if label_array.size == 0 or label_array.shape != score_array.shape:
        raise MetricError(
            f"Invalid image metric inputs: labels shape {label_array.shape}, "
            f"scores shape {score_array.shape}; expected equal non-empty vectors."
        )
    if not np.all(np.isfinite(score_array)):
        raise MetricError("Invalid image scores: expected only finite values.")
    classes = set(np.unique(label_array).tolist())
    if not classes <= {0, 1}:
        raise MetricError(f"Invalid labels {sorted(classes)}; expected binary values 0 and 1.")
    if classes != {0, 1}:
        raise MetricError(
            "Image AUROC and AUPR require both normal and anomalous labels; "
            f"received classes {sorted(classes)}."
        )
    return label_array, score_array


@dataclass(frozen=True)
class ImageMetrics:
    """Store threshold-free and threshold-dependent image metrics."""

    auroc: float
    aupr: float
    threshold: float
    f1: float
    precision: float
    recall: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable metric mapping."""

        return asdict(self)


def compute_image_metrics(
    labels: NumericInput,
    scores: NumericInput,
    *,
    threshold: float,
) -> ImageMetrics:
    """Calculate image metrics using a preselected validation threshold.

    Args:
        labels: Binary final-evaluation labels.
        scores: Continuous image anomaly scores.
        threshold: Threshold selected without final-test optimization. Scores
            greater than or equal to it are anomalous.

    Returns:
        Threshold-free AUROC/AUPR and threshold-dependent classification metrics.

    Raises:
        MetricError: If inputs are invalid or the threshold is not finite.

    Assumptions:
        The caller is responsible for calibration provenance. This function
        never tunes a threshold on its input labels.
    """

    label_array, score_array = binary_vectors(labels, scores)
    if not np.isfinite(threshold):
        raise MetricError("Invalid image threshold: expected a finite value.")
    predictions = (score_array >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        label_array,
        predictions,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(label_array, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp))
    false_positive_rate = float(fp / (tn + fp))
    false_negative_rate = float(fn / (fn + tp))
    return ImageMetrics(
        auroc=float(roc_auc_score(label_array, score_array)),
        aupr=float(average_precision_score(label_array, score_array)),
        threshold=float(threshold),
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
        specificity=specificity,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
    )
