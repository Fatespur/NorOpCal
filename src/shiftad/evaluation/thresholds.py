"""Validation-only threshold calibration protocols."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import f1_score

from shiftad.evaluation.image_metrics import MetricError, NumericInput, binary_vectors, to_numpy


@dataclass(frozen=True)
class ThresholdCalibration:
    """Record a selected threshold and its validation provenance."""

    threshold: float
    method: str
    validation_size: int
    validation_false_positive_rate: float
    quantile: float | None = None
    validation_f1: float | None = None

    def to_dict(self) -> dict[str, float | int | str | None]:
        """Return a JSON-serializable calibration record."""

        return asdict(self)


def calibrate_normal_quantile(
    normal_validation_scores: NumericInput,
    *,
    quantile: float = 0.99,
) -> ThresholdCalibration:
    """Select a threshold from normal validation scores only.

    Args:
        normal_validation_scores: Scores known to come from a normal validation
            split, never the final test set.
        quantile: Quantile in the open interval ``(0, 1)``.

    Returns:
        Threshold and observed normal validation false-positive rate.

    Raises:
        MetricError: If scores or the quantile are invalid.

    Assumptions:
        Scores greater than or equal to the threshold are classified anomalous.
    """

    scores = to_numpy(
        normal_validation_scores,
        dtype=np.dtype(np.float64),
    ).reshape(-1)
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise MetricError(
            "Normal quantile calibration requires a non-empty vector of finite scores."
        )
    if not 0.0 < quantile < 1.0:
        raise MetricError("Calibration quantile must be strictly between 0 and 1.")
    threshold = float(np.quantile(scores, quantile, method="higher"))
    false_positive_rate = float(np.mean(scores >= threshold))
    return ThresholdCalibration(
        threshold=threshold,
        method="normal_quantile_higher",
        validation_size=int(scores.size),
        validation_false_positive_rate=false_positive_rate,
        quantile=quantile,
    )


def calibrate_best_f1(
    validation_labels: NumericInput,
    validation_scores: NumericInput,
) -> ThresholdCalibration:
    """Select the best F1 threshold on an explicit validation split.

    Args:
        validation_labels: Binary validation labels.
        validation_scores: Continuous validation scores.

    Returns:
        The highest threshold among F1-optimal candidates and its validation
        diagnostics.

    Raises:
        MetricError: If the validation vectors are invalid.

    Assumptions:
        Passing final-test data violates the protocol; the explicit parameter
        names and persisted method make that misuse auditable.
    """

    labels, scores = binary_vectors(validation_labels, validation_scores)
    best_threshold = float(scores.min())
    best_f1 = -1.0
    for candidate in np.unique(scores):
        predictions = (scores >= candidate).astype(np.int64)
        candidate_f1 = float(f1_score(labels, predictions, zero_division=0))
        if candidate_f1 > best_f1 or (candidate_f1 == best_f1 and candidate > best_threshold):
            best_threshold = float(candidate)
            best_f1 = candidate_f1
    normal_scores = scores[labels == 0]
    false_positive_rate = float(np.mean(normal_scores >= best_threshold))
    return ThresholdCalibration(
        threshold=best_threshold,
        method="validation_best_f1",
        validation_size=int(scores.size),
        validation_false_positive_rate=false_positive_rate,
        validation_f1=best_f1,
    )
