"""Pixel-level metrics, including region-overlap AUPRO."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import auc, average_precision_score, roc_auc_score

from shiftad.evaluation.image_metrics import MetricError, NumericInput, to_numpy


def _pixel_arrays(
    masks: NumericInput,
    anomaly_maps: NumericInput,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    mask_array = to_numpy(masks, dtype=np.dtype(np.float64))
    score_array = to_numpy(anomaly_maps, dtype=np.dtype(np.float64))
    if mask_array.ndim == 2:
        mask_array = mask_array[None, ...]
    if score_array.ndim == 2:
        score_array = score_array[None, ...]
    if mask_array.ndim == 4 and mask_array.shape[1] == 1:
        mask_array = mask_array[:, 0]
    if score_array.ndim == 4 and score_array.shape[1] == 1:
        score_array = score_array[:, 0]
    if mask_array.shape != score_array.shape or mask_array.ndim != 3:
        raise MetricError(
            f"Invalid pixel metric shapes: masks {mask_array.shape}, maps "
            f"{score_array.shape}; expected equal [N, H, W] or [N, 1, H, W] arrays."
        )
    if mask_array.size == 0 or not np.all(np.isfinite(score_array)):
        raise MetricError("Pixel metrics require non-empty finite anomaly maps.")
    unique_masks = set(np.unique(mask_array).tolist())
    if not unique_masks <= {0.0, 1.0}:
        raise MetricError(
            f"Invalid mask values {sorted(unique_masks)}; expected binary values 0 and 1."
        )
    bool_masks = mask_array.astype(np.bool_)
    if not np.any(bool_masks) or np.all(bool_masks):
        raise MetricError("Pixel AUROC and AUPR require both background and anomalous mask pixels.")
    return bool_masks, score_array


def _connected_components(mask: NDArray[np.bool_]) -> list[NDArray[np.bool_]]:
    height, width = mask.shape
    visited = np.zeros_like(mask)
    components: list[NDArray[np.bool_]] = []
    for row, column in np.argwhere(mask):
        row_index = int(row)
        column_index = int(column)
        if visited[row_index, column_index]:
            continue
        component = np.zeros_like(mask)
        queue: deque[tuple[int, int]] = deque([(row_index, column_index)])
        visited[row_index, column_index] = True
        while queue:
            current_row, current_column = queue.popleft()
            component[current_row, current_column] = True
            for neighbor_row, neighbor_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= neighbor_row < height
                    and 0 <= neighbor_column < width
                    and mask[neighbor_row, neighbor_column]
                    and not visited[neighbor_row, neighbor_column]
                ):
                    visited[neighbor_row, neighbor_column] = True
                    queue.append((neighbor_row, neighbor_column))
        components.append(component)
    return components


def compute_aupro(
    masks: NumericInput,
    anomaly_maps: NumericInput,
    *,
    max_false_positive_rate: float = 0.3,
    num_thresholds: int = 200,
) -> float:
    """Compute normalized area under the per-region-overlap curve.

    Args:
        masks: Binary masks shaped ``[N, H, W]`` or ``[N, 1, H, W]``.
        anomaly_maps: Continuous pixel scores with the same shape.
        max_false_positive_rate: Background FPR integration limit.
        num_thresholds: Maximum number of sampled score thresholds.

    Returns:
        AUPRO normalized to ``[0, 1]`` over the configured FPR range.

    Raises:
        MetricError: If masks, scores, regions, background, or parameters are
            invalid.

    Assumptions:
        Four-connected foreground components define individual defect regions.
        FPR is measured over all background pixels across the evaluated set.
    """

    mask_array, score_array = _pixel_arrays(masks, anomaly_maps)
    if not 0.0 < max_false_positive_rate <= 1.0:
        raise MetricError("AUPRO max_false_positive_rate must be in (0, 1].")
    if num_thresholds < 2:
        raise MetricError("AUPRO num_thresholds must be at least 2.")
    regions: list[tuple[int, NDArray[np.bool_]]] = []
    for image_index, mask in enumerate(mask_array):
        regions.extend((image_index, component) for component in _connected_components(mask))
    if not regions:
        raise MetricError("AUPRO requires at least one connected anomaly region.")

    unique_scores = np.unique(score_array)
    if unique_scores.size > num_thresholds:
        thresholds = np.linspace(
            float(unique_scores.max()),
            float(unique_scores.min()),
            num=num_thresholds,
        )
    else:
        thresholds = unique_scores[::-1]

    background = ~mask_array
    background_count = int(background.sum())
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for threshold in thresholds:
        predicted = score_array >= threshold
        false_positive_rate = float(np.logical_and(predicted, background).sum())
        false_positive_rate /= background_count
        overlaps = [
            float(np.logical_and(predicted[image_index], region).sum() / region.sum())
            for image_index, region in regions
        ]
        points.append((false_positive_rate, float(np.mean(overlaps))))

    by_fpr: dict[float, float] = {}
    for false_positive_rate, overlap in points:
        by_fpr[false_positive_rate] = max(by_fpr.get(false_positive_rate, 0.0), overlap)
    sorted_points = sorted(by_fpr.items())
    x_all = np.asarray([point[0] for point in sorted_points], dtype=np.float64)
    y_all = np.asarray([point[1] for point in sorted_points], dtype=np.float64)

    within = x_all <= max_false_positive_rate
    x_curve = x_all[within].tolist()
    y_curve = y_all[within].tolist()
    if x_curve[-1] < max_false_positive_rate:
        upper_index = int(np.searchsorted(x_all, max_false_positive_rate, side="right"))
        lower_index = upper_index - 1
        if upper_index >= len(x_all):
            interpolated = y_all[-1]
        else:
            interpolated = np.interp(
                max_false_positive_rate,
                x_all[[lower_index, upper_index]],
                y_all[[lower_index, upper_index]],
            )
        x_curve.append(max_false_positive_rate)
        y_curve.append(float(interpolated))
    normalized_area = float(auc(x_curve, y_curve) / max_false_positive_rate)
    return float(np.clip(normalized_area, 0.0, 1.0))


@dataclass(frozen=True)
class PixelMetrics:
    """Store threshold-free and threshold-dependent pixel metrics."""

    auroc: float
    aupr: float
    threshold: float
    iou: float
    dice: float
    aupro: float
    aupro_max_false_positive_rate: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable pixel metric mapping."""

        return asdict(self)


def compute_pixel_metrics(
    masks: NumericInput,
    anomaly_maps: NumericInput,
    *,
    threshold: float,
    aupro_max_false_positive_rate: float = 0.3,
) -> PixelMetrics:
    """Calculate pixel metrics using a validation-selected threshold.

    Args:
        masks: Binary masks shaped ``[N, H, W]`` or ``[N, 1, H, W]``.
        anomaly_maps: Continuous anomaly maps with the same shape.
        threshold: Pixel threshold selected without final-test optimization.
        aupro_max_false_positive_rate: FPR range used for AUPRO.

    Returns:
        Pixel AUROC, AUPR, IoU, Dice, and AUPRO.

    Raises:
        MetricError: If inputs or threshold are invalid.
    """

    mask_array, score_array = _pixel_arrays(masks, anomaly_maps)
    if not np.isfinite(threshold):
        raise MetricError("Invalid pixel threshold: expected a finite value.")
    predictions = score_array >= threshold
    intersection = int(np.logical_and(predictions, mask_array).sum())
    union = int(np.logical_or(predictions, mask_array).sum())
    predicted_count = int(predictions.sum())
    target_count = int(mask_array.sum())
    iou = float(intersection / union) if union else 1.0
    denominator = predicted_count + target_count
    dice = float(2 * intersection / denominator) if denominator else 1.0
    flat_masks = mask_array.reshape(-1).astype(np.int64)
    flat_scores = score_array.reshape(-1)
    return PixelMetrics(
        auroc=float(roc_auc_score(flat_masks, flat_scores)),
        aupr=float(average_precision_score(flat_masks, flat_scores)),
        threshold=float(threshold),
        iou=iou,
        dice=dice,
        aupro=compute_aupro(
            mask_array,
            score_array,
            max_false_positive_rate=aupro_max_false_positive_rate,
        ),
        aupro_max_false_positive_rate=aupro_max_false_positive_rate,
    )
