"""Reference cases for image, pixel, AUPRO, and threshold metrics."""

import numpy as np
import pytest

from shiftad.evaluation.image_metrics import MetricError, compute_image_metrics
from shiftad.evaluation.pixel_metrics import compute_aupro, compute_pixel_metrics
from shiftad.evaluation.thresholds import calibrate_best_f1, calibrate_normal_quantile


def test_perfect_image_metrics() -> None:
    """Separated scores produce exact image classification metrics."""

    metrics = compute_image_metrics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        threshold=0.5,
    )
    assert metrics.auroc == pytest.approx(1.0)
    assert metrics.aupr == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.specificity == pytest.approx(1.0)
    assert metrics.false_positive_rate == pytest.approx(0.0)
    assert metrics.false_negative_rate == pytest.approx(0.0)


def test_image_metrics_require_both_classes() -> None:
    """Degenerate labels fail instead of returning a misleading AUROC."""

    with pytest.raises(MetricError, match="require both"):
        compute_image_metrics([0, 0], [0.1, 0.2], threshold=0.5)


def test_validation_threshold_protocols() -> None:
    """Quantile and best-F1 calibration preserve validation provenance."""

    quantile = calibrate_normal_quantile([0.1, 0.2, 0.3, 0.4], quantile=0.75)
    assert quantile.threshold == pytest.approx(0.4)
    assert quantile.validation_false_positive_rate == pytest.approx(0.25)
    assert quantile.method == "normal_quantile_higher"

    f1 = calibrate_best_f1([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert f1.threshold == pytest.approx(0.8)
    assert f1.validation_f1 == pytest.approx(1.0)
    assert f1.validation_false_positive_rate == pytest.approx(0.0)


def test_perfect_pixel_metrics_and_aupro() -> None:
    """A perfect anomaly map achieves one for all supported pixel metrics."""

    mask = np.zeros((1, 8, 8), dtype=np.uint8)
    mask[:, 2:6, 2:6] = 1
    anomaly_map = mask.astype(np.float64)
    metrics = compute_pixel_metrics(mask, anomaly_map, threshold=0.5)
    assert metrics.auroc == pytest.approx(1.0)
    assert metrics.aupr == pytest.approx(1.0)
    assert metrics.iou == pytest.approx(1.0)
    assert metrics.dice == pytest.approx(1.0)
    assert metrics.aupro == pytest.approx(1.0)
    assert compute_aupro(mask, anomaly_map, num_thresholds=2) == pytest.approx(1.0)


def test_pixel_metrics_accept_dataloader_singleton_channels() -> None:
    """Batched ``[N, 1, H, W]`` masks work without caller-side squeezing."""

    mask = np.zeros((2, 1, 4, 4), dtype=np.uint8)
    mask[:, :, 1:3, 1:3] = 1
    metrics = compute_pixel_metrics(mask, mask.astype(np.float64), threshold=0.5)
    assert metrics.auroc == pytest.approx(1.0)
    assert metrics.dice == pytest.approx(1.0)


def test_aupro_averages_individual_connected_regions() -> None:
    """Disconnected regions and FPR-boundary interpolation match analytically."""

    mask = np.zeros((1, 5, 5), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[0, 4, 4] = 1
    anomaly_map = np.zeros_like(mask, dtype=np.float64)
    anomaly_map[0, 0, 0] = 1.0
    score = compute_aupro(mask, anomaly_map, max_false_positive_rate=0.3)
    # PRO is 0.5 at FPR 0 and rises linearly toward 1.0 at FPR 1.
    assert score == pytest.approx(0.575)
