"""Deterministic anomaly heatmaps and overlays."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from shiftad.evaluation.image_metrics import MetricError

ImageInput = ArrayLike | torch.Tensor


def _rgb_array(image: ImageInput) -> NDArray[np.uint8]:
    array = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if array.ndim != 3:
        raise MetricError(
            f"Invalid visualization image shape {array.shape}; expected HWC or CHW RGB."
        )
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise MetricError(f"Invalid visualization channels {array.shape[-1]}; expected exactly 3.")
    float_array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(float_array)):
        raise MetricError("Visualization image contains non-finite values.")
    if float_array.max(initial=0.0) <= 1.0:
        float_array *= 255.0
    return np.clip(float_array, 0.0, 255.0).astype(np.uint8)


def _normalized_map(anomaly_map: ImageInput) -> NDArray[np.float64]:
    if isinstance(anomaly_map, torch.Tensor):
        array = anomaly_map.detach().cpu().numpy()
    else:
        array = np.asarray(anomaly_map)
    array = np.squeeze(array)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise MetricError(
            f"Invalid anomaly map shape or values {array.shape}; expected finite [H, W]."
        )
    float_array = np.asarray(array, dtype=np.float64)
    minimum = float(float_array.min())
    maximum = float(float_array.max())
    if maximum == minimum:
        return np.zeros_like(float_array)
    return (float_array - minimum) / (maximum - minimum)


def create_heatmap_overlay(
    image: ImageInput,
    anomaly_map: ImageInput,
    *,
    alpha: float = 0.45,
    colormap: str = "inferno",
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Create an RGB heatmap and alpha-blended image overlay.

    Args:
        image: RGB image in HWC or CHW layout.
        anomaly_map: Finite 2D anomaly scores with matching spatial dimensions.
        alpha: Heatmap contribution in ``[0, 1]``.
        colormap: Matplotlib colormap name.

    Returns:
        Heatmap and overlay as ``uint8`` HWC RGB arrays.

    Raises:
        MetricError: If shapes, values, alpha, or colormap are invalid.
    """

    if not 0.0 <= alpha <= 1.0:
        raise MetricError("Heatmap alpha must be between 0 and 1.")
    rgb = _rgb_array(image)
    normalized = _normalized_map(anomaly_map)
    if rgb.shape[:2] != normalized.shape:
        raise MetricError(
            f"Image shape {rgb.shape[:2]} and anomaly map shape {normalized.shape} must match."
        )
    try:
        cmap = matplotlib.colormaps[colormap]
    except KeyError as exc:
        raise MetricError(f"Unknown Matplotlib colormap {colormap!r}.") from exc
    heatmap = np.asarray(cmap(normalized, bytes=True)[..., :3], dtype=np.uint8)
    overlay = np.clip(
        (1.0 - alpha) * rgb.astype(np.float64) + alpha * heatmap.astype(np.float64),
        0.0,
        255.0,
    ).astype(np.uint8)
    return heatmap, overlay


def save_anomaly_visualization(
    image: ImageInput,
    anomaly_map: ImageInput,
    output_path: Path,
    *,
    alpha: float = 0.45,
    colormap: str = "inferno",
    dpi: int = 150,
) -> Path:
    """Render original, heatmap, and overlay panels to a PNG.

    Args:
        image: RGB image in HWC or CHW layout.
        anomaly_map: Finite aligned anomaly scores.
        output_path: Destination image path.
        alpha: Heatmap contribution in the overlay.
        colormap: Matplotlib colormap name.
        dpi: Positive output resolution.

    Returns:
        The written output path.

    Raises:
        MetricError: If inputs or ``dpi`` are invalid.
        OSError: If the destination cannot be written.
    """

    if dpi <= 0:
        raise MetricError("Visualization dpi must be positive.")
    rgb = _rgb_array(image)
    heatmap, overlay = create_heatmap_overlay(
        rgb,
        anomaly_map,
        alpha=alpha,
        colormap=colormap,
    )
    figure = Figure(figsize=(9, 3), constrained_layout=True)
    FigureCanvasAgg(figure)
    for axis, panel, title in zip(
        figure.subplots(1, 3),
        (rgb, heatmap, overlay),
        ("Input", "Anomaly map", "Overlay"),
        strict=True,
    ):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    figure.clear()
    return output_path
