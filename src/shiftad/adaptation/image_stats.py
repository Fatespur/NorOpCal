"""Deterministic image-space statistics for the shift estimator.

Computes the frozen image-statistic set declared in the Phase 6 protocol
(``configs/phase6/protocol.yaml`` ``image_statistic_set``, frozen at Gate A):

- brightness (mean luminance)
- contrast (standard deviation of luminance)
- channel means (per-channel mean)
- color-temperature proxy
- Laplacian sharpness
- high-frequency noise proxy
- resolution indicator (edge-energy ratio)

All functions are pure numpy, deterministic, and never use global RNG or
future information. Inputs are float32 RGB images in [0, 1] with shape
``[H, W, 3]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Frozen image-statistic names (protocol ``image_statistic_set``, Gate A).
IMAGE_STATISTIC_NAMES: tuple[str, ...] = (
    "brightness",
    "contrast",
    "mean_r",
    "mean_g",
    "mean_b",
    "color_temperature",
    "laplacian_sharpness",
    "hf_noise",
    "resolution_energy",
)

EPS = 1e-8


def _as_float32_array(image: Any) -> np.ndarray:
    """Validate and convert an image to a float32 [H, W, 3] array.

    Raises:
        TypeError: If the input is not array-like.
        ValueError: If the shape or dtype range is unsupported.
    """
    try:
        arr = np.asarray(image, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Image must be array-like, got {type(image).__name__}.") from exc
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Image must be [H, W, 3], got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError("Image must not be empty.")
    if float(arr.min()) < -EPS or float(arr.max()) > 1.0 + EPS:
        raise ValueError(
            f"Image values must lie in [0, 1], got range [{float(arr.min()):.3f}, "
            f"{float(arr.max()):.3f}]."
        )
    return arr


def _clamp01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def brightness(image: Any) -> float:
    """Mean luminance (mean over all channels)."""
    arr = _as_float32_array(image)
    return float(arr.mean())


def contrast(image: Any) -> float:
    """Standard deviation of luminance."""
    arr = _as_float32_array(image)
    return float(arr.std())


def channel_means(image: Any) -> tuple[float, float, float]:
    """Per-channel means (R, G, B)."""
    arr = _as_float32_array(image)
    means = tuple(float(arr[..., c].mean()) for c in range(3))
    r, g, b = means
    return (r, g, b)


def color_temperature(image: Any) -> float:
    """Color-temperature proxy: (mean_r - mean_b) / (mean_g + eps)."""
    arr = _as_float32_array(image)
    mean_r = float(arr[..., 0].mean())
    mean_g = float(arr[..., 1].mean())
    mean_b = float(arr[..., 2].mean())
    return (mean_r - mean_b) / (mean_g + EPS)


def laplacian_sharpness(image: Any) -> float:
    """Laplacian sharpness: mean absolute Laplacian over luminance."""
    arr = _as_float32_array(image)
    gray = arr.mean(axis=2)
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    padded = np.pad(gray, 1, mode="edge")
    # Manual 3x3 convolution (deterministic, no scipy dependency).
    out = np.zeros_like(gray)
    for di in range(3):
        for dj in range(3):
            out += kernel[di, dj] * padded[di : di + gray.shape[0], dj : dj + gray.shape[1]]
    return float(np.abs(out).mean())


def hf_noise(image: Any) -> float:
    """High-frequency noise proxy: mean absolute difference to a 2x2 box blur."""
    arr = _as_float32_array(image)
    gray = arr.mean(axis=2)
    blurred = (gray[:-1:2, :-1:2] + gray[:-1:2, 1::2] + gray[1::2, :-1:2] + gray[1::2, 1::2]) / 4.0
    # Align blurred with the original via upsampling by repetition.
    up = np.repeat(np.repeat(blurred, 2, axis=0), 2, axis=1)
    h, w = gray.shape
    up = up[:h, :w]
    return float(np.abs(gray - up).mean())


def resolution_energy(image: Any) -> float:
    """Resolution indicator: high-frequency edge energy over total energy.

    Uses the Laplacian map: ratio of |laplacian| above the median to the
    total |laplacian| sum. Higher for sharper/higher-resolution content.
    """
    arr = _as_float32_array(image)
    gray = arr.mean(axis=2)
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    padded = np.pad(gray, 1, mode="edge")
    lap = np.zeros_like(gray)
    for di in range(3):
        for dj in range(3):
            lap += kernel[di, dj] * padded[di : di + gray.shape[0], dj : dj + gray.shape[1]]
    abs_lap = np.abs(lap)
    total = float(abs_lap.sum())
    if total <= EPS:
        return 0.0
    threshold = float(np.median(abs_lap))
    return float(abs_lap[abs_lap > threshold].sum()) / total


def compute_image_stats(image: Any) -> dict[str, float]:
    """Compute the full frozen image-statistic set for one image.

    Args:
        image: Float32 RGB array in [0, 1] with shape [H, W, 3].

    Returns:
        A dict keyed by ``IMAGE_STATISTIC_NAMES`` with deterministic values.
    """
    arr = _as_float32_array(image)
    means = channel_means(arr)
    stats: dict[str, float] = {
        "brightness": brightness(arr),
        "contrast": contrast(arr),
        "mean_r": means[0],
        "mean_g": means[1],
        "mean_b": means[2],
        "color_temperature": color_temperature(arr),
        "laplacian_sharpness": laplacian_sharpness(arr),
        "hf_noise": hf_noise(arr),
        "resolution_energy": resolution_energy(arr),
    }
    return {name: float(stats[name]) for name in IMAGE_STATISTIC_NAMES}
