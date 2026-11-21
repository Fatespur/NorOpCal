"""Patch-score aggregation and spatial anomaly-map reconstruction."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor

from shiftad.methods.base import DetectorError


def _gaussian_kernel(sigma: float, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    radius = max(1, math.ceil(3.0 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, *kernel_2d.shape)


def patch_scores_to_anomaly_maps(
    patch_scores: Tensor,
    *,
    grid_shape: tuple[int, int],
    output_size: tuple[int, int],
    smoothing_sigma: float = 0.0,
) -> Tensor:
    """Reshape patch scores and interpolate them to image resolution.

    Args:
        patch_scores: Finite scores shaped ``[B, N]``.
        grid_shape: Patch height and width.
        output_size: Output image height and width.
        smoothing_sigma: Optional Gaussian standard deviation in output pixels.

    Returns:
        Finite anomaly maps shaped ``[B, H, W]``.
    """

    if (
        patch_scores.ndim != 2
        or patch_scores.shape[1] != grid_shape[0] * grid_shape[1]
        or min(output_size) <= 0
    ):
        raise DetectorError(
            f"Patch scores shape {tuple(patch_scores.shape)}, grid {grid_shape}, "
            f"and output {output_size} are incompatible."
        )
    if smoothing_sigma < 0 or not torch.isfinite(patch_scores).all():
        raise DetectorError("Anomaly-map scores must be finite and smoothing non-negative.")
    maps = patch_scores.reshape(-1, 1, *grid_shape)
    maps = functional.interpolate(
        maps,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    if smoothing_sigma > 0:
        kernel = _gaussian_kernel(
            smoothing_sigma,
            device=maps.device,
            dtype=maps.dtype,
        )
        padding = kernel.shape[-1] // 2
        maps = functional.conv2d(maps, kernel, padding=padding)
    result = maps[:, 0]
    if not torch.isfinite(result).all():
        raise DetectorError("Anomaly-map reconstruction produced NaN or Infinity.")
    return result


def aggregate_image_scores(
    patch_scores: Tensor,
    *,
    mode: str,
    top_k: int,
) -> Tensor:
    """Aggregate patch scores to one image score per batch item."""

    if patch_scores.ndim != 2 or patch_scores.shape[1] == 0:
        raise DetectorError("Image aggregation expects non-empty [B, N] patch scores.")
    if mode == "max":
        return patch_scores.max(dim=1).values
    if mode == "top_k_mean":
        if not 1 <= top_k <= patch_scores.shape[1]:
            raise DetectorError(f"top_k={top_k} is invalid for {patch_scores.shape[1]} patches.")
        return patch_scores.topk(top_k, dim=1).values.mean(dim=1)
    raise DetectorError(f"Unsupported image-score aggregation {mode!r}.")
