"""Spatial alignment and fusion for multi-layer patch tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor

from shiftad.backbones.base import BackboneError


@dataclass(frozen=True)
class FusedFeatures:
    """Store fused patch tokens, common grid, and transformation metadata."""

    features: Tensor
    grid_shape: tuple[int, int]
    metadata: dict[str, object]


class FeatureFuser:
    """Align patch grids and combine layers by concatenation or mean.

    Args:
        mode: ``concat`` or ``mean``.
        normalize: Apply numerically safe L2 normalization after fusion.
        target_grid: Optional explicit output patch grid. The finest input grid
            is used when omitted.
    """

    def __init__(
        self,
        *,
        mode: str = "concat",
        normalize: bool = True,
        target_grid: tuple[int, int] | None = None,
    ) -> None:
        if mode not in {"concat", "mean"}:
            raise BackboneError(f"Unsupported fusion mode {mode!r}; use concat or mean.")
        if target_grid is not None and min(target_grid) <= 0:
            raise BackboneError("Fusion target_grid values must be positive.")
        self.mode = mode
        self.normalize = normalize
        self.target_grid = target_grid

    def __call__(
        self,
        *,
        layer_features: dict[str, Tensor],
        patch_grid_shapes: dict[str, tuple[int, int]],
    ) -> FusedFeatures:
        """Align named patch tokens and produce one finite feature tensor.

        Raises:
            BackboneError: If names, shapes, dimensions, or values are invalid.
        """

        if not layer_features or set(layer_features) != set(patch_grid_shapes):
            raise BackboneError(
                "Fusion requires non-empty features with exactly matching grid names."
            )
        ordered_names = tuple(layer_features)
        target = self.target_grid or max(
            patch_grid_shapes.values(),
            key=lambda grid: grid[0] * grid[1],
        )
        aligned: list[Tensor] = []
        batch_size: int | None = None
        for name in ordered_names:
            features = layer_features[name]
            grid = patch_grid_shapes[name]
            if features.ndim != 3 or features.shape[1] != grid[0] * grid[1]:
                raise BackboneError(
                    f"Cannot fuse layer {name!r} with shape {tuple(features.shape)} "
                    f"and grid {grid}."
                )
            if batch_size is None:
                batch_size = features.shape[0]
            elif features.shape[0] != batch_size:
                raise BackboneError("Fusion layers have inconsistent batch sizes.")
            spatial = features.transpose(1, 2).reshape(
                features.shape[0],
                features.shape[2],
                grid[0],
                grid[1],
            )
            if grid != target:
                spatial = functional.interpolate(
                    spatial,
                    size=target,
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(spatial.flatten(2).transpose(1, 2))
        if self.mode == "concat":
            fused = torch.cat(aligned, dim=-1)
        else:
            dimensions = {item.shape[-1] for item in aligned}
            if len(dimensions) != 1:
                raise BackboneError(
                    "Mean fusion requires equal channel dimensions after alignment."
                )
            fused = torch.stack(aligned, dim=0).mean(dim=0)
        if self.normalize:
            fused = functional.normalize(fused, p=2, dim=-1, eps=1e-12)
        if not torch.isfinite(fused).all():
            raise BackboneError("Fused features contain NaN or Infinity.")
        return FusedFeatures(
            features=fused,
            grid_shape=target,
            metadata={
                "fusion_mode": self.mode,
                "normalized": self.normalize,
                "layer_names": list(ordered_names),
                "feature_dimension": fused.shape[-1],
                "patch_grid": list(target),
            },
        )
