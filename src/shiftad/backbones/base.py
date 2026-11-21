"""Common contracts for frozen patch-feature backbones."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor


class BackboneError(RuntimeError):
    """Report invalid inputs, feature outputs, or unavailable weights."""


@dataclass(frozen=True)
class FeatureBatch:
    """Store named patch-token tensors and their spatial grids.

    Attributes:
        layer_features: Mapping from stable layer names to ``[B, N, C]`` tensors.
        patch_grid_shapes: Spatial ``(height, width)`` for each named layer.
        input_size: Input image height and width.
        metadata: Backbone and weight provenance that is safe to serialize.
    """

    layer_features: dict[str, Tensor]
    patch_grid_shapes: dict[str, tuple[int, int]]
    input_size: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate layer names, tensor ranks, batch agreement, and patch counts.

        Raises:
            BackboneError: If the feature collection is empty or inconsistent.
        """

        if not self.layer_features:
            raise BackboneError("Backbone returned no layer features.")
        if set(self.layer_features) != set(self.patch_grid_shapes):
            raise BackboneError("Feature layer names and patch-grid names must match exactly.")
        batch_size: int | None = None
        for name, features in self.layer_features.items():
            if features.ndim != 3:
                raise BackboneError(
                    f"Layer {name!r} has shape {tuple(features.shape)}; expected [B, N, C]."
                )
            grid = self.patch_grid_shapes[name]
            if features.shape[1] != grid[0] * grid[1]:
                raise BackboneError(
                    f"Layer {name!r} has {features.shape[1]} tokens but grid {grid} "
                    f"requires {grid[0] * grid[1]}."
                )
            if batch_size is None:
                batch_size = features.shape[0]
            elif features.shape[0] != batch_size:
                raise BackboneError("All feature layers must preserve the same batch size.")
            if not torch.isfinite(features).all():
                raise BackboneError(f"Layer {name!r} contains NaN or Infinity.")


@runtime_checkable
class FeatureBackbone(Protocol):
    """Protocol implemented by frozen visual feature providers."""

    @property
    def device(self) -> torch.device:
        """Return the device on which feature extraction executes."""

    @property
    def is_real_dinov2(self) -> bool:
        """Return whether official DINOv2 architecture and weights are active."""

    def extract_features(self, images: Tensor) -> FeatureBatch:
        """Extract intermediate patch tokens from ``[B, C, H, W]`` images."""


def validate_image_batch(images: Tensor) -> tuple[int, int, int, int]:
    """Validate an RGB floating-point image batch.

    Args:
        images: Candidate tensor.

    Returns:
        Batch, channel, height, and width sizes.

    Raises:
        BackboneError: If rank, channels, dimensions, dtype, or values are invalid.
    """

    if images.ndim != 4:
        raise BackboneError(
            f"Backbone input has shape {tuple(images.shape)}; expected [B, C, H, W]."
        )
    batch, channels, height, width = images.shape
    if batch <= 0 or channels != 3 or height <= 0 or width <= 0:
        raise BackboneError(
            f"Backbone input has shape {tuple(images.shape)}; expected non-empty RGB images."
        )
    if not images.is_floating_point():
        raise BackboneError(
            f"Backbone input dtype is {images.dtype}; expected a floating-point tensor."
        )
    if not torch.isfinite(images).all():
        raise BackboneError("Backbone input contains NaN or Infinity.")
    return batch, channels, height, width
