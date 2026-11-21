"""Deterministic offline backbone used only for synthetic acceptance tests."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from shiftad.backbones.base import BackboneError, FeatureBatch, validate_image_batch


class DeterministicFixtureBackbone(nn.Module):
    """Generate deterministic patch descriptors without external weights.

    This provider is intentionally unsuitable for real anomaly experiments. It
    exists to test tensor flow, memory construction, serialization, and CLI
    behavior without network access.

    Args:
        layers: Stable non-negative fixture layer identifiers.
        patch_size: Non-overlapping pooling patch size.
        device: Compute device.
    """

    def __init__(
        self,
        layers: tuple[int, ...] = (0, 1),
        patch_size: int = 8,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if not layers or len(set(layers)) != len(layers) or min(layers) < 0:
            raise BackboneError("Fixture backbone layers must be unique non-negative integers.")
        if patch_size <= 0:
            raise BackboneError("Fixture backbone patch_size must be positive.")
        self.layers = layers
        self.patch_size = patch_size
        self.register_parameter(
            "_fixed_scale",
            nn.Parameter(torch.tensor(1.0), requires_grad=False),
        )
        self.to(device)
        self.eval()

    @property
    def device(self) -> torch.device:
        """Return the fixture parameter device."""

        return next(self.parameters()).device

    @property
    def is_real_dinov2(self) -> bool:
        """Return ``False`` because this is explicitly synthetic."""

        return False

    @torch.inference_mode()
    def extract_features(self, images: Tensor) -> FeatureBatch:
        """Pool RGB patches and derive stable multi-layer descriptors.

        Args:
            images: Floating-point RGB tensor shaped ``[B, 3, H, W]``.

        Returns:
            Named fixture features shaped ``[B, N, 4]``.

        Raises:
            BackboneError: If input dimensions are invalid or not patch-aligned.
        """

        _, _, height, width = validate_image_batch(images)
        if height % self.patch_size or width % self.patch_size:
            raise BackboneError(
                f"Fixture input {(height, width)} is not divisible by patch size {self.patch_size}."
            )
        images = images.to(device=self.device, dtype=torch.float32)
        pooled = functional.avg_pool2d(
            images,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        base = torch.cat((pooled, pooled.mean(dim=1, keepdim=True)), dim=1)
        tokens = base.flatten(2).transpose(1, 2)
        layer_features = {
            f"fixture.{layer}": tokens * float(layer + 1) + float(layer) / 10.0
            for layer in self.layers
        }
        grid = (height // self.patch_size, width // self.patch_size)
        result = FeatureBatch(
            layer_features=layer_features,
            patch_grid_shapes={name: grid for name in layer_features},
            input_size=(height, width),
            metadata={
                "backbone": "deterministic_fixture",
                "artifact_scope": "synthetic_fixture_acceptance_only",
                "patch_size": self.patch_size,
            },
        )
        result.validate()
        return result
