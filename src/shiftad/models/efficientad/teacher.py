"""Frozen teacher network for EfficientAD anomaly detection.

The teacher provides local feature maps that the student learns to predict.
In the original EfficientAD paper, the teacher is a Plain Discriminative
Network (PDN) pretrained on a generated dataset. To avoid silent external
weight downloads and ensure reproducibility, this implementation uses a
torchvision pretrained ResNet18 as the teacher feature extractor.

The teacher extracts features from layer2 of ResNet18, producing a feature
map of shape ``[B, 128, H/8, W/8]`` for input size ``[B, 3, H, W]``. For the
standard 256x256 input, this yields ``[B, 128, 32, 32]``.

The teacher is always frozen: ``requires_grad=False`` and ``eval()`` mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn


class TeacherNetwork(nn.Module):
    """Frozen pretrained feature extractor for teacher-student distillation.

    Args:
        out_channels: Number of output channels from the teacher. This is
            determined by the ResNet18 layer2, which is 128.
        pretrained: If True, load ImageNet pretrained weights. If False,
            use random initialization (only for fixtures).
        checkpoint: Optional local checkpoint path for custom teacher weights.
        allow_download: If True, permit torchvision to download pretrained
            weights when the cache is empty.
        use_cache: If True, permit reuse of cached torchvision weights.

    Raises:
        RuntimeError: If pretrained weights cannot be loaded and
            ``allow_download`` is False.

    Notes:
        The teacher is always frozen after construction. Callers must not
        modify its parameters.
    """

    #: Expected output channels from ResNet18 layer2.
    OUT_CHANNELS: int = 128

    def __init__(
        self,
        *,
        pretrained: bool = True,
        checkpoint: Path | None = None,
        allow_download: bool = False,
        use_cache: bool = True,
    ) -> None:
        super().__init__()
        self._out_channels = self.OUT_CHANNELS
        self._feature_extractor = self._build_feature_extractor(
            pretrained=pretrained,
            checkpoint=checkpoint,
            allow_download=allow_download,
            use_cache=use_cache,
        )
        self._freeze()

    def _build_feature_extractor(
        self,
        *,
        pretrained: bool,
        checkpoint: Path | None,
        allow_download: bool,
        use_cache: bool,
    ) -> nn.Module:
        """Build the ResNet18-based feature extractor.

        Args:
            pretrained: Whether to load pretrained weights.
            checkpoint: Optional local checkpoint path.
            allow_download: Permit network access for weights.
            use_cache: Permit cached weights reuse.

        Returns:
            A frozen feature extraction module.

        Raises:
            RuntimeError: If weights cannot be loaded.
        """

        import torchvision.models as tv_models

        if checkpoint is not None:
            if not checkpoint.is_file():
                raise RuntimeError(
                    f"Teacher checkpoint not found: {checkpoint}. "
                    "Provide a valid checkpoint path or set pretrained: true "
                    "with allow_download: true."
                )
            model = tv_models.resnet18(weights=None)
            payload: Any = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if isinstance(payload, dict) and "state_dict" in payload:
                payload = payload["state_dict"]
            model.load_state_dict(payload, strict=False)
        elif pretrained:
            if allow_download or use_cache:
                try:
                    model = tv_models.resnet18(weights="IMAGENET1K_V1")
                except (OSError, RuntimeError, ValueError) as exc:
                    if not allow_download:
                        raise RuntimeError(
                            f"Cannot load cached pretrained ResNet18 weights: {exc}. "
                            "Set allow_download: true to download, or provide a checkpoint."
                        ) from exc
                    raise RuntimeError(
                        f"Failed to download pretrained ResNet18 weights: {exc}"
                    ) from exc
            else:
                raise RuntimeError(
                    "Teacher requires pretrained weights but allow_download=false "
                    "and use_cache=false. Set allow_download: true or provide a checkpoint."
                )
        else:
            model = tv_models.resnet18(weights=None)

        # Extract conv1, bn1, relu, maxpool, layer1, layer2 as the feature extractor.
        return nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
        )

    def _freeze(self) -> None:
        """Freeze all parameters and set eval mode."""

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @property
    def out_channels(self) -> int:
        """Return the number of output feature channels."""

        return self._out_channels

    @torch.inference_mode()
    def forward(self, images: Tensor) -> Tensor:
        """Extract frozen teacher features.

        Args:
            images: Normalized RGB tensor shaped ``[B, 3, H, W]``.

        Returns:
            Feature map shaped ``[B, out_channels, H/8, W/8]``.

        Raises:
            RuntimeError: If the forward pass fails.
        """

        if images.ndim != 4 or images.shape[1] != 3:
            raise RuntimeError(
                f"Teacher expects [B, 3, H, W] input, got shape {tuple(images.shape)}."
            )
        return cast(Tensor, self._feature_extractor(images))

    def assert_frozen(self) -> None:
        """Assert that no parameter requires gradients.

        Raises:
            AssertionError: If any parameter requires gradients.
        """

        for name, param in self.named_parameters():
            if param.requires_grad:
                raise AssertionError(f"Teacher parameter {name!r} unexpectedly requires gradients.")


class FixtureTeacherNetwork(nn.Module):
    """Deterministic small teacher for offline fixture tests only.

    This network produces deterministic features without any pretrained
    weights, enabling CPU-only fixture tests without network access.

    Args:
        out_channels: Number of output channels (default 16 for small fixtures).
    """

    def __init__(self, out_channels: int = 16) -> None:
        super().__init__()
        self._out_channels = out_channels
        self._feature_extractor = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )
        self._freeze()

    def _freeze(self) -> None:
        """Freeze all parameters and set eval mode."""

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @property
    def out_channels(self) -> int:
        """Return the number of output feature channels."""

        return self._out_channels

    @torch.inference_mode()
    def forward(self, images: Tensor) -> Tensor:
        """Extract deterministic fixture features.

        Args:
            images: RGB tensor shaped ``[B, 3, H, W]``.

        Returns:
            Feature map shaped ``[B, out_channels, H/8, W/8]``.
        """

        if images.ndim != 4 or images.shape[1] != 3:
            raise RuntimeError(
                f"Fixture teacher expects [B, 3, H, W] input, got {tuple(images.shape)}."
            )
        return cast(Tensor, self._feature_extractor(images))

    def assert_frozen(self) -> None:
        """Assert that no parameter requires gradients."""

        for name, param in self.named_parameters():
            if param.requires_grad:
                raise AssertionError(
                    f"Fixture teacher parameter {name!r} unexpectedly requires gradients."
                )
