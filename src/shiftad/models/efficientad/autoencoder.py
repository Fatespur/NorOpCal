"""Autoencoder for EfficientAD global anomaly detection.

The autoencoder learns to reconstruct the student's AE channels on normal
images. At test time, high reconstruction error indicates global anomalies
that the student did not learn to represent correctly.

Architecture: A symmetric encoder-decoder with a bottleneck, operating at
the student's feature-map resolution.
"""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn


class AutoEncoder(nn.Module):
    """Trainable autoencoder for global anomaly detection.

    Args:
        in_channels: Number of input channels from the student's AE branch.
        hidden_channels: Hidden layer width in the encoder and decoder.
        bottleneck_channels: Bottleneck layer width.

    Notes:
        The autoencoder input and output have the same channel count and
        spatial resolution as the student's AE branch.
    """

    def __init__(
        self,
        *,
        in_channels: int = 128,
        hidden_channels: int = 128,
        bottleneck_channels: int = 64,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or hidden_channels <= 0 or bottleneck_channels <= 0:
            raise ValueError("All channel counts must be positive.")
        self._in_channels = in_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, bottleneck_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, stride=1, padding=1),
        )

    @property
    def in_channels(self) -> int:
        """Return the number of input/output channels."""

        return self._in_channels

    def forward(self, features: Tensor) -> Tensor:
        """Reconstruct the input feature map.

        Args:
            features: Student AE branch output shaped ``[B, in_channels, H, W]``.

        Returns:
            Reconstructed features with the same shape as input.

        Raises:
            RuntimeError: If input channels do not match configuration.
        """

        if features.ndim != 4:
            raise RuntimeError(
                f"AutoEncoder expects [B, C, H, W] input, got shape {tuple(features.shape)}."
            )
        if features.shape[1] != self._in_channels:
            raise RuntimeError(
                f"AutoEncoder input has {features.shape[1]} channels, expected {self._in_channels}."
            )
        return cast(Tensor, self.decoder(self.encoder(features)))


class FixtureAutoEncoder(nn.Module):
    """Compact autoencoder for offline fixture tests only.

    Args:
        in_channels: Number of input channels from the fixture student's AE branch.
    """

    def __init__(self, *, in_channels: int = 16) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        self._in_channels = in_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        )

    @property
    def in_channels(self) -> int:
        """Return the number of input/output channels."""

        return self._in_channels

    def forward(self, features: Tensor) -> Tensor:
        """Reconstruct fixture feature maps.

        Args:
            features: Student AE branch output shaped ``[B, in_channels, H, W]``.

        Returns:
            Reconstructed features with the same shape as input.
        """

        if features.ndim != 4:
            raise RuntimeError(
                f"Fixture AutoEncoder expects [B, C, H, W] input, "
                f"got shape {tuple(features.shape)}."
            )
        if features.shape[1] != self._in_channels:
            raise RuntimeError(
                f"Fixture AutoEncoder input has {features.shape[1]} channels, "
                f"expected {self._in_channels}."
            )
        return cast(Tensor, self.decoder(self.encoder(features)))
