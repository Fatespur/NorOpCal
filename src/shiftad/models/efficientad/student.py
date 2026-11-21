"""Student network for EfficientAD anomaly detection.

The student learns to predict the teacher's feature maps on normal images.
At test time, discrepancies between student and teacher outputs indicate
local anomalies. The student outputs ``teacher_channels + ae_channels``
channels: the first ``teacher_channels`` match the teacher, and the
remaining ``ae_channels`` serve as input to the autoencoder.

Architecture: A compact PDN-style CNN with three downsampling blocks,
mirroring the teacher's spatial reduction (stride 8 total).
"""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn


class StudentNetwork(nn.Module):
    """Trainable student network that learns to predict teacher features.

    Args:
        teacher_channels: Number of teacher output channels to predict.
        ae_channels: Additional channels for the autoencoder branch.
        input_channels: Input image channels (default 3 for RGB).

    Notes:
        The student outputs ``teacher_channels + ae_channels`` channels at
        the same spatial resolution as the teacher. The first
        ``teacher_channels`` are compared with the teacher for local anomaly
        detection, and the remaining ``ae_channels`` are reconstructed by
        the autoencoder for global anomaly detection.
    """

    def __init__(
        self,
        *,
        teacher_channels: int = 128,
        ae_channels: int = 128,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        if teacher_channels <= 0 or ae_channels <= 0:
            raise ValueError("teacher_channels and ae_channels must be positive.")
        if input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        self._teacher_channels = teacher_channels
        self._ae_channels = ae_channels
        self._out_channels = teacher_channels + ae_channels

        # PDN-style architecture: three blocks, each downsampling by 2.
        # Uses same-padding convolutions (k=3, p=1) so spatial output is
        # exactly H/8, matching the ResNet18 teacher's layer2 output.
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, self._out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self._out_channels, self._out_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
        )

    @property
    def out_channels(self) -> int:
        """Return the total number of output channels."""

        return self._out_channels

    @property
    def teacher_channels(self) -> int:
        """Return the number of channels that match the teacher."""

        return self._teacher_channels

    @property
    def ae_channels(self) -> int:
        """Return the number of channels for the autoencoder branch."""

        return self._ae_channels

    def forward(self, images: Tensor) -> Tensor:
        """Predict teacher features and produce AE input channels.

        Args:
            images: RGB tensor shaped ``[B, 3, H, W]``.

        Returns:
            Feature map shaped ``[B, out_channels, H/8, W/8]`` where the
            first ``teacher_channels`` correspond to the teacher prediction
            and the remaining ``ae_channels`` feed the autoencoder.

        Raises:
            RuntimeError: If input dimensions are invalid.
        """

        if images.ndim != 4 or images.shape[1] != 3:
            raise RuntimeError(
                f"Student expects [B, 3, H, W] input, got shape {tuple(images.shape)}."
            )
        return cast(Tensor, self.block3(self.block2(self.block1(images))))

    def split_output(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Split student output into teacher-matching and AE channels.

        Args:
            features: Student output shaped ``[B, out_channels, H, W]``.

        Returns:
            A tuple of (teacher_match, ae_input) where teacher_match has
            ``teacher_channels`` and ae_input has ``ae_channels``.
        """

        if features.shape[1] != self._out_channels:
            raise RuntimeError(
                f"Student output has {features.shape[1]} channels, expected {self._out_channels}."
            )
        return features[:, : self._teacher_channels], features[:, self._teacher_channels :]


class FixtureStudentNetwork(nn.Module):
    """Compact student for offline fixture tests only.

    Args:
        teacher_channels: Number of teacher output channels to predict.
        ae_channels: Additional channels for the autoencoder branch.
    """

    def __init__(
        self,
        *,
        teacher_channels: int = 16,
        ae_channels: int = 16,
    ) -> None:
        super().__init__()
        self._teacher_channels = teacher_channels
        self._ae_channels = ae_channels
        self._out_channels = teacher_channels + ae_channels
        self._feature_extractor = nn.Sequential(
            nn.Conv2d(3, self._out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self._out_channels, self._out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self._out_channels, self._out_channels, kernel_size=3, stride=2, padding=1),
        )

    @property
    def out_channels(self) -> int:
        """Return the total number of output channels."""

        return self._out_channels

    @property
    def teacher_channels(self) -> int:
        """Return the number of channels that match the teacher."""

        return self._teacher_channels

    @property
    def ae_channels(self) -> int:
        """Return the number of channels for the autoencoder branch."""

        return self._ae_channels

    def forward(self, images: Tensor) -> Tensor:
        """Predict features for fixture tests.

        Args:
            images: RGB tensor shaped ``[B, 3, H, W]``.

        Returns:
            Feature map shaped ``[B, out_channels, H/8, W/8]``.
        """

        if images.ndim != 4 or images.shape[1] != 3:
            raise RuntimeError(
                f"Fixture student expects [B, 3, H, W] input, got {tuple(images.shape)}."
            )
        return cast(Tensor, self._feature_extractor(images))

    def split_output(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Split student output into teacher-matching and AE channels.

        Args:
            features: Student output shaped ``[B, out_channels, H, W]``.

        Returns:
            A tuple of (teacher_match, ae_input).
        """

        if features.shape[1] != self._out_channels:
            raise RuntimeError(
                f"Fixture student output has {features.shape[1]} channels, "
                f"expected {self._out_channels}."
            )
        return features[:, : self._teacher_channels], features[:, self._teacher_channels :]
