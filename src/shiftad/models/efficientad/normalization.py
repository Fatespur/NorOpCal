"""Feature normalization for EfficientAD anomaly map computation.

During inference, the local and global anomaly maps are computed as pixel-wise
differences. To ensure stable and comparable magnitudes, the student and
teacher outputs can be channel-wise normalized using statistics (mean and
std) collected from the training set. This module stores and applies those
statistics.
"""

from __future__ import annotations

from torch import Tensor


class FeatureNormalizer:
    """Channel-wise mean/std normalization for feature maps.

    Stores per-channel mean and standard deviation computed from training
    data and applies normalization at inference time. When statistics are
    not provided, normalization is a no-op.

    Args:
        mean: Per-channel mean tensor shaped ``[C]``, or None.
        std: Per-channel standard deviation tensor shaped ``[C]``, or None.

    Notes:
        The normalizer is typically fit on teacher output statistics from
        the training set. Division by zero is prevented by clamping std to
        a minimum epsilon.
    """

    EPSILON: float = 1e-6

    def __init__(
        self,
        *,
        mean: Tensor | None = None,
        std: Tensor | None = None,
    ) -> None:
        if (mean is None) != (std is None):
            raise ValueError("Both mean and std must be provided or both must be None.")
        if mean is not None and std is not None:
            if mean.shape != std.shape:
                raise ValueError(f"Mean shape {tuple(mean.shape)} != std shape {tuple(std.shape)}.")
            if mean.ndim != 1:
                raise ValueError(f"Mean/std must be 1-D channel vectors, got ndim={mean.ndim}.")
        self._mean = mean
        self._std = std

    @property
    def is_fit(self) -> bool:
        """Return True if normalization statistics are available."""

        return self._mean is not None and self._std is not None

    @classmethod
    def fit_from_features(cls, features: Tensor) -> FeatureNormalizer:
        """Compute channel-wise mean and std from a batch of features.

        Args:
            features: Feature tensor shaped ``[N, C, H, W]`` collected from
                the training set.

        Returns:
            A fitted FeatureNormalizer.

        Raises:
            ValueError: If the input tensor has fewer than 4 dimensions.
        """

        if features.ndim != 4:
            raise ValueError(
                f"Fit expects [N, C, H, W] features, got shape {tuple(features.shape)}."
            )
        # Compute per-channel statistics across N, H, W.
        n, c, h, w = features.shape
        flat = features.reshape(n, c, h * w)
        mean = flat.mean(dim=(0, 2))
        std = flat.std(dim=(0, 2), unbiased=True)
        std = std.clamp(min=cls.EPSILON)
        return cls(mean=mean, std=std)

    def normalize(self, features: Tensor) -> Tensor:
        """Apply channel-wise normalization.

        Args:
            features: Feature tensor shaped ``[B, C, H, W]``.

        Returns:
            Normalized features with the same shape.

        Notes:
            If no statistics are available, returns the input unchanged.
        """

        if not self.is_fit:
            return features
        assert self._mean is not None and self._std is not None
        mean = self._mean.to(device=features.device, dtype=features.dtype)
        std = self._std.to(device=features.device, dtype=features.dtype)
        # Broadcast [C] -> [1, C, 1, 1]
        return (features - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

    def state_dict(self) -> dict[str, Tensor | None]:
        """Return a serializable state dictionary.

        Returns:
            A dictionary with optional ``mean`` and ``std`` tensors.
        """

        return {"mean": self._mean, "std": self._std}

    @classmethod
    def from_state_dict(cls, state: dict[str, Tensor | None]) -> FeatureNormalizer:
        """Reconstruct a normalizer from a saved state dictionary.

        Args:
            state: Dictionary with optional ``mean`` and ``std`` tensors.

        Returns:
            A reconstructed FeatureNormalizer.
        """

        mean = state.get("mean")
        std = state.get("std")
        if mean is not None and not isinstance(mean, Tensor):
            raise TypeError("Normalizer mean must be a tensor or None.")
        if std is not None and not isinstance(std, Tensor):
            raise TypeError("Normalizer std must be a tensor or None.")
        return cls(mean=mean, std=std)
