"""V1 shift estimator: deterministic, clean-reference calibrated.

Implements the Phase 6 protocol Section 5.1 estimator:

- source image/feature normalization fitted from source-normal data (D1);
- robust window statistics (median/MAD, frozen estimator choice);
- combined magnitude ``D_shift = lambda_img * D_img + lambda_mu * d_mu
  + lambda_sigma * d_sigma`` with frozen scales/weights;
- operating envelope per component with fail-closed refusal;
- no global RNG and no future access (deterministic transitions).

The fixture implementation remains for software safety tests only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shiftad.adaptation.base import (
    FIXTURE_SCOPE,
    SchemaRejectionError,
    ShiftEstimate,
)
from shiftad.adaptation.image_stats import IMAGE_STATISTIC_NAMES

EPS = 1e-8


@dataclass(frozen=True)
class SourceNormalStatistics:
    """Frozen source-normal reference statistics (fitted from D1 only).

    Attributes:
        image_mean: Per-statistic mean over source-normal images.
        image_std: Per-statistic std over source-normal images.
        mu_s: Source-normal feature mean (feature-level).
        sigma_s: Source-normal feature std (feature-level; same unit as the
            robust window MAD).
        n_samples: Number of source-normal samples used.
    """

    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]
    mu_s: tuple[float, ...]
    sigma_s: tuple[float, ...]
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize deterministically (protocol config-hash pattern)."""
        return {
            "image_mean": list(self.image_mean),
            "image_std": list(self.image_std),
            "mu_s": list(self.mu_s),
            "sigma_s": list(self.sigma_s),
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceNormalStatistics:
        """Reconstruct from :meth:`to_dict` output.

        Raises:
            SchemaRejectionError: If required fields are missing or malformed.
        """
        try:
            return cls(
                image_mean=tuple(float(v) for v in data["image_mean"]),
                image_std=tuple(float(v) for v in data["image_std"]),
                mu_s=tuple(float(v) for v in data["mu_s"]),
                sigma_s=tuple(float(v) for v in data["sigma_s"]),
                n_samples=int(data["n_samples"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaRejectionError(f"Malformed SourceNormalStatistics: {exc}") from exc


@dataclass(frozen=True)
class WindowStatistics:
    """Robust window statistics for the shift estimator.

    Attributes:
        image_stats: Image-statistic means over the window.
        mu_t: Feature mean over the window (None when unavailable).
        sigma_t: Feature covariance diagonal over the window (None when
            unavailable).
        occupancy: Fraction of the window with valid observations.
    """

    image_stats: dict[str, float] = field(default_factory=dict)
    mu_t: tuple[float, ...] | None = None
    sigma_t: tuple[float, ...] | None = None
    occupancy: float = 1.0


class ShiftEstimator(ABC):
    """Compute a typed shift estimate from image and feature statistics."""

    @abstractmethod
    def estimate(self, image_stats: dict[str, float], feature_stats: Any) -> ShiftEstimate:
        """Return the shift estimate for one sample.

        Args:
            image_stats: Image-space statistics (brightness, contrast, ...).
            feature_stats: Feature-space statistics for the current window.

        Returns:
            A deterministic shift estimate.
        """


class V1ShiftEstimator(ShiftEstimator):
    """Protocol-consistent shift estimator (candidate; frozen at Gate C).

    Args:
        source_statistics: Frozen source-normal reference statistics.
        lambda_img: Weight of the image-statistics displacement.
        lambda_mu: Weight of the feature-mean displacement.
        lambda_sigma: Weight of the feature-covariance displacement.
        window_length: Robust window length (median/MAD estimator).
        envelope: Per-component operating envelope as a dict of
            ``component -> (min, max)``; components outside the envelope
            refuse the estimate (fail-closed).
    """

    def __init__(
        self,
        source_statistics: SourceNormalStatistics,
        *,
        lambda_img: float = 1.0,
        lambda_mu: float = 1.0,
        lambda_sigma: float = 0.0,
        window_length: int = 8,
        envelope: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if len(source_statistics.image_mean) != len(IMAGE_STATISTIC_NAMES):
            raise SchemaRejectionError(
                "source_statistics.image_mean must match the frozen image-statistic set."
            )
        if window_length < 1:
            raise SchemaRejectionError("window_length must be >= 1.")
        self._source = source_statistics
        self._lambda_img = float(lambda_img)
        self._lambda_mu = float(lambda_mu)
        self._lambda_sigma = float(lambda_sigma)
        self._window_length = int(window_length)
        self._envelope = dict(envelope or {})

    # ------------------------------------------------------------------
    # Source normalization fitting (D1 source-normal only)
    # ------------------------------------------------------------------
    @classmethod
    def fit_source_normal(
        cls,
        image_stats: list[dict[str, float]],
        feature_rows: list[tuple[float, ...]] | None = None,
    ) -> SourceNormalStatistics:
        """Fit source image/feature normalization from source-normal data.

        Args:
            image_stats: Per-image statistic dicts (D1 source-normal only).
            feature_rows: Per-image feature rows (D1 source-normal only);
                None when feature statistics are unavailable.

        Returns:
            Frozen source-normal statistics.

        Raises:
            SchemaRejectionError: If the inputs are malformed.
        """
        if not image_stats:
            raise SchemaRejectionError("fit_source_normal requires at least one image.")
        names = list(IMAGE_STATISTIC_NAMES)
        matrix = np.asarray(
            [[float(stat[name]) for name in names] for stat in image_stats], dtype=np.float64
        )
        if matrix.shape[1] != len(names):
            raise SchemaRejectionError("image_statistic names must match the frozen set.")
        image_mean = tuple(float(v) for v in matrix.mean(axis=0))
        # Numerically-constant statistics (std below 1e-6) are treated as
        # noise-free references and normalized with unit scale.
        image_std = tuple(float(v) if v > 1e-6 else 1.0 for v in matrix.std(axis=0))

        mu_s: tuple[float, ...] = ()
        sigma_s: tuple[float, ...] = ()
        n_samples = len(image_stats)
        if feature_rows is not None:
            if len(feature_rows) != len(image_stats):
                raise SchemaRejectionError(
                    "feature_rows and image_stats must have the same length."
                )
            feat = np.asarray(feature_rows, dtype=np.float64)
            if feat.ndim != 2 or feat.shape[0] == 0:
                raise SchemaRejectionError("feature_rows must be a non-empty 2D array.")
            mu_s = tuple(float(v) for v in feat.mean(axis=0))
            # Standard deviation (same unit as the robust MAD in
            # ``robust_window``) so the sigma displacement is scale-consistent.
            sigma_s = tuple(float(v) for v in feat.std(axis=0))
        return SourceNormalStatistics(
            image_mean=image_mean,
            image_std=image_std,
            mu_s=mu_s,
            sigma_s=sigma_s,
            n_samples=n_samples,
        )

    # ------------------------------------------------------------------
    # Robust window statistics (frozen median/MAD estimator)
    # ------------------------------------------------------------------
    def robust_window(
        self,
        window: list[dict[str, float]],
        feature_window: list[tuple[float, ...]] | None = None,
    ) -> WindowStatistics:
        """Compute robust window statistics (median/MAD).

        Args:
            window: Image-statistic dicts for the current window.
            feature_window: Feature rows for the current window (optional).

        Returns:
            Robust window statistics with occupancy.

        Raises:
            SchemaRejectionError: If the window is empty or malformed.
        """
        if not window:
            raise SchemaRejectionError("robust_window requires a non-empty window.")
        names = list(IMAGE_STATISTIC_NAMES)
        matrix = np.asarray(
            [[float(stat[name]) for name in names] for stat in window], dtype=np.float64
        )
        occupancy = min(1.0, len(window) / self._window_length)
        image_stats = {name: float(np.median(matrix[:, i])) for i, name in enumerate(names)}
        mu_t: tuple[float, ...] | None = None
        sigma_t: tuple[float, ...] | None = None
        if feature_window is not None and feature_window:
            feat = np.asarray(feature_window, dtype=np.float64)
            mu_t = tuple(float(v) for v in np.median(feat, axis=0))
            sigma_t = tuple(
                float(v) for v in np.median(np.abs(feat - np.median(feat, axis=0)), axis=0)
            )
        return WindowStatistics(
            image_stats=image_stats,
            mu_t=mu_t,
            sigma_t=sigma_t,
            occupancy=occupancy,
        )

    # ------------------------------------------------------------------
    # Displacement computation
    # ------------------------------------------------------------------
    def _image_displacement(self, image_stats: dict[str, float]) -> float:
        names = list(IMAGE_STATISTIC_NAMES)
        if set(image_stats) != set(names):
            raise SchemaRejectionError(
                f"image_stats must contain exactly {names}, got {sorted(image_stats)}."
            )
        values = np.asarray([float(image_stats[name]) for name in names], dtype=np.float64)
        mean = np.asarray(self._source.image_mean, dtype=np.float64)
        std = np.asarray(self._source.image_std, dtype=np.float64)
        normalized = (values - mean) / std
        return float(np.linalg.norm(normalized, ord=2))

    def _feature_displacements(self, window: WindowStatistics) -> tuple[float, float]:
        d_mu = 0.0
        d_sigma = 0.0
        if window.mu_t is not None and self._source.mu_s:
            mu_s = np.asarray(self._source.mu_s, dtype=np.float64)
            mu_t = np.asarray(window.mu_t, dtype=np.float64)
            if mu_t.shape != mu_s.shape:
                raise SchemaRejectionError(
                    f"Feature dimension mismatch: mu_t {mu_t.shape} vs mu_s {mu_s.shape}."
                )
            d_mu = float(np.linalg.norm(mu_t - mu_s, ord=2))
        if window.sigma_t is not None and self._source.sigma_s:
            sigma_s = np.asarray(self._source.sigma_s, dtype=np.float64)
            sigma_t = np.asarray(window.sigma_t, dtype=np.float64)
            if sigma_t.shape != sigma_s.shape:
                raise SchemaRejectionError(
                    "Feature dimension mismatch: sigma_t "
                    f"{sigma_t.shape} vs sigma_s {sigma_s.shape}."
                )
            d_sigma = float(np.linalg.norm(sigma_t - sigma_s, ord=2))
        return d_mu, d_sigma

    def _check_envelope(self, displacement: dict[str, float]) -> str | None:
        """Return the failing component status, or None when in envelope."""
        for name, (lo, hi) in self._envelope.items():
            value = displacement.get(name)
            if value is None:
                return f"OUT_OF_ENVELOPE_MISSING:{name}"
            if not (lo <= value <= hi):
                return f"OUT_OF_ENVELOPE:{name}"
        return None

    # ------------------------------------------------------------------
    # Public estimate
    # ------------------------------------------------------------------
    def estimate(self, image_stats: dict[str, float], feature_stats: Any) -> ShiftEstimate:
        """Compute the shift estimate for one sample/window.

        Args:
            image_stats: Image-space statistics.
            feature_stats: A ``WindowStatistics`` instance (or None).

        Returns:
            A deterministic shift estimate with per-component contributions.

        Raises:
            SchemaRejectionError: If the inputs are malformed or unsupported.
        """
        if not isinstance(feature_stats, WindowStatistics) and feature_stats is not None:
            raise SchemaRejectionError("feature_stats must be a WindowStatistics instance or None.")
        d_img = self._image_displacement(image_stats)
        window = feature_stats if feature_stats is not None else WindowStatistics()
        d_mu, d_sigma = self._feature_displacements(window)
        magnitude = self._lambda_img * d_img + self._lambda_mu * d_mu + self._lambda_sigma * d_sigma
        components = {
            "image": self._lambda_img * d_img,
            "feature_mean": self._lambda_mu * d_mu,
            "feature_sigma": self._lambda_sigma * d_sigma,
        }
        envelope_status = self._check_envelope(components)
        if envelope_status is not None:
            return ShiftEstimate(
                magnitude=magnitude,
                components=components,
                confidence=window.occupancy,
                window_occupancy=window.occupancy,
                in_envelope=False,
                envelope_status=envelope_status,
            )
        return ShiftEstimate(
            magnitude=magnitude,
            components=components,
            confidence=window.occupancy,
            window_occupancy=window.occupancy,
            in_envelope=True,
            envelope_status="IN_ENVELOPE",
        )


class FixtureShiftEstimator(ShiftEstimator):
    """Deterministic fixture estimator (FIXTURE_SCOPE only).

    The fixture estimator returns a fixed zero-magnitude estimate. It exists
    to exercise the interface and causality invariants in tests; it is not a
    calibrated shift estimator and must not appear in formal results.
    """

    def __init__(self, *, magnitude: float = 0.0) -> None:
        self._magnitude = magnitude
        self.scope = FIXTURE_SCOPE

    def estimate(self, image_stats: dict[str, float], feature_stats: Any) -> ShiftEstimate:
        return ShiftEstimate(
            magnitude=self._magnitude,
            components={key: float(value) for key, value in sorted(image_stats.items())},
            confidence=1.0,
            window_occupancy=1.0,
            in_envelope=True,
            envelope_status="FIXTURE_IN_ENVELOPE",
        )
