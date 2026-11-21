"""Unified anomaly-detector contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import Tensor


class DetectorError(RuntimeError):
    """Report invalid model state, data leakage, or artifact loading."""


@dataclass(frozen=True)
class PredictionBatch:
    """Store image scores, pixel maps, timing, thresholds, and provenance."""

    image_scores: Tensor
    anomaly_maps: Tensor
    predicted_labels: Tensor | None
    threshold: float | None
    inference_times_ms: list[float]
    metadata: list[dict[str, Any]]


class AnomalyDetector(ABC):
    """Abstract interface shared by ShiftAD anomaly methods."""

    @abstractmethod
    def fit(self, train_loader: object) -> None:
        """Fit or construct model state from training data."""

    @abstractmethod
    def predict(self, images: Tensor) -> PredictionBatch:
        """Predict image scores and aligned anomaly maps."""

    @abstractmethod
    def save(self, output_dir: Path) -> None:
        """Save a complete, validated model artifact."""

    @classmethod
    @abstractmethod
    def load(cls, model_dir: Path) -> AnomalyDetector:
        """Load a complete, compatible model artifact."""

    def export(self, output_path: Path, export_format: str) -> None:
        """Reject deployment export until a method supplies a verified backend."""

        raise DetectorError(
            f"Export format {export_format!r} is not implemented for "
            f"{type(self).__name__}; Phase 8 will add verified deployment export "
            f"to {output_path}."
        )
