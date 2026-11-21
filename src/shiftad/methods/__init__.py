"""Unified anomaly-detector interfaces and implementations."""

from shiftad.methods.base import AnomalyDetector, DetectorError, PredictionBatch
from shiftad.methods.dino_memory import DinoMemoryDetector

__all__ = [
    "AnomalyDetector",
    "DetectorError",
    "DinoMemoryDetector",
    "PredictionBatch",
]
