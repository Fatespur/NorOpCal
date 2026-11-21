"""EfficientAD model components: teacher, student, autoencoder, and losses.

This package implements the EfficientAD architecture adapted for the ShiftAD
project. The design follows the teacher-student distillation paradigm combined
with an autoencoder for global anomaly detection, as described in:

    B. J. Holzinger et al., "EfficientAD: Accurate Visual Anomaly Detection at
    Millisecond-Level Latencies," WACV 2024.

Key adaptations from the original paper:
- The teacher uses a torchvision pretrained ResNet18 feature extractor instead
  of a separately pretrained PDN, avoiding silent external weight downloads.
- The student and autoencoder use compact PDN-style architectures.
- All components are configurable and integrate with the ShiftAD unified
  AnomalyDetector interface.

Reference code: https://github.com/nelson1425/EfficientAD (MIT License)
"""

from shiftad.models.efficientad.autoencoder import AutoEncoder
from shiftad.models.efficientad.losses import EfficientADLoss
from shiftad.models.efficientad.normalization import FeatureNormalizer
from shiftad.models.efficientad.student import StudentNetwork
from shiftad.models.efficientad.teacher import TeacherNetwork

__all__ = [
    "AutoEncoder",
    "EfficientADLoss",
    "FeatureNormalizer",
    "StudentNetwork",
    "TeacherNetwork",
]
