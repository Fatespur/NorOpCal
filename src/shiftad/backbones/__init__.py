"""Frozen visual backbones and patch-feature fusion."""

from shiftad.backbones.base import FeatureBackbone, FeatureBatch
from shiftad.backbones.dinov2 import DinoV2Backbone
from shiftad.backbones.feature_extractor import FeatureFuser, FusedFeatures
from shiftad.backbones.fixture import DeterministicFixtureBackbone

__all__ = [
    "DeterministicFixtureBackbone",
    "DinoV2Backbone",
    "FeatureBackbone",
    "FeatureBatch",
    "FeatureFuser",
    "FusedFeatures",
]
