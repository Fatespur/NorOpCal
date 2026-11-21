"""Construct configured frozen feature backbones."""

from __future__ import annotations

import torch

from shiftad.backbones.base import FeatureBackbone
from shiftad.backbones.dinov2 import DinoV2Backbone
from shiftad.backbones.fixture import DeterministicFixtureBackbone
from shiftad.backbones.resnet import SUPPORTED_MODELS as RESNET_MODELS
from shiftad.backbones.resnet import ResNetBackbone
from shiftad.utils.config import BackboneConfig

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def create_backbone(
    config: BackboneConfig,
    *,
    device: torch.device | str,
) -> FeatureBackbone:
    """Create the configured fixture, DINOv2, or ResNet provider.

    Args:
        config: Strictly validated backbone settings.
        device: Explicit execution device.

    Returns:
        A frozen feature provider.
    """

    if config.name == "deterministic_fixture":
        return DeterministicFixtureBackbone(
            layers=config.layers,
            patch_size=config.fixture_patch_size,
            device=device,
        )
    if config.name in RESNET_MODELS:
        return ResNetBackbone(
            model_name=config.name,
            layers=config.layers,
            device=device,
            dtype=config.dtype,
            checkpoint=config.checkpoint,
            allow_download=config.allow_download,
            use_cache=config.use_cache,
            expected_input_size=config.input_size,
        )
    return DinoV2Backbone(
        model_name=config.name,
        layers=config.layers,
        device=device,
        dtype=DTYPES[config.dtype],
        checkpoint=config.checkpoint,
        allow_download=config.allow_download,
        use_cache=config.use_cache,
        expected_input_size=config.input_size,
    )
