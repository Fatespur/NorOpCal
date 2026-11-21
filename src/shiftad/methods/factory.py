"""Detector construction and loading dispatch for unified experiment workflows.

This module provides factory functions that construct the correct
:class:`~shiftad.methods.base.AnomalyDetector` subclass from a model
configuration or a saved model manifest, allowing the CLI and workflow
layer to remain model-agnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from shiftad.backbones.base import FeatureBackbone
from shiftad.methods.base import AnomalyDetector, DetectorError
from shiftad.methods.dino_memory import DinoMemoryDetector
from shiftad.methods.efficientad import EfficientADConfig, EfficientADDetector
from shiftad.methods.patchcore import PatchCoreDetector
from shiftad.utils.config import (
    DinoMemoryConfig,
    EfficientADModelConfig,
    PatchCoreConfig,
)

#: Union of all model configuration types accepted by the factory.
ModelConfig = DinoMemoryConfig | PatchCoreConfig | EfficientADModelConfig


def create_detector(
    config: ModelConfig,
    *,
    device: torch.device | str = "cpu",
    backbone: FeatureBackbone | None = None,
) -> AnomalyDetector:
    """Construct the correct detector subclass from a model configuration.

    Args:
        config: Validated model configuration dataclass.
        device: CPU or available accelerator request.
        backbone: Optional injected provider for isolated tests
            (DINO memory and PatchCore only).

    Returns:
        A configured but unfitted anomaly detector.

    Raises:
        DetectorError: If the model configuration type is unsupported.
    """

    if isinstance(config, DinoMemoryConfig):
        return DinoMemoryDetector(config, device=device, backbone=backbone)
    if isinstance(config, PatchCoreConfig):
        return PatchCoreDetector(config, device=device, backbone=backbone)
    if isinstance(config, EfficientADModelConfig):
        return _create_efficientad(config, device=device)
    raise DetectorError(
        f"Unsupported model configuration type {type(config).__name__}; "
        "expected DinoMemoryConfig, PatchCoreConfig, or EfficientADModelConfig."
    )


def _create_efficientad(
    config: EfficientADModelConfig,
    *,
    device: torch.device | str,
) -> EfficientADDetector:
    """Convert frozen dataclass config to runtime config and create detector.

    Args:
        config: Validated EfficientAD model configuration.
        device: CPU or available accelerator request.

    Returns:
        A configured but unfitted EfficientADDetector.
    """

    runtime_config = EfficientADConfig(
        is_fixture=config.is_fixture,
        teacher_pretrained=config.teacher.pretrained,
        teacher_checkpoint=config.teacher.checkpoint,
        teacher_allow_download=config.teacher.allow_download,
        teacher_use_cache=config.teacher.use_cache,
        teacher_channels=config.teacher.out_channels,
        ae_channels=config.ae_channels,
        input_size=config.input_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        num_steps=config.training.num_steps,
        batch_size=config.training.batch_size,
        lambda_global=config.training.lambda_global,
        local_weight=config.scoring.local_weight,
        global_weight=config.scoring.global_weight,
        smoothing_sigma=config.scoring.smoothing_sigma,
        image_threshold=config.scoring.image_threshold,
        pixel_threshold=config.scoring.pixel_threshold,
        threshold_source=config.scoring.threshold_source,
        checkpoint_enabled=config.checkpoint.enabled,
        checkpoint_interval=config.checkpoint.interval,
        checkpoint_dir=config.checkpoint.dir,
        resume_from=config.checkpoint.resume_from,
        auto_resume=config.checkpoint.auto_resume,
        max_to_keep=config.checkpoint.max_to_keep,
    )
    return EfficientADDetector(runtime_config, device=device)


def load_detector(
    model_dir: Path,
    *,
    device: torch.device | str = "cpu",
    backbone: FeatureBackbone | None = None,
) -> AnomalyDetector:
    """Load a saved detector by reading its manifest model type.

    Args:
        model_dir: Directory produced by a detector ``save()`` call.
        device: CPU or available accelerator request.
        backbone: Optional injected provider for isolated tests
            (DINO memory and PatchCore only).

    Returns:
        A loaded and verified anomaly detector.

    Raises:
        DetectorError: If the manifest is missing, unreadable, or contains
            an unsupported model type.
    """

    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise DetectorError(f"Model manifest is missing: {manifest_path}.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DetectorError(f"Cannot read model manifest {manifest_path}: {exc}") from exc
    model_type = manifest.get("model_type")
    if model_type == "dino_memory":
        return DinoMemoryDetector.load(model_dir, device=device, backbone=backbone)
    if model_type == "patchcore":
        return PatchCoreDetector.load(model_dir, device=device, backbone=backbone)
    if model_type == "efficientad":
        return EfficientADDetector.load(model_dir, device=device)
    raise DetectorError(
        f"Unsupported model_type {model_type!r} in manifest {manifest_path}; "
        "expected 'dino_memory', 'patchcore', or 'efficientad'."
    )
