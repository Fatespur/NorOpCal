"""DINO memory detector, scoring, leakage, and serialization tests."""

import json
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from shiftad.backbones.fixture import DeterministicFixtureBackbone
from shiftad.data.fixture import SyntheticFixtureDataset, collate_anomaly_samples
from shiftad.methods.base import DetectorError
from shiftad.methods.dino_memory import DinoMemoryDetector
from shiftad.methods.scoring import (
    aggregate_image_scores,
    patch_scores_to_anomaly_maps,
)
from shiftad.utils.config import ExperimentConfig, load_experiment_config


def _config() -> ExperimentConfig:
    config = load_experiment_config(Path("configs/experiments/dino_memory_fixture.yaml"))
    assert config.model is not None
    return config


def _fitted_detector() -> DinoMemoryDetector:
    config = _config()
    assert config.model is not None
    detector = DinoMemoryDetector(config.model, device="cpu")
    train = SyntheticFixtureDataset("train", (32, 32))
    detector.fit(
        DataLoader(
            train,
            batch_size=2,
            collate_fn=collate_anomaly_samples,
        )
    )
    return detector


def test_detector_fit_predict_shapes_scores_and_reproducibility() -> None:
    """Normal fit produces deterministic image scores and aligned maps."""

    detector = _fitted_detector()
    test = SyntheticFixtureDataset("test", (32, 32))
    images = torch.stack([test[index]["image"] for index in range(len(test))])
    first = detector.predict(images)
    second = detector.predict(images)
    assert first.image_scores.shape == (4,)
    assert first.anomaly_maps.shape == (4, 32, 32)
    assert torch.equal(first.image_scores, second.image_scores)
    assert torch.equal(first.anomaly_maps, second.anomaly_maps)
    assert torch.isfinite(first.anomaly_maps).all()
    assert first.metadata[0]["artifact_scope"] == "synthetic_fixture_acceptance_only"
    assert detector.memory_bank is not None and detector.memory_bank.finalized


def test_detector_rejects_anomalous_training_labels() -> None:
    """An anomalous train batch is rejected as a leakage risk."""

    config = _config()
    assert config.model is not None
    detector = DinoMemoryDetector(config.model, device="cpu")
    batch = {
        "image": torch.ones(2, 3, 32, 32),
        "label": torch.tensor([0, 1]),
    }
    with pytest.raises(DetectorError, match="leakage"):
        detector.fit([batch])


def test_detector_serialization_round_trip_and_hash_validation(tmp_path: Path) -> None:
    """Saved memory reloads exactly and rejects tampered artifacts."""

    detector = _fitted_detector()
    test_images = torch.stack(
        [SyntheticFixtureDataset("test", (32, 32))[index]["image"] for index in range(4)]
    )
    expected = detector.predict(test_images)
    model_dir = tmp_path / "model"
    detector.save(model_dir)
    assert {
        "manifest.json",
        "config.resolved.yaml",
        "memory_bank.pt",
        "coreset_indices.pt",
        "metadata.json",
    } <= {path.name for path in model_dir.iterdir()}
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["real_dinov2_weights"] is False
    assert (
        yaml.safe_load((model_dir / "config.resolved.yaml").read_text(encoding="utf-8"))["name"]
        == "dino_memory"
    )
    loaded = DinoMemoryDetector.load(model_dir, device="cpu")
    actual = loaded.predict(test_images)
    assert torch.equal(actual.image_scores, expected.image_scores)
    assert torch.equal(actual.anomaly_maps, expected.anomaly_maps)

    (model_dir / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DetectorError, match="hash-mismatched"):
        DinoMemoryDetector.load(model_dir, device="cpu")


def test_spatial_map_and_image_aggregation_known_cases() -> None:
    """Single-patch location, batch order, top-k bounds, and smoothing are stable."""

    scores = torch.tensor([[0.0, 1.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    maps = patch_scores_to_anomaly_maps(
        scores,
        grid_shape=(2, 2),
        output_size=(2, 2),
    )
    assert torch.equal(maps, scores.reshape(2, 2, 2))
    assert maps[0].argmax() == 1
    assert maps[1].argmax() == 0
    smoothed = patch_scores_to_anomaly_maps(
        scores,
        grid_shape=(2, 2),
        output_size=(8, 8),
        smoothing_sigma=1.0,
    )
    assert smoothed.shape == (2, 8, 8)
    assert torch.equal(
        aggregate_image_scores(scores, mode="max", top_k=1),
        torch.tensor([1.0, 2.0]),
    )
    assert torch.equal(
        aggregate_image_scores(scores, mode="top_k_mean", top_k=2),
        torch.tensor([0.5, 1.0]),
    )
    with pytest.raises(DetectorError, match="top_k"):
        aggregate_image_scores(scores, mode="top_k_mean", top_k=5)


def test_fixture_backbone_dependency_can_be_injected() -> None:
    """Detector accepts a frozen provider without global state."""

    config = _config()
    assert config.model is not None
    backbone = DeterministicFixtureBackbone(layers=(0, 1), patch_size=8)
    detector = DinoMemoryDetector(config.model, backbone=backbone)
    assert detector.backbone is backbone
