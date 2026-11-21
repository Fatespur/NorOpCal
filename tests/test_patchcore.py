"""PatchCore detector, scoring, leakage, and serialization tests."""

import json
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from shiftad.backbones.fixture import DeterministicFixtureBackbone
from shiftad.data.fixture import SyntheticFixtureDataset, collate_anomaly_samples
from shiftad.memory.memory_bank import MemoryBankError
from shiftad.methods.base import DetectorError
from shiftad.methods.factory import create_detector, load_detector
from shiftad.methods.patchcore import PatchCoreDetector
from shiftad.utils.config import ExperimentConfig, PatchCoreConfig, load_experiment_config


def _config() -> ExperimentConfig:
    config = load_experiment_config(Path("configs/experiments/patchcore_fixture.yaml"))
    assert config.model is not None
    return config


def _fitted_detector() -> PatchCoreDetector:
    config = _config()
    assert config.model is not None
    assert isinstance(config.model, PatchCoreConfig)
    detector = PatchCoreDetector(config.model, device="cpu")
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
    assert isinstance(config.model, PatchCoreConfig)
    detector = PatchCoreDetector(config.model, device="cpu")
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
    assert manifest["model_type"] == "patchcore"
    assert (
        yaml.safe_load((model_dir / "config.resolved.yaml").read_text(encoding="utf-8"))["name"]
        == "patchcore"
    )
    loaded = PatchCoreDetector.load(model_dir, device="cpu")
    actual = loaded.predict(test_images)
    assert torch.equal(actual.image_scores, expected.image_scores)
    assert torch.equal(actual.anomaly_maps, expected.anomaly_maps)

    (model_dir / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DetectorError, match="hash-mismatched"):
        PatchCoreDetector.load(model_dir, device="cpu")


def test_factory_creates_patchcore_detector() -> None:
    """The detector factory dispatches to PatchCoreDetector for patchcore configs."""

    config = _config()
    assert config.model is not None
    assert isinstance(config.model, PatchCoreConfig)
    detector = create_detector(config.model, device="cpu")
    assert isinstance(detector, PatchCoreDetector)


def test_factory_loads_patchcore_manifest(tmp_path: Path) -> None:
    """The load_detector factory dispatches based on manifest model_type."""

    detector = _fitted_detector()
    model_dir = tmp_path / "model"
    detector.save(model_dir)
    loaded = load_detector(model_dir, device="cpu")
    assert isinstance(loaded, PatchCoreDetector)


def test_source_memory_is_immutable_after_fit() -> None:
    """Finalized source memory rejects additional features."""

    detector = _fitted_detector()
    assert detector.memory_bank is not None
    extra = torch.zeros(1, detector.memory_bank.feature_dimension)
    with pytest.raises(MemoryBankError, match="immutable"):
        detector.memory_bank.add(extra)


def test_fixture_backbone_dependency_can_be_injected() -> None:
    """Detector accepts a frozen provider without global state."""

    config = _config()
    assert config.model is not None
    assert isinstance(config.model, PatchCoreConfig)
    backbone = DeterministicFixtureBackbone(layers=(0, 1), patch_size=8)
    detector = PatchCoreDetector(config.model, backbone=backbone)
    assert detector.backbone is backbone


def test_predict_before_fit_raises() -> None:
    """Predict on an unfitted detector raises a clear error."""

    config = _config()
    assert config.model is not None
    assert isinstance(config.model, PatchCoreConfig)
    detector = PatchCoreDetector(config.model, device="cpu")
    images = torch.randn(2, 3, 32, 32)
    with pytest.raises(DetectorError, match="fitted or loaded"):
        detector.predict(images)


def test_default_patchcore_config_loads() -> None:
    """The default PatchCore experiment config parses and validates correctly."""

    config = load_experiment_config(Path("configs/models/patchcore_default.yaml"))
    assert config.model is not None
    assert isinstance(config.model, PatchCoreConfig)
    assert config.model.name == "patchcore"
    assert config.model.backbone.name == "wide_resnet50_2"
    assert tuple(config.model.backbone.layers) == (2, 3)
    assert config.model.backbone.input_size == (224, 224)
    assert config.model.memory.coreset.ratio == 0.1
    assert config.model.memory.nearest_neighbors.metric == "euclidean"
    assert config.model.memory.nearest_neighbors.k == 1


def test_anomalous_images_score_higher_than_normal() -> None:
    """Anomalous test images receive higher image scores than normal ones."""

    detector = _fitted_detector()
    test = SyntheticFixtureDataset("test", (32, 32))
    images = torch.stack([test[index]["image"] for index in range(len(test))])
    prediction = detector.predict(images)
    normal_scores = prediction.image_scores[:2]
    anomaly_scores = prediction.image_scores[2:]
    assert torch.all(anomaly_scores >= normal_scores.min())
