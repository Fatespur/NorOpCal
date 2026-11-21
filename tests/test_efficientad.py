"""EfficientAD detector, model component, teacher-freezing, and serialization tests."""

import json
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from shiftad.data.fixture import SyntheticFixtureDataset, collate_anomaly_samples
from shiftad.methods.base import DetectorError
from shiftad.methods.efficientad import EfficientADConfig, EfficientADDetector
from shiftad.methods.factory import create_detector, load_detector
from shiftad.models.efficientad.autoencoder import AutoEncoder, FixtureAutoEncoder
from shiftad.models.efficientad.losses import EfficientADLoss, global_loss, local_loss
from shiftad.models.efficientad.normalization import FeatureNormalizer
from shiftad.models.efficientad.student import FixtureStudentNetwork, StudentNetwork
from shiftad.models.efficientad.teacher import FixtureTeacherNetwork
from shiftad.utils.config import (
    EfficientADModelConfig,
    ExperimentConfig,
    load_experiment_config,
)


def _fixture_config() -> ExperimentConfig:
    """Load the EfficientAD fixture experiment configuration."""

    config = load_experiment_config(Path("configs/experiments/efficientad_fixture.yaml"))
    assert config.model is not None
    assert isinstance(config.model, EfficientADModelConfig)
    return config


def _runtime_config(**overrides: object) -> EfficientADConfig:
    """Build a small runtime config for fixture tests."""

    defaults: dict[str, object] = {
        "is_fixture": True,
        "teacher_pretrained": False,
        "teacher_allow_download": False,
        "teacher_use_cache": False,
        "teacher_channels": 16,
        "ae_channels": 16,
        "input_size": (32, 32),
        "num_steps": 10,
        "batch_size": 2,
        "image_threshold": 0.05,
        "pixel_threshold": 0.05,
        "threshold_source": "fixture_predeclared_not_dataset_tuned",
    }
    defaults.update(overrides)
    return EfficientADConfig(**defaults)  # type: ignore[arg-type]


def _fitted_detector() -> EfficientADDetector:
    """Return a detector fitted on the synthetic fixture dataset."""

    detector = EfficientADDetector(_runtime_config(), device="cpu")
    train = SyntheticFixtureDataset("train", (32, 32))
    detector.fit(
        DataLoader(
            train,
            batch_size=2,
            collate_fn=collate_anomaly_samples,
        )
    )
    return detector


# ---------------------------------------------------------------------------
# Model structure tests
# ---------------------------------------------------------------------------


class TestModelStructure:
    """Verify network output shapes for fixture and real architectures."""

    def test_fixture_teacher_output_shape(self) -> None:
        """Fixture teacher produces [B, C, H/8, W/8] features."""

        teacher = FixtureTeacherNetwork(out_channels=16)
        images = torch.randn(2, 3, 32, 32)
        output = teacher(images)
        assert output.shape == (2, 16, 4, 4)

    def test_fixture_student_output_shape(self) -> None:
        """Fixture student produces [B, C_total, H/8, W/8] features."""

        student = FixtureStudentNetwork(teacher_channels=16, ae_channels=16)
        images = torch.randn(2, 3, 32, 32)
        output = student(images)
        assert output.shape == (2, 32, 4, 4)

    def test_fixture_student_split_output(self) -> None:
        """Student output splits into teacher-match and AE channels."""

        student = FixtureStudentNetwork(teacher_channels=16, ae_channels=16)
        images = torch.randn(1, 3, 32, 32)
        output = student(images)
        teacher_match, ae_input = student.split_output(output)
        assert teacher_match.shape == (1, 16, 4, 4)
        assert ae_input.shape == (1, 16, 4, 4)

    def test_fixture_autoencoder_output_shape(self) -> None:
        """Fixture autoencoder preserves input shape."""

        ae = FixtureAutoEncoder(in_channels=16)
        features = torch.randn(2, 16, 4, 4)
        output = ae(features)
        assert output.shape == features.shape

    def test_real_student_output_shape(self) -> None:
        """Real student produces [B, 256, H/8, W/8] features."""

        student = StudentNetwork(teacher_channels=128, ae_channels=128)
        images = torch.randn(1, 3, 256, 256)
        output = student(images)
        assert output.shape == (1, 256, 32, 32)

    def test_real_autoencoder_output_shape(self) -> None:
        """Real autoencoder preserves input shape."""

        ae = AutoEncoder(in_channels=128)
        features = torch.randn(1, 128, 32, 32)
        output = ae(features)
        assert output.shape == features.shape

    def test_student_rejects_invalid_input(self) -> None:
        """Student rejects non-4D or non-RGB input."""

        student = FixtureStudentNetwork(teacher_channels=16, ae_channels=16)
        with pytest.raises(RuntimeError, match=r"expect.*\[B, 3, H, W\]"):
            student(torch.randn(2, 3, 32))

    def test_autoencoder_rejects_wrong_channels(self) -> None:
        """Autoencoder rejects input with wrong channel count."""

        ae = FixtureAutoEncoder(in_channels=16)
        with pytest.raises(RuntimeError, match="channels"):
            ae(torch.randn(1, 32, 4, 4))

    def test_batch_size_variation(self) -> None:
        """Networks handle varying batch sizes."""

        teacher = FixtureTeacherNetwork(out_channels=16)
        student = FixtureStudentNetwork(teacher_channels=16, ae_channels=16)
        ae = FixtureAutoEncoder(in_channels=16)
        for batch_size in (1, 3, 5):
            images = torch.randn(batch_size, 3, 32, 32)
            teacher_out = teacher(images)
            student_out = student(images)
            assert teacher_out.shape[0] == batch_size
            assert student_out.shape[0] == batch_size
            _, ae_input = student.split_output(student_out)
            ae_out = ae(ae_input)
            assert ae_out.shape[0] == batch_size


# ---------------------------------------------------------------------------
# Teacher freezing tests
# ---------------------------------------------------------------------------


class TestTeacherFrozen:
    """Verify that the teacher network remains frozen."""

    def test_fixture_teacher_parameters_no_grad(self) -> None:
        """Fixture teacher parameters do not require gradients."""

        teacher = FixtureTeacherNetwork(out_channels=16)
        for param in teacher.parameters():
            assert not param.requires_grad

    def test_fixture_teacher_assert_frozen(self) -> None:
        """assert_frozen() does not raise for a frozen teacher."""

        teacher = FixtureTeacherNetwork(out_channels=16)
        teacher.assert_frozen()

    def test_teacher_not_in_optimizer(self) -> None:
        """Teacher parameters are excluded from the optimizer."""

        detector = _fitted_detector()
        assert detector.teacher is not None
        assert detector.student is not None
        assert detector.autoencoder is not None
        student_params = set(id(p) for p in detector.student.parameters())
        ae_params = set(id(p) for p in detector.autoencoder.parameters())
        teacher_params = set(id(p) for p in detector.teacher.parameters())
        assert not (student_params & teacher_params)
        assert not (ae_params & teacher_params)

    def test_teacher_parameters_unchanged_after_fit(self) -> None:
        """Teacher parameters do not change during training."""

        detector = EfficientADDetector(_runtime_config(), device="cpu")
        assert detector.teacher is not None
        before = {n: p.clone() for n, p in detector.teacher.named_parameters()}
        train = SyntheticFixtureDataset("train", (32, 32))
        detector.fit(
            DataLoader(
                train,
                batch_size=2,
                collate_fn=collate_anomaly_samples,
            )
        )
        after = {n: p.clone() for n, p in detector.teacher.named_parameters()}
        for name in before:
            assert torch.equal(before[name], after[name]), (
                f"Teacher parameter {name!r} changed during fit()."
            )


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------


class TestLoss:
    """Verify loss computation properties."""

    def test_local_loss_is_finite(self) -> None:
        """Local loss produces a finite scalar."""

        student_match = torch.randn(2, 16, 4, 4)
        teacher_output = torch.randn(2, 16, 4, 4)
        loss = local_loss(student_match, teacher_output)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_global_loss_is_finite(self) -> None:
        """Global loss produces a finite scalar."""

        ae_reconstruction = torch.randn(2, 16, 4, 4)
        ae_target = torch.randn(2, 16, 4, 4)
        loss = global_loss(ae_reconstruction, ae_target)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_local_loss_backward(self) -> None:
        """Local loss supports backpropagation."""

        student_match = torch.randn(2, 16, 4, 4, requires_grad=True)
        teacher_output = torch.randn(2, 16, 4, 4)
        loss = local_loss(student_match, teacher_output)
        loss.backward()
        assert student_match.grad is not None

    def test_global_loss_backward(self) -> None:
        """Global loss supports backpropagation."""

        ae_reconstruction = torch.randn(2, 16, 4, 4, requires_grad=True)
        ae_target = torch.randn(2, 16, 4, 4)
        loss = global_loss(ae_reconstruction, ae_target)
        loss.backward()
        assert ae_reconstruction.grad is not None

    def test_combined_loss_weights(self) -> None:
        """Combined loss applies lambda_global correctly."""

        student_match = torch.randn(1, 16, 4, 4)
        teacher_output = torch.randn(1, 16, 4, 4)
        ae_reconstruction = torch.randn(1, 16, 4, 4)
        ae_target = torch.randn(1, 16, 4, 4)
        loss_fn = EfficientADLoss(lambda_global=2.0)
        losses = loss_fn(student_match, teacher_output, ae_reconstruction, ae_target)
        expected = losses["local"] + 2.0 * losses["global"]
        assert torch.allclose(losses["total"], expected)

    def test_local_loss_shape_mismatch_raises(self) -> None:
        """Local loss raises on shape mismatch."""

        with pytest.raises(RuntimeError, match="mismatch"):
            local_loss(torch.randn(1, 16, 4, 4), torch.randn(1, 16, 8, 8))

    def test_lambda_global_negative_raises(self) -> None:
        """EfficientADLoss rejects negative lambda_global."""

        with pytest.raises(ValueError, match="non-negative"):
            EfficientADLoss(lambda_global=-1.0)


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------


class TestFeatureNormalizer:
    """Verify feature normalization behavior."""

    def test_unfit_normalizer_is_noop(self) -> None:
        """Unfit normalizer returns input unchanged."""

        normalizer = FeatureNormalizer()
        assert not normalizer.is_fit
        features = torch.randn(1, 16, 4, 4)
        assert torch.equal(normalizer.normalize(features), features)

    def test_fit_then_normalize(self) -> None:
        """Fitted normalizer applies channel-wise normalization."""

        features = torch.randn(10, 16, 4, 4)
        normalizer = FeatureNormalizer.fit_from_features(features)
        assert normalizer.is_fit
        normalized = normalizer.normalize(features)
        assert normalized.shape == features.shape
        assert torch.isfinite(normalized).all()

    def test_state_dict_round_trip(self) -> None:
        """Normalizer state dict round-trips correctly."""

        features = torch.randn(5, 8, 4, 4)
        original = FeatureNormalizer.fit_from_features(features)
        state = original.state_dict()
        restored = FeatureNormalizer.from_state_dict(state)
        assert restored.is_fit
        input_tensor = torch.randn(1, 8, 4, 4)
        assert torch.equal(
            original.normalize(input_tensor),
            restored.normalize(input_tensor),
        )

    def test_partial_state_dict_raises(self) -> None:
        """Providing only mean or std raises."""

        with pytest.raises(ValueError, match="Both mean and std"):
            FeatureNormalizer(mean=torch.randn(8))


# ---------------------------------------------------------------------------
# Detector tests
# ---------------------------------------------------------------------------


class TestEfficientADDetector:
    """Verify detector fit, predict, serialization, and calibration."""

    def test_fit_predict_shapes_scores_and_reproducibility(self) -> None:
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

    def test_detector_rejects_anomalous_training_labels(self) -> None:
        """An anomalous train batch is rejected as a leakage risk."""

        detector = EfficientADDetector(_runtime_config(), device="cpu")
        batch = {
            "image": torch.ones(2, 3, 32, 32),
            "label": torch.tensor([0, 1]),
        }
        with pytest.raises(DetectorError, match="leakage"):
            detector.fit([batch])

    def test_predict_before_fit_raises(self) -> None:
        """Predict on an unfitted detector raises a clear error."""

        detector = EfficientADDetector(_runtime_config(), device="cpu")
        images = torch.randn(2, 3, 32, 32)
        with pytest.raises(DetectorError, match="fitted or loaded"):
            detector.predict(images)

    def test_predict_rejects_invalid_input(self) -> None:
        """Predict rejects non-4D or non-RGB input."""

        detector = _fitted_detector()
        with pytest.raises(DetectorError, match="\\[B, 3, H, W\\]"):
            detector.predict(torch.randn(2, 3, 32))

    def test_serialization_round_trip_and_hash_validation(self, tmp_path: Path) -> None:
        """Saved model reloads exactly and rejects tampered artifacts."""

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
            "teacher.pt",
            "student.pt",
            "autoencoder.pt",
            "normalizer.pt",
            "metadata.json",
        } <= {path.name for path in model_dir.iterdir()}
        manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["model_type"] == "efficientad"
        assert manifest["is_fixture"] is True
        assert (
            yaml.safe_load((model_dir / "config.resolved.yaml").read_text(encoding="utf-8"))[
                "is_fixture"
            ]
            is True
        )
        loaded = EfficientADDetector.load(model_dir, device="cpu")
        actual = loaded.predict(test_images)
        assert torch.allclose(actual.image_scores, expected.image_scores, atol=1e-6)
        assert torch.allclose(actual.anomaly_maps, expected.anomaly_maps, atol=1e-6)

        (model_dir / "metadata.json").write_text("{}", encoding="utf-8")
        with pytest.raises(DetectorError, match=r"(?i)hash-mismatched"):
            EfficientADDetector.load(model_dir, device="cpu")

    def test_factory_creates_efficientad_detector(self) -> None:
        """The detector factory dispatches to EfficientADDetector."""

        config = _fixture_config()
        assert config.model is not None
        assert isinstance(config.model, EfficientADModelConfig)
        detector = create_detector(config.model, device="cpu")
        assert isinstance(detector, EfficientADDetector)

    def test_factory_loads_efficientad_manifest(self, tmp_path: Path) -> None:
        """The load_detector factory dispatches based on manifest model_type."""

        detector = _fitted_detector()
        model_dir = tmp_path / "model"
        detector.save(model_dir)
        loaded = load_detector(model_dir, device="cpu")
        assert isinstance(loaded, EfficientADDetector)

    def test_anomalous_images_score_higher_than_normal(self) -> None:
        """Anomalous test images receive higher image scores than normal ones."""

        detector = _fitted_detector()
        test = SyntheticFixtureDataset("test", (32, 32))
        images = torch.stack([test[index]["image"] for index in range(len(test))])
        prediction = detector.predict(images)
        normal_scores = prediction.image_scores[:2]
        anomaly_scores = prediction.image_scores[2:]
        assert torch.all(anomaly_scores >= normal_scores.min())

    def test_build_metadata_recorded(self) -> None:
        """Fit records build metadata including parameter counts."""

        detector = _fitted_detector()
        assert detector.build_metadata["train_samples"] > 0
        assert detector.build_metadata["train_steps"] == 10
        assert detector.build_metadata["teacher_frozen"] is True
        assert detector.build_metadata["student_parameters"] > 0
        assert detector.build_metadata["autoencoder_parameters"] > 0

    def test_save_unfitted_raises(self, tmp_path: Path) -> None:
        """Saving an unfitted detector raises an error."""

        detector = EfficientADDetector(_runtime_config(), device="cpu")
        with pytest.raises(DetectorError, match="unfitted"):
            detector.save(tmp_path / "model")

    def test_load_missing_manifest_raises(self, tmp_path: Path) -> None:
        """Loading from a directory without manifest raises an error."""

        with pytest.raises(DetectorError, match="Manifest missing"):
            EfficientADDetector.load(tmp_path / "nonexistent", device="cpu")


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


class TestEfficientADConfig:
    """Verify configuration loading and validation."""

    def test_fixture_config_loads(self) -> None:
        """The fixture experiment config parses and validates correctly."""

        config = _fixture_config()
        assert config.model is not None
        assert isinstance(config.model, EfficientADModelConfig)
        assert config.model.name == "efficientad"
        assert config.model.is_fixture is True
        assert config.model.teacher.pretrained is False
        assert config.model.teacher.allow_download is False
        assert config.model.teacher.out_channels == 16
        assert config.model.ae_channels == 16
        assert config.model.training.num_steps == 5

    def test_default_model_config_loads(self) -> None:
        """The default EfficientAD model config parses correctly."""

        config = load_experiment_config(Path("configs/models/efficientad_default.yaml"))
        assert config.model is not None
        assert isinstance(config.model, EfficientADModelConfig)
        assert config.model.name == "efficientad"
        assert config.model.is_fixture is False
        assert config.model.teacher.pretrained is True
        assert config.model.teacher.allow_download is False
        assert config.model.teacher.out_channels == 128
        assert config.model.ae_channels == 128
        assert config.model.training.num_steps == 70000
        assert config.model.input_size == (256, 256)

    def test_phase45_base_config_uses_protocol(self) -> None:
        """The Phase 4.5 base config follows the shared protocol."""

        template = Path("configs/experiments/phase45/base_efficientad.yaml").read_text(
            encoding="utf-8"
        )
        resolved = template.format(category="bottle", seed=0, runtime_seed=42)
        import yaml as yaml_lib

        parsed = yaml_lib.safe_load(resolved)
        assert parsed["calibration"]["ratio"] == 0.20
        assert parsed["calibration"]["quantile"] == 0.95
        assert parsed["model"]["name"] == "efficientad"

    def test_build_memory_rejects_efficientad(self) -> None:
        """build-memory CLI rejects EfficientAD configs."""

        from shiftad.cli import main

        result = main(
            [
                "build-memory",
                "--config",
                "configs/experiments/efficientad_fixture.yaml",
            ]
        )
        assert result == 2
