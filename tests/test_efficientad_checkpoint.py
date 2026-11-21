"""EfficientAD checkpoint save/restore and resume tests.

All tests use synthetic fixture data and small step counts.  The cumulative
training step budget across this module is strictly bounded (see
``CUMULATIVE_STEP_BUDGET`` below).

Test step budget summary:
  - Module fixture (6 steps): trains 6 steps, saves checkpoint at step 6.
  - test_resume_continues_from_saved_step (18 steps): 12 initial + 6 resumed.
  - test_resume_skips_completed_training (6 steps): initial run only.
  - test_auto_resume_finds_latest (12 steps): single run with interval=6.
  - test_auto_resume_no_checkpoint_starts_from_zero (6 steps).
  - test_optimizer_step_count_after_resume (9 steps): 6 initial + 3 resumed.
  - test_hash_sidecar_write_failure_preserves_old_checkpoint (0 steps).
  - test_complete_marker_creation_failure_preserves_old_checkpoint (0 steps).
  - All other tests: 0 additional training steps.
  - Total: 6+18+6+12+6+9+0+0 = 57 steps (within the 100-step budget).
"""

from __future__ import annotations

import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from shiftad.data.fixture import SyntheticFixtureDataset, collate_anomaly_samples
from shiftad.methods.efficientad import (
    CHECKPOINT_VERSION,
    CheckpointError,
    EfficientADConfig,
    EfficientADDetector,
    NormalizerAccumulator,
)

#: Cumulative training step budget for this test module.
CUMULATIVE_STEP_BUDGET = 57


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fixture_config(
    *,
    num_steps: int = 6,
    **overrides: Any,
) -> EfficientADConfig:
    """Build a small fixture runtime config for checkpoint tests."""

    defaults: dict[str, Any] = {
        "is_fixture": True,
        "teacher_pretrained": False,
        "teacher_allow_download": False,
        "teacher_use_cache": False,
        "teacher_channels": 16,
        "ae_channels": 16,
        "input_size": (32, 32),
        "num_steps": num_steps,
        "batch_size": 2,
        "smoothing_sigma": 0.0,
    }
    defaults.update(overrides)
    return EfficientADConfig(**defaults)


def _train_loader() -> DataLoader[Any]:
    """Return a DataLoader for the synthetic fixture train split."""

    return DataLoader(
        SyntheticFixtureDataset("train", (32, 32)),
        batch_size=2,
        collate_fn=collate_anomaly_samples,
    )


def _make_optimizer(detector: EfficientADDetector) -> torch.optim.Optimizer:
    """Create an Adam optimizer matching the detector's trainable parameters."""

    assert detector.student is not None and detector.autoencoder is not None
    return torch.optim.Adam(
        list(detector.student.parameters()) + list(detector.autoencoder.parameters()),
        lr=detector.config.learning_rate,
        weight_decay=detector.config.weight_decay,
    )


@dataclass
class CheckpointSetup:
    """Bundle of objects created by the module-scoped fixture."""

    checkpoint_dir: Path
    checkpoint_path: Path
    hash_path: Path
    complete_path: Path
    config: EfficientADConfig
    compatibility_info: dict[str, Any]
    step: int
    student_state_dict: dict[str, Tensor]
    autoencoder_state_dict: dict[str, Tensor]


# ---------------------------------------------------------------------------
# Module-scoped fixture: train 6 steps, save checkpoint (6 steps total)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ckpt_setup(tmp_path_factory: pytest.TempPathFactory) -> CheckpointSetup:
    """Train for 6 steps and save a checkpoint at step 6.

    Training steps consumed: 6.
    """

    base = tmp_path_factory.mktemp("ckpt_module")
    ckpt_dir = base / "checkpoints"
    config = _fixture_config(num_steps=6)
    compatibility_info: dict[str, Any] = {
        "category": "widget",
        "seed": 42,
        "split_fingerprint": "test_split_fp_a",
    }
    detector = EfficientADDetector(config, device="cpu")
    detector.fit(
        _train_loader(),
        checkpoint_dir=ckpt_dir,
        checkpoint_interval=6,
        auto_resume=False,
        compatibility_info=compatibility_info,
    )
    ckpt_path = ckpt_dir / "checkpoint_step_00000006.pt"
    hash_path = ckpt_path.with_suffix(".pt.sha256")
    complete_path = ckpt_path.with_suffix(".pt.complete")
    assert ckpt_path.is_file(), "Checkpoint file was not created."
    assert hash_path.is_file(), "Checkpoint hash sidecar was not created."
    assert complete_path.is_file(), "Checkpoint complete marker was not created."
    student_sd = {
        n: p.clone()
        for n, p in detector.student.named_parameters()  # type: ignore[union-attr]
    }
    autoencoder_sd = {
        n: p.clone()
        for n, p in detector.autoencoder.named_parameters()  # type: ignore[union-attr]
    }
    return CheckpointSetup(
        checkpoint_dir=ckpt_dir,
        checkpoint_path=ckpt_path,
        hash_path=hash_path,
        complete_path=complete_path,
        config=config,
        compatibility_info=compatibility_info,
        step=6,
        student_state_dict=student_sd,
        autoencoder_state_dict=autoencoder_sd,
    )


def _fresh_detector(config: EfficientADConfig) -> EfficientADDetector:
    """Create a fresh detector with the given config (untrained)."""

    return EfficientADDetector(config, device="cpu")


# ---------------------------------------------------------------------------
# Phase 1.1 verification: checkpoint file creation and basic structure
# ---------------------------------------------------------------------------


class TestCheckpointCreation:
    """Verify checkpoint file creation, hash sidecar, complete marker, and structure."""

    def test_checkpoint_file_created(self, ckpt_setup: CheckpointSetup) -> None:
        """Checkpoint .pt file exists and is non-empty."""

        assert ckpt_setup.checkpoint_path.is_file()
        assert ckpt_setup.checkpoint_path.stat().st_size > 0

    def test_hash_sidecar_created(self, ckpt_setup: CheckpointSetup) -> None:
        """Hash sidecar .sha256 file exists and contains a valid hash."""

        assert ckpt_setup.hash_path.is_file()
        content = ckpt_setup.hash_path.read_text(encoding="utf-8").strip()
        parts = content.split()
        assert len(parts) >= 1
        assert len(parts[0]) == 64  # SHA-256 hex digest

    def test_complete_marker_created(self, ckpt_setup: CheckpointSetup) -> None:
        """Complete marker .complete file exists and is non-empty."""

        assert ckpt_setup.complete_path.is_file()
        content = ckpt_setup.complete_path.read_text(encoding="utf-8").strip()
        assert len(content) > 0

    def test_checkpoint_version_correct(self, ckpt_setup: CheckpointSetup) -> None:
        """Checkpoint contains the expected version number."""

        ckpt = torch.load(ckpt_setup.checkpoint_path, map_location="cpu", weights_only=False)
        assert ckpt["checkpoint_version"] == CHECKPOINT_VERSION

    def test_no_teacher_features_in_checkpoint(self, ckpt_setup: CheckpointSetup) -> None:
        """Checkpoint does not contain a complete teacher feature list."""

        ckpt = torch.load(ckpt_setup.checkpoint_path, map_location="cpu", weights_only=False)
        model_state = ckpt["model_state"]
        assert "teacher_features" not in model_state, (
            "Checkpoint must not contain teacher_features list."
        )
        assert "normalizer_accumulator" in model_state, (
            "Checkpoint must contain normalizer_accumulator state."
        )

    def test_temporary_file_not_valid_checkpoint(
        self,
        ckpt_setup: CheckpointSetup,
        tmp_path: Path,
    ) -> None:
        """Temporary files are not picked up as valid checkpoints."""

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        # Create a temp file that mimics a failed atomic write.
        (ckpt_dir / ".ckpt_tmp_abc.pt").write_bytes(b"garbage")
        assert EfficientADDetector._find_latest_checkpoint(ckpt_dir) is None

    def test_file_without_hash_sidecar_skipped(
        self,
        ckpt_setup: CheckpointSetup,
        tmp_path: Path,
    ) -> None:
        """A .pt file without a .sha256 sidecar is not a valid checkpoint."""

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "checkpoint_step_00000003.pt").write_bytes(b"garbage")
        # No .sha256 sidecar → should not be found.
        assert EfficientADDetector._find_latest_checkpoint(ckpt_dir) is None

    def test_file_without_complete_marker_skipped(
        self,
        ckpt_setup: CheckpointSetup,
        tmp_path: Path,
    ) -> None:
        """A .pt file with .sha256 but without .complete is not a valid checkpoint."""

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        pt = ckpt_dir / "checkpoint_step_00000003.pt"
        sha = pt.with_suffix(".pt.sha256")
        pt.write_bytes(b"garbage")
        sha.write_text(f"{'0' * 64}  {pt.name}\n", encoding="utf-8")
        # No .complete marker → should not be found.
        assert EfficientADDetector._find_latest_checkpoint(ckpt_dir) is None


# ---------------------------------------------------------------------------
# Phase 1.1 verification: Student and Autoencoder parameter restoration
# ---------------------------------------------------------------------------


class TestParameterRestoration:
    """Verify Student and Autoencoder parameters are correctly restored."""

    def test_student_params_restored(self, ckpt_setup: CheckpointSetup) -> None:
        """Student state_dict matches after checkpoint reload."""

        fresh = _fresh_detector(ckpt_setup.config)
        optimizer = _make_optimizer(fresh)
        fresh.load_checkpoint(
            ckpt_setup.checkpoint_path,
            optimizer=optimizer,
            compatibility_info=ckpt_setup.compatibility_info,
        )
        loaded_params = {
            n: p.clone()
            for n, p in fresh.student.named_parameters()  # type: ignore[union-attr]
        }

        assert ckpt_setup.student_state_dict.keys() == loaded_params.keys()
        for name in ckpt_setup.student_state_dict:
            assert torch.equal(ckpt_setup.student_state_dict[name], loaded_params[name]), (
                f"Student parameter {name!r} differs after checkpoint reload."
            )

    def test_autoencoder_params_restored(self, ckpt_setup: CheckpointSetup) -> None:
        """Autoencoder state_dict matches after checkpoint reload."""

        fresh = _fresh_detector(ckpt_setup.config)
        optimizer = _make_optimizer(fresh)
        fresh.load_checkpoint(
            ckpt_setup.checkpoint_path,
            optimizer=optimizer,
            compatibility_info=ckpt_setup.compatibility_info,
        )
        loaded_params = {
            n: p.clone()
            for n, p in fresh.autoencoder.named_parameters()  # type: ignore[union-attr]
        }

        assert ckpt_setup.autoencoder_state_dict.keys() == loaded_params.keys()
        for name in ckpt_setup.autoencoder_state_dict:
            assert torch.equal(ckpt_setup.autoencoder_state_dict[name], loaded_params[name]), (
                f"Autoencoder parameter {name!r} differs after checkpoint reload."
            )


# ---------------------------------------------------------------------------
# Phase 1.1 verification: optimizer state restoration
# ---------------------------------------------------------------------------


class TestOptimizerStateRestoration:
    """Verify optimizer state is correctly restored from checkpoint."""

    def test_optimizer_state_restored(self, ckpt_setup: CheckpointSetup) -> None:
        """Optimizer state_dict contains param_groups and state after reload."""

        fresh = _fresh_detector(ckpt_setup.config)
        optimizer = _make_optimizer(fresh)
        fresh.load_checkpoint(
            ckpt_setup.checkpoint_path,
            optimizer=optimizer,
            compatibility_info=ckpt_setup.compatibility_info,
        )
        opt_state = optimizer.state_dict()
        assert "param_groups" in opt_state
        assert "state" in opt_state
        # Adam stores exp_avg and exp_avg_sq for each parameter after at least one step.
        assert len(opt_state["state"]) > 0


# ---------------------------------------------------------------------------
# Phase 1.1 verification: step restoration
# ---------------------------------------------------------------------------


class TestStepRestoration:
    """Verify the training step is correctly restored."""

    def test_step_correctly_restored(self, ckpt_setup: CheckpointSetup) -> None:
        """Restored step equals the step at which the checkpoint was saved."""

        fresh = _fresh_detector(ckpt_setup.config)
        optimizer = _make_optimizer(fresh)
        step, _ = fresh.load_checkpoint(
            ckpt_setup.checkpoint_path,
            optimizer=optimizer,
            compatibility_info=ckpt_setup.compatibility_info,
        )
        assert step == ckpt_setup.step


# ---------------------------------------------------------------------------
# Phase 1.1 verification: RNG and normalizer state fields
# ---------------------------------------------------------------------------


class TestRngAndNormalizerState:
    """Verify RNG state and normalizer accumulator state fields exist in checkpoint."""

    def test_rng_state_fields_present(self, ckpt_setup: CheckpointSetup) -> None:
        """Checkpoint contains Python, NumPy, and PyTorch RNG state fields."""

        ckpt = torch.load(ckpt_setup.checkpoint_path, map_location="cpu", weights_only=False)
        rng = ckpt["rng_state"]
        assert "python" in rng
        assert "numpy" in rng
        assert "torch_cpu" in rng
        assert "torch_cuda" in rng

    def test_normalizer_state_present(self, ckpt_setup: CheckpointSetup) -> None:
        """Checkpoint model_state contains a normalizer entry."""

        ckpt = torch.load(ckpt_setup.checkpoint_path, map_location="cpu", weights_only=False)
        model_state = ckpt["model_state"]
        assert "normalizer" in model_state

    def test_normalizer_accumulator_state_present(self, ckpt_setup: CheckpointSetup) -> None:
        """Checkpoint model_state contains a normalizer_accumulator entry."""

        ckpt = torch.load(ckpt_setup.checkpoint_path, map_location="cpu", weights_only=False)
        model_state = ckpt["model_state"]
        assert "normalizer_accumulator" in model_state
        acc_state = model_state["normalizer_accumulator"]
        assert "count" in acc_state
        assert "mean" in acc_state
        assert "m2" in acc_state
        assert "num_channels" in acc_state

    def test_normalizer_accumulator_restored(self, ckpt_setup: CheckpointSetup) -> None:
        """Normalizer accumulator state is correctly restored from checkpoint."""

        fresh = _fresh_detector(ckpt_setup.config)
        optimizer = _make_optimizer(fresh)
        step, accumulator = fresh.load_checkpoint(
            ckpt_setup.checkpoint_path,
            optimizer=optimizer,
            compatibility_info=ckpt_setup.compatibility_info,
        )
        assert step == ckpt_setup.step
        assert accumulator.count > 0
        assert accumulator.num_channels == ckpt_setup.config.teacher_channels
        # Verify the accumulator can produce a fitted normalizer.
        normalizer = accumulator.to_normalizer()
        assert normalizer.is_fit

    def test_rng_state_restored(self, ckpt_setup: CheckpointSetup) -> None:
        """RNG state is restored after load_checkpoint (no exception)."""

        # Perturb RNG state to ensure it's different.
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)

        fresh = _fresh_detector(ckpt_setup.config)
        optimizer = _make_optimizer(fresh)
        fresh.load_checkpoint(
            ckpt_setup.checkpoint_path,
            optimizer=optimizer,
            compatibility_info=ckpt_setup.compatibility_info,
        )
        # If RNG state was restored, the next random draw should match the
        # state at checkpoint time (not seed 999).
        val_after_restore = torch.randint(0, 10000, (1,)).item()
        # Restore again to get the same state.
        fresh2 = _fresh_detector(ckpt_setup.config)
        opt2 = _make_optimizer(fresh2)
        fresh2.load_checkpoint(
            ckpt_setup.checkpoint_path,
            optimizer=opt2,
            compatibility_info=ckpt_setup.compatibility_info,
        )
        val_after_second_restore = torch.randint(0, 10000, (1,)).item()
        assert val_after_restore == val_after_second_restore


# ---------------------------------------------------------------------------
# Phase 1.1 verification: incompatible checkpoint rejection
# ---------------------------------------------------------------------------


class TestCompatibilityChecks:
    """Verify incompatible checkpoints are rejected with clear errors."""

    def test_incompatible_config_fingerprint_rejected(
        self,
        ckpt_setup: CheckpointSetup,
    ) -> None:
        """Checkpoint with different config fingerprint is rejected."""

        different_config = _fixture_config(num_steps=6, teacher_channels=32)
        fresh = _fresh_detector(different_config)
        optimizer = _make_optimizer(fresh)
        with pytest.raises(CheckpointError, match="fingerprint"):
            fresh.load_checkpoint(
                ckpt_setup.checkpoint_path,
                optimizer=optimizer,
                compatibility_info=ckpt_setup.compatibility_info,
            )

    def test_incompatible_category_rejected(
        self,
        ckpt_setup: CheckpointSetup,
    ) -> None:
        """Checkpoint with different category is rejected."""

        fresh = _fresh_detector(ckpt_setup.config)
        optimizer = _make_optimizer(fresh)
        wrong_info = {**ckpt_setup.compatibility_info, "category": "bottle"}
        with pytest.raises(CheckpointError, match="category"):
            fresh.load_checkpoint(
                ckpt_setup.checkpoint_path,
                optimizer=optimizer,
                compatibility_info=wrong_info,
            )

    def test_incompatible_seed_rejected(
        self,
        ckpt_setup: CheckpointSetup,
    ) -> None:
        """Checkpoint with different seed is rejected."""

        fresh = _fresh_detector(ckpt_setup.config)
        optimizer = _make_optimizer(fresh)
        wrong_info = {**ckpt_setup.compatibility_info, "seed": 0}
        with pytest.raises(CheckpointError, match="seed"):
            fresh.load_checkpoint(
                ckpt_setup.checkpoint_path,
                optimizer=optimizer,
                compatibility_info=wrong_info,
            )

    def test_incompatible_split_fingerprint_rejected(
        self,
        ckpt_setup: CheckpointSetup,
    ) -> None:
        """Checkpoint with different split_fingerprint is rejected."""

        fresh = _fresh_detector(ckpt_setup.config)
        optimizer = _make_optimizer(fresh)
        wrong_info = {**ckpt_setup.compatibility_info, "split_fingerprint": "split-B"}
        with pytest.raises(CheckpointError, match="split_fingerprint"):
            fresh.load_checkpoint(
                ckpt_setup.checkpoint_path,
                optimizer=optimizer,
                compatibility_info=wrong_info,
            )

    def test_incompatible_teacher_hash_rejected(
        self,
        ckpt_setup: CheckpointSetup,
        tmp_path: Path,
    ) -> None:
        """Checkpoint with different teacher hash is rejected."""

        # Create a detector and modify teacher weights to change the hash.
        config = _fixture_config(num_steps=6)
        detector = EfficientADDetector(config, device="cpu")
        with torch.no_grad():
            for param in detector.teacher.parameters():  # type: ignore[union-attr]
                param.add_(0.001)
                break

        # Save a checkpoint from this modified-teacher detector.
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "checkpoint_step_00000006.pt"
        optimizer = _make_optimizer(detector)
        acc = NormalizerAccumulator(config.teacher_channels)
        detector.save_checkpoint(
            ckpt_path,
            step=6,
            optimizer=optimizer,
            normalizer_accumulator=acc,
            compatibility_info=ckpt_setup.compatibility_info,
        )

        # Try to load with a fresh detector (original teacher weights).
        fresh = _fresh_detector(ckpt_setup.config)
        opt = _make_optimizer(fresh)
        with pytest.raises(CheckpointError, match="teacher"):
            fresh.load_checkpoint(
                ckpt_path,
                optimizer=opt,
                compatibility_info=ckpt_setup.compatibility_info,
            )

    def test_incompatible_version_rejected(
        self,
        ckpt_setup: CheckpointSetup,
        tmp_path: Path,
    ) -> None:
        """Checkpoint with wrong version is rejected."""

        ckpt = torch.load(ckpt_setup.checkpoint_path, map_location="cpu", weights_only=False)
        ckpt["checkpoint_version"] = 99

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        bad_path = ckpt_dir / "checkpoint_step_00000006.pt"
        torch.save(ckpt, bad_path)
        file_hash = hashlib.sha256(bad_path.read_bytes()).hexdigest()
        bad_path.with_suffix(".pt.sha256").write_text(
            f"{file_hash}  {bad_path.name}\n", encoding="utf-8"
        )
        bad_path.with_suffix(".pt.complete").write_text(f"{file_hash}\n6\n", encoding="utf-8")

        fresh = _fresh_detector(ckpt_setup.config)
        opt = _make_optimizer(fresh)
        with pytest.raises(CheckpointError, match="version"):
            fresh.load_checkpoint(
                bad_path,
                optimizer=opt,
                compatibility_info=ckpt_setup.compatibility_info,
            )

    def test_hash_sidecar_missing_rejected(
        self,
        ckpt_setup: CheckpointSetup,
        tmp_path: Path,
    ) -> None:
        """Checkpoint without hash sidecar is rejected."""

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        dst = ckpt_dir / ckpt_setup.checkpoint_path.name
        shutil.copy2(ckpt_setup.checkpoint_path, dst)
        shutil.copy2(ckpt_setup.complete_path, dst.with_suffix(".pt.complete"))
        # No .sha256 sidecar copied.
        fresh = _fresh_detector(ckpt_setup.config)
        optimizer = _make_optimizer(fresh)
        with pytest.raises(CheckpointError, match="sidecar missing"):
            fresh.load_checkpoint(
                dst,
                optimizer=optimizer,
                compatibility_info=ckpt_setup.compatibility_info,
            )

    def test_hash_mismatch_rejected(
        self,
        ckpt_setup: CheckpointSetup,
        tmp_path: Path,
    ) -> None:
        """Tampered hash sidecar is rejected."""

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        dst_pt = ckpt_dir / ckpt_setup.checkpoint_path.name
        dst_sha = dst_pt.with_suffix(".pt.sha256")
        dst_complete = dst_pt.with_suffix(".pt.complete")
        shutil.copy2(ckpt_setup.checkpoint_path, dst_pt)
        shutil.copy2(ckpt_setup.complete_path, dst_complete)
        dst_sha.write_text("0" * 64 + "  " + dst_pt.name + "\n", encoding="utf-8")
        fresh = _fresh_detector(ckpt_setup.config)
        optimizer = _make_optimizer(fresh)
        with pytest.raises(CheckpointError, match="hash mismatch"):
            fresh.load_checkpoint(
                dst_pt,
                optimizer=optimizer,
                compatibility_info=ckpt_setup.compatibility_info,
            )

    def test_missing_complete_marker_rejected(
        self,
        ckpt_setup: CheckpointSetup,
        tmp_path: Path,
    ) -> None:
        """Checkpoint without .complete marker is rejected."""

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        dst_pt = ckpt_dir / ckpt_setup.checkpoint_path.name
        dst_sha = dst_pt.with_suffix(".pt.sha256")
        shutil.copy2(ckpt_setup.checkpoint_path, dst_pt)
        shutil.copy2(ckpt_setup.hash_path, dst_sha)
        # No .complete marker copied.
        fresh = _fresh_detector(ckpt_setup.config)
        optimizer = _make_optimizer(fresh)
        with pytest.raises(CheckpointError, match="complete marker missing"):
            fresh.load_checkpoint(
                dst_pt,
                optimizer=optimizer,
                compatibility_info=ckpt_setup.compatibility_info,
            )


# ---------------------------------------------------------------------------
# Phase 1.1 verification: atomic save preserves old checkpoint
# ---------------------------------------------------------------------------


class TestAtomicSave:
    """Verify atomic write failure preserves the existing checkpoint."""

    def test_atomic_save_failure_preserves_old(
        self,
        ckpt_setup: CheckpointSetup,
        tmp_path: Path,
    ) -> None:
        """A failed save_checkpoint does not corrupt the existing checkpoint."""

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        old_pt = ckpt_dir / "checkpoint_step_00000006.pt"
        old_sha = old_pt.with_suffix(".pt.sha256")
        old_complete = old_pt.with_suffix(".pt.complete")
        shutil.copy2(ckpt_setup.checkpoint_path, old_pt)
        shutil.copy2(ckpt_setup.hash_path, old_sha)
        shutil.copy2(ckpt_setup.complete_path, old_complete)

        detector = _fresh_detector(ckpt_setup.config)
        optimizer = _make_optimizer(detector)
        acc = NormalizerAccumulator(ckpt_setup.config.teacher_channels)

        failing_path = ckpt_dir / "checkpoint_step_00000012.pt"
        with (
            patch(
                "shiftad.methods.efficientad.torch.save",
                side_effect=RuntimeError("simulated disk failure"),
            ),
            pytest.raises(RuntimeError, match="simulated disk failure"),
        ):
            detector.save_checkpoint(
                failing_path,
                step=12,
                optimizer=optimizer,
                normalizer_accumulator=acc,
                compatibility_info=ckpt_setup.compatibility_info,
            )

        # Old checkpoint is still intact and loadable.
        assert old_pt.is_file()
        assert old_sha.is_file()
        assert old_complete.is_file()
        fresh = _fresh_detector(ckpt_setup.config)
        opt = _make_optimizer(fresh)
        step, _ = fresh.load_checkpoint(
            old_pt,
            optimizer=opt,
            compatibility_info=ckpt_setup.compatibility_info,
        )
        assert step == 6

        # No temp files left behind.
        temp_files = list(ckpt_dir.glob(".ckpt_tmp_*"))
        assert len(temp_files) == 0

        # Failed checkpoint file does not exist.
        assert not failing_path.is_file()


# ---------------------------------------------------------------------------
# Phase 1.2: hash sidecar write failure preserves old checkpoint
# ---------------------------------------------------------------------------


class TestHashSidecarWriteFailure:
    """Verify hash sidecar write failure preserves old checkpoint.

    Training steps consumed: 0 (reuses module fixture checkpoint).
    """

    def test_hash_sidecar_write_failure_preserves_old_checkpoint(
        self,
        ckpt_setup: CheckpointSetup,
        tmp_path: Path,
    ) -> None:
        """Hash sidecar write failure preserves old checkpoint integrity.

        Simulates a failure during the .sha256 temp file replace step
        (after torch.save succeeded).  Verifies:
        - ``save_checkpoint()`` raises an exception.
        - Old checkpoint files remain intact.
        - Old checkpoint is still loadable.
        - New checkpoint is NOT recognised as a valid candidate.
        - ``_find_latest_checkpoint`` returns the old checkpoint.
        - No temp files are left behind.
        """

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        # Copy old valid checkpoint (step 6).
        old_pt = ckpt_dir / "checkpoint_step_00000006.pt"
        old_sha = old_pt.with_suffix(".pt.sha256")
        old_complete = old_pt.with_suffix(".pt.complete")
        shutil.copy2(ckpt_setup.checkpoint_path, old_pt)
        shutil.copy2(ckpt_setup.hash_path, old_sha)
        shutil.copy2(ckpt_setup.complete_path, old_complete)

        detector = _fresh_detector(ckpt_setup.config)
        optimizer = _make_optimizer(detector)
        acc = NormalizerAccumulator(ckpt_setup.config.teacher_channels)

        new_path = ckpt_dir / "checkpoint_step_00000012.pt"

        _original_replace = Path.replace

        def _failing_replace(self_path: Path, target: Path) -> None:
            if self_path.suffix == ".sha256":
                raise OSError("simulated hash sidecar write failure")
            return _original_replace(self_path, target)

        with (
            patch.object(Path, "replace", _failing_replace),
            pytest.raises(OSError, match="simulated hash sidecar write failure"),
        ):
            detector.save_checkpoint(
                new_path,
                step=12,
                optimizer=optimizer,
                normalizer_accumulator=acc,
                compatibility_info=ckpt_setup.compatibility_info,
            )

        # Old checkpoint is still intact.
        assert old_pt.is_file(), "Old checkpoint .pt should still exist."
        assert old_sha.is_file(), "Old checkpoint .sha256 should still exist."
        assert old_complete.is_file(), "Old checkpoint .complete should still exist."

        # Old checkpoint can still be loaded.
        fresh = _fresh_detector(ckpt_setup.config)
        opt = _make_optimizer(fresh)
        step, _ = fresh.load_checkpoint(
            old_pt,
            optimizer=opt,
            compatibility_info=ckpt_setup.compatibility_info,
        )
        assert step == 6, "Old checkpoint should load correctly (step 6)."

        # New checkpoint is NOT recognised as a valid candidate.
        candidates = EfficientADDetector._list_checkpoint_candidates(ckpt_dir)
        candidate_steps = [s for s, _ in candidates]
        assert 12 not in candidate_steps, (
            "Failed new checkpoint (step 12) must not appear in candidate list."
        )

        # _find_latest_checkpoint returns the old checkpoint.
        latest = EfficientADDetector._find_latest_checkpoint(ckpt_dir)
        assert latest is not None, "Should find at least one valid checkpoint."
        assert "00000006" in latest.name, (
            "Latest checkpoint should be the old step 6, not the failed step 12."
        )

        # No temp files left behind.
        temp_files = list(ckpt_dir.glob(".ckpt_tmp_*"))
        assert len(temp_files) == 0, f"Temp files left behind: {[f.name for f in temp_files]}"


# ---------------------------------------------------------------------------
# Phase 1.2: complete marker creation failure preserves old checkpoint
# ---------------------------------------------------------------------------


class TestCompleteMarkerCreationFailure:
    """Verify .complete marker creation failure preserves old checkpoint.

    Training steps consumed: 0 (reuses module fixture checkpoint).
    """

    def test_complete_marker_creation_failure_preserves_old_checkpoint(
        self,
        ckpt_setup: CheckpointSetup,
        tmp_path: Path,
    ) -> None:
        """.complete marker creation failure preserves old checkpoint integrity.

        Simulates a failure during the .complete temp file replace step
        (after .pt and .sha256 were successfully written).  Verifies:
        - ``save_checkpoint()`` raises an exception.
        - Old checkpoint files remain intact.
        - Old checkpoint is still loadable.
        - New checkpoint is NOT recognised as a valid candidate.
        - ``_find_latest_checkpoint`` returns the old checkpoint.
        - ``_auto_resume_from_latest`` selects the old checkpoint.
        - No temp files are left behind.
        """

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        # Copy old valid checkpoint (step 6).
        old_pt = ckpt_dir / "checkpoint_step_00000006.pt"
        old_sha = old_pt.with_suffix(".pt.sha256")
        old_complete = old_pt.with_suffix(".pt.complete")
        shutil.copy2(ckpt_setup.checkpoint_path, old_pt)
        shutil.copy2(ckpt_setup.hash_path, old_sha)
        shutil.copy2(ckpt_setup.complete_path, old_complete)

        detector = _fresh_detector(ckpt_setup.config)
        optimizer = _make_optimizer(detector)
        acc = NormalizerAccumulator(ckpt_setup.config.teacher_channels)

        new_path = ckpt_dir / "checkpoint_step_00000012.pt"

        _original_replace = Path.replace

        def _failing_replace(self_path: Path, target: Path) -> None:
            if self_path.suffix == ".complete":
                raise OSError("simulated complete marker creation failure")
            return _original_replace(self_path, target)

        with (
            patch.object(Path, "replace", _failing_replace),
            pytest.raises(OSError, match="simulated complete marker creation failure"),
        ):
            detector.save_checkpoint(
                new_path,
                step=12,
                optimizer=optimizer,
                normalizer_accumulator=acc,
                compatibility_info=ckpt_setup.compatibility_info,
            )

        # Old checkpoint is still intact.
        assert old_pt.is_file(), "Old checkpoint .pt should still exist."
        assert old_sha.is_file(), "Old checkpoint .sha256 should still exist."
        assert old_complete.is_file(), "Old checkpoint .complete should still exist."

        # Old checkpoint can still be loaded.
        fresh = _fresh_detector(ckpt_setup.config)
        opt = _make_optimizer(fresh)
        step, _ = fresh.load_checkpoint(
            old_pt,
            optimizer=opt,
            compatibility_info=ckpt_setup.compatibility_info,
        )
        assert step == 6, "Old checkpoint should load correctly (step 6)."

        # New checkpoint is NOT recognised as a valid candidate.
        candidates = EfficientADDetector._list_checkpoint_candidates(ckpt_dir)
        candidate_steps = [s for s, _ in candidates]
        assert 12 not in candidate_steps, (
            "Failed new checkpoint (step 12) must not appear in candidate list."
        )

        # _find_latest_checkpoint returns the old checkpoint.
        latest = EfficientADDetector._find_latest_checkpoint(ckpt_dir)
        assert latest is not None, "Should find at least one valid checkpoint."
        assert "00000006" in latest.name, (
            "Latest checkpoint should be the old step 6, not the failed step 12."
        )

        # auto_resume uses the old checkpoint.
        detector2 = _fresh_detector(ckpt_setup.config)
        opt2 = _make_optimizer(detector2)
        start_step, _acc, res_path = detector2._auto_resume_from_latest(
            ckpt_dir,
            optimizer=opt2,
            compatibility_info=ckpt_setup.compatibility_info,
        )
        assert start_step == 6, f"auto_resume should use old checkpoint (step 6), got {start_step}."
        assert res_path is not None and "00000006" in res_path.name, (
            "auto_resume should return the old checkpoint path."
        )

        # No temp files left behind.
        temp_files = list(ckpt_dir.glob(".ckpt_tmp_*"))
        assert len(temp_files) == 0, f"Temp files left behind: {[f.name for f in temp_files]}"


# ---------------------------------------------------------------------------
# Phase 1.1 verification: resume does not repeat completed steps
# ---------------------------------------------------------------------------


class TestResumeNoStepRepetition:
    """Verify resume continues from the saved step without repetition.

    Training steps consumed: 12 (initial run) + 6 (resumed run) = 18.
    """

    def test_resume_continues_from_saved_step(self, tmp_path: Path) -> None:
        """Resumed training starts from the checkpoint step, not from 0."""

        ckpt_dir = tmp_path / "checkpoints"
        config = _fixture_config(num_steps=12)
        compat = {"category": "widget", "seed": 42, "split_fingerprint": "test_split_fp_a"}

        # First run: train 12 steps, save checkpoints at step 6 and 12.
        detector1 = EfficientADDetector(config, device="cpu")
        detector1.fit(
            _train_loader(),
            checkpoint_dir=ckpt_dir,
            checkpoint_interval=6,
            auto_resume=False,
            compatibility_info=compat,
        )
        assert detector1.build_metadata["resumed_from_step"] == 0

        ckpt_at_6 = ckpt_dir / "checkpoint_step_00000006.pt"
        assert ckpt_at_6.is_file()

        # Second run: resume from step 6, train to step 12.
        detector2 = EfficientADDetector(config, device="cpu")
        detector2.fit(
            _train_loader(),
            checkpoint_dir=ckpt_dir,
            checkpoint_interval=6,
            resume_from=ckpt_at_6,
            auto_resume=False,
            compatibility_info=compat,
        )
        assert detector2.build_metadata["resumed_from_step"] == 6
        assert detector2.build_metadata["train_steps"] == 12

    def test_resume_skips_completed_training(self, tmp_path: Path) -> None:
        """Resume from final step skips the training loop entirely."""

        ckpt_dir = tmp_path / "checkpoints"
        config = _fixture_config(num_steps=6)
        compat = {"category": "widget", "seed": 42, "split_fingerprint": "test_split_fp_a"}

        detector1 = EfficientADDetector(config, device="cpu")
        detector1.fit(
            _train_loader(),
            checkpoint_dir=ckpt_dir,
            checkpoint_interval=6,
            auto_resume=False,
            compatibility_info=compat,
        )

        ckpt_at_6 = ckpt_dir / "checkpoint_step_00000006.pt"
        detector2 = EfficientADDetector(config, device="cpu")
        detector2.fit(
            _train_loader(),
            resume_from=ckpt_at_6,
            auto_resume=False,
            compatibility_info=compat,
        )
        assert detector2.build_metadata["resumed_from_step"] == 6
        assert detector2.is_fitted is True


# ---------------------------------------------------------------------------
# Phase 1.1 verification: step count verification after resume
# ---------------------------------------------------------------------------


class TestStepCountVerification:
    """Verify optimizer.step() is called exactly (target - resumed) times.

    Training steps consumed: 6 (initial) + 3 (resumed) = 9.
    """

    def test_optimizer_step_count_after_resume(self, tmp_path: Path) -> None:
        """Verify optimizer.step() call count matches expected resume delta."""

        ckpt_dir = tmp_path / "checkpoints"
        config = _fixture_config(num_steps=6)
        compat = {"category": "widget", "seed": 42, "split_fingerprint": "test_split_fp_a"}

        # First run: train 6 steps, save at step 3 and 6.
        detector1 = EfficientADDetector(config, device="cpu")
        detector1.fit(
            _train_loader(),
            checkpoint_dir=ckpt_dir,
            checkpoint_interval=3,
            auto_resume=False,
            compatibility_info=compat,
        )

        ckpt_at_3 = ckpt_dir / "checkpoint_step_00000003.pt"
        assert ckpt_at_3.is_file()

        # Second run: resume from step 3, count optimizer.step() calls.
        step_counter: dict[str, int] = {"count": 0}
        original_step = torch.optim.Adam.step

        def counting_step(self_param: Any, closure: Any = None) -> Any:
            step_counter["count"] += 1
            return original_step(self_param, closure)

        detector2 = EfficientADDetector(config, device="cpu")
        with patch.object(torch.optim.Adam, "step", counting_step):
            detector2.fit(
                _train_loader(),
                resume_from=ckpt_at_3,
                auto_resume=False,
                compatibility_info=compat,
            )

        # Resumed from step 3, target is 6 → 3 new optimizer.step() calls.
        assert step_counter["count"] == 3, (
            f"Expected 3 optimizer.step() calls after resume from step 3 to 6, "
            f"got {step_counter['count']}."
        )
        assert detector2.build_metadata["resumed_from_step"] == 3
        assert detector2.build_metadata["train_steps"] == 6


# ---------------------------------------------------------------------------
# Phase 1.1 verification: auto_resume finds latest checkpoint
# ---------------------------------------------------------------------------


class TestAutoResume:
    """Verify auto_resume finds and uses the latest checkpoint.

    Training steps consumed: 12 (single run with interval=6) + 6 (no checkpoint) = 18.
    """

    def test_auto_resume_finds_latest(self, tmp_path: Path) -> None:
        """auto_resume locates the checkpoint with the highest step."""

        ckpt_dir = tmp_path / "checkpoints"
        config = _fixture_config(num_steps=12)
        compat = {"category": "widget", "seed": 42, "split_fingerprint": "test_split_fp_a"}

        detector1 = EfficientADDetector(config, device="cpu")
        detector1.fit(
            _train_loader(),
            checkpoint_dir=ckpt_dir,
            checkpoint_interval=6,
            auto_resume=False,
            compatibility_info=compat,
        )

        latest = EfficientADDetector._find_latest_checkpoint(ckpt_dir)
        assert latest is not None
        assert "00000012" in latest.name

        detector2 = EfficientADDetector(config, device="cpu")
        detector2.fit(
            _train_loader(),
            checkpoint_dir=ckpt_dir,
            checkpoint_interval=6,
            auto_resume=True,
            compatibility_info=compat,
        )
        assert detector2.build_metadata["resumed_from_step"] == 12

    def test_auto_resume_no_checkpoint_starts_from_zero(
        self,
        tmp_path: Path,
    ) -> None:
        """auto_resume with no existing checkpoints starts from step 0."""

        ckpt_dir = tmp_path / "checkpoints"
        config = _fixture_config(num_steps=6)
        compat = {"category": "widget", "seed": 42, "split_fingerprint": "test_split_fp_a"}

        detector = EfficientADDetector(config, device="cpu")
        detector.fit(
            _train_loader(),
            checkpoint_dir=ckpt_dir,
            checkpoint_interval=6,
            auto_resume=True,
            compatibility_info=compat,
        )
        assert detector.build_metadata["resumed_from_step"] == 0

    def test_auto_resume_fallback_to_previous(
        self,
        ckpt_setup: CheckpointSetup,
        tmp_path: Path,
    ) -> None:
        """Auto-resume falls back when the latest checkpoint is corrupted."""

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        # Step 6: valid checkpoint (copy from module fixture).
        s6_pt = ckpt_dir / "checkpoint_step_00000006.pt"
        s6_sha = s6_pt.with_suffix(".pt.sha256")
        s6_complete = s6_pt.with_suffix(".pt.complete")
        shutil.copy2(ckpt_setup.checkpoint_path, s6_pt)
        shutil.copy2(ckpt_setup.hash_path, s6_sha)
        shutil.copy2(ckpt_setup.complete_path, s6_complete)

        # Step 12: corrupted hash (copy .pt but write wrong .sha256).
        s12_pt = ckpt_dir / "checkpoint_step_00000012.pt"
        s12_sha = s12_pt.with_suffix(".pt.sha256")
        s12_complete = s12_pt.with_suffix(".pt.complete")
        shutil.copy2(ckpt_setup.checkpoint_path, s12_pt)
        s12_sha.write_text("0" * 64 + "  " + s12_pt.name + "\n", encoding="utf-8")
        shutil.copy2(ckpt_setup.complete_path, s12_complete)

        # Auto-resume should skip step 12 (hash mismatch) and use step 6.
        detector = _fresh_detector(ckpt_setup.config)
        optimizer = _make_optimizer(detector)
        start_step, _acc, path = detector._auto_resume_from_latest(
            ckpt_dir,
            optimizer=optimizer,
            compatibility_info=ckpt_setup.compatibility_info,
        )
        assert start_step == 6
        assert path is not None
        assert "00000006" in path.name


# ---------------------------------------------------------------------------
# Phase 1.1 verification: resume_from priority and corrupt rejection
# ---------------------------------------------------------------------------


class TestResumeFromPriority:
    """Verify resume_from priority over auto_resume and corrupt rejection."""

    def test_resume_from_takes_priority_over_auto_resume(
        self,
        ckpt_setup: CheckpointSetup,
        tmp_path: Path,
    ) -> None:
        """resume_from takes priority over auto_resume."""

        config = _fixture_config(num_steps=6)
        detector = EfficientADDetector(config, device="cpu")

        with patch.object(
            EfficientADDetector,
            "_auto_resume_from_latest",
            side_effect=AssertionError("auto_resume should not be called when resume_from is set"),
        ):
            detector.fit(
                _train_loader(),
                checkpoint_dir=tmp_path / "checkpoints",
                resume_from=ckpt_setup.checkpoint_path,
                auto_resume=True,
                compatibility_info=ckpt_setup.compatibility_info,
            )
        assert detector.build_metadata["resumed_from_step"] == 6

    def test_resume_from_corrupt_fails_immediately(
        self,
        ckpt_setup: CheckpointSetup,
        tmp_path: Path,
    ) -> None:
        """Explicit resume_from with corrupted checkpoint fails immediately."""

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        corrupt_pt = ckpt_dir / "checkpoint_step_00000006.pt"
        corrupt_sha = corrupt_pt.with_suffix(".pt.sha256")
        corrupt_complete = corrupt_pt.with_suffix(".pt.complete")
        shutil.copy2(ckpt_setup.checkpoint_path, corrupt_pt)
        corrupt_sha.write_text("0" * 64 + "  " + corrupt_pt.name + "\n", encoding="utf-8")
        shutil.copy2(ckpt_setup.complete_path, corrupt_complete)

        config = _fixture_config(num_steps=6)
        detector = EfficientADDetector(config, device="cpu")
        with pytest.raises(CheckpointError, match="hash mismatch"):
            detector.fit(
                _train_loader(),
                resume_from=corrupt_pt,
                auto_resume=False,
                compatibility_info=ckpt_setup.compatibility_info,
            )


# ---------------------------------------------------------------------------
# Phase 1.1 verification: fixture Teacher deterministic initialization
# ---------------------------------------------------------------------------


class TestFixtureTeacherInit:
    """Verify fixture Teacher deterministic initialization and RNG isolation."""

    def test_fixture_teacher_deterministic(self) -> None:
        """Two fixture Teachers produce identical weight hashes."""

        config = _fixture_config(num_steps=1)
        detector1 = EfficientADDetector(config, device="cpu")
        hash1 = detector1._compute_teacher_state_hash()

        detector2 = EfficientADDetector(config, device="cpu")
        hash2 = detector2._compute_teacher_state_hash()

        assert hash1 == hash2, "Fixture teacher hashes should be identical."

    def test_fixture_teacher_rng_isolated(self) -> None:
        """Fixture Teacher creation preserves global RNG state.

        ``_init_networks`` saves and restores the global RNG state around
        fixture Teacher creation and ``_deterministic_init``.  This test
        verifies that a Student created *after* the Teacher block has the
        same weights as a Student created *without* any preceding Teacher
        creation, proving the Teacher block did not perturb the RNG.
        Student and Autoencoder use normal seed logic and are expected to
        consume global RNG state during their own initialisation.
        """

        from shiftad.models.efficientad.student import FixtureStudentNetwork

        # Seed, then create a full detector (teacher + student + AE).
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        config = _fixture_config(num_steps=1)
        detector = EfficientADDetector(config, device="cpu")
        detector_first_weight = next(iter(detector.student.parameters())).clone()  # type: ignore[union-attr]

        # Same seed, then create only a Student (no Teacher before it).
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        standalone_student = FixtureStudentNetwork(
            teacher_channels=config.teacher_channels,
            ae_channels=config.ae_channels,
        )
        standalone_first_weight = next(iter(standalone_student.parameters())).clone()

        assert torch.equal(detector_first_weight, standalone_first_weight), (
            "Student weights differ after Teacher creation, indicating the "
            "fixture Teacher block perturbed the global RNG state."
        )

    def test_fixture_teacher_not_degenerate(self) -> None:
        """Fixture Teacher feature channels are not all identical."""

        config = _fixture_config(num_steps=1)
        detector = EfficientADDetector(config, device="cpu")
        sample = torch.randn(1, 3, 32, 32)
        with torch.inference_mode():
            output = detector.teacher(sample)  # type: ignore[no-untyped-call]
        channel_means = output.mean(dim=(0, 2, 3))
        # Not all channel means should be identical (would indicate degeneration).
        assert not torch.allclose(channel_means, channel_means[0].expand_as(channel_means)), (
            "Teacher feature channels appear degenerated (all identical)."
        )


# ---------------------------------------------------------------------------
# Phase 1.1 verification: checkpoint cleanup (max_to_keep)
# ---------------------------------------------------------------------------


class TestCheckpointCleanup:
    """Verify old checkpoints are cleaned up when max_to_keep is exceeded."""

    def test_max_to_keep_removes_old_checkpoints(
        self,
        ckpt_setup: CheckpointSetup,
        tmp_path: Path,
    ) -> None:
        """Only max_to_keep checkpoints are retained after cleanup."""

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        for step in (1, 2, 3):
            pt = ckpt_dir / f"checkpoint_step_{step:08d}.pt"
            sha = pt.with_suffix(".pt.sha256")
            complete = pt.with_suffix(".pt.complete")
            pt.write_bytes(b"fake")
            sha.write_text(f"{'0' * 64}  {pt.name}\n", encoding="utf-8")
            complete.write_text(f"{'0' * 64}\n{step}\n", encoding="utf-8")

        EfficientADDetector._cleanup_old_checkpoints(ckpt_dir, max_to_keep=1)
        remaining_pt = sorted(ckpt_dir.glob("checkpoint_step_*.pt"))
        remaining_sha = sorted(ckpt_dir.glob("checkpoint_step_*.pt.sha256"))
        remaining_complete = sorted(ckpt_dir.glob("checkpoint_step_*.pt.complete"))
        assert len(remaining_pt) == 1
        assert len(remaining_sha) == 1
        assert len(remaining_complete) == 1
        assert "00000003" in remaining_pt[0].name

    def test_cleanup_deletes_complete_triplet(self, tmp_path: Path) -> None:
        """Cleanup deletes .pt, .sha256, and .complete for each evicted checkpoint."""

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        for step in (1, 2, 3):
            pt = ckpt_dir / f"checkpoint_step_{step:08d}.pt"
            sha = pt.with_suffix(".pt.sha256")
            complete = pt.with_suffix(".pt.complete")
            pt.write_bytes(b"fake")
            sha.write_text(f"{'0' * 64}  {pt.name}\n", encoding="utf-8")
            complete.write_text(f"{'0' * 64}\n{step}\n", encoding="utf-8")

        EfficientADDetector._cleanup_old_checkpoints(ckpt_dir, max_to_keep=1)

        # Steps 1 and 2 should be fully deleted (no orphan sidecars).
        for step in (1, 2):
            assert not (ckpt_dir / f"checkpoint_step_{step:08d}.pt").exists()
            assert not (ckpt_dir / f"checkpoint_step_{step:08d}.pt.sha256").exists()
            assert not (ckpt_dir / f"checkpoint_step_{step:08d}.pt.complete").exists()

        # Step 3 should remain with all three files.
        assert (ckpt_dir / "checkpoint_step_00000003.pt").exists()
        assert (ckpt_dir / "checkpoint_step_00000003.pt.sha256").exists()
        assert (ckpt_dir / "checkpoint_step_00000003.pt.complete").exists()

        # No orphan temp files.
        assert len(list(ckpt_dir.glob(".ckpt_tmp_*"))) == 0
