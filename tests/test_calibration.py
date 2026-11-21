"""Regression tests for the held-out normal calibration split protocol.

These tests verify that the calibration split is deterministic, disjoint,
and never references test data, and that both DINO Memory and PatchCore
use the same protocol for threshold calibration.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from shiftad.data.calibration_split import (
    CalibrationSplit,
    CalibrationSplitError,
    split_normal_indices,
    verify_no_overlap_with_test,
)
from shiftad.data.fixture import SyntheticFixtureDataset
from shiftad.methods.factory import load_detector
from shiftad.methods.patchcore import PatchCoreDetector
from shiftad.utils.config import CalibrationConfig, load_experiment_config
from shiftad.workflows.dino_memory import HELD_OUT_THRESHOLD_SOURCE, run_build_memory


def _patchcore_config(calibration_enabled: bool = False):
    """Load the fixture PatchCore config with optional calibration."""

    config = load_experiment_config(Path("configs/experiments/patchcore_fixture.yaml"))
    return replace(
        config,
        calibration=CalibrationConfig(
            enabled=calibration_enabled,
            ratio=0.25,
            seed=42,
            quantile=0.99,
        ),
    )


# ---------------------------------------------------------------------------
# 1. Split determinism
# ---------------------------------------------------------------------------


def test_split_determinism() -> None:
    """The same seed and size always produce the same split."""

    first = split_normal_indices(100, calibration_ratio=0.1, seed=7)
    second = split_normal_indices(100, calibration_ratio=0.1, seed=7)
    assert first.memory_indices == second.memory_indices
    assert first.calibration_indices == second.calibration_indices


# ---------------------------------------------------------------------------
# 2. Memory and calibration images don't overlap
# ---------------------------------------------------------------------------


def test_memory_and_calibration_disjoint() -> None:
    """Memory and calibration index sets are disjoint and cover all samples."""

    split = split_normal_indices(50, calibration_ratio=0.2, seed=0)
    memory_set = set(split.memory_indices)
    calibration_set = set(split.calibration_indices)
    assert not (memory_set & calibration_set)
    assert memory_set | calibration_set == set(range(50))


# ---------------------------------------------------------------------------
# 3. Test images don't enter split
# ---------------------------------------------------------------------------


def test_test_images_not_in_split() -> None:
    """verify_no_overlap_with_test passes when train and test IDs differ."""

    split = split_normal_indices(10, calibration_ratio=0.2, seed=0)
    train_ids = [f"widget/train/good/{idx:03d}" for idx in range(10)]
    test_ids = ["widget/test/good/000", "widget/test/broken/001"]
    verify_no_overlap_with_test(split, test_sample_ids=test_ids, train_sample_ids=train_ids)


def test_test_images_in_split_raises() -> None:
    """verify_no_overlap_with_test raises when a calibration ID is in test."""

    split = split_normal_indices(10, calibration_ratio=0.2, seed=0)
    train_ids = [f"widget/train/good/{i:03d}" for i in range(10)]
    leaked_id = train_ids[split.calibration_indices[0]]
    test_ids = [leaked_id]
    with pytest.raises(CalibrationSplitError, match="leakage"):
        verify_no_overlap_with_test(split, test_sample_ids=test_ids, train_sample_ids=train_ids)


# ---------------------------------------------------------------------------
# 4. Different seeds produce different splits
# ---------------------------------------------------------------------------


def test_different_seeds_produce_different_splits() -> None:
    """Different seeds yield different (but valid) calibration index sets."""

    first = split_normal_indices(100, calibration_ratio=0.1, seed=1)
    second = split_normal_indices(100, calibration_ratio=0.1, seed=2)
    assert first.calibration_indices != second.calibration_indices
    assert len(first.calibration_indices) == len(second.calibration_indices)


# ---------------------------------------------------------------------------
# 5. Calibration ratio boundaries
# ---------------------------------------------------------------------------


def test_calibration_ratio_boundaries() -> None:
    """Very small and very large ratios produce valid splits."""

    small = split_normal_indices(200, calibration_ratio=0.01, seed=0)
    assert len(small.calibration_indices) >= 1
    assert len(small.memory_indices) >= 1

    large = split_normal_indices(200, calibration_ratio=0.9, seed=0)
    assert len(large.calibration_indices) >= 1
    assert len(large.memory_indices) >= 1


# ---------------------------------------------------------------------------
# 6. Calibration subset empty or too large fails
# ---------------------------------------------------------------------------


def test_calibration_subset_too_large_fails() -> None:
    """A ratio that would leave no memory samples raises an error."""

    with pytest.raises(CalibrationSplitError, match="no memory samples"):
        split_normal_indices(3, calibration_ratio=0.99, seed=0)


def test_zero_samples_fails() -> None:
    """Zero total samples raises an error."""

    with pytest.raises(CalibrationSplitError):
        split_normal_indices(0, calibration_ratio=0.1, seed=0)


def test_invalid_ratio_fails() -> None:
    """Ratio outside (0, 1) raises an error."""

    with pytest.raises(CalibrationSplitError):
        split_normal_indices(10, calibration_ratio=0.0, seed=0)
    with pytest.raises(CalibrationSplitError):
        split_normal_indices(10, calibration_ratio=1.0, seed=0)


# ---------------------------------------------------------------------------
# 7. PatchCore uses held-out normal scores (fixture)
# ---------------------------------------------------------------------------


def test_patchcore_calibration_produces_held_out_thresholds(tmp_path: Path) -> None:
    """PatchCore build with calibration saves held-out thresholds."""

    config = _patchcore_config(calibration_enabled=True)
    config = replace(config, runtime=replace(config.runtime, output_dir=tmp_path / "out"))
    summary = run_build_memory(config)
    assert summary["calibration_enabled"] is True
    assert "calibration" in summary

    model_dir = tmp_path / "out" / "model"
    calibration_path = model_dir / "calibration.json"
    split_path = model_dir / "calibration_split.json"
    assert calibration_path.is_file()
    assert split_path.is_file()

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert calibration["threshold_source"] == HELD_OUT_THRESHOLD_SOURCE
    assert calibration["image_threshold"] is not None
    assert calibration["pixel_threshold"] is not None

    resolved = yaml.safe_load((model_dir / "experiment.resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["model"]["scoring"]["threshold_source"] == HELD_OUT_THRESHOLD_SOURCE
    assert resolved["model"]["scoring"]["image_threshold"] is not None


# ---------------------------------------------------------------------------
# 8. DINO Memory and PatchCore use same protocol
# ---------------------------------------------------------------------------


def test_dino_memory_calibration_uses_same_protocol(tmp_path: Path) -> None:
    """DINO Memory build with calibration uses the same threshold source."""

    config = load_experiment_config(Path("configs/experiments/dino_memory_fixture.yaml"))
    config = replace(
        config,
        calibration=CalibrationConfig(enabled=True, ratio=0.25, seed=42, quantile=0.99),
        runtime=replace(config.runtime, output_dir=tmp_path / "dino_out"),
    )
    summary = run_build_memory(config)
    assert summary["calibration_enabled"] is True
    model_dir = tmp_path / "dino_out" / "model"
    calibration = json.loads((model_dir / "calibration.json").read_text(encoding="utf-8"))
    assert calibration["threshold_source"] == HELD_OUT_THRESHOLD_SOURCE


# ---------------------------------------------------------------------------
# 9. Threshold source written to artifact
# ---------------------------------------------------------------------------


def test_threshold_source_in_build_summary(tmp_path: Path) -> None:
    """The build summary records the threshold source from calibration."""

    config = _patchcore_config(calibration_enabled=True)
    config = replace(config, runtime=replace(config.runtime, output_dir=tmp_path / "out"))
    summary = run_build_memory(config)
    assert summary["calibration"]["threshold_source"] == HELD_OUT_THRESHOLD_SOURCE


# ---------------------------------------------------------------------------
# 10. Legacy calibration disabled by default
# ---------------------------------------------------------------------------


def test_legacy_calibration_disabled_by_default() -> None:
    """CalibrationConfig defaults have enabled=True (not legacy)."""

    default = CalibrationConfig()
    assert default.enabled is True
    assert default.ratio == 0.1
    assert default.quantile == 0.995


def test_fixture_config_has_calibration_disabled() -> None:
    """Fixture configs explicitly disable calibration (too few samples)."""

    config = load_experiment_config(Path("configs/experiments/patchcore_fixture.yaml"))
    assert config.calibration.enabled is False


# ---------------------------------------------------------------------------
# 11. Serialization preserves split metadata
# ---------------------------------------------------------------------------


def test_split_serialization_round_trip(tmp_path: Path) -> None:
    """Split manifest saves and loads with full consistency."""

    original = split_normal_indices(100, calibration_ratio=0.15, seed=99)
    manifest_path = tmp_path / "split.json"
    original.save(manifest_path)
    loaded = CalibrationSplit.load(manifest_path)
    assert loaded.memory_indices == original.memory_indices
    assert loaded.calibration_indices == original.calibration_indices
    assert loaded.total_samples == original.total_samples
    assert loaded.calibration_ratio == original.calibration_ratio
    assert loaded.seed == original.seed
    loaded.validate_consistency()


# ---------------------------------------------------------------------------
# 12. No test label threshold selection
# ---------------------------------------------------------------------------


def test_no_test_labels_in_calibration_manifest(tmp_path: Path) -> None:
    """The calibration split manifest contains only train indices, never test."""

    config = _patchcore_config(calibration_enabled=True)
    config = replace(config, runtime=replace(config.runtime, output_dir=tmp_path / "out"))
    run_build_memory(config)
    model_dir = tmp_path / "out" / "model"
    split_data = json.loads((model_dir / "calibration_split.json").read_text(encoding="utf-8"))
    train = SyntheticFixtureDataset("train", (32, 32))
    train_size = len(train)
    for index in split_data["calibration_indices"] + split_data["memory_indices"]:
        assert 0 <= index < train_size
    mem_set = set(split_data["memory_indices"])
    cal_set = set(split_data["calibration_indices"])
    assert not (mem_set & cal_set)
    assert len(mem_set) + len(cal_set) == split_data["total_samples"]


def test_model_reload_preserves_calibrated_thresholds(tmp_path: Path) -> None:
    """A reloaded model retains calibrated thresholds from the build step."""

    config = _patchcore_config(calibration_enabled=True)
    config = replace(config, runtime=replace(config.runtime, output_dir=tmp_path / "out"))
    run_build_memory(config)
    model_dir = tmp_path / "out" / "model"
    loaded = load_detector(model_dir, device="cpu")
    assert isinstance(loaded, PatchCoreDetector)
    assert loaded.config.scoring.threshold_source == HELD_OUT_THRESHOLD_SOURCE
    assert loaded.config.scoring.image_threshold is not None
    assert loaded.config.scoring.pixel_threshold is not None


def test_split_consistency_validation_catches_overlap() -> None:
    """validate_consistency raises on overlapping indices."""

    bad = CalibrationSplit(
        memory_indices=(0, 1, 2),
        calibration_indices=(2, 3),
        total_samples=4,
        calibration_ratio=0.5,
        seed=0,
    )
    with pytest.raises(CalibrationSplitError, match="overlap"):
        bad.validate_consistency()
