"""Tests for strict experiment configuration loading."""

from pathlib import Path

import pytest

from shiftad.utils.config import (
    ConfigurationError,
    load_experiment_config,
    save_resolved_config,
)


def _write_config(path: Path, *, extra: str = "") -> None:
    path.write_text(
        "\n".join(
            [
                "dataset:",
                "  name: synthetic",
                "  root: data/synthetic",
                "  categories: [widget]",
                "  split: test",
                "  image_size: [32, 48]",
                "runtime:",
                "  seed: 11",
                "  device: cpu",
                "  output_dir: outputs/test",
                "  log_level: INFO",
                "experiment_name: unit_test",
                extra,
            ]
        ),
        encoding="utf-8",
    )


def test_load_and_save_resolved_config(tmp_path: Path) -> None:
    """A valid configuration is immutable and serializes deterministically."""

    source = tmp_path / "config.yaml"
    _write_config(source)
    config = load_experiment_config(source)
    assert config.dataset.categories == ("widget",)
    assert config.dataset.image_size == (32, 48)
    assert config.runtime.seed == 11

    destination = tmp_path / "nested" / "resolved.yaml"
    save_resolved_config(config, destination)
    assert load_experiment_config(destination) == config


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    """Unknown settings fail early instead of being silently ignored."""

    source = tmp_path / "invalid.yaml"
    _write_config(source, extra="invented_field: true")
    with pytest.raises(ConfigurationError, match="unknown fields"):
        load_experiment_config(source)


def test_missing_config_has_actionable_error(tmp_path: Path) -> None:
    """A missing file reports the received path and expected condition."""

    missing = tmp_path / "missing.yaml"
    with pytest.raises(ConfigurationError, match="existing YAML file"):
        load_experiment_config(missing)


def test_dino_memory_config_round_trip_and_nested_unknown_rejection(
    tmp_path: Path,
) -> None:
    """Phase 3 model fields round-trip and reject unrecognized nested controls."""

    source = Path("configs/experiments/dino_memory_fixture.yaml")
    config = load_experiment_config(source)
    assert config.model is not None
    assert config.model.backbone.layers == (0, 1)
    resolved = tmp_path / "resolved.yaml"
    save_resolved_config(config, resolved)
    assert load_experiment_config(resolved) == config

    text = source.read_text(encoding="utf-8").replace(
        "    normalize: true",
        "    normalize: true\n    invented_fusion: false",
    )
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown fields"):
        load_experiment_config(invalid)
