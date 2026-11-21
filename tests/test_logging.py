"""Tests for readable and structured ShiftAD logging."""

import json
from pathlib import Path

import pytest

from shiftad.utils.logging import configure_logging


def test_json_log_file_contains_structured_record(tmp_path: Path) -> None:
    """JSON logging writes a parseable record without process identifiers."""

    destination = tmp_path / "logs" / "run.jsonl"
    logger = configure_logging("INFO", log_file=destination, json_output=True)
    logger.info("foundation ready")

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "shiftad"
    assert payload["message"] == "foundation ready"
    assert "process" not in payload


def test_invalid_log_level_is_rejected() -> None:
    """An unknown log level does not silently fall back."""

    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging("VERBOSE")
