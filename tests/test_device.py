"""Tests for explicit compute-device resolution."""

import pytest
import torch

from shiftad.utils.device import DeviceError, resolve_device


def test_cpu_device_is_always_available() -> None:
    """An explicit CPU request resolves without accelerator assumptions."""

    assert resolve_device("cpu") == torch.device("cpu")


def test_invalid_device_reports_supported_values() -> None:
    """Malformed device strings fail with actionable supported choices."""

    with pytest.raises(DeviceError, match="auto, cpu, cuda"):
        resolve_device("not-a-device")
