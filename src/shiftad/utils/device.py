"""Explicit compute-device resolution with actionable failures."""

from __future__ import annotations

import torch


class DeviceError(RuntimeError):
    """Report an unavailable or unsupported requested compute device."""


def resolve_device(requested: str) -> torch.device:
    """Resolve an ``auto``, CPU, CUDA, or MPS device request.

    Args:
        requested: Device string accepted by PyTorch, or ``auto``.

    Returns:
        A validated PyTorch device.

    Raises:
        DeviceError: If the requested accelerator is unavailable or malformed.

    Assumptions:
        ``auto`` prioritizes CUDA, then MPS, then CPU.
    """

    normalized = requested.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        device = torch.device(normalized)
    except RuntimeError as exc:
        raise DeviceError(
            f"Invalid device {requested!r}: expected auto, cpu, cuda[:index], or mps."
        ) from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise DeviceError(
                f"Requested {requested!r}, but CUDA is unavailable. Use device: cpu or auto."
            )
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise DeviceError(
                f"Requested CUDA device {index}, but only {torch.cuda.device_count()} "
                "device(s) are visible."
            )
    if device.type == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise DeviceError("Requested MPS, but the backend is unavailable. Use device: cpu or auto.")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise DeviceError(f"Unsupported device type {device.type!r}; use auto, cpu, cuda, or mps.")
    return device
