"""Shift family registry and convenience application API.

The registry freezes the 13 protocol families. ``get_shift`` returns a
severity-bound instance; ``apply_shift`` applies one shift to an image and
optional mask with the derived seed.
"""

from __future__ import annotations

import numpy as np

from shiftad.shifts.base import Shift, ShiftError, ShiftResult, derive_seed
from shiftad.shifts.blur import DefocusBlurShift, MotionBlurShift
from shiftad.shifts.compression import JpegCompressionShift, ResolutionReductionShift
from shiftad.shifts.geometric import RotationShift, ScaleChangeShift, TranslationShift
from shiftad.shifts.noise import GaussianNoiseShift
from shiftad.shifts.photometric import (
    BrightnessShift,
    ColorTemperatureShift,
    ContrastShift,
    GammaShift,
    ShadowShift,
)

_SHIFT_CLASSES: tuple[type[Shift], ...] = (
    BrightnessShift,
    ContrastShift,
    GammaShift,
    ColorTemperatureShift,
    ShadowShift,
    GaussianNoiseShift,
    JpegCompressionShift,
    MotionBlurShift,
    DefocusBlurShift,
    ResolutionReductionShift,
    RotationShift,
    TranslationShift,
    ScaleChangeShift,
)

SHIFT_NAMES: tuple[str, ...] = tuple(cls.name for cls in _SHIFT_CLASSES)

# The only stochastic families, frozen by protocol Section 12.2 (T06).
STOCHASTIC_SHIFT_NAMES: frozenset[str] = frozenset({"gaussian_noise", "shadow"})

_BY_NAME: dict[str, type[Shift]] = {cls.name: cls for cls in _SHIFT_CLASSES}

CLEAN_SHIFT_NAME = "clean"
CLEAN_VARIANT = "none"


def list_shifts() -> list[str]:
    """Return the 13 canonical shift family names in protocol order."""

    return list(SHIFT_NAMES)


def get_shift(name: str, severity: int = 0) -> Shift:
    """Instantiate one frozen shift family at a severity level.

    Args:
        name: Canonical family name (see :func:`list_shifts`).
        severity: Severity level 0-4.

    Returns:
        A severity-bound shift instance.

    Raises:
        ShiftError: If the family name or severity is invalid.
    """

    if name not in _BY_NAME:
        raise ShiftError(f"Unknown shift family {name!r}; expected one of {sorted(_BY_NAME)}.")
    return _BY_NAME[name](severity)


def apply_shift(
    name: str,
    severity: int,
    image: np.ndarray,
    mask: np.ndarray | None,
    derived_seed: int,
) -> ShiftResult:
    """Apply one shift to an image and optional mask.

    Args:
        name: Canonical family name.
        severity: Severity level 0-4.
        image: ``float32`` RGB image with shape ``[H, W, 3]`` in [0, 1].
        mask: Binary mask with shape ``[H, W]`` or ``None``.
        derived_seed: Per-sample seed from :func:`derive_seed`.

    Returns:
        The applied shift result with realized parameters.
    """

    return get_shift(name, severity).apply(image, mask, derived_seed)


def derive_clean_seed(
    *,
    category: str,
    sample_id: str,
    base_seed: int = 42,
) -> int:
    """Derive the seed for a clean (Level 0) cell per protocol Section 4.1.

    Clean cells use ``shift_name="clean"``, ``shift_variant="none"`` and
    ``severity=0``; the transform is identity and consumes no RNG.
    """

    return derive_seed(
        category=category,
        sample_id=sample_id,
        shift_name=CLEAN_SHIFT_NAME,
        severity=0,
        base_seed=base_seed,
    )


def shift_variant(name: str) -> str:
    """Return the frozen variant label for one family."""

    return _BY_NAME[name].variant


def shift_is_stochastic(name: str) -> bool:
    """Return whether a family consumes the derived seed (protocol T06)."""

    if name not in _BY_NAME:
        raise ShiftError(f"Unknown shift family {name!r}.")
    return _BY_NAME[name].stochastic


def severity_names() -> dict[int, str]:
    """Return the frozen severity label mapping (protocol Section 2.2)."""

    return {0: "clean", 1: "mild", 2: "moderate", 3: "strong", 4: "severe"}


def validate_cell_identity(
    *,
    category: str,
    sample_id: str,
    shift_name: str,
    severity: int,
    base_seed: int,
) -> None:
    """Validate a cell identity before seed derivation (protocol Section 4)."""

    if not category:
        raise ShiftError("Category must be a non-empty string.")
    if not sample_id:
        raise ShiftError("sample_id must be a non-empty string.")
    if shift_name not in _BY_NAME and shift_name != CLEAN_SHIFT_NAME:
        raise ShiftError(f"Unknown shift family {shift_name!r}.")
    if not 0 <= severity <= 4:
        raise ShiftError(f"Severity must be between 0 and 4, got {severity}.")
    if not 0 <= base_seed <= 2**32 - 1:
        raise ShiftError("base_seed must be between 0 and 2**32 - 1 inclusive.")
