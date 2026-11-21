"""Public API for the frozen Phase 5 synthetic-shift package.

Exposes shift application, the registry, seed derivation, metadata, and
fingerprint helpers. All transforms are deterministic for a fixed derived
seed and operate at native source resolution.
"""

from __future__ import annotations

from shiftad.shifts.base import (
    BASE_SEED,
    NOMINAL_SOURCE_SIZE,
    PROTOCOL_VERSION,
    Shift,
    ShiftError,
    ShiftResult,
    derive_seed,
)
from shiftad.shifts.fingerprint import (
    compute_fingerprint,
    compute_suite_fingerprint,
    golden_vector_assertions,
    verify_fingerprint,
)
from shiftad.shifts.metadata import (
    compute_shift_metadata,
    file_sha256,
    image_sha256,
    mask_sha256,
    per_sample_fingerprint,
)
from shiftad.shifts.registry import (
    CLEAN_SHIFT_NAME,
    CLEAN_VARIANT,
    SHIFT_NAMES,
    STOCHASTIC_SHIFT_NAMES,
    apply_shift,
    derive_clean_seed,
    get_shift,
    list_shifts,
    severity_names,
    shift_is_stochastic,
    shift_variant,
    validate_cell_identity,
)

__all__ = [
    "BASE_SEED",
    "CLEAN_SHIFT_NAME",
    "CLEAN_VARIANT",
    "NOMINAL_SOURCE_SIZE",
    "PROTOCOL_VERSION",
    "SHIFT_NAMES",
    "STOCHASTIC_SHIFT_NAMES",
    "Shift",
    "ShiftError",
    "ShiftResult",
    "apply_shift",
    "compute_fingerprint",
    "compute_shift_metadata",
    "compute_suite_fingerprint",
    "derive_clean_seed",
    "derive_seed",
    "file_sha256",
    "get_shift",
    "golden_vector_assertions",
    "image_sha256",
    "list_shifts",
    "mask_sha256",
    "per_sample_fingerprint",
    "severity_names",
    "shift_is_stochastic",
    "shift_variant",
    "validate_cell_identity",
    "verify_fingerprint",
]
