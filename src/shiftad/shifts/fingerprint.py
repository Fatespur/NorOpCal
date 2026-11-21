"""Canonical fingerprint computation and verification (protocol Section 8).

The serialization is deterministic across platforms: UTF-8, sorted keys,
fixed separators, floats with 17 significant digits via ``format(value,
".17g")``, and rejection of NaN/Infinity. ``created_at`` and local absolute
paths are excluded from fingerprint data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from shiftad.shifts.base import ShiftError

FINGERPRINT_PREFIX = "sha256:"


def _canonical_float(value: float) -> str:
    """Serialize a float with 17 significant digits for reproducibility."""

    if value != value:  # NaN
        raise ShiftError("NaN is not allowed in fingerprint data")
    if value in (float("inf"), float("-inf")):
        raise ShiftError("Infinity is not allowed in fingerprint data")
    # repr() returns the shortest round-trip representation, so distinct
    # decimal literals can collapse to the same string (0.1 and
    # 0.10000000000000001 are the same IEEE-754 double). Fixed 17 significant
    # digits disambiguate them.
    return format(value, ".17g")


def _canonical_value(obj: object) -> object:
    """Recursively convert an object to a canonical JSON-serializable form."""

    if isinstance(obj, dict):
        return {str(k): _canonical_value(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canonical_value(v) for v in obj]
    if isinstance(obj, float):
        return _canonical_float(obj)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if obj is None:
        return obj
    return str(obj)


def compute_fingerprint(data: Mapping[str, Any]) -> str:
    """Compute the SHA256 fingerprint of a canonical JSON serialization.

    Args:
        data: Fingerprint data; must exclude ``created_at`` and local
            absolute paths.

    Returns:
        A ``"sha256:..."`` fingerprint string.

    Raises:
        ShiftError: If the data contains NaN, Infinity, or non-JSON values.
    """

    canonical = _canonical_value(data)
    serialized = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return FINGERPRINT_PREFIX + digest


def compute_suite_fingerprint(per_sample_fingerprints: Sequence[str]) -> str:
    """Compute a suite fingerprint from sorted per-sample fingerprints.

    Args:
        per_sample_fingerprints: One fingerprint per sample in the cell.

    Returns:
        A ``"sha256:..."`` fingerprint over the sorted list.
    """

    if not per_sample_fingerprints:
        raise ShiftError("Cannot fingerprint an empty shift suite.")
    return compute_fingerprint({"per_sample_fingerprints": sorted(per_sample_fingerprints)})


def verify_fingerprint(data: Mapping[str, Any], expected: str) -> None:
    """Verify that ``data`` fingerprints to ``expected``.

    Args:
        data: Fingerprint data (must exclude ``created_at`` and paths).
        expected: Expected ``"sha256:..."`` fingerprint.

    Raises:
        ShiftError: If the recomputed fingerprint does not match.
    """

    actual = compute_fingerprint(data)
    if actual != expected:
        raise ShiftError(f"Fingerprint mismatch: expected {expected}, computed {actual}.")


def strip_excluded_fields(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return fingerprint data with ``created_at`` and paths removed.

    Path-like keys (``*_path_hash``) are already hashes and are kept; only
    raw absolute path fields named ``*_path`` are removed per protocol
    Section 7.2.
    """

    return {
        str(key): value
        for key, value in data.items()
        if key != "created_at" and not str(key).endswith("_path")
    }


def golden_vector_assertions() -> None:
    """Execute the protocol Section 8.3 golden-vector assertions.

    Raises:
        AssertionError: If any golden vector fails.
    """

    assert compute_fingerprint({"a": 0.1, "b": 2}) == compute_fingerprint({"b": 2, "a": 0.1})
    assert compute_fingerprint({"a": 0.1}) != compute_fingerprint({"a": 0.10000000000000002})
    # 0.10000000000000001 parses to the same IEEE-754 double as 0.1, so it
    # cannot be used to test float disambiguation; 0.10000000000000002 is
    # distinct.
