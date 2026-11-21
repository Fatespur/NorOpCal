"""Per-sample shift metadata assembly (protocol Section 7.1).

Every shifted sample records source identity hashes, exact realized
parameters, source-resolution shapes, the model-specific resize target, mask
area tracking, and output hashes. ``created_at`` is excluded from
fingerprints (Section 8.1) but retained in the full metadata record.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from shiftad.shifts.base import PROTOCOL_VERSION, ShiftError
from shiftad.shifts.fingerprint import compute_fingerprint, strip_excluded_fields

SCHEMA_VERSION = "1.0"
IMPLEMENTATION_VERSION = "1.1.0"
IMAGE_BYTES_SERIALIZATION = "uint8_rgb_hwc_raw"
MASK_BYTES_SERIALIZATION = "uint8_mask_hw_raw"


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA256 of raw bytes."""

    return hashlib.sha256(data).hexdigest()


def image_bytes(image: np.ndarray) -> bytes:
    """Serialize a float32 [0, 1] RGB image to deterministic uint8 bytes.

    Args:
        image: ``float32`` image with shape ``[H, W, 3]``.

    Returns:
        Raw ``[H, W, 3]`` uint8 bytes in row-major order (round-trip of the
        protocol's uint8 bridge).
    """

    return np.clip(image * 255.0, 0.0, 255.0).round().astype(np.uint8).tobytes()


def mask_bytes(mask: np.ndarray | None) -> bytes | None:
    """Serialize a binary mask to deterministic uint8 bytes (0/1 values)."""

    if mask is None:
        return None
    return mask.astype(np.uint8).tobytes()


def image_sha256(image: np.ndarray) -> str:
    """Hash a shifted image with the recorded serialization."""

    return sha256_bytes(image_bytes(image))


def mask_sha256(mask: np.ndarray | None) -> str | None:
    """Hash a shifted mask with the recorded serialization."""

    serialized = mask_bytes(mask)
    return sha256_bytes(serialized) if serialized is not None else None


def file_sha256(path: str) -> str:
    """Hash a source file's raw bytes (used for source image/mask hashes).

    Args:
        path: Existing file path.

    Returns:
        Hex SHA256 of the file bytes.
    """

    with open(path, "rb") as handle:
        return sha256_bytes(handle.read())


def compute_shift_metadata(
    *,
    category: str,
    source_split: str,
    source_sample_id: str,
    source_image_path_hash: str,
    source_mask_path_hash: str | None,
    source_image_sha256: str,
    source_mask_sha256: str | None,
    shift_name: str,
    shift_variant: str,
    severity: int,
    exact_parameters: dict[str, Any],
    base_seed: int,
    derived_seed: int,
    transform_code_git_head: str,
    input_shape: list[int],
    output_shape: list[int],
    model_input_shape: list[int],
    mask_area_before: int | None,
    mask_area_after: int | None,
    defect_visible_after_shift: bool | None,
    output_image_sha256: str,
    output_mask_sha256: str | None,
    fixture: bool,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Assemble the protocol Section 7.1 per-sample metadata record.

    Args:
        category: MVTec AD category name.
        source_split: Dataset split, e.g. ``"test"``.
        source_sample_id: Manifest sample identifier.
        source_image_path_hash: SHA256 of the source image path string (or a
            stable relative identifier hash).
        source_mask_path_hash: SHA256 of the source mask path, or ``None``.
        source_image_sha256: SHA256 of the source image file bytes.
        source_mask_sha256: SHA256 of the source mask file bytes or ``None``.
        shift_name: Canonical shift family name.
        shift_variant: Frozen variant label.
        severity: Severity level 0-4.
        exact_parameters: Realized parameters after source-size scaling.
        base_seed: Frozen base seed.
        derived_seed: Per-sample derived seed.
        transform_code_git_head: Git commit hash of the transform code.
        input_shape: Source resolution ``[H_src, W_src]``.
        output_shape: Shifted source resolution (equals input).
        model_input_shape: Model-specific resize target.
        mask_area_before: Anomalous pixels before a geometric shift.
        mask_area_after: Anomalous pixels after a geometric shift.
        defect_visible_after_shift: Whether defects remain visible.
        output_image_sha256: Hash of the shifted image bytes.
        output_mask_sha256: Hash of the shifted mask bytes or ``None``.
        fixture: True for synthetic fixture scope.
        created_at: Explicit UTC timestamp; defaults to the current UTC time.
            Formal artifacts must record real UTC provenance; fixture
            workflows pass a fixed timestamp for byte-determinism.

    Returns:
        The complete metadata record including ``created_at``.
    """

    if severity not in {0, 1, 2, 3, 4}:
        raise ShiftError(f"Invalid severity {severity}.")
    if len(input_shape) != 2 or len(output_shape) != 2 or len(model_input_shape) != 2:
        raise ShiftError("input/output/model shapes must be [H, W] pairs.")
    if created_at is None:
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "source_dataset": "mvtec_ad",
        "category": category,
        "source_split": source_split,
        "source_sample_id": source_sample_id,
        "source_image_path_hash": source_image_path_hash,
        "source_mask_path_hash": source_mask_path_hash,
        "source_image_sha256": source_image_sha256,
        "source_mask_sha256": source_mask_sha256,
        "shift_name": shift_name,
        "shift_variant": shift_variant,
        "severity": severity,
        "exact_parameters": dict(exact_parameters),
        "base_seed": base_seed,
        "derived_seed": derived_seed,
        "implementation_version": IMPLEMENTATION_VERSION,
        "transform_code_git_head": transform_code_git_head,
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "model_input_shape": list(model_input_shape),
        "mask_area_before": mask_area_before,
        "mask_area_after": mask_area_after,
        "defect_visible_after_shift": defect_visible_after_shift,
        "image_sha256": output_image_sha256,
        "mask_sha256": output_mask_sha256,
        "image_bytes_serialization": IMAGE_BYTES_SERIALIZATION,
        "mask_bytes_serialization": MASK_BYTES_SERIALIZATION,
        "created_at": created_at,
        "fixture": fixture,
    }


def per_sample_fingerprint(metadata: dict[str, Any]) -> str:
    """Compute the per-sample shift fingerprint (protocol Section 8.2 #2).

    The fingerprint excludes ``created_at`` and raw path fields but keeps the
    source path hashes, exact parameters, output hashes, implementation
    version, and Git provenance.

    Args:
        metadata: Full metadata record from :func:`compute_shift_metadata`.
    """

    return compute_fingerprint(strip_excluded_fields(metadata))
