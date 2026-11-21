"""Paired benign-shift displacement construction and storage.

K0 instruction Section 13: for each permitted source-normal image ``x``,
transformation ``T_k``, and valid patch correspondence::

    Delta_k(x, p) = f(T_k(x))_{p'} - f(x)_p

Provenance sufficient to reconstruct every displacement is recorded: raw
identity, category, transform family, parameters/severity, transform seed,
clean patch coordinate, transformed patch coordinate, feature source/hash,
and correspondence validity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from torch import Tensor

from shiftad.route_k.correspondence import GRID, PATCH_SIZE, CorrespondenceMap


class DisplacementError(RuntimeError):
    """Report invalid displacement construction."""


@dataclass(frozen=True)
class PairedDisplacement:
    """One clean/transformed patch pair displacement.

    ``delta`` is ``f(T(x))_{p'} - f(x)_p`` in the fused normalized feature
    space; ``correspondence_valid`` marks usable oracle correspondences.
    """

    raw_identity: str
    category: str
    shift_family: str
    severity: int
    transform_seed: int
    clean_patch: int
    transformed_patch: int
    delta: np.ndarray  # [D] float32
    correspondence_valid: bool


@dataclass
class DisplacementStore:
    """In-memory collection of paired displacements.

    Keeps displacements of the *fit* severities for S1 identities (used by
    geometry fitting) and all severities for provenance/energy diagnostics.
    """

    displacements: list[PairedDisplacement] = field(default_factory=list)

    def add(
        self,
        *,
        raw_identity: str,
        category: str,
        shift_family: str,
        severity: int,
        transform_seed: int,
        clean_patch: int,
        transformed_patch: int,
        delta: Tensor | np.ndarray,
        correspondence_valid: bool,
    ) -> None:
        """Append one displacement record."""
        array = np.asarray(delta, dtype=np.float32)
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            raise DisplacementError(f"Invalid displacement for {raw_identity}: {array.shape}.")
        self.displacements.append(
            PairedDisplacement(
                raw_identity=raw_identity,
                category=category,
                shift_family=shift_family,
                severity=severity,
                transform_seed=transform_seed,
                clean_patch=int(clean_patch),
                transformed_patch=int(transformed_patch),
                delta=array,
                correspondence_valid=bool(correspondence_valid),
            )
        )

    def __len__(self) -> int:
        return len(self.displacements)

    def matrix(
        self,
        *,
        category: str | None = None,
        shift_family: str | None = None,
        severities: set[int] | None = None,
        raw_identities: set[str] | None = None,
        correspondence_valid_only: bool = True,
    ) -> tuple[np.ndarray, list[PairedDisplacement]]:
        """Return the displacement matrix ``[N, D]`` and the matching records."""
        selected: list[PairedDisplacement] = []
        for record in self.displacements:
            if category is not None and record.category != category:
                continue
            if shift_family is not None and record.shift_family != shift_family:
                continue
            if severities is not None and record.severity not in severities:
                continue
            if raw_identities is not None and record.raw_identity not in raw_identities:
                continue
            if correspondence_valid_only and not record.correspondence_valid:
                continue
            selected.append(record)
        if not selected:
            return np.empty((0, 0), dtype=np.float32), []
        matrix = np.stack([record.delta for record in selected]).astype(np.float32)
        return matrix, selected


def build_paired_displacements(
    clean_features: Tensor,
    shifted_features: Tensor,
    correspondence: CorrespondenceMap,
    *,
    raw_identity: str,
    category: str,
    shift_family: str,
    severity: int,
    transform_seed: int,
) -> list[PairedDisplacement]:
    """Build per-patch displacements for one clean/transformed image pair.

    Args:
        clean_features: Clean patch features ``[256, D]``.
        shifted_features: Transformed patch features ``[256, D]``.
        correspondence: Oracle correspondence map.
        raw_identity: Source raw image identity.
        category: MVTec AD category.
        shift_family: Shift family name.
        severity: Severity level.
        transform_seed: Derived transform seed.

    Returns:
        One displacement per output patch; records keep both patch indices and
        the correspondence validity flag.
    """
    if clean_features.ndim != 2 or shifted_features.ndim != 2:
        raise DisplacementError("Features must be [256, D] matrices.")
    if clean_features.shape != shifted_features.shape:
        raise DisplacementError("Clean and shifted feature shapes must match.")
    if clean_features.shape[0] != GRID * GRID:
        raise DisplacementError(f"Expected {GRID * GRID} patches, got {clean_features.shape[0]}.")
    records: list[PairedDisplacement] = []
    clean_np = clean_features.detach().cpu().numpy().astype(np.float32)
    shifted_np = shifted_features.detach().cpu().numpy().astype(np.float32)
    for out_index in range(GRID * GRID):
        source_index = int(correspondence.source_patch[out_index])
        valid = bool(correspondence.valid[out_index]) and source_index >= 0
        if valid:
            delta = shifted_np[out_index] - clean_np[source_index]
        else:
            delta = shifted_np[out_index] - clean_np[out_index]
        records.append(
            PairedDisplacement(
                raw_identity=raw_identity,
                category=category,
                shift_family=shift_family,
                severity=severity,
                transform_seed=transform_seed,
                clean_patch=int(source_index) if valid else -1,
                transformed_patch=out_index,
                delta=delta,
                correspondence_valid=valid,
            )
        )
    return records


def patch_index_to_row_col(patch_index: int) -> tuple[int, int]:
    """Convert a flat patch index to ``(col, row)`` grid coordinates."""
    if not 0 <= patch_index < GRID * GRID:
        raise DisplacementError(f"Patch index {patch_index} out of range.")
    return patch_index % GRID, patch_index // GRID


def patch_center_pixels(patch_index: int) -> tuple[float, float]:
    """Return the pixel center of a patch in the 224 grid (``(x, y)``)."""
    col, row = patch_index_to_row_col(patch_index)
    return (col * PATCH_SIZE + PATCH_SIZE / 2.0 - 0.5, row * PATCH_SIZE + PATCH_SIZE / 2.0 - 0.5)
