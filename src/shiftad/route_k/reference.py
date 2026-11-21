"""Immutable source-normal reference construction and nearest-reference search.

K0 instruction Section 14: reuse the accepted Phase 3 source-normal patch
memory / nearest-reference semantics wherever technically compatible. For
each query patch the source reference is::

    r* = argmin_{r in M_s} D(f(z_q), f(r))

The current query never becomes part of the reference bank; no target memory
is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from shiftad.memory.coreset import select_coreset
from shiftad.memory.nearest_neighbors import (
    NeighborSearchResult,
    TorchNearestNeighbors,
)


class ReferenceError(RuntimeError):
    """Report invalid reference construction or search."""


@dataclass(frozen=True)
class ReferenceBank:
    """Immutable source-normal patch memory with its coreset indices."""

    features: Tensor  # [M, D] coreset features (frozen)
    coreset_indices: Tensor  # [M] indices into the original feature pool
    original_size: int

    @property
    def size(self) -> int:
        return int(self.features.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.features.shape[1])


def build_reference_bank(
    normal_features: Tensor,
    *,
    coreset_method: str = "kcenter",
    coreset_ratio: float = 0.1,
    seed: int = 0,
    chunk_size: int = 1024,
) -> ReferenceBank:
    """Build the immutable source-normal reference bank.

    Args:
        normal_features: All source-normal patch features ``[N, D]`` (train/good).
        coreset_method: Coreset selection method (Phase 3 accepted: kcenter).
        coreset_ratio: Coreset compression ratio (Phase 3 accepted: 0.1).
        seed: Coreset RNG seed (Phase 3 uses seed 0).
        chunk_size: Chunk size for coreset distance computations.

    Returns:
        The frozen reference bank.

    Raises:
        ReferenceError: If the feature matrix is invalid or coreset fails.
    """
    features = normal_features.detach().cpu()
    if features.ndim != 2 or features.shape[0] == 0 or not torch.isfinite(features).all():
        raise ReferenceError(f"Invalid normal feature matrix {tuple(features.shape)}.")
    result = select_coreset(
        features,
        method=coreset_method,
        ratio=coreset_ratio,
        target_size=None,
        seed=seed,
        chunk_size=chunk_size,
    )
    return ReferenceBank(
        features=result.features.detach().cpu().contiguous(),
        coreset_indices=result.indices.detach().cpu(),
        original_size=result.original_size,
    )


class ReferenceIndex:
    """Exact nearest-reference search over the immutable bank."""

    def __init__(
        self,
        bank: ReferenceBank,
        *,
        metric: str = "euclidean",
        k: int = 1,
        chunk_size: int = 1024,
        device: torch.device | str = "cpu",
    ) -> None:
        if metric != "euclidean":
            raise ReferenceError("K0 reference search uses the frozen euclidean metric.")
        self.bank = bank
        self._index = TorchNearestNeighbors(
            metric=metric,
            k=k,
            chunk_size=chunk_size,
            device=device,
        )
        self._index.fit(bank.features)

    def search(self, query: Tensor) -> NeighborSearchResult:
        """Return the nearest reference distances and indices for ``[Q, D]``."""
        return self._index.search(query.detach().cpu())


def build_s1_neighborhood_index(
    s1_features: Tensor,
    *,
    k: int,
    chunk_size: int = 1024,
    device: torch.device | str = "cpu",
) -> TorchNearestNeighbors:
    """Build the S1-only clean-feature neighborhood index for local geometry.

    The neighborhood space is the frozen clean feature space; only S1
    identities may enter (K0 instruction Section 15.2).
    """
    features = s1_features.detach().cpu()
    if features.ndim != 2 or features.shape[0] == 0 or not torch.isfinite(features).all():
        raise ReferenceError(f"Invalid S1 feature matrix {tuple(features.shape)}.")
    index = TorchNearestNeighbors(metric="euclidean", k=k, chunk_size=chunk_size, device=device)
    index.fit(features)
    return index


def nearest_s1_neighbors(
    index: TorchNearestNeighbors,
    query_reference: Tensor,
) -> NeighborSearchResult:
    """Return the S1 neighborhood of one reference vector (or batch)."""
    return index.search(query_reference)


def s1_neighborhood_rows(
    result: NeighborSearchResult,
    *,
    s1_start_row: int,
    s1_patches_per_image: int,
) -> list[int]:
    """Map S1 neighborhood search rows back to (image, patch) flat rows.

    ``s1_start_row`` is the first flat row of the given S1 image block inside
    the S1 feature matrix; each image contributes ``s1_patches_per_image``
    consecutive rows (valid patches only).
    """
    rows: list[int] = []
    for row_index in result.indices.flatten().tolist():
        rows.append(s1_start_row + row_index)
    return rows


def make_s1_image_offset_map(valid_counts: list[int]) -> dict[int, int]:
    """Return S1 image ordinal -> first flat row, given per-image valid counts."""
    offsets: dict[int, int] = {}
    cumulative = 0
    for image_ordinal, count in enumerate(valid_counts):
        offsets[image_ordinal] = cumulative
        cumulative += count
    return offsets
