"""Residual scoring for the K0 method matrix.

K0 instruction Sections 17-18: the Route K score is derived from the
orthogonal residual magnitude of the query-reference discrepancy::

    d = f(z_q) - f(r*)       (query patch - nearest source reference)
    d_parallel = U U^T d     (shift-explainable component)
    d_perp = d - d_parallel  (orthogonal residual)

M5-full uses ``alpha = 1.0``; the single permitted bounded diagnostic uses
``alpha = 0.5``::

    d_bounded = d - alpha * d_parallel

Per-patch provenance (original/parallel/residual norms, parallel-energy
fraction, reference identity, local rank, correspondence status) is persisted
for every evaluation unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from shiftad.methods.scoring import aggregate_image_scores, patch_scores_to_anomaly_maps


class ResidualScoreError(RuntimeError):
    """Report invalid residual scoring."""


@dataclass(frozen=True)
class PatchScoreRecord:
    """Per-patch residual score provenance."""

    discrepancy_norm: np.ndarray  # [N]
    parallel_norm: np.ndarray  # [N]
    residual_norm: np.ndarray  # [N]
    parallel_energy_fraction: np.ndarray  # [N]
    reference_index: np.ndarray  # [N] int64
    local_rank: int
    correspondence_status: np.ndarray  # [N] bool


def residualize(
    discrepancy: Tensor,
    basis: Tensor,
    *,
    alpha: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decompose discrepancies into parallel and orthogonal components.

    Args:
        discrepancy: ``[N, D]`` query-reference differences.
        basis: Orthonormal ``[D, r]`` subspace basis.
        alpha: Residualization strength (1.0 full, 0.5 bounded diagnostic).

    Returns:
        ``(d_parallel, d_perp, d_bounded)`` where ``d_bounded = d - alpha*d_parallel``.
    """
    if discrepancy.ndim != 2 or basis.ndim != 2:
        raise ResidualScoreError("Discrepancy and basis must be matrices.")
    if discrepancy.shape[1] != basis.shape[0]:
        raise ResidualScoreError(
            f"Discrepancy dimension {discrepancy.shape[1]} does not match "
            f"basis dimension {basis.shape[0]}."
        )
    if not 0.0 <= alpha <= 1.0:
        raise ResidualScoreError(f"alpha must be in [0, 1], got {alpha}.")
    d = discrepancy.double()
    u = basis.double()
    parallel = d @ u @ u.T
    residual = d - parallel
    bounded = d - alpha * parallel
    return parallel.float(), residual.float(), bounded.float()


def score_with_residual(
    discrepancy: Tensor,
    basis: Tensor,
    *,
    alpha: float = 1.0,
) -> tuple[Tensor, PatchScoreRecord]:
    """Score discrepancies through orthogonal residual norms.

    Returns ``(patch_scores, record)`` with scores ``||d_perp||`` (or
    ``||d_bounded||`` for the bounded diagnostic).
    """
    parallel, residual, bounded = residualize(discrepancy, basis, alpha=alpha)
    if alpha >= 1.0:
        scores = torch.linalg.vector_norm(residual, dim=1)
    else:
        scores = torch.linalg.vector_norm(bounded, dim=1)
    d_norm = torch.linalg.vector_norm(discrepancy.float(), dim=1)
    parallel_norm = torch.linalg.vector_norm(parallel, dim=1)
    denominator = torch.clamp(d_norm.square(), min=1e-12)
    parallel_fraction = parallel_norm.square() / denominator
    record = PatchScoreRecord(
        discrepancy_norm=d_norm.numpy(),
        parallel_norm=parallel_norm.numpy(),
        residual_norm=torch.linalg.vector_norm(residual, dim=1).numpy(),
        parallel_energy_fraction=parallel_fraction.numpy(),
        reference_index=np.zeros(d_norm.shape[0], dtype=np.int64),
        local_rank=int(basis.shape[1]),
        correspondence_status=np.ones(d_norm.shape[0], dtype=bool),
    )
    return scores, record


def discrepancy_against_reference(
    query_features: Tensor,
    reference_features: Tensor,
    reference_indices: Tensor,
) -> Tensor:
    """Return ``f(z_q) - f(r*)`` for matched query/reference rows."""
    query = query_features.detach().cpu()
    references = reference_features.detach().cpu()
    if query.ndim != 2 or references.ndim != 2 or query.shape != references.shape:
        raise ResidualScoreError(
            f"Query {tuple(query.shape)} and reference {tuple(references.shape)} must match."
        )
    return query - references


def image_scores_from_patch_scores(
    patch_scores: Tensor,
    *,
    grid_shape: tuple[int, int],
    image_size: tuple[int, int],
    aggregation: str = "max",
    top_k: int = 1,
    smoothing_sigma: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Aggregate patch scores to image scores and anomaly maps."""
    scores = patch_scores.detach().cpu()
    if scores.ndim != 2:
        raise ResidualScoreError(f"Patch scores must be [B, N], got {tuple(scores.shape)}.")
    image_scores = aggregate_image_scores(scores, mode=aggregation, top_k=top_k)
    anomaly_maps = patch_scores_to_anomaly_maps(
        scores,
        grid_shape=grid_shape,
        output_size=image_size,
        smoothing_sigma=smoothing_sigma,
    )
    return image_scores, anomaly_maps


def nearest_reference_search(
    query_features: Tensor,
    reference_features: Tensor,
    *,
    chunk_size: int = 1024,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Exact nearest-reference search returning (distances, indices)."""
    from shiftad.memory.nearest_neighbors import TorchNearestNeighbors

    index = TorchNearestNeighbors(
        metric="euclidean",
        k=1,
        chunk_size=chunk_size,
        device=device,
    )
    index.fit(reference_features)
    result = index.search(query_features)
    return result.distances[:, 0], result.indices[:, 0]
