"""Subspace fitting for the K0 displacement geometry.

K0 instruction Sections 15-16: the primary Route K oracle conditions the
displacement basis on the retrieved source reference through a fixed-size
source-normal neighborhood (32 neighbors); the rank rule selects the smallest
rank explaining at least 90% of S1 benign-shift displacement energy with a
hard maximum of 16. Matched-rank controls (M3 random, M4 destroyed
correspondence) must use the same effective rank.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shiftad.route_k.displacement import PairedDisplacement


class SubspaceError(RuntimeError):
    """Report invalid subspace fitting."""


@dataclass(frozen=True)
class PcaResult:
    """PCA/SVD result for one displacement set."""

    basis: np.ndarray  # [D, r] orthonormal
    singular_values: np.ndarray  # [min(N, D)]
    explained_fraction: float  # cumulative energy at the selected rank
    effective_rank: int  # selected rank
    rank_for_energy: dict[int, int]  # energy threshold percent -> rank

    def project(self, vectors: np.ndarray) -> np.ndarray:
        """Project ``[N, D]`` vectors onto the basis (parallel component)."""
        return np.asarray(vectors @ self.basis @ self.basis.T)

    def residual(self, vectors: np.ndarray) -> np.ndarray:
        """Return the orthogonal residual ``[N, D]``."""
        return np.asarray(vectors - self.project(vectors))


def rank_for_energy(
    singular_values: np.ndarray,
    *,
    energy_fraction: float = 0.9,
    max_rank: int = 16,
) -> int:
    """Return the smallest rank whose cumulative energy reaches the threshold.

    Args:
        singular_values: Sorted (descending) singular values.
        energy_fraction: Cumulative energy threshold in ``(0, 1]``.
        max_rank: Hard maximum rank.

    Returns:
        The selected rank (at least 1, at most ``max_rank``).
    """
    if singular_values.size == 0:
        raise SubspaceError("Cannot select a rank from an empty spectrum.")
    values = np.asarray(singular_values, dtype=np.float64)
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise SubspaceError("Singular values must be finite and non-negative.")
    total = float(values.sum())
    if total <= 0.0:
        raise SubspaceError("Cannot select a rank from zero displacement energy.")
    cumulative = np.cumsum(values) / total
    target = min(1.0, max(energy_fraction, 0.0))
    rank = int(np.searchsorted(cumulative, target) + 1)
    return min(max(rank, 1), max_rank)


def _centered_spectrum(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(singular_values, right_vectors)`` of a mean-centered matrix.

    For ``[m, n]`` with ``m >= n`` uses the Gram matrix ``X^T X`` eigen
    decomposition (exact for the top-``n`` singular subspace, far cheaper
    than the full SVD); otherwise uses the direct SVD. Both are exact
    float64 computations; the returned singular values are the square roots
    of the Gram eigenvalues (clipped at zero for numerical noise).
    """
    samples, dimension = matrix.shape
    if samples >= dimension:
        gram = matrix.T @ matrix
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        singular_values = np.sqrt(np.clip(eigenvalues, 0.0, None))
        return singular_values, eigenvectors
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    return singular_values, vh.T


def fit_pca(displacements: np.ndarray, *, rank: int) -> PcaResult:
    """Fit a PCA basis of the requested rank to ``[N, D]`` displacements.

    Uses the (mean-centered) displacement spectrum and returns the exact
    orthonormal right singular vectors. ``rank`` must be in
    ``[1, min(N, D)]``.
    """
    matrix = np.asarray(displacements, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise SubspaceError(f"Displacement matrix must be [N, D] with N > 0, got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise SubspaceError("Displacement matrix contains non-finite values.")
    if not 1 <= rank <= min(matrix.shape):
        raise SubspaceError(f"Rank {rank} is invalid for shape {matrix.shape}.")
    # Mean-center: the displacement geometry is about directions of change,
    # not the absolute feature offset (a constant shift of all features is
    # not a direction).
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular_values, right_vectors = _centered_spectrum(centered)
    basis = right_vectors[:, :rank].astype(np.float32)
    total = float(singular_values.sum())
    rank_for_energy_90 = rank_for_energy(singular_values, energy_fraction=0.9, max_rank=16)
    rank_for_energy_80 = rank_for_energy(singular_values, energy_fraction=0.8, max_rank=16)
    rank_for_energy_95 = rank_for_energy(singular_values, energy_fraction=0.95, max_rank=16)
    explained = float(np.cumsum(singular_values)[:rank].sum() / total) if total > 0 else 0.0
    return PcaResult(
        basis=basis,
        singular_values=singular_values.astype(np.float32),
        explained_fraction=explained,
        effective_rank=rank,
        rank_for_energy={
            80: rank_for_energy_80,
            90: rank_for_energy_90,
            95: rank_for_energy_95,
        },
    )


def fit_pca_with_rank_rule(
    displacements: np.ndarray,
    *,
    energy_fraction: float = 0.9,
    max_rank: int = 16,
) -> PcaResult:
    """Fit PCA using the frozen K0 rank rule (smallest rank at energy threshold)."""
    matrix = np.asarray(displacements, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular_values, right_vectors = _centered_spectrum(centered)
    rank = rank_for_energy(singular_values, energy_fraction=energy_fraction, max_rank=max_rank)
    rank_80 = rank_for_energy(singular_values, energy_fraction=0.8, max_rank=max_rank)
    rank_95 = rank_for_energy(singular_values, energy_fraction=0.95, max_rank=max_rank)
    return PcaResult(
        basis=right_vectors[:, :rank].astype(np.float32),
        singular_values=singular_values.astype(np.float32),
        explained_fraction=float(np.cumsum(singular_values)[:rank].sum() / singular_values.sum()),
        effective_rank=rank,
        rank_for_energy={80: rank_80, 90: rank, 95: rank_95},
    )


def local_basis_from_neighbor_displacements(
    neighbor_displacements: list[PairedDisplacement],
    *,
    rank: int,
    feature_dimension: int,
) -> PcaResult:
    """Fit a local basis from the displacements of one reference neighborhood.

    ``neighbor_displacements`` must belong to S1 identities only and to the
    oracle shift family. Records with invalid correspondence are excluded.
    """
    selected = [record for record in neighbor_displacements if record.correspondence_valid]
    if len(selected) < max(rank, 2):
        raise SubspaceError(
            f"Neighborhood contains only {len(selected)} valid displacements for rank {rank}."
        )
    matrix = np.stack([record.delta for record in selected]).astype(np.float64)
    return fit_pca(matrix, rank=rank)


def random_orthonormal_basis(dimension: int, rank: int, *, seed: int) -> np.ndarray:
    """Return a deterministic random orthonormal basis ``[D, r]``.

    Uses a seeded Gaussian matrix and QR; the sign pattern is fixed by
    canonicalizing each column's largest-magnitude entry to positive.
    """
    if not 1 <= rank <= dimension:
        raise SubspaceError(f"Rank {rank} is invalid for dimension {dimension}.")
    generator = np.random.default_rng(seed)
    matrix = generator.standard_normal((dimension, rank))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    canonical = q * signs
    # Re-orthonormalize once for numerical safety after sign application.
    q2, r2 = np.linalg.qr(canonical)
    return np.asarray((q2 * np.sign(np.diag(r2))).astype(np.float32))


def destroyed_correspondence_permutation(count: int, *, seed: int) -> np.ndarray:
    """Return a deterministic permutation of ``count`` displacement rows.

    The permutation is a single fixed-point-free permutation on the row
    indices, generated from a seeded RNG; it destroys clean/transformed patch
    correspondence while keeping the same displacement multiset.
    """
    if count < 2:
        raise SubspaceError(f"Cannot destroy correspondence for {count} rows.")
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(count)
    if np.array_equal(permutation, np.arange(count)):
        # A fixed point everywhere would keep correspondence intact; swap the
        # first two rows deterministically to guarantee destruction.
        permutation[0], permutation[1] = permutation[1], permutation[0]
    return permutation


def subspace_similarity(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Return the principal-angle cosine similarity between two bases.

    Computes the smallest principal angle via SVD of the cross-Gram matrix
    (values in ``[0, 1]``; 1 means identical subspaces).
    """
    gram = np.asarray(basis_a, dtype=np.float64).T @ np.asarray(basis_b, dtype=np.float64)
    singular = np.linalg.svd(gram, compute_uv=False)
    return float(np.clip(singular[0], 0.0, 1.0))


def principal_angles(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """Return all principal-angle cosines between two bases (sorted descending)."""
    gram = np.asarray(basis_a, dtype=np.float64).T @ np.asarray(basis_b, dtype=np.float64)
    singular = np.linalg.svd(gram, compute_uv=False)
    return np.clip(singular, 0.0, 1.0)
