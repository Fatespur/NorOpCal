"""Deterministic, memory-bounded coreset selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


class CoresetError(ValueError):
    """Report invalid features or coreset selection controls."""


@dataclass(frozen=True)
class CoresetResult:
    """Store selected features and their source-row indices."""

    features: Tensor
    indices: Tensor
    method: str
    original_size: int


def _selection_size(size: int, ratio: float, target_size: int | None) -> int:
    if not 0 < ratio <= 1:
        raise CoresetError(f"Coreset ratio {ratio} is invalid; expected a value in (0, 1].")
    if target_size is not None and target_size <= 0:
        raise CoresetError("Coreset target_size must be positive when provided.")
    selected = target_size if target_size is not None else math.ceil(size * ratio)
    return min(size, selected)


def _kcenter_indices(features: Tensor, count: int, chunk_size: int) -> Tensor:
    if chunk_size <= 0:
        raise CoresetError("K-center chunk_size must be positive.")
    cpu_features = features.detach().to(device="cpu", dtype=torch.float32)
    center = torch.linalg.vector_norm(
        cpu_features - cpu_features.mean(dim=0, keepdim=True),
        dim=1,
    ).argmax()
    selected = [int(center)]
    minimum_distances = torch.full((len(cpu_features),), torch.inf)
    while len(selected) < count:
        current = cpu_features[selected[-1] : selected[-1] + 1]
        for start in range(0, len(cpu_features), chunk_size):
            stop = min(start + chunk_size, len(cpu_features))
            distances = torch.cdist(cpu_features[start:stop], current).squeeze(1)
            minimum_distances[start:stop] = torch.minimum(
                minimum_distances[start:stop],
                distances,
            )
        minimum_distances[selected] = -1
        selected.append(int(minimum_distances.argmax()))
    return torch.tensor(selected, dtype=torch.int64)


def select_coreset(
    features: Tensor,
    *,
    method: str,
    ratio: float = 1.0,
    target_size: int | None = None,
    seed: int = 0,
    chunk_size: int = 4096,
) -> CoresetResult:
    """Select a no-op, random, or K-center subset of ``[N, D]`` features.

    Args:
        features: Non-empty finite feature matrix.
        method: ``none``, ``random``, or ``kcenter``.
        ratio: Selected fraction when ``target_size`` is omitted.
        target_size: Optional explicit maximum selected count.
        seed: Random-selection seed.
        chunk_size: K-center distance-update chunk size.

    Returns:
        CPU feature copies and source indices in deterministic order.

    Raises:
        CoresetError: If inputs or controls are invalid.

    Notes:
        K-center stores one minimum-distance vector and calculates distances
        from one new center in chunks. It never allocates an ``N x N`` matrix.
    """

    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise CoresetError(
            f"Coreset features have shape {tuple(features.shape)}; expected non-empty [N, D]."
        )
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise CoresetError("Coreset features must be finite floating-point values.")
    if method not in {"none", "random", "kcenter"}:
        raise CoresetError(f"Unsupported coreset method {method!r}; use none, random, or kcenter.")
    size = features.shape[0]
    requested_count = _selection_size(size, ratio, target_size)
    count = size if method == "none" else requested_count
    if method == "none":
        indices = torch.arange(size, dtype=torch.int64)
    elif method == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        indices = torch.randperm(size, generator=generator)[:count].sort().values
    else:
        indices = _kcenter_indices(features, count, chunk_size)
    cpu_features = features.detach().to(device="cpu").index_select(0, indices).clone()
    return CoresetResult(
        features=cpu_features,
        indices=indices,
        method=method,
        original_size=size,
    )
