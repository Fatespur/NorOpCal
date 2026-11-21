"""Chunked exact PyTorch nearest-neighbour search with optional FAISS fallback."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as functional
from torch import Tensor

LOGGER = logging.getLogger(__name__)


class NeighborSearchError(RuntimeError):
    """Report invalid memory, query, metric, or neighbour-count inputs."""


@dataclass(frozen=True)
class NeighborSearchResult:
    """Store nearest distances and memory-row indices."""

    distances: Tensor
    indices: Tensor
    backend: str


class NearestNeighborIndex(Protocol):
    """Protocol for fitted nearest-neighbour backends."""

    @property
    def backend_name(self) -> str:
        """Return the active backend identifier."""

    def fit(self, memory: Tensor) -> None:
        """Fit a search index from ``[M, D]`` features."""

    def search(self, query: Tensor) -> NeighborSearchResult:
        """Return nearest distances and indices for ``[Q, D]`` queries."""


class TorchNearestNeighbors:
    """Perform exact distance search without a full query-memory matrix."""

    def __init__(
        self,
        *,
        metric: str = "euclidean",
        k: int = 1,
        chunk_size: int = 1024,
        device: torch.device | str = "cpu",
    ) -> None:
        if metric not in {"euclidean", "cosine"}:
            raise NeighborSearchError(
                f"Unsupported distance metric {metric!r}; use euclidean or cosine."
            )
        if k <= 0 or chunk_size <= 0:
            raise NeighborSearchError("Nearest-neighbour k and chunk_size must be positive.")
        self.metric = metric
        self.k = k
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        self._memory: Tensor | None = None

    @property
    def backend_name(self) -> str:
        """Return the active backend identifier."""

        return "torch"

    def fit(self, memory: Tensor) -> None:
        """Copy a finite ``[M, D]`` memory matrix to the configured device."""

        self._validate_matrix(memory, "memory")
        if self.k > memory.shape[0]:
            raise NeighborSearchError(
                f"Nearest-neighbour k={self.k} exceeds memory size {memory.shape[0]}."
            )
        if self.metric == "cosine" and torch.any(torch.linalg.vector_norm(memory, dim=1) <= 1e-12):
            raise NeighborSearchError("Cosine memory contains zero vectors.")
        self._memory = memory.detach().to(self.device).contiguous()

    @staticmethod
    def _validate_matrix(values: Tensor, name: str) -> None:
        if values.ndim != 2 or min(values.shape) <= 0:
            raise NeighborSearchError(
                f"Nearest-neighbour {name} shape {tuple(values.shape)}; expected [N, D]."
            )
        if not values.is_floating_point() or not torch.isfinite(values).all():
            raise NeighborSearchError(
                f"Nearest-neighbour {name} must contain finite floating-point values."
            )

    def _distances(self, query: Tensor, memory: Tensor) -> Tensor:
        if self.metric == "euclidean":
            return torch.cdist(query, memory)
        query_normalized = functional.normalize(query, dim=1, eps=1e-12)
        memory_normalized = functional.normalize(memory, dim=1, eps=1e-12)
        similarities: Tensor = torch.matmul(
            query_normalized,
            memory_normalized.transpose(0, 1),
        )
        return torch.ones_like(similarities) - similarities

    @torch.inference_mode()
    def search(self, query: Tensor) -> NeighborSearchResult:
        """Find exact neighbours using query and memory chunking."""

        if self._memory is None:
            raise NeighborSearchError("Nearest-neighbour index must be fitted before search.")
        self._validate_matrix(query, "query")
        if query.shape[1] != self._memory.shape[1]:
            raise NeighborSearchError(
                f"Query dimension {query.shape[1]} does not match memory dimension "
                f"{self._memory.shape[1]}."
            )
        if self.metric == "cosine" and torch.any(torch.linalg.vector_norm(query, dim=1) <= 1e-12):
            raise NeighborSearchError("Cosine query contains zero vectors.")
        queries = query.to(self.device)
        output_distances: list[Tensor] = []
        output_indices: list[Tensor] = []
        for query_start in range(0, len(queries), self.chunk_size):
            query_chunk = queries[query_start : query_start + self.chunk_size]
            best_distances = torch.full(
                (len(query_chunk), self.k),
                torch.inf,
                device=self.device,
            )
            best_indices = torch.full(
                (len(query_chunk), self.k),
                -1,
                dtype=torch.int64,
                device=self.device,
            )
            for memory_start in range(0, len(self._memory), self.chunk_size):
                memory_chunk = self._memory[memory_start : memory_start + self.chunk_size]
                distances = self._distances(query_chunk, memory_chunk)
                indices = (
                    torch.arange(
                        memory_start,
                        memory_start + len(memory_chunk),
                        device=self.device,
                    )
                    .unsqueeze(0)
                    .expand(len(query_chunk), -1)
                )
                candidates = torch.cat((best_distances, distances), dim=1)
                candidate_indices = torch.cat((best_indices, indices), dim=1)
                best_distances, positions = torch.topk(
                    candidates,
                    k=self.k,
                    dim=1,
                    largest=False,
                    sorted=True,
                )
                best_indices = candidate_indices.gather(1, positions)
            output_distances.append(best_distances.cpu())
            output_indices.append(best_indices.cpu())
        return NeighborSearchResult(
            distances=torch.cat(output_distances),
            indices=torch.cat(output_indices),
            backend="torch",
        )


class FaissNearestNeighbors:
    """Optional exact CPU FAISS backend loaded only when installed."""

    def __init__(self, *, metric: str, k: int) -> None:
        if metric not in {"euclidean", "cosine"} or k <= 0:
            raise NeighborSearchError("FAISS metric or neighbour count is invalid.")
        self.metric = metric
        self.k = k
        self._index: Any = None
        self._dimension: int | None = None

    @property
    def backend_name(self) -> str:
        """Return the active backend identifier."""

        return "faiss"

    def fit(self, memory: Tensor) -> None:
        """Build an exact FAISS flat index from finite CPU float32 features."""

        TorchNearestNeighbors._validate_matrix(memory, "memory")
        if self.k > memory.shape[0]:
            raise NeighborSearchError(
                f"Nearest-neighbour k={self.k} exceeds memory size {memory.shape[0]}."
            )
        values = memory.detach().cpu().to(torch.float32)
        if self.metric == "cosine":
            if torch.any(torch.linalg.vector_norm(values, dim=1) <= 1e-12):
                raise NeighborSearchError("Cosine memory contains zero vectors.")
            values = functional.normalize(values, dim=1)
        import importlib

        faiss = importlib.import_module("faiss")
        self._dimension = values.shape[1]
        self._index = (
            faiss.IndexFlatL2(self._dimension)
            if self.metric == "euclidean"
            else faiss.IndexFlatIP(self._dimension)
        )
        self._index.add(values.numpy())

    def search(self, query: Tensor) -> NeighborSearchResult:
        """Search the optional exact FAISS index."""

        if self._index is None or self._dimension is None:
            raise NeighborSearchError("FAISS index must be fitted before search.")
        TorchNearestNeighbors._validate_matrix(query, "query")
        if query.shape[1] != self._dimension:
            raise NeighborSearchError(
                f"Query dimension {query.shape[1]} does not match memory dimension "
                f"{self._dimension}."
            )
        values = query.detach().cpu().to(torch.float32)
        if self.metric == "cosine":
            if torch.any(torch.linalg.vector_norm(values, dim=1) <= 1e-12):
                raise NeighborSearchError("Cosine query contains zero vectors.")
            values = functional.normalize(values, dim=1)
        distances, indices = self._index.search(values.numpy(), self.k)
        distance_tensor = torch.from_numpy(distances)
        if self.metric == "euclidean":
            distance_tensor = distance_tensor.clamp_min(0).sqrt()
        else:
            distance_tensor = 1.0 - distance_tensor
        return NeighborSearchResult(
            distances=distance_tensor,
            indices=torch.from_numpy(indices).to(torch.int64),
            backend="faiss",
        )


def create_nearest_neighbor(
    *,
    backend: str,
    metric: str,
    k: int,
    chunk_size: int,
    device: torch.device | str,
) -> NearestNeighborIndex:
    """Create the available exact search backend.

    FAISS remains optional. Until its dedicated backend is enabled, requesting
    ``auto`` or ``faiss`` without an importable package records a warning and
    returns the tested PyTorch implementation.
    """

    if backend not in {"auto", "torch", "faiss"}:
        raise NeighborSearchError(f"Unsupported nearest-neighbour backend {backend!r}.")
    if backend in {"auto", "faiss"}:
        if importlib.util.find_spec("faiss") is None:
            LOGGER.warning("FAISS is unavailable; using the exact chunked PyTorch backend.")
        else:
            LOGGER.info("Using the optional exact FAISS backend.")
            return FaissNearestNeighbors(metric=metric, k=k)
    return TorchNearestNeighbors(
        metric=metric,
        k=k,
        chunk_size=chunk_size,
        device=device,
    )
