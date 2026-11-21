"""Normal memory, coreset, and nearest-neighbour correctness tests."""

import importlib.util
from pathlib import Path

import pytest
import torch

from shiftad.memory.coreset import CoresetError, select_coreset
from shiftad.memory.memory_bank import MemoryBank, MemoryBankError
from shiftad.memory.nearest_neighbors import (
    FaissNearestNeighbors,
    NeighborSearchError,
    TorchNearestNeighbors,
    create_nearest_neighbor,
)


def test_memory_bank_incremental_finalize_immutability_and_round_trip(
    tmp_path: Path,
) -> None:
    """Finalized source memory is exact, immutable, and safely reloadable."""

    bank = MemoryBank(metadata={"category": "widget"}, normalized=True)
    bank.add(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    bank.add(torch.tensor([[1.0, 1.0]]))
    assert bank.size == 3
    assert bank.original_size == 3
    bank.finalize()
    expected = bank.features
    with pytest.raises(MemoryBankError, match="immutable"):
        bank.add(torch.ones(1, 2))
    copied = bank.features
    copied.zero_()
    assert torch.equal(bank.features, expected)

    path = tmp_path / "memory_bank.pt"
    bank.save(path)
    loaded = MemoryBank.load(path)
    assert loaded.metadata == {"category": "widget"}
    assert loaded.normalized is True
    assert loaded.original_size == 3
    assert torch.equal(loaded.features, expected)
    assert loaded.memory_bytes == expected.numel() * expected.element_size()
    assert loaded.to("cpu").device.type == "cpu"
    if torch.cuda.is_available():
        assert loaded.to("cuda").device.type == "cuda"


def test_memory_bank_rejects_empty_and_dimension_mismatch() -> None:
    """Invalid memory state fails before it can contaminate source memory."""

    bank = MemoryBank()
    with pytest.raises(MemoryBankError, match="empty"):
        bank.finalize()
    bank.add(torch.ones(2, 3))
    with pytest.raises(MemoryBankError, match="dimension"):
        bank.add(torch.ones(1, 4))


def test_coreset_modes_are_deterministic_and_nonempty() -> None:
    """No-op, random, and K-center selectors return valid source indices."""

    features = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    no_op = select_coreset(features, method="none")
    assert torch.equal(no_op.indices, torch.arange(10))
    first = select_coreset(features, method="random", ratio=0.3, seed=7)
    second = select_coreset(features, method="random", ratio=0.3, seed=7)
    assert torch.equal(first.indices, second.indices)
    assert len(first.indices) == 3
    kcenter = select_coreset(
        features,
        method="kcenter",
        target_size=4,
        ratio=1.0,
        chunk_size=3,
    )
    assert len(kcenter.indices) == 4
    assert len(torch.unique(kcenter.indices)) == 4


def test_coreset_rejects_ratio_and_empty_inputs() -> None:
    """Coreset boundary violations produce clear errors."""

    with pytest.raises(CoresetError, match="non-empty"):
        select_coreset(torch.empty(0, 3), method="none")
    with pytest.raises(CoresetError, match="ratio"):
        select_coreset(torch.ones(2, 3), method="none", ratio=0.0)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_chunked_neighbor_search_matches_brute_force(metric: str) -> None:
    """Chunking returns the same distances and indices as a full matrix."""

    memory = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    query = torch.tensor([[0.9, 0.1], [1.8, 1.0]])
    search = TorchNearestNeighbors(metric=metric, k=2, chunk_size=1)
    search.fit(memory)
    result = search.search(query)
    if metric == "euclidean":
        brute = torch.cdist(query, memory)
    else:
        brute = 1.0 - torch.nn.functional.normalize(query, dim=1) @ (
            torch.nn.functional.normalize(memory, dim=1).T
        )
    expected_distances, expected_indices = brute.topk(2, largest=False, sorted=True)
    assert torch.allclose(result.distances, expected_distances)
    assert torch.equal(result.indices, expected_indices)


def test_neighbor_k_zero_vectors_and_faiss_fallback() -> None:
    """Invalid k and cosine zero vectors fail; unavailable FAISS falls back."""

    with pytest.raises(NeighborSearchError, match="exceeds"):
        index = TorchNearestNeighbors(k=3)
        index.fit(torch.ones(2, 4))
    cosine = TorchNearestNeighbors(metric="cosine")
    with pytest.raises(NeighborSearchError, match="zero"):
        cosine.fit(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    fallback = create_nearest_neighbor(
        backend="faiss",
        metric="euclidean",
        k=1,
        chunk_size=2,
        device="cpu",
    )
    expected_type = (
        TorchNearestNeighbors
        if importlib.util.find_spec("faiss") is None
        else FaissNearestNeighbors
    )
    assert isinstance(fallback, expected_type)
