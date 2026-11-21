"""Incremental normal-patch memory with immutable finalized state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


class MemoryBankError(RuntimeError):
    """Report invalid memory state, shape, mutation, or serialization."""


class MemoryBank:
    """Collect CPU patch features and freeze them for inference.

    Args:
        metadata: JSON-compatible category and configuration provenance.
        normalized: Whether stored features were L2-normalized.
    """

    def __init__(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        normalized: bool = True,
    ) -> None:
        self.metadata = dict(metadata or {})
        self.normalized = normalized
        self._chunks: list[Tensor] = []
        self._features: Tensor | None = None
        self._feature_dimension: int | None = None
        self.original_size = 0

    @property
    def finalized(self) -> bool:
        """Return whether the memory has entered immutable inference state."""

        return self._features is not None

    @property
    def size(self) -> int:
        """Return finalized rows or accumulated rows."""

        if self._features is not None:
            return self._features.shape[0]
        return sum(chunk.shape[0] for chunk in self._chunks)

    @property
    def feature_dimension(self) -> int:
        """Return the known feature dimension.

        Raises:
            MemoryBankError: If no features have been added.
        """

        if self._feature_dimension is None:
            raise MemoryBankError("Memory feature dimension is unknown before the first add.")
        return self._feature_dimension

    @property
    def memory_bytes(self) -> int:
        """Return tensor storage size in bytes."""

        return (
            self.size
            * self.feature_dimension
            * (self.features.element_size() if self.finalized else self._chunks[0].element_size())
        )

    @property
    def features(self) -> Tensor:
        """Return a detached clone of finalized CPU features.

        Raises:
            MemoryBankError: If finalization has not completed.
        """

        if self._features is None:
            raise MemoryBankError("Memory bank must be finalized before reading features.")
        return self._features.clone()

    def add(self, features: Tensor) -> None:
        """Append a finite ``[N, D]`` feature batch before finalization."""

        if self.finalized:
            raise MemoryBankError("Finalized source memory is immutable; add() is rejected.")
        if features.ndim != 2 or min(features.shape) <= 0:
            raise MemoryBankError(
                f"Memory features have shape {tuple(features.shape)}; expected non-empty [N, D]."
            )
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise MemoryBankError("Memory features must be finite floating-point values.")
        dimension = features.shape[1]
        if self._feature_dimension is not None and dimension != self._feature_dimension:
            raise MemoryBankError(
                f"Memory feature dimension {dimension} does not match {self._feature_dimension}."
            )
        self._feature_dimension = dimension
        copied = features.detach().to(device="cpu").contiguous().clone()
        self._chunks.append(copied)
        self.original_size += copied.shape[0]

    def accumulated_features(self) -> Tensor:
        """Return a CPU copy of all pre-finalization feature chunks."""

        if self.finalized:
            raise MemoryBankError("Accumulated features are unavailable after memory finalization.")
        if not self._chunks:
            raise MemoryBankError("Memory bank has no accumulated features.")
        return torch.cat(self._chunks, dim=0).clone()

    def finalize(self, features: Tensor | None = None) -> None:
        """Freeze accumulated or explicitly compressed features.

        Args:
            features: Optional compressed matrix derived from accumulated rows.

        Raises:
            MemoryBankError: If already finalized, empty, or dimensionally invalid.
        """

        if self.finalized:
            raise MemoryBankError("Memory bank is already finalized.")
        if not self._chunks:
            raise MemoryBankError("Cannot finalize an empty memory bank.")
        candidate = (
            self.accumulated_features()
            if features is None
            else features.detach().to(device="cpu").contiguous()
        )
        if (
            candidate.ndim != 2
            or candidate.shape[0] == 0
            or candidate.shape[1] != self.feature_dimension
            or not torch.isfinite(candidate).all()
        ):
            raise MemoryBankError(
                f"Final memory shape {tuple(candidate.shape)} is invalid for feature "
                f"dimension {self.feature_dimension}."
            )
        self._features = candidate.clone()
        self._chunks.clear()

    def to(self, device: torch.device | str) -> Tensor:
        """Return a copied finalized tensor on the requested device."""

        return self.features.to(device=device)

    def save(self, path: Path) -> None:
        """Save tensor-only state suitable for weights-only loading."""

        if not self.finalized:
            raise MemoryBankError("Cannot save a memory bank before finalization.")
        try:
            metadata_bytes = json.dumps(
                self.metadata,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise MemoryBankError(f"Memory metadata must be JSON serializable: {exc}") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": self.features,
                "normalized": torch.tensor(int(self.normalized), dtype=torch.int8),
                "original_size": torch.tensor(self.original_size, dtype=torch.int64),
                "metadata_utf8": torch.tensor(list(metadata_bytes), dtype=torch.uint8),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryBank:
        """Load trusted tensor-only memory state with pickle object loading disabled."""

        if not path.is_file():
            raise MemoryBankError(f"Memory bank file does not exist: {path}.")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise MemoryBankError(f"Cannot safely load memory bank {path}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("features"), Tensor):
            raise MemoryBankError(f"Memory bank {path} lacks a tensor 'features' field.")
        normalized = payload.get("normalized")
        original_size = payload.get("original_size")
        metadata_utf8 = payload.get("metadata_utf8")
        if (
            not isinstance(normalized, Tensor)
            or not isinstance(original_size, Tensor)
            or not isinstance(metadata_utf8, Tensor)
        ):
            raise MemoryBankError(f"Memory bank {path} has incomplete tensor metadata.")
        try:
            stored_metadata = json.loads(bytes(metadata_utf8.tolist()).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise MemoryBankError(
                f"Memory bank {path} has invalid encoded metadata: {exc}"
            ) from exc
        if not isinstance(stored_metadata, dict):
            raise MemoryBankError(f"Memory bank {path} metadata is not a mapping.")
        bank = cls(
            metadata=stored_metadata if metadata is None else metadata,
            normalized=bool(normalized.item()),
        )
        bank.add(payload["features"])
        bank.original_size = int(original_size.item())
        bank.finalize()
        return bank
