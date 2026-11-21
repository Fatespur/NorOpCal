"""Versioned target-domain adaptation state with snapshots and rollback.

The target state is the only mutable adaptation object. It stores the bounded
quarantine buffer, the target prototype, maturity counters, and a version
number. Every mutation bumps the version; snapshots allow exact rollback to a
previously validated version (protocol Section 5.3, 5.5).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shiftad.adaptation.base import (
    PROTOCOL_VERSION,
    AdaptationError,
    SchemaRejectionError,
)

_EMPTY_PROTOTYPE: tuple[float, ...] = ()


@dataclass
class TargetState:
    """Mutable, versioned target-domain adaptation state.

    Attributes:
        version: Monotonic state version; 0 is the initial (empty) state.
        quarantine: Bounded list of accepted candidate feature rows.
        prototype: Current target prototype (empty tuple before maturity).
        accepted_updates: Number of promoted batch updates so far.
        rejected_updates: Number of gate-rejected candidates observed.
        quarantine_capacity: Maximum quarantine length (frozen policy value).
        promotion_min_count: Minimum accepted rows before promotion.
        category: Category this state belongs to.
    """

    version: int = 0
    quarantine: list[tuple[float, ...]] = field(default_factory=list)
    prototype: tuple[float, ...] = _EMPTY_PROTOTYPE
    accepted_updates: int = 0
    rejected_updates: int = 0
    quarantine_capacity: int = 32
    promotion_min_count: int = 4
    category: str = ""

    def add_candidate(self, features: tuple[float, ...]) -> None:
        """Add one accepted candidate to the quarantine (bounded).

        Raises:
            SchemaRejectionError: If the feature row is malformed or its
                dimension differs from the established row dimension.
        """
        if not features or any(not isinstance(v, (int, float)) for v in features):
            raise SchemaRejectionError("Quarantine candidates must be non-empty numeric rows.")
        expected_dim = self._feature_dimension()
        if expected_dim is not None and len(features) != expected_dim:
            raise SchemaRejectionError(
                f"Candidate dimension {len(features)} != established dimension {expected_dim}."
            )
        if len(self.quarantine) >= self.quarantine_capacity:
            # Deterministic bounded eviction: drop the oldest row.
            self.quarantine.pop(0)
        self.quarantine.append(features)
        self.version += 1

    def _feature_dimension(self) -> int | None:
        """Return the established quarantine dimension (None when empty)."""
        if self.quarantine:
            return len(self.quarantine[0])
        if self.prototype:
            return len(self.prototype)
        return None

    def promote(self) -> bool:
        """Promote the quarantine into the prototype when mature.

        The promoted prototype is the component-wise mean of the quarantine
        rows. Returns True when a promotion occurred (and clears the buffer),
        False when the quarantine has not reached ``promotion_min_count``.

        Raises:
            SchemaRejectionError: If the quarantine rows have inconsistent
                dimensions.
        """
        if len(self.quarantine) < self.promotion_min_count:
            return False
        dims = {len(row) for row in self.quarantine}
        if len(dims) != 1:
            raise SchemaRejectionError(
                f"Inconsistent quarantine dimensions {sorted(dims)}; cannot promote."
            )
        dim = dims.pop()
        means: list[float] = []
        for i in range(dim):
            total = sum(row[i] for row in self.quarantine)
            means.append(total / len(self.quarantine))
        self.prototype = tuple(means)
        self.quarantine.clear()
        self.accepted_updates += 1
        self.version += 1
        return True

    def reject(self) -> None:
        """Record one gate rejection (deterministic counter)."""
        self.rejected_updates += 1
        self.version += 1

    def is_mature(self) -> bool:
        """Return whether the target prototype has been promoted at least once."""
        return self.accepted_updates > 0 and self.prototype != _EMPTY_PROTOTYPE

    def snapshot(self) -> TargetStateSnapshot:
        """Capture an immutable snapshot of the current state."""
        return TargetStateSnapshot(
            version=self.version,
            quarantine=tuple(self.quarantine),
            prototype=self.prototype,
            accepted_updates=self.accepted_updates,
            rejected_updates=self.rejected_updates,
            quarantine_capacity=self.quarantine_capacity,
            promotion_min_count=self.promotion_min_count,
            category=self.category,
        )

    def restore(self, snapshot: TargetStateSnapshot) -> None:
        """Restore exactly the state captured in the snapshot.

        Raises:
            AdaptationError: If the snapshot belongs to a different category.
        """
        if snapshot.category != self.category:
            raise AdaptationError(
                f"Cannot restore snapshot of category {snapshot.category!r} "
                f"into state of category {self.category!r}."
            )
        self.version = snapshot.version
        self.quarantine = list(snapshot.quarantine)
        self.prototype = snapshot.prototype
        self.accepted_updates = snapshot.accepted_updates
        self.rejected_updates = snapshot.rejected_updates
        self.quarantine_capacity = snapshot.quarantine_capacity
        self.promotion_min_count = snapshot.promotion_min_count

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state deterministically."""
        return {
            "schema_version": "1.0",
            "protocol_version": PROTOCOL_VERSION,
            "version": self.version,
            "quarantine": [list(row) for row in self.quarantine],
            "prototype": list(self.prototype),
            "accepted_updates": self.accepted_updates,
            "rejected_updates": self.rejected_updates,
            "quarantine_capacity": self.quarantine_capacity,
            "promotion_min_count": self.promotion_min_count,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetState:
        """Reconstruct a state from :meth:`to_dict` output.

        Raises:
            SchemaRejectionError: If required fields are missing, malformed,
                or the schema version is unsupported.
        """
        if data.get("schema_version") != "1.0":
            raise SchemaRejectionError(
                f"Unsupported TargetState schema_version {data.get('schema_version')!r}."
            )
        try:
            state = cls(
                version=int(data["version"]),
                quarantine=[tuple(float(v) for v in row) for row in data["quarantine"]],
                prototype=tuple(float(v) for v in data["prototype"]),
                accepted_updates=int(data["accepted_updates"]),
                rejected_updates=int(data["rejected_updates"]),
                quarantine_capacity=int(data["quarantine_capacity"]),
                promotion_min_count=int(data["promotion_min_count"]),
                category=str(data["category"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaRejectionError(f"Malformed TargetState mapping: {exc}") from exc
        return state

    def state_hash(self) -> str:
        """Return a canonical SHA256 of the state (excludes timestamps)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, path: Path) -> None:
        """Atomically persist the state as JSON (write, fsync, then rename).

        Raises:
            AdaptationError: If persistence fails.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent, delete=False
            ) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                tmp_path = Path(handle.name)
            tmp_path.replace(path)
        except OSError as exc:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise AdaptationError(f"Cannot persist TargetState to {path}: {exc}") from exc

    @classmethod
    def load(cls, path: Path) -> TargetState:
        """Load a state persisted by :meth:`save`.

        Raises:
            AdaptationError: If the file is missing or malformed.
        """
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AdaptationError(f"Cannot load TargetState from {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise AdaptationError(f"TargetState file {path} must contain a JSON object.")
        return cls.from_dict(data)


@dataclass(frozen=True)
class TargetStateSnapshot:
    """Immutable capture of a target state at a specific version."""

    version: int
    quarantine: tuple[tuple[float, ...], ...]
    prototype: tuple[float, ...]
    accepted_updates: int
    rejected_updates: int
    quarantine_capacity: int
    promotion_min_count: int
    category: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the snapshot with the same schema as :meth:`TargetState.to_dict`."""
        return {
            "schema_version": "1.0",
            "protocol_version": PROTOCOL_VERSION,
            "version": self.version,
            "quarantine": [list(row) for row in self.quarantine],
            "prototype": list(self.prototype),
            "accepted_updates": self.accepted_updates,
            "rejected_updates": self.rejected_updates,
            "quarantine_capacity": self.quarantine_capacity,
            "promotion_min_count": self.promotion_min_count,
            "category": self.category,
        }

    def state_hash(self) -> str:
        """Return the same canonical SHA256 as the restored state."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetStateSnapshot:
        """Reconstruct a snapshot from :meth:`to_dict` output.

        Raises:
            SchemaRejectionError: If required fields are missing, malformed,
                or the schema version is unsupported.
        """
        if data.get("schema_version") != "1.0":
            raise SchemaRejectionError(
                f"Unsupported snapshot schema_version {data.get('schema_version')!r}."
            )
        try:
            return cls(
                version=int(data["version"]),
                quarantine=tuple(tuple(float(v) for v in row) for row in data["quarantine"]),
                prototype=tuple(float(v) for v in data["prototype"]),
                accepted_updates=int(data["accepted_updates"]),
                rejected_updates=int(data["rejected_updates"]),
                quarantine_capacity=int(data["quarantine_capacity"]),
                promotion_min_count=int(data["promotion_min_count"]),
                category=str(data["category"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaRejectionError(f"Malformed snapshot mapping: {exc}") from exc

    def save(self, path: Path) -> None:
        """Atomically persist the snapshot as JSON (write, fsync, rename).

        Raises:
            AdaptationError: If persistence fails.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent, delete=False
            ) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                tmp_path = Path(handle.name)
            tmp_path.replace(path)
        except OSError as exc:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise AdaptationError(f"Cannot persist snapshot to {path}: {exc}") from exc

    @classmethod
    def load(cls, path: Path) -> TargetStateSnapshot:
        """Load a snapshot persisted by :meth:`save`.

        Raises:
            AdaptationError: If the file is missing or malformed.
        """
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AdaptationError(f"Cannot load snapshot from {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise AdaptationError(f"Snapshot file {path} must contain a JSON object.")
        return cls.from_dict(data)
