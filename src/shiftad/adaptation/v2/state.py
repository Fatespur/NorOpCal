"""Bounded category-isolated causal state for SAPCA."""

from __future__ import annotations

from dataclasses import dataclass

from shiftad.adaptation.v2.contracts import ProtectedStateEvent, SapcaContractError, canonical_hash


@dataclass(frozen=True)
class ProtectedStateSnapshot:
    """Immutable exact snapshot of permitted V2 state."""

    category: str
    last_sequence_id: int
    source_score_history: tuple[float, ...]
    used_certificate_hashes: tuple[str, ...]
    failure_count: int
    rollback_count: int
    disabled: bool
    disable_reason: str
    provenance_hashes: tuple[str, ...]

    def state_hash(self) -> str:
        """Return the canonical state hash."""
        return canonical_hash(self)


class ProtectedCausalStateStore:
    """Own only bounded score summaries, counters, and provenance."""

    def __init__(
        self,
        category: str,
        *,
        max_history: int = 32,
        provenance_hashes: tuple[str, ...] = (),
    ) -> None:
        if not category or max_history < 2:
            raise SapcaContractError("State requires a category and max_history >= 2.")
        self._max_history = max_history
        self._snapshot = ProtectedStateSnapshot(
            category=category,
            last_sequence_id=0,
            source_score_history=(),
            used_certificate_hashes=(),
            failure_count=0,
            rollback_count=0,
            disabled=False,
            disable_reason="",
            provenance_hashes=tuple(sorted(provenance_hashes)),
        )

    @property
    def category(self) -> str:
        """Return the isolated category."""
        return self._snapshot.category

    @property
    def disabled(self) -> bool:
        """Return whether actuation is fail-closed disabled."""
        return self._snapshot.disabled

    @property
    def history(self) -> tuple[float, ...]:
        """Return committed source scores ending at the last transaction."""
        return self._snapshot.source_score_history

    @property
    def rollback_count(self) -> int:
        """Return the rollback count."""
        return self._snapshot.rollback_count

    def snapshot(self) -> ProtectedStateSnapshot:
        """Return the immutable current snapshot."""
        return self._snapshot

    def state_hash(self) -> str:
        """Return the current canonical state hash."""
        return self._snapshot.state_hash()

    def commit(
        self,
        *,
        sequence_id: int,
        source_score: float,
        decision: str,
        certificate_hash: str | None = None,
    ) -> ProtectedStateEvent:
        """Atomically commit one causal observation after actuation."""
        before = self._snapshot
        if sequence_id <= before.last_sequence_id:
            raise SapcaContractError("Sequence identifiers must increase monotonically.")
        history = (*before.source_score_history, float(source_score))[-self._max_history :]
        certificates = before.used_certificate_hashes
        if certificate_hash is not None:
            if certificate_hash in certificates:
                raise SapcaContractError("Certificate replay detected.")
            certificates = (*certificates, certificate_hash)[-self._max_history :]
        after = ProtectedStateSnapshot(
            category=before.category,
            last_sequence_id=sequence_id,
            source_score_history=history,
            used_certificate_hashes=certificates,
            failure_count=before.failure_count,
            rollback_count=before.rollback_count,
            disabled=before.disabled,
            disable_reason=before.disable_reason,
            provenance_hashes=before.provenance_hashes,
        )
        self._snapshot = after
        return ProtectedStateEvent(
            category=after.category,
            sequence_id=sequence_id,
            pre_state_hash=before.state_hash(),
            post_state_hash=after.state_hash(),
            decision=decision,
            rollback_count=after.rollback_count,
        )

    def restore_and_disable(
        self, snapshot: ProtectedStateSnapshot, *, sequence_id: int, reason: str
    ) -> ProtectedStateEvent:
        """Restore an exact snapshot, increment rollback, and disable actuation."""
        if snapshot.category != self.category:
            raise SapcaContractError("Cannot restore a cross-category snapshot.")
        before_hash = self.state_hash()
        restored = ProtectedStateSnapshot(
            category=snapshot.category,
            last_sequence_id=snapshot.last_sequence_id,
            source_score_history=snapshot.source_score_history,
            used_certificate_hashes=snapshot.used_certificate_hashes,
            failure_count=snapshot.failure_count + 1,
            rollback_count=snapshot.rollback_count + 1,
            disabled=True,
            disable_reason=reason,
            provenance_hashes=snapshot.provenance_hashes,
        )
        self._snapshot = restored
        return ProtectedStateEvent(
            category=restored.category,
            sequence_id=sequence_id,
            pre_state_hash=before_hash,
            post_state_hash=restored.state_hash(),
            decision="ROLLBACK_DISABLED",
            rollback_count=restored.rollback_count,
        )

    def reset(self, category: str, provenance_hashes: tuple[str, ...]) -> None:
        """Deterministically reset category state without reusing prior history."""
        if category != self.category:
            raise SapcaContractError("Cross-category reset requires a new store.")
        self._snapshot = ProtectedStateSnapshot(
            category=category,
            last_sequence_id=0,
            source_score_history=(),
            used_certificate_hashes=(),
            failure_count=0,
            rollback_count=0,
            disabled=False,
            disable_reason="",
            provenance_hashes=tuple(sorted(provenance_hashes)),
        )
