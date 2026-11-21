"""Atomic state-and-audit controller for SAPCA."""

from __future__ import annotations

from shiftad.adaptation.v2.audit import SapcaAuditError, V2AuditLog
from shiftad.adaptation.v2.contracts import (
    ActuationResult,
    FrozenSourceObservation,
    V2AuditRecord,
    canonical_hash,
)
from shiftad.adaptation.v2.state import ProtectedCausalStateStore


class AuditAndSafetyController:
    """Commit causal state and audit before exposing a method result."""

    def __init__(self, state: ProtectedCausalStateStore, audit_log: V2AuditLog) -> None:
        self.state = state
        self.audit_log = audit_log

    def commit(
        self,
        observation: FrozenSourceObservation,
        result: ActuationResult,
        *,
        proposal_hash: str | None,
        certificate_hash: str | None,
        fixture_only: bool,
        formal_evidence: bool = False,
    ) -> V2AuditRecord:
        """Atomically commit state and audit, rolling back and disabling on failure."""
        snapshot = self.state.snapshot()
        try:
            event = self.state.commit(
                sequence_id=observation.sequence_id,
                source_score=observation.source_score,
                decision=result.reason,
                certificate_hash=certificate_hash,
            )
            observation_binding = canonical_hash(
                {
                    "category": observation.category,
                    "sequence_id": observation.sequence_id,
                    "source_score": observation.source_score,
                    "source_threshold": observation.source_threshold,
                    "weak_view_scores": observation.weak_view_scores,
                    "source_anchor_hashes": observation.source_anchor_hashes,
                }
            )
            record = V2AuditRecord(
                category=observation.category,
                sequence_id=observation.sequence_id,
                source_observation_hash=observation_binding,
                proposal_hash=proposal_hash,
                certificate_hash=certificate_hash,
                state_event=event,
                acted=result.acted,
                reason=result.reason,
                previous_hash=self.audit_log.previous_hash,
                fixture_only=fixture_only,
                formal_evidence=formal_evidence,
            )
            self.audit_log.append(record)
            return record
        except Exception as exc:
            self.state.restore_and_disable(
                snapshot, sequence_id=observation.sequence_id, reason=type(exc).__name__
            )
            if isinstance(exc, SapcaAuditError):
                raise
            raise SapcaAuditError(f"SAPCA transaction failed closed: {exc}") from exc
