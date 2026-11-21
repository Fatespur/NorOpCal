"""Causal stream engine and evaluator-only observation boundary.

The stream engine enforces the Phase 6 causal rule (protocol Section 4.1):
at stream time ``t`` adaptation may read only the current sample and state
derived from samples ``1..t``. Truth (labels, masks) lives on the evaluator
side and is joined only after each decision/update transaction commits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

from shiftad.adaptation.audit import AuditLog, AuditRecord
from shiftad.adaptation.base import (
    REASON_ADAPTATION_OFF,
    REASON_ROLLBACK,
    GateDecision,
    LeakageError,
    SchemaRejectionError,
    ShiftEstimate,
    SourceAnchor,
)
from shiftad.adaptation.safety import (
    REASON_ROLLBACK_COUNT,
    SafetyMonitor,
    require_valid_snapshot,
)
from shiftad.adaptation.state import TargetState, TargetStateSnapshot

#: Engine-side hard cap on automatic rollbacks: even a custom safety monitor
#: that never reports ROLLBACK_COUNT cannot drive the engine past this many
#: automatic rollbacks (persistent contamination must end in fail-closed
#: disablement, not an unbounded rollback loop).
MAX_AUTOMATIC_ROLLBACKS = 3


@dataclass(frozen=True)
class StreamTransaction:
    """Immutable record of one processed stream sample.

    Attributes:
        sequence_id: 1-based position in the stream.
        sample_id: Anonymous sample identifier.
        decision: The gate decision produced for this sample.
        state_version_before: State version before the transaction.
        state_version_after: State version after the transaction.
        audit: The audit record appended for this transaction.
    """

    sequence_id: int
    sample_id: str
    decision: GateDecision
    state_version_before: int
    state_version_after: int
    audit: AuditRecord


class StreamEvaluator(Protocol):
    """Evaluator-only truth holder (protocol Section 4.3).

    The evaluator owns truth and exposes it only after a decision/update
    transaction has committed. The adaptation object never receives a
    reference to this interface.
    """

    def join(self, sample_id: str) -> dict[str, Any] | None:
        """Return evaluator truth for a sample (label, mask) or None."""
        ...


@dataclass
class CausalStreamEngine:
    """Drives one deterministic stream with strict causality.

    Attributes:
        anchor: Immutable source anchor shared by all transactions.
        state: Mutable target state (the only mutable adaptation object).
        audit_log: Append-only audit log.
        enabled: Adaptation enablement (False is byte-equivalent to A0).
        warmup_samples: Number of leading samples during which accepted
            decisions do not mutate state (protocol warmup; the audit trail
            still records them).
    """

    anchor: SourceAnchor
    state: TargetState
    audit_log: AuditLog
    enabled: bool = True
    warmup_samples: int = 0
    safety_monitor: SafetyMonitor | None = None
    _transactions: list[StreamTransaction] = field(default_factory=list)
    _last_validated_snapshot: TargetStateSnapshot | None = field(
        default=None, repr=False, compare=False
    )
    _rollback_count: int = 0
    _accepted_count: int = 0
    _total_count: int = 0

    @property
    def rollback_count(self) -> int:
        """Number of automatic rollbacks executed by the safety monitor."""
        return self._rollback_count

    @property
    def last_validated_snapshot(self) -> TargetStateSnapshot | None:
        """Last snapshot validated by the safety monitor (or None)."""
        return self._last_validated_snapshot

    #: Truth-bearing dictionary keys rejected by the leakage firewall. The
    #: check is a substring match so variant keys (e.g. ``labels``) cannot
    #: bypass it. Note: this is intentionally fail-closed and may over-reject
    #: feature names containing these substrings (e.g. ``anomaly_score``).
    FORBIDDEN_TRUTH_KEYS: ClassVar[tuple[str, ...]] = (
        "label",
        "mask",
        "defect",
        "truth",
        "anomaly",
    )

    @classmethod
    def _reject_truth_keys(cls, payload: Any) -> None:
        """Recursively reject truth-bearing keys in any adaptation payload.

        Args:
            payload: Sample, decision mapping, shift-estimate mapping, or any
                nested mapping.

        Raises:
            LeakageError: If any key (at any nesting depth) contains a
                forbidden truth substring.
        """
        if isinstance(payload, dict):
            for key, value in payload.items():
                lowered = str(key).lower()
                for forbidden in cls.FORBIDDEN_TRUTH_KEYS:
                    if forbidden in lowered:
                        raise LeakageError(
                            "Adaptation payloads must not carry labels, masks, defect "
                            f"types, or truth; forbidden key {key!r}."
                        )
                cls._reject_truth_keys(value)
        elif isinstance(payload, (list, tuple)):
            for item in payload:
                cls._reject_truth_keys(item)

    def process(
        self,
        sample_id: str,
        sample: Any,
        decision: GateDecision,
        shift_estimate: ShiftEstimate | None = None,
        *,
        promote: bool = True,
    ) -> StreamTransaction:
        """Process one sample under the causal rule.

        The caller supplies a pre-computed gate decision (label-blind); this
        engine applies state mutation, audit append, and evaluator-visible
        bookkeeping. The evaluator may join truth only after this method
        returns.

        Args:
            sample_id: Anonymous sample identifier.
            sample: Current sample payload (never labels or masks).
            decision: Gate decision for this sample.
            shift_estimate: Optional shift estimate for the audit trail.
            promote: Whether a promotion attempt is allowed after an accepted
                decision.

        Returns:
            The committed transaction.

        Raises:
            LeakageError: If ``sample``, ``decision``, or ``shift_estimate``
                contains a forbidden truth field (recursively, including
                ``GateDecision.checks`` and ``ShiftEstimate.components``).
            SchemaRejectionError: If ``sample`` cannot be converted to a
                numeric feature row.
        """
        # Leakage firewall: every adaptation payload is validated recursively
        # before any state mutation or audit logging. Truth-bearing keys are
        # rejected in the sample, the decision checks, the shift components,
        # and any nested mapping.
        self._reject_truth_keys(sample)
        self._reject_truth_keys(decision.to_dict())
        if shift_estimate is not None:
            self._reject_truth_keys(shift_estimate.to_dict())
        # Fail-closed dimension check BEFORE any state mutation: a prototype
        # whose dimension differs from the source anchor would silently
        # truncate the safety-monitor distance computation.
        if (
            self.safety_monitor is not None
            and self.state.is_mature()
            and len(self.state.prototype) != len(self.anchor.mu_s)
        ):
            raise SchemaRejectionError(
                "Target prototype dimension does not match the source anchor; "
                "refusing to process samples under a mismatched configuration."
            )
        version_before = self.state.version
        in_warmup = len(self._transactions) < self.warmup_samples
        if decision.accepted and self.enabled and not in_warmup:
            try:
                values = [float(v) for v in sample]
            except (TypeError, ValueError) as exc:
                raise SchemaRejectionError(
                    f"Accepted sample must be a numeric feature row, got {sample!r}."
                ) from exc
            if not all(math.isfinite(v) for v in values):
                raise SchemaRejectionError(
                    "Accepted sample must contain only finite values "
                    "(NaN/inf would poison the prototype and blind the safety "
                    "monitor's distance check)."
                )
            row = tuple(values)
            self.state.add_candidate(row)
            if promote:
                self.state.promote()
        elif not decision.accepted and self.enabled and not in_warmup:
            self.state.reject()
        # When adaptation is disabled (fail-closed) or the stream is still in
        # warmup, the state is frozen: accepted decisions do not mutate it.
        # Warmup decisions are still appended to the audit trail.
        version_after = self.state.version
        action = "decision"
        audit = self.audit_log.append(
            sample_id=sample_id,
            decision=decision,
            shift_estimate=shift_estimate,
            state_version_before=version_before,
            state_version_after=version_after,
            action=action,
        )
        transaction = StreamTransaction(
            sequence_id=len(self._transactions) + 1,
            sample_id=sample_id,
            decision=decision,
            state_version_before=version_before,
            state_version_after=version_after,
            audit=audit,
        )
        self._transactions.append(transaction)
        self._total_count += 1
        if decision.accepted:
            self._accepted_count += 1
        # Automatic safety orchestration: after every committed transaction,
        # the safety monitor checks adaptation health and a fail-closed
        # verdict restores the last validated snapshot (or disables
        # adaptation when no validated snapshot exists yet).
        self._monitor_and_rollback()
        return transaction

    def _monitor_and_rollback(self) -> None:
        """Run the automatic safety-monitor orchestration (label-blind).

        When ``safety_monitor`` is set and adaptation is enabled, the monitor
        receives deterministic health signals (target/source distance,
        acceptance rate, rollback count) and its verdict drives the engine:

        - ``ok``: the current state becomes the new last validated snapshot;
        - ``fail_closed`` with a validated snapshot: exact rollback to it,
          rollback count incremented (audited as ``rollback``);
        - ``fail_closed`` without a snapshot: adaptation disabled
          (audited as ``disable``).

        The monitor never receives labels, masks, or future samples.
        """
        if not self.enabled or self.safety_monitor is None:
            return
        distance: float | None = None
        if self.state.is_mature():
            proto = self.state.prototype
            mu_s = self.anchor.mu_s
            if len(proto) != len(mu_s):
                raise SchemaRejectionError(
                    "Target prototype dimension does not match the source anchor; "
                    "refusing to compute a silently-truncated distance."
                )
            distance = float(sum((p - m) ** 2 for p, m in zip(proto, mu_s, strict=True)) ** 0.5)
        acceptance_rate = (
            self._accepted_count / self._total_count if self._total_count > 0 else None
        )
        verdict = self.safety_monitor.check(
            self.state,
            target_source_distance=distance,
            acceptance_rate=acceptance_rate,
            score_variance_ratio=None,
            rollback_count=self._rollback_count,
        )
        if verdict.ok:
            self._last_validated_snapshot = self.state.snapshot()
            return
        if verdict.fail_closed:
            if (
                verdict.reason == REASON_ROLLBACK_COUNT
                or self._rollback_count >= MAX_AUTOMATIC_ROLLBACKS
            ):
                # Rollback budget exhausted: repeated rollbacks indicate
                # persistent contamination; fail closed by disabling
                # adaptation instead of rolling back forever.
                self.disable_adaptation()
            elif self._last_validated_snapshot is not None:
                self.rollback_to(self._last_validated_snapshot)
                self._rollback_count += 1
            else:
                self.disable_adaptation()

    @property
    def transactions(self) -> tuple[StreamTransaction, ...]:
        """Return committed transactions in stream order (immutable view)."""
        return tuple(self._transactions)

    def disable_adaptation(self) -> AuditRecord:
        """Fail-closed disablement; records the event in the audit log."""
        self.enabled = False
        return self.audit_log.append(
            sample_id="",
            decision=GateDecision(accepted=False, reason_code=REASON_ADAPTATION_OFF),
            shift_estimate=None,
            state_version_before=self.state.version,
            state_version_after=self.state.version,
            action="disable",
        )

    def rollback_to(self, snapshot: Any, *, audit: AuditLog | None = None) -> AuditRecord:
        """Restore the target state to a validated snapshot (exact rollback).

        Args:
            snapshot: A ``TargetStateSnapshot`` captured earlier.
            audit: Optional separate audit log for the rollback event; when
                None the engine's own audit log records the event.

        Returns:
            The audit record for the rollback event.

        Raises:
            AdaptationError: If the snapshot is None or belongs to another
                category.
        """
        require_valid_snapshot(snapshot)
        version_before = self.state.version
        self.state.restore(snapshot)
        version_after = self.state.version
        target_audit = audit if audit is not None else self.audit_log
        return target_audit.append(
            sample_id="",
            decision=GateDecision(accepted=False, reason_code=REASON_ROLLBACK),
            shift_estimate=None,
            state_version_before=version_before,
            state_version_after=version_after,
            action="rollback",
        )

    @classmethod
    def from_paths(
        cls,
        anchor: SourceAnchor,
        state: TargetState,
        audit_path: Path,
        *,
        enabled: bool = True,
    ) -> CausalStreamEngine:
        """Construct an engine with a fresh (or existing) audit log on disk."""
        return cls(
            anchor=anchor,
            state=state,
            audit_log=AuditLog(audit_path),
            enabled=enabled,
        )
