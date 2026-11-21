"""Phase 6 core types: immutable source anchor, typed estimates, gate
decisions, fused scores, and the label-blind adaptation policy contract.

This module is the Phase 6-B interface foundation. It defines the boundary
types required by ``docs/PHASE6_PROTOCOL.md`` Section 4 (adaptation/evaluator
API boundary):

- ``SourceAnchor``: immutable source-domain safety anchor (backbone hash,
  source-memory hash, and source feature statistics);
- ``ShiftEstimate``: typed shift estimator output;
- ``GateDecision``: typed, machine-readable gate outcome;
- ``FusedScore``: source/target/consistency fusion output;
- ``AdaptationPolicy``: label-blind adaptation contract.

The adaptation object never receives labels, masks, defect types, future
samples, or evaluator outcomes (protocol Section 4.2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = "phase6_v1.0"

#: Rejection reason codes frozen by the protocol (Section 5.2).
REASON_SCORE_LIMIT = "SCORE_LIMIT"
REASON_CONSISTENCY = "CONSISTENCY"
REASON_SUSPICIOUS_AREA = "SUSPICIOUS_AREA"
REASON_INPUT_QUALITY = "INPUT_QUALITY"
REASON_SHIFT_MEANINGFUL = "SHIFT_MEANINGFUL"
REASON_IDENTITY = "IDENTITY"
REASON_RATE_LIMIT = "RATE_LIMIT"
REASON_ADAPTATION_OFF = "ADAPTATION_OFF"
REASON_ROLLBACK = "ROLLBACK"

#: Marker for deterministic fixture implementations used only in software
#: safety tests; forbidden as formal evidence (AGENTS.md Section 6).
FIXTURE_SCOPE = "synthetic_fixture_acceptance_only"

#: Marker for production implementations eligible as formal evidence
#: (frozen protocol parameters, full validation, no fixture shortcuts).
PRODUCTION_SCOPE = "production_formal_eligible"


class AdaptationError(RuntimeError):
    """Report invalid adaptation state, leakage, or artifact handling."""


class SchemaRejectionError(AdaptationError):
    """Report that an input violates the adaptation schema contract."""


class LeakageError(AdaptationError):
    """Report a forbidden information flow (labels, masks, or future data)."""


class SourceAnchorError(AdaptationError):
    """Report an invalid or mismatched immutable source anchor."""


@dataclass(frozen=True)
class SourceAnchor:
    """Immutable source-domain safety anchor.

    Attributes:
        category: MVTec AD category name.
        backbone_hash: SHA256 of the frozen backbone weights.
        source_memory_hash: SHA256 of the immutable source memory ``M_s``.
        mu_s: Source-normal feature mean (1D sequence).
        sigma_s: Source-normal feature covariance (2D sequence of rows).
        protocol_version: Protocol version string the anchor was built under.
    """

    category: str
    backbone_hash: str
    source_memory_hash: str
    mu_s: tuple[float, ...]
    sigma_s: tuple[tuple[float, ...], ...]
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize the anchor to a plain mapping (deterministic)."""
        return {
            "category": self.category,
            "backbone_hash": self.backbone_hash,
            "source_memory_hash": self.source_memory_hash,
            "mu_s": list(self.mu_s),
            "sigma_s": [list(row) for row in self.sigma_s],
            "protocol_version": self.protocol_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceAnchor:
        """Reconstruct an anchor from :meth:`to_dict` output.

        Raises:
            SourceAnchorError: If required fields are missing or malformed.
        """
        try:
            return cls(
                category=str(data["category"]),
                backbone_hash=str(data["backbone_hash"]),
                source_memory_hash=str(data["source_memory_hash"]),
                mu_s=tuple(float(v) for v in data["mu_s"]),
                sigma_s=tuple(tuple(float(v) for v in row) for row in data["sigma_s"]),
                protocol_version=str(data["protocol_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SourceAnchorError(f"Malformed SourceAnchor mapping: {exc}") from exc


@dataclass(frozen=True)
class ShiftEstimate:
    """Typed shift-estimator output.

    Attributes:
        magnitude: Scalar shift magnitude (``D_shift``).
        components: Per-component contributions keyed by component name.
        confidence: Confidence in the estimate in [0, 1].
        window_occupancy: Fraction of the window with valid observations.
        in_envelope: True when the operating envelope accepts the estimate.
        envelope_status: Machine-readable envelope verdict string.
    """

    magnitude: float
    components: dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    window_occupancy: float = 1.0
    in_envelope: bool = True
    envelope_status: str = "IN_ENVELOPE"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the estimate deterministically."""
        return {
            "magnitude": self.magnitude,
            "components": dict(sorted(self.components.items())),
            "confidence": self.confidence,
            "window_occupancy": self.window_occupancy,
            "in_envelope": self.in_envelope,
            "envelope_status": self.envelope_status,
        }


@dataclass(frozen=True)
class GateDecision:
    """Typed reliability-gate outcome.

    Attributes:
        accepted: True when the candidate may enter quarantine.
        reason_code: Machine-readable code (``REASON_*`` constant, or
            ``"ACCEPTED"`` when accepted).
        checks: Per-condition verdicts keyed by condition name.
    """

    accepted: bool
    reason_code: str
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the decision deterministically."""
        return {
            "accepted": self.accepted,
            "reason_code": self.reason_code,
            "checks": dict(sorted(self.checks.items())),
        }


@dataclass(frozen=True)
class FusedScore:
    """Source/target/consistency fusion output.

    Attributes:
        source_score: Score from the immutable source memory.
        target_score: Score from the target prototype (None before maturity).
        consistency_score: Weak-view consistency score (None when unused).
        fused_score: Final fused score.
        weights: Applied weights keyed by component name.
    """

    source_score: float
    target_score: float | None
    consistency_score: float | None
    fused_score: float
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the score deterministically."""
        return {
            "source_score": self.source_score,
            "target_score": self.target_score,
            "consistency_score": self.consistency_score,
            "fused_score": self.fused_score,
            "weights": dict(sorted(self.weights.items())),
        }


class AdaptationPolicy(ABC):
    """Label-blind adaptation contract (protocol Section 4.2).

    A policy observes one sample at a time together with the immutable
    source anchor and the current shift estimate. It must never receive
    labels, masks, defect types, future samples, or evaluator outcomes.
    """

    @abstractmethod
    def observe(
        self,
        sample: Any,
        anchor: SourceAnchor,
        shift_estimate: ShiftEstimate,
    ) -> GateDecision:
        """Process one sample and decide whether it may update target state.

        Args:
            sample: Current sample payload (features and/or image stats).
            anchor: Immutable source anchor.
            shift_estimate: Current shift estimate.

        Returns:
            A deterministic gate decision.

        Raises:
            LeakageError: If the payload contains forbidden truth fields.
            SchemaRejectionError: If the payload violates the schema.
        """

    @abstractmethod
    def is_adaptation_enabled(self) -> bool:
        """Return whether target-state adaptation is currently enabled."""
