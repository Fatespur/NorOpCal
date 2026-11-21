"""Proposal-only SAPCA interface and non-acting production default."""

from __future__ import annotations

import math
from typing import Protocol

from shiftad.adaptation.v2.contracts import (
    CorrectionProposal,
    ProposalContextView,
    SourceOnlyParameterPackageV2,
    canonical_hash,
)


class EligibilityProposer(Protocol):
    """Consume only proposal context and optionally request a correction."""

    version: str

    def propose(self, view: ProposalContextView, *, proposal_id: str) -> CorrectionProposal | None:
        """Return a proposal or None without authorizing actuation."""
        ...


class NoOpEligibilityProposer:
    """Production default that always abstains."""

    version = "sapca_noop_proposer_v1"

    def propose(self, view: ProposalContextView, *, proposal_id: str) -> None:
        """Return no proposal for every input."""
        del view, proposal_id
        return None


class MultisignalHistoryDriftProposer:
    """Acting proposer PROPOSER_MULTISIGNAL_HISTORY_DRIFT_V1 (Cycle #21 + #29).

    Consumes only ``ProposalContextView[12]`` and a validated v2 parameter
    package. Returns a ``CorrectionProposal`` or ``None`` (no proposal ->
    exact-source fallback). Never writes state, never issues a certificate,
    and never actuates.
    """

    version = "PROPOSER_MULTISIGNAL_HISTORY_DRIFT_V1"

    def __init__(
        self,
        package: SourceOnlyParameterPackageV2,
        policy_provenance_hash: str,
        *,
        delta_scale: float = 0.5,
        delta_min: float = 0.0,
    ) -> None:
        from shiftad.adaptation.v2.contracts import SapcaContractError as _Err

        if not policy_provenance_hash:
            raise _Err("policy provenance hash must be non-empty.")
        self._package = package
        self._provenance_hash = policy_provenance_hash
        if not math.isfinite(delta_scale) or delta_scale <= 0:
            raise _Err("delta_scale must be a positive finite literal.")
        if delta_min != 0.0:
            raise _Err("delta_min must be the frozen literal 0.0.")
        self._delta_scale = delta_scale
        self._delta_min = delta_min

    def _raw_delta(self, margin: float) -> float:
        """Frozen raw requested-delta function (RAW_DELTA_V1)."""
        delta = (
            self._delta_scale * (self._package.tau_margin - margin) * self._package.source_score_std
        )
        return min(max(delta, self._delta_min), self._package.correction_cap)

    def propose(self, view: ProposalContextView, *, proposal_id: str) -> CorrectionProposal | None:
        """Evaluate eligibility and return a bound proposal or None.

        ``proposal_id`` is accepted for interface compatibility; the canonical
        proposal ID is derived internally by the frozen Cycle #21 formula
        ``SHA256(canonical_json({category, sequence_id, requested_delta,
        expires_after_sequence, proposer_version, policy_provenance_hash}))``.
        """
        del proposal_id
        values = tuple(float(value) for value in view.values)
        if not all(math.isfinite(value) for value in values):
            return None
        history_median_shift = values[10]
        history_mad_ratio = values[11]
        margin = values[9]
        stats_l2 = math.sqrt(sum(value * value for value in values[0:9]))
        if history_median_shift >= self._package.tau_shift:
            return None
        if (
            history_mad_ratio <= self._package.tau_mad_low
            or history_mad_ratio >= self._package.tau_mad
        ):
            return None
        if stats_l2 >= self._package.tau_stats_env:
            return None
        if margin >= self._package.tau_margin:
            return None
        requested_delta = self._raw_delta(margin)
        if not math.isfinite(requested_delta) or requested_delta < 0:
            return None
        if requested_delta > self._package.correction_cap:
            return None
        expires_after_sequence = view.sequence_id + 5
        computed_id = canonical_hash(
            {
                "category": view.category,
                "sequence_id": view.sequence_id,
                "requested_delta": requested_delta,
                "expires_after_sequence": expires_after_sequence,
                "proposer_version": self.version,
                "policy_provenance_hash": self._provenance_hash,
            }
        )
        return CorrectionProposal(
            proposal_id=computed_id,
            category=view.category,
            sequence_id=view.sequence_id,
            requested_delta=requested_delta,
            expires_after_sequence=expires_after_sequence,
            proposer_version=self.version,
            policy_provenance_hash=self._provenance_hash,
        )
