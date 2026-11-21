"""Causal pce_v1 construction from source-only references."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any

from shiftad.adaptation.image_stats import IMAGE_STATISTIC_NAMES
from shiftad.adaptation.v2.contracts import (
    CertificateInputView,
    CorrectionProposal,
    FrozenSourceObservation,
    ProposalCertificateEvidence,
    ProposalContextView,
    SapcaContractError,
    SourceOnlyParameterPackage,
)


def _mad(values: Sequence[float], center: float) -> float:
    return float(median(abs(value - center) for value in values))


class ProposalCertificateEvidenceProvider:
    """Build dependency-separated PCE views using history ending at t-1."""

    def proposal_view(
        self,
        observation: FrozenSourceObservation,
        image_stats: Mapping[str, float],
        committed_score_history: Sequence[float],
        package: SourceOnlyParameterPackage | None,
        *,
        allow_fixture_package: bool = False,
        execution_permit: Any | None = None,
    ) -> ProposalContextView | None:
        """Build the proposal-only view or return None when fail-closed."""
        if package is None or package.provenance.category != observation.category:
            return None
        if package.provenance.fixture_only and not allow_fixture_package:
            return None
        if not package.production_usable and not allow_fixture_package and execution_permit is None:
            # Cycle #31R: a formal package may produce proposal evidence only
            # under a validated formal execution permit.
            return None
        if len(committed_score_history) < 2:
            return None
        stats = tuple(float(image_stats[name]) for name in IMAGE_STATISTIC_NAMES)
        if not all(math.isfinite(value) for value in stats):
            raise SapcaContractError("Image statistics must be finite.")
        standardized = tuple(
            (value - mean) / std
            for value, mean, std in zip(
                stats, package.image_stat_mean, package.image_stat_std, strict=True
            )
        )
        margin = (observation.source_score - package.source_score_mean) / package.source_score_std
        history = tuple(float(value) for value in committed_score_history)
        if not all(math.isfinite(value) for value in history):
            raise SapcaContractError("Committed history must be finite.")
        center = float(median(history))
        history_mad = _mad(history, center)
        if history_mad <= 0:
            return None
        values = (
            *standardized,
            margin,
            center - package.history_median_reference,
            history_mad / package.history_mad_reference,
        )
        return ProposalContextView(observation.category, observation.sequence_id, values)

    def certificate_view(
        self,
        observation: FrozenSourceObservation,
        proposal: CorrectionProposal,
        package: SourceOnlyParameterPackage,
        *,
        state_hash: str,
        weak_view_proposed_deltas: Sequence[float],
        anchor_envelope_violation: float,
        causal_state_influence: float,
    ) -> CertificateInputView:
        """Build the independent action-response view after a proposal exists."""
        weak_scores = observation.weak_view_scores
        score_disagreement = max(weak_scores) - min(weak_scores)
        proposed = tuple(float(value) for value in weak_view_proposed_deltas)
        if not proposed or not all(math.isfinite(value) for value in proposed):
            raise SapcaContractError("Weak-view proposed deltas must be finite and non-empty.")
        proposed_disagreement = max(proposed) - min(proposed)
        normalized_delta = proposal.requested_delta / package.correction_cap
        proposed_score = observation.source_score - proposal.requested_delta
        response = proposed_score - (package.source_threshold - package.epsilon_s)
        values = (
            normalized_delta,
            response,
            score_disagreement,
            proposed_disagreement,
            float(anchor_envelope_violation),
            float(causal_state_influence),
        )
        return CertificateInputView(
            category=observation.category,
            sequence_id=observation.sequence_id,
            values=values,
            source_anchor_hashes=observation.source_anchor_hashes,
            state_hash=state_hash,
        )

    @staticmethod
    def assemble(
        proposal_view: ProposalContextView,
        certificate_view: CertificateInputView,
        *,
        history_count: int,
    ) -> ProposalCertificateEvidence:
        """Assemble the one fixed evidence class from its isolated views."""
        return ProposalCertificateEvidence(
            proposal_view=proposal_view,
            certificate_view=certificate_view,
            history_count=history_count,
            validity_flags=("SOURCE_ONLY", "CAUSAL_T_MINUS_1", "FINITE"),
        )
