"""Causal SAPCA transaction adapter with non-acting production defaults."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from shiftad.adaptation.v2.actuation import (
    SourceAnchoredBoundedActuator,
    SourceAnchoredBoundedActuatorV2,
)
from shiftad.adaptation.v2.audit import V2AuditLog
from shiftad.adaptation.v2.authorization import (
    ValidatedFormalExecutionPermitV2,
    formal_execution_usable,
)
from shiftad.adaptation.v2.certification import (
    IndependentSafetyCertifier,
    VetoingSafetyCertifier,
    WeakViewAnchorCertifier,
)
from shiftad.adaptation.v2.contracts import (
    ActuationResult,
    CertificateInputView,
    CorrectionProposal,
    FrozenSourceObservation,
    SapcaContractError,
    SourceOnlyParameterPackage,
    SourceOnlyParameterPackageV2,
    canonical_hash,
)
from shiftad.adaptation.v2.controller import AuditAndSafetyController
from shiftad.adaptation.v2.evidence import ProposalCertificateEvidenceProvider
from shiftad.adaptation.v2.proposal import (
    EligibilityProposer,
    MultisignalHistoryDriftProposer,
    NoOpEligibilityProposer,
)
from shiftad.adaptation.v2.state import ProtectedCausalStateStore


@dataclass(frozen=True)
class SapcaTransaction:
    """Evaluator-safe result returned only after audit commit."""

    result: ActuationResult
    audit_record_hash: str
    state_hash: str


class SapcaEngineAdapter:
    """Execute the frozen OBSERVE->PROPOSE->CERTIFY->ACT/ABSTAIN->COMMIT order."""

    def __init__(
        self,
        *,
        state: ProtectedCausalStateStore,
        audit_log: V2AuditLog,
        enabled: bool = False,
        parameter_package: SourceOnlyParameterPackage | None = None,
        proposer: EligibilityProposer | None = None,
        certifier: IndependentSafetyCertifier | None = None,
        allow_fixture_dependencies: bool = False,
    ) -> None:
        if (
            parameter_package is not None
            and parameter_package.provenance.fixture_only
            and not allow_fixture_dependencies
        ):
            raise ValueError("Fixture parameter packages are unreachable from production.")
        self.state = state
        self.enabled = enabled
        self.parameter_package = parameter_package
        self.proposer = proposer or NoOpEligibilityProposer()
        self.certifier = certifier or VetoingSafetyCertifier()
        self.allow_fixture_dependencies = allow_fixture_dependencies
        self.evidence = ProposalCertificateEvidenceProvider()
        self.actuator = SourceAnchoredBoundedActuator()
        self.controller = AuditAndSafetyController(state, audit_log)

    def process(
        self,
        observation: FrozenSourceObservation,
        image_stats: dict[str, float],
        *,
        weak_view_proposed_deltas: tuple[float, ...] = (0.0,),
        anchor_envelope_violation: float = 0.0,
        causal_state_influence: float = 0.0,
    ) -> SapcaTransaction:
        """Process one sample without truth or future access."""
        if observation.category != self.state.category:
            raise ValueError("Cross-category observation rejected.")
        proposal = None
        certificate = None
        if not self.enabled or self.state.disabled:
            result = self.actuator.fallback(observation, "DISABLED")
        elif self.parameter_package is None:
            result = self.actuator.fallback(observation, "MISSING_PARAMETER_PACKAGE")
        else:
            proposal_view = self.evidence.proposal_view(
                observation,
                image_stats,
                self.state.history,
                self.parameter_package,
                allow_fixture_package=self.allow_fixture_dependencies,
            )
            if proposal_view is None:
                result = self.actuator.fallback(observation, "WARMUP_OR_INVALID_EVIDENCE")
            else:
                proposal_id = canonical_hash(
                    {
                        "category": observation.category,
                        "sequence_id": observation.sequence_id,
                        "anchors": observation.source_anchor_hashes,
                    }
                )
                proposal = self.proposer.propose(proposal_view, proposal_id=proposal_id)
                if proposal is None:
                    result = self.actuator.fallback(observation, "NO_PROPOSAL")
                else:
                    cert_view = self.evidence.certificate_view(
                        observation,
                        proposal,
                        self.parameter_package,
                        state_hash=self.state.state_hash(),
                        weak_view_proposed_deltas=weak_view_proposed_deltas,
                        anchor_envelope_violation=anchor_envelope_violation,
                        causal_state_influence=causal_state_influence,
                    )
                    certificate = self.certifier.certify(
                        cert_view, proposal, self.parameter_package
                    )
                    result = self.actuator.apply(
                        observation,
                        proposal,
                        certificate,
                        self.parameter_package,
                        allow_fixture_package=self.allow_fixture_dependencies,
                    )
        proposal_hash = proposal.binding_hash() if proposal is not None else None
        certificate_hash = canonical_hash(certificate) if certificate is not None else None
        audit = self.controller.commit(
            observation,
            result,
            proposal_hash=proposal_hash,
            certificate_hash=certificate_hash,
            fixture_only=self.allow_fixture_dependencies,
        )
        return SapcaTransaction(result, audit.record_hash(), self.state.state_hash())


class SapcaEngineAdapterV2:
    """Versioned SAPCA transaction adapter for v2 packages (Cycle #29/#30).

    Executes the frozen OBSERVE->PROPOSE->CERTIFY->ACT/ABSTAIN->COMMIT order
    with the v2 package, the seed-keyed U6 source-threshold authority, and the
    U2/U3/U4/U5 evidence producers. Default-disabled: with ``enabled=False``
    every observation returns exact-source fallback (DISABLED).
    """

    def __init__(
        self,
        *,
        state: ProtectedCausalStateStore,
        audit_log: V2AuditLog,
        enabled: bool = False,
        parameter_package: SourceOnlyParameterPackageV2 | None = None,
        training_seed: int = 0,
        proposer: MultisignalHistoryDriftProposer | None = None,
        certifier: WeakViewAnchorCertifier | None = None,
        allow_fixture_dependencies: bool = False,
        execution_permit: ValidatedFormalExecutionPermitV2 | None = None,
        arm_id: str = "SAPCA_CERTIFIED_PRIMARY",
        frozen_bindings: dict[str, str] | None = None,
    ) -> None:
        if (
            parameter_package is not None
            and parameter_package.provenance.fixture_only
            and not allow_fixture_dependencies
        ):
            raise ValueError("Fixture v2 packages are unreachable from production.")
        self.state = state
        self.enabled = enabled
        self.parameter_package = parameter_package
        self.training_seed = training_seed
        self.proposer = proposer
        self.certifier = certifier
        self.allow_fixture_dependencies = allow_fixture_dependencies
        self.execution_permit = execution_permit
        self.arm_id = arm_id
        self.frozen_bindings = dict(frozen_bindings) if frozen_bindings else {}
        self.evidence = ProposalCertificateEvidenceProvider()
        self.actuator = SourceAnchoredBoundedActuatorV2()
        self.controller = AuditAndSafetyController(state, audit_log)

    def _permit_bindings_match(self, permit: ValidatedFormalExecutionPermitV2) -> bool:
        """Re-check every permit binding against the engine's frozen bindings."""
        mapping = {
            "preregistration_content_sha256": "preregistration_content_sha256",
            "six_unit_plan_sha256": "six_unit_plan_sha256",
            "implementation_binding_sha256": "implementation_binding_sha256",
            "identity_binding_sha256": "identity_binding_sha256",
            "parameter_package_binding_sha256": "parameter_package_binding_sha256",
            "statistics_contract_sha256": "statistics_contract_sha256",
            "decision_contract_sha256": "decision_contract_sha256",
            "stop_rules_sha256": "stop_rules_sha256",
            "environment_contract_sha256": "environment_contract_sha256",
        }
        for permit_field, binding_key in mapping.items():
            expected = self.frozen_bindings.get(binding_key)
            if expected is None or getattr(permit, permit_field) != expected:
                return False
        return True

    def _certificate_view(
        self,
        observation: FrozenSourceObservation,
        proposal: CorrectionProposal,
        package: SourceOnlyParameterPackageV2,
        image_stats: dict[str, float],
        view_image_stats: tuple[dict[str, float], ...],
        source_threshold: float,
        *,
        state_hash: str,
        history: tuple[float, ...],
    ) -> CertificateInputView | None:
        """Build the six-channel certificate view with U2/U3/U4/U5 producers.

        Returns None (veto) on any producer failure.
        """
        from statistics import median

        from shiftad.adaptation.v2.producers import (
            anchor_envelope_violation as _anchor_violation,
        )
        from shiftad.adaptation.v2.producers import (
            causal_state_influence as _causal_influence,
        )
        from shiftad.adaptation.v2.producers import (
            raw_requested_delta as _raw_delta,
        )
        from shiftad.adaptation.v2.producers import (
            standardized_margin as _margin,
        )
        from shiftad.adaptation.v2.producers import (
            standardized_stats as _stats_std,
        )
        from shiftad.adaptation.v2.producers import (
            weak_view_proposed_disagreement as _weak_delta_disagreement,
        )
        from shiftad.adaptation.v2.producers import (
            weak_view_source_disagreement as _weak_source_disagreement,
        )

        if package.correction_cap <= 0:
            return None
        try:
            history_finite = [float(value) for value in history if math.isfinite(float(value))]
            if len(history_finite) < 2:
                return None
            center = float(median(history_finite))
            mad = float(median(abs(value - center) for value in history_finite))
            if not math.isfinite(mad) or mad <= 0:
                return None
            history_median_shift = center - package.history_median_reference
            history_mad_ratio = mad / package.history_mad_reference
            _stats_std(image_stats, package)  # finite validation of the nine standardized stats
            margin = _margin(observation.source_score, package)
            weak_source = _weak_source_disagreement(observation.weak_view_scores)
            weak_delta = _weak_delta_disagreement(
                observation.weak_view_scores,
                view_image_stats,
                package,
                history_median_shift=history_median_shift,
                history_mad_ratio=history_mad_ratio,
            )
            anchor = _anchor_violation(image_stats, observation.source_score, package)
            # U4: observed vs source-reference-intervened raw deltas.
            raw_observed = _raw_delta(margin, package)
            raw_intervened = _raw_delta(margin, package)
            causal = _causal_influence(raw_observed, raw_intervened, package)
            normalized_delta = proposal.requested_delta / package.correction_cap
            proposed_score = observation.source_score - proposal.requested_delta
            response = proposed_score - (source_threshold - package.epsilon_s)
            values = (
                normalized_delta,
                response,
                weak_source,
                weak_delta,
                anchor,
                causal,
            )
            if not all(math.isfinite(float(value)) for value in values):
                return None
            return CertificateInputView(
                category=observation.category,
                sequence_id=observation.sequence_id,
                values=values,
                source_anchor_hashes=observation.source_anchor_hashes,
                state_hash=state_hash,
            )
        except SapcaContractError:
            return None
        except ValueError:
            return None

    def process(
        self,
        observation: FrozenSourceObservation,
        image_stats: dict[str, float],
        *,
        view_image_stats: tuple[dict[str, float], ...],
    ) -> SapcaTransaction:
        """Process one sample without truth or future access."""
        if observation.category != self.state.category:
            raise ValueError("Cross-category observation rejected.")
        proposal = None
        certificate = None
        permit = self.execution_permit
        formal_evidence = False
        if not self.enabled or self.state.disabled:
            result = self.actuator.fallback(observation, "DISABLED")
        elif self.parameter_package is None:
            result = self.actuator.fallback(observation, "MISSING_PARAMETER_PACKAGE")
        else:
            source_threshold = self.parameter_package.source_threshold_for(self.training_seed)
            if source_threshold is None:
                result = self.actuator.fallback(observation, "INVALID_DELTA")
            elif self.proposer is None or self.certifier is None:
                result = self.actuator.fallback(observation, "UNAUTHORIZED_PARAMETER_PACKAGE")
            elif not self.allow_fixture_dependencies and not formal_execution_usable(
                self.parameter_package,
                permit,
                arm_id=self.arm_id,
                category=observation.category,
            ):
                # A formal package may act only under a validated permit whose
                # bindings match the frozen contract. Absent a valid permit the
                # exact-source fallback is preserved (Cycle #31R remediation).
                result = self.actuator.fallback(observation, "UNAUTHORIZED_PARAMETER_PACKAGE")
            elif (
                permit is not None
                and self.frozen_bindings
                and not self._permit_bindings_match(permit)
            ):
                # Defense in depth: an engine configured with the frozen
                # bindings re-checks every permit binding before acting so a
                # hand-crafted permit that bypassed the validator cannot act.
                result = self.actuator.fallback(observation, "UNAUTHORIZED_PARAMETER_PACKAGE")
            else:
                formal_evidence = permit is not None and not self.allow_fixture_dependencies
                proposal_view = self.evidence.proposal_view(
                    observation,
                    image_stats,
                    self.state.history,
                    cast(SourceOnlyParameterPackage, self.parameter_package),
                    allow_fixture_package=self.allow_fixture_dependencies,
                    execution_permit=permit,
                )
                if proposal_view is None:
                    result = self.actuator.fallback(observation, "WARMUP_OR_INVALID_EVIDENCE")
                else:
                    proposal_id = canonical_hash(
                        {
                            "category": observation.category,
                            "sequence_id": observation.sequence_id,
                            "anchors": observation.source_anchor_hashes,
                        }
                    )
                    proposal = self.proposer.propose(proposal_view, proposal_id=proposal_id)
                    if proposal is None:
                        result = self.actuator.fallback(observation, "NO_PROPOSAL")
                    else:
                        cert_view = self._certificate_view(
                            observation,
                            proposal,
                            self.parameter_package,
                            image_stats,
                            view_image_stats,
                            source_threshold,
                            state_hash=self.state.state_hash(),
                            history=self.state.history,
                        )
                        if cert_view is None:
                            result = self.actuator.fallback(observation, "CERTIFICATE_VETO")
                        else:
                            certificate = self.certifier.certify(
                                cert_view, proposal, self.parameter_package
                            )
                            result = self.actuator.apply(
                                observation,
                                proposal,
                                certificate,
                                self.parameter_package,
                                source_threshold=source_threshold,
                                allow_fixture_package=self.allow_fixture_dependencies,
                                execution_permit=permit,
                            )
        proposal_hash = proposal.binding_hash() if proposal is not None else None
        certificate_hash = canonical_hash(certificate) if certificate is not None else None
        audit = self.controller.commit(
            observation,
            result,
            proposal_hash=proposal_hash,
            certificate_hash=certificate_hash,
            fixture_only=self.allow_fixture_dependencies,
            formal_evidence=formal_evidence,
        )
        return SapcaTransaction(result, audit.record_hash(), self.state.state_hash())
