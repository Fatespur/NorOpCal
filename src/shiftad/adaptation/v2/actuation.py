"""Policy-free source-anchored bounded SAPCA actuation."""

from __future__ import annotations

import math

from shiftad.adaptation.v2.authorization import ValidatedFormalExecutionPermitV2
from shiftad.adaptation.v2.contracts import (
    ActuationResult,
    CorrectionProposal,
    FrozenSourceObservation,
    SafetyCertificate,
    SourceOnlyParameterPackage,
    SourceOnlyParameterPackageV2,
)


class SourceAnchoredBoundedActuator:
    """Apply only a valid certificate or return exact source behavior."""

    @staticmethod
    def fallback(observation: FrozenSourceObservation, reason: str) -> ActuationResult:
        """Return the exact source score and map."""
        return ActuationResult(
            source_score=observation.source_score,
            method_score=observation.source_score,
            source_map=observation.source_map,
            acted=False,
            reason=reason,
        )

    def apply(
        self,
        observation: FrozenSourceObservation,
        proposal: CorrectionProposal | None,
        certificate: SafetyCertificate | None,
        package: SourceOnlyParameterPackage | None,
        *,
        allow_fixture_package: bool = False,
    ) -> ActuationResult:
        """Validate all bindings and apply a downward bounded correction."""
        if proposal is None:
            return self.fallback(observation, "NO_PROPOSAL")
        if certificate is None:
            return self.fallback(observation, "CERTIFICATE_VETO")
        if package is None:
            return self.fallback(observation, "MISSING_PARAMETER_PACKAGE")
        if package.provenance.fixture_only and not allow_fixture_package:
            return self.fallback(observation, "FIXTURE_PACKAGE_REJECTED")
        if not package.production_usable and not allow_fixture_package:
            return self.fallback(observation, "UNAUTHORIZED_PARAMETER_PACKAGE")
        if certificate.proposal_hash != proposal.binding_hash():
            return self.fallback(observation, "PROPOSAL_HASH_MISMATCH")
        if certificate.proposal_id != proposal.proposal_id:
            return self.fallback(observation, "PROPOSAL_ID_MISMATCH")
        if (
            certificate.category != observation.category
            or proposal.category != observation.category
        ):
            return self.fallback(observation, "CATEGORY_MISMATCH")
        if (
            certificate.sequence_id != observation.sequence_id
            or proposal.sequence_id != observation.sequence_id
        ):
            return self.fallback(observation, "SEQUENCE_MISMATCH")
        if observation.sequence_id > min(
            certificate.expires_after_sequence, proposal.expires_after_sequence
        ):
            return self.fallback(observation, "EXPIRED")
        if certificate.source_anchor_hashes != observation.source_anchor_hashes:
            return self.fallback(observation, "ANCHOR_MISMATCH")
        if certificate.parameter_package_hash != package.package_hash:
            return self.fallback(observation, "PARAMETER_HASH_MISMATCH")
        delta = certificate.approved_delta
        if not math.isfinite(delta) or delta < 0:
            return self.fallback(observation, "INVALID_DELTA")
        if delta > proposal.requested_delta or delta > package.correction_cap:
            return self.fallback(observation, "CAP_VIOLATION")
        method_score = max(
            package.source_threshold - package.epsilon_s,
            observation.source_score - delta,
        )
        if not math.isfinite(method_score) or method_score > observation.source_score:
            return self.fallback(observation, "DIRECTION_VIOLATION")
        return ActuationResult(
            source_score=observation.source_score,
            method_score=method_score,
            source_map=observation.source_map,
            acted=delta > 0 and method_score < observation.source_score,
            reason="CERTIFIED_ACTION" if delta > 0 else "CERTIFIED_ZERO_DELTA",
            proposal_id=proposal.proposal_id,
            certificate_id=certificate.certificate_id,
        )


class SourceAnchoredBoundedActuatorV2:
    """Threshold-resolved v2 actuator (Cycle #29 U8 / instruction 7.10).

    Identical fail-closed semantics to the legacy actuator, except the source
    threshold is resolved from the v2 package's seed-keyed authority by the
    caller (``source_threshold`` argument); the package itself carries no
    category-only scalar threshold.
    """

    @staticmethod
    def fallback(observation: FrozenSourceObservation, reason: str) -> ActuationResult:
        """Return the exact source score and map."""
        return ActuationResult(
            source_score=observation.source_score,
            method_score=observation.source_score,
            source_map=observation.source_map,
            acted=False,
            reason=reason,
        )

    def apply(
        self,
        observation: FrozenSourceObservation,
        proposal: CorrectionProposal | None,
        certificate: SafetyCertificate | None,
        package: SourceOnlyParameterPackageV2 | None,
        *,
        source_threshold: float,
        allow_fixture_package: bool = False,
        execution_permit: ValidatedFormalExecutionPermitV2 | None = None,
    ) -> ActuationResult:
        """Validate all bindings and apply a downward bounded correction."""
        if proposal is None:
            return self.fallback(observation, "NO_PROPOSAL")
        if certificate is None:
            return self.fallback(observation, "CERTIFICATE_VETO")
        if package is None:
            return self.fallback(observation, "MISSING_PARAMETER_PACKAGE")
        if package.provenance.fixture_only and not allow_fixture_package:
            return self.fallback(observation, "FIXTURE_PACKAGE_REJECTED")
        if not package.production_usable and not allow_fixture_package and execution_permit is None:
            # Cycle #31R: a formal (non-fixture) package may act only under a
            # validated formal execution permit. Direct actuator invocation
            # without a valid permit stays fail-closed exact-source.
            return self.fallback(observation, "UNAUTHORIZED_PARAMETER_PACKAGE")
        if certificate.proposal_hash != proposal.binding_hash():
            return self.fallback(observation, "PROPOSAL_HASH_MISMATCH")
        if certificate.proposal_id != proposal.proposal_id:
            return self.fallback(observation, "PROPOSAL_ID_MISMATCH")
        if (
            certificate.category != observation.category
            or proposal.category != observation.category
            or package.provenance.category != observation.category
        ):
            return self.fallback(observation, "CATEGORY_MISMATCH")
        if (
            certificate.sequence_id != observation.sequence_id
            or proposal.sequence_id != observation.sequence_id
        ):
            return self.fallback(observation, "SEQUENCE_MISMATCH")
        if observation.sequence_id > min(
            certificate.expires_after_sequence, proposal.expires_after_sequence
        ):
            return self.fallback(observation, "EXPIRED")
        if certificate.source_anchor_hashes != observation.source_anchor_hashes:
            return self.fallback(observation, "ANCHOR_MISMATCH")
        if certificate.parameter_package_hash != package.compute_content_hash():
            return self.fallback(observation, "PARAMETER_HASH_MISMATCH")
        if not math.isfinite(source_threshold) or source_threshold <= 0:
            return self.fallback(observation, "INVALID_DELTA")
        delta = certificate.approved_delta
        if not math.isfinite(delta) or delta < 0:
            return self.fallback(observation, "INVALID_DELTA")
        if delta > proposal.requested_delta or delta > package.correction_cap:
            return self.fallback(observation, "CAP_VIOLATION")
        method_score = max(
            source_threshold - package.epsilon_s,
            observation.source_score - delta,
        )
        if not math.isfinite(method_score) or method_score > observation.source_score:
            return self.fallback(observation, "DIRECTION_VIOLATION")
        return ActuationResult(
            source_score=observation.source_score,
            method_score=method_score,
            source_map=observation.source_map,
            acted=delta > 0 and method_score < observation.source_score,
            reason="CERTIFIED_ACTION" if delta > 0 else "CERTIFIED_ZERO_DELTA",
            proposal_id=proposal.proposal_id,
            certificate_id=certificate.certificate_id,
        )
