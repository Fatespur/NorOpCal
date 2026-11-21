"""Independent SAPCA certification interface and vetoing production default."""

from __future__ import annotations

import math
from typing import Protocol

from shiftad.adaptation.v2.contracts import (
    CertificateInputView,
    CorrectionProposal,
    SafetyCertificate,
    SourceOnlyParameterPackage,
    SourceOnlyParameterPackageV2,
    canonical_hash,
)


class IndependentSafetyCertifier(Protocol):
    """Consume only certificate inputs and independently veto or certify."""

    version: str

    def certify(
        self,
        view: CertificateInputView,
        proposal: CorrectionProposal,
        package: SourceOnlyParameterPackage,
    ) -> SafetyCertificate | None:
        """Return a bound certificate or None."""
        ...


class VetoingSafetyCertifier:
    """Production default that never authorizes an action."""

    version = "sapca_veto_certifier_v1"

    def certify(
        self,
        view: CertificateInputView,
        proposal: CorrectionProposal,
        package: SourceOnlyParameterPackage,
    ) -> None:
        """Veto every proposal."""
        del view, proposal, package
        return None


class WeakViewAnchorCertifier:
    """Acting certifier CERTIFIER_WEAKVIEW_ANCHOR_V1 (Cycle #21 + #29).

    Consumes only ``CertificateInputView[6]``, the exact proposal, and a
    validated v2 package. Vetoes (returns ``None``) on any missing, nonfinite,
    out-of-band, or mismatched input; otherwise returns a bound
    ``SafetyCertificate`` with ``approved_delta`` and the eleven-field
    canonical certificate ID (U7).
    """

    version = "CERTIFIER_WEAKVIEW_ANCHOR_V1"

    def __init__(
        self,
        package: SourceOnlyParameterPackageV2,
        policy_provenance_hash: str,
    ) -> None:
        from shiftad.adaptation.v2.contracts import SapcaContractError as _Err

        if not policy_provenance_hash:
            raise _Err("certifier policy provenance hash must be non-empty.")
        self._package = package
        self._policy_provenance_hash = policy_provenance_hash

    def certify(
        self,
        view: CertificateInputView,
        proposal: CorrectionProposal,
        package: SourceOnlyParameterPackageV2,
    ) -> SafetyCertificate | None:
        """Return a bound certificate or None (veto)."""
        values = tuple(float(value) for value in view.values)
        if not all(math.isfinite(value) for value in values):
            return None
        if view.category != proposal.category or view.category != package.provenance.category:
            return None
        if view.sequence_id != proposal.sequence_id:
            return None
        normalized_delta, response, weak_source, weak_delta, anchor_violation, causal = values
        if normalized_delta < 0 or normalized_delta > 1.0:
            return None
        if anchor_violation != 0.0:
            return None
        if weak_source >= self._package.tau_weak_source:
            return None
        if weak_delta >= self._package.tau_weak_delta:
            return None
        if causal >= self._package.tau_causal:
            return None
        if response <= self._package.rho_low or response >= self._package.rho_high:
            return None
        if proposal.proposer_version != "PROPOSER_MULTISIGNAL_HISTORY_DRIFT_V1":
            return None
        approved_delta = min(proposal.requested_delta, self._package.correction_cap) * 1.0
        if not math.isfinite(approved_delta) or approved_delta < 0:
            return None
        if (
            approved_delta > proposal.requested_delta
            or approved_delta > self._package.correction_cap
        ):
            return None
        expires = min(proposal.expires_after_sequence, view.sequence_id + 5)
        proposal_hash = proposal.binding_hash()
        parameter_package_hash = package.compute_content_hash()
        identity_payload = {
            "certificate_schema_version": "pce_v1",
            "proposal_id": proposal.proposal_id,
            "proposal_hash": proposal_hash,
            "category": proposal.category,
            "sequence_id": proposal.sequence_id,
            "approved_delta": approved_delta,
            "source_anchor_hashes": view.source_anchor_hashes,
            "parameter_package_hash": parameter_package_hash,
            "proposer_policy_provenance_hash": proposal.policy_provenance_hash,
            "certifier_policy_provenance_hash": self._policy_provenance_hash,
            "expires_after_sequence": expires,
        }
        certificate_id = canonical_hash(identity_payload)
        return SafetyCertificate(
            certificate_id=certificate_id,
            schema_version="pce_v1",
            proposal_hash=proposal_hash,
            proposal_id=proposal.proposal_id,
            category=proposal.category,
            sequence_id=proposal.sequence_id,
            approved_delta=approved_delta,
            source_anchor_hashes=view.source_anchor_hashes,
            parameter_package_hash=parameter_package_hash,
            policy_provenance_hash=self._policy_provenance_hash,
            certifier_version=self.version,
            expires_after_sequence=expires,
        )
