"""Append-only adaptation audit log with hash-chain integrity.

Every gate decision, target-state update, snapshot, and rollback must be
recorded in the audit log (protocol Section 4.4, 5.5). The log is append-only:
records are never edited or deleted, and each record carries the hash of the
previous record so tampering is detectable.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shiftad.adaptation.base import (
    PROTOCOL_VERSION,
    AdaptationError,
    GateDecision,
    ShiftEstimate,
)

AUDIT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class AuditRecord:
    """One immutable audit entry for a single stream transaction.

    Attributes:
        sequence_id: Monotonic transaction index within the stream.
        sample_id: Anonymous sample identifier (never a label or mask).
        decision: The gate decision for this transaction.
        shift_estimate: The shift estimate seen by the policy.
        state_version_before: Target state version before the transaction.
        state_version_after: Target state version after the transaction.
        previous_hash: Hash of the previous record ("" for the first record).
        action: One of "decision", "promotion", "rollback", "snapshot",
            "disable", "enable".
    """

    sequence_id: int
    sample_id: str
    decision: GateDecision
    shift_estimate: ShiftEstimate | None
    state_version_before: int
    state_version_after: int
    action: str
    previous_hash: str
    self_hash: str = ""

    def _to_dict(self, *, include_self_hash: bool) -> dict[str, Any]:
        """Serialize the record deterministically (excludes timestamps)."""
        data: dict[str, Any] = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "sequence_id": self.sequence_id,
            "sample_id": self.sample_id,
            "decision": self.decision.to_dict(),
            "shift_estimate": (
                None if self.shift_estimate is None else self.shift_estimate.to_dict()
            ),
            "state_version_before": self.state_version_before,
            "state_version_after": self.state_version_after,
            "action": self.action,
            "previous_hash": self.previous_hash,
        }
        if include_self_hash:
            data["self_hash"] = self.self_hash
        return data

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record including its self-hash."""
        return self._to_dict(include_self_hash=True)

    def content_hash(self) -> str:
        """Canonical SHA256 of the record content excluding ``self_hash``."""
        canonical = json.dumps(
            self._to_dict(include_self_hash=False), sort_keys=True, separators=(",", ":")
        )
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def record_hash(self) -> str:
        """Canonical SHA256 of the full serialized record (incl. self_hash)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditLog:
    """Append-only audit log backed by a JSONL file.

    Records are appended with flush + fsync before ``append`` returns, so a
    committed record survives process crashes (crash-safe append). A truncated
    final line left by an interrupted write is recovered on load by dropping
    that incomplete record and recording the recovery event.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._records: list[AuditRecord] = []
        self._truncated_record_recovered = False
        self._file_existed_at_init = self.path.is_file()
        if self._file_existed_at_init:
            self._load()

    @property
    def truncated_record_recovered(self) -> bool:
        """True when loading dropped a truncated final record (crash recovery)."""
        return self._truncated_record_recovered

    def _load(self) -> None:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AdaptationError(f"Cannot read audit log {self.path}: {exc}") from exc
        lines = [line for line in raw.splitlines() if line.strip()]
        needs_rewrite = False
        if raw and not raw.endswith("\n"):
            # A crash between the JSON write and the newline write leaves a
            # final line that is valid JSON but was never committed (no
            # terminating newline). Drop it and rewrite so later appends
            # cannot merge into it.
            needs_rewrite = True
            lines = lines[:-1]
        for index, line in enumerate(lines):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                if index == len(lines) - 1:
                    # A truncated final record from an interrupted write is
                    # recoverable: drop it (it was never committed) and clear
                    # the remnant from disk so later appends cannot merge
                    # into it.
                    needs_rewrite = True
                    continue
                raise AdaptationError(f"Corrupt audit line {index} in {self.path}: {exc}") from exc
            self._records.append(self._record_from_dict(data))
        if needs_rewrite:
            self._truncated_record_recovered = True
            self._rewrite_disk()

    def _rewrite_disk(self) -> None:
        """Atomically rewrite the log from the committed in-memory records.

        Called after crash recovery to remove the truncated remnant so that
        subsequent appends start on a clean line boundary.

        Raises:
            AdaptationError: If the rewrite fails.
        """
        payload = "".join(
            json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
            for record in self._records
        )
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                tmp_path = Path(handle.name)
            tmp_path.replace(self.path)
        except OSError as exc:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise AdaptationError(
                f"Cannot rewrite audit log {self.path} after recovery: {exc}"
            ) from exc

    @staticmethod
    def _record_from_dict(data: dict[str, Any]) -> AuditRecord:
        try:
            decision_data = data["decision"]
            shift_data = data["shift_estimate"]
            return AuditRecord(
                sequence_id=int(data["sequence_id"]),
                sample_id=str(data["sample_id"]),
                decision=GateDecision(
                    accepted=bool(decision_data["accepted"]),
                    reason_code=str(decision_data["reason_code"]),
                    checks=dict(decision_data.get("checks", {})),
                ),
                shift_estimate=(
                    None
                    if shift_data is None
                    else ShiftEstimate(
                        magnitude=float(shift_data["magnitude"]),
                        components=dict(shift_data.get("components", {})),
                        confidence=float(shift_data.get("confidence", 1.0)),
                        window_occupancy=float(shift_data.get("window_occupancy", 1.0)),
                        in_envelope=bool(shift_data.get("in_envelope", True)),
                        envelope_status=str(shift_data.get("envelope_status", "IN_ENVELOPE")),
                    )
                ),
                state_version_before=int(data["state_version_before"]),
                state_version_after=int(data["state_version_after"]),
                action=str(data["action"]),
                previous_hash=str(data["previous_hash"]),
                self_hash=str(data.get("self_hash", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AdaptationError(f"Malformed audit record: {exc}") from exc

    def append(
        self,
        *,
        sample_id: str,
        decision: GateDecision,
        shift_estimate: ShiftEstimate | None,
        state_version_before: int,
        state_version_after: int,
        action: str = "decision",
    ) -> AuditRecord:
        """Append one record and persist it immediately.

        Raises:
            AdaptationError: If the log cannot be appended to.
        """
        previous_hash = self._records[-1].record_hash() if self._records else ""
        record = AuditRecord(
            sequence_id=len(self._records),
            sample_id=sample_id,
            decision=decision,
            shift_estimate=shift_estimate,
            state_version_before=state_version_before,
            state_version_after=state_version_after,
            action=action,
            previous_hash=previous_hash,
        )
        # Bind the record to its own content so that tampering with the last
        # record (which nothing else references) is still detectable.
        record = AuditRecord(
            sequence_id=record.sequence_id,
            sample_id=record.sample_id,
            decision=record.decision,
            shift_estimate=record.shift_estimate,
            state_version_before=record.state_version_before,
            state_version_after=record.state_version_after,
            action=record.action,
            previous_hash=record.previous_hash,
            self_hash=record.content_hash(),
        )
        line = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise AdaptationError(f"Cannot append to audit log {self.path}: {exc}") from exc
        self._records.append(record)
        return record

    @property
    def records(self) -> tuple[AuditRecord, ...]:
        """Return the in-memory records (immutable view)."""
        return tuple(self._records)

    def verify_integrity(self) -> list[str]:
        """Replay the hash chain, self-hashes, and the on-disk lines.

        An empty list means the log is intact and complete. Tampering with
        any record — including the last one — is detected because every
        record is bound to its own content hash and every on-disk line must
        re-serialize to the same canonical form as its in-memory record.
        """
        findings: list[str] = []
        expected_previous = ""
        for index, record in enumerate(self._records):
            if record.previous_hash != expected_previous:
                findings.append(
                    f"Record {index}: previous_hash {record.previous_hash!r} != "
                    f"expected {expected_previous!r}."
                )
            expected_previous = record.record_hash()
            if record.self_hash and record.self_hash != record.content_hash():
                findings.append(
                    f"Record {index}: self_hash {record.self_hash!r} does not match "
                    f"its content hash {record.content_hash()!r}."
                )
        if not self.path.is_file():
            if self._file_existed_at_init or self._records:
                # The log existed at load time, or committed records exist in
                # memory: an append-only audit file must never disappear.
                findings.append(f"Audit log file {self.path} is missing.")
        else:
            on_disk_lines = [
                line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            if len(on_disk_lines) != len(self._records):
                findings.append(
                    f"On-disk line count {len(on_disk_lines)} != in-memory record count "
                    f"{len(self._records)}."
                )
            for index, (record, line) in enumerate(zip(self._records, on_disk_lines, strict=False)):
                expected_line = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
                if line != expected_line:
                    findings.append(
                        f"Record {index}: on-disk line does not match the record content."
                    )
        return findings
