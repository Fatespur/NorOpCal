"""Crash-safe append-only audit persistence for SAPCA."""

from __future__ import annotations

import os
from pathlib import Path

from shiftad.adaptation.v2.contracts import V2AuditRecord, canonical_json


class SapcaAuditError(RuntimeError):
    """Report an audit persistence or chain-integrity failure."""


class V2AuditLog:
    """Append V2 records with flush/fsync before returning a result."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._records: list[V2AuditRecord] = []

    @property
    def records(self) -> tuple[V2AuditRecord, ...]:
        """Return committed records."""
        return tuple(self._records)

    @property
    def previous_hash(self) -> str:
        """Return the current chain head."""
        return self._records[-1].record_hash() if self._records else ""

    def append(self, record: V2AuditRecord) -> None:
        """Persist one correctly chained record before adding it in memory."""
        if record.previous_hash != self.previous_hash:
            raise SapcaAuditError("Audit previous-hash mismatch.")
        line = canonical_json(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise SapcaAuditError(f"Cannot persist SAPCA audit: {exc}") from exc
        self._records.append(record)

    def verify_integrity(self) -> list[str]:
        """Verify in-memory chain and exact on-disk canonical lines."""
        findings: list[str] = []
        previous = ""
        for index, record in enumerate(self._records):
            if record.previous_hash != previous:
                findings.append(f"record {index}: previous hash mismatch")
            previous = record.record_hash()
        if not self.path.is_file():
            return [*findings, "audit file missing"]
        lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line]
        if lines != [canonical_json(record) for record in self._records]:
            findings.append("on-disk audit content mismatch")
        return findings
