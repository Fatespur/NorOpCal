# Public Provenance Summary (sanitized)

Derived from internal sealed provenance. Machine-specific paths, personal
metadata, and internal workflow details have been redacted or excluded.
Scientific numerical content of every referenced artifact is unchanged.

## What this repository tracks and why

- `src/shiftad/` - frozen scientific implementation used for all paper
  numbers (byte-identical to the internal research baseline).
- `src/noropcal/` - thin compatibility facade delegating to that implementation.
- `tools/` - sealed experiment-chain drivers (PB6 third detector, PB7-P0/P1/P2,
  PB8 benchmark, PB8-S1 breadth audit) plus shared frozen libraries.
- `reports/` - lightweight sealed stage artifacts (tables, verdicts, manifests,
  per-entity results) backing every published number. Heavy binary score
  caches are intentionally NOT in Git; their authoritative manifest is
  `artifacts/frozen_scores_manifest.csv`.
- `scripts/` - Level-1 frozen verification (parity <= 1e-9) and Level-2
  reproduction orchestration.
- Full chain provenance: an internal migration manifest maps every tracked
  file to its origin and SHA256; a sanitized copy ships as
  `artifacts/migration_manifest.public.csv` (free-text notes removed).

## Excluded from the public tree (available privately)

- Sealed evidence files whose free text embeds historical machine-specific
  paths (asset-resolution reports, protocol bindings with absolute roots).
- Internal audit working files (scans, snapshots, gate ledgers).
- Revision-reserve experiments RR0-RR6 (private reviewer-response reserve;
  their existence is visible via `.gitignore`, contents stay private).
- Binary score caches (165 npz) - see manifest above.

## Integrity anchors

- Internal submission baseline tag: `tim-submission-baseline`
  (separate private history line; not part of this public candidate).
- Level-1 verification: `python scripts/verify_frozen_results.py`
