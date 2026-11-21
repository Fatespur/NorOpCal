# Reproduction

Two reproduction levels, by design:

| Level | What it does | Needs dataset? | Needs GPU? | Status in G2 |
|---|---|---|---|---|
| **1 — Frozen-artifact verification** | recomputes paper numbers from the shipped frozen score caches and compares to sealed artifacts at tolerance ≤ 1e-9 | no | no | `VERIFIED` (PARITY_PASS) |
| **2 — Full end-to-end** | re-runs detector scoring + calibration chain from raw MVTec AD 2 | yes | yes for EfficientAD | orchestrator built; `END_TO_END_FULL_RUN_NOT_EXECUTED_IN_G2` |

---

## Level 1 — Fast frozen verification

```bash
python scripts/verify_frozen_results.py            # full: 60 summary cells + 15 bootstrap cells
python scripts/verify_frozen_results.py --quick    # constants + principal B4/K=8 cells
```

What it does (no training, no downloads, CPU-only, minutes):

1. checks frozen constants (λ=0.25, Q50=0.015890181064605713) against the
   sealed spec AND the declarative protocol record;
2. verifies cache byte-integrity against the migration manifest;
3. proves deployable-path purity (policy threshold invariant under anomaly
   perturbation);
4. recomputes per-entity PB7-P1 TEST results with the ORIGINAL implementation
   and compares every aggregate metric vs `E_TEST_DETECTOR_SUMMARY.csv`;
5. full mode also replays the B=5000 category-aware bootstrap (seed 20260820)
   vs `E_TEST_BOOTSTRAP.csv`.

Verdict is printed as `PARITY_PASS` / `PARITY_FAIL` (exit code 0/1); reports
land in `outputs/verification/` (git-ignored).

G2 recorded result: **PARITY_PASS — max deviation ≤ 1e-9 tolerance; bootstrap
cells exact.**

Related exports:

```bash
python scripts/reproduce_tables.py    # Tables I-V + S1 from sealed CSVs -> outputs/verification/tables/
python scripts/reproduce_figures.py   # re-render Fig1-4 -> outputs/figures/regenerated/
```

Figure parity levels (honest): quantitative figure PDFs are
`CONTENT_IDENTICAL_AFTER_DATE_NORM`; SVG/PNG are DATA_PARITY (byte differences
from font/metadata only); Fig1 SVG is DATA_PARITY (later script revision).
No visual-identity claim is made.

---

## Level 2 — Full end-to-end reproduction

```bash
# validate environment + print the frozen stage chain (nothing executes)
python scripts/reproduce_main.py --mode dry-run

# include scoring prerequisites in the check
python scripts/reproduce_main.py --mode dry-run --with-scoring \
  --data-root D:/data/mvtec_ad_2 --work-root D:/work/efficientad_pb6

# execute inside an ISOLATED scratch copy (sealed artifacts never overwritten)
python scripts/reproduce_main.py --mode execute --scratch-dir D:/tmp/noropcal_repro \
  --data-root D:/data/mvtec_ad_2 --work-root D:/work/efficientad_pb6 --with-scoring
```

The chain is the sealed order PB7-P0 → PB7-P1 → PB7-P2 → PB8 (22 stage
scripts). Each stage self-verifies its inputs' hashes.

Stage status legend:

| Stage group | G2 status |
|---|---|
| Cache-generation provenance (`r03_*`, `p0_scores.py`, `pb6/d2_*`) | `REQUIRES_DATASET`, `REQUIRES_GPU` (EfficientAD) |
| Calibration/evaluation chain (PB7-P0..PB8) | runnable offline from caches; `NOT_RUN_IN_G2` end-to-end (orchestration dry-run validated) |
| Statistics & tables | `VERIFIED` via Level 1 + `reproduce_tables.py` |

Environment variables (see `configs/examples/dataset_roots.example.yaml`):
`NOROPCAL_MVTEC_AD2_ROOT`, `NOROPCAL_PB6_WORK_ROOT`,
`NOROPCAL_AEBAD_S_ROOT` (legacy AeBAD scripts only).

## Tests

```bash
pytest                                # fixture unit/integration tests
pytest -m parity                      # frozen-parity smoke suite
```

Known pre-existing failure: see `docs/KNOWN_ISSUES.md` (KI-1) — inherited,
not caused by, and intentionally not masked by G2.
