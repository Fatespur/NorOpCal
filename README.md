# NorOpCal

**Normal-Only Selective Operating-Point Calibration**

NorOpCal selectively and conservatively recalibrates the **operating point**
(the decision threshold) of a **frozen** industrial anomaly detector under
**acquisition shift**, using only a handful of **known-normal target
observations** — no target anomalies, no labels, no masks, and no
modification of the detector itself.

> Manuscript currently under review at *IEEE Transactions on Instrumentation
> and Measurement* (IEEE TIM).

## What problem does it solve?

Under acquisition shift, a deployed detector's frozen source threshold no
longer matches the target imaging condition: false negatives can explode.
NorOpCal treats this as a **threshold-only** calibration problem:

1. compute the **median-shift signal** `Δm` between source-normal and K
   target-normal score distributions;
2. apply a preregistered **correctability gate**: adapt iff `Δm ≥ Q50`
   (frozen `Q50 = 0.015890181064605713`); otherwise **abstain**;
3. on gate pass, apply the **conservative bounded actuator**
   `τ_t = τ_s + λ·Δm` with frozen `λ = 0.25`.

The detector's parameters, representation, score function, and ranking are
never touched. Abstention is a designed outcome, not a failure.

## Verify in one minute (no dataset, no GPU)

The repository ships the frozen image-score caches that every paper number is
computed from, so you can verify the main result offline:

```bash
pip install -r requirements.txt && pip install -e .
python scripts/verify_frozen_results.py --quick    # PARITY_PASS expected
python scripts/verify_frozen_results.py            # full parity sweep
```

Full reproduction from raw MVTec AD 2 (dataset required by license):

```bash
python scripts/reproduce_main.py --mode dry-run --with-scoring \
    --data-root /path/to/mvtec_ad_2
```

See `docs/REPRODUCTION.md` for the two-level design and honest status labels.

## Main Results (frozen; source-mapped)

Source of truth:
`reports/pb8_publication_benchmark/TABLE_1_MAIN_BENCHMARK.csv` (sealed;
hash-bound in `_audit/migration_manifest.csv`), summarized in
`PUBLICATION_GROUND_TRUTH.md §5` (`docs/reference/`). Principal setting:
DINOv2, K=8, 10 TEST-eligible units (fabric + walnuts).

| Policy | ΔBA pp [95% CI] | ΔFNR pp | Harm % | Coverage % |
|---|---|---|---|---|
| NoAdapt (`τ_s`) | 0 (ref; BA 0.5411) | 0 | 0 | 0 |
| Bounded-only (λ=0.25, no gate) | +3.64 [+3.52, +3.76] | −0.16 | 3.67 | 100 |
| Gate-only (λ=1.0, Q50) | +4.25 [+3.96, +4.55] | +7.49 | 37.67 | 40.33 |
| **NorOpCal (λ=0.25, Q50)** | **+3.23 [+2.70, +3.76]** | **+0.24** | **3.67** | **40.33** |

Scope facts carried by the sealed evidence: detector-specific verdicts
(DINOv2 useful-safe-selectable; PatchCore actuator-mismatch; EfficientAD no
headroom); K=4 unsafe; FULL not monotonically safer; residual harm localized
(walnuts). Regenerate any table yourself:
`python scripts/reproduce_tables.py`.

## Repository structure

```text
src/noropcal/             thin public facade (delegates to frozen impl)
src/shiftad/              scientific implementation: detectors (DINOv2,
                          PatchCore, EfficientAD), data adapters, backbones
tools/                    sealed experiment drivers PB6/PB7-*/PB8*/RR*
reports/                  sealed stage artifacts incl. frozen score caches
results/revision_reserve/ post-submission reserve analyses RR0-RR6
scripts/                  Level-1 verification + Level-2 orchestration
assets/figures/           final main-text figures (+generation provenance)
configs/                  dataset/detector configs, frozen protocol record
docs/                     METHOD / DATASETS / EXPERIMENT_PROTOCOL /
                          REPRODUCTION / KNOWN_ISSUES / reference/
tests/                    fixture-based unit tests + parity smoke suite
_audit/                   G1/G2 migration & engineering provenance
```

Naming note: the Python package keeps its historical codename `shiftad`;
`import noropcal` is provided as a compatibility facade. Renaming the package
internally is deferred engineering work — scientific fidelity outranks
cosmetics.

## Installation

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Python 3.11+. GPU only needed for re-scoring images from raw data.

## Datasets

All paper numbers use **MVTec AD 2** (8 categories, 39 category-condition
units). Raw datasets are NOT distributed; see `docs/DATASETS.md`.

## Documentation

- `docs/METHOD.md` — method summary and scope
- `docs/EXPERIMENT_PROTOCOL.md` — frozen protocol and firewall
- `docs/REPRODUCTION.md` — Level-1 / Level-2 details and status labels
- `docs/KNOWN_ISSUES.md` — honest open issues
- `THIRD_PARTY_NOTICES.md` — backbone/model attribution

## Citation

See `CITATION.cff` (no DOI/volume until acceptance).

## License

MIT — inherited verbatim from the parent research repository decision.
