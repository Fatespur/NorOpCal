# PB1-DS PB6-D4 — Offline Pre-D2 Checkpoint

Date: 2026-08-19
Stage: `PB1-DS-PB6-OFFLINE-PRE-D2-CHECKPOINT`
PB6 overall status: **`PB1DS_PB6_IN_PROGRESS_BLOCKED_AT_D2_PENDING_REAL_MVTEC_AD2_IMAGES_AND_EFFICIENTAD_WEIGHTS`**

## 0. Nature of this checkpoint

This endpoint summarizes the two **authorized offline tasks** only:

- **Task A** = PB5 minimal transparency erratum repair (E0→E4).
- **Task B** = PB6-D4 two-detector shadow reproduction precheck (S0→S4).

It is **not** a PB6 scientific completion, **not** third-detector evidence, and carries **no**
authorization for PB6-D2/D3/D5+ execution. D5 remains unauthorized; D2 remains blocked.

---

## 1. Task A — PB5 transparency-label erratum repair

| Step | Result |
|---|---|
| E0 audit | Passed — scope strictly transparency reporting labels; no scientific computation site consumes the erroneous slice |
| E1 canonical recovery | Passed — `all_sources_agree = true` (C1 freeze + PB3R/PB4 model-binding order == canonical 10) |
| E2 minimal repair | Passed — `pb1ds_pb5_common.FEATURE_COLUMNS` now reads `M0.sealed_canonical_feature_columns()` |
| E3 artifact regeneration | Passed — regenerated transparency artifact; `numeric_changed = false` everywhere |
| E4 regression/reseal | Passed — reseal audits pass; erratum statement written |

**Verdict (Task A): DONE.** The co-misdrop was semantic-only: coefficient *numbers, predictions,
ΔBA, Global Harm, and the PB5 scientific terminal were never affected — only the reporting labels
upon the frozen canonical coefficient order.

**Evidence**
- `reports/pb1ds_publication/pb5_transparency_erratum/PB1DS_PB5_ERRATUM_SCOPE_AUDIT.md`
- `reports/pb1ds_publication/pb5_transparency_erratum/PB1DS_PB5_ERRATUM_CANONICAL_FEATURE_BINDING.json`
- `reports/pb1ds_publication/pb5_transparency_erratum/PB1DS_PB5_ERRATUM_DIFF.csv` (`numeric_changed == false`)
- `reports/pb1ds_publication/pb5_transparency_erratum/PB1DS_PB5_TRANSPARENCY_ERRATUM.md`
- `reports/pb1ds_publication/pb5_publication_closure/PB1DS_PB5_C_COEFFICIENT_TRANSPARENCY.csv` (regenerated)

### Required-question answers (Task A)
1. **Was the erratum really scoped to transparency labels only?** → Yes. E0 enumerated every consumer of
   `pb1ds_pb5_common.FEATURE_COLUMNS` and confirmed none is a scientific computation site (no fit / scorer /
   threshold / bootstrap / GH / operating-point uses the slice).
2. **Do coefficient numbers change?** → No. E3 diff: `numeric_changed == false` for every `(fold, index)`.
3. **Is the canonical order restored from sealed sources?** → Yes. `sealed_canonical_feature_columns()`
   requires exact ordered equality with `C0_FEATURE_COLUMNS`, else raises (no silent guess).

---

## 2. Task B — PB6-D4 two-detector shadow reproduction precheck

| Step | Result |
|---|---|
| S0 protocol binding | Passed — two-detector-only; `no_efficientad = true`; `d5_authorized = false` |
| S1 selector OOF (A/C/D) | Passed — 39 outer folds, 234 selector rows, 78-entity two-detector population |
| S1b replay (NO_ADAPT / ALWAYS_ON_QM / BC_RANK) | Passed — 234 replay rows |
| S2 reproduction audit | **EXACT** — see metrics below |
| S3 no-EfficientAD guard | Passed — structural data-domain audit, `contaminated = false` |

**Terminal (S2):**
`PB1DS_PB6_D4_SHADOW_PRECHECK_PASS_TWO_DETECTOR_PROTOCOL_EXACTLY_REPRODUCED`

| Reproduction metric (pooled across 234 entities / 6 methods) | Value |
|---|---|
| selector probability `max \|diff\|` | **1.888e-13** (tolerance 1e-9) |
| action mismatches | **0** / 234 |
| entity-level downstream `max \|diff\|` | **5.55e-17** |
| pooled ΔBA `max \|diff\|` (pp) | **3.47e-15** |
| firewall (held-out unit not read pre-seal) | all 39 folds pass |

**Verdict (Task B): DONE — the PB6 two-detector code path exactly reproduces the sealed PB2/PUB0
two-detector results.** This is *code-path validation only*; it provides no scientific claim about
EfficientAD and does not lift the D2 block.

Two genuine defects were caught and fixed by the shadow re-execution:
1. **Metric-level ΔBA used the unconditional QM delta** for abstained entities (not the active
   `ba_used - ba_before` increment). Fixed in `_to_perunit_method_frame`; after fix `max ΔBA diff = 3.5e-15`.
2. **S3 guard was a naive substring scan** that falsely flagged the binding's own
   `no_efficientad` / `this_is_not_tri_detector` assertions, the D2 blocking status string, and numeric
   substrings (e.g. "117" inside decimals). Replaced with a structural data-domain audit.

**Evidence**
- Protocol: `shadow_precheck/PB1DS_PB6_D4_SHADOW_PROTOCOL_BINDING.json`
- Selector OOF: `shadow_precheck/PB1DS_PB6_D4_SHADOW_SELECTOR_OOF.csv`
- Replay OOF: `shadow_precheck/PB1DS_PB6_D4_SHADOW_REPLAY_OOF.csv`
- Model binding (coef sha256, features): `shadow_precheck/PB1DS_PB6_D4_SHADOW_MODEL_BINDING.json`
- Entity reproduction: `shadow_precheck/PB1DS_PB6_D4_SHADOW_ENTITY_REPRODUCTION.csv`
- Metric reproduction: `shadow_precheck/PB1DS_PB6_D4_SHADOW_METRIC_REPRODUCTION.csv`
- Contamination guard: `shadow_precheck/PB1DS_PB6_D4_SHADOW_CONTAMINATION_GUARD.json`
- Terminal: `shadow_precheck/PB1DS_PB6_D4_SHADOW_TERMINAL.json`
- Audit: `shadow_precheck/PB1DS_PB6_D4_SHADOW_REPRODUCTION_AUDIT.md`

### Required-question answers (Task B)
1. **Does the PB6 code path reproduce the sealed two-detector results?** → Yes, exactly
   (terminal EXACTLY_REPRODUCED; action mismatch 0; probability diff ~1.9e-13).
2. **Is any EfficientAD placeholder / tri-detector encoding present in the shadow artifacts?** → No.
   The structural guard scans selector/replay/entity/metric data frames and model bindings for forbidden
   tokens in columns, values, and keys; result `contaminated = false`.
3. **Does re-execution alter the sealed two-detector protocol?** → No. It reuses frozen upstream
   primitives, the sealed 39-fold split, canonical-10 features, and frozen threshold/benefit/QM definitions,
   seeded identically, and writes only into the isolated `shadow_precheck/` directory.

---

## 3. Blocked / forbidden (not executed, must remain so)

- **PB6-D2** real EfficientAD training/forward — **blocked**: no MVTec AD 2 eight-category raw images and
  no legal EfficientAD teacher pretrained weights; `allow_download = false`; synthetic/simulated evidence forbidden.
- **PB6-D3** tri-detector fitting — not performed (shadow is 2×2 only).
- **PB6-D5+** tri-detector scientific analysis — **unauthorized**; `d5_authorized = false`.

## 4. Next authorized work

The next authorized step is **PB6-D2**, but it is gated on: (a) the 8-category MVTec AD 2 raw images on a
permitted path, and (b) legal EfficientAD teacher pretrained weights, both confirmed before any D2
execution. Until then PB6 remains `IN_PROGRESS_BLOCKED_AT_D2`.

## 5. Required (必答) questions — consolidated

1. **PB5 erratum scope**: transparency labels only? → Yes (E0–E4; `numeric_changed == false`).
2. **PB5 scientific validity preserved**: coefficients/predictions/ΔBA/GH/terminal unaffected → Yes.
3. **PB6 shadow reproducibility**: two-detector protocol exactly reproduced → Yes (EXACTLY_REPRODUCED).
4. **EfficientAD contamination**: none in any shadow artifact → Yes (`contaminated = false`).
5. **PB6 status**: remains `IN_PROGRESS_BLOCKED_AT_D2`; `d5_authorized = false` → Yes.