# PB8-S1 — AD2 K8 Evidence-Breadth Supplementary Validation: Final Report

**Phase:** FINAL
**Stage nature:** `SUPPLEMENTARY EVIDENCE-BREADTH VALIDATION`
**Terminal:** `PB8_S1_NO_LEGAL_K8_BREADTH_EXPANSION` (T3)
**Date:** 2026-08-20
**Detector scope:** DINOv2 / `dinov2_vits14` only

---

## 1. Objective

Under a fully frozen NorOpCal method and anomaly firewall, audit the target-normal
support structure of all MVTec AD 2 categories and units, and determine whether a
supplementary support protocol, defined independently and solely from dataset
semantics and normal-only metadata, would legally admit more categories into the
fixed-`K=8` evaluation.

**Core question answered:**

> Can the K=8 evidence breadth be expanded using real, independent target-normal
> observations under a scientifically valid supplementary support definition,
> without changing the frozen method or choosing pooling rules from anomaly outcomes?

**Answer: NO.** This is a fully acceptable scientific result. It is reported, not hidden,
and the frozen PB8 principal result is preserved unchanged.

---

## 2. Frozen predecessor (unchanged)

| Policy element | Frozen value |
|---|---|
| Detector | `dinov2_vits14` |
| Actuator | `A_MEDIAN_RAW` |
| Offset | `Δτ_raw = median_target - median_source` |
| Shrinkage | `λ = 0.25` |
| Gate | `median_shift >= Q50` |
| Frozen Q50 | `0.015890181064605713` |
| Fallback | `NoAdapt` |
| Principal support | `K = 8` |
| Safety budget | `mean ΔFNR <= +0.02` |

Frozen PB8 principal result (verbatim, preserved, not modified):

```
fabric + walnuts
10 DINOv2 K8 units
ΔBA = +3.23 pp
95% CI = [2.70, 3.76] pp
mean ΔFNR = +0.24 pp
harm = 3.67%
coverage = 40.33%
```

---

## 3. Execution order and gates

Executed `P0 → A → B → C` under Phase-I (metadata / normal-only only).
Because the C gate returned **C-STOP**, the FREEZE / D / E / F stages are **NOT launched**.
No anomaly outcome was ever loaded; the outcome firewall held throughout.

| Stage | Verdict | Anomaly outcome loaded |
|---|---|---|
| P0 — Evidence freeze | `PASS` | False |
| A — Full AD2 target-normal census | `PASS` | False |
| B — K8 ineligibility root-cause audit | `PASS` | False |
| C — Semantic pooling admissibility audit | `C-STOP` | False |
| FREEZE / D / E / F | NOT RUN (no legal protocol) | — |

---

## 4. Census results (Phase-I, metadata + normal-only)

| Metric | Value |
|---|---|
| Units enumerated | 39 |
| Raw target-normal images (all units) | 315 |
| Unique images | 315 |
| Duplicate hash rows | 0 |
| TEST units | 26 |
| TEST K8-eligible units | 10 |
| TEST K8-eligible categories | `fabric`, `walnuts` |

All census rows matched the frozen 39-unit list and score-count ↔ raw-image-count
correspondence (A gate `PASS`).

---

## 5. K8 ineligibility root-cause audit (Stage B)

All 16 TEST-side K8-ineligible units were classified as **R1 (TRUE_DATA_SHORTAGE)**:

| Category | Units | per-domain n | Root cause |
|---|---|---|---|
| rice | 5 | 7 | R1 TRUE_DATA_SHORTAGE |
| sheet_metal | 5 | 4 | R1 TRUE_DATA_SHORTAGE |
| vial | 6 | 5 | R1 TRUE_DATA_SHORTAGE |

- R2 (unit fragmentation of a homogeneous pool): **0** — the multiple conditions in
  each category carry distinct acquisition regimes (distinct filename suffixes), so
  they are not fragments of one semantically homogeneous domain.
- R3 (duplicate / non-independent support): **0** — no duplicate-hash rows exist.
- R4 (other protocol constraint): **0**.

The B1 critical distinction is settled: the shortage is genuinely per acquisition
domain, not an artifact of how the original protocol partitioned the data.

---

## 6. Semantic pooling admissibility audit (Stage C)

Four candidate pooling schemes were registered and scored against S1–S7:

| ID | Pooling rule | S2 | S3 | Admissible |
|---|---|---|---|---|
| C1 | Pool all public target-normal across the acquisition conditions of one category | FAIL | FAIL | False |
| C2 | Pad short target pool with source-domain (regular) normals | FAIL | FAIL | False |
| C3 | Supplement with unlabeled test_private / test_private_mixed normals | FAIL | FAIL | False |
| C4 | Run with (n<8) support but report it as fixed K=8 | FAIL | FAIL | False |

**None passed.** The decisive blockers were S2 (same target-domain scientific meaning)
and S3 (deployment plausibility):

- AD2 acquisition conditions (regular / overexposed / underexposed / shift_N) are
  distinct regimes carried by the public filename suffix. Pooling across them mixes
  regimes that describe no single deployed setting.
- Source-domain supplementation is an explicitly forbidden operation
  ("source-normal filling target-normal").
- Private mixes carry no acquisition-condition identity, so target-domain semantics
  cannot be established.
- Relabeling `n<8` as `K8` is an explicitly forbidden pseudo-expansion.

Per C3 preference ordering, the most conservative reading was applied: because the
strongest semantic-equivalence criterion (S2) fails for every scheme, no scheme was
selected, even though cross-condition pooling would mechanically reach 8 images.

---

## 7. Final verdict

```
terminal = PB8_S1_NO_LEGAL_K8_BREADTH_EXPANSION   (T3)
breadth_tier = BREADTH-C   (no new legal supplementary K8 category)
utility_gate / safety_gate / category_consistency / repeat_robustness = NOT RUN
```

The original PB8 K8 evidence breadth is limited to `fabric + walnuts` (EB2
"LIMITED_BUT_REPORTABLE"), and this limitation is confirmed as intrinsic to the
dataset semantics rather than an artifact that a supplement could legally remove.

---

## 8. Claim handling

PB8 frozen claim is **not changed**. It remains:

> principal K8 evidence is limited to frozen K8-eligible acquisition-shift units.

No supplementary claim is added. The negative supplementary result is disclosed
(not hidden), exactly as required for T3/T4.

---

## 9. Limitations / NOT_VERIFIED

- The DEV-side categories (`can`, `fruit_jelly`, `wallplugs`) have sufficient
  per-domain counts (12, 5, 10) but are excluded by the frozen DEV/TEST split; they
  are not K8-evaluation candidates under the frozen protocol.
- No outcome evaluation was run, so no supplementary ΔBA / ΔFNR / harm / benefit
  numbers exist to report; none were computed or needed for this verdict.
- The breadth limitation is dataset-semantic; it does not claim that a different
  dataset or a future acquisition protocol could never admit more categories.

---

## 10. Required artifacts — delivery status

| Artifact | Status |
|---|---|
| `P0_S1_INPUT_MANIFEST.csv` | ✓ |
| `P0_ORIGINAL_PB8_SCOPE.json` | ✓ |
| `A_AD2_TARGET_NORMAL_CENSUS.csv` | ✓ |
| `A_CATEGORY_SUPPORT_SUMMARY.csv` | ✓ |
| `B_K8_INELIGIBILITY_ROOT_CAUSE.csv` | ✓ |
| `C_POOLING_CANDIDATE_REGISTRY.csv` | ✓ |
| `C_SEMANTIC_ADMISSIBILITY_REPORT.md` | ✓ |
| `F_PB8_S1_BREADTH_VERDICT.json` | ✓ |
| `PB8_S1_AD2_K8_BREADTH_FINAL_REPORT.md` | ✓ (this file) |
| `TABLE_S1_AD2_K8_EVIDENCE_BREADTH.csv` | ✓ |
| `P0_EXECUTION_LEDGER.csv` (FINAL row) | ✓ |
| `S1_FROZEN_SUPPLEMENTARY_PROTOCOL.json` | N/A (C-STOP; no protocol frozen) |
| `S1_FIREWALL_AUDIT.json` | N/A (C-STOP) |
| `D/E/F*` outcome artifacts | N/A (C-STOP; FREEZE/D/E/F not launched) |

Per the C-STOP branch, D/E/F outcome artifacts are intentionally not generated.