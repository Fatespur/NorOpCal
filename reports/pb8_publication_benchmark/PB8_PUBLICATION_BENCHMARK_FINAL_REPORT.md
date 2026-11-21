# PB8 Publication Benchmark Final Report

Generated: 2026-08-20T03:27:58Z  
Frozen predecessor: `PB7_P2_PUBLICATION_CANDIDATE_CONFIRMED_DINOV2_SPECIFIC`  
Final terminal: **T2 — PB8_PAPER_PACKAGE_READY_WITH_SCOPE_LIMITATIONS**

## Executive answer

**1. Is the frozen method strong enough to serve as the paper's central method?**  
**Yes, for a deliberately scoped DINOv2-specific paper.** The principal confirmatory effect is positive, its sealed confidence interval excludes zero, the mean ΔFNR remains well inside the preregistered budget, aggressive baselines expose the utility/safety distinction, the component evidence is interpretable, detector dependence is informative, and the incremental actuator is computationally small. It is not strong enough to support a detector-agnostic, broadly K=8-generalized, externally validated, or harm-free paper.

**2. What is the strongest evidence-supported claim?**  
On the 10 DINOv2 K=8 units from the two frozen TEST categories eligible under the acquisition-shift protocol (fabric and walnuts), a preregistered normal-only selective bounded threshold calibration improved mean balanced accuracy by **3.23 pp** (category-aware 95% CI **[2.70, 3.76] pp**) versus NoAdapt while mean FNR increased by **0.24 pp**, within the **+2 pp** safety budget. This is detector-specific operating-point evidence and does not establish representation/ranking gains, K=4 safety, cross-dataset validity, or harm-free generalization.

## P0 — Evidence and estimand freeze

- The publication manifest binds 30/30 required upstream artifacts by SHA-256.
- Principal hierarchy is DINOv2 K=8; pooled K=8 is secondary; PatchCore/EfficientAD are mechanism and negative-control evidence.
- Primary utility is ΔBA; primary safety is mean ΔFNR; harm is secondary safety.
- No principal policy element changed: `A_MEDIAN_RAW`, λ=0.25, Q50 gate, K=8, NoAdapt fallback.

## A — Comparator eligibility

- Six-source literature acquisition produced 140 source records and 106 unique records; 20 API records plus one live-verified ICML paper survived semantic review.
- Nine closest candidates were audited against target-information, detector/score compatibility, test-outcome tuning, and public implementation rules.
- Eligible direct literature comparators: **0**.
- Frozen verdict: `NO_DIRECT_LITERATURE_COMPARATOR_WITH_MATCHED_DEPLOYMENT_SEMANTICS`.
- The benchmark therefore uses the strongest protocol-compatible frozen baselines and does not manufacture a SOTA comparison.

## B — Final benchmark

Principal DINOv2 B5 (n=10 units; 2 eligible categories):

- BA: 0.5411 → 0.5733; ΔBA = 3.23 pp.
- Sealed 95% CI for ΔBA: [2.70, 3.76] pp.
- FPR: 0.2845 → 0.2176; ΔFPR = -6.70 pp.
- FNR: 0.6333 → 0.6358; ΔFNR = 0.24 pp.
- Harm/catastrophic harm = 3.67% / 3.67%; coverage = 40.33%.
- Benefit precision/recall = 0.68 / 0.40.
- B2 raw and B4 gate-only are explicitly `SAFETY FAIL`; B1 shows that stronger raw utility does not imply safety.

Secondary pooled B5 (n=30) has ΔBA=1.20 pp, ΔFNR=1.27 pp, harm=14.33%, and remains secondary by construction.

## C — Component and negative-control evidence

- Raw always-on (gate OFF, shrinkage OFF): ΔBA +5.58 pp but ΔFNR +6.31 pp and harm 37.67% — unsafe.
- Gate only: ΔBA +4.25 pp but ΔFNR +7.49 pp and harm 37.67% — unsafe.
- Bounded only: ΔBA +3.64 pp, ΔFNR -0.16 pp, harm 3.67%, coverage 100% — safe.
- Full method: ΔBA +3.23 pp, ΔFNR +0.24 pp, harm 3.67%, coverage 40.33% — safe/selective.
- At matched coverage, random abstention averaged ΔBA 1.47 pp (2.5–97.5% randomization interval [1.24, 1.70] pp), below the frozen gate. Random abstention has lower average harm, so the result supports gate utility/selectivity, not universal safety dominance.
- The frozen λ grid shows monotonic safety degradation as actuation becomes less conservative.

## D — Support and evidence breadth

- Total categories audited: 8; frozen TEST K=8-eligible categories: 2 (`fabric`, `walnuts`).
- K=8 units: 30 pooled, 10 DINOv2.
- K=4 is unsupported/unsafe; DINOv2 FULL has mean ΔFNR +2.82 pp and is not monotonically safer.
- Evidence-breadth terminal: **EB2 — LIMITED_BUT_REPORTABLE**.
- No duplication, replacement sampling, augmentation-as-independent-support, or `min(8,n)` pseudo-expansion was used.

## E — Mechanism and residual risk

- Median shift predicts correctability (historical AUC 0.712 full / 0.802 K=8) but not oracle correction direction or magnitude.
- The method shrinks a raw median offset; it does not estimate an oracle threshold.
- DINOv2: `USEFUL_SAFE_SELECTABLE`; PatchCore: `ACTUATOR_MISMATCH`; EfficientAD: `NO_HEADROOM`.
- All nine sealed harmful repeat-level cases are in walnuts; no post-hoc walnuts rule was introduced.
- Capability is threshold calibration / operating-point adaptation only; score ranking is fixed.

## F — Statistics

- Primary confirmatory intervals reuse the sealed PB7-P2 B=10,000 category-aware bootstrap.
- Component contrasts use one preregistered paired B=10,000 bootstrap seed (`20260820`).
- The 1,000-seed random-abstention interval is a randomization interval and is not mislabeled as a category-level sampling CI.
- Effect sizes and intervals are primary; no p-value family was added.
- Images, conditions, and support repeats are not treated as independent category replications.

## G — Deployment cost

- K=8 target-normal samples; no retraining, gradients, target anomalies, or extra parameters.
- Verified actuator-only CPU microbenchmark over 100,000 loops: p50 0.136 ms/entity; p95 0.535 ms/entity.
- Additional measured CPU memory: 3,632 bytes; additional GPU memory: 0 bytes; stored scalar overhead: 96 bytes.
- Base DINOv2 inference cost is **NOT_VERIFIED_IN_PB8**. No end-to-end runtime superiority claim is made.

## H–I — Publication assets and visual QA

- Five exact-name tables and five figures are present with frozen source snapshots and reproducible Python scripts.
- Each formal figure has PDF, editable-text SVG, 600-DPI PNG, and grayscale preview.
- Strict raster/vector checks and rendered human visual QA passed.
- The first auto-flowchart candidate passed machine checks but failed human English-label/layout QA; it is retained only as diagnostic evidence. The formal Figure 1 is the inspected code-native vector schematic.

## J–M — Claim, naming, and paper structure freeze

- Claim hierarchy is frozen in `FINAL_CLAIM_HIERARCHY.md`.
- Working name is **NorOpCal**, after a preliminary exact-name collision check; naming remains revisable without changing method semantics.
- `PAPER_SECTION_PLAN.md` binds every proposed subsection to evidence, tables/figures, and an exact permitted claim.
- `TABLE_5_SCOPE_AND_FAILURES.csv` keeps category breadth, external-validation absence, walnuts risk, detector specificity, K-support failures, and base-cost uncertainty visible.

## Final gate rationale

T1 is not used because category breadth, lack of a legal external validation setting, and localized walnuts risk materially narrow the paper claim. T3 is not used because no eligible matched published comparator displaced the frozen method, and the method retains independent safety/selectivity value. T4 is not used because the K=8 result is adequate for a transparently scoped detector- and eligibility-bounded central claim, but not for a generalized few-normal claim. T5 is not used because the final hierarchy has been reduced to match the evidence.

**Final terminal: `PB8_PAPER_PACKAGE_READY_WITH_SCOPE_LIMITATIONS`.** The method and experimental core are frozen and may proceed to formal section-by-section manuscript writing, subject to the exact claim hierarchy and limitations above.
