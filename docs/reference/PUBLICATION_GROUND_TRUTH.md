# PUBLICATION_GROUND_TRUTH.md — NorOpCal Default Publication Truth Source

> Status: **FROZEN (PW0.5, 2026-08-20; PW0.5A narrative-layering patch, 2026-08-20)**
> Terminal: `PW0_5_PUBLICATION_GROUND_TRUTH_COMPLETE_EVIDENCE_RECONCILED_CONTEXT_COMPRESSED_AND_MANUSCRIPT_BOUND`
> PW0.5A terminal: `PW0_5A_NARRATIVE_SCOPE_CLARIFICATION_COMPLETE_SCIENTIFIC_GROUND_TRUTH_UNCHANGED` —
> writing-semantics patch only; no scientific fact, number, method parameter, claim tier,
> prohibited claim, notation, equation, or provenance entry changed
> (audit: `reports/manuscript/PW0_5/PUBLICATION_GROUND_TRUTH_GATES_PW0_5A_ADDENDUM.md`).
> Upstream frozen sources: PB7-P0 (F4), PB7-P1 (T4), PB7-P2 (T1), PB8 (T2), PB8-S1 (T3), PW0, PW1.
> Every number below was re-read from its bound authoritative artifact during PW0.5.

> **PW2–PWn Default Context Rule**
>
> Subsequent paper-writing stages shall use this file as the default scientific ground-truth
> source. Historical PB artifacts should only be reopened when a specific evidence gap,
> inconsistency, or provenance check requires it. In default operation a PW stage needs only:
> (1) this file, (2) the current `NorOpCal_TIM_Manuscript.docx`, (3) the current PW stage
> instructions.
>
> Conflict rules: if this file conflicts with a historical exploratory artifact, this file wins.
> If it conflicts with a **later sealed** artifact, stop writing the affected claim and correct
> this file first from that artifact. Detailed registries remain in `reports/manuscript/PW0/`
> (claim registry C1–C16, non-claims, terminology, symbols, section map, figure/table registry,
> citation firewall); this file is the compact binding layer over them.

## Narrative Layering Rule (PW0.5A)

> Ground-truth qualifiers define the maximum evidential scope of empirical claims, but they do
> not need to be repeated verbatim in every manuscript sentence. General problem motivation,
> title-level positioning, methodological framing, contribution statements, and method
> definitions may use the broader framing of industrial anomaly detection under acquisition
> shift, provided that they do not imply unsupported empirical generalization.

- **Framing layer** — problem motivation, title-level positioning, methodological framing,
  contribution statements, method definitions, and number-free Abstract / Introduction /
  Conclusion prose — may use general industrial-anomaly-detection / acquisition-shift phrasing.
- **Result layer** — quantitative results, empirical-breadth statements, and any generalization
  claim — must still carry the required scope qualifiers explicitly (detector, K, eligibility,
  dataset/category), exactly as bound in §5–§7.
- Evidence-scope qualifiers must not be mechanically converted into unnecessary rhetorical
  weakening of framing sentences.
- This rule grants **no** relaxation of §7 prohibited claims or `MANUSCRIPT_NONCLAIMS.md`; it
  changes where qualifiers must appear, never what the evidence supports.

---

# 0. Publication Identity

| Item | Frozen value |
|---|---|
| Method name | **NorOpCal** |
| Formal expansion | **Normal-Only Selective Operating-Point Calibration** (PW0.5 directive; see §14 issue O1) |
| Historical codename | **ShiftAD** — historical/internal project codename only; never usable as the paper method name |
| Core task (frozen sentence) | Under acquisition shift, NorOpCal uses only a small set of target-normal observations to selectively and conservatively recalibrate the operating point of a frozen industrial anomaly detector without modifying the detector itself. |
| Consistency check | Matches the frozen method (gate + bounded threshold update, detector untouched). PASS. |
| Name status | `WORKING_NAME_AFTER_PRELIMINARY_COLLISION_CHECK` (PB8 audit, 2026-08-20); a final scholarly collision check MUST be re-executed before PW10 title freeze |
| Venue target | IEEE Transactions on Instrumentation and Measurement (IEEE TIM) |

Naming-conflict record (do not resolve silently again): PB8 `METHOD_NAME_CANDIDATES.md` froze the
working expansion "Conservative Normal-only Operating-Point Calibration". The PW0.5 directive
specifies "Normal-Only Selective Operating-Point Calibration". The short name **NorOpCal is
unchanged**; the PB8 name was explicitly marked revisable during drafting; PW0.5 adopts the
directive expansion as canonical. The manuscript does not yet contain any expansion (verified:
only the heading "IV. NorOpCal" exists), so no manuscript conflict arises; PW2 must introduce
the expansion exactly as frozen here.

# 1. Problem Setting

Deployment assumptions (all frozen; consistent with manuscript Section III, PW1):

- Known product category; unknown target acquisition condition (acquisition shift).
- Frozen anomaly detector: parameters, internal representation, score function `s(·)`, and the
  induced ranking are fixed at deployment; no fine-tuning, gradient update, retraining, or
  additional parameters.
- Frozen source operating threshold `τ_s`, derived from the source-normal score set `S_s^N`.
- K independent target images known in advance to be normal (K = 8 principal).
- No target anomalies, no anomaly labels, no masks, no target anomaly score distribution, no
  prevalence, no test labels, and no outcome-derived thresholds — unavailable and unusable,
  directly or indirectly, for selecting or applying the adaptation.
- The complete deployment information set is `(τ_s, S_s^N, S_t^N)`.

**What may change:** only the decision operating point (threshold `τ_t`).
**What never changes:** backbone, feature extractor, scoring function, detector parameters,
memory/prototype representation, ranking, and any target test labels.

Anti-misdescription rule: NorOpCal is **not** domain adaptation, detector adaptation,
representation adaptation, or test-time model adaptation. It is threshold-only operating-point
calibration on a frozen score scale.

# 2. Frozen NorOpCal Method

Single legal specification (PB7-P1 `E_FROZEN_METHOD_SPEC.json`, family registry
`A_ACTUATOR_FAMILY_REGISTRY.json`; no re-optimization of any component after PW0).

**Inputs.** `τ_s`; `S_s^N` (source-normal scores); `S_t^N` = K known-normal target scores; frozen
detector identity. Nothing else.

**Target-normal support size.** K ∈ {4, 8, FULL}. Semantics: K=4 auxiliary (unsafe, §5.2);
K=8 principal; FULL = every available target-normal observation per unit. Deployment
recommendation: **K = 8 is the minimum support size demonstrated safe under the frozen protocol,
within K8-eligible units, for the principal detector DINOv2** — an empirical finding under the
frozen evaluation, **not** a mathematical guarantee.

**Normal-score statistic (median).** `m_s = median(S_s^N)`; `m_t = median(S_t^N)`;
`Δm = m_t − m_s` (median shift; identical object to the raw correction `Δτ_raw`).

**Correctability gate.** Adapt iff `Δm ≥ Q50`, frozen **Q50 = 0.015890181064605713**
(PB7-P1 DEV-selected cutoff on the median-shift distribution). Gate fails → abstain.

**Actuator.** `A_MEDIAN_RAW` with conservative shrinkage **λ = 0.25**:
`τ_raw = τ_s + (m_t − m_s)`; `τ_t = τ_s + 0.25 · (τ_raw − τ_s)`; equivalently the applied update
is `τ_t = τ_s + 0.25 · Δm` (gate pass only).

**Abstention behavior.** Gate rejection → `τ_t = τ_s` (NoAdapt). Abstention is a designed
outcome, not a failure.

**Safety constraint (problem-level, preregistered).** `𝔼[ΔFNR] ≤ +0.02` (mean FNR degradation
budget, +2 pp), fixed before any evaluation.

Forbidden after freeze: re-optimizing λ, Q50, feature, gate, actuator, detector, or K; any
walnuts-specific rule; any post-hoc component addition.

# 3. Obsolete or Non-Final Research Routes

These must **never** be reintroduced as components of NorOpCal. Each is marked with its only
permissible role.

| Route | Status |
|---|---|
| Category-conditioned selector (PB5 family) | historical / not part of final method |
| Deformation features beyond median shift; deformation-based gating (PB7-P0 Stage B features) | analysis only (mechanism evidence) |
| Residualization | historical |
| BC_RANK / BC_QUANTILE threshold adaptation | historical baseline / analysis only; behaviorally equivalent top-k rank executors, not independent methods |
| Quantile-mapping actuator (QM, PB6) | analysis only; frozen QM = operating-point rebalancing on failed points, not capability improvement (post-hoc audit) |
| Benefit-Label Oracle / oracle thresholds O1–O5 | oracle diagnostic only (Level D; `DIAGNOSTIC ONLY` label mandatory) |
| Domain adaptation / feature alignment / representation learning | prohibited semantics (never a component) |
| Detector fine-tuning, pseudo-label anomaly adaptation, memory-bank adaptation | prohibited semantics |
| Contamination protocol ρ ∈ {0, 5, 10, 20%} (PB1DS-PB4/D8) | historical route; **not part of the frozen PB7/PB8 evidence base; no registered table/figure; no contamination claim is in contract** |
| Early ShiftAD branches (PB1–PB6 publication route) | historical; superseded by PB7-P0→P1→P2→PB8 chain for all manuscript numbers |

# 4. Experimental Protocol

**Dataset.** MVTec AD 2 (AD2) only. 8 categories, 39 category-condition units (vial 6
conditions, fruit_jelly 3, others 5); 315 unique target-normal images (0 duplicates). Unit =
category × acquisition condition; manuscript-visible entities are unit × detector (117 total;
39 per detector).

**DEV/TEST split (frozen).** DEV = can (12/unit), fruit_jelly (5), wallplugs (10) — 13 units,
used for λ/Q50/actuator selection only. TEST = fabric (11), rice (7), sheet_metal (4), vial (5),
walnuts (10) — 26 units, evaluation only.

**Detectors.** DINOv2 (`dinov2_vits14`, principal); PatchCore (`wide_resnet50_2` backbone,
mechanism control); EfficientAD (`efficientad`, mechanism control). EfficientAD is an
**in-house-trained instance** (70,000 steps, source-normal only, frozen teacher/checkpoint
hashes; PB1DS-PB6 D2/D4) — the manuscript must not imply official pretrained weights; exact
per-detector construction details for §5.2 must be transcribed from the PB1DS/PB7-P0 sealed
bindings (see O5).

**Support sizes.** K=4 (26 DINOv2 units, 5 TEST categories), K=8 (10 DINOv2 units, 2 categories),
FULL (26 units). Eligibility by independent target-normal count only, never by outcomes.
K-support robustness uses 200 preregistered support repeats (repeats ≠ category replications).
Entity-level K8 results use 30 sealed repeats per entity.

**K8-eligible units.** 10 TEST units: fabric (5) + walnuts (5). The 16 ineligible TEST units are
all R1 true-data-shortage (rice 7, sheet_metal 4, vial 5). Semantic pooling: 4 registered
candidates, 0 admissible (PB8-S1 C-STOP). Canonical sentence: "K=8 evaluation was restricted to
acquisition-shift units containing at least eight independent target-normal observations."

**Contamination.** Not part of the frozen protocol (see §3). No contamination experiment exists
in the PB7/PB8 evidence base.

**Statistical inference.** Category-aware bootstrap, B = 10,000 (PB7-P2 authoritative reuse);
95% CI. Random-abstention negative control: 1000 preregistered control seeds (randomization
interval, not a sampling CI). Primary utility metric: ΔBA = BA(τ_t) − BA(τ_s) vs NoAdapt.
Primary safety: mean ΔFNR vs +2 pp budget. Secondary: harm rate (entity-level event with
ΔBA < −0.02 or ΔFNR > +0.02), coverage, ΔFPR. Report hierarchy: DINOv2 principal before pooled
(pooled always labeled SECONDARY).

# 5. Main Numerical Results

All values read from sealed artifacts at PW0.5. Percent changes in percentage points (pp).

## 5.1 Main Benchmark (PB8 Table 1; DINOv2, K=8, 10 units, fabric + walnuts)

| Policy | ΔBA pp [95% CI] | ΔFPR pp | ΔFNR pp | Harm % | Coverage % | Safety |
|---|---|---|---|---|---|---|
| B0 NoAdapt (`τ_s`) | 0 (ref; BA 0.5411) | 0 | 0 | 0 | 0 | OK |
| B1 Unsafe-strong (A_T2_RAW, λ=1.0, no gate) | +8.05 [+5.41, +10.69] | −19.62 | +3.51 | 49.0 | 100 | FAIL |
| B2 Raw median (A_MEDIAN_RAW, λ=1.0, no gate) | +5.58 [+4.26, +6.91] | −17.48 | +6.31 | 37.67 | 100 | FAIL |
| B3 Bounded-only (λ=0.25, no gate) | +3.64 [+3.52, +3.76] | −7.12 | −0.16 | 3.67 | 100 | OK |
| B4 Gate-only (λ=1.0, Q50) | +4.25 [+3.96, +4.55] | −15.99 | +7.49 | 37.67 | 40.33 | FAIL |
| **B5 NorOpCal (λ=0.25, Q50 gate)** | **+3.23 [+2.70, +3.76]** | **−6.70** | **+0.24** | **3.67** | **40.33** | **OK** |
| B6_T2 (A_T2_RAW, λ=0.25, Q50) | +3.50 [+3.44, +3.55] | −7.38 | +0.38 | 5.67 | 40.33 | OK |
| B6_TAIL (A_TAIL_RAW, λ=0.25, Q50) | +2.94 [+1.97, +3.91] | −5.88 | 0.00 | 0.0 | 40.33 | OK |

Principal (B5): BA 0.5411 → 0.5733; bootstrap mean (B=10,000) +3.23 pp, CI [+2.70, +3.76];
mean ΔFNR +0.24 pp ≤ +2 pp budget. Baseline-label rule: PB8 labels (B5 = proposed); PB7-P1
called this policy B4 and its own B5 is an oracle diagnostic — never reuse PB7-P1 labels.
Selection was preregistered on DEV, not post hoc.

## 5.2 Few-Normal Support (PB8 Table 4 / `D_K_SUPPORT_FINAL.csv`)

| K | Group (units) | ΔBA pp | ΔFNR pp | Harm % | Coverage % | Safety freq (200 repeats) |
|---|---|---|---|---|---|---|
| 4 | DINOv2 (26) | +3.76 | **+2.85** | 28.59 | 49.87 | **0.0** |
| 4 | Pooled (78) | +1.05 | +2.34 | 23.42 | 40.17 | 0.095 |
| **8** | **DINOv2 (10)** | **+3.23** | **+0.24** | **3.67** | **40.33** | **1.000** |
| 8 | Pooled (30) | +1.20 | +1.27 | 14.33 | 34.22 | 0.965 |
| FULL | DINOv2 (26) | +3.72 | **+2.82** | 26.92 | 50.0 | 0.731 |
| FULL | Pooled (78) | +1.08 | +1.88 | 19.23 | 35.90 | 0.808 |

Frozen conclusion: **K=4 is unsupported/unsafe under the frozen policy; K=8 is the principal
safe few-normal setting, only within K8-eligible units; FULL is not monotonically safer**
(DINOv2 FULL ΔFNR +2.82 pp exceeds the +2 pp budget). Pooled K8 bootstrap: ΔBA +1.20 pp
[+1.01, +1.38], ΔFNR +1.27 pp, harm 14.33% (SECONDARY; never headline).

## 5.3 Contamination Robustness

**No frozen result exists.** Contamination experiments belong to the historical PB1DS route and
carry no registered table/figure/claim in the final paper. Any contamination statement
("robust under contamination", "degrades gracefully") is out of contract.

## 5.4 Detector-Specific Results (PB8 Table 2; frozen policy, K=8)

| Detector | ΔBA pp | ΔFPR pp | ΔFNR pp | Harm % | Coverage % | Verdict |
|---|---|---|---|---|---|---|
| DINOv2 | +3.23 | −6.70 | +0.24 | 3.67 | 40.33 | USEFUL_SAFE_SELECTABLE |
| PatchCore | −0.66 | −0.83 | +2.16 | 24.67 | 24.67 | ACTUATOR_MISMATCH (safety fail) |
| EfficientAD | +1.03 | −3.45 | +1.40 | 14.67 | 37.67 | NO_HEADROOM |

Oracle-conditioned safe-correctable counts (9/1/0 of 10) are `DIAGNOSTIC ONLY` (Level D).
Frozen conclusion: DINOv2 benefits; PatchCore is actuator-mismatched; EfficientAD has no safe
correctability headroom. These are detector-specific findings, **not** universal statements;
PatchCore/EfficientAD are mechanism controls, not "removed detectors".

## 5.5 Final Safety-Constrained Actuator Result (PB7-P1/P2 + PB8)

- Pooled ΔBA (B=10,000 bootstrap): +1.20 pp [+1.01, +1.38]; DINOv2 +3.23 pp [+2.70, +3.76]
  (CI lower bound > 0). Mean ΔFNR: pooled +1.27 pp; DINOv2 +0.24 pp. Harm: pooled 14.33%,
  DINOv2 3.67%. Coverage: pooled 34.22%, DINOv2 40.33%.
- λ safety curve (DINOv2 K8, gated): λ=0.25 → ΔBA +3.23 / ΔFNR +0.24 / harm 3.67%;
  λ=0.5 → +5.06 / +1.78 / 20%; λ=0.75 → +5.17 / +5.11 / 29%; λ=1.0 → +4.25 / +7.49 / 37.67%.
  Safety cost rises monotonically with λ on the frozen grid.
- Gate selectivity: NorOpCal beats coverage-matched random abstention by +1.75 pp ΔBA
  (randomization interval [+1.53, +1.99] pp, 1000 seeds). Nuance (mandatory when cited): on
  DINOv2 the gate primarily reduces coverage (always-on λ=0.25 is already safe there); pooled
  evidence supports additional safety value.
- Mechanism: median shift identifies correctability (AUC 0.712 full-population; 0.802 at K=8)
  but does **not** identify oracle threshold direction (max |Spearman| 0.203) or magnitude;
  NorOpCal does not estimate an oracle threshold.
- Residual harm: 9/78 harmful entities, all in walnuts; 89% with high source FNR; no
  post-hoc walnuts rule exists or may be implied.
- Confirmatory (PB7-P2): exact reproduction < 1e-9; leave-one-category-out 8/8 positive with
  ΔFNR ≤ +2 pp; 200 K8 repeats (DINOv2 positive frequency 1.000, safety frequency 1.000);
  B=10,000 bootstrap CI lower bound +2.70 pp > 0. Terminal T1
  `PB7_P2_PUBLICATION_CANDIDATE_CONFIRMED_DINOV2_SPECIFIC`.
- Deployment overhead (actuator only): 0.136 ms/entity p50, 0.535 ms p95, 3,632 B CPU memory,
  96 B storage, no gradients/retraining/extra parameters (100,000-loop CPU microbenchmark,
  10 entities). Base DINOv2 per-image runtime `NOT_VERIFIED_IN_PB8` — never claim it.
- Terminals: PB7-P1 T4 `PB7_P1_DETECTOR_SPECIFIC_SAFE_ACTUATOR_SUPPORTED`; PB8 T2
  `PB8_PAPER_PACKAGE_READY_WITH_SCOPE_LIMITATIONS`; PB8-S1 T3
  `PB8_S1_NO_LEGAL_K8_BREADTH_EXPANSION` (C-STOP; no ΔBA/ΔFNR produced; protocol evidence only).

Frozen summary sentence (if cited, keep "detector-specific"): **NorOpCal is a conservative
bounded abstaining normal-only threshold actuator with detector-specific utility, strongest in
the DINOv2 regime.**

# 6. Allowed Publication Claims

Qualifier binding (PW0.5A): the qualifiers below bind **result-layer** sentences — any
sentence reporting numbers or empirical breadth. Framing-layer prose follows the Narrative
Layering Rule.

**Tier A — Strongly Supported** (usable in Abstract / Introduction / Contributions / Conclusion):

| ID | Claim | Binding |
|---|---|---|
| A1 | Only K known-normal target observations (K=8 principal) are required; no target anomalies, labels, masks, or detector retraining | X |
| A2 | The detector remains fully frozen (parameters, representation, score function, ranking) | P |
| A3 | On the 10 frozen DINOv2 K=8 TEST-eligible units (fabric, walnuts), the preregistered policy improved mean BA by +3.23 pp (95% CI [+2.70, +3.76] pp) while mean ΔFNR stayed at +0.24 pp, within the +2 pp preregistered budget | P/C1/C2 |
| A4 | Conservative shrinkage (λ=0.25) is the principal safety control; λ=1.0 variants fail the safety budget | C3 |
| A5 | Selective bounded calibration keeps harm at 3.67% where unbounded always-on raw actuation reaches 37.67% under the same coverage-eligible units | C3/C15 |
| A6 | Correctability and safe threshold actuation are detector-dependent (DINOv2 useful-safe-selectable; PatchCore actuator-mismatch; EfficientAD no headroom) | M/C4 |
| A7 | The actuator adds negligible overhead (0.136 ms p50 per entity, CPU-only) and no parameters | X/C12 |
| A8 | No directly deployment-matched published comparator was identified in the audited literature (9 candidates audited, all ineligible) | C13 |
| A9 | The principal result reproduces exactly (< 1e-9) and is robust to leave-one-category-out, 200 support repeats, and confirmatory bootstrap | C14 |

**Tier B — Supported with Qualification** (carry the qualifier verbatim):

| ID | Claim | Required qualifier |
|---|---|---|
| B1 | Pooled K8 utility ≈ +1.20 pp ΔBA | "SECONDARY"; never before the DINOv2 result; label pooled |
| B2 | K=8 is the minimum practical support | "under the frozen protocol, within K8-eligible units, for DINOv2"; empirical, not a guarantee |
| B3 | The frozen gate supplies selectivity (+1.75 pp vs coverage-matched random abstention) | include the DINOv2 coverage-reduction nuance |
| B4 | Median shift identifies correctability (AUC 0.712 / 0.802) | "but not oracle threshold direction or magnitude" (Discussion use) |
| B5 | Residual harm is localized (9/78 entities, all walnuts) | disclose as limitation; no post-hoc rule may be implied |
| B6 | K=4 unsafe; FULL not monotonically safer | "under the frozen policy" (negative results; Results/Discussion use) |
| B7 | Safety is not monotonic in support size | empirical observation framing; no re-tuning attempted |

**Tier C — Exploratory / Do Not Elevate** (Discussion only; never Abstract or headline):

- Mechanism interpretation beyond frozen findings (e.g., why FULL medians overshoot the
  conservative bound; why walnuts harm concentrates with high source FNR).
- Category effects and deformation-feature findings beyond median shift.
- Speculative generalization to other shift types, datasets, or deployment regimes.
- Cross-dataset universality (explicitly NOT externally validated).

# 7. Prohibited or Unsupported Claims

Never state (each conflicts with sealed evidence; `OVERCLAIM_VIOLATION` if present):

1. NorOpCal improves all anomaly detectors / is detector-agnostic.
2. NorOpCal universally handles distribution shift.
3. NorOpCal performs domain adaptation or adapts representations/features.
4. NorOpCal uses anomaly supervision (any form).
5. K=4 is sufficient/safe.
6. FULL support is always superior / more support is safer.
7. The oracle result is BA-optimal equivalence for the method (oracle items are diagnostics).
8. BC_RANK and BC_QUANTILE are independent methods (behaviorally equivalent historical executors).
9. Contamination has no impact (no frozen contamination evidence exists at all).
10. Deformation features add statistically significant benefit beyond median shift.
11. EfficientAD results imply universal generalization.
12. AUROC/AUPR or ranking improvement (unchanged by construction).
13. "Outperforms the state of the art" / superiority over prior work (no matched comparator).
14. Harm-free, risk-free, or universally safe operation.
15. External-dataset validation (honest abstention recorded; AeBAD-S gate degenerates).
16. End-to-end runtime superiority (base detector runtime NOT_VERIFIED).
17. Broad industrial-category generalization at K=8 (only fabric + walnuts eligible).
18. "We selected fabric and walnuts" without the support-count eligibility context.

# 8. Frozen Terminology

| Use | Do Not Use |
|---|---|
| NorOpCal | ShiftAD as paper method name |
| acquisition shift | distribution/domain/covariate shift (except Related-Work concept contrast) |
| source-normal samples | source domain data |
| target-normal samples / known-normal target samples | target calibration anomalies; unlabeled target data |
| operating point | decision boundary |
| threshold correctability | calibratability, fixability |
| threshold actuator | adapter, adjuster, corrector |
| abstention / NoAdapt | calibration failure, rejection, fallback |
| conservative shrinkage (λ=0.25) | regularization, damping |
| median shift (Δm) | median drift |
| adaptation coverage | pass rate, hit rate |
| residual harm | failure cases |
| preregistered mean FNR degradation budget (+2 pp) | tolerance, margin |

Detector labels: DINOv2 / PatchCore / EfficientAD (roles: principal / mechanism control /
mechanism control). Dataset: MVTec AD 2 (AD2) at first mention. Capability phrasing allowed:
operating-point adaptation, threshold calibration, safe threshold correction, selective
threshold adaptation, normal-only adaptation, conservative bounded actuation, abstention.
Prohibited unless explicitly negated: representation enhancement/adaptation, feature
improvement/learning, anomaly discrimination improvement, ranking enhancement, AUROC improvement.

# 9. Frozen Mathematical Notation

| Symbol | Meaning | First defined in manuscript |
|---|---|---|
| `x`, `𝒳` | input image / image space | Sec. III ¶1 |
| `s(x)`, `s(·)` | anomaly score of the frozen detector (larger = more anomalous) | Sec. III ¶1 |
| `τ` | generic decision threshold | Sec. III ¶1 |
| `ŷ(x;τ)` | binary decision | Eq. (1) |
| `𝕀[·]` | indicator function | Sec. III ¶2 |
| `τ_s` | frozen source threshold (from `S_s^N`) | Sec. III ¶2 |
| `S_s^N` | source-normal score set | Sec. III ¶2 |
| `S_t^N` | target-normal score set (K known-normal scores) | Eq. (2) |
| `K` | number of target-normal support samples (principal K=8) | Sec. III ¶4 |
| `x_{t,1},…,x_{t,K}` | the K known-normal target images | Sec. III ¶4 |
| `τ_t` | adapted target threshold | Eq. (3) |
| `𝒜` | abstract adaptation operator (admissible-class definition) | Eq. (3) |
| `TPR(τ)`, `TNR(τ)`, `BA(τ)`, `FNR(τ)` | evaluation-time estimand functions | Sec. III ¶8 |
| `ΔBA` | `BA(τ_t) − BA(τ_s)` | Eq. (4) |
| `ΔFPR`, `ΔFNR` | induced changes in FPR / FNR | Sec. III ¶9 |
| `𝔼[·]` | mean over evaluated acquisition-shift units | Eq. (5) |
| `0.02` / +2 pp | preregistered safety budget | Eq. (5) |
| `m_s`, `m_t` | median source-/target-normal score | Sec. IV (PW2, pending) |
| `Δm` | median shift `m_t − m_s` (≡ `Δτ_raw`) | Sec. IV (PW2, pending) |
| `Q50` | frozen gate cutoff 0.015890181064605713 | Sec. IV (PW2, pending) |
| `λ` | conservative shrinkage coefficient (frozen 0.25) | Sec. IV (PW2, pending) |
| `Δτ_raw` | raw correction `m_t − m_s` (method-section name of `Δm`) | Sec. IV (PW2, pending) |

No symbol may be renamed by later PW stages; PW1-defined symbols are binding as written.

# 10. Frozen Equation Registry

Canonical TeX only; later PW stages must take equations from here, not regenerate variants.

| ID | Semantic role | Canonical TeX | Manuscript Eq. | Section | Inserted | Verified |
|---|---|---|---|---|---|---|
| E1 | binary decision rule | `\hat{y}(x;\tau) = \mathbb{I}\left[s(x) > \tau\right]` | (1) | III | yes | PASS (PW1) |
| E2 | target-normal score set | `S_t^N = \{s(x_{t,1}), \ldots, s(x_{t,K})\}` | (2) | III | yes | PASS (PW1) |
| E3 | abstract adaptation rule | `\tau_t = \mathcal{A}(\tau_s, S_s^N, S_t^N)` | (3) | III | yes | PASS (PW1) |
| E4 | utility | `\Delta\mathrm{BA} = \mathrm{BA}(\tau_t) - \mathrm{BA}(\tau_s)` | (4) | III | yes | PASS (PW1) |
| E5 | safety constraint | `\mathbb{E}[\Delta\mathrm{FNR}] \le 0.02` | (5) | III | yes | PASS (PW1) |
| E6 | median statistics | `m_s = \mathrm{median}(S_s^N), \quad m_t = \mathrm{median}(S_t^N)` | TBD | IV | no | — |
| E7 | median shift / raw correction | `\Delta\tau_{\mathrm{raw}} = m_t - m_s` | TBD | IV | no | — |
| E8 | correctability gate | `g(\Delta m) = \mathbb{I}\left[\Delta m \ge Q_{50}\right], \quad Q_{50} = 0.015890181064605713` | TBD | IV | no | — |
| E9 | bounded update / abstention | `\tau_t = \tau_s + \lambda\,\Delta\tau_{\mathrm{raw}} \;\text{if}\; g=1; \quad \tau_t = \tau_s \;\text{otherwise}; \quad \lambda = 0.25` | TBD | IV | no | — |

Numbering of E6–E9 continues from (5) as inserted by PW2; presentation (display vs inline) is a
PW2 decision, but the TeX semantics above are frozen. E1's manuscript run text uses the
equivalent non-scaled form `\mathbb{I}[s(x) > \tau]` (no `\left`/`\right`); both variants are
canonical-equivalent. Manuscript equations are currently LaTeX
text runs with `\[ \]` delimiters and Word `SEQ Equation` numbering, bookmarks Eq1–Eq5 (MathType/
OMML conversion pending; see O2).

# 11. Planned Figures

Five figures only (PB8 frozen assets, each PDF/PNG/SVG/grayscale + `SOURCE.csv` under
`reports/pb8_publication_benchmark/figures/`; captions anchored in `FIGURE_CAPTIONS.md`):

- **Fig. 1 — NorOpCal overview**: acquisition shift → K=8 known-normal support → frozen detector
  → median shift → Q50 gate → abstain or λ=0.25 bounded actuator → calibrated threshold. Sections
  1/3/4. No feature/score/ranking modification, no oracle estimation.
- **Fig. 2 — Safety–utility / λ tradeoff**: shrinkage controls safety; gate selectivity vs random
  abstention; +2 pp budget line. Sections 4.3/5.5/6.2–6.3/7.2. Never present λ=1.0 peaks as
  recommended; λ grid frozen on DEV.
- **Fig. 3 — Detector-specificity decomposition**: DINOv2 / PatchCore / EfficientAD verdicts;
  correctability identifiable, direction/magnitude not. Sections 5.2/6.4/7.1. `n_correctable_safe`
  is DIAGNOSTIC ONLY.
- **Fig. 4 — K-support robustness**: K4 unsafe; K8 principal within eligible units; FULL not
  monotonically safer. Sections 5.4/6.5/7.3. No monotonicity reading.
- **Fig. 5 — Residual harm map**: harm localized to walnuts; no post-hoc rule. Sections 7.4/8.

No new figure may be added without a separate frozen registration; no historical route becomes a
main figure.

# 12. Planned Tables

| Table | Content | Source artifact | Status |
|---|---|---|---|
| I (registry: Table 1) | Main DINOv2 K8 benchmark, B0–B6 | `TABLE_1_MAIN_BENCHMARK.csv`; stats `F_FINAL_STATISTICS.csv` | numbers frozen, verified |
| II (Table 2) | Detector specificity (verdicts + diagnostic counts) | `TABLE_2_DETECTOR_SPECIFICITY.csv` | frozen, verified |
| III (Table 3) | Component ablation (gate × shrinkage, paired CIs) | `TABLE_3_COMPONENT_ABLATION.csv`; `C_RANDOM_ABSTENTION_SUMMARY.json` | frozen, verified |
| IV (Table 4) | K-support (K4/K8/FULL + repeat frequencies) | `TABLE_4_K_SUPPORT.csv`; `D_K_SUPPORT_FINAL.csv` | frozen, verified |
| V (Table 5) | Scope/failures L1–L7 (limitations) | `TABLE_5_SCOPE_AND_FAILURES.csv` | frozen, verified |
| S1 | AD2 K8 evidence breadth (per-category support, R1 causes, pooling) | PB8-S1 `TABLE_S1_AD2_K8_EVIDENCE_BREADTH.csv` | frozen, verified |

All seven L-rows of Table V must be retained in meaning; no softening. Registry numbering is
Arabic (Table 1–5); final TIM Roman-numeral formatting is a PW10 presentation decision only.

# 13. Evidence-to-Section Mapping

| Section | Allowed claims / content | Figures/Tables |
|---|---|---|
| I. Introduction | P, C1, C4, C13, X; general problem and method framing permitted (Narrative Layering Rule); principal quantitative evidence remains explicitly scoped to the frozen DINOv2 K=8 evaluation; detailed eligibility breadth is disclosed in Experimental Setup / Results rather than repeated in high-level framing; no universality, no SOTA sentence | Fig. 1 |
| II. Related Work | C13, C9 positioning only; concept comparison; no fabricated quantitative comparison; citations must be VERIFIED first | none |
| III. Problem Formulation | P, C12 (inputs); symbols, deployment set, decision rule, utility/safety (already written, PW1) | Fig. 1 (permitted) |
| IV. Method (NorOpCal) | M, C3, C7, C9, C12; frozen formula verbatim (λ=0.25, Q50 value); no tuning narrative | Fig. 1, Fig. 2 |
| V. Experimental Setup | Protocol facts only (§4 of this file); dataset/firewall/statistics; detector construction from sealed bindings | Table IV, S1 |
| VI. Results | C1/C2/C14 (main), C15/C3 (baselines/ablation), C4/C9 (detector), C5/C8/C11 (K-support); all numbers via Claim IDs | Tables I–IV |
| VII. Analysis & Discussion | C9/C4 (mechanism), C3/C7 (tradeoff), C8/C5 (non-monotonicity), C10 (residual risk), C6 (breadth); no new numbers | Fig. 2–5 |
| VIII. Limitations | L1–L7 verbatim in meaning; C16 external abstention | Table V |
| IX. Conclusion | Tier A + necessary Tier B only; scoped DINOv2 claim | none |

Writing order (frozen): PW2 Method → PW3 Setup → PW4 Results → PW5 Analysis/Limitations →
PW6 Related Work → PW7 Introduction → PW8 Conclusion → PW9 Abstract → PW10 Title/audit.

# 14. Open Publication Issues

| Issue | Impact | Required before | Status |
|---|---|---|---|
| O1 Method formal expansion changed vs PB8 working expansion ("Conservative Normal-only…" → "Normal-Only Selective…") | naming only; semantics unchanged | PW2 first use of the expansion | RESOLVED (this file §0); manuscript Section IV now introduces the frozen expansion (PW2, 2026-08-20; superseded PB8 wording removed) |
| O2 Section III equations are LaTeX text runs (`\[ \]` + SEQ fields), not Word-native MathType/OMML | equation rendering for submission | PW2 / final production | PW2-PARTLY-RESOLVED (M0 repair complete: single noProof runs, no proofErr, numbers outside delimiters; actual MathType Toggle verified 9/9 forward + 9/9 back with byte-identical TeX recovery on a working copy; master intentionally keeps editable TeX text — final production toggle by the author at submission; render QA waived by user 2026-08-20) |
| O3 MVTec AD 2 dataset citation (CITATION_STATUS T10) not located/verified | §5.1 cannot cite dataset | PW3 | OPEN |
| O4 Detector provenance citations (DINOv2/PatchCore/EfficientAD; T11–T13) unverified | §5.2 citations | PW3 | OPEN |
| O5 Per-detector construction details (DINOv2 scoring rule, PatchCore memory/coreset, EfficientAD 70k-step in-house training) must be transcribed from PB1DS/PB7-P0 sealed bindings | §5.2 accuracy | PW3 | OPEN (pointer recorded; EfficientAD in-house status verified) |
| O6 Final scholarly name-collision recheck for NorOpCal | title freeze | PW10 | OPEN (scheduled) |
| O7 Base DINOv2 per-image runtime NOT_VERIFIED_IN_PB8 | cannot claim end-to-end runtime | never (permanent limitation L6) | FIXED_AS_LIMITATION |
| O8 AeBAD-S external validation: gate degenerates to always-on (median shifts 0.039–0.107 > Q50); honest abstention | no external-validation claim | never (limitation) | FIXED_AS_LIMITATION |

No unresolved numeric discrepancy exists between this file and sealed artifacts (contradiction
audit: PB8 Table 1 / F_FINAL_STATISTICS / PB8-S1 `principal_preserved` / PB7-P2 reproduction
agree at their respective precision; micro-differences frozen-estimand vs bootstrap mean are
documented in PW0 `MANUSCRIPT_EVIDENCE_INDEX.md` §E and both round to the reported pp values).

# 15. Evidence Provenance Index

| Ground-truth item | Source artifact | Status |
|---|---|---|
| λ=0.25, Q50=0.015890181064605713, A_MEDIAN_RAW, gate, bounding rule | `reports/pb7_p1_safety_actuator/E_FROZEN_METHOD_SPEC.json` | SEALED |
| Actuator family formulas (τ_raw definitions) | `reports/pb7_p1_safety_actuator/A_ACTUATOR_FAMILY_REGISTRY.json` | SEALED |
| Safety budget 0.02; DINOv2-before-pooled hierarchy | `reports/pb8_publication_benchmark/P0_PUBLICATION_ESTIMAND.json` | SEALED |
| Principal ΔBA/ΔFNR/harm/coverage (B5) + B0–B6 | `reports/pb8_publication_benchmark/TABLE_1_MAIN_BENCHMARK.csv` | SEALED |
| Bootstrap B=10,000 CIs; random-abstention control | `reports/pb8_publication_benchmark/F_FINAL_STATISTICS.csv` | SEALED |
| Random abstention summary | `reports/pb8_publication_benchmark/C_RANDOM_ABSTENTION_SUMMARY.json` | SEALED |
| K4/K8/FULL results + repeat frequencies | `reports/pb8_publication_benchmark/TABLE_4_K_SUPPORT.csv` | SEALED |
| Detector verdicts + K8 numbers | `reports/pb8_publication_benchmark/TABLE_2_DETECTOR_SPECIFICITY.csv` | SEALED |
| Component ablation (paired CIs) | `reports/pb8_publication_benchmark/TABLE_3_COMPONENT_ABLATION.csv` | SEALED |
| Correctability AUC 0.712/0.802; Spearman 0.203; λ curve; abstention nuance; capability semantics | `reports/pb8_publication_benchmark/E_MECHANISM_EVIDENCE.json` | SEALED |
| Limitations L1–L7 | `reports/pb8_publication_benchmark/TABLE_5_SCOPE_AND_FAILURES.csv` | SEALED |
| Overhead 0.136/0.535 ms; memory/storage | `reports/pb8_publication_benchmark/G_COMPUTATIONAL_OVERHEAD.csv` | SEALED |
| PB8 terminal T2 + strongest claim + claim boundaries | `reports/pb8_publication_benchmark/F_PB8_PUBLICATION_VERDICT.json` | SEALED |
| Comparator absence (9 audited, all ineligible) | `reports/pb8_publication_benchmark/A_EXTERNAL_COMPARATOR_ELIGIBILITY.csv` | SEALED |
| K8 breadth: 39/26/10 units; 16 R1; pooling 0/4; 315 images | `reports/pb8_s1_evidence_breadth/F_PB8_S1_BREADTH_VERDICT.json` | SEALED |
| Exact reproduction < 1e-9 | `reports/pb7_p2_publication_validation/A_EXACT_REPRODUCTION.json` | SEALED |
| Confirmatory bootstrap; LOCO; K8 stability | `reports/pb7_p2_publication_validation/D_CONFIRMATORY_BOOTSTRAP.csv`; `B_LOCO_CATEGORY_INFLUENCE.csv`; `C_K_SUPPORT_STABILITY.csv` | SEALED |
| Residual harm 9/78 walnuts; 89% high source FNR | `reports/pb7_p2_publication_validation/E_RESIDUAL_HARM.csv` | SEALED |
| PB7-P2 terminal T1 | `reports/pb7_p2_publication_validation/F_PB7_P2_PUBLICATION_VERDICT.json` | SEALED |
| Claim registry C1–C16; numeric-confusion guards | `reports/manuscript/PW0/MANUSCRIPT_EVIDENCE_INDEX.md` | FROZEN |
| Non-claims (13); terminology; symbol table; section map; figure/table registry | `reports/manuscript/PW0/` (corresponding files) | FROZEN |
| 39-unit dataset/DEV-TEST protocol | `reports/manuscript/PW0/MANUSCRIPT_DATASET_PROTOCOL.csv` | FROZEN |
| Section III equations/symbols as inserted | `reports/manuscript/PW1/PW1_EQUATION_REGISTRY.md`; `PW1_SYMBOL_AUDIT.md` | VERIFIED |
| Manuscript Section III text (consistency check) | `NorOpCal_TIM_Manuscript.docx` (SHA256 1228716d…) | VERIFIED at PW0.5 |
| Naming audit (NorOpCal working name, rejected names) | `reports/pb8_publication_benchmark/METHOD_NAME_CANDIDATES.md` | SEALED |
| Citation firewall (V1–V7 verified; T1–T14 to verify) | `reports/manuscript/PW0/CITATION_STATUS.md` | FROZEN |

---

*End of ground truth. Gates G1–G12: see `PUBLICATION_GROUND_TRUTH_GATES.md` in
`reports/manuscript/PW0_5/` for the machine-checkable audit record.*
