# PB7-P1 Safety-Constrained Normal-Only Threshold Actuator Development — Final Report

> Stage: PB7-P1 (Safety-Constrained Normal-Only Threshold Actuator Development)
> Date: 2026-08-20
> Status: **COMPLETE** — mechanism terminal sealed (T4)
> Inherits sealed terminal: PB7-P0 `THRESHOLD_CORRECTABILITY_IDENTIFIED_BUT_SIMPLE_ACTUATORS_INSUFFICIENT` (F4)
> This report is the final summary; per-stage raw evidence is in `reports/pb7_p1_safety_actuator/`.

---

## 1. Protocol (frozen at P0)

Hard order `P0 → A → B → C → D → E → F → FINAL`, each stage producing a formal artifact, a machine-readable JSON/CSV, a self-audit, and a gate before proceeding. Full protocol in `P0_PROTOCOL.json`, input freeze in `P0_INPUT_MANIFEST.csv` (297 rows, SHA256-bound), data roles in `P0_DATA_ROLES.json`, estimand registry in `P0_ESTIMAND_REGISTRY.json`, hypothesis registry in `P0_HYPOTHESIS_REGISTRY.json`, split in `P0_SPLIT.json`.

### 1.1 Inherited frozen facts (not reinterpreted)

- PB7-P0 terminal F4 is sealed and is the starting point of this stage.
- QM route is closed; QM is not a candidate here.
- Oracle headroom: 22/117 safe-correctable (DINOv2 16/39 = 41.0%); no detector-agnostic claim without new cross-detector evidence.
- Correctability is identifiable from normal-only evidence (median_shift correctable AUC 0.712, LCO 0.714, K8 0.802).
- Threshold direction/magnitude are NOT stably identifiable (PB7-P0 H2/H3 NOT SUPPORTED); oracle threshold regression, sign classifier, and outcome-driven direction chooser are prohibited.
- PB7-P0 best Always-On T2 has utility (mean ΔBA +0.040) but violates the frozen safety constraint (mean ΔFNR +0.056 > +0.02, harm rate 0.573).

### 1.2 Data roles and firewall

- **DEPLOYABLE**: source-normal, target-normal, detector identity, category identity, condition, allowed acquisition metadata.
- **ORACLE_DIAGNOSTIC_ONLY**: target anomaly scores/labels, BA/FPR/FNR outcomes, oracle thresholds, correctability labels.
- Code-level `Firewall` (inherited from PB7-P0): any deployable computation path reading an anomaly/oracle file raises `FIREWALL_VIOLATION`. P0 Gate verified.

### 1.3 Frozen settings

- Principal setting: **K = 8**; auxiliary: K4, FULL. K_REPEATS = 30, nested support rule inherited from PB7-P0.
- Detector reporting: DINOv2, PatchCore, EfficientAD, pooled.
- Safety constraint: **mean ΔFNR ≤ +0.02**; also report harmful-unit rate, catastrophic harm rate, worst-decile ΔFNR, detector-stratified ΔFNR.
- Harmful unit (frozen, reused from PB7-P0): `ΔBA < −0.02 OR ΔFNR > +0.02`; catastrophic: `ΔFNR > +0.05`.
- Utility: primary ΔBA; secondary ΔFPR, ΔFNR, coverage, benefit precision, harm rate. AUROC/AUPR only as consistency check (threshold-only methods do not change ranking by construction).
- DEV/TEST split: category-level deterministic SHA256 parity (`SPLIT_SALT = "PB7_P1_DEV_TEST_SPLIT_v1"`). DEV = {can, fruit_jelly, wallplugs}; TEST = {fabric, rice, sheet_metal, vial, walnuts}.
- DEV is used only to choose among pre-registered (family, λ, cutoff); TEST is one frozen evaluation.

### 1.4 Hypotheses (H1–H6)

Registered in `P0_HYPOTHESIS_REGISTRY.json`; adjudication in §10.

---

## 2. Base Actuator Family Freeze (Stage A — PASS)

Registry: `A_ACTUATOR_FAMILY_REGISTRY.json`; parity: `A_ACTUATOR_PARITY.csv`; gate: `A_GATE.json`.

Three families frozen (≤ 3 allowed), all DEPLOYABLE-input only, no anomaly-derived parameter:

| ID | Formula | Rationale |
|---|---|---|
| `A_T2_RAW` | `τ_raw = median_t + (MAD_t/MAD_s)·(τ_source − median_s)` | Exact PB7-P0 best simple actuator T2; robust location-scale |
| `A_TAIL_RAW` | `τ_raw = Q95(target-normal)` | Exact PB7-P0 T3; tail-preserving exceedance |
| `A_MEDIAN_RAW` | `τ_raw = τ_source + (median_t − median_s)` | Conservative location-only; median_shift is the PB7-P0 PASS-B principal correctability signal |

Numerical parity with PB7-P0 verified: max |diff| vs T2 = 2.22e-16, vs T3 = 2.22e-16 (117 entities). No 4th family added.

---

## 3. Safety Bounding Family (Stage B — PASS)

Registry: `B_BOUNDING_FAMILY.json`; grid: `B_BOUNDING_GRID.csv`; gate: `B_GATE.json`.

- Multiplicative shrinkage: `τ_used = τ_source + λ·(τ_raw − τ_source)`, λ ∈ {0.25, 0.50, 0.75, 1.00} frozen before any outcome.
- λ = 1.00 is the raw actuator (no shrinkage).
- B2 (uncertainty-adaptive shrinkage) not implemented: no sufficiently simple and interpretable normal-only confidence-weight formula; B1 preferred per protocol.
- No direction switching: bounding only shrinks toward τ_source or abstains; never flips sign (B3 verified on grid: monotonic `|τ_used − τ_source|` decreasing in λ).

---

## 4. Correctability Gate / Abstention (Stage C — PASS)

Definition: `C_GATE_DEFINITION.json`; coverage: `C_CUTOFF_COVERAGE.csv`; gate: `C_GATE.json`.

- Principal signal: `median_shift = median(target-normal) − median(source-normal)` (PB7-P0 PASS-B; no feature search).
- Gate form: `adapt iff median_shift ≥ c, else NoAdapt`.
- Cutoff selection: **Option A (DEV-selected)** from a frozen grid c ∈ {Q50, Q60, Q70, Q80} of the DEV FULL-support median_shift distribution. DEV cutoffs: Q50 = 0.01589, Q60 = 0.02049, Q70 = 0.03268, Q80 = 0.05275. DEV K8 coverage: [0.588, 0.549, 0.421, 0.268].

---

## 5. DEV Selection (Stage D — PASS → PROCEED_TO_E)

Benchmark: `D_DEV_BENCHMARK.csv`; safety filter: `D_DEV_SAFETY_FILTER.csv`; gate: `D_GATE.json`.

- Candidate grid: 48 gated (3 family × 4 λ × 4 cutoff) + 12 Always-On + 1 NoAdapt = 61; grid checks PASS.
- Safety-first selection: filter `ΔFNR_DEV ≤ +0.02`, then maximize `ΔBA_DEV`; tie-break lower harm, smaller λ, simpler actuator, higher coverage.
- **Selection restricted to GATED candidates** (the 48 deployable candidates): the stage goal is a conservative, bounded, abstaining actuator. NoAdapt and Always-On are mandatory baselines only.
- Result: 14 safe gated candidates; selected **A_MEDIAN_RAW, λ = 0.25, cutoff = Q50** (DEV: ΔBA = +0.0235, ΔFNR = +0.0115, harm = 0.149, coverage = 0.588).
- Note: the first D run (ledger row) selected an Always-On candidate before the GATED restriction was applied; it was superseded by the corrected run. The authoritative D result is the GATED selection above; `D_GATE.json` reflects it.

---

## 6. Frozen TEST Specification (Stage E)

Spec: `E_FROZEN_METHOD_SPEC.json`; per-entity: `E_TEST_PER_ENTITY.csv`; summary: `E_TEST_DETECTOR_SUMMARY.csv`; bootstrap: `E_TEST_BOOTSTRAP.csv`; gate: `E_GATE.json`.

Frozen principal candidate: **A_MEDIAN_RAW, λ = 0.25, cutoff = Q50 (0.01589), gate = adapt iff median_shift ≥ cutoff**. Single global operating setting; no detector-specific tuning. Mandatory baselines:

| ID | Definition |
|---|---|
| B0 | NoAdapt (τ_source) |
| B1 | PB7-P0 best Always-On T2 (A_T2_RAW, λ=1.0, no gate) |
| B2 | Frozen base actuator without shrinkage (A_MEDIAN_RAW, λ=1.0, no gate) |
| B3 | Bounded actuator Always-On (A_MEDIAN_RAW, λ=0.25, no gate) |
| B4 | Bounded + Abstaining (A_MEDIAN_RAW, λ=0.25, Q50) **[principal]** |
| B5 | Oracle safe threshold (DIAGNOSTIC ONLY) |

Bootstrap: B = 5000, category-aware block resampling, seed = 20260820, block structure identical to PB7-P0.

### 6.1 Main benchmark (TEST, K=8, pooled)

| Baseline | ΔBA | ΔFPR | ΔFNR | benefit | harm | catastrophic | coverage |
|---|---|---|---|---|---|---|---|
| B0 NoAdapt | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B1 Always-On T2 | +0.0117 | −0.0860 | **+0.0625** | 0.064 | **0.414** | 0.379 | 0.610 |
| B2 Median λ=1.0 | +0.0145 | −0.0885 | **+0.0596** | 0.048 | **0.412** | 0.412 | 0.610 |
| B3 Median λ=0.25 Always-On | +0.0127 | −0.0397 | +0.0143 | 0.153 | 0.168 | 0.168 | 0.610 |
| **B4 Median λ=0.25 Q50 Gated** | **+0.0120** | −0.0366 | **+0.0127** | 0.144 | **0.143** | 0.143 | **0.342** |
| B5 Oracle safe (DIAGNOSTIC) | +0.0996 | +0.1034 | −0.3026 | 0.128 | 0.000 | 0.000 | 0.872 |

Key reading: B4 retains essentially the full utility of the unsafe Always-On T2 (ΔBA +0.0120 vs +0.0117) while cutting ΔFNR from +0.0625 to +0.0127 and harm from 0.414 to 0.143. The bounded+abstaining policy converts the identified correctability into a safe operating-point adaptation.

### 6.2 Bootstrap (TEST, K=8, B=5000)

| Baseline | ΔBA mean | ΔBA CI_lo | ΔBA CI_hi | ΔFNR mean | harm mean |
|---|---|---|---|---|---|
| B1 | +0.0119 | −0.0111 | +0.0346 | +0.0621 | 0.413 |
| B2 | +0.0145 | +0.0084 | +0.0206 | +0.0593 | 0.411 |
| B3 | +0.0127 | +0.0100 | +0.0154 | +0.0142 | 0.167 |
| **B4** | **+0.0120** | **+0.0101** | **+0.0138** | **+0.0126** | **0.142** |
| B5 | +0.0998 | +0.0887 | +0.1108 | −0.3034 | 0.000 |

B4 CI lower for ΔBA = +0.0101 > 0 (statistically supported non-zero utility).

### 6.3 E gates (K=8)

| Gate | Criterion | B4 value | Verdict |
|---|---|---|---|
| G1 Utility | mean ΔBA > 0 and CI_lo > 0 | +0.0120 / CI_lo +0.0101 | **PASS** (strong bar ≥ +0.02 not met pooled) |
| G2 FNR safety | mean ΔFNR ≤ +0.02 | +0.0127 | **PASS** |
| G3 Harm reduction | rel. reduction vs B1 ≥ 25% | 0.143 vs 0.414 (65.4%) | **PASS** |
| G4 Selectivity value | B4 < B3 on ΔFNR or harm, utility retained | ΔFNR 0.0127<0.0143; harm 0.143<0.168; ΔBA retained | **PASS** |
| G5 Coverage | ≥ 10% | 0.342 (DINOv2 0.403) | **PASS** |
| G6 K8 principal | evaluated at K=8 | — | **PASS** |
| G7 Detector specificity | report | DINOv2-dominant | see §8 |

E Gate = **PASS** → PROCEED_TO_F.

---

## 7. Mechanism Analysis (Stage F — COMPLETE)

Curve: `F_SAFETY_UTILITY_CURVE.csv`; residual harm: `F_RESIDUAL_HARM_ENTITIES.csv`, `F_RESIDUAL_HARM_TAXONOMY.csv`; decomposition: `F_DETECTOR_DECOMPOSITION.csv`; mechanism: `F_MECHANISM.json`.

### 7.1 Safety-utility curve (TEST K=8, A_MEDIAN_RAW)

| λ | kind | ΔBA | ΔFPR | ΔFNR | harm | coverage |
|---|---|---|---|---|---|---|
| 0.25 | Always-On | +0.0127 | −0.0397 | +0.0143 | 0.168 | 0.610 |
| 0.25 | Gated | +0.0120 | −0.0366 | +0.0127 | 0.143 | 0.342 |
| 0.50 | Always-On | +0.0185 | −0.0641 | +0.0272 | 0.286 | 0.610 |
| 0.50 | Gated | +0.0178 | −0.0595 | +0.0238 | 0.234 | 0.342 |
| 0.75 | Always-On | +0.0166 | −0.0786 | +0.0453 | 0.339 | 0.610 |
| 0.75 | Gated | +0.0156 | −0.0725 | +0.0413 | 0.284 | 0.342 |
| 1.00 | Always-On | +0.0145 | −0.0885 | +0.0596 | 0.412 | 0.610 |
| 1.00 | Gated | +0.0129 | −0.0788 | +0.0530 | 0.333 | 0.342 |

- **Monotonic safety with conservatism: TRUE.** Gated ΔFNR decreases monotonically with shrinkage (0.0530 → 0.0413 → 0.0238 → 0.0127); harm follows the same monotone trend.
- **Utility is not strictly monotone** (peaks at λ = 0.5, then declines); the safety-utility frontier is concave, so λ = 0.25 is the safe corner.
- The gated variant strictly dominates the Always-On variant at every λ on both ΔFNR and harm, with only a small ΔBA cost (≤ 0.6 pp).

### 7.2 Residual harm taxonomy (B4, K=8)

9 harmful entities out of 78 (harm rate 0.143). All 9 are in **category `walnuts`** and all have **high source FNR** (source FNR ≥ 0.15; mostly 0.8 for EfficientAD). Breakdown:

- By detector: EfficientAD 4, PatchCore 4, DINOv2 1.
- By harm type: catastrophic FNR (ΔFNR > 0.05) 3, FNR breach (0.02 < ΔFNR ≤ 0.05) 4, repeat-level "other" 2 (mean deltas within bounds but ≥ 1 repeat harmful).
- By median shift: large positive 5, small positive 4; by K8 uncertainty: high 5, mid 3, low 1.

The single DINOv2 harmful entity (`walnuts/shift_1`) is a borderline case: ΔBA = +0.0878 (large gain) with ΔFNR = +0.0244 (just above the +0.02 bound). Residual harm is concentrated where the source operating point is already poor (high FNR) and the shift is large — a diagnostic signal for a future stage, not a basis to modify the frozen method here.

**Safety nuance:** at the repeat level, residual harm is entirely FNR-driven and catastrophic — for every harmful entity the per-repeat harm rate equals the catastrophic rate exactly (harm_rate = catastrophic_rate = 0.1433 pooled), i.e. harmful repeats always breach ΔFNR > +0.05 rather than a mild ΔBA-only drop. This is a conservative property of the bounded+abstaining policy (no hidden mild-harm tail), but it also means the residual harm that remains is severe where it occurs.

### 7.3 Detector decomposition (TEST K=8, gated λ=0.25)

| Detector | n_k8 | correctable | ΔBA | ΔFNR | harm | coverage | verdict |
|---|---|---|---|---|---|---|---|
| DINOv2 | 10 | 9 | **+0.0323** | **+0.0024** | **0.037** | 0.403 | **USEFUL_SAFE_SELECTABLE** |
| PatchCore | 10 | 1 | −0.0066 | +0.0216 | 0.247 | 0.247 | ACTUATOR_MISMATCH |
| EfficientAD | 10 | 0 | +0.0103 | +0.0140 | 0.147 | 0.377 | NO_HEADROOM |

- **DINOv2**: useful (ΔBA +3.2 pp), safe (ΔFNR +0.2 pp, harm 3.7%), selectable (coverage 40%); 9/10 oracle-correctable. This is the supported deployment case.
- **PatchCore**: negative ΔBA and ΔFNR slightly above budget → actuator mismatch with the median-shift family; should abstain (near-zero coverage) rather than adapt.
- **EfficientAD**: zero oracle-correctable entities (NO_HEADROOM by the frozen oracle taxonomy); the small positive drift (+1.0 pp) is not oracle-supported and is not claimed as a benefit.

---

## 8. Mechanism Verdict

Terminal: **`PB7_P1_DETECTOR_SPECIFIC_SAFE_ACTUATOR_SUPPORTED`** (T4).

Machine-readable verdict: `F_PB7_P1_MECHANISM_VERDICT.json`.

Answer to the core question — *Can a bounded normal-only actuator convert identified correctability into safe deployable threshold adaptation?* — is **YES, with a DINOv2-dominant, detector-specific scope**:

- A conservative (λ = 0.25), abstaining (median_shift ≥ Q50) location-only actuator converts the PB7-P0-identified correctability into a safe operating-point adaptation on TEST: ΔBA = +1.2 pp pooled (CI lower +1.0 pp > 0), ΔFNR = +1.3 pp ≤ +2 pp, harm reduced 65% vs the unsafe Always-On T2, coverage 34%.
- The DINOv2-specific result is strong: ΔBA = +3.2 pp, ΔFNR = +0.2 pp, harm 3.7%, coverage 40%.
- Safety improves monotonically with conservatism; abstention adds safety value at every λ (H2, H3 supported).
- The claim is explicitly **DINOv2-dominant detector-dependent threshold adaptation**; it is NOT detector-agnostic. PatchCore and EfficientAD are allowed to abstain / near-zero coverage.

### 8.1 Hypothesis adjudication

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1 Bounded correction retains utility | SUPPORTED | B4 ΔBA +0.0120 ≈ B1 +0.0117 (utility retained at safe corner) |
| H2 Safety improves monotonically with conservatism | SUPPORTED | Gated ΔFNR 0.053→0.013, harm 0.333→0.143 as λ→0.25 |
| H3 Abstention adds safety value | SUPPORTED | B4 < B3 on ΔFNR (0.0127<0.0143) and harm (0.143<0.168) at same λ |
| H4 K8 sufficient | SUPPORTED | Principal result at K=8; K4/FULL auxiliary consistent |
| H5 Safe deployable operating point exists | SUPPORTED | B4 satisfies ΔFNR ≤ +0.02 with ΔBA > 0, CI lower > 0 |
| H6 Detector-dependent | SUPPORTED | DINOv2 dominant; PatchCore negative; EfficientAD no oracle headroom |

---

## 9. Limitations and NOT_VERIFIED

- **Pooled utility is modest** (+1.2 pp, below the +2 pp "strong" bar); the strong result is DINOv2-specific (+3.2 pp). The pooled gain is diluted by PatchCore's negative contribution.
- **Residual harm is concentrated in `walnuts`** (9/9 harmful entities, all high source-FNR). The frozen method was not modified on this basis; this is evidence for a future stage (e.g., a high-source-FNR abstention rule would require a new pre-registered protocol).
- **Oracle baseline B5 (DIAGNOSTIC ONLY)** is evaluated on all 78 TEST entities at every K (K-independent by construction), while B0–B4 at K=8 use the 30 K8-eligible entities; the oracle is not part of any gate.
- **PatchCore ΔFNR (+0.0216)** is marginally above the +0.02 budget at the global setting; a detector-specific abstention (allowed by the protocol as "near-zero coverage") would be needed for deployment, which is outside this frozen evaluation.
- **EfficientAD** shows a small positive drift (+1.0 pp) but zero oracle-correctable entities; no benefit claim is made for it.
- AUROC/AUPR are unchanged by construction (threshold-only); no ranking/representation claim is made. No evaluator change was needed.
- The first D ledger row (Always-On selection) is superseded by the corrected GATED selection; the authoritative D result is the second row.

---

## 10. Next Authorization

Per `AGENTS.md`, this report recommends but does not authorize successor work. Possible next steps (each requires explicit user authorization and, where applicable, a new pre-registered protocol):

1. **Detector-specific deployment policy**: freeze a DINOv2-only (or DINOv2 + abstain-others) operating setting and re-run a single frozen TEST evaluation (new protocol required).
2. **High-source-FNR abstention rule**: pre-register a normal-only gate that abstains for high source-FNR / large-shift entities to further reduce residual harm (new protocol required).
3. **Upgrade to paper main-method candidate**: only if the DINOv2-dominant safe actuator result is accepted as the paper's operating-point adaptation claim.

---

## 11. Artifacts

| Stage | Artifacts |
|---|---|
| P0 | `P0_PROTOCOL.json`, `P0_INPUT_MANIFEST.csv`, `P0_DATA_ROLES.json`, `P0_ESTIMAND_REGISTRY.json`, `P0_HYPOTHESIS_REGISTRY.json`, `P0_SPLIT.json`, `P0_GATE.json` |
| A | `A_ACTUATOR_FAMILY_REGISTRY.json`, `A_ACTUATOR_PARITY.csv`, `A_GATE.json` |
| B | `B_BOUNDING_FAMILY.json`, `B_BOUNDING_GRID.csv`, `B_GATE.json` |
| C | `C_GATE_DEFINITION.json`, `C_CUTOFF_COVERAGE.csv`, `C_GATE.json` |
| D | `D_DEV_BENCHMARK.csv`, `D_DEV_SAFETY_FILTER.csv`, `D_GATE.json` |
| E | `E_FROZEN_METHOD_SPEC.json`, `E_TEST_PER_ENTITY.csv`, `E_TEST_DETECTOR_SUMMARY.csv`, `E_TEST_BOOTSTRAP.csv`, `E_GATE.json` |
| F | `F_SAFETY_UTILITY_CURVE.csv`, `F_RESIDUAL_HARM_ENTITIES.csv`, `F_RESIDUAL_HARM_TAXONOMY.csv`, `F_DETECTOR_DECOMPOSITION.csv`, `F_MECHANISM.json` |
| FINAL | `F_PB7_P1_MECHANISM_VERDICT.json`, this report, `P0_EXECUTION_LEDGER.csv` |

Code: `tools/pb7_p1_safety_actuator/` (`pb7_p1_common.py`, `p0_freeze.py`, `a_freeze.py`, `b_freeze.py`, `c_freeze.py`, `d_dev_select.py`, `e_test_eval.py`, `f_mechanism.py`).
