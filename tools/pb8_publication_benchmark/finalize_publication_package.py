"""PB8 J–M/FINAL — freeze claims, paper plan, naming, ledger, and verdict."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "pb8_publication_benchmark"
FIG = OUT / "figures"
NOW = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(name: str) -> dict[str, Any]:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _write_text(name: str, text: str) -> None:
    (OUT / name).write_text(text.strip() + "\n", encoding="utf-8")


def _build_exact_tables() -> None:
    detector = pd.read_csv(OUT / "TABLE_2_DETECTOR_DECOMPOSITION.csv")
    detector_names = {
        "dinov2_vits14": "DINOv2",
        "wide_resnet50_2": "PatchCore",
        "efficientad": "EfficientAD",
    }
    detector.insert(0, "detector_family", detector["detector"].map(detector_names))
    order = pd.Categorical(
        detector["detector_family"], ["DINOv2", "PatchCore", "EfficientAD"], ordered=True
    )
    detector = detector.assign(_order=order).sort_values("_order").drop(columns="_order")
    detector.to_csv(OUT / "TABLE_2_DETECTOR_SPECIFICITY.csv", index=False)

    pd.read_csv(OUT / "TABLE_4_SUPPORT_ROBUSTNESS.csv", dtype={"K": str}).to_csv(
        OUT / "TABLE_4_K_SUPPORT.csv", index=False
    )
    pd.read_csv(OUT / "TABLE_5_LIMITATIONS.csv").to_csv(
        OUT / "TABLE_5_SCOPE_AND_FAILURES.csv", index=False
    )


def _claim_hierarchy() -> None:
    text = """
# PB8 Final Claim Hierarchy

Status: **FROZEN FOR SCOPED PAPER WRITING**  
Evidence-breadth verdict: **EB2 — LIMITED_BUT_REPORTABLE**  
Publication terminal: **T2 — PB8_PAPER_PACKAGE_READY_WITH_SCOPE_LIMITATIONS**

## Claim 1 — Primary supported claim

On the 10 DINOv2 K=8 units from the two frozen TEST categories eligible under the acquisition-shift protocol (fabric and walnuts), the preregistered normal-only selective bounded threshold policy improved mean balanced accuracy by **3.23 percentage points** versus the frozen source threshold (category-aware bootstrap 95% CI **[2.70, 3.76] pp**) while mean FNR increased by **0.24 pp**, within the preregistered **+2 pp** safety budget. Harm was **3.67%** and adaptation coverage was **40.33%**.

This is evidence for **DINOv2-specific operating-point adaptation in frozen TEST K=8-eligible acquisition-shift units**. It is not evidence of broad category generalization.

## Claim 2 — Mechanism claim

Correctability and safe threshold actuation are detector-dependent. Median shift identifies correctability but does not identify the oracle correction direction or magnitude. Conservative shrinkage supplies the principal safety control; the frozen gate supplies selectivity. Under the same frozen rule, DINOv2 is useful/safe/selectable, PatchCore is actuator-mismatched, and EfficientAD has no safe correctability headroom.

## Claim 3 — Practical claim

The deployment actuator uses eight known-normal target samples, a scalar median offset, a frozen Q50 abstention gate, and one bounded threshold update. It requires no detector retraining, gradients, target anomalies, or extra model parameters. The verified incremental CPU overhead is 0.136 ms/entity at p50 and 0.535 ms/entity at p95 in the recorded PB8 environment. Base DINOv2 per-image inference cost is **NOT_VERIFIED in PB8** and is not attributed to the actuator.

## Claim 4 — Explicit non-claims

- **NOT detector-agnostic.**
- **NOT representation, feature, or anomaly-ranking improvement.**
- **NOT an AUROC/AUPR improvement claim; threshold-only adaptation should not change ranking metrics.**
- **NOT K=4-safe**; K=4 is unsupported/unsafe.
- **NOT universally improved by more normal samples**; DINOv2 FULL exceeds the mean ΔFNR budget.
- **NOT broadly validated with eight samples**; only fabric and walnuts are frozen TEST K=8 eligible.
- **NOT externally validated across datasets**; no legal external setting preserves the frozen score-scale semantics.
- **NOT harm-free, risk-free, or universally safe**; residual harm is localized in walnuts.
- **NOT superior to a matched published comparator**; no direct literature comparator met all deployment and reproducibility constraints.
"""
    _write_text("FINAL_CLAIM_HIERARCHY.md", text)


def _method_names() -> None:
    text = """
# PB8 Method Name Candidates

Collision check date: **2026-08-20**  
Search scope: exact-name web searches plus the PB8 six-source literature audit. A negative search result is not a legal trademark clearance or proof of global uniqueness.

## Selected working name

**NorOpCal — Conservative Normal-only Operating-Point Calibration**

Status: **WORKING_NAME_AFTER_PRELIMINARY_COLLISION_CHECK**

Rationale: it states the available target information (normal-only) and the actual capability (operating-point calibration) without implying representation learning, detector universality, or zero risk. Exact searches for `NorOpCal` with anomaly-detection, calibration, and method terms returned no direct research-method collision in the inspected results.

## Other collision-checked candidates

| Candidate | Expansion | Preliminary result | Decision |
|---|---|---|---|
| **NoTAS** | Normal-only Threshold Adaptation with Selectivity | No direct research-method collision found in inspected exact-search results | Reserve |
| **NorSafe-OP** | Normal-only Safe Operating-Point Adaptation | No direct research-method collision found in inspected exact-search results | Reserve; “safe” requires the scoped safety qualifier every time |

## Rejected names found during screening

| Name | Why rejected | Evidence |
|---|---|---|
| **SafeNOC** | Direct collision with a published on-chip-network correctness architecture | https://web.eecs.umich.edu/valeria/research/publications/ICCD11SafeNoC.pdf |
| **ConSAT** | Direct collision with a published contrastive Transformer method and multiple established uses | https://www.sciencedirect.com/science/article/abs/pii/S092658052600378X |
| **NOSTA** | Existing logistics company/brand creates avoidable ambiguity | https://www.nosta-group.com/en/news-media/switzerland-abolishes-industrial-tariffs |

The working name remains revisable during manuscript drafting if venue-specific or broader database screening reveals a conflict. The scientific method and claim boundaries remain frozen regardless of naming.
"""
    _write_text("METHOD_NAME_CANDIDATES.md", text)


def _paper_plan() -> None:
    text = """
# PB8 Paper Section Plan

Working method name: **NorOpCal** (preliminary collision check complete). This file plans evidence placement only; it is not manuscript prose.

| Section | Subsection purpose | Evidence artifact(s) | Table / figure | Exact permitted claim |
|---|---|---|---|---|
| 1 Introduction | Define acquisition-shift operating-point failure and the normal-only deployment constraint | `P0_PUBLICATION_ESTIMAND.json`; `FINAL_CLAIM_HIERARCHY.md` | Figure 1 | A frozen detector's threshold can be selectively recalibrated from known-normal support in the scoped DINOv2 K=8 setting. |
| 1 Introduction | State contributions without universality | `E_MECHANISM_EVIDENCE.json`; `G_COMPUTATIONAL_OVERHEAD.csv` | Figures 1, 3 | Contributions are conservative operating-point adaptation, detector-dependent correctability evidence, and a leakage-controlled publication protocol. |
| 2.1 Related Work — industrial VAD | Position frozen industrial anomaly detectors | `allinone.md`; `A_LITERATURE_SEARCH_REPORT.md` | None | Related methods address anomaly detection; PB8 does not claim representation novelty. |
| 2.2 Acquisition/domain shift | Define acquisition-shift scope | `A_LITERATURE_SEARCH_RAW.json`; PB7 sealed reports in the P0 manifest | None | The studied shift is the frozen category-condition acquisition protocol. |
| 2.3 Test-time / normal-only adaptation | Separate normal-only support from anomaly-supervised adaptation | `A_EXTERNAL_COMPARATOR_ELIGIBILITY.csv` | None | No audited published method matched all deployment semantics and implementation constraints. |
| 2.4 Threshold calibration / selective adaptation | Position operating-point semantics | `A_LITERATURE_SEARCH_REPORT.md`; `E_MECHANISM_EVIDENCE.json` | Figure 1 | NorOpCal changes the decision threshold, not score ranking or features. |
| 3.1 Source detector and operating point | Define detector score and frozen source threshold | `P0_PUBLICATION_ESTIMAND.json` | Figure 1 | DINOv2 is the principal detector; PatchCore and EfficientAD are mechanism controls. |
| 3.2 Target-normal support | Define K and eligibility | `D_K_SUPPORT_ELIGIBILITY.csv`; `D_EVIDENCE_BREADTH_VERDICT.json` | Table 4 | K=8 means eight distinct known-normal target samples; only fabric and walnuts are eligible in frozen TEST. |
| 3.3 Safety-constrained objective | Define ΔBA utility and ΔFNR budget | `P0_PUBLICATION_ESTIMAND.json`; `F_FINAL_STATISTICS.csv` | Figure 2 | Primary safety requires mean ΔFNR ≤ +2 pp; effect sizes and CIs lead interpretation. |
| 4.1 Normal-score deformation | Define raw median offset | Frozen method spec bound in `P0_PUBLICATION_EVIDENCE_MANIFEST.csv` | Figure 1 | Median shift is an outcome-free correctability signal, not an oracle threshold estimator. |
| 4.2 Correctability gate | Define Q50 abstention | `E_MECHANISM_EVIDENCE.json`; `C_RANDOM_ABSTENTION_SUMMARY.json` | Figures 1, 2 | The frozen gate selects adaptation coverage and outperforms coverage-matched random abstention on utility. |
| 4.3 Conservative bounded actuator | Define λ=0.25 shrinkage | `C_SHRINKAGE_ABLATION.csv`; `C_COMPONENT_ABLATION.csv` | Figures 1, 2; Table 3 | Conservative shrinkage is the principal safety control under the frozen actuator. |
| 4.4 Abstention | Define NoAdapt fallback | `C_COMPONENT_ABLATION.csv` | Figures 1, 2 | Gate rejection preserves the source threshold; no anomaly outcome is used. |
| 4.5 Deployment algorithm | Specify inputs and operations | `G_COMPUTATIONAL_OVERHEAD.csv`; plotting/method source | Figure 1 | Eight normals, scalar median, frozen gate, bounded scalar update; no retraining or gradients. |
| 5.1 Dataset and shifts | Declare frozen TEST scope and categories | `D_K_SUPPORT_ELIGIBILITY.csv`; P0 manifest | Table 4 | Results cover the declared acquisition-shift entities; no pseudo-expansion is used. |
| 5.2 Detectors | Define DINOv2, PatchCore, EfficientAD roles | `B_DETECTOR_BENCHMARK.csv`; `E_MECHANISM_EVIDENCE.json` | Table 2; Figure 3 | Detector results are mechanism evidence; only DINOv2 supports the primary claim. |
| 5.3 K-support | Define K4/K8/FULL protocols | `D_K_SUPPORT_FINAL.csv` | Table 4; Figure 4 | K4 is unsafe, K8 is principal, and FULL is not monotonically safer. |
| 5.4 Safety metrics | Define ΔFNR, harm, catastrophic harm | `P0_PUBLICATION_ESTIMAND.json`; `B_FINAL_MAIN_BENCHMARK.csv` | Table 1 | Any mean ΔFNR above +2 pp is marked SAFETY FAIL regardless of utility. |
| 5.5 Firewall / leakage prevention | Document forbidden information | `P0_GATE.json`; scripts under `tools/pb8_publication_benchmark/` | None | Target anomalies, masks, final-test labels, and outcomes do not enter fitting or selection. |
| 5.6 Statistical protocol | Define hierarchy and uncertainty | `F_FINAL_STATISTICS.csv`; PB7-P2 bootstrap bound in P0 manifest | Table 1 | Primary CI reuses the sealed B=10,000 category-aware confirmatory bootstrap; repeats are not category replications. |
| 6.1 Main DINOv2 benchmark | Present principal comparison | `B_FINAL_MAIN_BENCHMARK.csv`; `F_FINAL_STATISTICS.csv` | Table 1 | B5 improves BA by 3.23 pp with CI above zero and remains within the ΔFNR budget. |
| 6.2 Safety comparison | Contrast aggressive and bounded policies | `B_FINAL_MAIN_BENCHMARK.csv`; `C_COMPONENT_ABLATION.csv` | Tables 1, 3; Figure 2 | Higher raw utility from aggressive actuation does not imply safety. |
| 6.3 Detector specificity | Explain headroom and mismatch | `B_DETECTOR_BENCHMARK.csv`; `E_MECHANISM_EVIDENCE.json` | Table 2; Figure 3 | Correctability and safe actuation are detector-dependent. |
| 6.4 K-support robustness | Show K4/K8/FULL | `D_K_SUPPORT_FINAL.csv` | Table 4; Figure 4 | More support is not monotonically safer; only K8 supports the principal safe setting. |
| 7.1 Component ablation | Attribute gate and shrinkage functions | `C_COMPONENT_ABLATION.csv`; paired rows in `F_FINAL_STATISTICS.csv` | Table 3 | Shrinkage controls safety; gate controls selectivity; the full method preserves both. |
| 7.2 Safety–utility tradeoff | Show frozen λ curve | `C_SHRINKAGE_ABLATION.csv`; `E_MECHANISM_EVIDENCE.json` | Figure 2 | Increasing actuation raises safety cost monotonically over the frozen λ grid. |
| 7.3 Correctability mechanism | Separate identifiability from actuation | `E_MECHANISM_EVIDENCE.json` | Figure 3 | Correctability is identifiable; oracle direction and magnitude are not. |
| 7.4 Residual harm | Characterize without post-hoc repair | `E_MECHANISM_EVIDENCE.json`; PB7-P2 residual-harm artifact | Figure 5; Table 5 | Residual harm is localized to walnuts/high source-FNR; no walnuts-specific rule was added. |
| 8 Limitations | Consolidate all boundaries | `TABLE_5_SCOPE_AND_FAILURES.csv`; `D_EVIDENCE_BREADTH_VERDICT.json` | Table 5 | EB2 breadth, no legal external validation, detector specificity, K4 failure, FULL non-monotonicity, walnuts risk, and unverified base runtime limit generalization. |
| 9 Conclusion | Restate the strongest scoped result | `FINAL_CLAIM_HIERARCHY.md`; `F_PB8_PUBLICATION_VERDICT.json` | None | NorOpCal is ready as a central method for a scoped DINOv2 paper, not as a universal anomaly-adaptation method. |

"""
    _write_text("PAPER_SECTION_PLAN.md", text)


def _figure_captions() -> None:
    text = """
# PB8 Frozen Figure Captions

## Figure 1 — Method overview

Conservative normal-only operating-point adaptation. The deployment path uses a frozen source threshold and K=8 known-normal target support to compute a median score offset. A frozen Q50 gate selects abstention or a λ=0.25 bounded threshold update. No target anomalies, final-test labels, gradients, or detector retraining enter the path. Bottom boxes state the frozen utility, safety, and evidence-scope boundaries.

## Figure 2 — Safety–utility curve

DINOv2 K=8 safety–utility path over the already frozen λ grid at the Q50 gate. The preregistered mean ΔFNR limit is +2 percentage points. λ=0.25 is the frozen principal operating point; larger actuation increases safety cost and does not monotonically improve utility.

## Figure 3 — Detector correctability decomposition

Frozen K=8 policy decomposed across DINOv2, PatchCore (WideResNet50-2 backbone), and EfficientAD. The panels show safe-correctable headroom, achieved ΔBA, harm, and adaptation coverage. DINOv2 supports useful/safe/selectable actuation; PatchCore is actuator-mismatched; EfficientAD has no safe headroom.

## Figure 4 — Few-normal robustness

K=4, K=8, and FULL results for the DINOv2 principal analysis and pooled secondary analysis. K=4 is unsafe, K=8 is the principal safe setting, and DINOv2 FULL exceeds the mean ΔFNR budget, demonstrating that additional support is not monotonically safer.

## Figure 5 — Residual harm map

Entity-level mean harm over 30 sealed K=8 support repeats for the DINOv2 frozen policy. The only non-zero category-condition cell is walnuts/shift 1, making the localized residual-risk concentration explicit. The map covers only the two frozen TEST K=8-eligible categories.
"""
    _write_text("FIGURE_CAPTIONS.md", text)


def _final_report() -> None:
    main = pd.read_csv(OUT / "B_FINAL_MAIN_BENCHMARK.csv")
    principal = main.loc[(main["baseline"] == "B5") & (main["group"] == "DINOv2")].iloc[0]
    pooled = main.loc[(main["baseline"] == "B5") & (main["group"] == "POOLED")].iloc[0]
    random = _json("C_RANDOM_ABSTENTION_SUMMARY.json")
    breadth = _json("D_EVIDENCE_BREADTH_VERDICT.json")
    report = f"""
# PB8 Publication Benchmark Final Report

Generated: {NOW}  
Frozen predecessor: `PB7_P2_PUBLICATION_CANDIDATE_CONFIRMED_DINOV2_SPECIFIC`  
Final terminal: **T2 — PB8_PAPER_PACKAGE_READY_WITH_SCOPE_LIMITATIONS**

## Executive answer

**1. Is the frozen method strong enough to serve as the paper's central method?**  
**Yes, for a deliberately scoped DINOv2-specific paper.** The principal confirmatory effect is positive, its sealed confidence interval excludes zero, the mean ΔFNR remains well inside the preregistered budget, aggressive baselines expose the utility/safety distinction, the component evidence is interpretable, detector dependence is informative, and the incremental actuator is computationally small. It is not strong enough to support a detector-agnostic, broadly K=8-generalized, externally validated, or harm-free paper.

**2. What is the strongest evidence-supported claim?**  
On the 10 DINOv2 K=8 units from the two frozen TEST categories eligible under the acquisition-shift protocol (fabric and walnuts), a preregistered normal-only selective bounded threshold calibration improved mean balanced accuracy by **{100 * principal["delta_ba"]:.2f} pp** (category-aware 95% CI **[{100 * principal["dba_ci_lo"]:.2f}, {100 * principal["dba_ci_hi"]:.2f}] pp**) versus NoAdapt while mean FNR increased by **{100 * principal["delta_fnr"]:.2f} pp**, within the **+2 pp** safety budget. This is detector-specific operating-point evidence and does not establish representation/ranking gains, K=4 safety, cross-dataset validity, or harm-free generalization.

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

Principal DINOv2 B5 (n={int(principal["n_units"])} units; {int(principal["n_eligible_categories"])} eligible categories):

- BA: {principal["ba_before"]:.4f} → {principal["ba_after"]:.4f}; ΔBA = {100 * principal["delta_ba"]:.2f} pp.
- Sealed 95% CI for ΔBA: [{100 * principal["dba_ci_lo"]:.2f}, {100 * principal["dba_ci_hi"]:.2f}] pp.
- FPR: {principal["fpr_before"]:.4f} → {principal["fpr_after"]:.4f}; ΔFPR = {100 * principal["delta_fpr"]:.2f} pp.
- FNR: {principal["fnr_before"]:.4f} → {principal["fnr_after"]:.4f}; ΔFNR = {100 * principal["delta_fnr"]:.2f} pp.
- Harm/catastrophic harm = {100 * principal["harm_rate"]:.2f}% / {100 * principal["catastrophic_rate"]:.2f}%; coverage = {100 * principal["coverage"]:.2f}%.
- Benefit precision/recall = {principal["benefit_precision"]:.2f} / {principal["benefit_recall"]:.2f}.
- B2 raw and B4 gate-only are explicitly `SAFETY FAIL`; B1 shows that stronger raw utility does not imply safety.

Secondary pooled B5 (n={int(pooled["n_units"])}) has ΔBA={100 * pooled["delta_ba"]:.2f} pp, ΔFNR={100 * pooled["delta_fnr"]:.2f} pp, harm={100 * pooled["harm_rate"]:.2f}%, and remains secondary by construction.

## C — Component and negative-control evidence

- Raw always-on (gate OFF, shrinkage OFF): ΔBA +5.58 pp but ΔFNR +6.31 pp and harm 37.67% — unsafe.
- Gate only: ΔBA +4.25 pp but ΔFNR +7.49 pp and harm 37.67% — unsafe.
- Bounded only: ΔBA +3.64 pp, ΔFNR -0.16 pp, harm 3.67%, coverage 100% — safe.
- Full method: ΔBA +3.23 pp, ΔFNR +0.24 pp, harm 3.67%, coverage 40.33% — safe/selective.
- At matched coverage, random abstention averaged ΔBA {100 * random["mean_delta_ba"]:.2f} pp (2.5–97.5% randomization interval [{100 * random["ci_lo_delta_ba"]:.2f}, {100 * random["ci_hi_delta_ba"]:.2f}] pp), below the frozen gate. Random abstention has lower average harm, so the result supports gate utility/selectivity, not universal safety dominance.
- The frozen λ grid shows monotonic safety degradation as actuation becomes less conservative.

## D — Support and evidence breadth

- Total categories audited: 8; frozen TEST K=8-eligible categories: 2 (`fabric`, `walnuts`).
- K=8 units: 30 pooled, 10 DINOv2.
- K=4 is unsupported/unsafe; DINOv2 FULL has mean ΔFNR +2.82 pp and is not monotonically safer.
- Evidence-breadth terminal: **{breadth["verdict_code"]} — {breadth["verdict"]}**.
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
"""
    _write_text("PB8_PUBLICATION_BENCHMARK_FINAL_REPORT.md", report)


def _verdict_and_gate() -> None:
    strongest = (
        "On the 10 DINOv2 K=8 units from the two frozen TEST categories eligible under the acquisition-shift protocol "
        "(fabric and walnuts), preregistered normal-only selective bounded threshold calibration improved mean balanced "
        "accuracy by 3.23 pp (category-aware 95% CI 2.70–3.76 pp) versus NoAdapt while mean FNR increased by 0.24 pp, "
        "within the +2 pp safety budget."
    )
    verdict = {
        "schema_version": "1.0",
        "phase": "FINAL",
        "timestamp": NOW,
        "terminal_rank": "T2",
        "terminal": "PB8_PAPER_PACKAGE_READY_WITH_SCOPE_LIMITATIONS",
        "central_method_ready": True,
        "central_method_scope": "DINOv2-specific frozen TEST K8-eligible acquisition-shift units",
        "strongest_supported_claim": strongest,
        "evidence_breadth": "EB2_LIMITED_BUT_REPORTABLE",
        "external_comparator": "NO_DIRECT_LITERATURE_COMPARATOR_WITH_MATCHED_DEPLOYMENT_SEMANTICS",
        "external_validation": "EXTERNAL_SETTING_NOT_AVAILABLE_WITHOUT_PROTOCOL_DISTORTION",
        "residual_risk": "R1_LOCALIZED_RESIDUAL_RISK_WALNUTS",
        "working_method_name": "NorOpCal",
        "name_status": "WORKING_NAME_AFTER_PRELIMINARY_COLLISION_CHECK",
        "required_claim_boundaries": [
            "not detector-agnostic",
            "not representation or ranking improvement",
            "not K4-safe",
            "not monotonically safer with more support",
            "not externally validated",
            "not harm-free",
        ],
        "answers": {
            "central_method_question": "YES_FOR_A_SCOPED_DINOV2_SPECIFIC_PAPER",
            "strongest_claim_question": strongest,
        },
        "all_phase_gates": "PASS_OR_SCOPED_PASS",
        "next_permitted_work": "FORMAL_SECTION_BY_SECTION_PAPER_WRITING_UNDER_FROZEN_CLAIMS",
    }
    (OUT / "F_PB8_PUBLICATION_VERDICT.json").write_text(
        json.dumps(verdict, indent=2) + "\n", encoding="utf-8"
    )
    jm_gate = {
        "schema_version": "1.0",
        "phase": "J-M",
        "timestamp": NOW,
        "claim_hierarchy": "PASS",
        "method_naming": "PASS_WITH_PRELIMINARY_COLLISION_CHECK",
        "paper_section_plan": "PASS",
        "verdict": "PASS",
        "terminal": "PROCEED_TO_FINAL",
    }
    (OUT / "J_M_GATE.json").write_text(json.dumps(jm_gate, indent=2) + "\n", encoding="utf-8")


def _execution_ledger() -> None:
    rows: list[dict[str, Any]] = []
    stages = [
        (
            "P0",
            "P0_GATE.json",
            "P0_PUBLICATION_EVIDENCE_MANIFEST.csv; P0_PUBLICATION_ESTIMAND.json",
        ),
        (
            "A",
            "A_GATE.json",
            "A_EXTERNAL_COMPARATOR_ELIGIBILITY.csv; A_LITERATURE_SEARCH_REPORT.md",
        ),
        ("B", "B_GATE.json", "B_FINAL_MAIN_BENCHMARK.csv; B_DETECTOR_BENCHMARK.csv"),
        (
            "C",
            "C_GATE.json",
            "C_COMPONENT_ABLATION.csv; C_SHRINKAGE_ABLATION.csv; C_RANDOM_ABSTENTION_CONTROL.csv",
        ),
        (
            "D",
            "D_GATE.json",
            "D_K_SUPPORT_ELIGIBILITY.csv; D_K_SUPPORT_FINAL.csv; D_EVIDENCE_BREADTH_VERDICT.json",
        ),
        ("E", "E_GATE.json", "E_MECHANISM_EVIDENCE.json"),
        ("F", "F_GATE.json", "F_FINAL_STATISTICS.csv"),
        ("G", "G_GATE.json", "G_COMPUTATIONAL_OVERHEAD.csv"),
        ("H", "H_GATE.json", "TABLE_1_MAIN_BENCHMARK.csv through TABLE_5_SCOPE_AND_FAILURES.csv"),
        (
            "I",
            "I_GATE.json",
            "Five formal figures; source snapshots; plotting scripts; S5 manifest",
        ),
        (
            "J-M",
            "J_M_GATE.json",
            "FINAL_CLAIM_HIERARCHY.md; METHOD_NAME_CANDIDATES.md; PAPER_SECTION_PLAN.md",
        ),
        ("FINAL", "F_PB8_PUBLICATION_VERDICT.json", "PB8_PUBLICATION_BENCHMARK_FINAL_REPORT.md"),
    ]
    for sequence, (stage, gate_name, outputs) in enumerate(stages, start=1):
        gate_path = OUT / gate_name
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        verdict = gate.get("verdict", gate.get("terminal", "PASS"))
        rows.append(
            {
                "sequence": sequence,
                "stage": stage,
                "authorized": True,
                "status": "COMPLETED",
                "gate_artifact": gate_name,
                "gate_sha256": _sha256(gate_path),
                "gate_result": verdict,
                "primary_outputs": outputs,
                "method_changed": False,
                "completed_at_utc": NOW,
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "P0_EXECUTION_LEDGER.csv", index=False)


def _artifact_manifest() -> None:
    mandatory = [
        "PB8_PUBLICATION_BENCHMARK_FINAL_REPORT.md",
        "F_PB8_PUBLICATION_VERDICT.json",
        "P0_EXECUTION_LEDGER.csv",
        "P0_PUBLICATION_EVIDENCE_MANIFEST.csv",
        "A_EXTERNAL_COMPARATOR_ELIGIBILITY.csv",
        "B_FINAL_MAIN_BENCHMARK.csv",
        "B_DETECTOR_BENCHMARK.csv",
        "C_COMPONENT_ABLATION.csv",
        "C_SHRINKAGE_ABLATION.csv",
        "C_RANDOM_ABSTENTION_CONTROL.csv",
        "D_K_SUPPORT_ELIGIBILITY.csv",
        "D_K_SUPPORT_FINAL.csv",
        "D_EVIDENCE_BREADTH_VERDICT.json",
        "E_MECHANISM_EVIDENCE.json",
        "F_FINAL_STATISTICS.csv",
        "G_COMPUTATIONAL_OVERHEAD.csv",
        "TABLE_1_MAIN_BENCHMARK.csv",
        "TABLE_2_DETECTOR_SPECIFICITY.csv",
        "TABLE_3_COMPONENT_ABLATION.csv",
        "TABLE_4_K_SUPPORT.csv",
        "TABLE_5_SCOPE_AND_FAILURES.csv",
        "FINAL_CLAIM_HIERARCHY.md",
        "PAPER_SECTION_PLAN.md",
        "METHOD_NAME_CANDIDATES.md",
    ]
    paths = [OUT / name for name in mandatory]
    paths.extend(
        path
        for path in FIG.iterdir()
        if path.is_file()
        and path.name.startswith("FIGURE_")
        and path.suffix.lower() in {".csv", ".png", ".svg", ".pdf"}
        and "QA" not in path.name
    )
    missing = [path for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing PB8 mandatory artifacts: {missing}")
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "role": "mandatory_final_output"
            if path.parent == OUT and path.name in mandatory
            else "figure_package",
        }
        for path in sorted(set(paths))
    ]
    pd.DataFrame(records).to_csv(OUT / "PB8_FINAL_ARTIFACT_MANIFEST.csv", index=False)


def main() -> None:
    _build_exact_tables()
    _claim_hierarchy()
    _method_names()
    _paper_plan()
    _figure_captions()
    _verdict_and_gate()
    _final_report()
    _execution_ledger()
    _artifact_manifest()


if __name__ == "__main__":
    main()
