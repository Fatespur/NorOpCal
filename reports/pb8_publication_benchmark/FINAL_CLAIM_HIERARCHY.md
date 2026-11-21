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
