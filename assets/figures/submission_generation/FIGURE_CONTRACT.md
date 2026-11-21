# NorOpCal IEEE TIM Figure Contract

## Figure 1

- **Core conclusion:** Under acquisition shift, NorOpCal leaves the anomaly detector and score ranking frozen and selectively changes only the decision threshold from source-normal statistics and eight known-normal target images.
- **Archetype:** Schematic-led method figure.
- **Primary evidence:** Frozen method semantics: median score deformation, Q50 gate, abstention, and a shrinkage-based threshold actuator.
- **Reviewer risk:** The diagram must not imply target-anomaly access, detector training, feature adaptation, score adaptation, ranking changes, or oracle-threshold estimation.

## Figure 2

- **Core conclusion:** NorOpCal provides positive utility while remaining inside the aggregate +2 pp mean-ΔFNR budget, and the achieved response is detector-specific.
- **Archetype:** Two-panel quantitative grid with identical axes.
- **Panel (a):** Frozen policy-level utility–safety points.
- **Panel (b):** Frozen detector-level utility–safety points.
- **Reviewer risk:** Safety-budget membership is descriptive for the frozen aggregate values; it is not a universal guarantee. EfficientAD must not be styled as equivalent to DINOv2, and PatchCore must remain visible as a failure case.

## Figure 3

- **Core conclusion:** The real gate carries information beyond coverage-matched random abstention, while normal-score deformation is an actionable but incomplete diagnostic rather than a correctability oracle.
- **Archetype:** Conservative two-panel diagnostic summary.
- **Panel (a):** Point and preregistered randomization interval for gate advantage.
- **Panel (b):** Exact AUC summary points only; no reconstructed ROC curve or raw-score distribution.
- **Reviewer risk:** The randomization interval is not relabeled as a sampling confidence interval. AUC values are explicitly oracle-conditioned diagnostics and are not deployment oracles.

## Journal and export contract

- Target: IEEE Transactions on Instrumentation and Measurement.
- Backend: Python/matplotlib only.
- Primary width: 7.16 in double-column; additional 3.50 in single-column variants.
- Typography: Arial/Helvetica-compatible sans serif; 6–9 pt at final size.
- Encoding: color plus marker/line/fill redundancy; dedicated grayscale previews.
- Exports: editable-text SVG, embedded-font PDF, and 600-DPI PNG.
