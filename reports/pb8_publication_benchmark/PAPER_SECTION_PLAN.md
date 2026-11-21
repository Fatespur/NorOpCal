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
