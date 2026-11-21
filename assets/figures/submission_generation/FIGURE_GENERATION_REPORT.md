# NorOpCal IEEE TIM Figure Generation Report

## Scope and outcome

This package contains publication-ready candidate figures for an IEEE Transactions on Instrumentation and Measurement submission. It does not modify the manuscript, rerun experiments, recompute metrics, or alter any sealed artifact. All quantitative panels are rendered from explicit frozen-value CSV snapshots in `data/`.

The recommended main-text set is:

1. `Fig1_NorOpCal_Framework` — method framework (required).
2. `Utility_across_Frozen_Support_Settings_REFINED` — independent frozen-setting utility evidence.
3. `Safety_across_Frozen_Support_Settings_REFINED` — independent frozen-setting safety evidence.

`Fig3_Gate_Informativeness_and_Correctability` is deliberately conservative and is best treated as an appendix/supplement candidate unless the manuscript has space and the figure is separately registered. No manuscript figure registry was changed.

## Export bundle

The two frozen support-setting result figures are independent 3.50 x 2.85-in single-column SVG, PDF, and 600-dpi PNG files. Each has a grayscale print-readability preview, a 90%-scale preview, a Python/matplotlib generation path, and the same traceable CSV input snapshot. Other previously generated package figures retain their existing dimensions and layouts.

SVG text is kept editable (`svg.fonttype = none`). PDFs use TrueType-compatible font type 42 and contain no detected Type 3 font object. SVGs contain no embedded raster image.

## Evidence fidelity

### Figure 1

The diagram encodes only the frozen NorOpCal semantics: frozen detector and ranking; source-normal statistics; eight known-normal target images; median normal-score deformation; frozen Q50 selective gate; abstention; and the threshold-only shrinkage update `tau_t = tau_s + 0.25 Delta m`. It explicitly excludes target anomalies, labels, masks, the target anomaly-score distribution, retraining, and fine-tuning. The small score profiles are labeled schematic and are not reconstructed empirical distributions.

### Frozen support-setting result figures

The rejected two-dimensional decision map has been replaced by two independent single-panel categorical plots. The utility figure reports mean DeltaBA of `+3.76`, `+3.23`, and `+3.72 pp` for `K = 4`, `K = 8`, and `Full`. The safety figure reports mean DeltaFNR of `+2.85`, `+0.24`, and `+2.82 pp`, respectively, against the frozen `+2 pp` safety budget. Only `K = 8` carries stored category-aware bootstrap 95% intervals: DeltaBA `[2.70, 3.76] pp` and DeltaFNR `[0.00, 0.49] pp`. `K = 4` and `Full` remain point-only. They use 26 DINOv2 units, whereas `K = 8` uses 10 eligible units, so the comparisons are not pure support-size causal effects.

### Figure 3

Panel (a) reports the exact gate advantage `+1.75 pp` and preregistered randomization interval `[1.53, 1.99] pp` over 1,000 control seeds. The interval is not described as a sampling confidence interval. Panel (b) shows only the supplied diagnostic AUC summaries: full population `0.712` and `K = 8` subset `0.802`. It is explicitly labeled oracle-conditioned and not a deployment oracle. No raw distribution or ROC curve was reconstructed.

## Suggested captions

**Fig. 1. NorOpCal framework under acquisition shift.** The anomaly detector, representation, score function, and ranking remain frozen. NorOpCal uses source-normal statistics and `K = 8` known-normal target images to estimate median normal-score deformation. A frozen Q50 gate selectively authorizes a conservative threshold-only correction; rejected cases retain the source threshold. Target anomalies, labels, masks, anomaly-score distributions, retraining, and fine-tuning are unavailable.

**Utility across frozen support settings.** Mean change in balanced accuracy under the frozen DINOv2 policy. The deep-blue diamond marks the frozen `K = 8` operating point; its whisker is the stored category-aware bootstrap 95% confidence interval. `K = 4` and `Full` are point estimates only because no matching 95% sampling intervals were stored. `K = 8` uses the frozen 10-unit eligible subset, whereas `K = 4` and `Full` use the broader 26-unit population; cross-setting differences are therefore not pure support-size effects.

**Safety across frozen support settings.** Mean change in false-negative rate under the same frozen settings. The dashed line marks the preregistered `ΔFNR ≤ 2 pp` aggregate safety budget, and the lightly tinted region lies outside that budget. The deep-blue `K = 8` diamond carries its stored category-aware bootstrap 95% confidence interval; `K = 4` and `Full` remain point estimates only. Population definitions are the same as in the utility figure.

**Fig. 3. Gate informativeness and correctability diagnostics.** (a) The real selective gate improves utility by `+1.75 pp` relative to coverage-matched random abstention; whiskers show the `[1.53, 1.99] pp` preregistered randomization interval over 1,000 control seeds. (b) Oracle-conditioned correctability-diagnostic AUC is `0.712` on the full population and `0.802` for the `K = 8` subset. These values are diagnostic summaries, not deployment oracles.

## QA record

- Python generation completed for both independent frozen support-setting figures.
- Strict size/DPI checks target 3.50 x 2.85 in; each final PNG is 2,100 x 1,710 px at approximately 600 dpi.
- Both SVGs retain editable text, contain a viewBox, and contain no `<image>` or base64 raster payload.
- Both PDFs use TrueType-compatible font configuration and contain no detected Type 3 font marker.
- Color, grayscale, and 90%-scale previews were visually inspected. No clipped labels, marker/interval clipping, text overlap, or unreadable titles remained.
- Both figures passed exact point/interval binding, missing-interval disclosure, label-boundary separation, single-column size, and grayscale-readability checks.

## Known limits

- The package is a venue-styled candidate asset set and is not automatically part of the sealed manuscript figure registry.
- Figure 3 intentionally remains a summary-only diagnostic because raw diagnostic distributions were not used or reconstructed.
- PDF font auditing used object-marker inspection because `pypdf`, Poppler, and Ghostscript were not installed; the source also explicitly sets `pdf.fonttype = 42`.
- No physical printer proof or external color-vision simulator was available. Grayscale previews and redundant shape/line encodings were checked locally.

## Reproduction

From the repository root:

```powershell
python reports/manuscript/NorOpCal_TIM_figures/make_all_figures.py --output-dir reports/manuscript/NorOpCal_TIM_figures
python reports/manuscript/NorOpCal_TIM_figures/make_refined_figures.py --output-dir reports/manuscript/NorOpCal_TIM_figures
```

The scripts require Python, matplotlib, NumPy, and Pillow. The package does not require manuscript editing or experiment execution.
