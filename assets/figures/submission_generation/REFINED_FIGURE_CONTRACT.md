# Refined IEEE TIM Figure Contract

## Scope

This refinement changes only visual encoding, typography, layout, and statistical presentation. It does not alter the manuscript, the frozen scientific values, method ranking, safety threshold, chance level, or detector-specific interpretation. Existing figure assets remain untouched; refined candidates use the `_REFINED` suffix.

## Frozen support-setting result figures

### Utility across frozen support settings

- **Core conclusion:** All three frozen settings have positive utility point estimates; `K = 8` is highlighted only because it is the frozen operating point, not because it has the largest utility estimate.
- **Archetype:** Independent single-panel categorical point-estimate plot.
- **Encoding:** Mean `DeltaBA` for `K = 4`, `K = 8`, and `Full`; only `K = 8` carries the stored category-aware bootstrap 95% CI `[2.70, 3.76] pp`.
- **Reference:** A light dashed horizontal line at `DeltaBA = 0` denotes no utility gain.

### Safety across frozen support settings

- **Core conclusion:** `K = 8` remains inside the predefined `mean DeltaFNR <= 2 pp` safety budget, whereas the `K = 4` and `Full` point estimates exceed it.
- **Archetype:** Independent single-panel categorical point-estimate plot with a policy boundary.
- **Encoding:** Mean `DeltaFNR` for the same settings; only `K = 8` carries the stored category-aware bootstrap 95% CI `[0.00, 0.49] pp`.
- **Reference:** A dashed line at `DeltaFNR = 2 pp` separates the white safety-feasible region from an extremely light warm-gray outside-budget region.

### Shared review constraints

- The figures are separate files, not panels and not a two-dimensional utility--safety map.
- `K = 8` uses a deep-blue diamond; `K = 4` and `Full` use neutral-gray circles. No points are joined.
- `K = 4` and `Full` remain point-only because no matching stored intervals exist.
- `K = 4` and `Full` use 26 DINOv2 units whereas `K = 8` uses 10 eligible units. Cross-setting differences are frozen-setting comparisons rather than pure support-size effects; this limitation remains in the captions.

## Figure 3

- **Core conclusion:** The frozen selective gate is informative relative to coverage-matched random abstention, while correctability AUC is a diagnostic rather than a deployment-time oracle.
- **Archetype:** Compact statistical evidence figure.
- **Panel (a):** horizontal point and frozen randomization interval because no stored per-seed gate-advantage series exists. The stored random-control file contains 1,000 seed-level `delta_ba` values, but deriving a new per-seed advantage series is outside this visual-only task.
- **Panel (b):** horizontal dot/forest plot. The full-population AUC uses its existing 95% category-aware block-bootstrap interval (`B=2000`); the K=8 AUC remains point-only because no frozen CI/percentile interval was found.
- **Review risk:** the figure must not imply deployment-time access to oracle correctability or fabricate uncertainty.

## Journal and export contract

- Target: IEEE Transactions on Instrumentation and Measurement.
- Backend: Python/matplotlib only.
- Final size: each support-setting result figure is 3.50 x 2.85 in (IEEE single column), with a 90% scale QA preview; the Gate/Correctability figure retains its existing dimensions.
- Typeface: Times New Roman for all English text; STIX math glyphs for mathematical notation.
- Text: 7.6--8.25 pt for axes and labels; 6.25--6.85 pt only for subordinate region/boundary annotations.
- Palette: deep blue `#2F5597` for the principal result, neutral grays for secondary evidence, white background.
- Exports: editable-text SVG, font-embedded PDF, 600-dpi PNG, grayscale preview, and 90% scale preview.
