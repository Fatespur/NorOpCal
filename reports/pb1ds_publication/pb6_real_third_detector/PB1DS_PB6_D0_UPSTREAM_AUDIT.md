# PB1-DS PB6 D0 — Upstream & Protocol Audit

Date: 2026-08-18T17:31:19Z

## Result
**all_pass: True** — every frozen definition PB6 must inherit is uniquely recoverable.

## Checks
- [x] `PB5_terminal_correct`: terminal=PB1DS_PB5_COMPLETE_TRANSPARENCY_MECHANISM_AND_PUBLICATION_CLAIMS_RECONCILED
- [x] `PB6_authorized`: PB6_authorized=True
- [x] `units_39`: units=39 (constant 39)
- [x] `entities_78`: entities=78 (2 detectors)
- [x] `categories_unique`: cats=['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
- [x] `conditions_unique`: conds=['overexposed', 'underexposed', 'shift_1', 'shift_2', 'shift_3']
- [x] `benefit_definition`: delta_ba >= +0.02 AND delta_fpr <= -0.05 AND delta_fnr <= +0.02
- [x] `qm_quantile`: QM quantile=0.95
- [x] `threshold_rule`: precision(ADAPT->BENEFIT) >= 0.50; among feasible thresholds maximize mean Delta_BA; tie-break 1 lower Global Harm, tie-
- [x] `ten_features`: n_features=10 (sealed C1/C0 freeze and frozen model binding feature_order_if_z both match canonical 10)
- [!] `p5_feature_col_slice_NOT_VERIFIED` (NOT_VERIFIED): P5.FEATURE_COLUMNS=['mean_shift', 'median_shift', 'd_q50', 'std_ratio', 'mad_ratio', 'iqr_ratio', 'd_q90', 'd_q95', 'd_q99', 'q99_q95_change'] slices first-10 of md0sr(14) differing from canonical ['mean_shift', 'median_shift', 'mad_ratio', 'std_ratio', 'd_q90', 'd_q95', 'd_q99', 'upper_tail_spread_change', 'wasserstein', 'ks']; only used to NAME coef indices in pb5_run T3b transparency; frozen model binding coef/order is canonical. NOT_VERIFIED PB5 doc drift (erratum candidate).
- [x] `scaler_null_or_documented`: standardization diagnostic: False
- [x] `bc_canonical`: BC_TARGET_NORMAL_THRESHOLD_FAMILY (representative BC_RANK_BASED_THRESHOLD)
- [x] `block_bootstrap_5000`: B=5000
- [x] `contamination_grid`: rho=[0.0, 0.05, 0.1, 0.2]
- [x] `contamination_repeats`: repeats=20
- [x] `detector_encoding_binary_today`: frozen detectors=['dinov2_vits14', 'wide_resnet50_2']
- [x] `efficientad_not_wired`: no EfficientAD detector level in the sealed 78-entity protocol frame
- [x] `k8_semantics`: primary K=8

## Unit population (authoritative, from sealed `PB1DS_PUB0_PER_UNIT.csv`)
- 39 units = 39; FULL = 78 entities (2 detectors); third detector adds 39 -> expected 117 entities.
- 8 categories: can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts.
## Protocol binding notes
See `PB1DS_PB6_PROTOCOL_BINDING.json`.
