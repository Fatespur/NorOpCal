# PB1-DS PB6-D4 — Two-Detector Shadow Reproduction Audit

Generated: 2026-08-19T01:26:02Z

## Nature
CODE-PATH VALIDATION. Re-executes the frozen two-detector (PatchCore + DINOv2)
protocol from RAW sealed cached scores and compares against sealed PB2/PUB0
results. NOT a new experiment, NOT tri-detector evidence, NOT PB6 scientific
completion.

## Hard gates
- **S1_population_39_78**: True — 39 outer folds from sealed split manifest
- **S2_fold_split_holdout**: True — held-out unit not read pre-seal
- **S3_feature_semantics_canonical10**: True — ['mean_shift', 'median_shift', 'mad_ratio', 'std_ratio', 'd_q90', 'd_q95', 'd_q99', 'upper_tail_spread_change', 'wasserstein', 'ks']
- **S4_model_family_logistic**: True — LogisticRegression l2 lbfgs balanced; scaler=null
- **S5_action_reproduction**: True
- **S6_downstream_reproduction**: True

## Entity-level reproduction (n=234)
- selector probability max |diff| = 1.8884893648873913e-13
- action mismatches = 0
- max downstream diff = 5.551115123125783e-17

## Metric-level reproduction
- max |ΔBA| diff (pp) = 3.469446951953614e-15

## Terminal
`PB1DS_PB6_D4_SHADOW_PRECHECK_PASS_TWO_DETECTOR_PROTOCOL_EXACTLY_REPRODUCED`

## PB6 status
- PB6 overall: `PB1DS_PB6_IN_PROGRESS_BLOCKED_AT_D2_PENDING_REAL_MVTEC_AD2_IMAGES_AND_EFFICIENTAD_WEIGHTS`
- D5_authorized = false
- D2_blocked = true
