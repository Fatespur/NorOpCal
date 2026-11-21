# PB7-P2 Publication-Readiness Confirmatory Validation — Final Report

- Date: 2026-08-20T01:50:00Z
- Stage: PB7-P2 (CONFIRMATORY / PUBLICATION-READINESS)
- Inherits terminal: `PB7_P1_DETECTOR_SPECIFIC_SAFE_ACTUATOR_SUPPORTED`
- Frozen principal policy: A_MEDIAN_RAW, lambda=0.25, cutoff=Q50 (0.015890), K=8, safety budget mean dFNR <= +0.02

## 1. Exact Reproduction (Stage A)

- Verdict: **PASS** (tolerance < 1e-9)
- Max per-entity abs diff: dBA=9.71e-17, dFNR=9.71e-17, harm=6.94e-17
- Pooled K8: dBA=0.00e+00, dFNR=1.56e-17
- DINOv2 K8: dBA=2.08e-17, dFNR=3.90e-18

## 2. Leave-One-Category-Out Influence Audit (Stage B)

- Verdict: **PASS**; full DINOv2 dBA=0.0323
- LOO runs: 8; dBA>0 in 8/8; dFNR<=0.02 in 8/8
- Max influence: 0.0053 (category walnuts, 16.4% of total)
- No single category >50% utility: True
- Category-fragile flag: False

## 3. K-Support Confirmatory Audit (Stage C)

- Verdict: **PASS**; repeats=200

| Setting | median dBA | positive freq | safety freq | median dFNR |
|---|---|---|---|---|
| 4 POOLED | 0.0104 | 1.000 | 0.095 | 0.0231 |
| 4 DINOv2 | 0.0365 | 1.000 | 0.000 | 0.0282 |
| 8 POOLED | 0.0125 | 1.000 | 0.965 | 0.0133 |
| 8 DINOv2 | 0.0336 | 1.000 | 1.000 | 0.0000 |
| FULL POOLED | 0.0108 | 0.167 | 0.808 | 0.0188 |
| FULL DINOv2 | 0.0372 | 0.385 | 0.731 | 0.0282 |

- K8 gate: positive freq=1.000 (>=0.90), safety freq=0.965 (>=0.90), median dBA=0.0125 vs FULL 0.0108 (retains 116%)
- K8 Jaccard: pooled=0.646, DINOv2=0.854

## 4. Confirmatory Bootstrap (Stage D)

- B=10000, seed=20260821 (pre-registered)

| Group | mean dBA | 95% CI dBA | mean dFNR | 95% CI dFNR | harm | coverage |
|---|---|---|---|---|---|---|
| POOLED | 0.0120 | [0.0101, 0.0138] | 0.0127 | [0.0000, 0.0253] | 0.1439 | 0.3428 |
| DINOv2 | 0.0323 | [0.0270, 0.0376] | 0.0025 | [0.0000, 0.0049] | 0.0368 | 0.4035 |

- D3 publication significance: **PASS** (DINOv2 dBA CI lower=0.0270 > 0; dFNR mean=0.0025 <= +0.02)
- Paired comparisons (DINOv2): Gated vs NoAdapt dBA=0.0323 (CI lo=0.0270); Gated vs PB7-P0 T2 dFNR=-0.0653 (CI hi=-0.0200)
- H5 selectivity nuance: pooled safety advantage supported (dFNR diff=-0.0016, harm diff=-0.0245); DINOv2-specific dFNR advantage NOT_SUPPORTED (always-on already safe on DINOv2; the gate acts as coverage reduction there).

## 5. Residual Harm Audit (Stage E)

- Verdict: **R1_LOCALIZED_RESIDUAL_RISK** (R1 LOCALIZED_RESIDUAL_RISK)
- n_harm=9/78 (harm rate=0.115); n_catastrophic=9
- Concentration: single category=True (all in walnuts, top share=1.00); detector-wide failure=False; condition systemic failure=False; DINOv2 harm rate=0.038 (n=1)

| category | condition | detector | source_fnr | median_shift | dBA | dFNR | harm_type |
|---|---|---|---|---|---|---|---|
| walnuts | overexposed | efficientad | 0.800 | 0.0452 | 0.0600 | 0.0600 | catastrophic_fnr |
| walnuts | shift_1 | dinov2_vits14 | 0.467 | 0.1028 | 0.0878 | 0.0244 | fnr_breach |
| walnuts | shift_1 | efficientad | 0.800 | 0.0207 | -0.0033 | 0.0267 | fnr_breach |
| walnuts | shift_1 | wide_resnet50_2 | 0.533 | 0.0572 | -0.0589 | 0.1178 | catastrophic_fnr |
| walnuts | shift_2 | efficientad | 0.800 | 0.0305 | -0.0222 | 0.0444 | fnr_breach |
| walnuts | shift_2 | wide_resnet50_2 | 0.533 | 0.0080 | -0.0133 | 0.0267 | fnr_breach |
| walnuts | shift_3 | efficientad | 0.800 | 0.0018 | -0.0044 | 0.0089 | dba_loss |
| walnuts | shift_3 | wide_resnet50_2 | 0.533 | 0.0189 | 0.0139 | 0.0556 | catastrophic_fnr |
| walnuts | underexposed | wide_resnet50_2 | 0.533 | 0.0052 | -0.0078 | 0.0156 | dba_loss |

## 6. Detector-Specificity Confirmation (Stage F)

- Classification persists vs sealed PB7-P1: **True**

| detector | n_k8 | n_correctable_safe | mean dBA | mean dFNR | harm | coverage | verdict |
|---|---|---|---|---|---|---|---|
| dinov2_vits14 | 10 | 9 | 0.0323 | 0.0024 | 0.037 | 0.403 | USEFUL_SAFE_SELECTABLE |
| wide_resnet50_2 | 10 | 1 | -0.0066 | 0.0216 | 0.247 | 0.247 | ACTUATOR_MISMATCH |
| efficientad | 10 | 0 | 0.0103 | 0.0140 | 0.147 | 0.377 | NO_HEADROOM |

- Claim frozen: **True** — "Correctability and safe threshold actuation are detector-dependent."

## 7. External-Setting Validation (Stage G)

- Verdict: **EXTERNAL_SETTING_NOT_AVAILABLE_WITHOUT_PROTOCOL_DISTORTION**
- AeBAD-S median_shift by condition: background=0.0672, illumination=0.0393, view=0.1068, same=0.0450
- Frozen cutoff=0.015890; all AeBAD-S median_shift above cutoff: True
- Reason: The frozen Q50 cutoff (0.015890181064605713) is the median of the MVTec AD 2 DEV median_shift distribution and is score-scale specific. All AeBAD-S median_shift values (0.039-0.107) exceed the frozen cutoff, so applying it unchanged makes the gate never abstain (equivalent to Always-On). Re-tuning lambda/Q50 for AeBAD-S is forbidden by G2 (it would no longer be confirmatory validation). Original MVTec AD offers no legal target-shift definition matching the acquisition-shift + source-normal/target-normal setting without manufacturing a pseudo-validation. Therefore no external setting is available without protocol distortion.

## 8. Limitations

- **Residual harm localized to walnuts** (9/9 harmful entities; all high source-FNR). The method is retained with this limitation documented; walnuts high-source-FNR entities are the known residual risk.
- **K4 is not safe** (K4 pooled safety freq=0.095, DINOv2=0.000; K4 dFNR exceeds +0.02). Only K8 is the safe few-normal operating point.
- **FULL is not safe on DINOv2** (dFNR=0.0282 > +0.02); the K8 gate abstains more and is the safe principal setting.
- **K8 evaluation scope**: only fabric and walnuts are K8-eligible (n>=8 target-normal); rice/sheet_metal/vial (n<8) are excluded from K8 but included in FULL/LOO.
- **H5 nuance**: on DINOv2 the gate provides no dFNR advantage over Always-On (always-on is already safe there); the gate's safety value is pooled-level (protects PatchCore/EfficientAD).
- **No external validation**: frozen Q50 cutoff is dataset/score-scale specific and not transferable to AeBAD-S; re-tuning is forbidden.

## 9. Terminal

- **T1 PB7_P2_PUBLICATION_CANDIDATE_CONFIRMED_DINOV2_SPECIFIC**

## 10. Next Authorization

- With terminal T1, the next authorized stage is **PB8 — Publication Benchmark + Ablation + Paper Claim Freeze** (final paper main table, strong baselines, ablation, mechanism figures, statistical tables, computational overhead, method naming, final claim hierarchy, paper section planning). PB7-P2 does not do paper packaging.

## Evidence tables

- `A_REPRODUCTION.csv`, `A_REPRODUCTION_PER_ENTITY.csv`
- `B_LOCO_CATEGORY_INFLUENCE.csv`
- `C_K_SUPPORT_STABILITY.csv`, `C_K8_REPEAT_DISTRIBUTION.csv`, `C_K8_REPEAT_DISTRIBUTION_DINOV2.csv`, `C_GATE_SELECTION_JACCARD.csv`
- `D_CONFIRMATORY_BOOTSTRAP.csv`, `D_PAIRED_COMPARISON.csv`, `D_POLICY_*.csv`
- `E_RESIDUAL_HARM.csv`, `E_HARMFUL_VS_NONHARMFUL.csv`, `E_CATASTROPHIC_CONCENTRATION.csv`
- `F_DETECTOR_CONFIRMATION.csv`, `F_DETECTOR_CONFIRMATION_PER_ENTITY.csv`
- `G_EXTERNAL_VALIDATION.csv`
- `P0_CONFIRMATORY_INPUT_MANIFEST.csv`, `P0_HYPOTHESIS_REGISTRY.json`, `P0_EXECUTION_LEDGER.csv`
