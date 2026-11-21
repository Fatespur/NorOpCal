# PB1-DS-PB6 — D8 K8 Support Bridge + Contamination Bridge

- K8 bridge rows: 531 (FULL + K8 per entity x candidate)
- K8-eligible units: 20 (FULL_n >= 8)
- Contamination rows: 14400 (K8 x 4 rho x 20 repeats x candidate)
- Adapt decisions frozen from D5; QM re-estimated from the changed support.
- Contaminants injected from the entity's own regular-defect pool (frozen PB4 draw rule); n_cont = ceil(rho * 8).

## K8 aggregate (frozen estimands)

        estimand support              candidate  n_entities  mean_delta_ba  mean_delta_fpr  mean_delta_fnr  coverage  benefit_precision  global_harm
common-78 (FULL)    FULL               A_GLOBAL         117      -0.069030       -0.111513        0.249573  0.128205           0.333333     0.017094
  common-40 (K8)      K8               A_GLOBAL          60      -0.015202       -0.119596        0.150000  0.133333           0.625000     0.016667
common-40 (FULL)    FULL               A_GLOBAL          60      -0.002109       -0.145783        0.150000  0.133333           0.625000     0.016667
common-78 (FULL)    FULL        D_CATEGORY_ONLY         117      -0.069030       -0.111513        0.249573  0.170940           0.550000     0.017094
  common-40 (K8)      K8        D_CATEGORY_ONLY          60      -0.015202       -0.119596        0.150000  0.333333           0.550000     0.066667
common-40 (FULL)    FULL        D_CATEGORY_ONLY          60      -0.002109       -0.145783        0.150000  0.333333           0.550000     0.033333
common-78 (FULL)    FULL C_CATEGORY_CONDITIONED         117      -0.069030       -0.111513        0.249573  0.170940           0.450000     0.017094
  common-40 (K8)      K8 C_CATEGORY_CONDITIONED          60      -0.015202       -0.119596        0.150000  0.300000           0.500000     0.066667
common-40 (FULL)    FULL C_CATEGORY_CONDITIONED          60      -0.002109       -0.145783        0.150000  0.300000           0.500000     0.033333

## Contamination aggregate (primary K=8)

   estimand   support              candidate  n_entities  mean_delta_ba  mean_delta_fpr  mean_delta_fnr  coverage  benefit_precision  global_harm
 K8 rho=0.0 K8_CONTAM               A_GLOBAL        1200      -0.016172       -0.117655            0.15  0.133333              0.625     0.024167
K8 rho=0.05 K8_CONTAM               A_GLOBAL        1200      -0.005383       -0.139234            0.15  0.133333              0.625     0.023333
 K8 rho=0.1 K8_CONTAM               A_GLOBAL        1200      -0.005541       -0.138918            0.15  0.133333              0.625     0.023333
 K8 rho=0.2 K8_CONTAM               A_GLOBAL        1200       0.001145       -0.152289            0.15  0.133333              0.625     0.023333
 K8 rho=0.0 K8_CONTAM        D_CATEGORY_ONLY        1200      -0.016172       -0.117655            0.15  0.333333              0.550     0.065833
K8 rho=0.05 K8_CONTAM        D_CATEGORY_ONLY        1200      -0.005383       -0.139234            0.15  0.333333              0.550     0.064167
 K8 rho=0.1 K8_CONTAM        D_CATEGORY_ONLY        1200      -0.005541       -0.138918            0.15  0.333333              0.550     0.064167
 K8 rho=0.2 K8_CONTAM        D_CATEGORY_ONLY        1200       0.001145       -0.152289            0.15  0.333333              0.550     0.065000
 K8 rho=0.0 K8_CONTAM C_CATEGORY_CONDITIONED        1200      -0.016172       -0.117655            0.15  0.300000              0.500     0.065000
K8 rho=0.05 K8_CONTAM C_CATEGORY_CONDITIONED        1200      -0.005383       -0.139234            0.15  0.300000              0.500     0.063333
 K8 rho=0.1 K8_CONTAM C_CATEGORY_CONDITIONED        1200      -0.005541       -0.138918            0.15  0.300000              0.500     0.063333
 K8 rho=0.2 K8_CONTAM C_CATEGORY_CONDITIONED        1200       0.001145       -0.152289            0.15  0.300000              0.500     0.065000
