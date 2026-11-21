# Method — NorOpCal

NorOpCal (**Nor**mal-Only **S**elective **Op**erating-Point **Cal**ibration)
recalibrates only the decision threshold of a frozen anomaly detector using K
known-normal target observations.

This page is a summary. The binding scientific record is
`docs/reference/PUBLICATION_GROUND_TRUTH.md` (frozen PW0.5); the canonical
implementation is `tools/pb7_p1_safety_actuator/pb7_p1_common.py` with frozen
semantics shared from `tools/pb7_p0_threshold_correctability/pb7_p0_common.py`.

## Deployment information set

`(\tau_s, S_s^N, S_t^N)` — the frozen source threshold, source-normal scores,
and `K` known-normal target scores. Nothing else. No target anomalies, labels,
masks, prevalence, or test outcomes are usable by the deployable decision
(code-level firewall: `pb7_p0_common.Firewall`).

## Frozen policy (single legal specification)

1. **Statistic.** `m_s = median(S_s^N)`, `m_t = median(S_t^N)`,
   `\Delta m = m_t - m_s`.
2. **Correctability gate.** adapt iff `\Delta m \ge Q_{50}` with frozen
   `Q_{50} = 0.015890181064605713`; otherwise abstain.
3. **Bounded actuator.** `A_MEDIAN_RAW`: `\tau_t = \tau_s + \lambda \Delta m`
   with frozen `\lambda = 0.25`.
4. **Abstention.** gate rejection sets `\tau_t = \tau_s` (NoAdapt). Abstention
   is a designed outcome, not a failure.
5. **Safety budget (preregistered).** `\mathbb{E}[\Delta FNR] \le +0.02`.

Support sizes: `K \in {4, 8, FULL}`; K=4 auxiliary (unsafe), K=8 principal,
FULL = all available target normals. Support sampling uses the frozen nested
permutation rule (`pb7_p0_common.support_indices`, SHA256-derived seeds,
30 sealed repeats per entity for entity-level K results).

## What NorOpCal is NOT

Not domain adaptation, not detector/representation adaptation, not test-time
model adaptation. The backbone, score function, ranking, and all detector
parameters are fixed at deployment. Threshold-only changes leave AUROC/AUPR
unchanged by construction.

## Detector scope (frozen verdicts)

| Detector | Role | Verdict |
|---|---|---|
| DINOv2 (`dinov2_vits14`) | principal | USEFUL_SAFE_SELECTABLE |
| PatchCore (`wide_resnet50_2`) | mechanism control | ACTUATOR_MISMATCH |
| EfficientAD (in-house 70k steps) | mechanism control | NO_HEADROOM |

Correctability and safe threshold actuation are **detector-dependent**
findings, not universal claims.

## Principal result

DINOv2, K=8, 10 TEST-eligible units (fabric + walnuts): mean ΔBA
+3.23 pp (category-aware bootstrap B=10,000 CI [+2.70, +3.76] pp), mean ΔFNR
+0.24 pp (within the +2 pp budget), harm rate 3.67%, coverage 40.33%.

## Known limitations (frozen L-rows, abridged)

- Residual harm concentrates in `walnuts` entities with high source FNR; no
  post-hoc category rule exists or may be added.
- K=8 eligibility holds only where ≥8 independent target-normal images exist
  (10 of 26 TEST units); broader breadth is data-limited (PB8-S1).
- External-dataset validation was honestly abstained (AeBAD-S gate degenerates).
- Base-detector runtime was not measured in PB8; overhead claims cover the
  actuator only (0.136 ms p50 per entity, CPU).
