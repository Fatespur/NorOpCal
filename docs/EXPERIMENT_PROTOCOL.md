# Experiment Protocol (frozen)

This file summarizes the frozen evaluation protocol. Binding sources:
`docs/reference/PUBLICATION_GROUND_TRUTH.md` and the sealed per-stage
artifacts under `reports/` (each stage directory carries its own
`P0_*` protocol/manifest files with SHA256-bound inputs).

## Population

- 39 category-condition units × 3 detectors = 117 entities.
- Detectors: `dinov2_vits14` (principal), `wide_resnet50_2` (PatchCore),
  `efficientad` (in-house, 70,000 steps, source-normal-only training,
  seed 0, frozen teacher/checkpoint hashes — PB1DS-PB6 D2/D4).

## Operating point

- Source threshold: `\tau_s = Q_{95}` of source-normal image scores.
- Detection rule: anomaly iff `score > \tau` (strict).
- Benefit (frozen): `\Delta BA \ge +0.02`, `\Delta FPR \le -0.05`,
  `\Delta FNR \le +0.02` vs the source operating point.
- Harmful entity: `\Delta BA < -0.02` or `\Delta FNR > +0.02`.

## Support protocol

- `K ∈ {4, 8, FULL}`, nested sampling (`S4 ⊂ S8 ⊂ S_FULL`), deterministic
  SHA256-derived seeds per `(K, category, condition, detector, repeat)`.
- Entity-level K results: 30 sealed repeats averaged per entity.
- K-support robustness: 200 preregistered support repeats (PB7-P2 C-stage).
- Eligibility by independent target-normal count only; never by outcomes.

## Selection vs evaluation

- DEV categories (can, fruit_jelly, wallplugs): used once to select
  `(family, λ, cutoff)` = `(A_MEDIAN_RAW, 0.25, Q50)` from the preregistered
  grid; safety-first selection restricted to gated candidates.
- TEST categories: single frozen evaluation, no post-hoc tuning of any kind.

## Statistics

- Primary utility: mean ΔBA = BA(τ_t) − BA(τ_s) vs NoAdapt.
- Primary safety: mean ΔFNR vs the preregistered +2 pp budget.
- Inference: category-aware bootstrap, B = 10,000 (authoritative PB7-P2
  reuse), 95% percentile CIs; report hierarchy DINOv2 before pooled
  (pooled always labeled SECONDARY).
- Random-abstention negative control: 1000 preregistered control seeds
  (randomization interval, not a sampling CI).

## Data-role firewall

Deployable computation may read only source-normal / target-normal scores and
identity metadata. Anomaly scores are ORACLE_DIAGNOSTIC_ONLY and every read
is recorded (`pb7_p0_common.Firewall`); deployable outputs must call
`assert_deployable_clean()`.

## Revision reserve (RR0–RR6)

Post-submission analyses under `tools/revision_reserve/` +
`results/revision_reserve/` re-use the same frozen caches and policy; nothing
was retrained, re-tuned, or re-split. RR wording is fixed to "Additional
post-submission analysis conducted in response to reviewer concerns." RR6 is a
disclosed negative control (a bounded two-sided comparator dominates aggregate
metrics via downward corrections that the upward-only gate refuses by design).
