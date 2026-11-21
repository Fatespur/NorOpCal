# Known Issues & Boundaries

Honest, non-hidden engineering and scientific boundaries affecting this release.

## KI-1 — Reproduction Scope: Level 1 & 2 vs. Level 3

- **Level 1 (Core Algorithms & APIs)**: Fully verified via `python -m pytest -v tests/` against self-contained implementations and sealed protocol specifications (`noropcal.load_frozen_spec()`).
- **Level 2 (Synthetic Demos & Diagnostics)**: Verified via `scripts/demo.py` and `assets/figures/generate_all_figures.py`.
- **Level 3 (Monolithic Research Experiments)**: Heavyweight components — including proprietary raw industrial datasets (MVTec AD 2), multi-seed detector checkpoint files, and historical internal tools (`tools/pb6`, `tools/pb7`) — are retained in internal research archives and are intentionally **not redistributed** in this public GitHub repository. Level 1/2 passing must not be equated with Level 3 direct reproduction.

## KI-2 — Detector-Specific Applicability & Hypotheses

- **Finding**: Positive threshold calibration headroom (+3.23 pp) was verified on DINOv2 (`dinov2_vits14`) under $K=8$ normal target samples on 10 qualified TEST units.
- **Controls**: Mechanism controls demonstrated that PatchCore suffered from feature space shift mismatches (`ACTUATOR_MISMATCH`), while EfficientAD showed negligible recalibration headroom (`NO_HEADROOM`).
- **Policy**: NorOpCal does not claim universal performance gains across arbitrary anomaly detectors. Mechanistic explanations represent scientific hypotheses based on empirical observations, not absolute mathematical proofs.

## KI-3 — Statistical Safety Budget vs. Individual Guarantees

- The preregistered safety constraint $\mathbb{E}[\Delta FNR] \le +0.02$ represents an aggregate expected budget across the designated test distribution.
- It does not constitute a deterministic individual safety guarantee for every isolated sample or every individual acquisition condition.
