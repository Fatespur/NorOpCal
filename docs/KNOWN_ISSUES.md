# Known Issues & Boundaries

Honest, non-hidden engineering and scientific boundaries affecting this release.

## KI-1 — Lightweight Open-Source Scope vs. Internal Research Monorepo

- **Context**: NorOpCal was developed as part of a broader industrial anomaly detection research initiative (serving as the historical algorithmic predecessor to MARC-AD).
- **Boundary**: Heavyweight experimental components — including hundreds of gigabytes of raw industrial images (MVTec AD 2), multi-seed detector checkpoint files, and historical internal tools (`tools/pb6`, `tools/pb7`) — are retained in internal research archives and are intentionally **not redistributed** in this public GitHub repository.
- **Verification Guarantee**: The core calibration logic is completely self-contained within `src/noropcal`, verified via automated unit tests (`tests/test_noropcal.py`), and frozen specification parameters (`noropcal.load_frozen_spec()`).

## KI-2 — Detector-Specific Applicability

- **Finding**: Positive threshold calibration headroom (+3.23 pp) was verified on DINOv2 (`dinov2_vits14`) under $K=8$ normal target samples on 10 qualified TEST units.
- **Controls**: Mechanism controls demonstrated that PatchCore suffered from feature space shift mismatches (`ACTUATOR_MISMATCH`), while EfficientAD showed negligible recalibration headroom (`NO_HEADROOM`).
- **Policy**: NorOpCal does not claim universal performance gains across arbitrary anomaly detectors.

## KI-3 — Dataset Availability & External Abstention

- Full reproduction of benchmark tables requires user acquisition of the proprietary MVTec AD 2 dataset and setting `NOROPCAL_MVTEC_AD2_ROOT`.
- External dataset transfer to AeBAD-S was honestly abstained in published findings due to severe ambient drift causing gate saturation.
