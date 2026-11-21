# Configs

Physical layout preserved from the parent research repository for provenance;
logical roles are documented here.

| Path | Logical role |
|---|---|
| `datasets/mvtec_ad.yaml` | MVTec AD / AD 2 dataset adapter config used by the scoring entry points |
| `models/*.yaml` | **Detector construction configs** (logical `detectors/`): `dino_memory.yaml` (DINOv2 principal), `patchcore_default.yaml`, `efficientad_default.yaml` (frozen 70,000-step protocol) |
| `experiments/{dino_memory,patchcore,efficientad}_fixture.yaml` | Synthetic-fixture configs consumed by the unit tests (test assets only) |
| `experiments/phase45/` | Phase-4.5 protocol/base templates asserted by detector tests (historical provenance) |
| `protocols/noropcal_frozen_protocol.yaml` | Declarative record of the frozen NorOpCal protocol. **NOT an execution input** — authoritative values live in `tools/pb7_p1_safety_actuator/pb7_p1_common.py` and the sealed `reports/pb7_p1_safety_actuator/E_FROZEN_METHOD_SPEC.json` |
| `examples/` | Minimal example configs for user-provided dataset roots |

Detector config paths are referenced directly by tests
(`tests/test_efficientad.py`) and scoring tools
(`tools/pb6/d2_efficientad_score.py`); do not rename without updating those
references.
