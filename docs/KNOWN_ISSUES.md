# Known Issues

Honest, non-hidden issues affecting this repository.

## KI-1 — Pre-existing test failure (inherited from the parent research repo)

- Test: `tests/test_efficientad.py::TestEfficientADConfig::test_default_model_config_loads`
- Symptom: asserts `teacher.allow_download is False`, but
  `configs/models/efficientad_default.yaml` sets `allow_download: true`.
- Status: **PRE_EXISTING_FAILURE** — reproduced identically in the parent
  research repository against the same config before any G2 change.
- Policy: intentionally NOT "fixed" by editing the config or the assertion,
  because both are part of the frozen scientific configuration provenance.
  Resolving it requires an author decision on which side is canonical.

## KI-2 — Full end-to-end reproduction not executed during G2

- `scripts/reproduce_main.py --mode execute` is implemented (isolated scratch
  copy; sealed artifacts never overwritten) but was NOT run during G2:
  running the full chain takes substantial compute time and G2 forbids
  re-running large sealed experiments unattended.
- Level-1 verification (`scripts/verify_frozen_results.py`) IS executed and
  passes at tolerance 1e-9.

## KI-3 — Figure byte-parity limits

- Re-rendered quantitative figure PDFs are content-identical to shipped
  assets only after blanking matplotlib date metadata
  (`CONTENT_IDENTICAL_AFTER_DATE_NORM`). SVG/PNG bytes differ due to font and
  encoder metadata. See `scripts/reproduce_figures.py` output.
- The shipped `Fig1_NorOpCal_Overall_Framework.svg` predates a later revision
  of `make_fig1.py`; regeneration produces `Fig1_NorOpCal_Framework.*`
  (DATA_PARITY level).

## KI-4 — AeBAD scripts require a legacy dataset root

- `tools/r03_d0_extract_patchcore.py` targets AeBAD_S (historical route,
  external validation honestly abstained in the paper). It now requires
  `NOROPCAL_AEBAD_S_ROOT`; without the dataset these scripts cannot run.
  This does not affect any paper number.
