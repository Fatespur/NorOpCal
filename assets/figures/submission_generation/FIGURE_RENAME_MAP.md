# Figure Rename Map (2026-08-21)

This curation step renamed the four final main-text figures to formal submission names and moved all deprecated/duplicate/QA exports to `_archive_old_versions/`. File contents were not modified; every SHA-256 recorded in `pipeline_manifest.json` still matches the renamed file. The manifest remains the historical record of the S5 generation run and was not edited.

## Final main-text figures (current names)

| Final name | Previous name | Basis |
|---|---|---|
| `Fig1_NorOpCal_Overall_Framework.svg` | `Figure 1. Overall framework of NorOpCal for normal-only selective operating-point calibration under acquisition shift..svg` | Final adopted framework diagram (only format available) |
| `Fig2_Gate_Informativeness_and_Correctability_Diagnostic.{pdf,png,svg}` | `Fig3_Gate_Informativeness_and_Correctability_REFINED.{pdf,png,svg}` | Latest two-panel diagnostic: (a) gate informativeness, (b) correctability diagnostic |
| `Fig3_Utility_Across_Frozen_Support_Settings.{pdf,png,svg}` | `Utility_across_Frozen_Support_Settings_REFINED.{pdf,png,svg}` | Final single-panel mean DeltaBA plot for K=4/K=8/Full |
| `Fig4_Safety_Across_Frozen_Support_Settings.{pdf,png,svg}` | `Safety_across_Frozen_Support_Settings_REFINED.{pdf,png,svg}` | Final single-panel mean DeltaFNR plot for K=4/K=8/Full |

## Deprecated assets moved to `_archive_old_versions/`

- `Fig2_Utility_Safety_and_Detector_Specificity_*` (all variants) — early composite utility-safety two-panel map, superseded by the two independent Fig. 3 / Fig. 4 plots.
- `Fig3_Gate_Informativeness_and_Correctability.{pdf,png,svg}` and `..._grayscale.png` — pre-refinement version.
- `Fig3_Gate_Informativeness_and_Correctability_singlecol.*` — single-column variant of the pre-refinement version.
- `*_REFINED_grayscale.png` and `*_REFINED_90pct_preview.png` for the final figures — QA-only previews, not manuscript assets.
- `__pycache__/` — Python bytecode cache.

## Notes

- Figure numbering follows the intended main-text order: Fig. 1 framework, Fig. 2 gate/correctability diagnostic, Fig. 3 utility, Fig. 4 safety. The figures have not yet been inserted into `NorOpCal_TIM_Manuscript.docx`; insertion and caption registration are a separate later step.
- Generation scripts (`make_*.py`, `figure_*.py`, `build_manifest.py`), contracts/reports (`FIGURE_CONTRACT.md`, `REFINED_FIGURE_CONTRACT.md`, `FIGURE_GENERATION_REPORT.md`), `pipeline_manifest.json`, and `data/` snapshots are intentionally kept in place for reproducibility; they are not figure assets.
- Regenerating figures with the scripts will emit the old filenames; the mapping above restores the formal names.
