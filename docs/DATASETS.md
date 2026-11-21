# Datasets

The paper uses exactly one benchmark dataset:

## MVTec AD 2 (AD2)

- 8 categories: can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs,
  walnuts.
- Unit = category × acquisition condition (39 units total; e.g. vial has 6
  conditions, fruit_jelly 3, others 5).
- Evaluation reads image-level anomaly scores from the *public* test splits:
  `test_public/good` images provide source/condition normal evidence and
  `test_public/bad` provides the regular defect evaluation pool.
- 315 unique target-normal images across conditions; 0 duplicates (PB8-S1
  census).

**The raw dataset is NOT distributed with this repository.** Obtain MVTec AD 2
from the official source and point the scoring entry points at it:

```bash
# PowerShell
$env:NOROPCAL_MVTEC_AD2_ROOT = "D:\path\to\mvtec_ad_2"
$env:NOROPCAL_PB6_WORK_ROOT  = "D:\path\to\efficientad_workdir"
```

Expected layout (as consumed by `src/shiftad/data/mvtec_ad.py` and the
scoring tools):

```text
<root>/
  <category>/
    train/good/*.png          # source-normal training split
    test_public/good/<cond>_<n>.png
    test_public/bad/<cond>_<n>.png
```

Condition names are derived from filename prefixes
(`overexposed`, `underexposed`, `shift_1..3`; condition enumeration is
data-driven in `tools/pb6/pb6_entry_determination.py`).

## Frozen DEV / TEST split (category level)

- DEV (selection only): can, fruit_jelly, wallplugs — 13 units.
- TEST (evaluation only): fabric, rice, sheet_metal, vial, walnuts — 26 units.
- Assignment is deterministic (SHA256 parity, salt `PB7_P1_DEV_TEST_SPLIT_v1`,
  `pb7_p1_common.split_side`).

K=8-eligible TEST units require ≥8 independent target-normal images:
fabric (5 units) + walnuts (5 units). The 16 ineligible units are true
target-normal shortage (PB8-S1 root-cause audit); semantic pooling candidates
were audited and none admissible.

## AeBAD / AeBAD_S

Historical development evidence for closed research routes only. No external-
validation claim exists in the paper (the frozen policy's gate degenerates to
always-on on AeBAD-S; honest abstention recorded). AeBAD caches are not part
of this repository.
