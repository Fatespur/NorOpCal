#!/usr/bin/env python3
"""PB1-DS-PB6 entry determination — read-only, data-driven a∧b∧c check.

Step 1 of the post-D2 driver task. Enumerates per-category condition sets from
the actual test_public image filenames (no hard-coded counts or condition sets),
then verifies:

  a. score caches complete: 1 train + 2 x |CONDS{cat}| public files per category
  b. terminal checkpoints complete: checkpoint_step_00070000.pt + .complete
  c. execution manifest complete: exactly 8 rows, all status=complete, seed=0

Read-only; never writes anything.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env_dir(var: str) -> Path:
    """Resolve a machine-specific root from an environment variable.

    Path injection only; experiment logic is unchanged. Set
    NOROPCAL_MVTEC_AD2_ROOT / NOROPCAL_PB6_WORK_ROOT explicitly.
    """
    value = os.environ.get(var)
    if not value:
        raise SystemExit(
            f"missing required environment variable {var}; path injection only, "
            "experiment logic unchanged"
        )
    return Path(value)


MV2_ROOT = _env_dir("NOROPCAL_MVTEC_AD2_ROOT")
WORK_ROOT = _env_dir("NOROPCAL_PB6_WORK_ROOT")
SCORE_DIR = ROOT / "reports" / "p0_dataset_opportunity" / "_cache" / "scores"
PB6_DIR = ROOT / "reports" / "pb1ds_publication" / "pb6_real_third_detector"
CATEGORIES = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]
FROZEN_STEPS = 70000

_SUFFIX_RE = re.compile(r"^\d+_(?P<cond>.+?)(?:_mask)?\.png$")


def enumerate_conditions(cat: str) -> list[str]:
    conds: set[str] = set()
    for split in ("good", "bad"):
        d = MV2_ROOT / cat / "test_public" / split
        if not d.is_dir():
            continue
        for p in d.glob("*.png"):
            m = _SUFFIX_RE.match(p.name)
            if not m:
                continue
            cond = m.group("cond")
            if cond.endswith("_mask"):
                continue
            conds.add(cond)
    return sorted(conds)


def check_caches(cat: str, conds: list[str]) -> tuple[bool, list[str]]:
    expected = [f"mv2_{cat}_train_efficientad_scores.npz"]
    for cond in conds:
        expected.append(f"mv2_{cat}_public_good_{cond}_efficientad_scores.npz")
        expected.append(f"mv2_{cat}_public_bad_{cond}_efficientad_scores.npz")
    missing = [f for f in expected if not (SCORE_DIR / f).is_file()]
    return (not missing, missing)


def check_checkpoints(cat: str) -> tuple[bool, list[str]]:
    ckpt = WORK_ROOT / cat / "checkpoints" / f"checkpoint_step_{FROZEN_STEPS:08d}.pt"
    done = WORK_ROOT / cat / "checkpoints" / f"checkpoint_step_{FROZEN_STEPS:08d}.pt.complete"
    missing = []
    if not ckpt.is_file():
        missing.append(str(ckpt))
    if not done.is_file():
        missing.append(str(done))
    return (not missing, missing)


def read_manifest() -> list[dict[str, str]]:
    import csv

    p = PB6_DIR / "PB1DS_PB6_EFFICIENTAD_EXECUTION_MANIFEST.csv"
    if not p.is_file():
        return []
    with p.open(newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def main() -> int:
    print("=== PB1-DS-PB6 entry determination (read-only) ===")
    conds_by_cat: dict[str, list[str]] = {}
    a_ok = True
    b_ok = True
    for cat in CATEGORIES:
        conds = enumerate_conditions(cat)
        conds_by_cat[cat] = conds
        a, a_missing = check_caches(cat, conds)
        b, b_missing = check_checkpoints(cat)
        a_ok = a_ok and a
        b_ok = b_ok and b
        print(f"  {cat}: |CONDS|={len(conds)} conds={conds}")
        print(f"      a.cache_ok={a} (expected 1+2x{len(conds)}={1 + 2 * len(conds)} files)")
        if a_missing:
            print(f"      missing cache: {a_missing}")
        print(f"      b.checkpoint_ok={b}")
        if b_missing:
            print(f"      missing checkpoint: {b_missing}")

    manifest = read_manifest()
    c_ok = (
        len(manifest) == len(CATEGORIES)
        and all(r.get("status") == "complete" for r in manifest)
        and all(str(r.get("seed", "")) == "0" for r in manifest)
        and all(r.get("category") in CATEGORIES for r in manifest)
    )
    print(f"  manifest: rows={len(manifest)} c_ok={c_ok}")
    if not c_ok:
        for r in manifest:
            print(f"      {r.get('category')} status={r.get('status')} seed={r.get('seed')}")

    print()
    print(f"  a (caches complete)      = {a_ok}")
    print(f"  b (checkpoints complete) = {b_ok}")
    print(f"  c (manifest complete)    = {c_ok}")
    verdict = "D2_COMPLETE" if (a_ok and b_ok and c_ok) else "D2_INCOMPLETE"
    print(f"  VERDICT: {verdict}")
    return 0 if a_ok and b_ok and c_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
