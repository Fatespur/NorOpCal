#!/usr/bin/env python3
"""PB1-DS-PB6 statistics — B=5000 category-condition block bootstrap.

Computes block-bootstrap 95% CIs for each selector/control from the D5 OOF, using
the frozen pb1ds_c0_common.block_bootstrap_oof (category-condition blocks sampled
together, preserving all three detectors of a unit). B = 5000 as frozen.

Per candidate, the OOF rows are filtered to that candidate and passed verbatim to
the frozen bootstrap; the frozen block resampling pairs each (category,condition)
block as one atomic unit. Negative results pass through unmodified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pb1ds_c0_common as C0  # noqa: E402  (frozen block bootstrap)

PB6_DIR = ROOT / "reports" / "pb1ds_publication" / "pb6_real_third_detector"
CANDIDATES = ["A_GLOBAL", "D_CATEGORY_ONLY", "C_CATEGORY_CONDITIONED"]
CONTROLS = ["NO_ADAPT", "ALWAYS_ON_QM", "ORACLE"]
METHODS = CANDIDATES + CONTROLS
BOOT_REPLICATES = 5000
SEED = 42


def main() -> int:
    oof = pd.read_csv(PB6_DIR / "PB1DS_PB6_D5_OOF_PER_UNIT.csv")
    results: dict[str, Any] = {}
    for cand in METHODS:
        sub = oof[oof["candidate"] == cand]
        if sub.empty:
            results[cand] = {"error": "no rows"}
            continue
        results[cand] = C0.block_bootstrap_oof(sub, n_boot=BOOT_REPLICATES, seed=SEED)
        results[cand]["n_entities"] = len(sub)

    (PB6_DIR / "PB1DS_PB6_BLOCK_BOOTSTRAP_5000.json").write_text(
        json.dumps({"replicates": BOOT_REPLICATES, "seed": SEED, "methods": results}, indent=2),
        encoding="utf-8",
    )

    lines: list[str] = [
        "# PB1-DS-PB6 — B=5000 Block Bootstrap CIs",
        "",
        f"- Replicates: {BOOT_REPLICATES}, seed {SEED}, block=(category,condition).",
        "",
    ]
    for cand, r in results.items():
        if "error" in r:
            lines.append(f"- {cand}: {r['error']}")
            continue
        lines.append(
            f"- {cand}: delta_ba_vs_noadapt "
            f"mean={r['delta_ba_vs_noadapt']['mean']:.4f} "
            f"CI=[{r['delta_ba_vs_noadapt']['ci_low']:.4f}, "
            f"{r['delta_ba_vs_noadapt']['ci_high']:.4f}] "
            f"(n={r['n_entities']})"
        )
    (PB6_DIR / "PB1DS_PB6_BLOCK_BOOTSTRAP_5000.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
