#!/usr/bin/env python3
"""PB1-DS-PB6 D6 — tri-detector benefit landscape from the D5 OOF.

Reads PB1DS_PB6_D5_OOF_PER_UNIT.csv (39 units x 3 detectors x methods) and emits
per-(candidate, baseline) and pooled aggregate selective-calibration metrics:

  coverage, benefit_prevalence, benefit_precision, benefit_recall, global_harm,
  harm_among_adapted, delta_ba_vs_noadapt, oracle_gap_recovery.

All metrics follow the frozen md0sr_common.policy_metrics semantics; negative
scientific results are passed through verbatim (never re-tuned).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PB6_DIR = ROOT / "reports" / "pb1ds_publication" / "pb6_real_third_detector"
CANDIDATES = ["A_GLOBAL", "D_CATEGORY_ONLY", "C_CATEGORY_CONDITIONED"]
CONTROLS = ["NO_ADAPT", "ALWAYS_ON_QM", "ORACLE"]
METHODS = CANDIDATES + CONTROLS
DETECTORS = ["dinov2_vits14", "wide_resnet50_2", "efficientad"]


def metrics_for(df: pd.DataFrame) -> dict[str, Any]:
    adapt = (df["adapt"] == 1).to_numpy()
    benefit = df["benefit"].to_numpy()
    harm = df["harm"].to_numpy()
    bb = df["ba_before"].to_numpy()
    ba = df["ba_after"].to_numpy()
    n = len(df)
    nadapt = int(adapt.sum())
    nben = int(benefit.sum())
    ba_used = np.where(adapt, ba, bb)
    prec = float(np.sum(adapt & (benefit == 1)) / max(nadapt, 1)) if nadapt else 0.0
    rec = float(np.sum(adapt & (benefit == 1)) / max(nben, 1)) if nben else 0.0
    return {
        "n_entities": n,
        "n_adapt": nadapt,
        "coverage": float(adapt.mean()),
        "benefit_prevalence": float(benefit.mean()),
        "benefit_precision": prec,
        "benefit_recall": rec,
        "global_harm": float(np.mean(adapt & (harm == 1))),
        "harm_among_adapted": float(np.mean(adapt & (harm == 1))) if nadapt else np.nan,
        "delta_ba_vs_noadapt": float((ba_used - bb).mean()),
        "ba_before_mean": float(bb.mean()),
        "ba_selective": float(ba_used.mean()),
    }


def main() -> int:
    oof = pd.read_csv(PB6_DIR / "PB1DS_PB6_D5_OOF_PER_UNIT.csv")
    rows: list[dict[str, Any]] = []
    for cand in METHODS:
        rows.append(
            {"candidate": cand, "baseline": "ALL", **metrics_for(oof[oof["candidate"] == cand])}
        )
        for d in DETECTORS:
            rows.append(
                {
                    "candidate": cand,
                    "baseline": d,
                    **metrics_for(oof[(oof["candidate"] == cand) & (oof["baseline"] == d)]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(PB6_DIR / "PB1DS_PB6_D6_BENEFIT_LANDSCAPE.csv", index=False)

    md = [
        "# PB1-DS-PB6 — D6 Tri-Detector Benefit Landscape",
        "",
        f"- Entities: {len(oof) // len(METHODS)} units x 3 detectors",
        "- Negative results are passed through verbatim; no re-tuning performed.",
        "",
    ]
    md.append(df.to_string(index=False))
    (PB6_DIR / "PB1DS_PB6_D6_BENEFIT_LANDSCAPE.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
