#!/usr/bin/env python3
"""PB1-DS-PB6 D7 — tri-detector benchmark landscape (incl. canonical BC_RANK).

Replays the frozen canonical external baseline BC_RANK_BASED_THRESHOLD (the
sealed representative of the BC_TARGET_NORMAL_THRESHOLD_FAMILY) as an
unconditional recalibrator on the tri-detector frame (39 units x 3 detectors =
117 entities), then merges it with the sealed D5 OOF (A_GLOBAL,
D_CATEGORY_ONLY, C_CATEGORY_CONDITIONED, NO_ADAPT, ALWAYS_ON_QM, ORACLE) and
emits the full benchmark + benefit landscape.

The BC_RANK actuator is the frozen pb1ds_pb2_common.actuator_bc_rank_threshold
(per-entity operating threshold = Q95(target-normal), no score remap,
always_on). No refit / no threshold retune / no reselection is performed.

Negative scientific results pass through verbatim; this stage never re-tunes to
force a PASS.
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

import md0sr_common as C  # noqa: E402
import pb1ds_c0_common as M0  # noqa: E402  (classify_change)
import pb1ds_pb2_common as B2  # noqa: E402  (frozen BC_RANK actuator)
import pb6_tri_common as T  # noqa: E402

PB6_DIR = ROOT / "reports" / "pb1ds_publication" / "pb6_real_third_detector"
CANDIDATES = ["A_GLOBAL", "D_CATEGORY_ONLY", "C_CATEGORY_CONDITIONED"]
CONTROLS = ["NO_ADAPT", "ALWAYS_ON_QM", "ORACLE"]
BC_RANK = "BC_RANK_BASED_THRESHOLD"
METHODS = CANDIDATES + CONTROLS + [BC_RANK]
DETECTORS = ["dinov2_vits14", "wide_resnet50_2", "efficientad"]
Q95 = 0.95


def metrics_for(df: pd.DataFrame) -> dict[str, Any]:
    """Same frozen policy-metric semantics as D6 (md0sr policy metrics)."""
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


def bc_rank_row(cat: str, cond: str, det: str, fold_index: int) -> dict[str, Any]:
    """Frozen BC_RANK_BASED_THRESHOLD row (always_on, tau=Q95(target-normal))."""
    src = C.load_image_scores(C.source_rel(cat, det))
    cnd = C.load_image_scores(C.cond_normal_rel(cat, cond, det))
    anm = C.load_image_scores(C.anomaly_rel(cat, det))
    labels = np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))])
    tau_before = float(np.percentile(src, 100.0 * Q95))
    mb = C.detection_metrics(np.concatenate([cnd, anm]), labels, tau_before)
    after_n, after_a, tau_after = B2.actuator_bc_rank_threshold(src, cnd, anm)
    ma = C.detection_metrics(np.concatenate([after_n, after_a]), labels, tau_after)
    dba = ma["ba"] - mb["ba"]
    dfpr = ma["fpr"] - mb["fpr"]
    dfnr = ma["fnr"] - mb["fnr"]
    benefit, harm, neutral = M0.classify_change(dba, dfpr, dfnr)
    if benefit == 1:
        outcome = "BENEFIT"
    elif harm == 1:
        outcome = "HARM"
    else:
        outcome = "NEUTRAL"
    return {
        "outer_fold_index": fold_index,
        "category": cat,
        "condition": cond,
        "baseline": det,
        "baseline_name": T.TRI_DET_NAMES[det],
        "candidate": BC_RANK,
        "selector_prob": float("nan"),
        "selector_threshold": None,
        "adapt": 1,
        "ba_before": float(mb["ba"]),
        "ba_after": float(ma["ba"]),
        "delta_ba": float(dba),
        "ba_used": float(ma["ba"]),
        "fpr_before": float(mb["fpr"]),
        "fpr_after": float(ma["fpr"]),
        "delta_fpr": float(dfpr),
        "fpr_used": float(ma["fpr"]),
        "fnr_before": float(mb["fnr"]),
        "fnr_after": float(ma["fnr"]),
        "delta_fnr": float(dfnr),
        "fnr_used": float(ma["fnr"]),
        "f1_before": float(mb["f1"]),
        "f1_after": float(ma["f1"]),
        "f1_used": float(ma["f1"]),
        "benefit": int(benefit),
        "harm": int(harm),
        "neutral": int(neutral),
        "policy_outcome": outcome,
    }


def main() -> int:
    oof = pd.read_csv(PB6_DIR / "PB1DS_PB6_D5_OOF_PER_UNIT.csv")

    # fold index per unit from the sealed D5 split manifest
    split = __import__("json").loads(
        (PB6_DIR / "PB1DS_PB6_D5_SPLIT_MANIFEST.json").read_text(encoding="utf-8")
    )
    fold_by_unit = {(f["test"][0], f["test"][1]): int(f["index"]) for f in split["outer_folds"]}

    units = sorted(set(zip(oof["category"], oof["condition"], strict=True)))
    bc_rows: list[dict[str, Any]] = []
    for cat, cond in units:
        for det in DETECTORS:
            bc_rows.append(bc_rank_row(cat, cond, det, fold_by_unit[(cat, cond)]))
    bc = pd.DataFrame(bc_rows)

    frame = pd.concat([oof, bc], ignore_index=True)
    frame.to_csv(PB6_DIR / "PB1DS_PB6_BENCHMARK_FRAME.csv", index=False)

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        rows.append(
            {
                "candidate": method,
                "baseline": "ALL",
                **metrics_for(frame[frame["candidate"] == method]),
            }
        )
        for d in DETECTORS:
            rows.append(
                {
                    "candidate": method,
                    "baseline": d,
                    **metrics_for(frame[(frame["candidate"] == method) & (frame["baseline"] == d)]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(PB6_DIR / "PB1DS_PB6_BENCHMARK_LANDSCAPE.csv", index=False)

    md = [
        "# PB1-DS-PB6 — D7 Tri-Detector Benchmark Landscape",
        "",
        f"- Entities: {len(frame) // len(METHODS)} units x 3 detectors "
        f"({len(frame)} rows; {len(METHODS)} methods incl. frozen BC_RANK replay)",
        "- Methods: " + ", ".join(METHODS),
        "- BC_RANK_BASED_THRESHOLD: frozen canonical BC family representative "
        "(per-entity Q95(target-normal) threshold, always_on, no score remap).",
        "- Negative results are passed through verbatim; no re-tuning performed.",
        "",
    ]
    md.append(df.to_string(index=False))
    (PB6_DIR / "PB1DS_PB6_BENCHMARK_LANDSCAPE.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
