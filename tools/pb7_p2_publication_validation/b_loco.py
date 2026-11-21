"""PB7-P2 Stage B — Leave-One-Category-Out Influence Audit.

For each of the 8 categories, remove it and re-run the FROZEN principal policy
(A_MEDIAN_RAW lambda=0.25, median_shift gate, K=8) on the remaining TEST
evaluation set:

  - If the removed category is a TEST category: evaluation set shrinks to the
    other 4 TEST categories; the frozen Q50 cutoff is kept unchanged.
  - If the removed category is a DEV category: evaluation set stays the 5 TEST
    categories, but the Q50 cutoff is rebuilt strictly by the frozen DEV
    procedure (Q50 of the FULL-support median_shift distribution over the
    remaining DEV categories only; normal-only, never test-anomaly).

Per LOO run we report DINOv2 delta_BA, delta_FNR, harm, coverage, benefit
precision/recall. Influence_i = dBA_full - dBA_without_i.

Robustness gate (pre-registered):
  - >= 7/8 LOO runs: DINOv2 delta_BA > 0
  - >= 7/8 LOO runs: DINOv2 delta_FNR <= +0.02
  - no single-category removal leads to DINOv2 delta_BA <= -0.01
  - no single category provides > 50% of the total DINOv2 utility
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p1_safety_actuator"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p2_publication_validation"))

import md0sr_common as C  # type: ignore[import-not-found]  # noqa: E402
import pb7_p1_common as P1  # type: ignore[import-not-found]  # noqa: E402
import pb7_p2_common as P2  # type: ignore[import-not-found]  # noqa: E402

OUT = P2.OUT
CATEGORIES = list(C.MVTEC_CATEGORIES)
DEV_CATS = ["can", "fruit_jelly", "wallplugs"]
TEST_CATS = ["fabric", "rice", "sheet_metal", "vial", "walnuts"]
DET = "dinov2_vits14"


def eval_dinov2(test_cats: list[str], cutoff: float) -> dict[str, Any]:
    """Evaluate the frozen principal policy on a TEST category subset (DINOv2)."""
    fw = P2.Firewall()
    ents = P2.entity_list()
    rows: list[dict[str, Any]] = []
    for cat, cond, det in ents:
        if cat not in test_cats or det != DET:
            continue
        src = P2.load_src(cat, det, fw)
        cnd = P2.load_cnd(cat, cond, det, fw)
        anm = P2.load_anm(cat, det, fw)
        tau_source = P2.source_threshold(src)
        r = P1.evaluate_entity(
            cat, cond, det, "A_MEDIAN_RAW", 0.25, cutoff, "8", src, cnd, anm, tau_source
        )
        rows.append(
            {
                "category": cat,
                "condition": cond,
                "detector": det,
                "delta_ba": r["delta_ba"],
                "delta_fpr": r["delta_fpr"],
                "delta_fnr": r["delta_fnr"],
                "benefit": r["benefit"],
                "harm": r["harm"],
                "catastrophic": r["catastrophic"],
                "coverage": r["coverage"],
            }
        )
    df = pd.DataFrame(rows)
    s = P2.summary_metrics(df)
    return {"n_entities": len(df), **s}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # full reference (no removal)
    full = eval_dinov2(TEST_CATS, P2.PRINCIPAL_CUTOFF)
    full_dba = float(full["mean_delta_ba"])

    rows: list[dict[str, Any]] = []
    for cat in CATEGORIES:
        removed_side = "DEV" if cat in DEV_CATS else "TEST"
        if removed_side == "DEV":
            dev_remaining = [c for c in DEV_CATS if c != cat]
            cutoff = P2.rebuild_cutoff_q50(dev_remaining)
            test_cats = TEST_CATS
        else:
            cutoff = P2.PRINCIPAL_CUTOFF
            test_cats = [c for c in TEST_CATS if c != cat]
        res = eval_dinov2(test_cats, cutoff)
        influence = full_dba - res["mean_delta_ba"]
        rows.append(
            {
                "removed_category": cat,
                "removed_side": removed_side,
                "cutoff_used": cutoff,
                "n_entities": res["n_entities"],
                "mean_delta_ba": res["mean_delta_ba"],
                "mean_delta_fpr": res["mean_delta_fpr"],
                "mean_delta_fnr": res["mean_delta_fnr"],
                "harm_rate": res["harm_rate"],
                "catastrophic_rate": res["catastrophic_rate"],
                "coverage": res["coverage"],
                "benefit_precision": res["benefit_precision"],
                "benefit_recall": res["benefit_recall"],
                "influence_dba": influence,
                "dba_gt_0": bool(res["mean_delta_ba"] > 0),
                "dfnr_le_002": bool(res["mean_delta_fnr"] <= P2.DFNR_MAX + 1e-12),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "B_LOCO_CATEGORY_INFLUENCE.csv", index=False)

    # ---- robustness gate ----------------------------------------------------- #
    n_dba_gt0 = int(df["dba_gt_0"].sum())
    n_dfnr_le = int(df["dfnr_le_002"].sum())
    max_influence = float(df["influence_dba"].max())
    max_influence_cat = str(df.loc[df["influence_dba"].idxmax(), "removed_category"])
    no_neg_001 = bool((df["mean_delta_ba"] > -0.01).all())
    no_single_gt_50pct = bool(max_influence <= 0.5 * full_dba) if full_dba > 0 else False

    gate_checks = {
        "full_dba": full_dba,
        "n_loco_runs": len(df),
        "n_runs_dba_gt_0": n_dba_gt0,
        "n_runs_dfnr_le_002": n_dfnr_le,
        "max_influence_dba": max_influence,
        "max_influence_category": max_influence_cat,
        "no_run_dba_le_neg001": no_neg_001,
        "no_single_category_gt_50pct_utility": no_single_gt_50pct,
    }
    ok = n_dba_gt0 >= 7 and n_dfnr_le >= 7 and no_neg_001 and no_single_gt_50pct
    gate = {
        "schema_version": "1.0",
        "stage": "PB7_P2_PUBLICATION_READINESS_CONFIRMATORY_VALIDATION",
        "phase": "B",
        "timestamp": ts,
        "checks": gate_checks,
        "verdict": "PASS" if ok else "FAIL",
        "terminal": "PROCEED_TO_C" if ok else "PROCEED_TO_C_WITH_CATEGORY_FRAGILE",
        "category_fragile_flag": not ok,
    }
    (OUT / "B_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    # ---- ledger -------------------------------------------------------------- #
    ledger = pd.read_csv(OUT / "P0_EXECUTION_LEDGER.csv")
    ledger = pd.concat(
        [
            ledger,
            pd.DataFrame(
                [
                    {
                        "stage": "B",
                        "step": "loo_category_influence_audit",
                        "status": "COMPLETE",
                        "timestamp": ts,
                        "artifacts": "B_LOCO_CATEGORY_INFLUENCE.csv, B_GATE.json",
                        "note": (
                            f"B gate {'PASS' if ok else 'FAIL'}; dBA>0 in {n_dba_gt0}/8, "
                            f"dFNR<=0.02 in {n_dfnr_le}/8, max influence={max_influence:.4f} "
                            f"({max_influence_cat})"
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ledger.to_csv(OUT / "P0_EXECUTION_LEDGER.csv", index=False)

    print(json.dumps(gate, indent=2))
    print("\nLOO DINOv2 results:")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
