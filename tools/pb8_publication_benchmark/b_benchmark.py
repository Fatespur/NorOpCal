"""PB8 B — final benchmark.

Evaluates the canonical frozen baseline registry (B0-B7) on the TEST split at
K8 (primary), K4 and FULL (auxiliary), verifies that the frozen proposed policy
(B5) reproduces the sealed PB7-P2 GATED result, and emits the main benchmark
tables + B_GATE.json. No outcome-driven baseline add/remove.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "pb8_publication_benchmark"))

import pb8_common as P8  # type: ignore[import-not-found]  # noqa: E402

OUT = P8.OUT
OUT.mkdir(parents=True, exist_ok=True)

DET = "dinov2_vits14"
K_GRID = ["4", "8", "FULL"]

REQUIRED_METRIC_COLUMNS = [
    "ba",
    "fpr",
    "fnr",
    "delta_ba",
    "delta_fpr",
    "delta_fnr",
    "harm",
]


def valid_metric_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only entities eligible for the requested K-support estimand.

    The frozen PB7 evaluator preserves an entity row with NaN metrics when
    ``K > target_normal_n``. Keeping those rows in the per-entity artifact is
    useful for auditing exclusions, but publication aggregates must exclude
    them before counting eligible categories or averaging absolute metrics.
    """
    return df.dropna(subset=REQUIRED_METRIC_COLUMNS).copy()


def ba_before_after(row: pd.Series) -> tuple[float, float]:
    return float(row["ba"] - row["delta_ba"]), float(row["ba"])


def fpr_before_after(row: pd.Series) -> tuple[float, float]:
    return float(row["fpr"] - row["delta_fpr"]), float(row["fpr"])


def fnr_before_after(row: pd.Series) -> tuple[float, float]:
    return float(row["fnr"] - row["delta_fnr"]), float(row["fnr"])


def main() -> None:
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # ---- Evaluate all canonical baselines at K4/K8/FULL (fresh, frozen eval) - #
    frames: list[pd.DataFrame] = []
    for bl in P8.BASELINES:
        for k in K_GRID:
            df = P8.evaluate_policy_per_entity(bl["family"], bl["lambda"], bl["cutoff"], k)
            df.insert(0, "baseline", bl["id"])
            df.insert(1, "baseline_name", bl["name"])
            df.insert(2, "role", bl["role"])
            frames.append(df)
    per_entity = pd.concat(frames, ignore_index=True)
    per_entity.to_csv(OUT / "B_PER_ENTITY_ALL.csv", index=False)

    # ---- Verify B5 (frozen proposed) reproduces sealed PB7-P2 GATED ---------- #
    sealed = pd.read_csv(P8.P2_OUT / "D_POLICY_GATED.csv")
    b5 = per_entity[(per_entity.baseline == "B5") & (per_entity.K == "8")]
    b5 = (
        b5[
            [
                "category",
                "condition",
                "detector",
                "delta_ba",
                "delta_fpr",
                "delta_fnr",
                "benefit",
                "harm",
                "catastrophic",
                "coverage",
            ]
        ]
        .sort_values(["category", "condition", "detector"])
        .reset_index(drop=True)
    )
    sealed = sealed.sort_values(["category", "condition", "detector"]).reset_index(drop=True)
    merge = b5.merge(sealed, on=["category", "condition", "detector"], suffixes=("_new", "_sealed"))
    max_dba = float(np.max(np.abs(merge["delta_ba_new"] - merge["delta_ba_sealed"])))
    max_dfnr = float(np.max(np.abs(merge["delta_fnr_new"] - merge["delta_fnr_sealed"])))
    max_harm = float(np.max(np.abs(merge["harm_new"] - merge["harm_sealed"])))
    repro_ok = max(max_dba, max_dfnr, max_harm) < 1e-9
    repro = {
        "n_entities": len(merge),
        "max_abs_diff_delta_ba": max_dba,
        "max_abs_diff_delta_fnr": max_dfnr,
        "max_abs_diff_harm": max_harm,
        "reproduction_ok": bool(repro_ok),
    }
    (OUT / "B_REPRODUCTION_CHECK.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")

    # ---- Summary metrics per baseline x detector x K -------------------------- #
    rows: list[dict] = []
    for bl in P8.BASELINES:
        for k in K_GRID:
            sub = per_entity[(per_entity.baseline == bl["id"]) & (k == per_entity.K)]
            for det_name in [None, DET, "wide_resnet50_2", "efficientad"]:
                s = P8.summary_for(sub, det_name, k)
                rows.append(
                    {
                        "baseline": bl["id"],
                        "baseline_name": bl["name"],
                        "role": bl["role"],
                        "K": k,
                        "group": "POOLED" if det_name is None else det_name,
                        **s,
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "B_BASELINE_SUMMARY.csv", index=False)

    # ---- Bootstrap CI (B=10000 category-aware, PB7-P2 seed) for K8 ----------- #
    boot_rows: list[dict] = []
    for bl in P8.BASELINES:
        sub = per_entity[(per_entity.baseline == bl["id"]) & (per_entity.K == "8")]
        for det_name in [None, DET]:
            df = sub if det_name is None else sub[sub.detector == det_name]
            b = P8.confirmatory_bootstrap(df, seed=P8.CONF_BOOT_SEED, b=P8.CONF_BOOT_B)
            boot_rows.append(
                {
                    "baseline": bl["id"],
                    "group": "POOLED" if det_name is None else "DINOv2",
                    **b,
                }
            )
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT / "B_BOOTSTRAP_CI.csv", index=False)

    # ---- Main benchmark table (DINOv2 K8 primary, POOLED K8 secondary) ------- #
    main_rows: list[dict] = []
    for bl in P8.BASELINES:
        sub = per_entity[(per_entity.baseline == bl["id"]) & (per_entity.K == "8")]
        dino = sub[sub.detector == DET]
        pooled = sub
        for group, df in [("DINOv2", dino), ("POOLED", pooled)]:
            valid = valid_metric_rows(df)
            s = P8.summary_metrics(valid)
            b = P8.confirmatory_bootstrap(valid, seed=P8.CONF_BOOT_SEED, b=P8.CONF_BOOT_B)
            ba_b = np.mean([ba_before_after(r)[0] for _, r in valid.iterrows()])
            ba_a = np.mean([ba_before_after(r)[1] for _, r in valid.iterrows()])
            fpr_b = np.mean([fpr_before_after(r)[0] for _, r in valid.iterrows()])
            fpr_a = np.mean([fpr_before_after(r)[1] for _, r in valid.iterrows()])
            fnr_b = np.mean([fnr_before_after(r)[0] for _, r in valid.iterrows()])
            fnr_a = np.mean([fnr_before_after(r)[1] for _, r in valid.iterrows()])
            n_cats = int(valid["category"].nunique())
            safety_fail = bool(s["mean_delta_fnr"] > P8.DFNR_MAX + 1e-12)
            main_rows.append(
                {
                    "baseline": bl["id"],
                    "baseline_name": bl["name"],
                    "role": bl["role"],
                    "group": group,
                    "K": "8",
                    "n_units": int(s["n"]),
                    "n_eligible_categories": n_cats,
                    "ba_before": ba_b,
                    "ba_after": ba_a,
                    "delta_ba": s["mean_delta_ba"],
                    "dba_ci_lo": b["boot_ci_lo_delta_ba"],
                    "dba_ci_hi": b["boot_ci_hi_delta_ba"],
                    "fpr_before": fpr_b,
                    "fpr_after": fpr_a,
                    "delta_fpr": s["mean_delta_fpr"],
                    "fnr_before": fnr_b,
                    "fnr_after": fnr_a,
                    "delta_fnr": s["mean_delta_fnr"],
                    "harm_rate": s["harm_rate"],
                    "catastrophic_rate": s["catastrophic_rate"],
                    "coverage": s["coverage"],
                    "benefit_precision": s["benefit_precision"],
                    "benefit_recall": s["benefit_recall"],
                    "safety_fail": safety_fail,
                    "safety_label": "SAFETY FAIL" if safety_fail else "OK",
                }
            )
    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(OUT / "B_FINAL_MAIN_BENCHMARK.csv", index=False)

    # ---- Detector benchmark (K8, all detectors, principal B5 + key baselines) - #
    det_rows: list[dict] = []
    for bl in P8.BASELINES:
        sub = per_entity[(per_entity.baseline == bl["id"]) & (per_entity.K == "8")]
        for det_name in [DET, "wide_resnet50_2", "efficientad"]:
            df = sub[sub.detector == det_name]
            s = P8.summary_metrics(df)
            det_rows.append(
                {
                    "baseline": bl["id"],
                    "detector": det_name,
                    "n_units": int(s["n"]),
                    "delta_ba": s["mean_delta_ba"],
                    "delta_fpr": s["mean_delta_fpr"],
                    "delta_fnr": s["mean_delta_fnr"],
                    "harm_rate": s["harm_rate"],
                    "catastrophic_rate": s["catastrophic_rate"],
                    "coverage": s["coverage"],
                    "benefit_precision": s["benefit_precision"],
                    "benefit_recall": s["benefit_recall"],
                    "safety_fail": bool(s["mean_delta_fnr"] > P8.DFNR_MAX + 1e-12),
                }
            )
    det_df = pd.DataFrame(det_rows)
    det_df.to_csv(OUT / "B_DETECTOR_BENCHMARK.csv", index=False)

    # ---- Oracle safe threshold (B7) from sealed PB7-P0 headroom -------------- #
    oracle = pd.read_csv(P8.P0_OUT / "A2_ORACLE_HEADROOM.csv")
    oracle_test = oracle[
        oracle[["category", "condition", "detector"]].apply(
            lambda r: P8.split_side(r["category"]) == "TEST", axis=1
        )
    ]
    oracle_rows: list[dict] = []
    for det_name in [DET, "wide_resnet50_2", "efficientad", None]:
        df = oracle_test if det_name is None else oracle_test[oracle_test.detector == det_name]
        oracle_rows.append(
            {
                "baseline": "B7_ORACLE_SAFE",
                "group": "POOLED" if det_name is None else det_name,
                "n_units": len(df),
                "delta_ba": float(df["delta_ba_safe"].mean()),
                "delta_fpr": float(df["delta_fpr_safe"].mean()),
                "delta_fnr": float(df["delta_fnr_safe"].mean()),
                "coverage": float(df["correctable_safe"].mean()),
                "note": "DIAGNOSTIC UPPER BOUND; not a deployable comparator",
            }
        )
    oracle_df = pd.DataFrame(oracle_rows)
    oracle_df.to_csv(OUT / "B_ORACLE_SAFE.csv", index=False)

    # ---- Gate ----------------------------------------------------------------- #
    dino_main = main_df[(main_df.group == "DINOv2") & (main_df.baseline == "B5")].iloc[0]
    pooled_main = main_df[(main_df.group == "POOLED") & (main_df.baseline == "B5")].iloc[0]
    b1_dino = main_df[(main_df.group == "DINOv2") & (main_df.baseline == "B1")].iloc[0]
    b3_dino = main_df[(main_df.group == "DINOv2") & (main_df.baseline == "B3")].iloc[0]

    dba_ci_lo_ok = bool(dino_main["dba_ci_lo"] > 0)
    safety_ok = bool(dino_main["delta_fnr"] <= P8.DFNR_MAX + 1e-12)
    lower_harm_than_aggressive = bool(dino_main["harm_rate"] < b1_dino["harm_rate"] - 1e-12)
    repro_ok_bool = bool(repro_ok)
    gate_pass = dba_ci_lo_ok and safety_ok and lower_harm_than_aggressive and repro_ok_bool

    gate = {
        "schema_version": "1.0",
        "stage": "PB8_PUBLICATION_BENCHMARK_ABLATION_CLAIM_FREEZE",
        "phase": "B",
        "timestamp": ts,
        "checks": {
            "exact_reproduction_of_pb7p2_gated": repro_ok_bool,
            "max_abs_diff_delta_ba": max_dba,
            "max_abs_diff_delta_fnr": max_dfnr,
            "dino_dba_ci_lo_gt_0": dba_ci_lo_ok,
            "dino_dba_ci_lo": dino_main["dba_ci_lo"],
            "dino_dfnr_le_002": safety_ok,
            "dino_dfnr": dino_main["delta_fnr"],
            "dino_harm_lt_b1": lower_harm_than_aggressive,
            "dino_harm": dino_main["harm_rate"],
            "b1_dino_harm": b1_dino["harm_rate"],
            "b3_dino_dfnr": b3_dino["delta_fnr"],
            "b1_dino_dfnr": b1_dino["delta_fnr"],
            "pooled_dba_ci_lo": pooled_main["dba_ci_lo"],
            "pooled_dfnr": pooled_main["delta_fnr"],
        },
        "verdict": "PASS" if gate_pass else "FAIL",
        "terminal": "PROCEED_TO_C",
    }
    (OUT / "B_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(
        f"B done: repro_ok={repro_ok_bool}, DINOv2 B5 dBA={dino_main['delta_ba']:.4f} "
        f"CI_lo={dino_main['dba_ci_lo']:.4f}, dFNR={dino_main['delta_fnr']:.4f}, "
        f"verdict {gate['verdict']}"
    )


if __name__ == "__main__":
    main()
