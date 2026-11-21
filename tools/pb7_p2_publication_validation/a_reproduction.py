"""PB7-P2 Stage A — Exact Reproduction of the PB7-P1 principal result.

Before any new robustness test, the PB7-P1 principal result (B4 =
A_MEDIAN_RAW lambda=0.25 cutoff=Q50) must be reproduced exactly from the
frozen inputs using the frozen method. The recomputed per-entity metrics are
compared against the sealed PB7-P1 E_TEST_PER_ENTITY.csv / summary.

Gate: absolute difference < 1e-9 for pooled and DINOv2 delta_ba, delta_fnr,
harm, coverage at K=8. On failure the terminal is
PB7_P2_STOP_REPRODUCTION_FAILURE and no further stage runs.
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

import pb7_p1_common as P1  # type: ignore[import-not-found]  # noqa: E402
import pb7_p2_common as P2  # type: ignore[import-not-found]  # noqa: E402

OUT = P2.OUT
TOL = 1e-9


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    sealed = pd.read_csv(P1.OUT / "E_TEST_PER_ENTITY.csv")

    fw = P2.Firewall()
    ents = P2.entity_list()
    test_ents = [e for e in ents if P2.split_side(e[0]) == "TEST"]

    rows: list[dict[str, Any]] = []
    for cat, cond, det in test_ents:
        src = P2.load_src(cat, det, fw)
        cnd = P2.load_cnd(cat, cond, det, fw)
        anm = P2.load_anm(cat, det, fw)
        tau_source = P2.source_threshold(src)
        for k_value in P2.K_GRID:
            r = P1.evaluate_entity(
                cat,
                cond,
                det,
                "A_MEDIAN_RAW",
                0.25,
                P2.PRINCIPAL_CUTOFF,
                k_value,
                src,
                cnd,
                anm,
                tau_source,
            )
            rows.append(
                {
                    "category": cat,
                    "condition": cond,
                    "detector": det,
                    "baseline": "B4",
                    "K": k_value,
                    "delta_ba": r["delta_ba"],
                    "delta_fpr": r["delta_fpr"],
                    "delta_fnr": r["delta_fnr"],
                    "benefit": r["benefit"],
                    "harm": r["harm"],
                    "catastrophic": r["catastrophic"],
                    "coverage": r["coverage"],
                    "n_repeats": r["n_repeats"],
                }
            )
    df_new = pd.DataFrame(rows)
    df_new.to_csv(OUT / "A_REPRODUCTION_PER_ENTITY.csv", index=False)

    # ---- compare per-entity against sealed ---------------------------------- #
    sealed_b4 = sealed[sealed.baseline == "B4"]
    merged = df_new.merge(
        sealed_b4[
            [
                "category",
                "condition",
                "detector",
                "K",
                "delta_ba",
                "delta_fpr",
                "delta_fnr",
                "harm",
                "catastrophic",
                "coverage",
            ]
        ],
        on=["category", "condition", "detector", "K"],
        suffixes=("_new", "_sealed"),
    )
    diffs: dict[str, list[float]] = {
        k: [] for k in ["delta_ba", "delta_fpr", "delta_fnr", "harm", "catastrophic", "coverage"]
    }
    for k in diffs:
        diffs[k] = (merged[f"{k}_new"] - merged[f"{k}_sealed"]).abs().tolist()
    max_diff = {k: float(max(v)) if v else float("nan") for k, v in diffs.items()}
    n_mismatch = int((merged["delta_ba_new"] - merged["delta_ba_sealed"]).abs().gt(TOL).sum())

    # ---- summary comparison (pooled + DINOv2, K=8) -------------------------- #
    def summary_for(df: pd.DataFrame, group: str, k_value: str) -> dict[str, float]:
        sub = df[k_value == df.K]
        if group is not None:
            sub = sub[sub.detector == group]
        return P2.summary_metrics(sub)

    ref_rows: list[dict[str, Any]] = []
    for k_value in P2.K_GRID:
        for group, det_name in [("POOLED", None), ("DINOv2", "dinov2_vits14")]:
            s_new = summary_for(df_new, det_name, k_value)
            s_sealed = summary_for(sealed_b4, det_name, k_value)
            ref_rows.append(
                {
                    "K": k_value,
                    "group": group,
                    "new_mean_delta_ba": s_new["mean_delta_ba"],
                    "sealed_mean_delta_ba": s_sealed["mean_delta_ba"],
                    "diff_delta_ba": abs(s_new["mean_delta_ba"] - s_sealed["mean_delta_ba"]),
                    "new_mean_delta_fnr": s_new["mean_delta_fnr"],
                    "sealed_mean_delta_fnr": s_sealed["mean_delta_fnr"],
                    "diff_delta_fnr": abs(s_new["mean_delta_fnr"] - s_sealed["mean_delta_fnr"]),
                    "new_harm_rate": s_new["harm_rate"],
                    "sealed_harm_rate": s_sealed["harm_rate"],
                    "diff_harm": abs(s_new["harm_rate"] - s_sealed["harm_rate"]),
                    "new_coverage": s_new["coverage"],
                    "sealed_coverage": s_sealed["coverage"],
                    "diff_coverage": abs(s_new["coverage"] - s_sealed["coverage"]),
                }
            )
    df_ref = pd.DataFrame(ref_rows)
    df_ref.to_csv(OUT / "A_REPRODUCTION.csv", index=False)

    # ---- gate: K=8 pooled + DINOv2, all four metrics < 1e-9 ------------------ #
    k8 = df_ref[df_ref.K == "8"]
    gate_checks = {}
    ok = True
    for group in ["POOLED", "DINOv2"]:
        row = k8[k8.group == group].iloc[0]
        for metric in ["delta_ba", "delta_fnr", "harm", "coverage"]:
            d = float(row[f"diff_{metric}"])
            gate_checks[f"{group}_{metric}_diff_lt_1e9"] = d < TOL
            gate_checks[f"{group}_{metric}_diff"] = d
            if d >= TOL:
                ok = False
    gate_checks["n_per_entity_mismatch_gt_1e9"] = n_mismatch
    if n_mismatch > 0:
        ok = False

    gate = {
        "schema_version": "1.0",
        "stage": "PB7_P2_PUBLICATION_READINESS_CONFIRMATORY_VALIDATION",
        "phase": "A",
        "timestamp": ts,
        "tolerance": TOL,
        "checks": gate_checks,
        "max_per_entity_abs_diff": max_diff,
        "verdict": "PASS" if ok else "FAIL",
        "terminal": "PROCEED_TO_B" if ok else "PB7_P2_STOP_REPRODUCTION_FAILURE",
    }
    (OUT / "A_EXACT_REPRODUCTION.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    # ---- ledger -------------------------------------------------------------- #
    ledger = pd.read_csv(OUT / "P0_EXECUTION_LEDGER.csv")
    ledger = pd.concat(
        [
            ledger,
            pd.DataFrame(
                [
                    {
                        "stage": "A",
                        "step": "exact_reproduction",
                        "status": "COMPLETE" if ok else "FAIL",
                        "timestamp": ts,
                        "artifacts": "A_REPRODUCTION.csv, A_REPRODUCTION_PER_ENTITY.csv, "
                        "A_EXACT_REPRODUCTION.json",
                        "note": (
                            f"A gate {'PASS' if ok else 'FAIL'}; max per-entity abs diff "
                            f"dBA={max_diff['delta_ba']:.2e} dFNR={max_diff['delta_fnr']:.2e} "
                            f"harm={max_diff['harm']:.2e} cov={max_diff['coverage']:.2e}"
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ledger.to_csv(OUT / "P0_EXECUTION_LEDGER.csv", index=False)

    print(json.dumps(gate, indent=2))
    print("\nReproduction summary (K=8):")
    print(df_ref[df_ref.K == "8"].round(10).to_string(index=False))


if __name__ == "__main__":
    main()
