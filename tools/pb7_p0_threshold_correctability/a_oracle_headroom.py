"""PB7-P0 Stage A — Oracle Threshold Headroom Characterization.

Diagnosis only. No deployable method is developed here.

For every one of the 117 detector x unit entities, this stage:
  A1. records the source operating point (tau_source, BA/FPR/FNR before,
      AUROC, anomaly prevalence, counts);
  A2. constructs the full threshold -> BA/FPR/FNR curve on the frozen grid,
      and extracts tau*_BA (O1), tau*_SAFE (O2), their deltas, and the
      best achievable deltas;
  A3. reports detector-stratified headroom (DINOv2 / PatchCore / EfficientAD
      separately, plus pooled);
  A4. assigns each entity a pre-registered correctability taxonomy.

The taxonomy definitions are frozen in this module BEFORE any outcome is
computed (protocol: "定义必须事前固定"). All outputs are ORACLE_DIAGNOSTIC_ONLY.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p0_threshold_correctability"))

import md0sr_common as C  # type: ignore[import-not-found]  # noqa: E402
import pb7_p0_common as P  # type: ignore[import-not-found]  # noqa: E402

OUT = P.OUT

# --------------------------------------------------------------------------- #
# Frozen taxonomy constants (pre-registered at Stage A, before outcomes)
# --------------------------------------------------------------------------- #
NEAR_OPTIMAL_DBA = 0.01        # SOURCE_ALREADY_NEAR_OPTIMAL: oracle delta_ba < +0.01
UNSAFE_ONLY_DBA = 0.02         # UNSAFE_ONLY: unconstrained BA gain >= +0.02, safe gain < +0.02
AMBIGUOUS_BA_TOL = 0.005       # plateau tolerance for AMBIGUOUS_FLAT_CURVE
AMBIGUOUS_WIDTH_FRAC = 0.25    # plateau width >= 25% of pooled score range -> ambiguous
GENERALIZED_HEADROOM_MIN_N = 8     # H1 / A-gate: >= 8 units
GENERALIZED_HEADROOM_MIN_FRAC = 0.20  # H1 / A-gate: >= 20% of units

TAXONOMY_ORDER: List[str] = [
    "SAFE_THRESHOLD_CORRECTABLE",
    "UNSAFE_ONLY",
    "SOURCE_ALREADY_NEAR_OPTIMAL",
    "AMBIGUOUS_FLAT_CURVE",
    "NO_HEADROOM",
]


def auroc(cnd: np.ndarray, anm: np.ndarray) -> float:
    scores = np.concatenate([cnd, anm])
    labels = np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))])
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(C.roc_auc_score(labels, scores))


def plateau_width(grid: np.ndarray, ba_curve: np.ndarray, best_ba: float,
                  tol: float) -> float:
    """Width (in score units) of the tau region with BA within tol of best_ba."""
    mask = ba_curve >= best_ba - tol
    if not mask.any():
        return 0.0
    return float(np.max(grid[mask]) - np.min(grid[mask]))


def taxonomy(d: Dict[str, Any], pooled_range: float) -> Dict[str, Any]:
    """Pre-registered correctability taxonomy (flags + primary label).

    Flags (an entity may satisfy more than one):
      SAFE_THRESHOLD_CORRECTABLE : frozen benefit at tau*_SAFE
      UNSAFE_ONLY                : unconstrained BA gain >= +0.02 but no safe gain >= +0.02
      SOURCE_ALREADY_NEAR_OPTIMAL: safe and unconstrained oracle delta_ba both < +0.01
      AMBIGUOUS_FLAT_CURVE       : safe-optimum plateau covers >= 25% of pooled score range
      NO_HEADROOM                : fallback (no meaningful headroom anywhere)
    Primary label priority: SAFE > UNSAFE_ONLY > NEAR_OPTIMAL > AMBIGUOUS > NO_HEADROOM.
    """
    dba_safe = d["delta_ba_safe"]
    dba_ba = d["delta_ba_ba"]
    f_safe = int(d["correctable_safe"])
    f_unsafe = int((dba_ba >= UNSAFE_ONLY_DBA) and (dba_safe < UNSAFE_ONLY_DBA))
    f_near = int((dba_safe < NEAR_OPTIMAL_DBA) and (dba_ba < NEAR_OPTIMAL_DBA))
    width = d["safe_plateau_width"]
    f_amb = int((pooled_range > 0) and (width >= AMBIGUOUS_WIDTH_FRAC * pooled_range))
    f_none = int(not (f_safe or f_unsafe or f_near or f_amb))
    if f_safe:
        primary = "SAFE_THRESHOLD_CORRECTABLE"
    elif f_unsafe:
        primary = "UNSAFE_ONLY"
    elif f_near:
        primary = "SOURCE_ALREADY_NEAR_OPTIMAL"
    elif f_amb:
        primary = "AMBIGUOUS_FLAT_CURVE"
    else:
        primary = "NO_HEADROOM"
    return {
        "f_safe_correctable": f_safe,
        "f_unsafe_only": f_unsafe,
        "f_source_near_optimal": f_near,
        "f_ambiguous_flat": f_amb,
        "f_no_headroom": f_none,
        "primary_taxonomy": primary,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fw = P.Firewall()
    ents = P.entity_list()

    rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []

    for cat, cond, det in ents:
        src = P.load_src(cat, det, fw)
        cnd = P.load_cnd(cat, cond, det, fw)
        anm = P.load_anm(cat, det, fw)
        tau_src = P.source_threshold(src)

        labels = np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))])
        scores = np.concatenate([cnd, anm])
        before = C.detection_metrics(scores, labels, tau_src)

        o1 = P.oracle_ba_optimal(src, cnd, anm, tau_src)
        o2 = P.oracle_safe_optimal(src, cnd, anm, tau_src)
        corr = P.correctable_safe(before, o2)

        grid = P.candidate_grid(src, cnd, anm, tau_src)
        ba_curve = np.array([C.detection_metrics(scores, labels, float(t))["ba"] for t in grid])
        safe_best_ba = o2["ba"]
        safe_plateau_w = plateau_width(grid, ba_curve, safe_best_ba, AMBIGUOUS_BA_TOL)
        pooled_range = float(np.max(scores) - np.min(scores))

        d = {
            "category": cat, "condition": cond, "detector": det,
            "tau_source": tau_src,
            "ba_before": before["ba"], "fpr_before": before["fpr"],
            "fnr_before": before["fnr"],
            "auroc": auroc(cnd, anm),
            "anomaly_prevalence": len(anm) / (len(cnd) + len(anm)),
            "n_normal": int(len(cnd)), "n_anomaly": int(len(anm)),
            # O1
            "tau_star_ba": o1["tau_star"], "ba_ba": o1["ba"],
            "fpr_ba": o1["fpr"], "fnr_ba": o1["fnr"],
            "delta_tau_ba": o1["delta_tau"], "delta_ba_ba": o1["delta_ba"],
            "delta_fpr_ba": o1["delta_fpr"], "delta_fnr_ba": o1["delta_fnr"],
            # O2
            "tau_star_safe": o2["tau_star"], "ba_safe": o2["ba"],
            "fpr_safe": o2["fpr"], "fnr_safe": o2["fnr"],
            "delta_tau_safe": o2["delta_tau"], "delta_ba_safe": o2["delta_ba"],
            "delta_fpr_safe": o2["delta_fpr"], "delta_fnr_safe": o2["delta_fnr"],
            "n_feasible": o2["n_feasible"],
            "correctable_safe": corr,
            "safe_plateau_width": safe_plateau_w,
            "pooled_range": pooled_range,
        }
        d.update(taxonomy(d, pooled_range))
        rows.append(d)

        for t, b in zip(grid, ba_curve):
            m = C.detection_metrics(scores, labels, float(t))
            curve_rows.append({
                "category": cat, "condition": cond, "detector": det,
                "tau": float(t), "ba": m["ba"], "fpr": m["fpr"], "fnr": m["fnr"],
            })

    df = pd.DataFrame(rows)
    df_curves = pd.DataFrame(curve_rows)

    # ---- A1: source operating point table -------------------------------- #
    a1_cols = ["category", "condition", "detector", "tau_source", "ba_before",
               "fpr_before", "fnr_before", "auroc", "anomaly_prevalence",
               "n_normal", "n_anomaly"]
    df[a1_cols].to_csv(OUT / "A1_SOURCE_OPERATING_POINT.csv", index=False)

    # ---- A2: oracle threshold curves + headroom table --------------------- #
    df_curves.to_csv(OUT / "A2_ORACLE_THRESHOLD_CURVES.csv", index=False)
    df.to_csv(OUT / "A2_ORACLE_HEADROOM.csv", index=False)

    # ---- A3: detector-stratified headroom summary ------------------------- #
    summary_rows: List[Dict[str, Any]] = []
    groups = [("POOLED", df)] + [(det, df[df["detector"] == det]) for det in P.DETECTORS]
    for name, g in groups:
        n = len(g)
        n_safe = int(g["f_safe_correctable"].sum())
        summary_rows.append({
            "detector": name, "n_entities": n,
            "n_safe_correctable": n_safe,
            "frac_safe_correctable": n_safe / n if n else 0.0,
            "n_unsafe_only": int(g["f_unsafe_only"].sum()),
            "n_source_near_optimal": int(g["f_source_near_optimal"].sum()),
            "n_ambiguous_flat": int(g["f_ambiguous_flat"].sum()),
            "n_no_headroom": int(g["f_no_headroom"].sum()),
            "mean_delta_ba_safe": float(g["delta_ba_safe"].mean()),
            "median_delta_ba_safe": float(g["delta_ba_safe"].median()),
            "mean_delta_tau_safe": float(g["delta_tau_safe"].mean()),
            "median_delta_tau_safe": float(g["delta_tau_safe"].median()),
            "mean_delta_fpr_safe": float(g["delta_fpr_safe"].mean()),
            "mean_delta_fnr_safe": float(g["delta_fnr_safe"].mean()),
            "mean_delta_ba_ba": float(g["delta_ba_ba"].mean()),
            "mean_auroc": float(g["auroc"].mean()),
            "n_delta_ba_safe_ge_002": int((g["delta_ba_safe"] >= 0.02).sum()),
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT / "A3_DETECTOR_HEADROOM_SUMMARY.csv", index=False)

    # ---- A4: taxonomy table ---------------------------------------------- #
    tax_cols = ["category", "condition", "detector"] + [
        "f_safe_correctable", "f_unsafe_only", "f_source_near_optimal",
        "f_ambiguous_flat", "f_no_headroom", "primary_taxonomy"]
    df[tax_cols].to_csv(OUT / "A4_CORRECTABILITY_TAXONOMY.csv", index=False)

    # ---- A Gate ----------------------------------------------------------- #
    det_safe = {det: int(df[df["detector"] == det]["f_safe_correctable"].sum())
                for det in P.DETECTORS}
    det_n = {det: int((df["detector"] == det).sum()) for det in P.DETECTORS}
    h1_any_detector = any(
        n_safe >= GENERALIZED_HEADROOM_MIN_N or
        (n_safe / det_n[det] >= GENERALIZED_HEADROOM_MIN_FRAC)
        for det, n_safe in det_safe.items()
    )
    n_safe_pooled = int(df["f_safe_correctable"].sum())
    n_safe_headroom_pooled = int((df["delta_ba_safe"] >= 0.02).sum())
    generalized_headroom = (
        n_safe_headroom_pooled >= GENERALIZED_HEADROOM_MIN_N or
        (n_safe_headroom_pooled / len(df) >= GENERALIZED_HEADROOM_MIN_FRAC)
    )
    gate_pass = h1_any_detector and generalized_headroom
    gate = {
        "schema_version": "1.0",
        "stage": "PB7_P0_THRESHOLD_CORRECTABILITY_IDENTIFIABILITY_AUDIT",
        "phase": "A",
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": {
            "h1_any_detector_safe_correctable": h1_any_detector,
            "h1_detail": {det: {"n_safe_correctable": det_safe[det],
                                "frac": det_safe[det] / det_n[det] if det_n[det] else 0.0}
                          for det in P.DETECTORS},
            "generalized_threshold_headroom": generalized_headroom,
            "generalized_detail": {
                "n_delta_ba_safe_ge_002_pooled": n_safe_headroom_pooled,
                "frac_delta_ba_safe_ge_002_pooled": n_safe_headroom_pooled / len(df),
            },
            "n_entities": len(df),
        },
        "verdict": "PASS" if gate_pass else "FAIL",
        "terminal": "PROCEED_TO_B" if gate_pass else
                    "THRESHOLD_ROUTE_CLOSED_INSUFFICIENT_ORACLE_HEADROOM",
    }
    (OUT / "A_GATE.json").write_text(
        json.dumps(gate, indent=2, sort_keys=False), encoding="utf-8")

    # ---- ledger (PB7-P0 execution ledger) --------------------------------- #
    ledger_path = OUT / "P0_EXECUTION_LEDGER.csv"
    ledger = pd.read_csv(ledger_path) if ledger_path.exists() else pd.DataFrame(
        columns=["stage", "step", "status", "timestamp", "artifacts", "note"])
    ledger = pd.concat([ledger, pd.DataFrame([{
        "stage": "A",
        "step": "oracle_threshold_headroom_characterization",
        "status": "COMPLETE" if gate_pass else "FAIL",
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifacts": "A1_SOURCE_OPERATING_POINT.csv, A2_ORACLE_THRESHOLD_CURVES.csv, "
                     "A2_ORACLE_HEADROOM.csv, A3_DETECTOR_HEADROOM_SUMMARY.csv, "
                     "A4_CORRECTABILITY_TAXONOMY.csv, A_GATE.json",
        "note": f"A gate {'PASS -> PROCEED_TO_B' if gate_pass else 'FAIL -> terminal'}; "
                f"pooled safe-correctable={n_safe_pooled}/{len(df)}",
    }])], ignore_index=True)
    ledger.to_csv(ledger_path, index=False)

    print(json.dumps(gate, indent=2))
    print("\nDetector headroom summary:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
