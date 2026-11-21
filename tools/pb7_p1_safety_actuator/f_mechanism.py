"""PB7-P1 Stage F — Mechanism Analysis (runs regardless of E success/failure).

F1 — Safety-utility curve: for the frozen principal actuator family
      (A_MEDIAN_RAW) on TEST at K=8, sweep lambda in {0.25,0.50,0.75,1.00}
      for both Always-On and Gated (frozen Q50 cutoff) variants. Report
      delta_BA vs delta_FNR, harm vs coverage, delta_FPR vs delta_FNR.

F2 — Residual harm taxonomy: characterize the residual harmful entities of the
      principal B4 (K=8) policy by detector, category, condition, median_shift
      magnitude, K8 median-shift uncertainty, and source working point.

F3 — Detector decomposition: classify each detector as
      (useful+safe+selectable) / no-headroom / weak-identifiability /
      actuator-mismatch, using TEST evidence only (no outcome-driven tuning).

All analysis is diagnostic; the frozen method is not modified here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p1_safety_actuator"))

import md0sr_common as C  # type: ignore[import-not-found]  # noqa: E402
import pb7_p1_common as P  # type: ignore[import-not-found]  # noqa: E402

OUT = P.OUT
PB7P0 = P.PB7P0
PRINCIPAL_FAMILY = "A_MEDIAN_RAW"
CUTOFF_KEY = "Q50"
SAFE_DFNR = P.DFNR_MAX  # +0.02


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    cdef = json.loads((OUT / "C_GATE_DEFINITION.json").read_text(encoding="utf-8"))
    cutoff_value = float(cdef["cutoff_values"][CUTOFF_KEY])

    fw = P.Firewall()
    ents = P.entity_list()
    test_ents = [e for e in ents if P.split_side(e[0]) == "TEST"]
    head = pd.read_csv(PB7P0 / "A2_ORACLE_HEADROOM.csv")

    # ---- precompute per-entity deployable + diagnostic context --------------- #
    ctx: List[Dict[str, Any]] = []
    for cat, cond, det in test_ents:
        src = P.load_src(cat, det, fw)
        cnd_full = P.load_cnd(cat, cond, det, fw)
        anm = P.load_anm(cat, det, fw)
        tau_source = P.source_threshold(src)
        n = len(cnd_full)
        h = head[(head.category == cat) & (head.condition == cond) & (head.detector == det)].iloc[0]
        # K8 median-shift uncertainty (bootstrap SE over K8 support repeats)
        if n >= 8:
            shifts = []
            for r in range(P.K_REPEATS):
                idx = P.support_indices(cat, cond, det, n, "8", r)
                shifts.append(float(np.median(cnd_full[idx]) - np.median(src)))
            k8_se = float(np.std(shifts))
            k8_mean_shift = float(np.mean(shifts))
        else:
            k8_se = float("nan")
            k8_mean_shift = float("nan")
        # source working point (diagnostic)
        labels = np.concatenate([np.zeros(len(cnd_full)), np.ones(len(anm))])
        scores = np.concatenate([cnd_full, anm])
        before = C.detection_metrics(scores, labels, tau_source)
        ctx.append({
            "category": cat, "condition": cond, "detector": det,
            "n_full": n, "k8_eligible": int(n >= 8),
            "median_shift_full": float(np.median(cnd_full) - np.median(src)),
            "k8_mean_shift": k8_mean_shift, "k8_se_shift": k8_se,
            "source_ba": before["ba"], "source_fnr": before["fnr"], "source_fpr": before["fpr"],
            "correctable_safe": int(h["correctable_safe"]),
            "delta_tau_safe": float(h["delta_tau_safe"]),
        })
    df_ctx = pd.DataFrame(ctx)

    # ---- F1: safety-utility curve (TEST K=8, A_MEDIAN_RAW) ------------------- #
    curve_rows: List[Dict[str, Any]] = []
    for lam in P.LAMBDA_GRID:
        for gate in [False, True]:
            rows = []
            for cat, cond, det in test_ents:
                src = P.load_src(cat, det, fw)
                cnd_full = P.load_cnd(cat, cond, det, fw)
                anm = P.load_anm(cat, det, fw)
                tau_source = P.source_threshold(src)
                n = len(cnd_full)
                if n < 8:
                    continue
                r = P.evaluate_entity(cat, cond, det, PRINCIPAL_FAMILY, lam,
                                      cutoff_value if gate else 0.0, "8",
                                      src, cnd_full, anm, tau_source)
                rows.append({"category": cat, "condition": cond, "detector": det,
                             "delta_ba": r["delta_ba"], "delta_fpr": r["delta_fpr"],
                             "delta_fnr": r["delta_fnr"], "benefit": r["benefit"],
                             "harm": r["harm"], "catastrophic": r["catastrophic"],
                             "coverage": r["coverage"]})
            df = pd.DataFrame(rows)
            s = P.summary_metrics(df)
            curve_rows.append({"family": PRINCIPAL_FAMILY, "lambda": lam,
                               "gate": gate, "kind": "GATED" if gate else "ALWAYS_ON",
                               **s, "n_entities": int(len(df))})
    df_curve = pd.DataFrame(curve_rows)
    df_curve.to_csv(OUT / "F_SAFETY_UTILITY_CURVE.csv", index=False)

    # ---- F2: residual harm taxonomy (principal B4 K=8) ----------------------- #
    per = pd.read_csv(OUT / "E_TEST_PER_ENTITY.csv")
    b4 = per[(per.baseline == "B4") & (per.K == "8")].copy()
    b4 = b4.merge(df_ctx, on=["category", "condition", "detector"], how="left")
    harmful = b4[b4["harm"] > 0].copy()
    # harm magnitude / type
    harmful["harm_type"] = np.where(harmful["delta_fnr"] > P.CATASTROPHIC_DFNR, "catastrophic_fnr",
                                    np.where(harmful["delta_fnr"] > P.HARM_DFNR, "fnr_breach",
                                             np.where(harmful["delta_ba"] < P.HARM_DBA, "ba_drop", "other")))
    harmful["median_shift_bin"] = pd.cut(harmful["median_shift_full"],
                                         bins=[-np.inf, -0.02, 0.0, 0.02, np.inf],
                                         labels=["large_neg", "small_neg", "small_pos", "large_pos"])
    harmful["k8_uncert_bin"] = pd.cut(harmful["k8_se_shift"],
                                      bins=[-np.inf, 0.005, 0.010, np.inf],
                                      labels=["low", "mid", "high"])
    harmful["source_fnr_bin"] = pd.cut(harmful["source_fnr"],
                                       bins=[-np.inf, 0.05, 0.15, np.inf],
                                       labels=["low", "mid", "high"])
    harmful.to_csv(OUT / "F_RESIDUAL_HARM_ENTITIES.csv", index=False)

    tax_rows: List[Dict[str, Any]] = []
    for col in ["detector", "category", "condition", "harm_type",
                "median_shift_bin", "k8_uncert_bin", "source_fnr_bin"]:
        tab = harmful.groupby(col).agg(
            n_harm=("harm", "size"),
            mean_dba=("delta_ba", "mean"),
            mean_dfnr=("delta_fnr", "mean"),
            n_catastrophic=("catastrophic", "sum"),
        ).reset_index()
        tab["dim"] = col
        tab = tab.rename(columns={col: "value"})
        tax_rows.append(tab)
    df_tax = pd.concat(tax_rows, ignore_index=True)
    df_tax.to_csv(OUT / "F_RESIDUAL_HARM_TAXONOMY.csv", index=False)

    # ---- F3: detector decomposition ------------------------------------------- #
    det_rows: List[Dict[str, Any]] = []
    for det in P.DETECTORS:
        # per-detector summary for gated lambda=0.25 K=8
        rows = []
        for cat, cond, d in test_ents:
            if d != det:
                continue
            src = P.load_src(cat, d, fw)
            cnd_full = P.load_cnd(cat, cond, d, fw)
            anm = P.load_anm(cat, d, fw)
            tau_source = P.source_threshold(src)
            n = len(cnd_full)
            if n < 8:
                continue
            r = P.evaluate_entity(cat, cond, d, PRINCIPAL_FAMILY, 0.25, cutoff_value,
                                  "8", src, cnd_full, anm, tau_source)
            rows.append({"category": cat, "condition": cond, "detector": d,
                         "delta_ba": r["delta_ba"], "delta_fpr": r["delta_fpr"],
                         "delta_fnr": r["delta_fnr"], "benefit": r["benefit"],
                         "harm": r["harm"], "catastrophic": r["catastrophic"],
                         "coverage": r["coverage"]})
        dfd = pd.DataFrame(rows)
        s = P.summary_metrics(dfd)
        n_correctable = int(df_ctx[(df_ctx.detector == det) & (df_ctx.correctable_safe == 1)].shape[0])
        n_k8 = int(df_ctx[(df_ctx.detector == det) & (df_ctx.k8_eligible == 1)].shape[0])
        # decomposition verdict (diagnostic, TEST evidence)
        if s["mean_delta_ba"] >= P.DBA_MIN and s["mean_delta_fnr"] <= SAFE_DFNR + 1e-12 \
                and s["coverage"] >= 0.10:
            verdict = "USEFUL_SAFE_SELECTABLE"
        elif n_correctable == 0:
            verdict = "NO_HEADROOM"
        elif s["mean_delta_ba"] < 0:
            verdict = "ACTUATOR_MISMATCH"
        else:
            verdict = "WEAK_IDENTIFIABILITY"
        det_rows.append({
            "detector": det, "n_k8": n_k8, "n_correctable_safe": n_correctable,
            "mean_delta_ba": s["mean_delta_ba"], "mean_delta_fpr": s["mean_delta_fpr"],
            "mean_delta_fnr": s["mean_delta_fnr"], "harm_rate": s["harm_rate"],
            "coverage": s["coverage"], "verdict": verdict,
        })
    df_det = pd.DataFrame(det_rows)
    df_det.to_csv(OUT / "F_DETECTOR_DECOMPOSITION.csv", index=False)

    # ---- F gate / mechanism summary ------------------------------------------- #
    # monotonic safety trend check: mean_delta_fnr should be non-increasing as
    # lambda decreases (0.25 <= 0.50 <= 0.75 <= 1.00) for the gated variant.
    gated_curve = df_curve[df_curve.kind == "GATED"].sort_values("lambda")
    dfnr_by_lam = gated_curve.set_index("lambda")["mean_delta_fnr"].to_dict()
    monotonic_safety = all(dfnr_by_lam[lam] <= dfnr_by_lam[lam2] + 1e-12
                           for lam, lam2 in [(0.25, 0.50), (0.50, 0.75), (0.75, 1.00)])
    dba_by_lam = gated_curve.set_index("lambda")["mean_delta_ba"].to_dict()
    monotonic_utility = all(dba_by_lam[lam] >= dba_by_lam[lam2] - 1e-12
                            for lam, lam2 in [(0.25, 0.50), (0.50, 0.75), (0.75, 1.00)])

    mech = {
        "schema_version": "1.0",
        "stage": "PB7_P1_SAFETY_CONSTRAINED_ACTUATOR_DEVELOPMENT",
        "phase": "F",
        "timestamp": ts,
        "principal_family": PRINCIPAL_FAMILY,
        "cutoff_key": CUTOFF_KEY,
        "f1_safety_utility_curve": {
            "monotonic_safety_with_conservatism": monotonic_safety,
            "monotonic_utility_with_aggressiveness": monotonic_utility,
            "gated_dfnr_by_lambda": dfnr_by_lam,
            "gated_dba_by_lambda": dba_by_lam,
        },
        "f2_residual_harm": {
            "n_harmful_entities": int(len(harmful)),
            "n_total_entities": int(len(b4)),
            "harm_rate": float(b4["harm"].mean()),
            "n_catastrophic": int((b4["catastrophic"] > 0).sum()),
            "harm_type_counts": harmful["harm_type"].value_counts().to_dict(),
        },
        "f3_detector_decomposition": det_rows,
        "verdict": "COMPLETE",
    }
    (OUT / "F_MECHANISM.json").write_text(json.dumps(mech, indent=2), encoding="utf-8")

    # ---- ledger --------------------------------------------------------------- #
    ledger_path = OUT / "P0_EXECUTION_LEDGER.csv"
    ledger = pd.read_csv(ledger_path)
    ledger = pd.concat([ledger, pd.DataFrame([{
        "stage": "F",
        "step": "mechanism_analysis",
        "status": "COMPLETE",
        "timestamp": ts,
        "artifacts": "F_SAFETY_UTILITY_CURVE.csv, F_RESIDUAL_HARM_ENTITIES.csv, "
                     "F_RESIDUAL_HARM_TAXONOMY.csv, F_DETECTOR_DECOMPOSITION.csv, F_MECHANISM.json",
        "note": (f"F complete; monotonic_safety={monotonic_safety}; "
                 f"n_harmful={len(harmful)}; det verdicts="
                 + ";".join(f"{r['detector']}:{r['verdict']}" for r in det_rows)),
    }])], ignore_index=True)
    ledger.to_csv(ledger_path, index=False)

    print(json.dumps(mech, indent=2))
    print("\nSafety-utility curve (TEST K=8, A_MEDIAN_RAW):")
    print(df_curve[["lambda", "kind", "mean_delta_ba", "mean_delta_fpr", "mean_delta_fnr",
                    "harm_rate", "coverage"]].round(4).to_string(index=False))
    print("\nDetector decomposition:")
    print(df_det.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
