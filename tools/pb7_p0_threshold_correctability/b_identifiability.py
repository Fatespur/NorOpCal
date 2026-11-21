"""PB7-P0 Stage B — Normal-Only Deformation vs Oracle Correctability.

This stage asks whether oracle threshold correctability can be identified from
normal-only evidence. No complex method is constructed here.

Frozen feature set (B1): the BSEL-registered deformation features
(median_shift, mean_shift, mad_ratio, std_ratio, d_q90, d_q95, d_q99,
upper-tail spread q99_q95_change, wasserstein, ks) PLUS the three pre-registered
threshold-specific analytic features T1/T2/T3. No outcome-driven additions.

Targets (B2, ORACLE_DIAGNOSTIC_ONLY): sign(delta_tau*_SAFE), |delta_tau*_SAFE|,
CORRECTABLE_SAFE, oracle delta_BA, oracle delta_FPR.

B3 computes, for EVERY feature (no cherry-picking), univariate identifiability:
Spearman rho with delta_tau*, sign AUC, CORRECTABLE_SAFE AUC, natural sign-rule
accuracy, bootstrap CIs, detector-stratified results, and leave-one-category-out.

B4 computes the trivial baselines B0-B4.

B Gate: PASS-A (sign identifiability) OR PASS-B (correctability identifiability).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p0_threshold_correctability"))

import md0sr_common as C  # type: ignore[import-not-found]  # noqa: E402
import pb7_p0_common as P  # type: ignore[import-not-found]  # noqa: E402

OUT = P.OUT

# --------------------------------------------------------------------------- #
# Frozen B1 feature set (pre-registered; no outcome-driven additions)
# --------------------------------------------------------------------------- #
# BSEL-registered deformation features (from md0sr_common.compute_raw_features)
BSEL_FEATURES: List[str] = [
    "median_shift", "mean_shift", "mad_ratio", "std_ratio",
    "d_q90", "d_q95", "d_q99", "q99_q95_change",
    "wasserstein", "ks",
]
# Threshold-specific analytic features (pre-registered at B, before outcomes)
THRESHOLD_FEATURES: List[str] = ["T1", "T2", "T3"]
ALL_FEATURES: List[str] = BSEL_FEATURES + THRESHOLD_FEATURES

# Natural sign-rule directions (pre-registered, physically motivated):
#   shift features: positive shift -> more target-normal above tau_source ->
#                   raise threshold to cut FPR  => sign(delta_tau) = sign(feature)
#   ratio features: ratio > 1 -> wider target spread => sign = sign(ratio - 1)
#   T1: F_target_normal(tau_source); nominal = Q95 = 0.95
#       T1 > 0.95 -> more mass below -> fewer exceed -> lower threshold
#   T2: target q95 - source q95 (identical to d_q95) => sign = sign(feature)
#   T3: P_target_normal(score > tau_source); nominal = 1 - Q95 = 0.05
#       T3 > 0.05 -> more exceed -> raise threshold
NOMINAL_Q95 = P.Q95
NOMINAL_EXCEED = 1.0 - P.Q95
NATURAL_SIGN: Dict[str, str] = {
    "median_shift": "shift", "mean_shift": "shift",
    "d_q90": "shift", "d_q95": "shift", "d_q99": "shift",
    "q99_q95_change": "shift",
    "mad_ratio": "ratio", "std_ratio": "ratio",
    "T1": "t1", "T2": "shift", "T3": "t3",
}
# wasserstein / ks are non-negative magnitude features: no natural sign rule.

# --------------------------------------------------------------------------- #
# Frozen B-gate criteria (pre-registered)
# --------------------------------------------------------------------------- #
SIGN_RHO_MIN = 0.30          # PASS-A: |pooled Spearman rho| >= 0.30
SIGN_AUC_MIN = 0.60          # PASS-A: pooled sign AUC >= 0.60
SIGN_RHO_CI_EXCLUDES_ZERO = True   # bootstrap 95% CI of rho excludes 0
N_DETECTORS_DIRECTION = 2    # PASS-A: >= 2 detectors same-sign rho as pooled
LCO_RHO_FRAC = 0.5           # PASS-A: min leave-category-out |rho| >= 0.5*|rho_pooled|
CORR_AUC_MIN = 0.60          # PASS-B: pooled CORRECTABLE_SAFE AUC >= 0.60
CORR_RANKBIS_MIN = 0.30      # PASS-B: |rank-biserial| >= 0.30
CORR_LCO_AUC_MIN = 0.55      # PASS-B: mean leave-category-out AUC >= 0.55
BOOT_B = 2000                # bootstrap resamples (category-aware block)
BOOT_SEED = 20260820


def natural_sign(feature: str, value: float) -> int:
    kind = NATURAL_SIGN.get(feature)
    if kind == "shift":
        return 1 if value > 0 else (-1 if value < 0 else 0)
    if kind == "ratio":
        return 1 if value > 1 else (-1 if value < 1 else 0)
    if kind == "t1":
        return -1 if value > NOMINAL_Q95 else (1 if value < NOMINAL_Q95 else 0)
    if kind == "t3":
        return 1 if value > NOMINAL_EXCEED else (-1 if value < NOMINAL_EXCEED else 0)
    return 0  # no natural rule (wasserstein / ks)


def compute_features(src: np.ndarray, cnd: np.ndarray, tau_source: float) -> Dict[str, float]:
    """Frozen feature computation: BSEL deformation features + T1/T2/T3."""
    raw = C.compute_raw_features(src, cnd)
    f: Dict[str, float] = {}
    for name in BSEL_FEATURES:
        f[name] = float(raw[name])
    # T1: source-threshold percentile under target-normal
    f["T1"] = float(np.mean(cnd <= tau_source))
    # T2: target-normal q95 - source-normal q95 (== d_q95, registered explicitly)
    f["T2"] = float(np.percentile(cnd, 95) - np.percentile(src, 95))
    # T3: target-normal exceedance rate at tau_source
    f["T3"] = float(np.mean(cnd > tau_source))
    return f


def sign_auc(feat: np.ndarray, sign_labels: np.ndarray) -> float:
    """AUC of feature vs binary sign label (+1 vs -1), direction-agnostic."""
    if len(np.unique(sign_labels)) < 2:
        return float("nan")
    return float(C.roc_auc_score(sign_labels, feat))


def rank_biserial(feat: np.ndarray, labels: np.ndarray) -> float:
    """Rank-biserial correlation of a feature with a binary label."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    r = stats.rankdata(feat)
    r0 = r[labels == 0]
    r1 = r[labels == 1]
    n0, n1 = len(r0), len(r1)
    if n0 == 0 or n1 == 0:
        return float("nan")
    return float((2.0 * (r1.mean() - r0.mean())) / (n0 + n1))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def block_bootstrap_ci(metric_fn, cats: np.ndarray, values: Dict[str, np.ndarray],
                       b: int = BOOT_B, seed: int = BOOT_SEED) -> Tuple[float, float, float]:
    """Category-aware block bootstrap of a metric (mean, 2.5%, 97.5%)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(cats)
    ests = np.empty(b)
    for i in range(b):
        sample_cats = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.isin(cats, sample_cats)
        sub = {k: v[idx] for k, v in values.items()}
        ests[i] = metric_fn(sub)
    ests = ests[~np.isnan(ests)]
    if len(ests) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(ests)), float(np.percentile(ests, 2.5)), float(np.percentile(ests, 97.5))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fw = P.Firewall()
    ents = P.entity_list()
    head = pd.read_csv(OUT / "A2_ORACLE_HEADROOM.csv")

    rows: List[Dict[str, Any]] = []
    for cat, cond, det in ents:
        src = P.load_src(cat, det, fw)
        cnd = P.load_cnd(cat, cond, det, fw)   # FULL target-normal support (Stage B)
        tau_src = P.source_threshold(src)
        feats = compute_features(src, cnd, tau_src)
        h = head[(head.category == cat) & (head.condition == cond) & (head.detector == det)].iloc[0]
        row = {"category": cat, "condition": cond, "detector": det,
               "tau_source": tau_src, "delta_tau_safe": h["delta_tau_safe"],
               "abs_delta_tau_safe": abs(h["delta_tau_safe"]),
               "correctable_safe": int(h["correctable_safe"]),
               "oracle_delta_ba": h["delta_ba_safe"],
               "oracle_delta_fpr": h["delta_fpr_safe"],
               "sign_delta_tau": int(np.sign(h["delta_tau_safe"]))}
        row.update(feats)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "B_FEATURES.csv", index=False)

    cats = df["category"].to_numpy()
    dets = df["detector"].to_numpy()
    dt = df["delta_tau_safe"].to_numpy()
    sign = df["sign_delta_tau"].to_numpy()
    corr = df["correctable_safe"].to_numpy()

    # ---- B3: univariate identifiability (ALL features, no cherry-picking) - #
    ident_rows: List[Dict[str, Any]] = []
    for feat in ALL_FEATURES:
        x = df[feat].to_numpy()
        # pooled Spearman with delta_tau*
        rho = spearman(x, dt)
        # sign discrimination (entities with delta_tau != 0)
        nz = sign != 0
        auc_sign = sign_auc(x[nz], sign[nz]) if nz.sum() >= 2 else float("nan")
        # correctability discrimination
        auc_corr = sign_auc(x, corr) if len(np.unique(corr)) >= 2 else float("nan")
        rbis = rank_biserial(x, corr)
        # natural sign-rule accuracy (only features with a natural rule)
        nat_acc = float("nan")
        if feat in NATURAL_SIGN:
            pred = np.array([natural_sign(feat, v) for v in x])
            nz2 = (sign != 0) & (pred != 0)
            nat_acc = float(np.mean(pred[nz2] == sign[nz2])) if nz2.sum() > 0 else float("nan")
        # detector-stratified rho
        det_rho: Dict[str, float] = {}
        for d in P.DETECTORS:
            m = dets == d
            det_rho[d] = spearman(x[m], dt[m])
        # leave-one-category-out rho
        lco_rho: Dict[str, float] = {}
        for c in np.unique(cats):
            m = cats != c
            lco_rho[str(c)] = spearman(x[m], dt[m])
        # leave-one-category-out correctability AUC
        lco_auc: Dict[str, float] = {}
        for c in np.unique(cats):
            m = cats != c
            lco_auc[str(c)] = sign_auc(x[m], corr[m]) if len(np.unique(corr[m])) >= 2 else float("nan")
        # bootstrap CIs (category-aware block)
        vals = {"x": x, "dt": dt, "sign": sign, "corr": corr, "cats": cats}
        rho_ci = block_bootstrap_ci(
            lambda s: spearman(s["x"], s["dt"]), cats, {"x": x, "dt": dt})
        auc_ci = block_bootstrap_ci(
            lambda s: sign_auc(s["x"][s["sign"] != 0], s["sign"][s["sign"] != 0]),
            cats, {"x": x, "sign": sign})
        corr_auc_ci = block_bootstrap_ci(
            lambda s: sign_auc(s["x"], s["corr"]), cats, {"x": x, "corr": corr})
        ident_rows.append({
            "feature": feat,
            "spearman_rho_delta_tau": rho,
            "rho_ci_lo": rho_ci[1], "rho_ci_hi": rho_ci[2],
            "sign_auc": auc_sign,
            "sign_auc_ci_lo": auc_ci[1], "sign_auc_ci_hi": auc_ci[2],
            "natural_sign_accuracy": nat_acc,
            "correctable_auc": auc_corr,
            "correctable_auc_ci_lo": corr_auc_ci[1], "correctable_auc_ci_hi": corr_auc_ci[2],
            "correctable_rank_biserial": rbis,
            "n_nonzero_sign": int(nz.sum()),
            **{f"rho_{d}": det_rho[d] for d in P.DETECTORS},
            **{f"lco_rho_{c}": lco_rho[str(c)] for c in np.unique(cats)},
            **{f"lco_corr_auc_{c}": lco_auc[str(c)] for c in np.unique(cats)},
        })
    ident = pd.DataFrame(ident_rows)
    ident.to_csv(OUT / "B_UNIVARIATE_IDENTIFIABILITY.csv", index=False)

    # ---- B4: trivial baselines -------------------------------------------- #
    n = len(df)
    frac_zero = float(np.mean(sign == 0))
    frac_pos = float(np.mean(sign == 1))
    frac_neg = float(np.mean(sign == -1))
    nz = sign != 0
    frac_pos_nz = float(np.mean(sign[nz] == 1)) if nz.sum() else float("nan")
    frac_neg_nz = float(np.mean(sign[nz] == -1)) if nz.sum() else float("nan")
    # B0 always no-adapt: predicts sign=0 for all -> accuracy = frac_zero
    # B1 always raise: predicts sign=+1 -> accuracy = frac_pos
    # B2 always lower: predicts sign=-1 -> accuracy = frac_neg
    # B3 detector-only prior: per-detector majority sign (among non-zero)
    b3_acc = 0.0
    for d in P.DETECTORS:
        m = dets == d
        s = sign[m]
        nzm = s != 0
        if nzm.sum() == 0:
            continue
        maj = 1 if np.mean(s[nzm] == 1) >= np.mean(s[nzm] == -1) else -1
        b3_acc += float(np.mean(s[nzm] == maj)) * nzm.sum()
    b3_acc = b3_acc / max(int(nz.sum()), 1)
    # B4 single normal-only scalar: T3 natural sign rule accuracy
    t3 = df["T3"].to_numpy()
    pred_t3 = np.array([natural_sign("T3", v) for v in t3])
    nz4 = (sign != 0) & (pred_t3 != 0)
    b4_acc = float(np.mean(pred_t3[nz4] == sign[nz4])) if nz4.sum() > 0 else float("nan")
    baselines = pd.DataFrame([{
        "baseline": "B0_ALWAYS_NO_ADAPT", "description": "predict no change for all",
        "sign_accuracy": frac_zero, "n": n,
    }, {
        "baseline": "B1_ALWAYS_RAISE", "description": "predict raise for all",
        "sign_accuracy": frac_pos, "n": n,
    }, {
        "baseline": "B2_ALWAYS_LOWER", "description": "predict lower for all",
        "sign_accuracy": frac_neg, "n": n,
    }, {
        "baseline": "B3_DETECTOR_ONLY_PRIOR", "description": "per-detector majority sign",
        "sign_accuracy": b3_acc, "n": int(nz.sum()),
    }, {
        "baseline": "B4_T3_EXCEEDANCE_RULE", "description": "T3 natural sign rule",
        "sign_accuracy": b4_acc, "n": int(nz4.sum()),
    }])
    baselines.to_csv(OUT / "B_TRIVIAL_BASELINES.csv", index=False)

    # ---- B Gate ------------------------------------------------------------ #
    # PASS-A: at least one feature with stable sign identifiability
    pass_a_rows = []
    for _, r in ident.iterrows():
        if np.isnan(r["spearman_rho_delta_tau"]) or np.isnan(r["sign_auc"]):
            continue
        rho = r["spearman_rho_delta_tau"]
        rho_ci_excl = not (r["rho_ci_lo"] <= 0 <= r["rho_ci_hi"])
        auc_ok = r["sign_auc"] >= SIGN_AUC_MIN
        rho_ok = abs(rho) >= SIGN_RHO_MIN
        # direction consistency across >= 2 detectors
        det_rhos = [r[f"rho_{d}"] for d in P.DETECTORS]
        same_sign = sum(1 for v in det_rhos if not np.isnan(v) and np.sign(v) == np.sign(rho))
        # leave-category-out not single-category-driven
        lco_vals = [r[f"lco_rho_{c}"] for c in np.unique(cats)]
        lco_ok = all(not np.isnan(v) and np.sign(v) == np.sign(rho)
                     and abs(v) >= LCO_RHO_FRAC * abs(rho) for v in lco_vals)
        ok = rho_ok and rho_ci_excl and auc_ok and same_sign >= N_DETECTORS_DIRECTION and lco_ok
        pass_a_rows.append({"feature": r["feature"], "pass_a": int(ok),
                            "rho": rho, "sign_auc": r["sign_auc"],
                            "rho_ci_excludes_zero": int(rho_ci_excl),
                            "n_detectors_same_sign": same_sign, "lco_ok": int(lco_ok)})
    pass_a_df = pd.DataFrame(pass_a_rows)
    pass_a = bool(pass_a_df["pass_a"].max()) if len(pass_a_df) else False
    best_pass_a = pass_a_df[pass_a_df["pass_a"] == 1].to_dict("records") if len(pass_a_df) else []

    # PASS-B: at least one feature distinguishes CORRECTABLE_SAFE
    pass_b_rows = []
    for _, r in ident.iterrows():
        if np.isnan(r["correctable_auc"]) or np.isnan(r["correctable_rank_biserial"]):
            continue
        auc_ok = r["correctable_auc"] >= CORR_AUC_MIN
        rbis_ok = abs(r["correctable_rank_biserial"]) >= CORR_RANKBIS_MIN
        lco_aucs = [r[f"lco_corr_auc_{c}"] for c in np.unique(cats)]
        lco_ok = np.nanmean(lco_aucs) >= CORR_LCO_AUC_MIN
        ok = (auc_ok or rbis_ok) and lco_ok
        pass_b_rows.append({"feature": r["feature"], "pass_b": int(ok),
                            "correctable_auc": r["correctable_auc"],
                            "rank_biserial": r["correctable_rank_biserial"],
                            "mean_lco_auc": float(np.nanmean(lco_aucs))})
    pass_b_df = pd.DataFrame(pass_b_rows)
    pass_b = bool(pass_b_df["pass_b"].max()) if len(pass_b_df) else False
    best_pass_b = pass_b_df[pass_b_df["pass_b"] == 1].to_dict("records") if len(pass_b_df) else []

    gate_pass = pass_a or pass_b
    gate = {
        "schema_version": "1.0",
        "stage": "PB7_P0_THRESHOLD_CORRECTABILITY_IDENTIFIABILITY_AUDIT",
        "phase": "B",
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": {
            "pass_a_sign_identifiability": pass_a,
            "pass_a_best": best_pass_a,
            "pass_b_correctability_identifiability": pass_b,
            "pass_b_best": best_pass_b,
            "trivial_baselines": {
                "B0_always_no_adapt": frac_zero,
                "B1_always_raise": frac_pos,
                "B2_always_lower": frac_neg,
                "B3_detector_only_prior": b3_acc,
                "B4_T3_exceedance_rule": b4_acc,
                "best_trivial_sign_accuracy": max(frac_zero, frac_pos, frac_neg, b3_acc),
            },
            "n_entities": n,
            "n_nonzero_sign": int(nz.sum()),
        },
        "verdict": "PASS" if gate_pass else "FAIL",
        "terminal": "PROCEED_TO_C" if gate_pass else
                    "THRESHOLD_ROUTE_CLOSED_NORMAL_ONLY_SIGNAL_NOT_IDENTIFIABLE",
    }
    (OUT / "B_GATE.json").write_text(json.dumps(gate, indent=2, sort_keys=False), encoding="utf-8")

    # ---- ledger ------------------------------------------------------------ #
    ledger_path = OUT / "P0_EXECUTION_LEDGER.csv"
    ledger = pd.read_csv(ledger_path) if ledger_path.exists() else pd.DataFrame(
        columns=["stage", "step", "status", "timestamp", "artifacts", "note"])
    ledger = pd.concat([ledger, pd.DataFrame([{
        "stage": "B",
        "step": "normal_only_deformation_vs_oracle_correctability",
        "status": "COMPLETE" if gate_pass else "FAIL",
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifacts": "B_FEATURES.csv, B_UNIVARIATE_IDENTIFIABILITY.csv, "
                     "B_TRIVIAL_BASELINES.csv, B_GATE.json",
        "note": f"B gate {'PASS -> PROCEED_TO_C' if gate_pass else 'FAIL -> terminal'}; "
                f"PASS-A={pass_a}, PASS-B={pass_b}",
    }])], ignore_index=True)
    ledger.to_csv(ledger_path, index=False)

    print(json.dumps(gate, indent=2))
    print("\nUnivariate identifiability (all features):")
    cols = ["feature", "spearman_rho_delta_tau", "sign_auc", "natural_sign_accuracy",
            "correctable_auc", "correctable_rank_biserial"]
    print(ident[cols].round(3).to_string(index=False))
    print("\nTrivial baselines:")
    print(baselines.to_string(index=False))


if __name__ == "__main__":
    main()
