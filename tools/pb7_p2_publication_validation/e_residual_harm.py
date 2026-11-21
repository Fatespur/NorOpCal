"""PB7-P2 Stage E — Residual Harm Audit.

Adjudicates whether the frozen principal policy's residual harm (PB7-P1:
9/78, concentrated in walnuts high-source-FNR entities) exposes a structural
failure that overturns the deployment claim (R2 SYSTEMIC_SAFETY_FAILURE) or is
a localized residual risk (R1 LOCALIZED_RESIDUAL_RISK).

Frozen policy (never modified): A_MEDIAN_RAW, lambda=0.25, cutoff=Q50, K=8.
This is a CONFIRMATORY audit: no method change, no outcome-driven repair.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p1_safety_actuator"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p2_publication_validation"))

import pb7_p2_common as P2  # type: ignore[import-not-found]  # noqa: E402

OUT = P2.OUT
DET = "dinov2_vits14"
K = P2.PRINCIPAL_K
N_REPEATS = P2.K_REPEATS  # 30 (PB7-P1 E protocol)


def eval_entity(cat: str, cond: str, det: str) -> dict[str, Any]:
    """Evaluate the frozen gated policy at K=8 on one TEST entity (mean over repeats)."""
    fw = P2.Firewall()
    src = P2.load_src(cat, det, fw)
    cnd_full = P2.load_cnd(cat, cond, det, fw)
    anm = P2.load_anm(cat, det, fw)
    tau_source = P2.source_threshold(src)

    labels = np.concatenate([np.zeros(len(cnd_full)), np.ones(len(anm))])
    scores = np.concatenate([cnd_full, anm])
    before = P2.C_detection_metrics(scores, labels, tau_source)

    # FULL-support correctability signal (deployable)
    med_shift_full = P2.correctability_score(src, cnd_full, tau_source)

    # per-repeat evaluation (K=8)
    reps = P2.evaluate_entity_repeats(
        cat,
        cond,
        det,
        P2.PRINCIPAL_FAMILY,
        P2.PRINCIPAL_LAMBDA,
        P2.PRINCIPAL_CUTOFF,
        K,
        src,
        cnd_full,
        anm,
        tau_source,
        n_repeats=N_REPEATS,
    )

    dba = float(np.mean([r["delta_ba"] for r in reps]))
    dfpr = float(np.mean([r["delta_fpr"] for r in reps]))
    dfnr = float(np.mean([r["delta_fnr"] for r in reps]))
    harm = float(np.mean([r["harm"] for r in reps]))
    catastrophic = float(np.mean([r["catastrophic"] for r in reps]))
    coverage = float(np.mean([r["coverage"] for r in reps]))
    benefit = float(np.mean([r["benefit"] for r in reps]))

    # correction magnitude (mean tau_used - tau_source over repeats)
    taus = [r["tau"] for r in reps]
    correction = float(np.mean([t - tau_source for t in taus]))
    adapted_fnr = float(before["fnr"] + dfnr)

    # support variance: SE of the K-sampled median shift over repeats
    k8_shifts: list[float] = []
    n = len(cnd_full)
    k = int(K)
    for r in range(N_REPEATS):
        if k > n:
            break
        idx = P2.support_indices(cat, cond, det, n, K, r)
        k8_shifts.append(P2.correctability_score(src, cnd_full[idx], tau_source))
    k8_se = float(np.std(k8_shifts) / np.sqrt(max(len(k8_shifts), 1)))

    # source threshold percentiles (diagnostic)
    src_sorted = np.sort(src)
    pct_src = float(np.searchsorted(src_sorted, tau_source) / len(src_sorted) * 100.0)
    cnd_sorted = np.sort(cnd_full)
    pct_cnd = float(np.searchsorted(cnd_sorted, tau_source) / len(cnd_sorted) * 100.0)

    return {
        "category": cat,
        "condition": cond,
        "detector": det,
        "source_ba": float(before["ba"]),
        "source_fnr": float(before["fnr"]),
        "source_fpr": float(before["fpr"]),
        "median_shift_full": med_shift_full,
        "k8_mean_shift": float(np.mean(k8_shifts)) if k8_shifts else float("nan"),
        "k8_se_shift": k8_se,
        "correction_magnitude": correction,
        "adapted_fnr": adapted_fnr,
        "delta_ba": dba,
        "delta_fpr": dfpr,
        "delta_fnr": dfnr,
        "benefit": benefit,
        "harm": harm,
        "catastrophic": catastrophic,
        "coverage": coverage,
        "source_tau_pct_source": pct_src,
        "source_tau_pct_target": pct_cnd,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    ents = P2.entity_list()
    rows: list[dict[str, Any]] = []
    for cat, cond, det in ents:
        if P2.split_side(cat) != "TEST":
            continue
        rows.append(eval_entity(cat, cond, det))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "E_RESIDUAL_HARM_ALL_ENTITIES.csv", index=False)

    # ---- E1: harmful entity characterization --------------------------------- #
    harmful = df[df.harm > 0].copy()
    harmful["harm_type"] = np.where(
        harmful["delta_fnr"] > P2.CATASTROPHIC_DFNR,
        "catastrophic_fnr",
        np.where(harmful["delta_fnr"] > P2.HARM_DFNR, "fnr_breach", "dba_loss"),
    )
    harmful.to_csv(OUT / "E_RESIDUAL_HARM.csv", index=False)

    # ---- E2: harmful vs non-harmful (diagnostic only) ------------------------ #
    diag_cols = [
        "source_fnr",
        "source_fpr",
        "median_shift_full",
        "k8_se_shift",
        "source_tau_pct_target",
        "source_ba",
    ]
    comp_rows: list[dict[str, Any]] = []
    for grp, sub in [("HARMFUL", harmful), ("NON_HARMFUL", df[df.harm <= 0])]:
        rec = {"group": grp, "n": len(sub)}
        for c in diag_cols:
            rec[f"{c}_mean"] = float(sub[c].mean())
            rec[f"{c}_median"] = float(sub[c].median())
        comp_rows.append(rec)
    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(OUT / "E_HARMFUL_VS_NONHARMFUL.csv", index=False)

    # ---- E3: catastrophic concentration -------------------------------------- #
    def conc(df_: pd.DataFrame, dim: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for val, sub in df_.groupby(dim):
            out.append(
                {
                    "dim": dim,
                    "value": val,
                    "n_entities": len(sub),
                    "n_harm": int((sub.harm > 0).sum()),
                    "harm_rate": float((sub.harm > 0).mean()),
                    "mean_dba": float(sub.delta_ba.mean()),
                    "mean_dfnr": float(sub.delta_fnr.mean()),
                    "n_catastrophic": int((sub.catastrophic > 0).sum()),
                }
            )
        return out

    conc_rows = []
    for dim in ["category", "condition", "detector"]:
        conc_rows.extend(conc(df, dim))
    # source-FNR bin (high = source_fnr > 0.5, mid = 0.2..0.5, low = < 0.2)
    df["source_fnr_bin"] = pd.cut(
        df["source_fnr"], bins=[-np.inf, 0.2, 0.5, np.inf], labels=["low", "mid", "high"]
    )
    conc_rows.extend(conc(df, "source_fnr_bin"))
    df_conc = pd.DataFrame(conc_rows)
    df_conc.to_csv(OUT / "E_CATASTROPHIC_CONCENTRATION.csv", index=False)

    # ---- E4: R1/R2 adjudication ---------------------------------------------- #
    n_total = len(df)
    n_harm = int((df.harm > 0).sum())
    n_catastrophic = int((df.catastrophic > 0).sum())
    harm_rate = n_harm / n_total if n_total else float("nan")

    # concentration checks
    cat_harm = df[df.harm > 0].groupby("category").size()
    top_cat_share = float(cat_harm.max() / n_harm) if n_harm else 0.0
    single_cat_concentration = (len(cat_harm) == 1) or (top_cat_share >= 0.9)
    high_sfnr_share = float((df[df.harm > 0].source_fnr > 0.5).mean()) if n_harm else 0.0
    high_sfnr_concentration = high_sfnr_share >= 0.9

    # detector-wide failure: a detector with harm spread across >= 2 categories
    det_harm = df[df.harm > 0].groupby("detector")["category"].nunique()
    detector_wide_failure = bool((det_harm >= 2).any())

    # condition systemic failure: a condition with harm across >= 2 categories
    # (harm within a single category is a localized category effect, not a
    # condition-wide systemic failure)
    cond_harm = df[df.harm > 0].groupby("condition")["category"].nunique()
    condition_systemic = bool((cond_harm >= 2).any())

    # DINOv2 (principal) harm check
    dino_harm = df[(df.detector == DET) & (df.harm > 0)]
    dino_harm_rate = len(dino_harm) / max(len(df[df.detector == DET]), 1)

    systemic = (
        detector_wide_failure
        or condition_systemic
        or (harm_rate > 0.30)
        or (dino_harm_rate > 0.20 and len(dino_harm) >= 3)
    )
    verdict = "R2_SYSTEMIC_SAFETY_FAILURE" if systemic else "R1_LOCALIZED_RESIDUAL_RISK"

    adjudication = {
        "schema_version": "1.0",
        "stage": "PB7_P2_PUBLICATION_READINESS_CONFIRMATORY_VALIDATION",
        "phase": "E",
        "timestamp": ts,
        "n_entities": n_total,
        "n_harm": n_harm,
        "harm_rate": harm_rate,
        "n_catastrophic": n_catastrophic,
        "concentration": {
            "harm_categories": sorted(cat_harm.index.astype(str).tolist()),
            "top_category_share": top_cat_share,
            "single_category_concentration": single_cat_concentration,
            "high_source_fnr_share": high_sfnr_share,
            "high_source_fnr_concentration": high_sfnr_concentration,
            "detector_wide_failure": detector_wide_failure,
            "condition_systemic_failure": condition_systemic,
            "dinov2_harm_rate": dino_harm_rate,
            "dinov2_n_harm": len(dino_harm),
        },
        "verdict": verdict,
        "claim_consequence": (
            "LOCALIZED_RESIDUAL_RISK: the method may be retained and the "
            "limitation written into the paper."
            if verdict == "R1_LOCALIZED_RESIDUAL_RISK"
            else "SYSTEMIC_SAFETY_FAILURE: the paper main method cannot be promoted directly."
        ),
    }
    (OUT / "E_GATE.json").write_text(json.dumps(adjudication, indent=2), encoding="utf-8")

    # ---- ledger --------------------------------------------------------------- #
    ledger = pd.read_csv(OUT / "P0_EXECUTION_LEDGER.csv")
    ledger = pd.concat(
        [
            ledger,
            pd.DataFrame(
                [
                    {
                        "stage": "E",
                        "step": "residual_harm_audit",
                        "status": "COMPLETE",
                        "timestamp": ts,
                        "artifacts": "E_RESIDUAL_HARM.csv, E_HARMFUL_VS_NONHARMFUL.csv, "
                        "E_CATASTROPHIC_CONCENTRATION.csv, E_GATE.json",
                        "note": (
                            f"E verdict {verdict}; n_harm={n_harm}/{n_total}, "
                            f"harm_rate={harm_rate:.3f}, "
                            f"top-category share={top_cat_share:.2f}, "
                            f"high-source-FNR share={high_sfnr_share:.2f}"
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ledger.to_csv(OUT / "P0_EXECUTION_LEDGER.csv", index=False)

    print(json.dumps(adjudication, indent=2))
    print("\nHarmful entities:")
    print(
        harmful[
            [
                "category",
                "condition",
                "detector",
                "source_ba",
                "source_fnr",
                "median_shift_full",
                "correction_magnitude",
                "adapted_fnr",
                "delta_ba",
                "delta_fnr",
                "harm_type",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\nHarmful vs non-harmful (diagnostic):")
    print(df_comp.round(4).to_string(index=False))
    print("\nConcentration:")
    print(df_conc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
