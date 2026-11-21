"""PB7-P2 Stage C — K-Support Confirmatory Audit.

For the frozen principal policy (A_MEDIAN_RAW lambda=0.25 cutoff=Q50), evaluate
K4 / K8 / FULL on TEST with >= 100 independent support repeats per eligible
entity (200 used here, pre-registered seed list inherited from PB7-P0).

Per-repeat metrics (delta_BA, delta_FPR, delta_FNR, harm, coverage, selected
entity set) are reported pooled and DINOv2-only. Stability statistics:

  - median / 5/25/75/95 percentile of per-repeat pooled delta_BA
  - positive-delta_BA frequency
  - safety-pass frequency (delta_FNR <= +0.02)
  - harmful-repeat frequency
  - gate-selection Jaccard stability (mean pairwise Jaccard of selected sets)

Principal K8 gate (pre-registered):
  - utility stability: >= 90% repeats delta_BA > 0
  - safety stability: >= 90% repeats delta_FNR <= +0.02
  - strong stability: median K8 delta_BA >= 60% of FULL delta_BA
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
N_REPEATS = P2.C_REPEATS  # 200
DET = "dinov2_vits14"


def jaccard_stability(selected_sets: list[list[int]]) -> float:
    """Mean pairwise Jaccard of selected entity-id sets across repeats."""
    if len(selected_sets) < 2:
        return float("nan")
    total = 0.0
    count = 0
    for i in range(len(selected_sets)):
        si = set(selected_sets[i])
        for j in range(i + 1, len(selected_sets)):
            sj = set(selected_sets[j])
            union = si | sj
            if len(union) == 0:
                continue
            total += len(si & sj) / len(union)
            count += 1
    return float(total / count) if count else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    fw = P2.Firewall()
    ents = P2.entity_list()
    test_ents = [e for e in ents if P2.split_side(e[0]) == "TEST"]

    # entity ids for Jaccard
    ent_ids = {e: i for i, e in enumerate(test_ents)}

    # ---- per-entity per-repeat evaluation ------------------------------------ #
    # store per K: list over entities of per-repeat dict lists
    per_entity: dict[str, dict[tuple, list[dict[str, Any]]]] = {k: {} for k in ["4", "8"]}
    full_rows: list[dict[str, Any]] = []

    for cat, cond, det in test_ents:
        src = P2.load_src(cat, det, fw)
        cnd = P2.load_cnd(cat, cond, det, fw)
        anm = P2.load_anm(cat, det, fw)
        tau_source = P2.source_threshold(src)
        n = len(cnd)
        for k_value in ["4", "8"]:
            if n < int(k_value):
                continue
            reps = P2.evaluate_entity_repeats(
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
                n_repeats=N_REPEATS,
            )
            per_entity[k_value][(cat, cond, det)] = reps
        # FULL deterministic
        reps_full = P2.evaluate_entity_repeats(
            cat,
            cond,
            det,
            "A_MEDIAN_RAW",
            0.25,
            P2.PRINCIPAL_CUTOFF,
            "FULL",
            src,
            cnd,
            anm,
            tau_source,
            n_repeats=1,
        )
        r = reps_full[0]
        full_rows.append({"category": cat, "condition": cond, "detector": det, "K": "FULL", **r})

    df_full = pd.DataFrame(full_rows)

    # ---- per-repeat pooled / DINOv2 frames ----------------------------------- #
    def per_repeat_frame(k_value: str, det_filter: str) -> pd.DataFrame:
        """Frame with one row per repeat: pooled (or DINOv2) mean metrics."""
        recs: list[dict[str, Any]] = []
        ents_k = per_entity[k_value]
        n_reps = max((len(v) for v in ents_k.values()), default=0)
        for r in range(n_reps):
            dba, dfpr, dfnr, harm, cat, cov = [], [], [], [], [], []
            for (_, _, det_), reps in ents_k.items():
                if det_filter is not None and det_ != det_filter:
                    continue
                if r >= len(reps):
                    continue
                m = reps[r]
                dba.append(m["delta_ba"])
                dfpr.append(m["delta_fpr"])
                dfnr.append(m["delta_fnr"])
                harm.append(m["harm"])
                cat.append(m["catastrophic"])
                cov.append(m["coverage"])
            recs.append(
                {
                    "repeat": r,
                    "n_entities": len(dba),
                    "delta_ba": float(np.mean(dba)) if dba else float("nan"),
                    "delta_fpr": float(np.mean(dfpr)) if dfpr else float("nan"),
                    "delta_fnr": float(np.mean(dfnr)) if dfnr else float("nan"),
                    "harm_rate": float(np.mean(harm)) if harm else float("nan"),
                    "catastrophic_rate": float(np.mean(cat)) if cat else float("nan"),
                    "coverage": float(np.mean(cov)) if cov else float("nan"),
                }
            )
        return pd.DataFrame(recs)

    def stability_stats(df: pd.DataFrame) -> dict[str, float]:
        dba = df["delta_ba"].dropna().to_numpy()
        dfnr = df["delta_fnr"].dropna().to_numpy()
        harm = df["harm_rate"].dropna().to_numpy()
        return {
            "n_repeats": len(df),
            "median_delta_ba": float(np.median(dba)),
            "p5_delta_ba": float(np.percentile(dba, 5)),
            "p25_delta_ba": float(np.percentile(dba, 25)),
            "p75_delta_ba": float(np.percentile(dba, 75)),
            "p95_delta_ba": float(np.percentile(dba, 95)),
            "positive_dba_freq": float((dba > 0).mean()),
            "safety_pass_freq": float((dfnr <= P2.DFNR_MAX + 1e-12).mean()),
            "harmful_repeat_freq": float((harm > 0).mean()),
            "median_delta_fnr": float(np.median(dfnr)),
            "p95_delta_fnr": float(np.percentile(dfnr, 95)),
            "median_coverage": float(np.median(df["coverage"].dropna())),
        }

    # ---- gate-selection Jaccard stability ------------------------------------ #
    jac_rows: list[dict[str, Any]] = []
    for k_value in ["4", "8"]:
        for det_filter, label in [(None, "POOLED"), (DET, "DINOv2")]:
            ents_k = per_entity[k_value]
            selected_sets: list[list[int]] = []
            n_reps = max((len(v) for v in ents_k.values()), default=0)
            for r in range(n_reps):
                sel = [
                    ent_ids[(cat_, cond_, det_)]
                    for (cat_, cond_, det_), reps in ents_k.items()
                    if det_filter is None or det_ == det_filter
                    if r < len(reps) and reps[r]["selected"] == 1
                ]
                selected_sets.append(sel)
            jac_rows.append(
                {
                    "K": k_value,
                    "group": label,
                    "jaccard_stability": jaccard_stability(selected_sets),
                    "n_repeats": len(selected_sets),
                }
            )
    df_jac = pd.DataFrame(jac_rows)
    df_jac.to_csv(OUT / "C_GATE_SELECTION_JACCARD.csv", index=False)

    # ---- stability tables ----------------------------------------------------- #
    stab_rows: list[dict[str, Any]] = []
    for k_value in ["4", "8"]:
        for det_filter, label in [(None, "POOLED"), (DET, "DINOv2")]:
            fr = per_repeat_frame(k_value, det_filter)
            s = stability_stats(fr)
            stab_rows.append({"K": k_value, "group": label, **s})
    # FULL (deterministic)
    for det_filter, label in [(None, "POOLED"), (DET, "DINOv2")]:
        sub = df_full if det_filter is None else df_full[df_full.detector == det_filter]
        s = P2.summary_metrics(sub)
        stab_rows.append(
            {
                "K": "FULL",
                "group": label,
                "n_repeats": 1,
                "median_delta_ba": s["mean_delta_ba"],
                "p5_delta_ba": s["mean_delta_ba"],
                "p25_delta_ba": s["mean_delta_ba"],
                "p75_delta_ba": s["mean_delta_ba"],
                "p95_delta_ba": s["mean_delta_ba"],
                "positive_dba_freq": float((sub["delta_ba"] > 0).mean()),
                "safety_pass_freq": float((sub["delta_fnr"] <= P2.DFNR_MAX + 1e-12).mean()),
                "harmful_repeat_freq": float((sub["harm"] > 0).mean()),
                "median_delta_fnr": s["mean_delta_fnr"],
                "p95_delta_fnr": s["mean_delta_fnr"],
                "median_coverage": s["coverage"],
            }
        )
    df_stab = pd.DataFrame(stab_rows)
    df_stab.to_csv(OUT / "C_K_SUPPORT_STABILITY.csv", index=False)

    # ---- K8 repeat distribution (pooled + DINOv2) ---------------------------- #
    k8_pooled = per_repeat_frame("8", None)
    k8_pooled.to_csv(OUT / "C_K8_REPEAT_DISTRIBUTION.csv", index=False)
    k8_dino = per_repeat_frame("8", DET)
    k8_dino.to_csv(OUT / "C_K8_REPEAT_DISTRIBUTION_DINOV2.csv", index=False)

    # ---- principal K8 gate ---------------------------------------------------- #
    p8 = k8_pooled.dropna(subset=["delta_ba", "delta_fnr"])
    full_pooled_dba = float(df_full["delta_ba"].mean())
    util_stab = float((p8["delta_ba"] > 0).mean())
    safe_stab = float((p8["delta_fnr"] <= P2.DFNR_MAX + 1e-12).mean())
    med_k8 = float(np.median(p8["delta_ba"]))
    strong_stab = med_k8 >= 0.6 * full_pooled_dba

    checks = {
        "full_pooled_dba": full_pooled_dba,
        "k8_median_dba": med_k8,
        "k8_positive_dba_freq": util_stab,
        "k8_safety_pass_freq": safe_stab,
        "k8_strong_stability_median_ge_60pct_full": strong_stab,
        "k8_jaccard_pooled": float(
            df_jac[(df_jac.K == "8") & (df_jac.group == "POOLED")]["jaccard_stability"].iloc[0]
        ),
        "k8_jaccard_dinov2": float(
            df_jac[(df_jac.K == "8") & (df_jac.group == "DINOv2")]["jaccard_stability"].iloc[0]
        ),
    }
    ok = util_stab >= 0.90 and safe_stab >= 0.90 and strong_stab
    gate = {
        "schema_version": "1.0",
        "stage": "PB7_P2_PUBLICATION_READINESS_CONFIRMATORY_VALIDATION",
        "phase": "C",
        "timestamp": ts,
        "n_repeats": N_REPEATS,
        "checks": checks,
        "verdict": "PASS" if ok else "FAIL",
        "terminal": "PROCEED_TO_D"
        if ok
        else "PROCEED_TO_D_WITH_FEW_NORMAL_ROBUSTNESS_INSUFFICIENT",
    }
    (OUT / "C_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    # ---- ledger --------------------------------------------------------------- #
    ledger = pd.read_csv(OUT / "P0_EXECUTION_LEDGER.csv")
    ledger = pd.concat(
        [
            ledger,
            pd.DataFrame(
                [
                    {
                        "stage": "C",
                        "step": "k_support_confirmatory_audit",
                        "status": "COMPLETE",
                        "timestamp": ts,
                        "artifacts": "C_K_SUPPORT_STABILITY.csv, C_K8_REPEAT_DISTRIBUTION.csv, "
                        "C_K8_REPEAT_DISTRIBUTION_DINOV2.csv, C_GATE_SELECTION_JACCARD.csv, "
                        "C_GATE.json",
                        "note": (
                            f"C gate {'PASS' if ok else 'FAIL'}; K8 positive freq={util_stab:.3f}, "
                            f"safety freq={safe_stab:.3f}, median dBA={med_k8:.4f} vs FULL "
                            f"{full_pooled_dba:.4f}"
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ledger.to_csv(OUT / "P0_EXECUTION_LEDGER.csv", index=False)

    print(json.dumps(gate, indent=2))
    print("\nK-support stability:")
    print(df_stab.round(4).to_string(index=False))
    print("\nGate-selection Jaccard:")
    print(df_jac.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
