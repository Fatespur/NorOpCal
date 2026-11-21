"""PB7-P1 Stage D — DEV Selection (safety-first constrained).

Evaluates the pre-registered candidate grid on the DEV split at the principal
K=8 setting and selects a single global operating setting (family, lambda,
cutoff) under a safety-first rule:

  1. filter candidates with mean delta_FNR_DEV <= +0.02
  2. among safe candidates maximize mean delta_BA_DEV
  3. tie-break: lower harm rate, smaller lambda, simpler actuator, higher coverage

Candidate grid (frozen at P0):
  - 48 gated candidates (3 family x 4 lambda x 4 cutoff)
  - 12 Always-On counterparts (3 family x 4 lambda, no gate)
  - 1 NoAdapt baseline (T0)

DEV outcomes are used only to select the global setting; TEST is untouched.
If no candidate passes the DEV safety filter, terminal D_NO_SAFE_DEV_CANDIDATE.
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
sys.path.insert(0, str(ROOT / "tools" / "pb7_p1_safety_actuator"))

import md0sr_common as C  # type: ignore[import-not-found]  # noqa: E402
import pb7_p1_common as P  # type: ignore[import-not-found]  # noqa: E402

OUT = P.OUT
PB7P0 = P.PB7P0
SAFE_DFNR = P.DFNR_MAX  # +0.02

# Simplicity rank for tie-break (lower = simpler)
FAMILY_SIMPLICITY = {"A_MEDIAN_RAW": 0, "A_TAIL_RAW": 1, "A_T2_RAW": 2}


def build_candidates() -> List[Dict[str, Any]]:
    cands: List[Dict[str, Any]] = []
    for fam in P.ACTUATOR_FAMILIES:
        for lam in P.LAMBDA_GRID:
            for ck in P.CUTOFF_KEYS:
                cands.append({"family": fam, "lambda": lam, "cutoff_key": ck,
                              "gate": True, "kind": "GATED"})
            cands.append({"family": fam, "lambda": lam, "cutoff_key": None,
                          "gate": False, "kind": "ALWAYS_ON"})
    cands.append({"family": None, "lambda": None, "cutoff_key": None,
                  "gate": False, "kind": "NO_ADAPT"})
    return cands


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    fw = P.Firewall()
    ents = P.entity_list()
    dev_ents = [e for e in ents if P.split_side(e[0]) == "DEV"]

    # ---- cutoff values from Stage C (DEV FULL-support median_shift) ------- #
    cdef = json.loads((OUT / "C_GATE_DEFINITION.json").read_text(encoding="utf-8"))
    cutoffs: Dict[str, float] = {k: float(v) for k, v in cdef["cutoff_values"].items()}

    # ---- precompute per-entity data (K=8 principal) ------------------------ #
    ent_data: List[Dict[str, Any]] = []
    for cat, cond, det in dev_ents:
        src = P.load_src(cat, det, fw)
        cnd_full = P.load_cnd(cat, cond, det, fw)
        anm = P.load_anm(cat, det, fw)
        tau_source = P.source_threshold(src)
        n = len(cnd_full)
        if n < 8:
            continue  # K8-ineligible; excluded from DEV K8 selection
        labels = np.concatenate([np.zeros(len(cnd_full)), np.ones(len(anm))])
        scores = np.concatenate([cnd_full, anm])
        before = C.detection_metrics(scores, labels, tau_source)
        idx_sets = [P.support_indices(cat, cond, det, n, "8", r) for r in range(P.K_REPEATS)]
        ent_data.append({
            "cat": cat, "cond": cond, "det": det, "src": src, "cnd_full": cnd_full,
            "anm": anm, "tau_source": tau_source, "labels": labels, "scores": scores,
            "before": before, "idx_sets": idx_sets,
        })

    # ---- evaluate a candidate on DEV K8 ------------------------------------ #
    def eval_candidate(family: str | None, lam: float | None, cutoff_key: str | None,
                       gate: bool) -> Dict[str, Any]:
        c = cutoffs[cutoff_key] if (gate and cutoff_key) else None
        rows: List[Dict[str, Any]] = []
        for ed in ent_data:
            acc = {"delta_ba": [], "delta_fpr": [], "delta_fnr": [],
                   "benefit": [], "harm": [], "catastrophic": [], "coverage": []}
            for idx in ed["idx_sets"]:
                cnd_k = ed["cnd_full"][idx]
                if gate:
                    score = P.correctability_score(ed["src"], cnd_k, ed["tau_source"])
                    if not P.gate_adapt(score, c):
                        tau = ed["tau_source"]
                    else:
                        tau = P.bounded_tau(family, ed["src"], cnd_k, ed["tau_source"], lam)
                elif family is not None:
                    tau = P.bounded_tau(family, ed["src"], cnd_k, ed["tau_source"], lam)
                else:
                    tau = ed["tau_source"]
                m = C.detection_metrics(ed["scores"], ed["labels"], tau)
                acc["delta_ba"].append(m["ba"] - ed["before"]["ba"])
                acc["delta_fpr"].append(m["fpr"] - ed["before"]["fpr"])
                acc["delta_fnr"].append(m["fnr"] - ed["before"]["fnr"])
                acc["benefit"].append(int(C.benefit_label(ed["before"], m)))
                acc["harm"].append(int((m["ba"] - ed["before"]["ba"]) < P.HARM_DBA
                                       or (m["fnr"] - ed["before"]["fnr"]) > P.HARM_DFNR))
                acc["catastrophic"].append(int((m["fnr"] - ed["before"]["fnr"]) > P.CATASTROPHIC_DFNR))
                acc["coverage"].append(1.0 if abs(tau - ed["tau_source"]) > 1e-12 else 0.0)
            rows.append({
                "category": ed["cat"], "condition": ed["cond"], "detector": ed["det"],
                "delta_ba": float(np.mean(acc["delta_ba"])),
                "delta_fpr": float(np.mean(acc["delta_fpr"])),
                "delta_fnr": float(np.mean(acc["delta_fnr"])),
                "benefit": float(np.mean(acc["benefit"])),
                "harm": float(np.mean(acc["harm"])),
                "catastrophic": float(np.mean(acc["catastrophic"])),
                "coverage": float(np.mean(acc["coverage"])),
            })
        df = pd.DataFrame(rows)
        s = P.summary_metrics(df)
        return {"family": family, "lambda": lam, "cutoff_key": cutoff_key,
                "gate": gate, "kind": ("GATED" if gate else
                                       ("ALWAYS_ON" if family else "NO_ADAPT")),
                **s, "n_entities": int(len(df))}

    # ---- evaluate all candidates ------------------------------------------- #
    cands = build_candidates()
    results: List[Dict[str, Any]] = []
    for cd in cands:
        results.append(eval_candidate(cd["family"], cd["lambda"], cd["cutoff_key"], cd["gate"]))
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT / "D_DEV_BENCHMARK.csv", index=False)

    # ---- D2: safety-first selection ---------------------------------------- #
    # Selection is restricted to the 48 GATED deployable candidates (the
    # "48 deployable candidates" of the frozen grid). NoAdapt and Always-On
    # counterparts are mandatory baselines (evaluated and reported) but are not
    # eligible for selection: the stage goal is a conservative, bounded,
    # abstaining actuator, and E-stage compares B3 (bounded Always-On) vs B4
    # (bounded + abstaining) as baselines of the selected gated setting.
    df_res["safe"] = df_res["mean_delta_fnr"] <= SAFE_DFNR + 1e-12
    gated = df_res[df_res["kind"] == "GATED"].copy()
    safe = gated[gated["safe"]].copy()
    if len(safe) == 0:
        selected = None
        no_safe = True
    else:
        no_safe = False
        # maximize mean_delta_ba among safe; tie-break: lower harm, smaller lambda,
        # simpler actuator, higher coverage
        safe["tie_lam"] = safe["lambda"].fillna(0.0)
        safe["tie_sim"] = safe["family"].map(lambda f: FAMILY_SIMPLICITY.get(f, 99))
        safe = safe.sort_values(
            by=["mean_delta_ba", "harm_rate", "tie_lam", "tie_sim", "coverage"],
            ascending=[False, True, True, True, False])
        selected = safe.iloc[0].to_dict()

    # ---- D3: gate ----------------------------------------------------------- #
    checks = {
        "candidate_grid_frozen": len(cands) == 61,
        "n_gated_48": int((df_res["kind"] == "GATED").sum()) == 48,
        "n_always_on_12": int((df_res["kind"] == "ALWAYS_ON").sum()) == 12,
        "n_no_adapt_1": int((df_res["kind"] == "NO_ADAPT").sum()) == 1,
        "no_new_family": True,
    }
    if no_safe:
        gate_pass = all(checks.values())  # grid checks still pass; selection yields no safe
        terminal = "D_NO_SAFE_DEV_CANDIDATE"
    else:
        gate_pass = all(checks.values()) and selected["mean_delta_fnr"] <= SAFE_DFNR + 1e-12
        terminal = "PROCEED_TO_E" if gate_pass else "D_NO_SAFE_DEV_CANDIDATE"

    gate = {
        "schema_version": "1.0",
        "stage": "PB7_P1_SAFETY_CONSTRAINED_ACTUATOR_DEVELOPMENT",
        "phase": "D",
        "timestamp": ts,
        "checks": checks,
        "no_safe_dev_candidate": no_safe,
        "selected": selected,
        "n_safe_candidates": int(len(safe)),
        "verdict": "PASS" if gate_pass else "FAIL",
        "terminal": terminal,
    }
    (OUT / "D_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    # ---- safety filter table (gated candidates only; Always-On/NoAdapt baselines) - #
    safe_rows = df_res[df_res["safe"] & (df_res["kind"] == "GATED")].sort_values(
        "mean_delta_ba", ascending=False)
    safe_rows.to_csv(OUT / "D_DEV_SAFETY_FILTER.csv", index=False)

    # ---- ledger ------------------------------------------------------------- #
    ledger_path = OUT / "P0_EXECUTION_LEDGER.csv"
    ledger = pd.read_csv(ledger_path)
    sel_desc = "NONE" if selected is None else (
        f"{selected['family']} lam={selected['lambda']} c={selected['cutoff_key']} "
        f"dBA={selected['mean_delta_ba']:.4f} dFNR={selected['mean_delta_fnr']:.4f}")
    ledger = pd.concat([ledger, pd.DataFrame([{
        "stage": "D",
        "step": "dev_selection_safety_first",
        "status": "COMPLETE" if gate_pass else "FAIL",
        "timestamp": ts,
        "artifacts": "D_DEV_BENCHMARK.csv, D_DEV_SAFETY_FILTER.csv, D_GATE.json",
        "note": (f"D gate {'PASS' if gate_pass else 'FAIL'}; terminal={terminal}; "
                 f"n_safe={int(len(safe))}; selected={sel_desc}"),
    }])], ignore_index=True)
    ledger.to_csv(ledger_path, index=False)

    print(json.dumps(gate, indent=2))
    print("\nTop-10 safe DEV candidates (by mean_delta_ba):")
    if len(safe_rows):
        print(safe_rows[["family", "lambda", "cutoff_key", "mean_delta_ba",
                         "mean_delta_fnr", "harm_rate", "coverage"]].head(10).round(4).to_string(index=False))
    else:
        print("  (none)")
    print("\nBest Always-On (for reference):")
    ao = df_res[df_res["kind"] == "ALWAYS_ON"].sort_values("mean_delta_ba", ascending=False)
    print(ao[["family", "lambda", "mean_delta_ba", "mean_delta_fnr", "harm_rate"]].head(5).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
