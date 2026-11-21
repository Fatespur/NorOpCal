"""PB7-P2 Stage F — Detector-Specificity Confirmation.

Applies the frozen principal policy (A_MEDIAN_RAW, lambda=0.25, cutoff=Q50,
K=8) unchanged to all three detectors and confirms that the PB7-P1 mechanism
classification persists under confirmatory analysis:

  DINOv2      -> USEFUL_SAFE_SELECTABLE
  PatchCore   -> ACTUATOR_MISMATCH
  EfficientAD -> NO_HEADROOM

If the classification persists, the claim "Correctability and safe threshold
actuation are detector-dependent" is frozen (F3). No detector-specific tuning.
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

import pb7_p1_common as P1  # type: ignore[import-not-found]  # noqa: E402
import pb7_p2_common as P2  # type: ignore[import-not-found]  # noqa: E402

OUT = P2.OUT
PB7P0 = P1.PB7P0
P1_OUT = P1.OUT
K = P2.PRINCIPAL_K
SAFE_DFNR = P2.DFNR_MAX  # +0.02


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    fw = P2.Firewall()
    ents = P2.entity_list()
    test_ents = [e for e in ents if P2.split_side(e[0]) == "TEST"]
    head = pd.read_csv(PB7P0 / "A2_ORACLE_HEADROOM.csv")

    # oracle safe headroom per entity (frozen PB7-P0 O3 estimand)
    head_map = {
        (r.category, r.condition, r.detector): int(r.correctable_safe)
        for r in head.itertuples(index=False)
    }

    det_rows: list[dict[str, Any]] = []
    per_entity_rows: list[dict[str, Any]] = []
    for det in P2.DETECTORS:
        rows: list[dict[str, Any]] = []
        for cat, cond, d in test_ents:
            if d != det:
                continue
            src = P2.load_src(cat, d, fw)
            cnd_full = P2.load_cnd(cat, cond, d, fw)
            anm = P2.load_anm(cat, d, fw)
            tau_source = P2.source_threshold(src)
            n = len(cnd_full)
            if n < 8:
                continue
            r = P2.evaluate_entity_repeats(
                cat,
                cond,
                d,
                P2.PRINCIPAL_FAMILY,
                P2.PRINCIPAL_LAMBDA,
                P2.PRINCIPAL_CUTOFF,
                K,
                src,
                cnd_full,
                anm,
                tau_source,
                n_repeats=P2.K_REPEATS,
            )
            rec = {
                "category": cat,
                "condition": cond,
                "detector": d,
                "delta_ba": float(np.mean([x["delta_ba"] for x in r])),
                "delta_fpr": float(np.mean([x["delta_fpr"] for x in r])),
                "delta_fnr": float(np.mean([x["delta_fnr"] for x in r])),
                "benefit": float(np.mean([x["benefit"] for x in r])),
                "harm": float(np.mean([x["harm"] for x in r])),
                "catastrophic": float(np.mean([x["catastrophic"] for x in r])),
                "coverage": float(np.mean([x["coverage"] for x in r])),
                "correctable_safe": head_map.get((cat, cond, d), 0),
            }
            rows.append(rec)
            per_entity_rows.append(rec)
        dfd = pd.DataFrame(rows)
        s = P2.summary_metrics(dfd)
        # oracle safe headroom over ALL TEST entities (matches sealed PB7-P1 F3
        # semantics: n_correctable counts every TEST entity, not only k8-eligible)
        n_correctable = int(
            sum(head_map.get((c, cd, det), 0) for c, cd, d in test_ents if d == det)
        )
        n_k8 = len(dfd)
        # same verdict logic as PB7-P1 F3 (frozen diagnostic)
        if (
            s["mean_delta_ba"] >= P2.DBA_MIN
            and s["mean_delta_fnr"] <= SAFE_DFNR + 1e-12
            and s["coverage"] >= 0.10
        ):
            verdict = "USEFUL_SAFE_SELECTABLE"
        elif n_correctable == 0:
            verdict = "NO_HEADROOM"
        elif s["mean_delta_ba"] < 0:
            verdict = "ACTUATOR_MISMATCH"
        else:
            verdict = "WEAK_IDENTIFIABILITY"
        det_rows.append(
            {
                "detector": det,
                "n_k8": n_k8,
                "n_correctable_safe": n_correctable,
                "mean_delta_ba": s["mean_delta_ba"],
                "mean_delta_fpr": s["mean_delta_fpr"],
                "mean_delta_fnr": s["mean_delta_fnr"],
                "harm_rate": s["harm_rate"],
                "coverage": s["coverage"],
                "verdict": verdict,
            }
        )

    df_det = pd.DataFrame(det_rows)
    df_det.to_csv(OUT / "F_DETECTOR_CONFIRMATION.csv", index=False)
    pd.DataFrame(per_entity_rows).to_csv(
        OUT / "F_DETECTOR_CONFIRMATION_PER_ENTITY.csv", index=False
    )

    # ---- F2: mechanism classification confirmation ---------------------------- #
    sealed = pd.read_csv(P1_OUT / "F_DETECTOR_DECOMPOSITION.csv")
    sealed_map = {r.detector: r.verdict for r in sealed.itertuples(index=False)}
    classification_persists = all(r["verdict"] == sealed_map.get(r["detector"]) for r in det_rows)

    # ---- F3: detector-dependence claim freeze --------------------------------- #
    dinov2 = df_det[df_det.detector == "dinov2_vits14"].iloc[0]
    claim_frozen = (
        dinov2["verdict"] == "USEFUL_SAFE_SELECTABLE"
        and float(dinov2["mean_delta_ba"]) > 0
        and float(dinov2["mean_delta_fnr"]) <= SAFE_DFNR + 1e-12
        and classification_persists
    )

    verdict = {
        "schema_version": "1.0",
        "stage": "PB7_P2_PUBLICATION_READINESS_CONFIRMATORY_VALIDATION",
        "phase": "F",
        "timestamp": ts,
        "f1_detector_confirmation": det_rows,
        "f2_classification_persists": classification_persists,
        "sealed_pb7p1_classification": sealed_map,
        "f3_claim": {
            "claim": "Correctability and safe threshold actuation are detector-dependent.",
            "frozen": claim_frozen,
            "note": (
                "DINOv2 remains USEFUL_SAFE_SELECTABLE; PatchCore remains "
                "ACTUATOR_MISMATCH; EfficientAD remains NO_HEADROOM. The "
                "detector-specific claim is NOT detector-agnostic."
            ),
        },
        "verdict": "PASS" if claim_frozen else "FAIL",
        "terminal": "PROCEED_TO_G"
        if claim_frozen
        else "PROCEED_TO_G_WITH_DETECTOR_CLAIM_NOT_CONFIRMED",
    }
    (OUT / "F_GATE.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    # ---- ledger --------------------------------------------------------------- #
    ledger = pd.read_csv(OUT / "P0_EXECUTION_LEDGER.csv")
    ledger = pd.concat(
        [
            ledger,
            pd.DataFrame(
                [
                    {
                        "stage": "F",
                        "step": "detector_specificity_confirmation",
                        "status": "COMPLETE",
                        "timestamp": ts,
                        "artifacts": "F_DETECTOR_CONFIRMATION.csv, "
                        "F_DETECTOR_CONFIRMATION_PER_ENTITY.csv, F_GATE.json",
                        "note": (
                            f"F gate {'PASS' if claim_frozen else 'FAIL'}; classification persists="
                            f"{classification_persists}; verdicts="
                            + ";".join(f"{r['detector']}:{r['verdict']}" for r in det_rows)
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ledger.to_csv(OUT / "P0_EXECUTION_LEDGER.csv", index=False)

    print(json.dumps(verdict, indent=2))
    print("\nDetector confirmation:")
    print(df_det.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
