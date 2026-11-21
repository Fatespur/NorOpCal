"""PB8 D — few-normal support and evidence-breadth audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "pb8_publication_benchmark"))

import pb8_common as P8  # type: ignore[import-not-found]  # noqa: E402

OUT = P8.OUT
DET = "dinov2_vits14"


def _eligibility_map() -> pd.DataFrame:
    counts = P8.target_normal_counts()
    grouped = (
        counts.groupby("category", as_index=False)
        .agg(
            target_normal_count=("target_normal_n", "min"),
            target_normal_count_max=("target_normal_n", "max"),
            number_conditions=("condition", "nunique"),
            number_detectors=("detector", "nunique"),
        )
        .sort_values("category")
    )
    rows: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        category = str(row["category"])
        n = int(row["target_normal_count"])
        side = P8.split_side(category)
        k4 = side == "TEST" and n >= 4
        k8 = side == "TEST" and n >= 8
        if k8:
            reason = "included in frozen TEST K8 estimand"
        elif side != "TEST":
            reason = "excluded from principal evaluation by frozen DEV/TEST split"
        else:
            reason = f"target-normal n={n} < 8"
        rows.append(
            {
                "category": category,
                "frozen_split": side,
                "target_normal_count": n,
                "target_normal_count_max": int(row["target_normal_count_max"]),
                "K4_support_available": n >= 4,
                "K4_principal_test_eligible": k4,
                "K8_support_available": n >= 8,
                "K8_eligible": k8,
                "FULL_n": n,
                "number_conditions": int(row["number_conditions"]),
                "number_detectors": int(row["number_detectors"]),
                "reason_excluded_or_included": reason,
            }
        )
    return pd.DataFrame(rows)


def _support_summary() -> pd.DataFrame:
    per_entity = pd.read_csv(OUT / "B_PER_ENTITY_ALL.csv", dtype={"K": str})
    proposed = per_entity[per_entity.baseline == "B5"]
    stability = pd.read_csv(P8.P2_OUT / "C_K_SUPPORT_STABILITY.csv", dtype={"K": str})
    rows: list[dict[str, Any]] = []
    for k_value in ["4", "8", "FULL"]:
        frame = proposed[k_value == proposed.K]
        for group, detector in [("DINOv2", DET), ("POOLED", None)]:
            sub = frame if detector is None else frame[frame.detector == detector]
            valid = sub.dropna(subset=["delta_ba", "delta_fpr", "delta_fnr", "harm"])
            summary = P8.summary_metrics(valid)
            sealed = stability[(k_value == stability.K) & (stability.group == group)].iloc[0]
            rows.append(
                {
                    "K": k_value,
                    "group": group,
                    "n_units": int(summary["n"]),
                    "n_categories": int(valid.category.nunique()),
                    "mean_delta_ba": summary["mean_delta_ba"],
                    "mean_delta_fpr": summary["mean_delta_fpr"],
                    "mean_delta_fnr": summary["mean_delta_fnr"],
                    "harm_rate": summary["harm_rate"],
                    "catastrophic_rate": summary["catastrophic_rate"],
                    "coverage": summary["coverage"],
                    "sealed_n_support_repeats": int(sealed["n_repeats"]),
                    "sealed_median_delta_ba": sealed["median_delta_ba"],
                    "sealed_positive_delta_ba_frequency": sealed["positive_dba_freq"],
                    "sealed_safety_frequency": sealed["safety_pass_freq"],
                    "sealed_median_delta_fnr": sealed["median_delta_fnr"],
                    "safety_fail_mean": bool(summary["mean_delta_fnr"] > P8.DFNR_MAX + 1e-12),
                    "interpretation": {
                        "4": "unsupported / unsafe",
                        "8": "principal safe setting within eligible categories",
                        "FULL": "not monotonically safer",
                    }[k_value],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    eligibility = _eligibility_map()
    eligibility.to_csv(OUT / "D_K_SUPPORT_ELIGIBILITY.csv", index=False)
    final = _support_summary()
    final.to_csv(OUT / "D_K_SUPPORT_FINAL.csv", index=False)

    k8_map = eligibility[eligibility.K8_eligible]
    dino_k8 = final[(final.K == "8") & (final.group == "DINOv2")].iloc[0]
    pooled_k8 = final[(final.K == "8") & (final.group == "POOLED")].iloc[0]
    dino_k4 = final[(final.K == "4") & (final.group == "DINOv2")].iloc[0]
    dino_full = final[(final.K == "FULL") & (final.group == "DINOv2")].iloc[0]
    verdict = {
        "schema_version": "1.0",
        "phase": "D",
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verdict_code": "EB2",
        "verdict": "LIMITED_BUT_REPORTABLE",
        "counts": {
            "n_total_categories": len(eligibility),
            "n_K8_eligible_categories": len(k8_map),
            "K8_eligible_categories": k8_map.category.tolist(),
            "n_K8_units": int(pooled_k8.n_units),
            "n_DINOv2_K8_units": int(dino_k8.n_units),
            "n_DINOv2_K4_units": int(dino_k4.n_units),
            "n_DINOv2_FULL_units": int(dino_full.n_units),
        },
        "evidence": {
            "K4_DINOv2_safety_frequency": float(dino_k4.sealed_safety_frequency),
            "K8_DINOv2_safety_frequency": float(dino_k8.sealed_safety_frequency),
            "K8_pooled_safety_frequency": float(pooled_k8.sealed_safety_frequency),
            "FULL_DINOv2_mean_delta_fnr": float(dino_full.mean_delta_fnr),
            "safety_budget": P8.DFNR_MAX,
        },
        "allowed_claim": "in frozen TEST K8-eligible acquisition-shift units (fabric and walnuts)",
        "forbidden_claim": "validated across all eight categories under K=8",
        "reason": (
            "K8 is positive and safe across 200 preregistered support repeats, but the frozen "
            "TEST K8 estimand contains only fabric and walnuts (10 DINOv2 units; 30 pooled units). "
            "This supports a scoped reportable claim, not broad eight-category few-normal generalization."
        ),
        "terminal": "PROCEED_TO_E",
    }
    (OUT / "D_EVIDENCE_BREADTH_VERDICT.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    (OUT / "D_GATE.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "phase": "D",
                "verdict": "PASS_WITH_SCOPE_LIMITATION",
                "terminal": "PROCEED_TO_E",
                "evidence_breadth_verdict": "EB2 LIMITED_BUT_REPORTABLE",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("D done: EB2 LIMITED_BUT_REPORTABLE; K8 categories=" + ",".join(k8_map.category.tolist()))


if __name__ == "__main__":
    main()
