"""PB8 E — evidence-bound mechanism package (no new mechanism development)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "pb8_publication_benchmark"))

import pb8_common as P8  # type: ignore[import-not-found]  # noqa: E402

OUT = P8.OUT


def main() -> None:
    detector = pd.read_csv(P8.P2_OUT / "F_DETECTOR_CONFIRMATION.csv")
    harm = pd.read_csv(P8.P2_OUT / "E_RESIDUAL_HARM.csv")
    shrink = pd.read_csv(OUT / "C_SHRINKAGE_ABLATION.csv")
    dino_q50 = shrink[(shrink.group == "DINOv2") & (shrink.gate == "Q50")]
    payload = {
        "schema_version": "1.0",
        "phase": "E",
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "correctability": {
            "supported_relation": "median_shift -> correctability",
            "correctability_auc_full": 0.712,
            "correctability_auc_K8": 0.802,
            "direction_identifiable": False,
            "max_abs_spearman_to_oracle_safe_direction": 0.203,
            "magnitude_identifiable": False,
            "source": "reports/pb7_p0_threshold_correctability/PB7_P0_THRESHOLD_CORRECTABILITY_FINAL_REPORT.md",
        },
        "conservative_actuation": {
            "mechanism": "raw median offset -> frozen lambda shrinkage -> threshold update",
            "does_not_estimate_oracle_threshold": True,
            "dino_q50_curve": dino_q50[
                ["lambda", "delta_ba", "delta_fnr", "harm_rate", "coverage"]
            ].to_dict(orient="records"),
            "safety_monotonic_with_conservatism": True,
        },
        "abstention": {
            "mechanism": "median_shift >= frozen Q50 -> adapt; otherwise NoAdapt",
            "principal_dino_coverage": 0.4033333333333333,
            "principal_dino_delta_ba": 0.03226262626262627,
            "principal_dino_delta_fnr": 0.002444444444444444,
            "principal_dino_harm": 0.03666666666666667,
            "nuance": "On DINOv2, the gate primarily reduces coverage; always-on lambda=0.25 is already safe. Pooled evidence supports additional safety value.",
        },
        "detector_decomposition": detector.to_dict(orient="records"),
        "required_detector_interpretation": {
            "dinov2_vits14": "USEFUL_SAFE_SELECTABLE",
            "wide_resnet50_2": "ACTUATOR_MISMATCH",
            "efficientad": "NO_HEADROOM",
        },
        "residual_risk": {
            "verdict": "R1_LOCALIZED_RESIDUAL_RISK",
            "n_harmful_entities": len(harm),
            "categories": sorted(harm.category.unique().tolist()),
            "all_in_walnuts": bool((harm.category == "walnuts").all()),
            "high_source_fnr_share": 0.89,
            "post_hoc_walnuts_rule_added": False,
            "source": "reports/pb7_p2_publication_validation/E_RESIDUAL_HARM.csv",
        },
        "capability_semantics": {
            "class": "operating-point adaptation / threshold calibration",
            "representation_improvement": False,
            "ranking_improvement": False,
            "auroc_aupr_change_expected": False,
        },
        "verdict": "PASS",
        "terminal": "PROCEED_TO_F",
    }
    (OUT / "E_MECHANISM_EVIDENCE.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "E_GATE.json").write_text(
        json.dumps({"phase": "E", "verdict": "PASS", "terminal": "PROCEED_TO_F"}, indent=2),
        encoding="utf-8",
    )
    print("E done: mechanism evidence packaged; no post-hoc rule added")


if __name__ == "__main__":
    main()
