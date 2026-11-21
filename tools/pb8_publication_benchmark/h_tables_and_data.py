"""PB8 H — build frozen publication tables and figure source snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "pb8_publication_benchmark"
FIG = OUT / "figures"


def _write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False)


def _main_table(main: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "baseline",
        "baseline_name",
        "role",
        "n_units",
        "n_eligible_categories",
        "ba_before",
        "ba_after",
        "delta_ba",
        "dba_ci_lo",
        "dba_ci_hi",
        "delta_fpr",
        "delta_fnr",
        "harm_rate",
        "coverage",
        "safety_label",
    ]
    return main.loc[main["group"].eq("DINOv2"), columns].reset_index(drop=True)


def _detector_table(detectors: pd.DataFrame, mechanism: dict[str, object]) -> pd.DataFrame:
    principal = detectors.loc[detectors["baseline"].eq("B5")].copy()
    decomposition = pd.DataFrame(mechanism["detector_decomposition"])
    principal = principal.merge(
        decomposition[["detector", "n_correctable_safe", "verdict"]],
        on="detector",
        how="left",
        validate="one_to_one",
    )
    return principal[
        [
            "detector",
            "n_units",
            "n_correctable_safe",
            "delta_ba",
            "delta_fpr",
            "delta_fnr",
            "harm_rate",
            "coverage",
            "safety_fail",
            "verdict",
        ]
    ].rename(columns={"n_units": "n_k8"})


def _limitations_table() -> pd.DataFrame:
    rows = [
        {
            "limitation_id": "L1",
            "scope": "Evidence breadth",
            "evidence": "Only fabric and walnuts satisfy the frozen TEST K=8 eligibility rule; 10 DINOv2 and 30 pooled units.",
            "claim_constraint": "Do not claim broad industrial-category generalization.",
            "status": "LIMITED_BUT_REPORTABLE",
        },
        {
            "limitation_id": "L2",
            "scope": "Residual harm",
            "evidence": "All nine sealed harmful DINOv2 repeat-level cases are in walnuts; 89% have high source FNR.",
            "claim_constraint": "Report localized residual risk; do not add a post-hoc walnuts rule.",
            "status": "R1_LOCALIZED_RESIDUAL_RISK",
        },
        {
            "limitation_id": "L3",
            "scope": "Detector transfer",
            "evidence": "WideResNet50-2 is actuator-mismatched and EfficientAD has no safe correctability headroom under the frozen rule.",
            "claim_constraint": "The positive claim is DINOv2-specific, not detector-agnostic.",
            "status": "DETECTOR_SPECIFIC",
        },
        {
            "limitation_id": "L4",
            "scope": "Support size",
            "evidence": "K=4 is unsafe and FULL is not monotonically safer than K=8 under the frozen policy.",
            "claim_constraint": "Do not claim smaller or larger support sets are safe by monotonicity.",
            "status": "K8_ONLY",
        },
        {
            "limitation_id": "L5",
            "scope": "External comparator",
            "evidence": "No literature method matched the frozen deployment semantics without adding a new protocol choice.",
            "claim_constraint": "Report an audited absence of a fair direct comparator, not superiority over prior work.",
            "status": "NO_MATCHED_COMPARATOR",
        },
        {
            "limitation_id": "L6",
            "scope": "Compute",
            "evidence": "Threshold actuation overhead is measured on CPU; base DINOv2 per-image runtime is not verified in PB8.",
            "claim_constraint": "Claim negligible actuator overhead only; do not claim end-to-end runtime superiority.",
            "status": "BASE_COST_NOT_VERIFIED",
        },
        {
            "limitation_id": "L7",
            "scope": "Capability",
            "evidence": "The method changes an operating threshold and does not change feature ranking.",
            "claim_constraint": "Do not claim AUROC/AUPR or representation improvement.",
            "status": "OPERATING_POINT_ONLY",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    main_benchmark = pd.read_csv(OUT / "B_FINAL_MAIN_BENCHMARK.csv")
    detectors = pd.read_csv(OUT / "B_DETECTOR_BENCHMARK.csv")
    component = pd.read_csv(OUT / "C_COMPONENT_ABLATION.csv")
    support = pd.read_csv(OUT / "D_K_SUPPORT_FINAL.csv", dtype={"K": str})
    entities = pd.read_csv(OUT / "B_PER_ENTITY_ALL.csv")
    mechanism = json.loads((OUT / "E_MECHANISM_EVIDENCE.json").read_text(encoding="utf-8"))

    table_1 = _main_table(main_benchmark)
    table_2 = _detector_table(detectors, mechanism)
    table_3 = component.loc[component["group"].eq("DINOv2")].reset_index(drop=True)
    table_4 = support.reset_index(drop=True)
    table_5 = _limitations_table()
    for frame, name in [
        (table_1, "TABLE_1_MAIN_BENCHMARK.csv"),
        (table_2, "TABLE_2_DETECTOR_DECOMPOSITION.csv"),
        (table_3, "TABLE_3_COMPONENT_ABLATION.csv"),
        (table_4, "TABLE_4_SUPPORT_ROBUSTNESS.csv"),
        (table_5, "TABLE_5_LIMITATIONS.csv"),
    ]:
        _write(frame, name)

    curve = pd.DataFrame(mechanism["conservative_actuation"]["dino_q50_curve"])
    no_adapt = pd.DataFrame(
        [{"lambda": 0.0, "delta_ba": 0.0, "delta_fnr": 0.0, "harm_rate": 0.0, "coverage": 0.0}]
    )
    figure_2 = pd.concat([no_adapt, curve], ignore_index=True)
    figure_2.insert(
        0, "policy", ["NoAdapt", "lambda=0.25", "lambda=0.50", "lambda=0.75", "lambda=1.00"]
    )
    figure_2.to_csv(FIG / "FIGURE_2_SAFETY_UTILITY_SOURCE.csv", index=False)

    figure_3 = table_2.copy()
    figure_3["safe_headroom_fraction"] = figure_3["n_correctable_safe"] / figure_3["n_k8"]
    figure_3.to_csv(FIG / "FIGURE_3_DETECTOR_DECOMPOSITION_SOURCE.csv", index=False)
    table_4.to_csv(FIG / "FIGURE_4_SUPPORT_ROBUSTNESS_SOURCE.csv", index=False)

    figure_5 = entities.loc[
        entities["baseline"].eq("B5")
        & entities["K"].astype(str).eq("8")
        & entities["detector"].eq("dinov2_vits14")
        & entities["delta_ba"].notna(),
        ["category", "condition", "harm", "delta_ba", "delta_fnr", "coverage", "n_repeats"],
    ].copy()
    assert len(figure_5) == 10
    figure_5.to_csv(FIG / "FIGURE_5_RESIDUAL_HARM_SOURCE.csv", index=False)

    gate = {
        "schema_version": "1.0",
        "phase": "H",
        "tables": 5,
        "figure_source_snapshots": 4,
        "principal_table_rows": len(table_1),
        "detector_table_rows": len(table_2),
        "limitations": len(table_5),
        "verdict": "PASS",
        "terminal": "PROCEED_TO_I",
    }
    (OUT / "H_GATE.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
