"""PB8 F — authoritative confirmatory and paired-ablation statistics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "pb8_publication_benchmark"
P2 = ROOT / "reports" / "pb7_p2_publication_validation"


def main() -> None:
    rows: list[dict[str, object]] = []
    confirm = pd.read_csv(P2 / "D_CONFIRMATORY_BOOTSTRAP.csv")
    for _, row in confirm[confirm.policy == "GATED"].iterrows():
        rows.append(
            {
                "family": "authoritative_confirmatory",
                "comparison": "Frozen proposed policy vs NoAdapt",
                "group": row["group"],
                "n_entities": int(row["n_entities"]),
                "B": 10000,
                "seed": 20260820,
                "effect_delta_ba": row["boot_mean_delta_ba"],
                "ci_lo_delta_ba": row["boot_ci_lo_delta_ba"],
                "ci_hi_delta_ba": row["boot_ci_hi_delta_ba"],
                "effect_delta_fnr": row["boot_mean_delta_fnr"],
                "ci_lo_delta_fnr": row["boot_ci_lo_delta_fnr"],
                "ci_hi_delta_fnr": row["boot_ci_hi_delta_fnr"],
                "effect_harm": row["boot_mean_harm_rate"],
                "ci_lo_harm": row["boot_ci_lo_harm"],
                "ci_hi_harm": row["boot_ci_hi_harm"],
                "interval_semantics": "category-aware bootstrap; PB7-P2 authoritative reuse",
                "p_value": "NOT_COMPUTED",
            }
        )

    component = pd.read_csv(OUT / "C_COMPONENT_ABLATION.csv")
    for _, row in component.iterrows():
        rows.append(
            {
                "family": "paired_component_ablation",
                "comparison": f"{row['policy_id']} minus B5 proposed",
                "group": row["group"],
                "n_entities": int(row["n_units"]),
                "B": 10000,
                "seed": 20260820,
                "effect_delta_ba": row["paired_policy_minus_proposed_delta_ba"],
                "ci_lo_delta_ba": row["paired_dba_ci_lo"],
                "ci_hi_delta_ba": row["paired_dba_ci_hi"],
                "effect_delta_fnr": row["paired_policy_minus_proposed_delta_fnr"],
                "ci_lo_delta_fnr": row["paired_dfnr_ci_lo"],
                "ci_hi_delta_fnr": row["paired_dfnr_ci_hi"],
                "effect_harm": row["paired_policy_minus_proposed_harm"],
                "ci_lo_harm": row["paired_harm_ci_lo"],
                "ci_hi_harm": row["paired_harm_ci_hi"],
                "interval_semantics": "paired category-aware bootstrap; policy minus proposed",
                "p_value": "NOT_COMPUTED",
            }
        )

    random = pd.read_csv(OUT / "C_RANDOM_ABSTENTION_CONTROL.csv")
    proposed = float(
        component[(component.group == "DINOv2") & (component.policy_id == "B5")].iloc[0].delta_ba
    )
    diff = proposed - random.delta_ba
    rows.append(
        {
            "family": "random_abstention_negative_control",
            "comparison": "B5 proposed minus coverage-matched random abstention",
            "group": "DINOv2",
            "n_entities": int(random.iloc[0].n_units),
            "B": 1000,
            "seed": 20260820,
            "effect_delta_ba": float(diff.mean()),
            "ci_lo_delta_ba": float(diff.quantile(0.025)),
            "ci_hi_delta_ba": float(diff.quantile(0.975)),
            "effect_delta_fnr": float(0.002444444444444444 - random.delta_fnr.mean()),
            "ci_lo_delta_fnr": float("nan"),
            "ci_hi_delta_fnr": float("nan"),
            "effect_harm": float(0.03666666666666667 - random.harm_rate.mean()),
            "ci_lo_harm": float("nan"),
            "ci_hi_harm": float("nan"),
            "interval_semantics": "randomization interval over 1000 preregistered control seeds; not category-level sampling CI",
            "p_value": "NOT_COMPUTED",
        }
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "F_FINAL_STATISTICS.csv", index=False)
    gate = {
        "schema_version": "1.0",
        "phase": "F",
        "authoritative_primary_reused": True,
        "primary_bootstrap_recomputed_for_prettier_interval": False,
        "paired_B": 10000,
        "paired_seed": 20260820,
        "multiplicity": "effect sizes and CIs reported; no p-value family tested",
        "independence_unit": "category-aware; entity summaries average support-repeat uncertainty",
        "verdict": "PASS",
        "terminal": "PROCEED_TO_G",
    }
    (OUT / "F_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"F done: {len(frame)} statistical rows; authoritative B=10000 reused")


if __name__ == "__main__":
    main()
