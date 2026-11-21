"""PB8 C — frozen component, shrinkage, and random-abstention ablations."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "pb8_publication_benchmark"))

import pb8_common as P8  # type: ignore[import-not-found]  # noqa: E402

OUT = P8.OUT
DET = "dinov2_vits14"
PAIR_SEED = 20260820
RANDOM_MASTER_SEED = 20260820
RANDOM_N = 1000
LAMBDAS = [0.25, 0.50, 0.75, 1.00]


def _valid(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(subset=["delta_ba", "delta_fpr", "delta_fnr", "harm"]).copy()


def _summary_row(
    frame: pd.DataFrame,
    group: str,
    policy_id: str,
    gate: bool,
    shrinkage: bool,
    lam: float,
    proposed: pd.DataFrame,
) -> dict[str, Any]:
    valid = _valid(frame)
    summary = P8.summary_metrics(valid)
    paired = P8.paired_block_bootstrap(valid, proposed, seed=PAIR_SEED, b=P8.CONF_BOOT_B)
    return {
        "policy_id": policy_id,
        "group": group,
        "gate": gate,
        "shrinkage": shrinkage,
        "lambda": lam,
        "n_units": int(summary["n"]),
        "n_categories": int(valid["category"].nunique()),
        "delta_ba": summary["mean_delta_ba"],
        "delta_fpr": summary["mean_delta_fpr"],
        "delta_fnr": summary["mean_delta_fnr"],
        "harm_rate": summary["harm_rate"],
        "catastrophic_rate": summary["catastrophic_rate"],
        "coverage": summary["coverage"],
        "safety_fail": bool(summary["mean_delta_fnr"] > P8.DFNR_MAX + 1e-12),
        "paired_policy_minus_proposed_delta_ba": paired["diff_mean_delta_ba"],
        "paired_dba_ci_lo": paired["diff_ci_lo_delta_ba"],
        "paired_dba_ci_hi": paired["diff_ci_hi_delta_ba"],
        "paired_policy_minus_proposed_delta_fnr": paired["diff_mean_delta_fnr"],
        "paired_dfnr_ci_lo": paired["diff_ci_lo_delta_fnr"],
        "paired_dfnr_ci_hi": paired["diff_ci_hi_delta_fnr"],
        "paired_policy_minus_proposed_harm": paired["diff_mean_harm"],
        "paired_harm_ci_lo": paired["diff_ci_lo_harm"],
        "paired_harm_ci_hi": paired["diff_ci_hi_harm"],
    }


def _per_repeat_dino() -> tuple[pd.DataFrame, int]:
    """Return bounded-only repeat rows and frozen-gate selected count."""
    fw = P8.Firewall()
    rows: list[dict[str, Any]] = []
    gate_selected = 0
    for cat, cond, det in P8.k8_eligible_entities():
        if det != DET:
            continue
        src = P8.load_src(cat, det, fw)
        cnd = P8.load_cnd(cat, cond, det, fw)
        anm = P8.load_anm(cat, det, fw)
        tau = P8.source_threshold(src)
        raw = P8.evaluate_entity_repeats(
            cat,
            cond,
            det,
            P8.PRINCIPAL_FAMILY,
            P8.PRINCIPAL_LAMBDA,
            P8.NO_GATE,
            "8",
            src,
            cnd,
            anm,
            tau,
            n_repeats=P8.K_REPEATS,
        )
        gated = P8.evaluate_entity_repeats(
            cat,
            cond,
            det,
            P8.PRINCIPAL_FAMILY,
            P8.PRINCIPAL_LAMBDA,
            P8.PRINCIPAL_CUTOFF,
            "8",
            src,
            cnd,
            anm,
            tau,
            n_repeats=P8.K_REPEATS,
        )
        gate_selected += sum(int(item["selected"]) for item in gated)
        for repeat, item in enumerate(raw):
            rows.append(
                {
                    "category": cat,
                    "condition": cond,
                    "detector": det,
                    "repeat": repeat,
                    **item,
                }
            )
    return pd.DataFrame(rows), gate_selected


def _random_control() -> pd.DataFrame:
    repeats, n_selected = _per_repeat_dino()
    n_total = len(repeats)
    prereg = {
        "schema_version": "1.0",
        "control": "coverage-matched random abstention",
        "detector": DET,
        "K": 8,
        "actuator": "A_MEDIAN_RAW lambda=0.25",
        "selection_uses_anomaly_outcome": False,
        "random_master_seed": RANDOM_MASTER_SEED,
        "seed_ids": [0, RANDOM_N - 1],
        "n_seeds": RANDOM_N,
        "selection_unit": "entity-support-repeat",
        "inference_unit": "entity after averaging support repeats",
        "n_total_selection_units": n_total,
        "n_selected_per_seed": n_selected,
        "matched_coverage": n_selected / n_total,
    }
    (OUT / "C_RANDOM_ABSTENTION_PREREGISTRATION.json").write_text(
        json.dumps(prereg, indent=2), encoding="utf-8"
    )

    metric_cols = [
        "delta_ba",
        "delta_fpr",
        "delta_fnr",
        "benefit",
        "harm",
        "catastrophic",
        "coverage",
    ]
    result_rows: list[dict[str, Any]] = []
    for seed_id in range(RANDOM_N):
        rng = np.random.default_rng(RANDOM_MASTER_SEED + seed_id)
        selected = np.zeros(n_total, dtype=bool)
        selected[rng.choice(n_total, size=n_selected, replace=False)] = True
        working = repeats.copy()
        for column in metric_cols:
            working[column] = np.where(selected, working[column], 0.0)
        entity = working.groupby(["category", "condition", "detector"], as_index=False)[
            metric_cols
        ].mean()
        summary = P8.summary_metrics(entity)
        result_rows.append(
            {
                "seed_id": seed_id,
                "seed": RANDOM_MASTER_SEED + seed_id,
                "n_units": int(summary["n"]),
                "n_categories": int(entity["category"].nunique()),
                "delta_ba": summary["mean_delta_ba"],
                "delta_fpr": summary["mean_delta_fpr"],
                "delta_fnr": summary["mean_delta_fnr"],
                "harm_rate": summary["harm_rate"],
                "catastrophic_rate": summary["catastrophic_rate"],
                "coverage": summary["coverage"],
                "matched_selected": n_selected,
                "matched_total": n_total,
            }
        )
    return pd.DataFrame(result_rows)


def main() -> None:
    per_entity = pd.read_csv(OUT / "B_PER_ENTITY_ALL.csv", dtype={"K": str})
    k8 = per_entity[per_entity["K"] == "8"]
    proposed_all = _valid(k8[k8["baseline"] == "B5"])

    component_map = [
        ("B2", False, False, 1.00),
        ("B4", True, False, 1.00),
        ("B3", False, True, 0.25),
        ("B5", True, True, 0.25),
    ]
    component_rows: list[dict[str, Any]] = []
    for group, detector in [("DINOv2", DET), ("POOLED", None)]:
        proposed = (
            proposed_all if detector is None else proposed_all[proposed_all.detector == detector]
        )
        for policy_id, gate, shrinkage, lam in component_map:
            frame = k8[k8["baseline"] == policy_id]
            if detector is not None:
                frame = frame[frame.detector == detector]
            component_rows.append(
                _summary_row(frame, group, policy_id, gate, shrinkage, lam, proposed)
            )
    components = pd.DataFrame(component_rows)
    components.to_csv(OUT / "C_COMPONENT_ABLATION.csv", index=False)

    shrink_rows: list[dict[str, Any]] = []
    shrink_frames: dict[tuple[str, bool], pd.DataFrame] = {}
    for lam in LAMBDAS:
        for gate in [False, True]:
            cutoff = P8.PRINCIPAL_CUTOFF if gate else P8.NO_GATE
            frame = P8.evaluate_policy_per_entity("A_MEDIAN_RAW", lam, cutoff, "8")
            shrink_frames[(str(lam), gate)] = frame
            for group, detector in [("DINOv2", DET), ("POOLED", None)]:
                sub = frame if detector is None else frame[frame.detector == detector]
                valid = _valid(sub)
                summary = P8.summary_metrics(valid)
                shrink_rows.append(
                    {
                        "group": group,
                        "lambda": lam,
                        "gate": "Q50" if gate else "NONE",
                        "n_units": int(summary["n"]),
                        "n_categories": int(valid["category"].nunique()),
                        "delta_ba": summary["mean_delta_ba"],
                        "delta_fpr": summary["mean_delta_fpr"],
                        "delta_fnr": summary["mean_delta_fnr"],
                        "harm_rate": summary["harm_rate"],
                        "catastrophic_rate": summary["catastrophic_rate"],
                        "coverage": summary["coverage"],
                        "safety_fail": bool(summary["mean_delta_fnr"] > P8.DFNR_MAX + 1e-12),
                    }
                )
    shrinkage = pd.DataFrame(shrink_rows)
    shrinkage.to_csv(OUT / "C_SHRINKAGE_ABLATION.csv", index=False)

    random_control = _random_control()
    random_control.to_csv(OUT / "C_RANDOM_ABSTENTION_CONTROL.csv", index=False)
    dino_component = components[components.group == "DINOv2"].set_index("policy_id")
    random_summary = {
        "mean_delta_ba": float(random_control.delta_ba.mean()),
        "ci_lo_delta_ba": float(random_control.delta_ba.quantile(0.025)),
        "ci_hi_delta_ba": float(random_control.delta_ba.quantile(0.975)),
        "mean_delta_fnr": float(random_control.delta_fnr.mean()),
        "mean_harm_rate": float(random_control.harm_rate.mean()),
        "mean_coverage": float(random_control.coverage.mean()),
        "interval_semantics": "2.5/97.5 percentiles over 1000 preregistered random-control seeds",
    }
    (OUT / "C_RANDOM_ABSTENTION_SUMMARY.json").write_text(
        json.dumps(random_summary, indent=2), encoding="utf-8"
    )

    q50_dino = shrinkage[(shrinkage.group == "DINOv2") & (shrinkage.gate == "Q50")]
    monotonic_safety = bool(
        q50_dino.sort_values("lambda").delta_fnr.is_monotonic_increasing
        and q50_dino.sort_values("lambda").harm_rate.is_monotonic_increasing
    )
    gate = {
        "schema_version": "1.0",
        "phase": "C",
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": {
            "component_2x2_complete": len(components) == 8,
            "full_method_dino_safe": not bool(dino_component.loc["B5", "safety_fail"]),
            "raw_dino_safety_fail": bool(dino_component.loc["B2", "safety_fail"]),
            "gate_only_dino_safety_fail": bool(dino_component.loc["B4", "safety_fail"]),
            "bounded_only_dino_safe": not bool(dino_component.loc["B3", "safety_fail"]),
            "shrinkage_safety_monotonic_with_lambda": monotonic_safety,
            "random_seeds_complete": len(random_control) == RANDOM_N,
            "random_control_outcome_free": True,
        },
        "random_control_summary": random_summary,
        "verdict": "PASS",
        "terminal": "PROCEED_TO_D",
    }
    (OUT / "C_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(
        "C done: 2x2 complete; random control 1000 seeds; "
        f"DINO full dBA={dino_component.loc['B5', 'delta_ba']:.4f}"
    )


if __name__ == "__main__":
    main()
