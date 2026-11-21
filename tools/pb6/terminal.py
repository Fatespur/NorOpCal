#!/usr/bin/env python3
"""PB1-DS-PB6 scientific terminal — deterministic tri-detector verdict.

Reads the sealed D5 OOF (39 units x 3 detectors x methods), the D7 benchmark
frame (incl. the frozen BC_RANK replay) and the B=5000 block-bootstrap JSON,
then applies the frozen decision tree below and seals exactly one allowed
scientific terminal.

Frozen decision constants (documented here; never tuned to force a PASS):
  MIN_PRECISION       = 0.50  (frozen operating-point precision gate)
  UTIL_EPS            = 1e-9  (utility strictly positive)
  HARM_EPS            = 1e-6  (harm reduction strictly below AlwaysOn)
  PREV_NEGLIGIBLE     = 0.05  (EfficientAD benefit prevalence ~ zero)
  PREV_MATERIAL       = 0.15  (reference benefit prevalence material)

Decision tree (deterministic, data-driven):
  1. core_supported = (D.gh < AO.gh) and (C.gh < AO.gh)          [safety]
                      and (D.prec >= 0.50) and (C.prec >= 0.50)  [precision gate]
                      and (D.dba > 0) and (C.dba > 0)            [utility]
     if not core_supported -> D-D (does not support the core selective
     recalibration claim).
  2. material detector heterogeneity (EfficientAD vs the two references):
        eff_prev <= 0.05 and ref_prev >= 0.15   (EfficientAD has essentially
                                                 no recalibration benefit)
        or eff_prec < 0.50 and ref_prec >= 0.50 (EfficientAD selective fails
                                                 the precision gate)
        or eff_dba <= 0 and ref_dba > 0         (EfficientAD utility
                                                 non-positive while refs positive)
     if hetero -> D-C (material detector heterogeneity).
  3. attenuated utility: eff_dba < ref_dba and tri_dba > 0
     (the third detector dilutes the pooled utility but the selective safety
     story holds) -> D-B.
  4. otherwise -> D-A (tri-detector validation supports the utility-safety
     Pareto).

Any scientific negative result is passed through verbatim; this stage never
re-tunes, re-selects, or re-runs to force a PASS.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PB6_DIR = ROOT / "reports" / "pb1ds_publication" / "pb6_real_third_detector"
CANDIDATES = ["A_GLOBAL", "D_CATEGORY_ONLY", "C_CATEGORY_CONDITIONED"]
CONTROLS = ["NO_ADAPT", "ALWAYS_ON_QM", "ORACLE"]
BC_RANK = "BC_RANK_BASED_THRESHOLD"
THIRD_DETECTOR = "efficientad"

MIN_PRECISION = 0.50
UTIL_EPS = 1e-9
HARM_EPS = 1e-6
PREV_NEGLIGIBLE = 0.05
PREV_MATERIAL = 0.15

TERMINAL_A = "PB1DS_PB6_COMPLETE_REAL_THIRD_DETECTOR_VALIDATION_SUPPORTS_UTILITY_SAFETY_PARETO"
TERMINAL_B = (
    "PB1DS_PB6_COMPLETE_THIRD_DETECTOR_REPLICATION_SUPPORTS_SELECTIVE_SAFETY_WITH_"
    "ATTENUATED_UTILITY"
)
TERMINAL_C = "PB1DS_PB6_COMPLETE_REAL_THIRD_DETECTOR_REVEALS_MATERIAL_DETECTOR_HETEROGENEITY"
TERMINAL_D = (
    "PB1DS_PB6_COMPLETE_REAL_THIRD_DETECTOR_DOES_NOT_SUPPORT_CORE_SELECTIVE_RECALIBRATION_CLAIM"
)
ALLOWED_TERMINALS = {TERMINAL_A, TERMINAL_B, TERMINAL_C, TERMINAL_D}


def metrics_for(df: pd.DataFrame) -> dict[str, float]:
    """Frozen policy-metric semantics (same as D6/D7)."""
    if df.empty:
        return {
            "n_entities": 0,
            "coverage": np.nan,
            "benefit_prevalence": np.nan,
            "benefit_precision": np.nan,
            "benefit_recall": np.nan,
            "global_harm": np.nan,
            "harm_among_adapted": np.nan,
            "delta_ba_vs_noadapt": np.nan,
            "ba_before_mean": np.nan,
            "ba_selective": np.nan,
        }
    adapt = (df["adapt"] == 1).to_numpy()
    benefit = df["benefit"].to_numpy()
    harm = df["harm"].to_numpy()
    bb = df["ba_before"].to_numpy()
    ba = df["ba_after"].to_numpy()
    n = len(df)
    nadapt = int(adapt.sum())
    nben = int(benefit.sum())
    ba_used = np.where(adapt, ba, bb)
    prec = float(np.sum(adapt & (benefit == 1)) / max(nadapt, 1)) if nadapt else 0.0
    rec = float(np.sum(adapt & (benefit == 1)) / max(nben, 1)) if nben else 0.0
    return {
        "n_entities": n,
        "coverage": float(adapt.mean()),
        "benefit_prevalence": float(benefit.mean()),
        "benefit_precision": prec,
        "benefit_recall": rec,
        "global_harm": float(np.mean(adapt & (harm == 1))),
        "harm_among_adapted": float(np.mean(adapt & (harm == 1))) if nadapt else np.nan,
        "delta_ba_vs_noadapt": float((ba_used - bb).mean()),
        "ba_before_mean": float(bb.mean()),
        "ba_selective": float(ba_used.mean()),
    }


def _mean(a: float, b: float) -> float:
    vals = [v for v in (a, b) if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def decide_terminal(
    oof: pd.DataFrame, bench: pd.DataFrame, boot: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Deterministic single-terminal decision (documented frozen tree)."""
    info: dict[str, Any] = {}

    def m(cand: str, frame: pd.DataFrame) -> dict[str, float]:
        return metrics_for(frame[frame["candidate"] == cand])

    D = m("D_CATEGORY_ONLY", oof)
    C = m("C_CATEGORY_CONDITIONED", oof)
    A = m("A_GLOBAL", oof)
    AO = m("ALWAYS_ON_QM", oof)
    OR = m("ORACLE", oof)
    BC = m(BC_RANK, bench)

    safety_kept = bool(
        D["global_harm"] < AO["global_harm"] - HARM_EPS
        and C["global_harm"] < AO["global_harm"] - HARM_EPS
    )
    prec_ok = bool(
        D["benefit_precision"] >= MIN_PRECISION and C["benefit_precision"] >= MIN_PRECISION
    )
    util_pos = bool(D["delta_ba_vs_noadapt"] > UTIL_EPS and C["delta_ba_vs_noadapt"] > UTIL_EPS)
    core_supported = safety_kept and prec_ok and util_pos

    info["pooled"] = {
        "D_CATEGORY_ONLY": D,
        "C_CATEGORY_CONDITIONED": C,
        "A_GLOBAL": A,
        "ALWAYS_ON_QM": AO,
        "ORACLE": OR,
        BC_RANK: BC,
    }
    info["core_supported"] = core_supported
    info["safety_kept"] = safety_kept
    info["precision_gate_ok"] = prec_ok
    info["utility_positive"] = util_pos

    if not core_supported:
        return TERMINAL_D, info

    eff = oof[oof["baseline"] == THIRD_DETECTOR]
    ref = oof[oof["baseline"] != THIRD_DETECTOR]
    eff_D = m("D_CATEGORY_ONLY", eff)
    eff_C = m("C_CATEGORY_CONDITIONED", eff)
    ref_D = m("D_CATEGORY_ONLY", ref)
    ref_C = m("C_CATEGORY_CONDITIONED", ref)

    sel_eff = oof[oof["candidate"].isin(["D_CATEGORY_ONLY", "C_CATEGORY_CONDITIONED"])]
    eff_prev = float(sel_eff[sel_eff["baseline"] == THIRD_DETECTOR]["benefit"].mean())
    ref_prev = float(sel_eff[sel_eff["baseline"] != THIRD_DETECTOR]["benefit"].mean())
    eff_prec = _mean(eff_D["benefit_precision"], eff_C["benefit_precision"])
    ref_prec = _mean(ref_D["benefit_precision"], ref_C["benefit_precision"])
    eff_dba = _mean(eff_D["delta_ba_vs_noadapt"], eff_C["delta_ba_vs_noadapt"])
    ref_dba = _mean(ref_D["delta_ba_vs_noadapt"], ref_C["delta_ba_vs_noadapt"])

    hetero = bool(
        (eff_prev <= PREV_NEGLIGIBLE and ref_prev >= PREV_MATERIAL)
        or (eff_prec < MIN_PRECISION and ref_prec >= MIN_PRECISION)
        or (eff_dba <= UTIL_EPS and ref_dba > UTIL_EPS)
    )
    info["per_detector"] = {
        "eff_D": eff_D,
        "eff_C": eff_C,
        "ref_D": ref_D,
        "ref_C": ref_C,
        "eff_benefit_prevalence": eff_prev,
        "ref_benefit_prevalence": ref_prev,
        "eff_precision": eff_prec,
        "ref_precision": ref_prec,
        "eff_delta_ba": eff_dba,
        "ref_delta_ba": ref_dba,
    }
    info["material_heterogeneity"] = hetero

    if hetero:
        return TERMINAL_C, info

    tri_dba = _mean(D["delta_ba_vs_noadapt"], C["delta_ba_vs_noadapt"])
    attenuated = bool(eff_dba < ref_dba - UTIL_EPS and tri_dba > UTIL_EPS)
    info["attenuated_utility"] = attenuated
    info["tri_delta_ba"] = tri_dba

    if attenuated:
        return TERMINAL_B, info

    return TERMINAL_A, info


def main() -> int:
    oof = pd.read_csv(PB6_DIR / "PB1DS_PB6_D5_OOF_PER_UNIT.csv")
    bench = pd.read_csv(PB6_DIR / "PB1DS_PB6_BENCHMARK_FRAME.csv")
    boot = json.loads((PB6_DIR / "PB1DS_PB6_BLOCK_BOOTSTRAP_5000.json").read_text(encoding="utf-8"))

    terminal, info = decide_terminal(oof, bench, boot)

    # supporting evidence: bootstrap CI lower bounds for the selective candidates
    boot_info: dict[str, Any] = {}
    for cand in CANDIDATES:
        r = boot.get("methods", {}).get(cand, {})
        if "delta_ba_vs_noadapt" in r:
            boot_info[cand] = {
                "mean": r["delta_ba_vs_noadapt"]["mean"],
                "ci_low": r["delta_ba_vs_noadapt"]["ci_low"],
                "ci_high": r["delta_ba_vs_noadapt"]["ci_high"],
            }
        else:
            boot_info[cand] = r
    info["bootstrap_5000"] = boot_info

    legal = terminal in ALLOWED_TERMINALS
    record = {
        "stage": "PB1-DS-PB6",
        "terminal": terminal,
        "terminal_legal": legal,
        "decision": info,
    }
    (PB6_DIR / "PB1DS_PB6_TERMINAL.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8"
    )

    lines: list[str] = [
        "# PB1-DS-PB6 — Scientific Terminal",
        "",
        f"- Terminal: **{terminal}**",
        f"- Terminal legal: {legal}",
        f"- Core claim supported: {info['core_supported']} "
        f"(safety_kept={info['safety_kept']}, precision_gate={info['precision_gate_ok']}, "
        f"utility_positive={info['utility_positive']})",
        f"- Material detector heterogeneity: {info.get('material_heterogeneity', 'n/a')}",
        f"- Attenuated utility: {info.get('attenuated_utility', 'n/a')}",
        "",
        "## Pooled tri-detector metrics (117 entities)",
        "",
    ]
    for cand in [
        "D_CATEGORY_ONLY",
        "C_CATEGORY_CONDITIONED",
        "A_GLOBAL",
        "ALWAYS_ON_QM",
        "ORACLE",
        BC_RANK,
    ]:
        m = info["pooled"][cand]
        lines.append(
            f"- {cand}: n={m['n_entities']} coverage={m['coverage']:.4f} "
            f"prec={m['benefit_precision']:.4f} gh={m['global_harm']:.4f} "
            f"dba={m['delta_ba_vs_noadapt']:.4f}"
        )
    lines.append("")
    lines.append("## Per-detector selective metrics (D/C)")
    lines.append("")
    pd_ = info.get("per_detector")
    if pd_:
        lines.append(
            f"- EfficientAD: benefit_prevalence={pd_['eff_benefit_prevalence']:.4f} "
            f"precision={pd_['eff_precision']:.4f} dba={pd_['eff_delta_ba']:.4f}"
        )
        lines.append(
            f"- References  : benefit_prevalence={pd_['ref_benefit_prevalence']:.4f} "
            f"precision={pd_['ref_precision']:.4f} dba={pd_['ref_delta_ba']:.4f}"
        )
    lines.append("")
    lines.append("## B=5000 block-bootstrap CI (delta_ba_vs_noadapt)")
    lines.append("")
    for cand, r in boot_info.items():
        if "ci_low" in r:
            lines.append(
                f"- {cand}: mean={r['mean']:.4f} CI=[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
            )
        else:
            lines.append(f"- {cand}: {r}")
    (PB6_DIR / "PB1DS_PB6_TERMINAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nPB6 scientific terminal = {terminal}")
    return 0 if legal else 1


if __name__ == "__main__":
    raise SystemExit(main())
