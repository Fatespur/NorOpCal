#!/usr/bin/env python3
"""PB1-DS-PB6 D8 — tri-detector K8 support bridge + contamination bridge.

Evaluates how the frozen tri-detector selective policies (fresh-fitted in D5)
behave when the target-normal support changes, holding the D5 adapt decisions
fixed (the same frozen semantics as PB3/PB4: adapt decisions frozen; the
recalibration transform/QM is re-estimated from the changed support).

  K8 bridge        : FULL support (all target normals) vs K=8 support (8 sampled
                     target normals). Estimands frozen by the protocol binding:
                     common-78 (FULL) and common-40 (K8). K8 sampling follows the
                     frozen nested seed rule (PB1DS_PB3_common) extended to the
                     third detector by the same deterministic seed identity.
  contamination    : rho in {0.0, 0.05, 0.1, 0.2}, 20 repeats, at the primary
                     K=8 operating point. Contaminants are drawn from the
                     entity's own 15-sample regular-defect set with the frozen
                     PB1DS_PB4_common.draw_contaminant_ids rule (injection, not
                     replacement); n_cont = ceil(rho * 8).

Downstream metrics follow the frozen QM semantics of
pb1ds_pb3_common.compute_unit_at_k(method_after='QM'): normal side mapped
through the monotone quantile-map transform referenced on the changed support;
anomaly side through the frozen anomaly-self-referenced map; threshold Q95(src).

Negative scientific results pass through verbatim; no re-tuning is performed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import md0sr_common as C  # noqa: E402
import pb1ds_c0_common as M0  # noqa: E402  (classify_change)
import pb1ds_pb3_common as B3  # noqa: E402  (frozen nested K sampling / QM transform)
import pb1ds_pb4_common as PB4  # noqa: E402  (frozen contamination draw)
import pb6_tri_common as T  # noqa: E402

PB6_DIR = ROOT / "reports" / "pb1ds_publication" / "pb6_real_third_detector"
CANDIDATES = ["A_GLOBAL", "D_CATEGORY_ONLY", "C_CATEGORY_CONDITIONED"]
DETECTORS = ["dinov2_vits14", "wide_resnet50_2", "efficientad"]
RHO_GRID = [0.0, 0.05, 0.1, 0.2]
REPEATS = 20
PRIMARY_K = 8
Q95 = 0.95
ESTIMAND_FULL = "common-78 (FULL)"
ESTIMAND_K8 = "common-40 (K8)"


def tri_k8_indices(cat: str, cond: str, det: str, n: int, repeat: int = 1) -> np.ndarray:
    """Deterministic K=8 target-normal indices (frozen nested seed rule).

    Mirrors pb1ds_pb3_common.support_indices for the third detector: one base
    permutation per (entity, repeat) from the frozen canonical seed identity,
    first 8 indices sorted. n is the actual target-normal count from the cache.
    """
    rng = np.random.default_rng(B3.seed_int(B3.seed_identity("BASE", cat, cond, det, repeat)))
    return np.sort(rng.permutation(n)[:PRIMARY_K])


def after_metrics_qm(
    src: np.ndarray, cnd: np.ndarray, anm: np.ndarray, support_k: np.ndarray, tau: float
) -> dict[str, float]:
    """Frozen QM downstream for a changed support (compute_unit_at_k 'QM')."""
    ref_n = B3.quantile_map_transform(src, support_k)
    used_n = np.asarray([ref_n(x) for x in cnd], dtype=float)
    used_a = C.quantile_mapping(src, anm)
    labels = np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))])
    return C.detection_metrics(np.concatenate([used_n, used_a]), labels, tau)


def bridge_row(
    cat: str,
    cond: str,
    det: str,
    candidate: str,
    support: str,
    rho: float,
    repeat_id: int,
    adapt: int,
    mb: dict[str, float],
    ma: dict[str, float],
    benefit_label: int,
) -> dict[str, Any]:
    dba = ma["ba"] - mb["ba"]
    dfpr = ma["fpr"] - mb["fpr"]
    dfnr = ma["fnr"] - mb["fnr"]
    _, harm, neutral = M0.classify_change(dba, dfpr, dfnr)
    return {
        "category": cat,
        "condition": cond,
        "baseline": det,
        "baseline_name": T.TRI_DET_NAMES[det],
        "candidate": candidate,
        "support": support,
        "rho": rho,
        "repeat_id": repeat_id,
        "n_cont": 0,
        "adapt": adapt,
        "ba_before": mb["ba"],
        "ba_after": ma["ba"],
        "delta_ba": float(dba),
        "fpr_before": mb["fpr"],
        "fpr_after": ma["fpr"],
        "delta_fpr": float(dfpr),
        "fnr_before": mb["fnr"],
        "fnr_after": ma["fnr"],
        "delta_fnr": float(dfnr),
        "benefit": int(benefit_label),
        "harm": int(harm),
        "neutral": int(neutral),
    }


def main() -> int:
    oof = pd.read_csv(PB6_DIR / "PB1DS_PB6_D5_OOF_PER_UNIT.csv")
    adapt_map = {
        (r["category"], r["condition"], r["baseline"], r["candidate"]): int(r["adapt"])
        for _, r in oof.iterrows()
    }
    benefit_map = {
        (r["category"], r["condition"], r["baseline"], r["candidate"]): int(r["benefit"])
        for _, r in oof.iterrows()
    }

    units = sorted(set(zip(oof["category"], oof["condition"], strict=True)))
    k8_rows: list[dict[str, Any]] = []
    contam_rows: list[dict[str, Any]] = []
    k8_eligible: set[tuple[str, str]] = set()

    for cat, cond in units:
        for det in DETECTORS:
            src = C.load_image_scores(C.source_rel(cat, det))
            cnd = C.load_image_scores(C.cond_normal_rel(cat, cond, det))
            anm = C.load_image_scores(C.anomaly_rel(cat, det))
            tau = float(np.percentile(src, 100.0 * Q95))
            labels = np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))])
            mb = C.detection_metrics(np.concatenate([cnd, anm]), labels, tau)
            n = len(cnd)
            if n >= PRIMARY_K:
                k8_eligible.add((cat, cond))

            for cand in CANDIDATES:
                key = (cat, cond, det, cand)
                adapt = adapt_map[key]
                benefit_label = benefit_map[key]

                # FULL support (matches D5)
                ma_full = after_metrics_qm(src, cnd, anm, cnd, tau)
                k8_rows.append(
                    bridge_row(
                        cat, cond, det, cand, "FULL", 0.0, 0, adapt, mb, ma_full, benefit_label
                    )
                )

                # K8 support (deterministic single draw, repeat=1)
                if n >= PRIMARY_K:
                    idx8 = tri_k8_indices(cat, cond, det, n)
                    ma_k8 = after_metrics_qm(src, cnd, anm, cnd[idx8], tau)
                    k8_rows.append(
                        bridge_row(
                            cat, cond, det, cand, "K8", 0.0, 1, adapt, mb, ma_k8, benefit_label
                        )
                    )

                # contamination bridge at primary K=8, 20 repeats
                if n >= PRIMARY_K:
                    for rho in RHO_GRID:
                        for rep in range(1, REPEATS + 1):
                            # Frozen PB4 injection semantics extended to the third
                            # detector: K8 base indices from the frozen nested seed
                            # rule (tri_k8_indices), contaminants from the frozen
                            # PB4.draw_contaminant_ids rule. The frozen
                            # build_contaminated_support cannot be reused because its
                            # sealed full_n_map() only knows the two reference
                            # detectors (KeyError on the efficientad entity id).
                            normal_idx = tri_k8_indices(cat, cond, det, n, rep)
                            n_cont = PB4.contam_count(str(PRIMARY_K), rho, n)
                            contam_idx = (
                                np.asarray(
                                    PB4.draw_contaminant_ids(cat, cond, det, rho, rep, n_cont),
                                    dtype=int,
                                )
                                if n_cont > 0
                                else np.asarray([], dtype=int)
                            )
                            support = np.concatenate([cnd[normal_idx], anm[contam_idx]])
                            ma_rho = after_metrics_qm(src, cnd, anm, support, tau)
                            row = bridge_row(
                                cat,
                                cond,
                                det,
                                cand,
                                "K8_CONTAM",
                                rho,
                                rep,
                                adapt,
                                mb,
                                ma_rho,
                                benefit_label,
                            )
                            row["n_cont"] = int(n_cont)
                            contam_rows.append(row)

    k8 = pd.DataFrame(k8_rows)
    k8.to_csv(PB6_DIR / "PB1DS_PB6_K8_BRIDGE.csv", index=False)

    # ---- K8 aggregate (frozen estimands) ----------------------------------
    full_rows = k8[k8["support"] == "FULL"]
    k8_only = k8[k8["support"] == "K8"]
    agg_rows: list[dict[str, Any]] = []
    for cand in CANDIDATES:
        # common-78 (FULL): FULL support on the FULL-eligible population
        sub = full_rows[full_rows["candidate"] == cand]
        agg_rows.append(_agg_row(ESTIMAND_FULL, "FULL", cand, sub))
        # common-40 (K8): K8 support on the K8-eligible population
        sub = k8_only[k8_only["candidate"] == cand]
        agg_rows.append(_agg_row(ESTIMAND_K8, "K8", cand, sub))
        # contrast: FULL support on the K8-eligible population (within-set)
        sub = full_rows[
            (full_rows["candidate"] == cand)
            & full_rows[["category", "condition"]].apply(tuple, axis=1).isin(k8_eligible)
        ]
        agg_rows.append(_agg_row("common-40 (FULL)", "FULL", cand, sub))
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(PB6_DIR / "PB1DS_PB6_K8_AGGREGATE.csv", index=False)

    contam = pd.DataFrame(contam_rows)
    contam.to_csv(PB6_DIR / "PB1DS_PB6_CONTAMINATION_BRIDGE.csv", index=False)

    # ---- contamination aggregate (per candidate x rho) ----------------------
    cagg_rows: list[dict[str, Any]] = []
    for cand in CANDIDATES:
        for rho in RHO_GRID:
            sub = contam[(contam["candidate"] == cand) & np.isclose(contam["rho"], rho)]
            cagg_rows.append(_agg_row(f"K8 rho={rho}", "K8_CONTAM", cand, sub))
    cagg = pd.DataFrame(cagg_rows)
    cagg.to_csv(PB6_DIR / "PB1DS_PB6_CONTAMINATION_AGGREGATE.csv", index=False)

    md = [
        "# PB1-DS-PB6 — D8 K8 Support Bridge + Contamination Bridge",
        "",
        f"- K8 bridge rows: {len(k8)} (FULL + K8 per entity x candidate)",
        f"- K8-eligible units: {len(k8_eligible)} (FULL_n >= {PRIMARY_K})",
        f"- Contamination rows: {len(contam)} "
        f"(K8 x {len(RHO_GRID)} rho x {REPEATS} repeats x candidate)",
        "- Adapt decisions frozen from D5; QM re-estimated from the changed support.",
        "- Contaminants injected from the entity's own regular-defect pool "
        "(frozen PB4 draw rule); n_cont = ceil(rho * 8).",
        "",
        "## K8 aggregate (frozen estimands)",
        "",
    ]
    md.append(agg.to_string(index=False))
    md.append("")
    md.append("## Contamination aggregate (primary K=8)")
    md.append("")
    md.append(cagg.to_string(index=False))
    (PB6_DIR / "PB1DS_PB6_D8_BRIDGES.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


def _agg_row(estimand: str, support: str, cand: str, sub: pd.DataFrame) -> dict[str, Any]:
    if sub.empty:
        return {
            "estimand": estimand,
            "support": support,
            "candidate": cand,
            "n_entities": 0,
            "mean_delta_ba": np.nan,
            "mean_delta_fpr": np.nan,
            "mean_delta_fnr": np.nan,
            "coverage": np.nan,
            "benefit_precision": np.nan,
            "global_harm": np.nan,
        }
    adapt = (sub["adapt"] == 1).to_numpy()
    nadapt = int(adapt.sum())
    prec = float(np.sum(adapt & (sub["benefit"] == 1)) / max(nadapt, 1)) if nadapt else 0.0
    # selective estimand (frozen D6/terminal semantics): used = after if adapt else before
    bb = sub["ba_before"].to_numpy()
    ba = sub["ba_after"].to_numpy()
    fprb = sub["fpr_before"].to_numpy()
    fpra = sub["fpr_after"].to_numpy()
    fnrb = sub["fnr_before"].to_numpy()
    fnra = sub["fnr_after"].to_numpy()
    ba_used = np.where(adapt, ba, bb)
    fpr_used = np.where(adapt, fpra, fprb)
    fnr_used = np.where(adapt, fnra, fnrb)
    return {
        "estimand": estimand,
        "support": support,
        "candidate": cand,
        "n_entities": len(sub),
        "mean_delta_ba": float((ba_used - bb).mean()),
        "mean_delta_fpr": float((fpr_used - fprb).mean()),
        "mean_delta_fnr": float((fnr_used - fnrb).mean()),
        "coverage": float(adapt.mean()),
        "benefit_precision": prec,
        "global_harm": float(np.mean(adapt & (sub["harm"] == 1))),
    }


if __name__ == "__main__":
    raise SystemExit(main())
