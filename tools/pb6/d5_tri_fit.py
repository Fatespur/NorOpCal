#!/usr/bin/env python3
"""PB1-DS-PB6 D5 — tri-detector nested leave-one-(category,condition)-out OOF.

Protocol-frozen re-execution of the PUB0 known-category / unknown-condition
machinery over THREE detectors (dinov2_vits14, wide_resnet50_2, efficientad).

For every PRIMARY-PAIRED (category, condition) unit (39 units):
  OUTER_TEST = ALL three detectors of that unit; OUTER_DEV = every other eligible
               unit (same-category other conditions ARE allowed, category is
               deployment-known).
  INNER      = leave-one-(category,condition)-out within OUTER_DEV picks C and the
               operating threshold (max mean delta_BA, precision>=0.50, ties ->
               lower harm, then lower coverage).
  SEAL       = fold model + operating point written to PB1DS_PB6_D5_OUTER_SEALS.jsonl
               BEFORE any test-outcome access.
  TEST       = held-out unit decided from normal-only scores with the sealed
               model/threshold; outcome (benefit/harm/metrics) read only after seal.

Candidates: A_GLOBAL (Z+d), D_CATEGORY_ONLY (c+d), C_CATEGORY_CONDITIONED (Z+c+d).
Controls:   NO_ADAPT, ALWAYS_ON_QM, ORACLE.

Only outer-test rows are kept. All candidate decisions are computed independently.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pb1ds_c1_common as M  # noqa: E402  (AccessGuard / audit_fold / provenance helpers)
import pb6_tri_common as T  # noqa: E402

PB6_DIR = ROOT / "reports" / "pb1ds_publication" / "pb6_real_third_detector"
CANDIDATES = ["A_GLOBAL", "D_CATEGORY_ONLY", "C_CATEGORY_CONDITIONED"]
CONTROLS = ["NO_ADAPT", "ALWAYS_ON_QM", "ORACLE"]


def _src(cat: str, bl: str) -> str:
    return f"mv2_{cat}_public_good_regular_{bl}_scores.npz"


def _cnd(cat: str, cond: str, bl: str) -> str:
    return f"mv2_{cat}_public_good_{cond}_{bl}_scores.npz"


def _anm(cat: str, bl: str) -> str:
    return f"mv2_{cat}_public_bad_regular_{bl}_scores.npz"


def discover_units() -> list[tuple[str, str]]:
    """All (category, condition) units from the authoritative D2 asset resolution
    whose score caches are present for ALL three detectors (auto-discovered from
    the real cache, never assumed uniform)."""
    import pandas as pd

    df = pd.read_csv(PB6_DIR / "PB1DS_PB6_D2_ASSET_RESOLUTION.csv")
    units: list[tuple[str, str]] = []
    for cat, cond in df[["category", "condition"]].drop_duplicates().itertuples(index=False):
        try:
            for bl in T.TRI_DETECTORS:
                T.load_image_scores(_src(cat, bl))
                T.load_image_scores(_cnd(cat, cond, bl))
                T.load_image_scores(_anm(cat, bl))
        except FileNotFoundError:
            continue
        units.append((cat, cond))
    return sorted(units)


def unit_groups(df: pd.DataFrame) -> np.ndarray:
    return (df["category"] + "|" + df["condition"]).to_numpy()


def loco_oof(cand: str, df: pd.DataFrame, Cv: float) -> np.ndarray:
    """Inner leave-one-unit-out selector probabilities aligned to df rows."""
    y = df["benefit"].to_numpy()
    groups = unit_groups(df)
    loo = LeaveOneGroupOut()
    scores = np.full(len(df), np.nan, dtype=float)
    for tr_idx, te_idx in loo.split(df, y, groups):
        if len(np.unique(y[tr_idx])) < 2:
            continue
        X_tr = T.encode_candidate(cand, df)[tr_idx]
        X_te = T.encode_candidate(cand, df)[te_idx]
        clf = T.fit_selector(X_tr, y[tr_idx], Cv)
        scores[te_idx] = clf.predict_proba(X_te)[:, 1]
    return scores


def inner_select(cand: str, dev_df: pd.DataFrame) -> dict[str, Any]:
    bb = dev_df["ba_before"].to_numpy()
    ba = dev_df["ba_after"].to_numpy()
    benef = dev_df["benefit"].to_numpy()
    harm = dev_df["harm"].to_numpy()
    rows: list[dict[str, Any]] = []
    for Cv in T.C_GRID:
        oof = loco_oof(cand, dev_df, Cv)
        m = ~np.isnan(oof)
        n_scored = int(m.sum())
        if n_scored == 0:
            rows.append(
                {
                    "C": Cv,
                    "threshold": None,
                    "inner_util": -np.inf,
                    "inner_coverage": np.inf,
                    "inner_harm": np.inf,
                    "inner_precision": 0.0,
                    "n_scored": 0,
                }
            )
            continue
        thr = T.select_threshold(oof[m], benef[m], harm[m], bb[m], ba[m], T.THRESH_GRID)
        if thr is None:
            rows.append(
                {
                    "C": Cv,
                    "threshold": None,
                    "inner_util": -np.inf,
                    "inner_coverage": np.inf,
                    "inner_harm": np.inf,
                    "inner_precision": 0.0,
                    "n_scored": n_scored,
                }
            )
            continue
        adapt = oof[m] > thr
        util = float((np.where(adapt, ba[m], bb[m]) - bb[m]).mean())
        cov = float(adapt.mean())
        hgt = float(np.mean(adapt & (harm[m] == 1)))
        prec = float(benef[m][adapt].mean()) if int(adapt.sum()) > 0 else 0.0
        rows.append(
            {
                "C": Cv,
                "threshold": thr,
                "inner_util": util,
                "inner_coverage": cov,
                "inner_harm": hgt,
                "inner_precision": prec,
                "n_scored": n_scored,
            }
        )
    valid = [r for r in rows if r["threshold"] is not None]
    if not valid:
        return {"selected_C": None, "threshold": None, "inner_rows": rows}
    best = max(
        valid, key=lambda r: (r["inner_util"], -r["inner_harm"], -r["inner_coverage"], -r["C"])
    )
    oof_best = loco_oof(cand, dev_df, best["C"])
    mb = ~np.isnan(oof_best)
    auprc = float("nan")
    auroc = float("nan")
    if int(mb.sum()) > 0:
        yy = dev_df["benefit"].to_numpy()[mb]
        try:
            auprc = float(average_precision_score(yy, oof_best[mb]))
        except Exception:
            auprc = float("nan")
        if len(np.unique(yy)) >= 2:
            try:
                auroc = float(roc_auc_score(yy, oof_best[mb]))
            except Exception:
                auroc = float("nan")
    return {
        "selected_C": best["C"],
        "threshold": best["threshold"],
        "inner_rows": rows,
        "inner_oof_auroc_dev": auroc,
        "inner_oof_auprc_dev": auprc,
        "selected_row": best,
    }


def compute_decision(
    cand: str, clf_best, thr: float | None, test_norm: pd.DataFrame, n_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    if clf_best is None or thr is None:
        return np.zeros(n_rows), np.zeros(n_rows, dtype=float)
    X = T.encode_candidate(cand, test_norm)
    prob = clf_best.predict_proba(X)[:, 1]
    return (prob > thr).astype(int), prob


def main() -> int:

    units = discover_units()
    assert len(units) == 39, f"expected 39 units, got {len(units)}"

    split_manifest = {
        "units": [[c, q] for c, q in units],
        "outer_folds": [
            {
                "index": i,
                "test": [c, q],
                "dev": [[cc, qq] for cc, qq in units if (cc, qq) != (c, q)],
            }
            for i, (c, q) in enumerate(units)
        ],
    }
    (PB6_DIR / "PB1DS_PB6_D5_SPLIT_MANIFEST.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )

    code_sha = _sha256(Path(__file__).resolve())
    config_sha = _sha256(PB6_DIR / "PB1DS_PB6_PROTOCOL_BINDING.json")

    start = time.time()
    oof_rows: list[dict[str, Any]] = []
    seals: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    for fi, (tc, tq) in enumerate(units):
        test_unit = (tc, tq)
        dev_units = [(c, q) for (c, q) in units if (c, q) != test_unit]

        guard = M.AccessGuard()
        dev_df = T.build_tri_frame(dev_units, guard)  # pre-seal: dev reads only

        sel: dict[str, dict[str, Any]] = {}
        clf: dict[str, Any] = {}
        for cand in CANDIDATES:
            s = inner_select(cand, dev_df)
            sel[cand] = s
            model = None
            if s["selected_C"] is not None:
                model = T.fit_selector(
                    T.encode_candidate(cand, dev_df), dev_df["benefit"].to_numpy(), s["selected_C"]
                )
            clf[cand] = model

        # seal before test-outcome access
        seals.append(
            {
                "fold_index": fi,
                "outer_test_category": tc,
                "outer_test_condition": tq,
                "outer_development_units": [list(u) for u in dev_units],
                "candidates": {
                    cand: {
                        "selected_C": sel[cand]["selected_C"],
                        "operating_threshold": sel[cand]["threshold"],
                        "inner_oof_auroc_dev": sel[cand]["inner_oof_auroc_dev"],
                        "inner_oof_auprc_dev": sel[cand]["inner_oof_auprc_dev"],
                        "no_valid_operating_point": sel[cand]["selected_C"] is None,
                    }
                    for cand in CANDIDATES
                },
                "feature_set": T.candidate_columns("C_CATEGORY_CONDITIONED"),
                "quantile_mapping": {
                    "source_threshold_quantile": 0.95,
                    "deterministic": True,
                    "monotonic": True,
                },
                "code_sha256": code_sha,
                "config_sha256": config_sha,
                "sealed_before_test_outcome_access": True,
                "frozen_at": _now(),
            }
        )

        guard.seal()
        test_norm = T.build_normal_only_tri_frame(test_unit)
        adapt: dict[str, np.ndarray] = {}
        prob: dict[str, np.ndarray] = {}
        for cand in CANDIDATES:
            a, p = compute_decision(
                cand, clf[cand], sel[cand]["threshold"], test_norm, len(test_norm)
            )
            adapt[cand] = a
            prob[cand] = p
        test_full = T.build_tri_frame([test_unit], guard)
        audit = M.audit_fold(test_unit, guard)
        audits.append(audit)

        for pos, (_, r) in enumerate(test_full.iterrows()):
            bl = r["baseline"]
            for cand in CANDIDATES + CONTROLS:
                if cand == "NO_ADAPT":
                    ad = 0
                elif cand == "ALWAYS_ON_QM":
                    ad = 1
                elif cand == "ORACLE":
                    ad = int(r["benefit"])
                else:
                    ad = int(adapt[cand][pos])
                ba_u = float(r["ba_after"]) if ad == 1 else float(r["ba_before"])
                fpr_u = float(r["fpr_after"]) if ad == 1 else float(r["fpr_before"])
                fnr_u = float(r["fnr_after"]) if ad == 1 else float(r["fnr_before"])
                f1_u = float(r["f1_after"]) if ad == 1 else float(r["f1_before"])
                if ad == 0:
                    outcome = "ABSTAIN"
                elif int(r["benefit"]) == 1:
                    outcome = "BENEFIT"
                elif int(r["harm"]) == 1:
                    outcome = "HARM"
                else:
                    outcome = "NEUTRAL"
                oof_rows.append(
                    {
                        "outer_fold_index": fi,
                        "category": r["category"],
                        "condition": r["condition"],
                        "baseline": bl,
                        "baseline_name": r["baseline_name"],
                        "candidate": cand,
                        "selector_prob": float(prob[cand][pos])
                        if cand in CANDIDATES
                        else float("nan"),
                        "selector_threshold": sel[cand]["threshold"]
                        if cand in CANDIDATES
                        else None,
                        "adapt": ad,
                        "ba_before": float(r["ba_before"]),
                        "ba_after": float(r["ba_after"]),
                        "delta_ba": float(r["delta_ba"]),
                        "ba_used": ba_u,
                        "fpr_before": float(r["fpr_before"]),
                        "fpr_after": float(r["fpr_after"]),
                        "delta_fpr": float(r["delta_fpr"]),
                        "fpr_used": fpr_u,
                        "fnr_before": float(r["fnr_before"]),
                        "fnr_after": float(r["fnr_after"]),
                        "delta_fnr": float(r["delta_fnr"]),
                        "fnr_used": fnr_u,
                        "f1_before": float(r["f1_before"]),
                        "f1_after": float(r["f1_after"]),
                        "f1_used": f1_u,
                        "benefit": int(r["benefit"]),
                        "harm": int(r["harm"]),
                        "neutral": int(r["neutral"]),
                        "policy_outcome": outcome,
                    }
                )

        print(
            f"[{fi:02d}] {tc}x{tq} dev={len(dev_units)} "
            + " ".join(
                f"{c}:C={sel[c]['selected_C']},thr={sel[c]['threshold']},adapt={int(adapt[c].sum())}"
                for c in CANDIDATES
            )
        )

    oof = pd.DataFrame(oof_rows)
    oof.to_csv(PB6_DIR / "PB1DS_PB6_D5_OOF_PER_UNIT.csv", index=False)
    with (PB6_DIR / "PB1DS_PB6_D5_OUTER_SEALS.jsonl").open("w", encoding="utf-8") as fh:
        for s in seals:
            fh.write(json.dumps(s, default=str) + "\n")
    (PB6_DIR / "PB1DS_PB6_D5_ACCESS_AUDIT.json").write_text(
        json.dumps(audits, indent=2, default=str), encoding="utf-8"
    )

    n_entities = len(oof) // len(CANDIDATES + CONTROLS)
    print("\n" + "=" * 70)
    print(
        f"PB1DS-PB6 D5 complete in {round(time.time() - start, 1)}s. "
        f"OOF rows={len(oof)} (39 units x 3 detectors x {len(CANDIDATES + CONTROLS)} methods = "
        f"{39 * 3 * len(CANDIDATES + CONTROLS)} expected). entities={n_entities}"
    )
    print(f"All firewall_pass: {all(a['firewall_pass'] for a in audits)}")
    return 0 if all(a["firewall_pass"] for a in audits) else 1


def _sha256(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
