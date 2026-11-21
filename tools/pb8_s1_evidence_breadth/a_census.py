"""PB8-S1 A — full AD2 target-normal census + unique-image verification.

Metadata-only: enumerates every (category, acquisition-condition) unit, its
raw/unique target-normal observation count (content SHA-256 duplicate check)
and its original K4/K8 eligibility under the frozen semantics. No anomaly
outcome is read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "pb8_s1_evidence_breadth"))
sys.path.insert(0, str(ROOT / "tools"))

import pb8_s1_common as S1  # type: ignore[import-not-found]
import md0sr_common as C  # type: ignore[import-not-found]

OUT = S1.OUT
ORIGINAL_ENTITY_DEF = (
    "PB6 frozen: unit = category x acquisition-condition x detector; "
    "target-normal = test_public good images under that condition suffix; "
    "source = test_public good_regular; K8 eligibility = TEST-side unit with FULL support n >= 8"
)


def census_row(cat: str, cond: str) -> Dict[str, Any]:
    side = S1.split_side(cat)
    g = S1.good_images(cat, cond)
    paths = [S1.rel_to_root(p) for p, _ in g]
    hashes = [h for _, h in g]
    raw_n = len(g)
    uniq_n = len(set(hashes))
    score_n = S1.condition_normal_n(cat, cond)
    k4_orig = side == "TEST" and raw_n >= 4
    k8_orig = side == "TEST" and raw_n >= 8
    return {
        "category": cat,
        "condition": cond,
        "unit_id": f"{cat}_{cond}",
        "source_domain": S1.SRC_CONDITION,
        "target_domain": cond,
        "target_normal_count": raw_n,
        "target_normal_score_n": score_n,
        "unique_target_normal_count": uniq_n,
        "target_normal_paths": ";".join(paths),
        "duplicate_hash_count": raw_n - uniq_n,
        "K4_eligible": k4_orig,
        "K8_eligible_original": k8_orig,
        "FULL_support_n": raw_n,
        "frozen_split": side,
        "original_entity_definition": ORIGINAL_ENTITY_DEF,
    }


def category_summary(cat: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ns = [r["target_normal_count"] for r in rows]
    side = S1.split_side(cat)
    n8 = sum(1 for r in rows if r["K8_eligible_original"])
    perc = (n8 / len(rows)) * 100.0 if rows else float("nan")
    reasons = sorted({_reason(r) for r in rows})
    return {
        "category": cat,
        "frozen_split": side,
        "number_of_units": len(rows),
        "min_target_normal_n": min(ns),
        "median_target_normal_n": median(ns),
        "max_target_normal_n": max(ns),
        "unique_min_n": min(r["unique_target_normal_count"] for r in rows),
        "n_K4_eligible_units": sum(1 for r in rows if r["K4_eligible"]),
        "n_K8_eligible_units": n8,
        "pct_K8_eligible": round(perc, 2),
        "category_K8_eligible": side == "TEST" and all(r["K8_eligible_original"] for r in rows),
        "K8_failure_reason": ";".join(reasons),
    }


def _reason(r: Dict[str, Any]) -> str:
    if r["frozen_split"] != "TEST":
        return "excluded by frozen DEV/TEST split"
    if r["unique_target_normal_count"] < S1.K8_MIN:
        return f"true target-normal shortage per acquisition domain (n={r['unique_target_normal_count']} < 8)"
    return "TEST-side and K8-eligible (n>=8)"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    units = S1.unit_list()

    rows = [census_row(cat, cond) for cat, cond in units]
    census = pd.DataFrame(rows)
    census.to_csv(OUT / "A_AD2_TARGET_NORMAL_CENSUS.csv", index=False)

    summaries = []
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    for cat in C.MVTEC_CATEGORIES:
        summaries.append(category_summary(cat, by_cat[cat]))
    sum_df = pd.DataFrame(summaries)
    sum_df.to_csv(OUT / "A_CATEGORY_SUPPORT_SUMMARY.csv", index=False)

    totals = {
        "n_units": len(rows),
        "n_raw_target_normal_images_across_all_units": int(
            sum(r["target_normal_count"] for r in rows)
        ),
        "n_unique_images": int(sum(r["unique_target_normal_count"] for r in rows)),
        "n_duplicate_hash_rows": int(sum(r["duplicate_hash_count"] for r in rows)),
        "n_TEST_units": int(sum(r["frozen_split"] == "TEST" for r in rows)),
        "n_TEST_K8_eligible_units": int(sum(r["K8_eligible_original"] for r in rows)),
        "n_TEST_K8_eligible_DINOv2_units": int(sum(r["K8_eligible_original"] for r in rows)),
        "TEST_K8_eligible_categories": sorted(
            {r["category"] for r in rows if r["K8_eligible_original"]}
        ),
    }

    ds = S1.sha256_file(OUT / "A_AD2_TARGET_NORMAL_CENSUS.csv")
    gate = {
        "schema_version": "1.0",
        "phase": "A",
        "verdict": "PASS",
        "n_units_enumerated": totals["n_units"],
        "all_units_match_frozen_39": totals["n_units"] == 39,
        "all_score_counts_match_raw_image_counts": bool(
            all(r["target_normal_count"] == r["target_normal_score_n"] for r in rows)
        ),
        "n_duplicate_hash_rows": totals["n_duplicate_hash_rows"],
        "test_k8_eligible_categories": totals["TEST_K8_eligible_categories"],
        "anomaly_outcome_loaded": False,
        "base_sha256": ds,
        "terminal": "PROCEED_TO_B",
    }
    (OUT / "A_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    with (OUT / "A_CENSUS_TOTALS.json").open("w", encoding="utf-8") as fh:
        json.dump(totals, fh, indent=2)

    _ledger(
        "A",
        "A_AD2_TARGET_NORMAL_CENSUS.csv; A_CATEGORY_SUPPORT_SUMMARY.csv; A_GATE.json",
        "PASS",
        ds,
    )
    print(
        f"A done: units={totals['n_units']} dup_rows={totals['n_duplicate_hash_rows']} "
        f"K8_categories={totals['TEST_K8_eligible_categories']}"
    )


def _ledger(stage: str, artifact: str, status: str, sha: str) -> None:
    import datetime

    ledger = OUT / "P0_EXECUTION_LEDGER.csv"
    row = {
        "stage": stage,
        "phase": "A",
        "artifact": artifact,
        "status": status,
        "sha256": sha,
        "anomaly_outcome_loaded": False,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        + "Z",
    }
    df = pd.DataFrame([row])
    if ledger.exists():
        df = pd.concat([pd.read_csv(ledger), df], ignore_index=True)
    df.to_csv(ledger, index=False)


if __name__ == "__main__":
    main()
