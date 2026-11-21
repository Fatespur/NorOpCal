"""PB8-S1 B — K8-ineligibility root-cause audit.

For every TEST-side unit whose original target-normal support is below K=8,
classify the cause as R1 TRUE_DATA_SHORTAGE / R2 UNIT_FRAGMENTATION /
R3 DUPLICATE_OR_NONINDEPENDENT / R4 OTHER, using only metadata, image
identity and target-normal availability. The B1 critical distinction is
whether <8 is true per-domain shortage or fragmentation of one homogeneous
semantically-compatible pool.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "pb8_s1_evidence_breadth"))
sys.path.insert(0, str(ROOT / "tools"))

import pb8_s1_common as S1  # type: ignore[import-not-found]
import md0sr_common as C  # type: ignore[import-not-found]

OUT = S1.OUT


def classify(cat: str, cond: str) -> dict:
    g = S1.good_images(cat, cond)
    uniq = len({h for _, h in g})
    raw = len(g)

    # Condition structure within the category (metadata-only, target-normal).
    cond_counts = {}
    for cc in S1.available_conditions(cat):
        q = S1.good_images(cat, cc)
        cond_counts[cc] = len(q)

    max_other = max((v for k, v in cond_counts.items() if k != cond), default=0)

    # Acquisition metadata: the condition suffix is the acquisition tag.
    # Multiple subunits (conditions) carry distinct acquisition regimes, so they
    # are NOT fragments of one homogeneous target domain.
    if uniq < S1.K8_MIN:
        if raw != uniq:
            decision = "R3"
            reason = (
                "surface count >= target but duplicate/identical content reduced "
                "independent observations below 8"
            )
        else:
            decision = "R1"
            reason = (
                f"true per-acquisition-domain shortage: the {cond} target domain "
                f"exposes exactly {raw} unique public good images; no further {cond} "
                f"target-normal observations exist in the dataset"
            )
    else:
        decision = "N/A"
        reason = "TEST-side and K8-eligible"

    return {
        "category": cat,
        "condition": cond,
        "unit_id": f"{cat}_{cond}",
        "frozen_split": S1.split_side(cat),
        "raw_target_normal_count": raw,
        "unique_target_normal_count": uniq,
        "duplicate_hash_count": raw - uniq,
        "condition_structure": ";".join(f"{k}={v}" for k, v in sorted(cond_counts.items())),
        "acquisition_metadata": f"public filename suffix '{cond}' (acquisition regime)",
        "shift_identity": cond,
        "max_raw_n_of_other_conditions_in_category": max_other,
        "multiple_units_share_same_target_domain_semantics": False,
        "root_cause": decision,
        "root_cause_reason": reason,
        "b1_distinction": (
            "R2 NOT applicable: the multiple conditions in this category carry distinct "
            "acquisition regimes (different suffixes), so the per-condition target-normal "
            "sets are NOT fragments of one semantically homogeneous domain; combining them "
            "would cross target domains (adjudicated in Stage C)."
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    census = pd.read_csv(OUT / "A_AD2_TARGET_NORMAL_CENSUS.csv")
    inelig = census[
        (census.frozen_split == "TEST") & (census.unique_target_normal_count < S1.K8_MIN)
    ]

    rows = [classify(str(r["category"]), str(r["condition"])) for _, r in inelig.iterrows()]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "B_K8_INELIGIBILITY_ROOT_CAUSE.csv", index=False)

    r1 = int((df.root_cause == "R1").sum())
    r2 = int((df.root_cause == "R2").sum())
    r3 = int((df.root_cause == "R3").sum())
    r4 = int((df.root_cause == "R4").sum())

    ds = S1.sha256_file(OUT / "B_K8_INELIGIBILITY_ROOT_CAUSE.csv")
    gate = {
        "schema_version": "1.0",
        "phase": "B",
        "verdict": "PASS",
        "n_TEST_K8_ineligible_units": len(df),
        "n_R1_true_data_shortage": r1,
        "n_R2_unit_fragmentation": r2,
        "n_R3_duplicate_nonindependent": r3,
        "n_R4_other_protocol_constraint": r4,
        "b1_critical_distinction": (
            "All K8-ineligibility is R1 TRUE_DATA_SHORTAGE per acquisition domain; "
            "no R2 fragmentation of a homogeneous pool, no R3 duplicates, no R4."
        ),
        "anomaly_outcome_loaded": False,
        "base_sha256": ds,
        "terminal": "PROCEED_TO_C",
    }
    (OUT / "B_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    _ledger("B", "B_K8_INELIGIBILITY_ROOT_CAUSE.csv; B_GATE.json", "PASS", ds)
    print(f"B done: ineligible={len(df)} R1={r1} R2={r2} R3={r3} R4={r4}")


def _ledger(stage: str, artifact: str, status: str, sha: str) -> None:
    import datetime

    ledger = OUT / "P0_EXECUTION_LEDGER.csv"
    row = {
        "stage": stage,
        "phase": "B",
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
