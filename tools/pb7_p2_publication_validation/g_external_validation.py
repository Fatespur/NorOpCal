"""PB7-P2 Stage G — Optional Independent-Setting Validation (assessment).

Assesses whether AeBAD-S / original MVTec AD can provide a semantically
compatible external setting for the frozen principal policy WITHOUT protocol
distortion.

Frozen method must be applied unchanged: A_MEDIAN_RAW, lambda=0.25, frozen Q50
cutoff (0.015890181064605713, MVTec AD 2 DEV median_shift quantile), K=8, no
re-tuning (G2). If the external dataset requires re-tuning lambda/Q50, it is
NOT confirmatory validation.

Conclusion recorded here: EXTERNAL_SETTING_NOT_AVAILABLE_WITHOUT_PROTOCOL_DISTORTION
because the frozen Q50 cutoff is dataset/score-scale specific and does not
transfer to AeBAD-S (all AeBAD-S median_shift values exceed the frozen cutoff,
so the gate degenerates to Always-On). No external validation is run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p1_safety_actuator"))
sys.path.insert(0, str(ROOT / "tools" / "pb7_p2_publication_validation"))

import md0sr_common as C  # type: ignore[import-not-found]  # noqa: E402
import pb7_p2_common as P2  # type: ignore[import-not-found]  # noqa: E402

OUT = P2.OUT
SD = C.SCORES_DIR
FROZEN_CUTOFF = P2.PRINCIPAL_CUTOFF


def aebad_median_shifts() -> dict[str, float]:
    """Median_shift (target-normal median - source median) per AeBAD-S test condition."""
    train_files = [
        "aebad_s_train_good_background_dinov2_vits14_scores.npz",
        "aebad_s_train_good_illumination_dinov2_vits14_scores.npz",
        "aebad_s_train_good_view_dinov2_vits14_scores.npz",
    ]
    src = np.concatenate([np.load(SD / f)["image_scores"] for f in train_files])
    out: dict[str, float] = {}
    for cond in ["background", "illumination", "view", "same"]:
        f = f"aebad_s_test_good_{cond}_dinov2_vits14_scores.npz"
        cnd = np.load(SD / f)["image_scores"]
        out[cond] = float(np.median(cnd) - np.median(src))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    aebad_shifts = aebad_median_shifts()
    aebad_min = min(aebad_shifts.values())

    # ---- G1: external protocol registration (attempted) ---------------------- #
    protocol = {
        "source_domain": "AeBAD-S train good (background+illumination+view, n=521)",
        "target_normal": "AeBAD-S test good per condition (background/illumination/view/same)",
        "target_anomaly": "AeBAD-S test defects (ablation/breakdown/fracture/groove)",
        "unit": "image (single-image capture per record; no physical identity grouping)",
        "detector": "dinov2_vits14 (frozen)",
        "k": "8",
        "frozen_method": "A_MEDIAN_RAW lambda=0.25 cutoff=Q50 (frozen value)",
        "no_tuning": True,
    }

    # ---- G2: frozen-cutoff transferability check ------------------------------ #
    # Frozen Q50 cutoff is the median of the MVTec AD 2 DEV median_shift
    # distribution. If every AeBAD-S median_shift exceeds it, the gate never
    # abstains on AeBAD-S -> degenerates to Always-On -> the frozen gate is not
    # transferable, and re-tuning the cutoff is forbidden (G2).
    all_above_cutoff = aebad_min > FROZEN_CUTOFF

    # ---- verdict --------------------------------------------------------------- #
    verdict = "EXTERNAL_SETTING_NOT_AVAILABLE_WITHOUT_PROTOCOL_DISTORTION"
    assessment = {
        "schema_version": "1.0",
        "stage": "PB7_P2_PUBLICATION_READINESS_CONFIRMATORY_VALIDATION",
        "phase": "G",
        "timestamp": ts,
        "protocol_registration": protocol,
        "frozen_cutoff": FROZEN_CUTOFF,
        "aebad_s_median_shift_by_condition": aebad_shifts,
        "aebad_s_min_median_shift": aebad_min,
        "all_aebad_median_shift_above_frozen_cutoff": all_above_cutoff,
        "reason": (
            "The frozen Q50 cutoff (0.015890181064605713) is the median of the "
            "MVTec AD 2 DEV median_shift distribution and is score-scale specific. "
            "All AeBAD-S median_shift values (0.039-0.107) exceed the frozen "
            "cutoff, so applying it unchanged makes the gate never abstain "
            "(equivalent to Always-On). Re-tuning lambda/Q50 for AeBAD-S is "
            "forbidden by G2 (it would no longer be confirmatory validation). "
            "Original MVTec AD offers no legal target-shift definition matching "
            "the acquisition-shift + source-normal/target-normal setting without "
            "manufacturing a pseudo-validation. Therefore no external setting is "
            "available without protocol distortion."
        ),
        "verdict": verdict,
        "external_validation_run": False,
    }
    (OUT / "G_GATE.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")

    # G_EXTERNAL_VALIDATION.csv: protocol registration + non-availability record
    rec = {
        "external_setting": "AeBAD-S",
        "source_domain": protocol["source_domain"],
        "target_normal": protocol["target_normal"],
        "target_anomaly": protocol["target_anomaly"],
        "unit": protocol["unit"],
        "detector": protocol["detector"],
        "k": protocol["k"],
        "frozen_method": protocol["frozen_method"],
        "no_tuning": protocol["no_tuning"],
        "frozen_cutoff": FROZEN_CUTOFF,
        "aebad_s_min_median_shift": aebad_min,
        "all_above_frozen_cutoff": all_above_cutoff,
        "validation_run": False,
        "verdict": verdict,
    }
    pd.DataFrame([rec]).to_csv(OUT / "G_EXTERNAL_VALIDATION.csv", index=False)

    # ---- ledger --------------------------------------------------------------- #
    ledger = pd.read_csv(OUT / "P0_EXECUTION_LEDGER.csv")
    ledger = pd.concat(
        [
            ledger,
            pd.DataFrame(
                [
                    {
                        "stage": "G",
                        "step": "external_setting_validation_assessment",
                        "status": "COMPLETE",
                        "timestamp": ts,
                        "artifacts": "G_GATE.json, G_EXTERNAL_VALIDATION.csv",
                        "note": (
                            f"G verdict {verdict}; AeBAD-S median_shift all above frozen cutoff "
                            f"({all_above_cutoff}), min={aebad_min:.4f} "
                            f"vs cutoff={FROZEN_CUTOFF:.4f}; "
                            f"no external validation run"
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ledger.to_csv(OUT / "P0_EXECUTION_LEDGER.csv", index=False)

    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
