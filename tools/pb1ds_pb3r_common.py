"""PB1-DS PB3-R shared library — End-to-End Few-Normal Frozen-Inference Re-execution.

PB3-R is an EXECUTION-SEMANTICS REPAIR / COMPLETION of PB3, not a new experiment,
method, or hyperparameter search. PB3 already disclosed that D/C/A_GLOBAL reused
the sealed PUB0 selector operating outputs and that K sensitivity was expressed
only through downstream recalibration (QM) re-estimation. PB3-R re-executes the
*full* chain under sampled target-normal support:

    S_K -> Z_K -> frozen selector -> p_K -> frozen operating threshold
         -> action_K -> QM_K -> downstream metrics

The frozen selector of a held-out unit is the per-outer-fold LogisticRegression
whose coefficients, intercept, and operating threshold are sealed in
PB1DS_C1_OUTER_SEALS.jsonl (A_GLOBAL, C_CATEGORY_CONDITIONED) and
PB1DS_PUB0_OUTER_SEALS.jsonl (D_CATEGORY_ONLY). No selector is refit, no
threshold is reselected, no feature is changed; only the SUPPORT used to
compute the canonical 10 deformation features is resampled (nested, from the
sealed PB3 support-seed manifest, read verbatim).

Scope (frozen, publication-locked): K grid {4,8,FULL} with the sealed 30-repeat
manifest; only frozen methods are executed. Z_K is recomputed from the K-sampled
support; D's selector has no Z input and is therefore K-invariant by construction
(verified: probability SD must be 0 and action flips 0).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import md0sr_common as C
import pb1ds_c0_common as M0
import pb1ds_pb1f_common as F
import pb1ds_pb3_common as B3

PB3R_OUT = Path("reports/pb1ds_publication/pb3r_end_to_end")

# inherited frozen upstream terminals (verified read-only in R0)
PB1_TERMINAL = "PB1DS_PB1_COMPLETE_PUBLICATION_BENCHMARK_PROTOCOL_FROZEN"
PB1F_TERMINAL = "PB1DS_PB1F_COMPLETE_FINAL_PROTOCOL_BINDING_VERIFIED"
PB2_TERMINAL = "PB1DS_PB2_COMPLETE_UTILITY_SAFETY_PARETO_SUPPORTED"
PB3_PRE0_TERMINAL = "PB1DS_PB3_PRE0_COMPLETE_SEMANTIC_AND_ESTIMAND_INTEGRITY_VERIFIED"
PB3_TERMINAL = "PB1DS_PB3_COMPLETE_FEW_NORMAL_ROBUSTNESS_SUPPORTED_WITH_MINIMUM_SUPPORT_REQUIREMENT"

# PB3-R allowed terminal states (section 38); exactly one is emitted.
TERM_E2E_CONFIRMED = "PB1DS_PB3R_COMPLETE_PB3_MINIMUM_SUPPORT_CONCLUSION_END_TO_END_CONFIRMED"
TERM_QM_BOTTLENECK = "PB1DS_PB3R_COMPLETE_LOW_K_BOTTLENECK_PRIMARILY_RECALIBRATION_ESTIMATION"
TERM_K8_SEL_INSTAB = (
    "PB1DS_PB3R_COMPLETE_MINIMUM_SUPPORT_K8_CONFIRMED_WITH_LOW_K_SELECTOR_INSTABILITY"
)
TERM_NOT_CONFIRMED = "PB1DS_PB3R_COMPLETE_PRIOR_PB3_MINIMUM_SUPPORT_CONCLUSION_NOT_CONFIRMED"

TERM_STOP_UPSTREAM = "PB1DS_PB3R_STOP_UPSTREAM_INTEGRITY_FAILURE"
TERM_STOP_FEATURE = "PB1DS_PB3R_STOP_FROZEN_FEATURE_NOT_VALID_AT_REGISTERED_K"
TERM_STOP_REPRO = "PB1DS_PB3R_STOP_FULL_INFERENCE_REPRODUCTION_FAILURE"
TERM_STOP_D_INV = "PB1DS_PB3R_STOP_D_K_INVARIANCE_VIOLATION"
TERM_STOP_EXEC = "PB1DS_PB3R_STOP_EXECUTION_INTEGRITY_FAILURE"

RESEAL_OK_TERMINALS = (
    TERM_E2E_CONFIRMED,
    TERM_QM_BOTTLENECK,
    TERM_K8_SEL_INSTAB,
)

# Strong stability gates (section 17): interpretation gates, not tuning knobs.
C_K8_ACTION_AGREE_OK = 0.90
C_K8_RANK_CORR_OK = 0.80
C_K4_ACTION_AGREE_MODERATE = 0.80

IMPLEMENTED = "IMPLEMENTED"
VERIFIED = "VERIFIED"


# --------------------------------------------------------------------------- #
# Frozen selector parameter binding (sealed, read-only)
# --------------------------------------------------------------------------- #
def load_c1_models() -> Dict[int, Dict[str, Dict[str, Any]]]:
    """fold_index -> candidate -> frozen {coef, intercept, threshold, selected_C}.
    Candidate set: A_GLOBAL, B_CATEGORY_CENTERED, C_CATEGORY_CONDITIONED."""
    out: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for line in (F.C1_OUT / "PB1DS_C1_OUTER_SEALS.jsonl").open(encoding="utf-8"):
        s = json.loads(line)
        fi = int(s["fold_index"])
        per = {}
        for cand in ("A_GLOBAL", "C_CATEGORY_CONDITIONED"):
            c = s["candidates"][cand]
            per[cand] = {
                "coef": [float(x) for x in c["selector_coefficients"]],
                "intercept": float(c["selector_intercept"]),
                "threshold": c["operating_threshold"],
                "selected_C": c["selected_C"],
                "no_valid": bool(c["no_valid_operating_point"]),
            }
        out[fi] = per
    return out


def load_d_models() -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for line in (F.PUB0_OUT / "PB1DS_PUB0_OUTER_SEALS.jsonl").open(encoding="utf-8"):
        s = json.loads(line)
        out[int(s["fold_index"])] = {
            "coef": [float(x) for x in s["selector_coefficients"]],
            "intercept": float(s["selector_intercept"]),
            "threshold": s["operating_threshold"],
            "selected_C": s["selected_C"],
            "no_valid": bool(s["no_valid_operating_point"]),
        }
    return out


def entity_to_fold() -> Dict[str, int]:
    """entity_id -> outer_fold_index (from the sealed C1 per-unit OOF)."""
    oof = pd.read_csv(F.C1_OUT / "PB1DS_C1_OOF_PER_UNIT.csv")
    oof = oof[oof["candidate"] == "C_CATEGORY_CONDITIONED"]
    return {
        f"{r['category']}|{r['condition']}|{r['baseline']}": int(r["outer_fold_index"])
        for _, r in oof.iterrows()
    }


def load_oof_all() -> pd.DataFrame:
    return pd.read_csv(F.PUB0_OUT / "PB1DS_PUB0_OOF_ALL.csv")


def load_category_only_oof() -> pd.DataFrame:
    return pd.read_csv(F.PUB0_OUT / "PB1DS_PUB0_CATEGORY_ONLY_OOF.csv")


def load_full_reconciliation() -> pd.DataFrame:
    return pd.read_csv(F.PB1F_OUT / "PB1DS_PB1F_FULL_SUPPORT_RECONCILIATION.csv")


def load_seed_manifest() -> pd.DataFrame:
    p = B3.PRE0_OUT / "PB1DS_PB3_PRE0_SUPPORT_SEED_MANIFEST.jsonl"
    return pd.read_json(p, lines=True)


# feature order + encoding geometry (frozen canonical)
FEATURE_COLUMNS: List[str] = list(M0.C0_FEATURE_COLUMNS)
CATS_SORTED = sorted(C.MVTEC_CATEGORIES)
CAT_REF = CATS_SORTED[0]
CAT_ONEHOT_COLS = [c for c in CATS_SORTED if c != CAT_REF]  # 7 indicator columns
DET_PATCHCORE = "wide_resnet50_2"


def cat_onehot_vec(category: str) -> np.ndarray:
    return np.array([1.0 if category == c else 0.0 for c in CAT_ONEHOT_COLS], dtype=float)


def det_vec(detector: str) -> float:
    return 1.0 if detector == DET_PATCHCORE else 0.0


def encode_X(candidate: str, z10: Sequence[float], category: str, detector: str) -> np.ndarray:
    """Frozen encoding for the given candidate; A/C take Z, D never does."""
    z = np.asarray([float(x) for x in z10], dtype=float)
    detv = det_vec(detector)
    if candidate == "D_CATEGORY_ONLY":
        return np.r_[cat_onehot_vec(category), detv]
    if candidate == "A_GLOBAL":
        return np.r_[z, detv]
    if candidate == "C_CATEGORY_CONDITIONED":
        return np.r_[z, cat_onehot_vec(category), detv]
    raise ValueError(candidate)


def sigmoid(z: float) -> float:
    z = float(np.clip(z, -500.0, 500.0))
    return 1.0 / (1.0 + np.exp(-z))


def forward_prob(
    candidate: str, z10: Sequence[float], category: str, detector: str, model: Dict[str, Any]
) -> float:
    """FULLY FROZEN forward: sigmoid(X @ coef + intercept). No fit, no refit."""
    X = encode_X(candidate, z10, category, detector)
    return float(
        sigmoid(float(np.dot(X, np.asarray(model["coef"], dtype=float)) + model["intercept"]))
    )


def action_from_prob(prob: float, model: Dict[str, Any]) -> int:
    if model["no_valid"] or model["threshold"] is None:
        return 0
    return int(prob > float(model["threshold"]))


def entity_parts(entity_id: str) -> Tuple[str, str, str]:
    """cat|cond|det -> (category, condition, detector). Frozen PB1DS identity."""
    return B3.entity_parts(entity_id)


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def sha256_arr(a: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()
    ).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, rel: str) -> Path:
    p = PB3R_OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def write_json(obj: Any, rel: str) -> Path:
    p = PB3R_OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p
