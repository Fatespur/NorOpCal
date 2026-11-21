"""PB1-DS PB4 shared library — Target-Normal Contamination Robustness.

PB4 evaluates the PB3-R sealed *frozen* selective-recalibration pipeline under
contaminated target-normal support. It is a robustness evaluation only:

    S_rho -> Z_rho -> h^frozen -> t^frozen -> action_rho -> QM_rho -> downstream

No method / selector / feature / threshold / QM / benefit / baseline / K /
seed is refit, retuned, extended, or redesigned. All frozen parameters are read
verbatim from the sealed C1/PUB0 outer seals (the same binding PB3-R wrote) and
cross-checked against the sealed PB1DS_PB3R_FROZEN_MODEL_BINDING.json.

Frozen contamination protocol (recovered verbatim from the sealed
PB1DS_PB1_CONTAMINATION_PROTOCOL.json; documented in PB4-R0):
    grid       : {0, 5, 10, 20}%; 20 repeats; 0% = clean control
    injection  : n_cont = ceil(rho * K) anomalies drawn from the entity's own
                 15-sample regular-defect set, INJECTED into the target-normal
                 support pool (NOT replacement). Effective rate =
                 n_cont / (K + n_cont), reported alongside nominal rho.
    scope      : PB1 froze "primary at K=FULL, plus each feasible K cell".
                 PB4 therefore executes {K=4, K=8, K=FULL}; K=8 is reported as
                 the publication-primary practical operating point (PB3-R).
    seed       : PB1 sealed the DETERMINISTIC seed-generation rule
                 (canonical string "PB1|K|<K_value>|cat|cond|det|repeat"); no
                 per-repeat contamination manifest was sealed, so the rule is
                 strictly rebuilt (K_value slot = "RHO<rho_post_round>") and the
                 identity + hash are frozen into PB4-R0.

Repeat binding (key integrity decision, R0-documented): contamination repeat
r in 1..20 is PAIRED with the sealed K-support repeat r from
PB1DS_PB3_PRE0_SUPPORT_SEED_MANIFEST.jsonl (support_ids reused verbatim, no
re-roll). Hence rho=0 reproduces the PB3-R per-cell at the same repeat exactly,
which makes the R1 clean-reproduction gate a hard, cell-level gate. At K=FULL
the support is deterministic and contamination only injects; each repeat r is
then an independent contaminant draw.

Selector semantics:
    A_GLOBAL            p = h_A(Z_rho)      (contaminated Z, frozen A model)
    C_CATEGORY_COND     p = h_C(Z_rho, c, d)(contaminated Z, frozen C model)
    D_CATEGORY_ONLY     p = h_D(c, d)       (feature-free -> contamination-invariant
                                             by construction; QM still contaminated)
    BC_TARGET_NORMAL_THRESHOLD_FAMILY        canonical family; BC_Q95 operand
                                             recomputed from the contaminated support
    ALWAYS_ON_QM                             unconditional QM on contaminated support
    NO_ADAPT                                 no adaptation -> contamination-invariant
Publication lock: the BC family canonical representative is frozen; BC_RANK /
BC_QUANTILE are NOT separate baselines.
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
import pb1ds_pb1_common as P
import pb1ds_pb1f_common as F
import pb1ds_pb2_run as B2R
import pb1ds_pb3_common as B3
import pb1ds_pb3r_common as R

PB4_OUT = Path("reports/pb1ds_publication/pb4_contamination")

# Inherited frozen upstream terminal (PB3-R, verified read-only in R0)
PB3R_TERMINAL = R.TERM_E2E_CONFIRMED

# --------------------------------------------------------------------------- #
# Frozen contamination protocol (recovered from PB1 sealed protocol, verbatim)
# --------------------------------------------------------------------------- #
RHO_GRID = [0.0, 0.05, 0.10, 0.20]
CONTAM_REPEATS = 20
ANOM_POOL_SIZE = 15  # entity's own 15-sample regular-defect set (verified)
CONTAM_SCOPE = "primary_at_FULL_plus_each_feasible_K_cell"  # PB1 sealed
K_GRID_PB4 = ["4", "8", "FULL"]
PRIMARY_K = "8"  # publication-primary practical operating point (PB3-R)

# Released allowed scientific terminal states (PB4 §34)
TERM_A = "PB1DS_PB4_COMPLETE_SELECTIVE_RECALIBRATION_ROBUST_TO_PREREGISTERED_CONTAMINATION_RANGE"
TERM_B = "PB1DS_PB4_COMPLETE_SELECTIVE_RECALIBRATION_ROBUST_WITH_FINITE_CONTAMINATION_TOLERANCE"
TERM_C = "PB1DS_PB4_COMPLETE_CONTAMINATION_PRESERVES_UTILITY_BUT_DEGRADES_INTERVENTION_SAFETY"
TERM_D = (
    "PB1DS_PB4_COMPLETE_CONTAMINATION_DEGRADES_UTILITY_WITH_SELECTIVE_SAFETY_ADVANTAGE_PRESERVED"
)
TERM_E = (
    "PB1DS_PB4_COMPLETE_DEFORMATION_REFINEMENT_ADDS_CONTAMINATION_SENSITIVITY_BEYOND_CATEGORY_PRIOR"
)
TERM_F = (
    "PB1DS_PB4_COMPLETE_TARGET_SUPPORT_CONTAMINATION_BREAKS_SELECTIVE_RECALIBRATION_RELIABILITY"
)

# Integrity STOP states (allowed, never a scientific FAIL)
STOP_PROTOCOL = "PB1DS_PB4_STOP_PROTOCOL_NOT_UNIQUELY_RECOVERABLE"
STOP_BINDING = "PB1DS_PB4_STOP_FROZEN_POLICY_BINDING_FAILURE"
STOP_REPRO = "PB1DS_PB4_STOP_CLEAN_REPRODUCTION_FAILURE"
STOP_SEED = "PB1DS_PB4_STOP_SEED_BINDING_FAILURE"
STOP_CONTAM = "PB1DS_PB4_STOP_CONTAMINATION_CONSTRUCTION_FAILURE"
STOP_PROP = "PB1DS_PB4_STOP_CONTAMINATION_PROPAGATION_INTEGRITY_FAILURE"
STOP_STAT = "PB1DS_PB4_STOP_STATISTICAL_EXECUTION_INCOMPLETE"

VALID_TERMINALS = {TERM_A, TERM_B, TERM_C, TERM_D, TERM_E, TERM_F}
VALID_STOPS = {
    STOP_PROTOCOL,
    STOP_BINDING,
    STOP_REPRO,
    STOP_SEED,
    STOP_CONTAM,
    STOP_PROP,
    STOP_STAT,
}

# Frozen methods (publication-locked; PB3, unchanged)
METHODS = [
    "NO_ADAPT",
    "ALWAYS_ON_QM",
    B3.BC_FAMILY_CANONICAL,
    "D_CATEGORY_ONLY",
    "C_CATEGORY_CONDITIONED",
    "A_GLOBAL",
]
Z_SELECTORS = ["C_CATEGORY_CONDITIONED", "A_GLOBAL"]  # Z-recomputed selectors
SELECTOR_METHODS = ["D_CATEGORY_ONLY", "C_CATEGORY_CONDITIONED", "A_GLOBAL"]
QM_SELECTORS = ["D_CATEGORY_ONLY", "C_CATEGORY_CONDITIONED", "A_GLOBAL"]

_TRANSFORM = {
    "NO_ADAPT": ("NONE", False),
    "ALWAYS_ON_QM": ("QM", False),
    B3.BC_FAMILY_CANONICAL: ("RANK", False),
    "D_CATEGORY_ONLY": ("QM", True),
    "C_CATEGORY_CONDITIONED": ("QM", True),
    "A_GLOBAL": ("QM", True),
}

# Material degradation interpretation (frozen, result-independent; PB1 catast.
# alerts: Delta_BA <= -2pp, Global Harm >= 10%)
MATERIAL_DBA_PP = 2.0
MATERIAL_HARM = 0.02
CATASTROPHIC_DBA_PP = -2.0
CATASTROPHIC_HARM = 0.10
# Empirical contamination robustness boundary decision thresholds (publication
# -facing, frozen interpretation, NOT tuning knobs):
BOUNDARY_UTIL_CATASTROPHIC_PP = -2.0  # mean Delta_BA drop below -2pp => material loss
BOUNDARY_UTIL_MATERIAL_PP = -1.0  # mean Delta_BA drop beyond -1pp => material
BOUNDARY_HARM_CATASTROPHIC = 0.10  # Global Harm >= 10% => catastrophic


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
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


def write_json(obj: Any, rel: str) -> Path:
    p = PB4_OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p


def write_csv(df: pd.DataFrame, rel: str) -> Path:
    p = PB4_OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def write_md(text: str, rel: str) -> Path:
    p = PB4_OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# Frozen model binding (identical rebuild + hash compare to PB3-R)
# --------------------------------------------------------------------------- #
def build_model_binding() -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Rebuild the frozen A/C/D binding from sealed seals and cross-check the
    coefficient / threshold / intercept hashes against the sealed PB3-R file."""
    c1_models = R.load_c1_models()
    d_models = R.load_d_models()
    fold_via_model = _fold_via_models()
    oof = pd.read_csv(F.C1_OUT / "PB1DS_C1_OOF_PER_UNIT.csv")
    oof = oof[oof["candidate"] == "C_CATEGORY_CONDITIONED"]
    fold_via_oof = {
        f"{r['category']}|{r['condition']}": int(r["outer_fold_index"]) for _, r in oof.iterrows()
    }
    entity_fold: Dict[str, int] = {
        f"{r['category']}|{r['condition']}|{r['baseline']}": int(r["outer_fold_index"])
        for _, r in oof.iterrows()
    }
    domain = sorted(set(fold_via_model) | set(fold_via_oof))
    fold_mismatch = [d for d in domain if fold_via_model.get(d) != fold_via_oof.get(d)]

    binding: Dict[str, Any] = {"A_GLOBAL": {}, "C_CATEGORY_CONDITIONED": {}, "D_CATEGORY_ONLY": {}}
    for fold, mod in sorted(c1_models.items()):
        for cand in ("A_GLOBAL", "C_CATEGORY_CONDITIONED"):
            m = mod[cand]
            binding[cand][str(fold)] = {
                "fold_index": fold,
                "coef": m["coef"],
                "intercept": m["intercept"],
                "operating_threshold": m["threshold"],
                "selected_C": m["selected_C"],
                "no_valid_operating_point": m["no_valid"],
                "coef_sha256": sha256_arr(np.asarray(m["coef"], dtype=np.float64)),
                "feature_order_if_z": list(M0.C0_FEATURE_COLUMNS),
                "x_dim": len(m["coef"]),
            }
    for fold, mod in sorted(d_models.items()):
        binding["D_CATEGORY_ONLY"][str(fold)] = {
            "fold_index": fold,
            "coef": mod["coef"],
            "intercept": mod["intercept"],
            "operating_threshold": mod["threshold"],
            "selected_C": mod["selected_C"],
            "no_valid_operating_point": mod["no_valid"],
            "coef_sha256": sha256_arr(np.asarray(mod["coef"], dtype=np.float64)),
            "feature_order_if_z": None,
            "x_dim": len(mod["coef"]),
        }
    binding["_meta"] = {
        "source_c1_seals": "reports/pb1ds_publication/c1_category_conditioned/PB1DS_C1_OUTER_SEALS.jsonl",
        "source_pub0_seals": "reports/pb1ds_publication/pub0_attribution_freeze/PB1DS_PUB0_OUTER_SEALS.jsonl",
        "c1_seals_sha256": sha256_file(F.C1_OUT / "PB1DS_C1_OUTER_SEALS.jsonl"),
        "pub0_seals_sha256": sha256_file(F.PUB0_OUT / "PB1DS_PUB0_OUTER_SEALS.jsonl"),
        "fold_domain_consistent": not fold_mismatch,
        "fold_mismatches": fold_mismatch,
        "n_folds_c1": len(c1_models),
        "n_folds_pub0": len(d_models),
        "n_entity_fold_mappings": len(entity_fold),
    }
    return binding, entity_fold


def _fold_via_models() -> Dict[str, int]:
    m: Dict[str, int] = {}
    for line in (F.C1_OUT / "PB1DS_C1_OUTER_SEALS.jsonl").open(encoding="utf-8"):
        s = json.loads(line)
        m[f"{s['outer_test_category']}|{s['outer_test_condition']}"] = int(s["fold_index"])
    return m


def binding_matches_pb3r(binding: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Compare every fold's coef hash + threshold + intercept to PB3-R sealed file."""
    prev = json.loads(
        (R.PB3R_OUT / "PB1DS_PB3R_FROZEN_MODEL_BINDING.json").read_text(encoding="utf-8")
    )
    mism: List[str] = []
    for cand in ("A_GLOBAL", "C_CATEGORY_CONDITIONED", "D_CATEGORY_ONLY"):
        for fold_s, b in binding[cand].items():
            pb = prev.get(cand, {}).get(fold_s)
            if pb is None:
                mism.append(f"{cand} fold {fold_s}: missing in PB3R binding")
                continue
            if b["coef_sha256"] != pb.get("coef_sha256"):
                mism.append(f"{cand} fold {fold_s}: coef hash differs")
            if abs(float(b["operating_threshold"]) - float(pb["operating_threshold"])) > 1e-12:
                mism.append(f"{cand} fold {fold_s}: threshold differs")
            if abs(float(b["intercept"]) - float(pb["intercept"])) > 1e-12:
                mism.append(f"{cand} fold {fold_s}: intercept differs")
    return (not mism), mism


# --------------------------------------------------------------------------- #
# Frozen selector forward (identical to PB3-R)
# --------------------------------------------------------------------------- #
def forward(method, z10, cat, det, fold, c1_models, d_models) -> Tuple[float, int]:
    if method == "D_CATEGORY_ONLY":
        mod = d_models[fold]
        prob = R.forward_prob("D_CATEGORY_ONLY", [], cat, det, mod)
    else:
        mod = c1_models[fold][method]
        prob = R.forward_prob(method, z10, cat, det, mod)
    return float(prob), R.action_from_prob(prob, mod)


# --------------------------------------------------------------------------- #
# Contamination seed rule (frozen canonical reconstruction, R0-documented)
# --------------------------------------------------------------------------- #
def contam_identity(
    rho_value: float, category: str, condition: str, detector: str, repeat: int
) -> str:
    """Contaminant-draw seed identity.

    Reconstructs the PB1 canonical SHA256 seed rule (P.seed_identity) with the
    per-cell ``k_value`` slot holding the frozen contamination level in exact
    POST-ROUND form ("RHO0.05", "RHO0.10", "RHO0.20"). Deterministic and
    result-independent; the identity string and its hash are frozen into R0.
    """
    return P.seed_identity(f"RHO{rho_value:.2f}", category, condition, detector, repeat)


def contam_seed(rho_value: float, category: str, condition: str, detector: str, repeat: int) -> int:
    return int(
        sha256_text(contam_identity(rho_value, category, condition, detector, repeat))[:8], 16
    )


def draw_contaminant_ids(
    category: str, condition: str, detector: str, rho_value: float, repeat: int, n_cont: int
) -> List[int]:
    """Deterministic distinct contaminant indices into the entity's own 15-anomaly
    pool. One permutation per (entity,rho,repeat), sliced by ceil(rho*K); hence for
    a fixed (entity,rho,repeat) the smaller-K contaminant sets are subsets of the
    larger-K sets (deterministic nested draw)."""
    rng = np.random.default_rng(contam_seed(rho_value, category, condition, detector, repeat))
    perm = rng.permutation(ANOM_POOL_SIZE)
    return sorted(int(x) for x in perm[:n_cont])


def contam_count(k_value: str, rho_value: float, full_n: int) -> int:
    """n_cont = ceil(rho * K); K in {4,8,FULL} where FULL uses the entity's FULL_n."""
    K = full_n if k_value == "FULL" else int(k_value)
    return int(np.ceil(rho_value * K))


def effective_rate(k_value: str, rho_value: float, full_n: int) -> float:
    K = full_n if k_value == "FULL" else int(k_value)
    nc = contam_count(k_value, rho_value, full_n)
    return nc / (K + nc) if (K + nc) > 0 else 0.0


# --------------------------------------------------------------------------- #
# Contaminated support construction + propagation
# --------------------------------------------------------------------------- #
def k_support_idx(entity_id: str, k_value: str, repeat: int) -> np.ndarray:
    """K-sampled target-normal index array, paired with contamination repeat r.

    Rereads the sealed PRE0 support seed manifest semantics (identical to
    PB3-R): NESTED S4 ⊂ S8 ⊆ SFULL from one base permutation per (entity,repeat),
    FULL = all normals. repeat is the contamination repeat (1..20) that equals
    the K-support repeat. This makes rho=0 reproduce the PB3-R cell exactly."""
    cat, cond, det = R.entity_parts(entity_id)
    if k_value == "FULL":
        return np.arange(int(B3.full_n_map()[entity_id]))
    return np.asarray(B3.support_indices(entity_id, k_value, repeat, cat, cond, det), dtype=int)


def build_contaminated_support(
    entity_id: str,
    k_value: str,
    rho_value: float,
    repeat: int,
    cnd: np.ndarray,
    anm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (support_scores_rho, normal_idx, contam_idx, n_cont).

    INJECTION semantics (PB1 sealed): keep all target-normals, ADD n_cont =
    ceil(rho*K) anomalies drawn from the entity's own anomaly pool. support_scores
    = cnd[normal_idx] ++ anm[contam_idx]. At rho=0 n_cont=0 and support_scores is
    exactly the clean K-support. n_cont <= ANOM_POOL_SIZE enforced.
    """
    full_n = int(len(cnd))
    normal_idx = k_support_idx(entity_id, k_value, repeat)
    n_cont = contam_count(k_value, rho_value, full_n)
    if n_cont == 0:
        return cnd[normal_idx], normal_idx, np.asarray([], dtype=int), 0
    if n_cont > ANOM_POOL_SIZE:
        raise ValueError(f"n_cont {n_cont} exceeds anomaly pool {ANOM_POOL_SIZE}")
    cat, cond, det = R.entity_parts(entity_id)
    contam_idx = np.asarray(
        draw_contaminant_ids(cat, cond, det, rho_value, repeat, n_cont), dtype=int
    )
    support = np.concatenate([cnd[normal_idx], anm[contam_idx]])
    return support, normal_idx, contam_idx, n_cont


def compute_z_rho(src: np.ndarray, support_scores_rho: np.ndarray) -> Dict[str, float]:
    """Recompute ALL 10 canonical deformation features from the CONTAMINATED
    support. 10/10 features recomputed — no clean-Z reuse, no interpolation."""
    if len(support_scores_rho) == 0:
        raise ValueError("empty contaminated support")
    ftr = M0.compute_c0_features(src, support_scores_rho)
    return {k: float(ftr[k]) for k in M0.C0_FEATURE_COLUMNS}


def exec_cell(
    entity_id: str,
    cat: str,
    cond: str,
    det: str,
    k_value: str,
    rho_value: float,
    repeat: int,
    support_scores_rho: np.ndarray,
    normal_idx: np.ndarray,
    contam_idx: np.ndarray,
    n_cont: int,
    z_rho: Dict[str, float],
    cnd: np.ndarray,
    anm: np.ndarray,
    src: np.ndarray,
    tau: float,
    benefit_label: int,
    fold: int,
    c1_models,
    d_models,
) -> List[Dict[str, Any]]:
    """Full contamination propagation for a (entity,K,rho,repeat) cell:
    S_rho -> Z_rho -> frozen selector -> p_rho -> frozen threshold ->
    action_rho -> QM_rho -> downstream metrics. Each method returns one row."""
    labels = np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))])
    before = C.detection_metrics(np.concatenate([cnd, anm]), labels, tau)
    z10 = [z_rho[f] for f in M0.C0_FEATURE_COLUMNS]
    out: List[Dict[str, Any]] = []
    for method in METHODS:
        aft, is_sel = _TRANSFORM[method]
        if is_sel:
            prob, ad = forward(method, z10, cat, det, fold, c1_models, d_models)
        else:
            prob, ad = float("nan"), (0 if method == "NO_ADAPT" else 1)
        if method == "NO_ADAPT":
            after, tau_used = before, tau
        else:
            r_ = B3.compute_unit_at_k(entity_id, cnd, anm, src, tau, support_scores_rho, aft)
            after, tau_used = r_["after"], r_["tau_used"]
        used_ba = after["ba"] if ad == 1 else before["ba"]
        used_fpr = after["fpr"] if ad == 1 else before["fpr"]
        used_fnr = after["fnr"] if ad == 1 else before["fnr"]
        used_f1 = after["f1"] if ad == 1 else before["f1"]
        dba = used_ba - before["ba"]
        dfpr = used_fpr - before["fpr"]
        dfnr = used_fnr - before["fnr"]
        _, harm, neutral = M0.classify_change(dba, dfpr, dfnr)
        row: Dict[str, Any] = {
            "entity_id": entity_id,
            "category": cat,
            "condition": cond,
            "detector": det,
            "K": k_value,
            "FULL_n": int(len(cnd)),
            "rho": rho_value,
            "repeat_id": repeat,
            "n_cont": n_cont,
            "contam_ids": ";".join(str(int(i)) for i in contam_idx),
            "method": method,
            "selector_prob": prob,
            "action": ad,
            "tau_used": float(tau_used),
            "fold_index": fold,
            "ba_before": before["ba"],
            "ba_after": after["ba"],
            "fpr_before": before["fpr"],
            "fpr_after": after["fpr"],
            "fnr_before": before["fnr"],
            "fnr_after": after["fnr"],
            "f1_before": before["f1"],
            "f1_after": after["f1"],
            "used_ba": used_ba,
            "used_fpr": used_fpr,
            "used_fnr": used_fnr,
            "used_f1": used_f1,
            "delta_ba": dba,
            "delta_fpr": dfpr,
            "delta_fnr": dfnr,
            "benefit_label": benefit_label,
            "harm": harm,
            "neutral": neutral,
        }
        if method in ("C_CATEGORY_CONDITIONED", "A_GLOBAL", "D_CATEGORY_ONLY"):
            row.update({f"z_{f}": z_rho[f] for f in M0.C0_FEATURE_COLUMNS})
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# per-(entity,K,rho) frame (repeat-aggregated) + frozen frame
# --------------------------------------------------------------------------- #
def per_entity_frame(per_repeat: pd.DataFrame) -> pd.DataFrame:
    if per_repeat.empty:
        return pd.DataFrame()
    grp = per_repeat.groupby(["entity_id", "K", "rho", "method"], as_index=False)
    g = grp.agg(
        category=("category", "first"),
        condition=("condition", "first"),
        detector=("detector", "first"),
        FULL_n=("FULL_n", "first"),
        fold_index=("fold_index", "first"),
        adapt=("action", "mean"),
        harm_r=("harm", "mean"),
        neutral_r=("neutral", "mean"),
        selector_prob=("selector_prob", "mean"),
        used_ba=("used_ba", "mean"),
        used_fpr=("used_fpr", "mean"),
        used_fnr=("used_fnr", "mean"),
        used_f1=("used_f1", "mean"),
        ba_before=("ba_before", "first"),
        fpr_before=("fpr_before", "first"),
        fnr_before=("fnr_before", "first"),
        f1_before=("f1_before", "first"),
        ba_after=("ba_after", "mean"),
        fpr_after=("fpr_after", "mean"),
        fnr_after=("fnr_after", "mean"),
        f1_after=("f1_after", "mean"),
        benefit_label=("benefit_label", "first"),
        delta_ba=("delta_ba", "mean"),
        delta_fpr=("delta_fpr", "mean"),
        delta_fnr=("delta_fnr", "mean"),
        prob_sd=("selector_prob", "std"),
    ).reset_index(drop=True)
    g["used_ba"] = np.where(
        g["adapt"].to_numpy() > 0.5, g["ba_after"].to_numpy(), g["ba_before"].to_numpy()
    )
    g["harm"] = np.where(g["adapt"].to_numpy() > 0.5, g["harm_r"].to_numpy(), 0.0)
    g["neutral"] = np.where(
        g["adapt"].to_numpy() > 0.5, g["neutral_r"].to_numpy(), 1.0 - g["neutral_r"].to_numpy()
    )
    g["benefit"] = g["benefit_label"]
    g["adapt_bin"] = (g["adapt"].to_numpy() > 0.5).astype(int)
    return g


def frozen_frame(g: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": g["method"],
            "category": g["category"],
            "condition": g["condition"],
            "baseline": g["detector"],
            "adapt": g["adapt_bin"],
            "benefit": g["benefit"],
            "harm": g["harm"],
            "neutral": g["neutral"],
            "selector_prob": g["selector_prob"],
            "ba_before": g["ba_before"],
            "ba_after": g["used_ba"],
            "fpr_before": g["fpr_before"],
            "fpr_after": g["used_fpr"],
            "fnr_before": g["fnr_before"],
            "fnr_after": g["used_fnr"],
            "f1_before": g["f1_before"],
            "f1_after": g["used_f1"],
            "entity_id": g["entity_id"],
            "FULL_n": g["FULL_n"],
            "delta_ba": g["delta_ba"],
            "delta_fpr": g["delta_fpr"],
            "delta_fnr": g["delta_fnr"],
        }
    )


def frame_for(
    g: pd.DataFrame, method: str, K: str, rho: float, entities: Sequence[str]
) -> pd.DataFrame:
    sub = g[
        (g["method"] == method)
        & (g["K"] == K)
        & (np.isclose(g["rho"], rho))
        & (g["entity_id"].isin(entities))
    ]
    sub = sub.set_index("entity_id").loc[list(entities)].reset_index()
    return frozen_frame(sub)


def aggregate_for(
    g: pd.DataFrame, method: str, K: str, rho: float, entities: Sequence[str]
) -> Dict[str, float]:
    return B2R.aggregate(frame_for(g, method, K, rho, entities))


def e78_entities() -> List[str]:
    return sorted(B3.list_entities())


def e40_entities() -> List[str]:
    return sorted(e for e in B3.list_entities() if int(B3.full_n_map()[e]) >= 8)
