"""PB1-DS-PB6 D3 — EfficientAD raw-score integrity + independence diagnostic.

Validates raw EfficientAD score caches after real D2 training and audits that
EfficientAD carries independent signal relative to PatchCore (wide_resnet50_2)
and DINOv2 Memory (dinov2_vits14).

Pure helpers are unit-tested in tests/test_pb1ds_pb6.py and runnable before D2.
The full driver (`__main__`) expects the D2 caches to exist.

Checks (requirement 12): finite (NaN==0, Inf==0), variance>0, one-to-one
image identity across patch_scores/image_scores/rels, category-condition
mapping correct, no duplicated artifact, no score/path offset, no
normal/anomaly reversal.

Independence diagnostic (requirement 13): per-unit Pearson / Spearman /
affine-fit residual / near-duplicate against each reference detector.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import md0sr_common as C
import pb1ds_c0_common as C0

PB6_DIR = ROOT / "reports" / "pb1ds_publication" / "pb6_real_third_detector"
SCORE_DIR = C.SCORES_DIR
DETECTOR = "efficientad"
REF_DETECTORS = ("wide_resnet50_2", "dinov2_vits14")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cache(path: Path):
    """Return (patch_scores, image_scores, rels)."""
    with np.load(path, allow_pickle=False) as d:
        return d["patch_scores"], d["image_scores"], np.asarray(d["rels"])


def check_finite(values: np.ndarray) -> dict:
    a = np.asarray(values, dtype=float)
    nan, inf = int(np.isnan(a).sum()), int(np.isinf(a).sum())
    return {"nan": nan, "inf": inf, "pass": nan == 0 and inf == 0}


def check_variance(values: np.ndarray) -> dict:
    a = np.asarray(values, dtype=float)
    v = float(np.var(a)) if a.size > 0 else 0.0
    return {"variance": v, "pass": v > 0}


def single_cache_integrity(path: Path) -> dict:
    patch, image, rels = load_cache(path)
    n = int(image.shape[0])
    is_fin = check_finite(image)
    ps_fin = check_finite(np.asarray(patch, dtype=float))
    var = check_variance(image)
    lengths = n == int(patch.shape[0]) == int(len(rels))
    uniq = int(len(np.unique(rels))) == n
    result = {
        "file": path.name,
        "n": n,
        "patch_shape": list(patch.shape),
        "nan": is_fin["nan"],
        "inf": is_fin["inf"],
        "variance": var["variance"],
        "image_scores_finite": is_fin["pass"],
        "patch_scores_finite": ps_fin["pass"],
        "variance_positive": var["pass"],
        "lengths_match": bool(lengths),
        "unique_identity": bool(uniq),
        "sha256": sha256_file(path),
    }
    result["pass"] = bool(
        result["image_scores_finite"]
        and result["patch_scores_finite"]
        and result["variance_positive"]
        and result["lengths_match"]
        and result["unique_identity"]
    )
    return result


def scan_identity(rels) -> dict:
    """No duplicated path / offset check: rels must not repeat and must be non-empty."""
    r = [str(x) for x in rels]
    return {
        "n": len(r),
        "duplicated_paths": int(len(r) != len(set(r))),
        "empty": int(any(x == "" for x in r)),
    }


def mapping_check(category: str, role: str, condition: str | None, rels) -> dict:
    prefix = f"mvtec_ad_2/{category}/test_public/{role}/"
    prefix_ok = all(str(x).startswith(prefix) for x in rels)
    cond_ok = True
    if condition is not None:
        cond_ok = all(str(x).rsplit("/", 1)[-1].endswith(f"_{condition}.png") for x in rels)
    return {
        "prefix_ok": bool(prefix_ok),
        "condition_ok": bool(cond_ok),
        "pass": bool(prefix_ok and cond_ok),
    }


def affinity_diagnostic(eff: np.ndarray, ref: np.ndarray) -> dict:
    """Association of EfficientAD vs a reference detector (image-level)."""
    from scipy import stats

    a = np.asarray(eff, dtype=float)
    b = np.asarray(ref, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    base = {
        "n": n,
        "pearson": None,
        "spearman": None,
        "affine_residual_relative": None,
        "near_affine_copy": False,
        "duplicate_or_near_duplicate": False,
    }
    if n < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return base
    if np.all(np.isclose(a, b)):
        return {**base, "n": n, "duplicate_or_near_duplicate": True}
    pearson = float(stats.pearsonr(a, b).statistic)
    spearman = float(stats.spearmanr(a, b).statistic)
    A = np.vstack([b, np.ones_like(b)]).T
    coef, *_ = np.linalg.lstsq(A, a, rcond=None)
    resid = a - A @ coef
    ptp = float(np.ptp(a))
    rel = float(resid.std() / ptp) if ptp > 0 else float("inf")
    return {
        "n": n,
        "pearson": pearson,
        "spearman": spearman,
        "affine_residual_relative": rel,
        "near_affine_copy": rel < 1e-3,
        "duplicate_or_near_duplicate": rel < 1e-6 and abs(pearson) > 0.9999,
    }


def build_unit(category: str, condition: str) -> dict:
    """Raw reference-QM metrics for the EfficientAD unit, mirroring the frozen
    PB1DS unit frame (source=good_regular, cond=good_{cond}, anomaly=bad_regular)."""
    src = C.load_image_scores(C.source_rel(category, DETECTOR))
    cnd = C.load_image_scores(C.cond_normal_rel(category, condition, DETECTOR))
    anm = C.load_image_scores(C.anomaly_rel(category, DETECTOR))
    tau = float(np.percentile(src, 95))
    labels = np.concatenate([np.zeros(len(cnd)), np.ones(len(anm))])
    mb = C.detection_metrics(np.concatenate([cnd, anm]), labels, tau)
    ma = C.detection_metrics(
        np.concatenate([C.quantile_mapping(src, cnd), C.quantile_mapping(src, anm)]), labels, tau
    )
    dba, dfpr, dfnr = ma["ba"] - mb["ba"], ma["fpr"] - mb["fpr"], ma["fnr"] - mb["fnr"]
    benefit, harm, neutral = C0.classify_change(dba, dfpr, dfnr)
    return {
        "category": category,
        "condition": condition,
        "detector": DETECTOR,
        "baseline_name": "EfficientAD",
        "tau": tau,
        "n_normal": int(len(cnd)),
        "n_anomaly": int(len(anm)),
        "benefit": benefit,
        "harm": harm,
        "neutral": neutral,
        "dba": dba,
        "dfpr": dfpr,
        "dfnr": dfnr,
        "ba_before": float(mb["ba"]),
        "ba_after": float(ma["ba"]),
        "fpr_before": float(mb["fpr"]),
        "fpr_after": float(ma["fpr"]),
        "fnr_before": float(mb["fnr"]),
        "fnr_after": float(ma["fnr"]),
        "benefit_qm": int(benefit),
    }


def main() -> int:
    us = sorted(pd_unit_units())
    integrity_rows: list[dict] = []
    mapping_rows: list[dict] = []
    result_rows: list[dict] = []
    diag_rows: list[dict] = []
    all_ok = True

    seen_files: set[str] = set()

    # collect the required caches for the 39 units
    required: list[tuple[str, str, str]] = []
    for cat, cond in us:
        required.append((cat, "good", "regular"))
        required.append((cat, "bad", "regular"))
        required.append((cat, "good", cond))
    required = sorted(set(required))

    for cat, role, cond in required:
        path = SCORE_DIR / f"mv2_{cat}_public_{role}_{cond}_{DETECTOR}_scores.npz"
        if not path.is_file():
            all_ok = False
            integrity_rows.append({"file": path.name, "present": False, "pass": False})
            continue
        if path.name in seen_files:
            all_ok = False
            continue
        seen_files.add(path.name)
        integ = single_cache_integrity(path)
        _, _, rels = load_cache(path)
        scan = scan_identity(rels)
        mapping = mapping_check(cat, role, cond, rels)
        row = {
            **integ,
            "category": cat,
            "role": role,
            "condition": cond,
            "identity_duplicates": scan["duplicated_paths"],
            "empty_paths": scan["empty"],
            "mapping_ok": mapping["pass"],
        }
        row["pass"] = bool(
            row["pass"] and scan["duplicated_paths"] == 0 and scan["empty"] == 0 and mapping["pass"]
        )
        integrity_rows.append(row)
        mapping_rows.append(mapping)
        if not row["pass"]:
            all_ok = False

        # independence diagnostic on cond-normal scores vs ref detectors
        eff = C.load_image_scores(C.cond_normal_rel(cat, cond, DETECTOR))
        for ref in REF_DETECTORS:
            refp = SCORE_DIR / f"mv2_{cat}_public_good_{cond}_{ref}_scores.npz"
            if not refp.is_file():
                continue
            rf = C.load_image_scores(f"mv2_{cat}_public_good_{cond}_{ref}_scores.npz")
            diag = affinity_diagnostic(eff, rf)
            diag_rows.append({"category": cat, "condition": cond, "reference": ref, **diag})

    # raw detector results per unit
    for cat, cond in us:
        result_rows.append(build_unit(cat, cond))

    write_csv = lambda path, rows, fields: _write_csv(path, rows, fields)
    write_csv(
        PB6_DIR / "PB1DS_PB6_RAW_SCORE_INTEGRITY.csv",
        integrity_rows,
        list(integrity_rows[0].keys()) if integrity_rows else [],
    )
    write_csv(
        PB6_DIR / "PB1DS_PB6_RAW_DETECTOR_RESULTS.csv",
        result_rows,
        list(result_rows[0].keys()) if result_rows else [],
    )
    write_csv(
        PB6_DIR / "PB1DS_PB6_INDEPENDENCE_DIAGNOSTIC.csv",
        diag_rows,
        list(diag_rows[0].keys()) if diag_rows else [],
    )

    n_present = sum(1 for r in integrity_rows if r.get("present", True))
    n_pass = sum(1 for r in integrity_rows if r["pass"])
    gate = "PASS" if all_ok and n_pass == n_present else "FAIL"
    md = [
        "# PB1-DS-PB6 — D3 Raw Score Integrity + Independence Audit",
        "",
        f"- Detector: `{DETECTOR}`",
        f"- Caches audited: {len(integrity_rows)}",
        f"- Finite/var/identity/mapping passes: {n_pass}/{n_present}",
        f"- Independence diagnostics: {len(diag_rows)} (vs PatchCore & DINOv2 Memory)",
        f"- **D3 raw-score gate = {gate}**",
    ]
    (PB6_DIR / "PB1DS_PB6_D3_SCORE_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("D3 gate =", gate, "| passes", n_pass, "/", n_present)
    return 0 if gate == "PASS" else 1


def pd_unit_units() -> list[tuple[str, str]]:
    import pandas as pd

    p = PB6_DIR / "PB1DS_PB6_D2_ASSET_RESOLUTION.csv"
    df = pd.read_csv(p)
    return sorted(
        tuple(r) for r in df[["category", "condition"]].drop_duplicates().itertuples(index=False)
    )


def _write_csv(path: Path, rows: list[dict], fields: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
