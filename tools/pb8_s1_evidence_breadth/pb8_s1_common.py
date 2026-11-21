"""PB8-S1 shared library — AD2 K8 evidence-breadth supplementary validation.

This stage is SUPPLEMENTARY EVIDENCE-BREADTH VALIDATION only. It never modifies
the frozen NorOpCal method or the frozen PB8 principal result.

Phase-I (P0/C/A/B) operates on metadata, image identity, acquisition metadata
and target-normal availability only. It deliberately has NO loader for anomaly
outcomes, BA/FPR/FNR, harm, or benefit. Everything needed to build the support
census and adjudicate semantic pooling is normal-only.

If C-PASS (a legal supplementary pooling protocol exists), a separate Phase-II
module may load outcomes. If C-STOP, Phase-II never runs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import md0sr_common as C  # type: ignore[import-not-found]

OUT = ROOT / "reports" / "pb8_s1_evidence_breadth"

# --------------------------------------------------------------------------- #
# Frozen predecessor constants (re-read, never re-derived; from the sealed
# PB7-P1 E_FROZEN_METHOD_SPEC.json principal candidate / PB8 estimand).
# --------------------------------------------------------------------------- #
Q50_CUTOFF = 0.015890181064605713  # frozen median_shift gate cutoff (PB7-P1 E)
LAMBDA = 0.25  # frozen shrinkage
PRINCIPAL_K = "8"
K8_MIN = 8  # K8 eligibility: FULL_support_n >= 8
DFNR_MAX = 0.02  # primary safety budget: mean dFNR <= +0.02
DETECTOR = "dinov2_vits14"  # PB8-S1 success criterion is DINOv2-only

# --------------------------------------------------------------------------- #
# Raw AD2 root (metadata-side only: resolves the manifest, no hard-coded path)
# --------------------------------------------------------------------------- #
P0_MANIFEST = ROOT / "reports" / "p0_dataset_opportunity" / "P0_DATASET_MANIFEST.json"


def dataset_root() -> Path:
    obj = json.loads(P0_MANIFEST.read_text(encoding="utf-8"))
    root = obj["datasets"]["MVTec_AD_2"]["root"]
    p = Path(root)
    if not p.exists():
        raise FileNotFoundError(f"MVTec AD 2 root not found: {p}")
    return p


SRC_CONDITION = "regular"


# --------------------------------------------------------------------------- #
# Authoritative 39-unit list (frozen PB6 D5 OOF split manifest)
# --------------------------------------------------------------------------- #
def unit_list() -> List[Tuple[str, str]]:
    """Frozen 39 (category, condition) acquisition-shift units."""
    oof = (
        ROOT
        / "reports"
        / "pb1ds_publication"
        / "pb6_real_third_detector"
        / "PB1DS_PB6_D5_OOF_PER_UNIT.csv"
    )
    cols = []
    with oof.open(encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        for i, name in enumerate(header):
            cols.append((i, name.strip()))
    idx = {name: i for i, name in cols}
    units = set()
    for line in oof.open(encoding="utf-8").readlines()[1:]:
        if not line.strip():
            continue
        parts = line.strip().split(",")
        units.add((parts[idx["category"]], parts[idx["condition"]]))
    return sorted(units, key=lambda u: (u[0], u[1]))


# --------------------------------------------------------------------------- #
# Frozen DEV/TEST split (identical deterministic rule to PB7-P1/PB8)
# --------------------------------------------------------------------------- #
SPLIT_SALT = "PB7_P1_DEV_TEST_SPLIT_v1"


def split_side(category: str) -> str:
    h = int(hashlib.sha256(f"{SPLIT_SALT}|{category}".encode("utf-8")).hexdigest()[:8], 16)
    return "DEV" if h % 2 == 0 else "TEST"


# --------------------------------------------------------------------------- #
# Normal-only score counts (Phase-I safe; reads src/cond normal arrays only)
# --------------------------------------------------------------------------- #
def _score_count(rel: str) -> int:
    p = C.SCORES_DIR / rel
    if not p.exists():
        raise FileNotFoundError(f"missing normal score file: {p}")
    return int(len(np.asarray(dict(np.load(p))["image_scores"])))


def source_normal_n(cat: str) -> int:
    return _score_count(C.source_rel(cat, DETECTOR))


def condition_normal_n(cat: str, cond: str) -> int:
    return _score_count(C.cond_normal_rel(cat, cond, DETECTOR))


# --------------------------------------------------------------------------- #
# Raw image census (content SHA-256; duplicate detection at byte level)
# --------------------------------------------------------------------------- #
def good_images(cat: str, cond: str) -> List[Tuple[Path, str]]:
    """Return [(absolute_path, content_sha256)] for public good images whose
    filename suffix matches the acquisition condition ``cond``.

    Pattern is {ID}_{cond}.png as frozen in the P0 dataset inventory
    (condition metadata is carried by the public filename suffix).
    """
    good_dir = dataset_root() / cat / "test_public" / "good"
    suffix = f"_{cond}.png"
    out: List[Tuple[Path, str]] = []
    for p in sorted(good_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == ".png" and p.name.endswith(suffix):
            out.append((p, sha256_file(p)))
    return out


def rel_to_root(p: Path) -> str:
    return str(p.relative_to(dataset_root()))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_count(hashes: List[str]) -> int:
    return len(set(hashes))


# --------------------------------------------------------------------------- #
# MVTec AD 2 condition registry (metadata only)
# --------------------------------------------------------------------------- #
# The acquisition conditions present vary by category (e.g. vial adds shift_4,
# fruit_jelly omits shift_2/shift_3). These are records of what the raw good
# tree exposes per category, read from the disk; they never carry anomaly info.
def available_conditions(cat: str) -> List[str]:
    good_dir = dataset_root() / cat / "test_public" / "good"
    conds: set[str] = set()
    for p in good_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".png":
            name = p.name
            if name.count("_") >= 1:
                conds.add(name.split("_", 1)[1][:-4])
    return sorted(conds)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()
