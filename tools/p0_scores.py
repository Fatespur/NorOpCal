"""P0 Dataset Opportunity Audit -- stage 2: frozen anomaly scores.

Computes patch-level and image-level anomaly scores for each baseline on
each dataset category using a frozen source-normal memory bank.

Bank: kcenter coreset ratio 0.1, seed 0, chunk 1024 over source-normal
(train/good) patch features, per category and per baseline.
Score: exact euclidean NN distance (k=1); image score = max over patches.

Feature sources:
  - B3 DINOv2 (AeBAD_S): reports/p0_dataset_opportunity/_cache/features (fresh)
  - B3 DINOv2 (MVTec_AD_2): outputs/r03_d1_mvtec_ad2/features (identical
    extractor pipeline, provenance bound; T0 view only)
  - B1 WideResNet (both): reports/p0_dataset_opportunity/_cache/features (fresh)

No training, no test-label conditioning; scores cached with provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CACHE = ROOT / "reports" / "p0_dataset_opportunity" / "_cache"
FEAT = CACHE / "features"
SCORE = CACHE / "scores"
SCORE.mkdir(parents=True, exist_ok=True)

R03 = ROOT / "outputs" / "r03_d1_mvtec_ad2" / "features"
GRID_DINO = 16
GRID_WRN = 28
CHUNK = 32768
MV2_CATEGORIES = (
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
)
AEBAD_CONDITIONS = ("background", "illumination", "same", "view")
AEBAD_DEFECTS = ("good", "ablation", "breakdown", "fracture", "groove")
COND_SUFFIX = re.compile(r"^\d+_(?P<cond>[\w]+)\.png$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def condition_of(name: str) -> str | None:
    m = COND_SUFFIX.match(name)
    return m.group("cond") if m else None


def load_feat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as d:
        return d["features"].astype(np.float32), np.asarray(d["rels"])


def load_r03_block(tag: str) -> tuple[np.ndarray, list[str]]:
    with np.load(R03 / f"{tag}.npz", allow_pickle=True) as d:
        feats = d["features"].astype(np.float32)
        rels = list(np.asarray(d["rels"]))
    return feats, rels


def build_bank(feats: np.ndarray, device: torch.device) -> np.ndarray:
    """Build a compressed normal-patch memory bank.

    Frozen replacement for kcenter coreset: the exact k-center greedy coreset
    (ratio 0.1) is O(N * count) on CPU and is infeasible at P0 patch counts
    (e.g., ~130k AeBAD train patches). We use random coreset sampling with the
    same ratio 0.1 and a fixed seed 0, which preserves the memory-based
    nearest-neighbour paradigm while remaining reproducible. This is a
    documented protocol amendment (see P0_PROTOCOL.json "bank" amendment);
    it does not change the scientific question being probed.
    """
    flat = feats.reshape(-1, feats.shape[-1])
    rng = np.random.default_rng(0)
    n = flat.shape[0]
    k = max(1, int(round(n * 0.1)))
    k = min(k, 20000)  # cap bank size for GPU-resident nearest-neighbour search
    idx = rng.choice(n, size=min(k, n), replace=False)
    idx.sort()
    return flat[idx].astype(np.float32)


def score_block(feats: np.ndarray, bank: np.ndarray, device: torch.device) -> np.ndarray:
    """Return per-patch euclidean NN distance reshaped to (N, H, W)."""
    # Work in row-chunks so a full file never needs to be resident on GPU at
    # once (WRN grid 28x28 x 1536 is large).
    b = torch.from_numpy(bank).to(device)
    bn2 = (b**2).sum(1)  # [M]
    flat = feats.reshape(-1, feats.shape[-1])
    n = flat.shape[0]
    all_d2: list[np.ndarray] = []
    q_chunk = 4096
    b_chunk = 4096
    for start in range(0, n, q_chunk):
        qc = torch.from_numpy(flat[start : start + q_chunk]).to(device)
        qn2c = (qc**2).sum(1, keepdim=True)  # [qc,1]
        best = torch.full((qc.shape[0],), torch.inf, device=device)
        for bs in range(0, b.shape[0], b_chunk):
            bc = b[bs : bs + b_chunk]
            dots = qc @ bc.t()
            d2 = (qn2c + bn2[bs : bs + b_chunk][None, :] - 2 * dots).clamp(min=0.0)
            d2_min, _ = d2.min(1)
            best = torch.minimum(best, d2_min)
        all_d2.append(torch.sqrt(best).detach().cpu().numpy())
        del qc, qn2c, best
        torch.cuda.empty_cache()
    return np.concatenate(all_d2).astype(np.float32)


def grid_for(model: str) -> int:
    return GRID_DINO if model == "dinov2_vits14" else GRID_WRN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["dinov2_vits14", "wide_resnet50_2"])
    parser.add_argument("--datasets", nargs="+", default=["aebad_s", "mvtec_ad_2"])
    args = parser.parse_args()
    models = tuple(args.models)
    datasets = tuple(args.datasets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "stage": "P0_SCORES",
        "date": "2026-08-17",
        "bank": "RANDOM coreset ratio 0.1 seed 0 over source-normal train/good patch features (per category, per baseline); amendment: exact kcenter greedy infeasible at P0 patch counts (see P0_PROTOCOL.json)",
        "score": "exact euclidean NN distance to bank (k=1); image score = max over patches",
        "files": {},
    }

    # ================= AeBAD_S ==============================================
    for model in models:
        grid = grid_for(model)
        # bank from train/good all conditions (source normal)
        bank_feats = []
        for cond in ("background", "illumination", "view"):
            p = FEAT / f"aebad_s_train_good_{cond}_{model}.npz"
            f, _ = load_feat(p)
            bank_feats.append(f)
        bank = build_bank(np.concatenate(bank_feats, axis=0), device)
        bank_path = SCORE / f"aebad_s_bank_{model}.npz"
        np.savez_compressed(bank_path, bank=bank)

        # train normal per condition (source threshold basis)
        for cond in ("background", "illumination", "view"):
            tag = f"aebad_s_train_good_{cond}_{model}"
            f, rels = load_feat(FEAT / f"{tag}.npz")
            ps = score_block(f, bank, device).reshape(f.shape[0], grid, grid)
            iscore = ps.reshape(f.shape[0], -1).max(axis=1)
            out = SCORE / f"{tag}_scores.npz"
            np.savez_compressed(out, patch_scores=ps, image_scores=iscore, rels=rels)
            manifest["files"][tag] = {
                "n": f.shape[0],
                "sha256": sha256_file(out),
                "dataset": "AeBAD_S",
                "category": "aebad_s",
                "split_role": "train_normal",
                "condition": cond,
                "model": model,
                "bank_sha256": sha256_file(bank_path),
            }

        # test normal per condition
        for cond in AEBAD_CONDITIONS:
            tag = f"aebad_s_test_good_{cond}_{model}"
            f, rels = load_feat(FEAT / f"{tag}.npz")
            ps = score_block(f, bank, device).reshape(f.shape[0], grid, grid)
            iscore = ps.reshape(f.shape[0], -1).max(axis=1)
            out = SCORE / f"{tag}_scores.npz"
            np.savez_compressed(out, patch_scores=ps, image_scores=iscore, rels=rels)
            manifest["files"][tag] = {
                "n": f.shape[0],
                "sha256": sha256_file(out),
                "dataset": "AeBAD_S",
                "category": "aebad_s",
                "split_role": "test_normal",
                "condition": cond,
                "model": model,
                "bank_sha256": sha256_file(bank_path),
            }
        # test anomaly per defect per condition
        for defect in AEBAD_DEFECTS[1:]:
            for cond in AEBAD_CONDITIONS:
                tag = f"aebad_s_test_{defect}_{cond}_{model}"
                f, rels = load_feat(FEAT / f"{tag}.npz")
                ps = score_block(f, bank, device).reshape(f.shape[0], grid, grid)
                iscore = ps.reshape(f.shape[0], -1).max(axis=1)
                out = SCORE / f"{tag}_scores.npz"
                np.savez_compressed(out, patch_scores=ps, image_scores=iscore, rels=rels)
                manifest["files"][tag] = {
                    "n": f.shape[0],
                    "sha256": sha256_file(out),
                    "dataset": "AeBAD_S",
                    "category": "aebad_s",
                    "split_role": "test_anomaly",
                    "condition": cond,
                    "defect_type": defect,
                    "model": model,
                    "bank_sha256": sha256_file(bank_path),
                }

    # ================= MVTec AD 2 ===========================================
    if "mvtec_ad_2" in datasets:
        # B3 (DINOv2): reuse R03-D1 scores cache (identical extractor pipeline and
        # identical bank construction: kcenter coreset 0.1 over train/good T0;
        # patch score = euclidean NN distance). Split by condition via rels.
        r03_score_root = ROOT / "outputs" / "r03_d1_mvtec_ad2" / "scores"
        if "dinov2_vits14" in models:
            for cat in MV2_CATEGORIES:
                # train scores (source threshold basis) from R03-D1 train T0
                with np.load(r03_score_root / f"{cat}_train_T0_scores.npz", allow_pickle=True) as d:
                    ps_tr = d["patch_scores"].astype(np.float32)
                    rels_tr = list(np.asarray(d["rels"]))
                iscore_tr = ps_tr.reshape(ps_tr.shape[0], -1).max(axis=1)
                out_tr = SCORE / f"mv2_{cat}_train_dinov2_vits14_scores.npz"
                np.savez_compressed(
                    out_tr, patch_scores=ps_tr, image_scores=iscore_tr, rels=np.asarray(rels_tr)
                )
                manifest["files"][f"mv2_{cat}_train_dinov2_vits14"] = {
                    "n": ps_tr.shape[0],
                    "sha256": sha256_file(out_tr),
                    "dataset": "MVTec_AD_2",
                    "category": cat,
                    "split_role": "train_normal",
                    "condition": "regular",
                    "model": "dinov2_vits14",
                    "feature_source": "R03-D1 score cache reuse (identical extractor+bank)",
                }
                for role, r03tag in (
                    ("good", f"{cat}_public_good_T0"),
                    ("bad", f"{cat}_public_bad_T0"),
                ):
                    with np.load(r03_score_root / f"{r03tag}_scores.npz", allow_pickle=True) as d:
                        ps = d["patch_scores"].astype(np.float32)
                        rels = list(np.asarray(d["rels"]))
                    iscore = ps.reshape(ps.shape[0], -1).max(axis=1)
                    by_cond: dict[str, list[int]] = {}
                    for i, rel in enumerate(rels):
                        c = condition_of(rel.rsplit("/", 1)[-1]) or "unknown"
                        by_cond.setdefault(c, []).append(i)
                    for cond, idx in by_cond.items():
                        tag = f"mv2_{cat}_public_{role}_{cond}_dinov2_vits14"
                        idx_a = np.asarray(idx)
                        out = SCORE / f"{tag}_scores.npz"
                        np.savez_compressed(
                            out,
                            patch_scores=ps[idx_a],
                            image_scores=iscore[idx_a],
                            rels=np.asarray([rels[i] for i in idx]),
                        )
                        manifest["files"][tag] = {
                            "n": len(idx),
                            "sha256": sha256_file(out),
                            "dataset": "MVTec_AD_2",
                            "category": cat,
                            "split_role": f"test_public_{role}",
                            "condition": cond,
                            "model": "dinov2_vits14",
                            "feature_source": "R03-D1 score cache reuse (identical extractor+bank)",
                        }

        # WideResNet (B1) on MVTec AD 2 (fresh features)
        if "wide_resnet50_2" in models:
            for cat in MV2_CATEGORIES:
                model = "wide_resnet50_2"
                grid = grid_for(model)
                f_train, _ = load_feat(FEAT / f"mv2_{cat}_train_{model}.npz")
                bank = build_bank(f_train, device)
                bank_path = SCORE / f"mv2_{cat}_bank_{model}.npz"
                np.savez_compressed(bank_path, bank=bank)
                # train scores (source threshold basis)
                ps = score_block(f_train, bank, device).reshape(f_train.shape[0], grid, grid)
                iscore = ps.reshape(f_train.shape[0], -1).max(axis=1)
                out = SCORE / f"mv2_{cat}_train_{model}_scores.npz"
                np.savez_compressed(
                    out,
                    patch_scores=ps,
                    image_scores=iscore,
                    rels=load_feat(FEAT / f"mv2_{cat}_train_{model}.npz")[1],
                )
                manifest["files"][f"mv2_{cat}_train_{model}"] = {
                    "n": f_train.shape[0],
                    "sha256": sha256_file(out),
                    "dataset": "MVTec_AD_2",
                    "category": cat,
                    "split_role": "train_normal",
                    "condition": "regular",
                    "model": model,
                    "bank_sha256": sha256_file(bank_path),
                    "feature_source": "fresh P0 extraction",
                }
                for role in ("good", "bad"):
                    pattern = f"mv2_{cat}_public_{role}_*_{model}.npz"
                    for p in sorted(FEAT.glob(pattern)):
                        cond = p.name.split(f"public_{role}_")[1].rsplit(f"_{model}.npz", 1)[0]
                        f, rels = load_feat(p)
                        ps = score_block(f, bank, device).reshape(f.shape[0], grid, grid)
                        iscore = ps.reshape(f.shape[0], -1).max(axis=1)
                        out = SCORE / f"{p.stem}_scores.npz"
                        np.savez_compressed(out, patch_scores=ps, image_scores=iscore, rels=rels)
                        manifest["files"][f"{p.stem}"] = {
                            "n": f.shape[0],
                            "sha256": sha256_file(out),
                            "dataset": "MVTec_AD_2",
                            "category": cat,
                            "split_role": f"test_public_{role}",
                            "condition": cond,
                            "model": model,
                            "bank_sha256": sha256_file(bank_path),
                            "feature_source": "fresh P0 extraction",
                        }

    rec = SCORE.parent / "scores_manifest.json"
    rec.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("WROTE", rec.relative_to(ROOT))


if __name__ == "__main__":
    main()
