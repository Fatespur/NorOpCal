"""PB1-DS-PB6 D2 — real EfficientAD training + scoring on MVTec AD 2.

For each of the 8 frozen categories:
  1. Train EfficientAD (student + autoencoder) on the source-normal
     ``train/good`` split only, using the canonical torchvision ResNet18
     teacher and the preregistered frozen training budget
     (``num_steps=70000``, ``batch_size=4``, ``lr=1e-3``, seed 0).
  2. Run a real frozen forward to produce per-image ``image_scores``
     (max of the fused pixel anomaly map, matching the frozen P0 score-cache
     semantic) and per-pixel ``patch_scores`` for every evaluated image.
  3. Write score caches in the exact shared schema consumed by the frozen
     PB1DS frame (``reports/p0_dataset_opportunity/_cache/scores``):
       - ``mv2_{cat}_train_efficientad_scores.npz``          (provenance)
       - ``mv2_{cat}_public_good_{cond}_efficientad_scores.npz``
       - ``mv2_{cat}_public_bad_{cond}_efficientad_scores.npz``
     each with keys ``patch_scores`` (N,H,W) float32, ``image_scores`` (N,)
     float32, ``rels`` (N,) paths ``mvtec_ad_2/{cat}/.../{file}.png``.

Leakage: training uses source-normal ``train/good`` evidence ONLY. Target-
condition normal support, anomalies, anomaly labels, and downstream ShiftAD
benefit labels are never used for training or for selecting checkpoints.

Idempotent: a category is skipped only if its final checkpoint (step
num_steps) exists and all required score caches are already present and valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from shiftad.data.fixture import collate_anomaly_samples
from shiftad.data.mvtec_ad import MVTecADDataset
from shiftad.data.preprocessing import load_rgb_tensor
from shiftad.methods.factory import create_detector
from shiftad.utils.config import ExperimentConfig
from shiftad.utils.device import resolve_device
from shiftad.utils.seed import seed_everything

def _env_dir(var: str) -> Path:
    """Resolve a machine-specific root from the environment (path injection only).

    The historical sealed run used private local roots; no private absolute
    path is hard-coded as a public default. Set the variables explicitly:

    - ``NOROPCAL_MVTEC_AD2_ROOT``: MVTec AD 2 dataset root
    - ``NOROPCAL_PB6_WORK_ROOT``: EfficientAD training/checkpoint work dir
    """
    value = os.environ.get(var)
    if not value:
        raise SystemExit(
            f"missing required environment variable {var}; set it to the local "
            "root (path injection only, experiment logic unchanged)"
        )
    return Path(value)


AD2_ROOT = _env_dir("NOROPCAL_MVTEC_AD2_ROOT")
CONFIG_YAML = ROOT / "configs" / "models" / "efficientad_default.yaml"
SCORE_DIR = ROOT / "reports" / "p0_dataset_opportunity" / "_cache" / "scores"
PB6_DIR = ROOT / "reports" / "pb1ds_publication" / "pb6_real_third_detector"
WORK_ROOT = _env_dir("NOROPCAL_PB6_WORK_ROOT")

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
COND_SUFFIX = re.compile(r"^\d+_(?P<cond>[\w]+)\.png$")
DETECTOR = "efficientad"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_config(category: str, work_dir: Path) -> ExperimentConfig:
    raw = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    raw["dataset"] = {
        "name": "mvtec_ad",
        "root": str(AD2_ROOT),
        "categories": [category],
        "split": "train",
        "image_size": [256, 256],
    }
    # Frozen preregistered PB6 realization: seed 0, canonical budget.
    raw["runtime"] = {
        "seed": 0,
        "device": "cuda",
        "output_dir": str(work_dir / "run"),
        "log_level": "INFO",
        "batch_size": 4,
    }
    raw["calibration"] = {"enabled": False}
    # Authorized canonical teacher: torchvision ResNet18 IMAGENET1K_V1 cache.
    raw["model"]["teacher"] = {
        "pretrained": True,
        "checkpoint": None,
        "allow_download": True,
        "use_cache": True,
        "out_channels": 128,
    }
    raw["model"]["is_fixture"] = False
    return ExperimentConfig.from_mapping(raw)


def image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
    )


def condition_of(name: str) -> str | None:
    m = COND_SUFFIX.match(name)
    return m.group("cond") if m else None


def batch_predict(detector, paths: list[Path], batch_size: int, image_size: tuple[int, int]):
    """Yield (image_scores, patch_maps) arrays as parallel lists of np arrays."""
    image_scores: list[np.ndarray] = []
    patch_maps: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        tensors = [load_rgb_tensor(p, image_size)[0] for p in chunk]
        batch = torch.stack(tensors)
        pred = detector.predict(batch)
        image_scores.append(np.asarray(pred.image_scores, dtype=np.float32))
        patch_maps.append(
            np.stack([m.detach().cpu().numpy().astype(np.float32) for m in pred.anomaly_maps])
        )
    return (
        np.concatenate(image_scores) if image_scores else np.empty(0, dtype=np.float32),
        np.concatenate(patch_maps) if patch_maps else np.empty((0, 0, 0), dtype=np.float32),
    )


def write_cache(path: Path, patch_scores, image_scores, rels) -> None:
    np.savez_compressed(
        path,
        patch_scores=np.asarray(patch_scores, dtype=np.float32),
        image_scores=np.asarray(image_scores, dtype=np.float32),
        rels=np.asarray(rels),
    )


def _frozen_num_steps() -> int:
    raw = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    return int(raw["model"]["training"]["num_steps"])


def category_done(category: str, work_dir: Path) -> bool:
    ckpt_dir = work_dir / "checkpoints"
    final = ckpt_dir / f"checkpoint_step_{_frozen_num_steps():08d}.pt"
    if not final.is_file():
        return False
    required = [
        SCORE_DIR / f"mv2_{category}_train_{DETECTOR}_scores.npz",
        SCORE_DIR / f"mv2_{category}_public_bad_regular_{DETECTOR}_scores.npz",
    ]
    good_dir = AD2_ROOT / category / "test_public" / "good"
    for p in image_paths(good_dir):
        c = condition_of(p.name)
        if c:
            required.append(SCORE_DIR / f"mv2_{category}_public_good_{c}_{DETECTOR}_scores.npz")
    return all(p.is_file() for p in required)


def process_category(category: str, device: torch.device, image_size: tuple[int, int]) -> dict:
    work_dir = WORK_ROOT / category
    work_dir.mkdir(parents=True, exist_ok=True)
    if category_done(category, work_dir):
        return {"category": category, "status": "skipped_already_done"}

    cfg = make_config(category, work_dir)
    seed_everything(cfg.runtime.seed, deterministic_algorithms=True)

    ckpt_dir = work_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = MVTecADDataset(
        Path(cfg.dataset.root),
        tuple(cfg.dataset.categories),
        "train",
        tuple(cfg.dataset.image_size),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.runtime.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_anomaly_samples,
    )
    n_train = len(train_dataset)
    print(
        f"[{category}] training EfficientAD on {n_train} train/good images, seed={cfg.runtime.seed}",
        flush=True,
    )
    t0 = time.time()

    detector = create_detector(cfg.model, device=device)
    compatibility_info = {"category": category, "seed": cfg.runtime.seed, "split_fingerprint": None}
    detector.fit(
        train_loader,
        checkpoint_dir=ckpt_dir,
        checkpoint_interval=10000,
        auto_resume=True,
        max_to_keep=3,
        compatibility_info=compatibility_info,
    )
    train_s = round(time.time() - t0, 1)
    print(f"[{category}] training done in {train_s}s", flush=True)

    final_ckpt = ckpt_dir / f"checkpoint_step_{cfg.model.training.num_steps:08d}.pt"
    # fallback scan for the actual final checkpoint name
    if not final_ckpt.is_file():
        cands = sorted(ckpt_dir.glob("checkpoint_step_*.pt"))
        final_ckpt = cands[-1] if cands else None
    ckpt_hash = sha256_file(final_ckpt) if final_ckpt and final_ckpt.is_file() else ""

    rel_prefix = f"mvtec_ad_2/{category}"
    out: dict = {
        "category": category,
        "train_images": n_train,
        "train_steps": cfg.model.training.num_steps,
        "training_seconds": train_s,
        "seed": cfg.runtime.seed,
        "device": str(device),
        "checkpoint": str(final_ckpt) if final_ckpt else "",
        "checkpoint_sha256": ckpt_hash,
        "caches": {},
    }

    def score_write(files: list[Path], role: str) -> None:
        if not files:
            return
        isc, pm = batch_predict(detector, files, 4, image_size)
        # group by condition
        by_cond: dict[str, list[int]] = {}
        for i, p in enumerate(files):
            c = condition_of(p.name) or "unknown"
            by_cond.setdefault(c, []).append(i)
        for cond, idx in by_cond.items():
            idx_a = np.asarray(idx, dtype=int)
            tag = f"mv2_{category}_public_{role}_{cond}_{DETECTOR}"
            path = SCORE_DIR / f"{tag}_scores.npz"
            rels = [f"{rel_prefix}/test_public/{role}/{files[i].name}" for i in idx]
            write_cache(path, pm[idx_a], isc[idx_a], rels)
            out["caches"][f"public_{role}_{cond}"] = {
                "file": path.name,
                "n": int(len(idx)),
                "sha256": sha256_file(path),
            }
            print(f"[{category}] wrote {path.name} n={len(idx)}", flush=True)

    # train/good provenance cache
    train_files = image_paths(AD2_ROOT / category / "train" / "good")
    if train_files:
        isc, pm = batch_predict(detector, train_files, 4, image_size)
        tag = f"mv2_{category}_train_{DETECTOR}"
        path = SCORE_DIR / f"{tag}_scores.npz"
        rels = [f"{rel_prefix}/train/good/{p.name}" for p in train_files]
        write_cache(path, pm, isc, rels)
        out["caches"]["train"] = {
            "file": path.name,
            "n": len(train_files),
            "sha256": sha256_file(path),
        }

    score_write(image_paths(AD2_ROOT / category / "test_public" / "good"), "good")
    score_write(image_paths(AD2_ROOT / category / "test_public" / "bad"), "bad")

    out["status"] = "complete"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=list(MV2_CATEGORIES))
    args = parser.parse_args()
    categories = [c for c in MV2_CATEGORIES if c in args.categories]

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    PB6_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    image_size = (256, 256)

    results: list[dict] = []
    for cat in categories:
        try:
            results.append(process_category(cat, device, image_size))
        except Exception as exc:  # record and continue; do not silently drop
            results.append(
                {"category": cat, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            )
            print(f"[{cat}] ERROR {exc}", flush=True)

    # Execution manifest
    manifest_path = PB6_DIR / "PB1DS_PB6_EFFICIENTAD_EXECUTION_MANIFEST.csv"
    import csv

    rows = []
    fieldnames = [
        "category",
        "condition",
        "unit_count",
        "train_images",
        "training_seconds",
        "seed",
        "device",
        "checkpoint_sha256",
        "status",
        "error",
    ]
    for r in results:
        caches = r.get("caches", {})
        units = sorted({k.split("_")[-1] for k in caches if k.startswith("public_good_")})
        rows.append(
            {
                "category": r["category"],
                "condition": ",".join(units),
                "unit_count": len(units),
                "train_images": r.get("train_images", ""),
                "training_seconds": r.get("training_seconds", ""),
                "seed": r.get("seed", ""),
                "device": r.get("device", ""),
                "checkpoint_sha256": r.get("checkpoint_sha256", ""),
                "status": r.get("status", ""),
                "error": r.get("error", ""),
            }
        )
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Score manifest
    score_rows = []
    for r in results:
        for cond, meta in r.get("caches", {}).items():
            score_rows.append(
                {
                    "category": r["category"],
                    "role_condition": cond,
                    "file": meta["file"],
                    "n": meta["n"],
                    "sha256": meta["sha256"],
                    "training_seconds": r.get("training_seconds", ""),
                }
            )
    with (PB6_DIR / "PB1DS_PB6_EFFICIENTAD_SCORE_MANIFEST.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.DictWriter(
            fh, fieldnames=["category", "role_condition", "file", "n", "sha256", "training_seconds"]
        )
        w.writeheader()
        w.writerows(score_rows)

    print("D2 done. results:", json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
