"""R03-D0 stage 5: generate PatchCore (alternative detector) features,
reference bank, NN scores and 224x224 anomaly maps over AeBAD-S (3 views).

Semantics: Phase-4 PatchCore (wide_resnet50_2 layers [2,3], concat + L2,
target_grid [14,14], kcenter coreset 0.1 seed 0 chunk 1024 over train/good,
exact euclidean NN distance k=1, bilinear 224 map, smoothing 0).
No R03 outcome is computed here.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shiftad.backbones.feature_extractor import FeatureFuser  # noqa: E402
from shiftad.backbones.factory import create_backbone  # noqa: E402
from shiftad.backbones.resnet import ResNetBackbone  # noqa: E402
from shiftad.memory.coreset import select_coreset  # noqa: E402

def _env_path(var: str) -> Path:
    """Resolve a machine-specific root from an environment variable.

    Path injection only; experiment logic is unchanged.
    """
    value = os.environ.get(var)
    if not value:
        raise SystemExit(
            f"missing required environment variable {var}; path injection only, "
            "experiment logic unchanged"
        )
    return Path(value)


DATA = _env_path("NOROPCAL_AEBAD_S_ROOT")
OUT = ROOT / "outputs" / "r03_d0_detector_semantics"
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 224
SHIFT = 8
GRID = 14
BATCH = 16
CHUNK = 16384
DEFECT_TYPES = ["ablation", "breakdown", "fracture", "groove"]
CONDITIONS = ["same", "background", "illumination", "view"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reflect_pad(img: np.ndarray, pad: int) -> np.ndarray:
    if img.ndim == 3:
        return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    return np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")


def forward_shift(img: np.ndarray, dx: int) -> np.ndarray:
    padded = reflect_pad(img, SHIFT)
    lo = SHIFT - dx
    return np.ascontiguousarray(padded[SHIFT : SHIFT + SIZE, lo : lo + SIZE])


def load_rgb_resized(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im.load()
        rgb = im.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        resized = np.asarray(
            Image.fromarray((arr * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    return resized


def enumerate_images() -> dict[str, list[tuple[str, Path]]]:
    blocks: dict[str, list[tuple[str, Path]]] = {}
    for cond in ["background", "illumination", "view"]:
        d = DATA / "train" / "good" / cond
        if d.is_dir():
            key = f"train_{cond}"
            blocks.setdefault(key, [])
            for p in sorted(d.glob("*.png")):
                if p.name.startswith("._"):
                    continue
                blocks[key].append((p.relative_to(DATA).as_posix(), p))
    for cond in CONDITIONS:
        d = DATA / "test" / "good" / cond
        if d.is_dir():
            key = f"testgood_{cond}"
            blocks.setdefault(key, [])
            for p in sorted(d.glob("*.png")):
                if p.name.startswith("._"):
                    continue
                blocks[key].append((p.relative_to(DATA).as_posix(), p))
    for cond in CONDITIONS:
        key = f"testanom_{cond}"
        blocks.setdefault(key, [])
        for dt in DEFECT_TYPES:
            d = DATA / "test" / dt / cond
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.png")):
                if p.name.startswith("._"):
                    continue
                blocks[key].append((p.relative_to(DATA).as_posix(), p))
    return blocks


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    backbone = ResNetBackbone(
        model_name="wide_resnet50_2",
        layers=(2, 3),
        device=device,
        dtype="float32",
        checkpoint=None,
        allow_download=False,
        use_cache=True,
        expected_input_size=(SIZE, SIZE),
    )
    backbone.eval()
    fuser = FeatureFuser(mode="concat", normalize=True, target_grid=(GRID, GRID))

    blocks = enumerate_images()
    total = sum(len(v) for v in blocks.values())
    print("images:", total)

    feats_dir = OUT / "features"
    feats_dir.mkdir(exist_ok=True)
    with torch.inference_mode():
        for key, entries in blocks.items():
            for view, dx in (("T0", 0), ("T1", +SHIFT), ("T2", -SHIFT)):
                out_path = feats_dir / f"{key}_{view}.npz"
                if out_path.is_file():
                    print("skip", out_path.name)
                    continue
                feats = []
                for start in range(0, len(entries), BATCH):
                    batch = entries[start : start + BATCH]
                    images = []
                    for rel, path in batch:
                        img = load_rgb_resized(path)
                        images.append(img if view == "T0" else forward_shift(img, dx))
                    tensors = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device)
                    out = backbone.extract_features(tensors)
                    fused = fuser(
                        layer_features=out.layer_features,
                        patch_grid_shapes=out.patch_grid_shapes,
                    )
                    feats.append(fused.features.detach().cpu().numpy())
                arr = np.concatenate(feats, axis=0).astype(np.float32)
                rels = np.array([rel for rel, _ in entries], dtype=object)
                np.savez_compressed(out_path, features=arr, rels=rels)
                print("wrote", out_path.name, arr.shape)

    # ---- bank (frozen operator deployment semantics) -------------------------
    train_pool = []
    for cond in ["background", "illumination", "view"]:
        with np.load(feats_dir / f"train_{cond}_T0.npz", allow_pickle=True) as d:
            train_pool.append(d["features"])
    all_clean = np.concatenate(train_pool, axis=0).reshape(-1, 1536).astype(np.float32)
    print("train pool:", all_clean.shape)
    bank_path = OUT / "bank" / "bank.npz"
    bank_path.parent.mkdir(exist_ok=True)
    if bank_path.is_file():
        bank = np.load(bank_path)["bank"]
    else:
        res = select_coreset(torch.from_numpy(all_clean), method="kcenter", ratio=0.1,
                             target_size=None, seed=0, chunk_size=1024)
        bank = res.features.numpy().astype(np.float32)
        np.savez_compressed(bank_path, bank=bank)
    print("bank:", bank.shape)

    # ---- NN scores + maps ------------------------------------------------------
    bank_t = torch.from_numpy(bank).to(device)
    bn2 = (bank_t ** 2).sum(1)
    scores_dir = OUT / "scores"
    maps_dir = OUT / "maps"
    scores_dir.mkdir(exist_ok=True)
    maps_dir.mkdir(exist_ok=True)
    with torch.inference_mode():
        for key, entries in blocks.items():
            for view in ("T0", "T1", "T2"):
                tag = f"{key}_{view}"
                sp = scores_dir / f"{tag}_scores.npz"
                if sp.is_file():
                    continue
                with np.load(feats_dir / f"{tag}.npz", allow_pickle=True) as d:
                    feats, rels = d["features"], d["rels"]
                flat = feats.reshape(-1, 1536)
                q = torch.from_numpy(flat).to(device)
                qn2 = (q ** 2).sum(1, keepdim=True)
                ds = []
                for st in range(0, q.shape[0], CHUNK):
                    qc = q[st : st + CHUNK]
                    d2 = (qn2[st : st + CHUNK] + bn2[None, :] - 2 * qc @ bank_t.t()).clamp(min=0.0)
                    ds.append(torch.sqrt(d2.min(1).values).detach().cpu().numpy())
                patch_scores = np.concatenate(ds).astype(np.float32).reshape(len(entries), GRID, GRID)
                np.savez_compressed(sp, patch_scores=patch_scores, rels=rels)
                t = torch.from_numpy(patch_scores).unsqueeze(1)
                maps = F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
                np.savez_compressed(maps_dir / f"{tag}_maps.npz", maps=maps.numpy()[:, 0].astype(np.float32), rels=rels)
                print("wrote", tag)

    manifest = {
        "schema_version": "1.0", "stage": "R03-D0", "date": "2026-08-16",
        "detector": "patchcore (wide_resnet50_2 layers[2,3] concat+L2 target_grid[14,14])",
        "bank": {"rows": int(bank.shape[0]), "sha256": sha256_file(bank_path), "pool": int(all_clean.shape[0])},
        "files": {},
    }
    for d in (feats_dir, scores_dir, maps_dir):
        for p in sorted(d.glob("*.npz")):
            manifest["files"][str(p.relative_to(OUT))] = {"sha256": sha256_file(p)}
    (OUT / "extraction_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("WROTE extraction manifest; bank sha256:", manifest["bank"]["sha256"])


if __name__ == "__main__":
    main()
