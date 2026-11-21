"""R03-D1 stage 17: source-only detector construction inputs and public
inference (frozen DINOv2 feature extraction over MVTec AD 2).

Semantics identical to R03-F0/R03-F0R:
  - backbone dinov2_vits14, layers (2,5,8,11), concat + L2, 224x224
  - preprocessing: PIL RGB float32 [0,1], bilinear resize native->224
  - views T0/T1(+8)/T2(-8) with reflection padding (registration-validated)
  - per scenario: train/good T0 (bank + calibration source); test_public
    good/bad T0/T1/T2 (public inference)

Firewall: only allowlisted paths (train/good, validation/good, test_public)
are ever opened; private splits are hard-denied.
"""

from __future__ import annotations

import hashlib
import os
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shiftad.route_k.features import K0FeatureExtractor  # noqa: E402

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


MV2_ROOT = _env_path("NOROPCAL_MVTEC_AD2_ROOT")
OUT = ROOT / "outputs" / "r03_d1_mvtec_ad2" / "features"
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 224
SHIFT = 8
BATCH = 16
SCENARIOS = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
DENY = ("test_private", "test_private_mixed")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def firewall_check(rel: Path) -> None:
    r = rel.as_posix()
    if any(d in r for d in DENY):
        raise SystemExit(f"FIREWALL DENIED path: {r}")


def reflect_pad(img: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")


def forward_shift(img: np.ndarray, dx: int) -> np.ndarray:
    """Frozen forward view transform (T1 dx=+8 content right; T2 dx=-8 content left)."""
    padded = reflect_pad(img, SHIFT)
    lo = SHIFT - dx
    return np.ascontiguousarray(padded[SHIFT : SHIFT + SIZE, lo : lo + SIZE])


def load_rgb_resized(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im.load()
        rgb = im.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        resized = (
            np.asarray(
                Image.fromarray((arr * 255).astype(np.uint8)).resize(
                    (SIZE, SIZE), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
            / 255.0
        )
    return resized


def enumerate_images(scenario: str, split: str) -> list[tuple[str, Path]]:
    """split in {train, public_good, public_bad} -> (relpath, path)."""
    sroot = MV2_ROOT / scenario
    if split == "train":
        d = sroot / "train" / "good"
    elif split == "public_good":
        d = sroot / "test_public" / "good"
    elif split == "public_bad":
        d = sroot / "test_public" / "bad"
    else:
        raise SystemExit(f"unknown split {split}")
    out = []
    for p in sorted(d.glob("*.png")):
        if p.name.startswith("._"):
            continue
        firewall_check(p.relative_to(MV2_ROOT))
        out.append((p.relative_to(MV2_ROOT).as_posix(), p))
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    extractor = K0FeatureExtractor(
        model_name="dinov2_vits14",
        layers=(2, 5, 8, 11),
        input_size=(SIZE, SIZE),
        device=device,
    )
    extractor.backbone.eval()

    manifest: dict[str, object] = {}
    total = 0
    for scenario in SCENARIOS:
        for split, views in (
            ("train", ("T0",)),
            ("public_good", ("T0", "T1", "T2")),
            ("public_bad", ("T0", "T1", "T2")),
        ):
            items = enumerate_images(scenario, split)
            if not items:
                raise SystemExit(f"empty split {scenario}/{split}")
            rels = [r for r, _ in items]
            for view in views:
                tag = f"{scenario}_{split}_{view}"
                npz_path = OUT / f"{tag}.npz"
                if npz_path.is_file():
                    manifest[tag] = {
                        "n": len(items),
                        "sha256": sha256_file(npz_path),
                        "cached": True,
                    }
                    print("skip", tag)
                    continue
                features = np.zeros((len(items), 256, 1536), dtype=np.float32)
                with torch.inference_mode():
                    for i, (_, path) in enumerate(items):
                        img = load_rgb_resized(path)
                        if view == "T1":
                            img = forward_shift(img, +SHIFT)
                        elif view == "T2":
                            img = forward_shift(img, -SHIFT)
                        features[i] = extractor.extract(img).numpy()
                np.savez_compressed(npz_path, features=features, rels=np.asarray(rels))
                manifest[tag] = {"n": len(items), "sha256": sha256_file(npz_path), "cached": False}
                total += len(items)
                print("wrote", tag, features.shape)

    record = {
        "schema_version": "1.0",
        "stage": "R03-D1",
        "date": "2026-08-16",
        "pipeline": "frozen DINOv2 vits14 layers(2,5,8,11) concat+L2 224x224; PIL bilinear native->224; T1/T2 = +-8px translation with reflection padding",  # noqa: E501
        "identity": "image identity = (scenario, split-role, filename)",
        "firewall": "only train/good, validation/good, test_public/* opened; private splits hard-denied",  # noqa: E501
        "files": manifest,
        "image_views_total": total,
    }
    rec = ROOT / "outputs" / "r03_d1_mvtec_ad2" / "features_manifest.json"
    rec.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print("WROTE", rec.relative_to(ROOT), "| image-views:", total)


if __name__ == "__main__":
    main()
