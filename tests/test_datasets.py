"""Tests for MVTec AD parsing, isolation, masks, and validation."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from shiftad.data.base import DatasetError, DatasetIntegrityError
from shiftad.data.mvtec_ad import MVTecADDataset
from shiftad.data.validation import validate_mvtec_ad


def test_mvtec_train_and_test_samples_follow_common_contract(mvtec_root: Path) -> None:
    """Train contains normal-only data and test returns aligned anomaly masks."""

    train = MVTecADDataset(mvtec_root, ("widget",), "train", (16, 16))
    assert len(train) == 1
    train_sample = train[0]
    assert train_sample["label"] == 0
    assert train_sample["mask"] is None
    assert train_sample["image"].shape == (3, 16, 16)
    assert train_sample["metadata"]["split"] == "train"

    test = MVTecADDataset(mvtec_root, ("widget",), "test", (16, 16))
    assert len(test) == 2
    anomaly = next(sample for sample in test if sample["label"] == 1)
    assert set(anomaly) == {
        "image",
        "label",
        "mask",
        "category",
        "defect_type",
        "image_path",
        "sample_id",
        "metadata",
    }
    assert anomaly["defect_type"] == "crack"
    assert anomaly["mask"] is not None
    assert anomaly["mask"].shape == (1, 16, 16)
    assert torch.all((anomaly["mask"] == 0) | (anomaly["mask"] == 1))
    assert int(anomaly["mask"].sum()) == 64


def test_image_and_mask_resize_preserves_spatial_alignment(tmp_path: Path) -> None:
    """A resized defect region must remain aligned with its binary mask."""

    root = tmp_path / "mvtec_ad"
    image_path = root / "widget" / "test" / "crack" / "sample.png"
    mask_path = root / "widget" / "ground_truth" / "crack" / "sample_mask.png"
    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[2:6, 2:6, 0] = 255
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)

    sample = MVTecADDataset(root, ("widget",), "test", (16, 16))[0]
    resized_region = sample["image"][0] >= 0.5
    resized_mask = sample["mask"][0].bool()
    assert torch.equal(resized_region, resized_mask)


def test_train_anomaly_directory_is_rejected(
    mvtec_root: Path,
    image_writer: Callable[[Path, int], None],
) -> None:
    """An anomalous training folder fails before memory construction."""

    image_writer(mvtec_root / "widget" / "train" / "crack" / "leak.png", 150)
    with pytest.raises(DatasetIntegrityError, match="Training leakage risk"):
        MVTecADDataset(mvtec_root, ("widget",), "train")

    report = validate_mvtec_ad(mvtec_root, ("widget",))
    assert not report.valid
    assert "TRAIN_ANOMALY_DIRECTORY" in {issue.code for issue in report.issues}


def test_missing_mask_is_reported_and_rejected(mvtec_root: Path) -> None:
    """Missing anomaly masks produce a machine-readable validation reason."""

    mask = mvtec_root / "widget" / "ground_truth" / "crack" / "002_mask.png"
    mask.unlink()
    with pytest.raises(DatasetIntegrityError, match="Missing mask"):
        MVTecADDataset(mvtec_root, ("widget",), "test")
    report = validate_mvtec_ad(mvtec_root, ("widget",))
    assert "MISSING_MASK" in {issue.code for issue in report.issues}


def test_corrupt_image_and_empty_category_are_reported(mvtec_root: Path) -> None:
    """Corruption and empty configured categories never pass validation."""

    corrupt = mvtec_root / "widget" / "test" / "good" / "broken.png"
    corrupt.write_bytes(b"not an image")
    report = validate_mvtec_ad(mvtec_root, ("widget",))
    assert "CORRUPT_IMAGE" in {issue.code for issue in report.issues}

    (mvtec_root / "empty").mkdir()
    empty_report = validate_mvtec_ad(mvtec_root, ("empty",))
    codes = {issue.code for issue in empty_report.issues}
    assert {"EMPTY_TRAIN_GOOD", "EMPTY_TEST_SPLIT"} <= codes


def test_missing_root_has_actionable_adapter_error(tmp_path: Path) -> None:
    """A missing root names the received path and expected condition."""

    missing = tmp_path / "missing"
    with pytest.raises(DatasetError, match="existing dataset root"):
        MVTecADDataset(missing, ("widget",), "test")
    report = validate_mvtec_ad(missing, ("widget",))
    assert report.issues[0].code == "ROOT_NOT_FOUND"
