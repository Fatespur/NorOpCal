"""Synthetic fixtures that mimic the MVTec AD directory contract."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def write_rgb(path: Path, value: int, *, size: tuple[int, int] = (8, 8)) -> None:
    """Write a deterministic RGB fixture image.

    Args:
        path: Destination image path.
        value: Base uint8 intensity.
        size: Width and height.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    array[..., 1] = np.clip(value + 10, 0, 255)
    Image.fromarray(array).save(path)


def write_mask(path: Path, *, size: tuple[int, int] = (8, 8)) -> None:
    """Write a deterministic centered binary mask fixture.

    Args:
        path: Destination mask path.
        size: Width and height.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((size[1], size[0]), dtype=np.uint8)
    array[2:6, 2:6] = 255
    Image.fromarray(array).save(path)


@pytest.fixture
def mvtec_root(tmp_path: Path) -> Path:
    """Create a minimal valid MVTec AD category without public data."""

    root = tmp_path / "mvtec_ad"
    write_rgb(root / "widget" / "train" / "good" / "000.png", 80)
    write_rgb(root / "widget" / "test" / "good" / "001.png", 90)
    write_rgb(root / "widget" / "test" / "crack" / "002.png", 120)
    write_mask(root / "widget" / "ground_truth" / "crack" / "002_mask.png")
    return root


@pytest.fixture
def image_writer() -> Callable[[Path, int], None]:
    """Expose the RGB fixture writer to dataset failure tests."""

    return write_rgb
