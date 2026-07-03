from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from data.dataset import PairedGarmentDataset


def _write_sample(root: Path) -> None:
    for sub in ("images/train", "targets/train", "masks/train"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    source[:, :4] = [255, 0, 0]
    target = np.zeros((8, 8, 3), dtype=np.uint8)
    target[:, 4:] = [0, 255, 0]
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, 4:] = 255
    Image.fromarray(source).save(root / "images/train/000001.jpg")
    Image.fromarray(target).save(root / "targets/train/000001.jpg")
    Image.fromarray(mask).save(root / "masks/train/000001.png")
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "split", "source_path", "target_path", "mask_path"])
        writer.writeheader()
        writer.writerow({"id": "000001", "split": "train", "source_path": "images/train/000001.jpg", "target_path": "targets/train/000001.jpg", "mask_path": "masks/train/000001.png"})


def test_dataset_shapes_values_and_agnostic(tmp_path: Path) -> None:
    _write_sample(tmp_path)
    ds = PairedGarmentDataset(tmp_path, tmp_path / "metadata.csv", width=8, height=8)
    item = ds[0]
    assert item["source"].shape == (3, 8, 8)
    assert item["target"].shape == (3, 8, 8)
    assert item["mask"].shape == (1, 8, 8)
    assert set(item["mask"].unique().tolist()).issubset({0.0, 1.0})
    masked = item["mask"].bool().expand_as(item["agnostic"])
    assert abs(item["agnostic"][masked].mean().item() - 0.5) < 1e-6


def test_horizontal_flip_is_synchronized(tmp_path: Path) -> None:
    _write_sample(tmp_path)
    ds = PairedGarmentDataset(tmp_path, tmp_path / "metadata.csv", width=8, height=8, horizontal_flip_prob=1.0)
    item = ds[0]
    assert item["mask"][0, :, :4].mean().item() == 1.0


def test_color_jitter_keeps_mask_geometry(tmp_path: Path) -> None:
    _write_sample(tmp_path)
    ds = PairedGarmentDataset(
        tmp_path,
        tmp_path / "metadata.csv",
        width=8,
        height=8,
        color_jitter={"enabled": True, "brightness": 0.01, "contrast": 0.01, "saturation": 0.01, "hue": 0.0},
    )
    item = ds[0]
    assert item["mask"].shape == (1, 8, 8)
    assert item["mask"][0, :, 4:].mean().item() == 1.0
