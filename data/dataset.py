from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from data.mask_utils import make_agnostic
from data.transforms import PairedTransform, TransformConfig


class PairedGarmentDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: str | Path,
        metadata: str | Path,
        split: str = "train",
        width: int = 512,
        height: int = 768,
        horizontal_flip_prob: float = 0.0,
        gray_value: float = 0.5,
        color_jitter: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.metadata = Path(metadata)
        self.split = split
        self.gray_value = gray_value
        color_jitter = color_jitter or {}
        self.transform = PairedTransform(
            TransformConfig(
                width=width,
                height=height,
                horizontal_flip_prob=horizontal_flip_prob,
                color_jitter_enabled=bool(color_jitter.get("enabled", False)),
                brightness=float(color_jitter.get("brightness", 0.02)),
                contrast=float(color_jitter.get("contrast", 0.02)),
                saturation=float(color_jitter.get("saturation", 0.02)),
                hue=float(color_jitter.get("hue", 0.0)),
            )
        )
        self.rows = self._load_rows()
        if not self.rows:
            raise ValueError(f"No samples found for split={split!r} in {self.metadata}")

    def _load_rows(self) -> list[dict[str, str]]:
        with self.metadata.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [row for row in reader if row.get("split") == self.split]

    def __len__(self) -> int:
        return len(self.rows)

    def _open_rgb(self, rel_path: str) -> Image.Image:
        path = self.root / rel_path
        if not path.exists():
            raise FileNotFoundError(path)
        return Image.open(path).convert("RGB")

    def _open_mask(self, rel_path: str) -> Image.Image:
        path = self.root / rel_path
        if not path.exists():
            raise FileNotFoundError(path)
        return Image.open(path).convert("L")

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        source = self._open_rgb(row["source_path"])
        target = self._open_rgb(row["target_path"])
        mask = self._open_mask(row["mask_path"])
        source, target, mask = self.transform(source, target, mask)
        source_t = pil_to_tensor(source).float() / 255.0
        target_t = pil_to_tensor(target).float() / 255.0
        mask_t = (pil_to_tensor(mask).float() >= 127.5).float()
        agnostic = make_agnostic(source_t, mask_t, self.gray_value)
        return {
            "source": source_t,
            "target": target_t,
            "mask": mask_t,
            "agnostic": agnostic,
            "sample_id": row["id"],
        }
