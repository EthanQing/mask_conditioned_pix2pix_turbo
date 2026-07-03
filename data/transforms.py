from __future__ import annotations

from dataclasses import dataclass
from random import Random

import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode


@dataclass(frozen=True)
class TransformConfig:
    width: int = 512
    height: int = 768
    horizontal_flip_prob: float = 0.0
    color_jitter_enabled: bool = False
    brightness: float = 0.02
    contrast: float = 0.02
    saturation: float = 0.02
    hue: float = 0.0


class PairedTransform:
    def __init__(self, config: TransformConfig, seed: int | None = None) -> None:
        self.config = config
        self.rng = Random(seed)

    def __call__(self, source: Image.Image, target: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
        size = (self.config.height, self.config.width)
        source = TF.resize(source, size, interpolation=InterpolationMode.BICUBIC, antialias=True)
        target = TF.resize(target, size, interpolation=InterpolationMode.BICUBIC, antialias=True)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)
        if self.config.horizontal_flip_prob > 0 and self.rng.random() < self.config.horizontal_flip_prob:
            source = TF.hflip(source)
            target = TF.hflip(target)
            mask = TF.hflip(mask)
        if self.config.color_jitter_enabled:
            brightness = self.rng.uniform(max(0.0, 1.0 - self.config.brightness), 1.0 + self.config.brightness)
            contrast = self.rng.uniform(max(0.0, 1.0 - self.config.contrast), 1.0 + self.config.contrast)
            saturation = self.rng.uniform(max(0.0, 1.0 - self.config.saturation), 1.0 + self.config.saturation)
            hue = self.rng.uniform(-self.config.hue, self.config.hue)
            source = TF.adjust_brightness(source, brightness)
            target = TF.adjust_brightness(target, brightness)
            source = TF.adjust_contrast(source, contrast)
            target = TF.adjust_contrast(target, contrast)
            source = TF.adjust_saturation(source, saturation)
            target = TF.adjust_saturation(target, saturation)
            if hue:
                source = TF.adjust_hue(source, hue)
                target = TF.adjust_hue(target, hue)
        return source, target, mask
