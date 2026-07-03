from __future__ import annotations

import torch
import torch.nn.functional as F


def _kernel(radius: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    size = radius * 2 + 1
    return torch.ones((1, 1, size, size), device=device, dtype=dtype)


def dilate_mask(mask: torch.Tensor, radius: int = 8) -> torch.Tensor:
    if radius <= 0:
        return mask.clamp(0, 1)
    return F.max_pool2d(mask.float(), kernel_size=radius * 2 + 1, stride=1, padding=radius).to(mask.dtype)


def erode_mask(mask: torch.Tensor, radius: int = 8) -> torch.Tensor:
    if radius <= 0:
        return mask.clamp(0, 1)
    return 1.0 - dilate_mask(1.0 - mask.float(), radius=radius)


def gaussian_blur(mask: torch.Tensor, radius: int = 16, sigma: float | None = None) -> torch.Tensor:
    if radius <= 0:
        return mask.clamp(0, 1)
    sigma = sigma or max(radius / 3.0, 1e-6)
    size = radius * 2 + 1
    coords = torch.arange(size, device=mask.device, dtype=mask.dtype) - radius
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_x = kernel_1d.view(1, 1, 1, size)
    kernel_y = kernel_1d.view(1, 1, size, 1)
    x = F.pad(mask.float(), (radius, radius, 0, 0), mode="replicate")
    x = F.conv2d(x, kernel_x.float())
    x = F.pad(x, (0, 0, radius, radius), mode="replicate")
    x = F.conv2d(x, kernel_y.float())
    return x.clamp(0, 1).to(mask.dtype)


def make_soft_mask(mask: torch.Tensor, dilate_radius: int = 12, blur_radius: int = 16) -> torch.Tensor:
    binary = (mask > 0.5).to(mask.dtype)
    return gaussian_blur(dilate_mask(binary, dilate_radius), blur_radius).clamp(0, 1)


def boundary_band(mask: torch.Tensor, radius: int = 8) -> torch.Tensor:
    binary = (mask > 0.5).to(mask.dtype)
    return (dilate_mask(binary, radius) - erode_mask(binary, radius)).clamp(0, 1)


def make_agnostic(source: torch.Tensor, mask: torch.Tensor, gray_value: float = 0.5) -> torch.Tensor:
    return source * (1.0 - mask) + float(gray_value) * mask
