from __future__ import annotations

import torch


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask = mask.to(dtype=pred.dtype)
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(1)
    denom = mask.sum() * pred.shape[1] + eps
    return ((pred - target).abs() * mask).sum() / denom


def masked_charbonnier(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    mask = mask.to(dtype=pred.dtype)
    denom = mask.sum() * pred.shape[1] + 1e-6
    return (torch.sqrt((pred - target) ** 2 + eps**2) * mask).sum() / denom
