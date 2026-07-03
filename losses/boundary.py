from __future__ import annotations

import torch

from data.mask_utils import boundary_band
from losses.masked_losses import masked_l1


def boundary_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, radius: int = 8) -> torch.Tensor:
    boundary = boundary_band(mask, radius=radius)
    return masked_l1(pred, target, boundary)
