from __future__ import annotations

import torch

from data.mask_utils import boundary_band, dilate_mask, erode_mask, make_soft_mask


def test_dilate_erode_and_feather_ranges() -> None:
    mask = torch.zeros(1, 1, 9, 9)
    mask[:, :, 4, 4] = 1
    dilated = dilate_mask(mask, radius=1)
    assert dilated.sum().item() == 9
    eroded = erode_mask(dilated, radius=1)
    assert eroded[:, :, 4, 4].item() == 1
    soft = make_soft_mask(mask, dilate_radius=1, blur_radius=2)
    assert soft.min().item() >= 0
    assert soft.max().item() <= 1
    boundary = boundary_band(dilated, radius=1)
    assert boundary.shape == mask.shape


def test_empty_and_full_masks_do_not_crash() -> None:
    empty = torch.zeros(1, 1, 8, 8)
    full = torch.ones(1, 1, 8, 8)
    assert make_soft_mask(empty).shape == empty.shape
    assert make_soft_mask(full).shape == full.shape
