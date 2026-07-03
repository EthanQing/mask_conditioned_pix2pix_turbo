from __future__ import annotations

import torch
from torch import nn


def expand_unet_conv_in(unet: nn.Module, new_in_channels: int = 5) -> nn.Module:
    conv = unet.conv_in
    if conv.in_channels == new_in_channels:
        return unet
    if new_in_channels < conv.in_channels:
        raise ValueError(f"new_in_channels={new_in_channels} must be >= current channels={conv.in_channels}")
    new_conv = nn.Conv2d(
        new_in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, : conv.in_channels].copy_(conv.weight)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    unet.conv_in = new_conv
    if hasattr(unet, "config"):
        unet.config.in_channels = new_in_channels
    return unet
