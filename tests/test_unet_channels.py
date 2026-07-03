from __future__ import annotations

import torch
from torch import nn

from models.unet_utils import expand_unet_conv_in


class TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(4, 8, 3, padding=1)
        self.config = type("Config", (), {"in_channels": 4})()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_in(x)


def test_expand_unet_conv_in_migrates_weights_and_forward() -> None:
    unet = TinyUNet()
    old_weight = unet.conv_in.weight.detach().clone()
    old_bias = unet.conv_in.bias.detach().clone()
    expand_unet_conv_in(unet, 5)
    assert unet.conv_in.in_channels == 5
    assert torch.allclose(unet.conv_in.weight[:, :4], old_weight)
    assert torch.allclose(unet.conv_in.weight[:, 4], torch.zeros_like(unet.conv_in.weight[:, 4]))
    assert torch.allclose(unet.conv_in.bias, old_bias)
    out = unet(torch.randn(2, 5, 8, 8))
    assert out.shape == (2, 8, 8, 8)
