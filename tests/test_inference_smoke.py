from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from models.mask_conditioned_turbo import MaskConditionedTurbo


class LatentDist:
    def __init__(self, x: torch.Tensor) -> None:
        self.x = x

    def sample(self) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(self.x[:, :4] if self.x.shape[1] >= 4 else self.x.repeat(1, 2, 1, 1), 8)


class TinyVAE(nn.Module):
    config = SimpleNamespace(scaling_factor=1.0)

    def encode(self, x: torch.Tensor):
        return SimpleNamespace(latent_dist=LatentDist(torch.cat([x, x[:, :1]], dim=1)))

    def decode(self, z: torch.Tensor):
        up = torch.nn.functional.interpolate(z[:, :3], scale_factor=8, mode="nearest")
        return SimpleNamespace(sample=up * 2 - 1)


class TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(5, 4, 1)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor):
        del timestep, encoder_hidden_states
        return SimpleNamespace(sample=self.conv_in(sample))


class TinyScheduler:
    def set_timesteps(self, steps: int, device: torch.device) -> None:
        self.timesteps = torch.tensor([999], device=device)

    def step(self, noise_pred: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor, return_dict: bool = True):
        del timestep, return_dict
        return SimpleNamespace(prev_sample=sample - noise_pred * 0.0)


def test_cpu_tiny_forward_no_nan() -> None:
    model = MaskConditionedTurbo.__new__(MaskConditionedTurbo)
    nn.Module.__init__(model)
    model.base_model = "tiny"
    model.gray_value = 0.5
    model.guidance_scale = 1.0
    model.vae = TinyVAE()
    model.unet = TinyUNet()
    model.scheduler = TinyScheduler()
    source = torch.rand(1, 3, 32, 32)
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 8:24, 8:24] = 1
    text = torch.zeros(1, 77, 8)
    out = model(source=source, mask=mask, fixed_text_embedding=text)
    assert out["final_pred"].shape == source.shape
    assert torch.isfinite(out["final_pred"]).all()
