from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import logging as diffusers_logging
from torch import nn

from data.mask_utils import make_agnostic, make_soft_mask
from models.lora import merged_lora_state_dict
from models.unet_utils import expand_unet_conv_in

diffusers_logging.set_verbosity_error()


class MaskConditionedTurbo(nn.Module):
    def __init__(
        self,
        base_model: str | Path = "stabilityai/sd-turbo",
        gray_value: float = 0.5,
        guidance_scale: float = 1.0,
        torch_dtype: torch.dtype = torch.float16,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.base_model = str(base_model)
        self.gray_value = gray_value
        self.guidance_scale = guidance_scale
        self.vae = AutoencoderKL.from_pretrained(self.base_model, subfolder="vae", torch_dtype=torch_dtype)
        self.unet = UNet2DConditionModel.from_pretrained(self.base_model, subfolder="unet", torch_dtype=torch_dtype)
        self.scheduler = DDPMScheduler.from_pretrained(self.base_model, subfolder="scheduler")
        expand_unet_conv_in(self.unet, 5)
        self.vae.requires_grad_(False)
        if device is not None:
            self.to(device)

    @property
    def scaling_factor(self) -> float:
        return float(getattr(self.vae.config, "scaling_factor", 0.18215))

    def enable_memory_efficient_attention(self, enable_xformers: bool = True) -> None:
        if enable_xformers:
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    def encode_image(self, image_01: torch.Tensor) -> torch.Tensor:
        image = image_01 * 2.0 - 1.0
        posterior = self.vae.encode(image).latent_dist
        return posterior.sample() * self.scaling_factor

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        image = self.vae.decode(latents / self.scaling_factor).sample
        return (image / 2.0 + 0.5).clamp(0, 1)

    def _timesteps(self, device: torch.device) -> torch.Tensor:
        self.scheduler.set_timesteps(1, device=device)
        return self.scheduler.timesteps

    def forward(
        self,
        source: torch.Tensor,
        mask: torch.Tensor,
        fixed_text_embedding: torch.Tensor,
        target: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        del target
        source = source.clamp(0, 1)
        mask = mask.clamp(0, 1)
        agnostic = make_agnostic(source, mask, self.gray_value)
        agnostic_latent = self.encode_image(agnostic)
        latent_mask = F.interpolate(mask, size=agnostic_latent.shape[-2:], mode="nearest")
        unet_input = torch.cat([agnostic_latent, latent_mask.to(agnostic_latent.dtype)], dim=1)
        timesteps = self._timesteps(source.device)
        timestep = timesteps[:1].expand(source.shape[0])
        if fixed_text_embedding.shape[0] == 1 and source.shape[0] > 1:
            fixed_text_embedding = fixed_text_embedding.expand(source.shape[0], -1, -1)
        noise_pred = self.unet(unet_input, timestep, encoder_hidden_states=fixed_text_embedding).sample
        step = self.scheduler.step(noise_pred, timesteps[0], agnostic_latent, return_dict=True)
        pred_latent = step.prev_sample
        raw_pred = self.decode_latents(pred_latent)
        soft_mask = make_soft_mask(mask)
        final_pred = raw_pred * soft_mask + source * (1.0 - soft_mask)
        if not return_dict:
            return final_pred
        return {
            "raw_pred": raw_pred,
            "final_pred": final_pred,
            "agnostic": agnostic,
            "soft_mask": soft_mask,
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.unet.save_pretrained(path / "unet", state_dict=merged_lora_state_dict(self.unet))
        (path / "model_info.json").write_text('{"format":"mask-conditioned-turbo-v1"}', encoding="utf-8")

    def load_unet_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        unet_path = path / "unet" if (path / "unet").exists() else path
        self.unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=next(self.unet.parameters()).dtype)
        expand_unet_conv_in(self.unet, 5)
