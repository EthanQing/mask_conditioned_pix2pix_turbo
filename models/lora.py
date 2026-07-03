from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn


ATTENTION_TARGET_SUFFIXES = ("to_q", "to_k", "to_v", "to_out.0")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float | None = None) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank
        self.lora_down = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=torch.float32)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=torch.float32)
        self.base.requires_grad_(False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_in = x.to(dtype=self.lora_down.weight.dtype)
        lora_out = self.lora_up(self.lora_down(lora_in)) * self.scaling
        return base_out + lora_out.to(dtype=base_out.dtype)

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_up.weight @ self.lora_down.weight) * self.scaling

    def merge(self) -> None:
        with torch.no_grad():
            self.base.weight.add_(self.delta_weight().to(device=self.base.weight.device, dtype=self.base.weight.dtype))

    def unmerge(self) -> None:
        with torch.no_grad():
            self.base.weight.sub_(self.delta_weight().to(device=self.base.weight.device, dtype=self.base.weight.dtype))


def _is_attention_lora_target(name: str) -> bool:
    if ".attn1." not in name and ".attn2." not in name:
        return False
    return name.endswith(ATTENTION_TARGET_SUFFIXES)


def apply_attention_lora(unet: nn.Module, rank: int = 8, alpha: float | None = None) -> int:
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for module_name, module in unet.named_modules():
        for child_name, child in module.named_children():
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear) and _is_attention_lora_target(full_name):
                replacements.append((module, child_name, child))

    for parent, child_name, child in replacements:
        wrapped = LoRALinear(child, rank=rank, alpha=alpha)
        if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
            parent[int(child_name)] = wrapped
        else:
            setattr(parent, child_name, wrapped)

    return len(replacements)


def iter_lora_layers(module: nn.Module):
    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield child


def keep_lora_trainable_params_fp32(module: nn.Module) -> None:
    for layer in iter_lora_layers(module):
        layer.lora_down.to(dtype=torch.float32)
        layer.lora_up.to(dtype=torch.float32)


def merge_lora_layers(module: nn.Module) -> None:
    for layer in iter_lora_layers(module):
        layer.merge()


def unmerge_lora_layers(module: nn.Module) -> None:
    for layer in iter_lora_layers(module):
        layer.unmerge()


def lora_state_dict(module: nn.Module, include_conv_in: bool = True) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters():
        if ".lora_down." in name or ".lora_up." in name:
            state[name] = param.detach().cpu()
        elif include_conv_in and name.startswith("conv_in."):
            state[name] = param.detach().cpu()
    return state


def merged_lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    lora_layers = dict((name, layer) for name, layer in module.named_modules() if isinstance(layer, LoRALinear))
    if not lora_layers:
        return module.state_dict()

    skip_prefixes = tuple(f"{name}." for name in lora_layers)
    merged: dict[str, torch.Tensor] = {}
    for name, layer in lora_layers.items():
        delta = layer.delta_weight().to(device=layer.base.weight.device, dtype=layer.base.weight.dtype)
        merged[f"{name}.weight"] = (layer.base.weight.detach() + delta).cpu()
        if layer.base.bias is not None:
            merged[f"{name}.bias"] = layer.base.bias.detach().cpu()

    for key, value in module.state_dict().items():
        if key.startswith(skip_prefixes):
            continue
        merged[key] = value.detach().cpu()
    return merged


def load_lora_state_dict(module: nn.Module, path: str | Path) -> None:
    state = torch.load(path, map_location="cpu")
    if "trainable" in state:
        state = state["trainable"]
    own_state = dict(module.named_parameters())
    missing: list[str] = []
    for name, value in state.items():
        param = own_state.get(name)
        if param is None:
            missing.append(name)
            continue
        param.data.copy_(value.to(device=param.device, dtype=param.dtype))
    if missing:
        raise RuntimeError(f"Could not load {len(missing)} adapter parameters: {missing[:10]}")
