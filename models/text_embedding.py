from __future__ import annotations

from pathlib import Path

import torch


def save_text_embedding(path: str | Path, prompt_embeds: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompt_embeds": prompt_embeds.detach().cpu()}, path)


def load_text_embedding(path: str | Path, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    obj = torch.load(Path(path), map_location="cpu")
    if isinstance(obj, dict):
        embeds = obj["prompt_embeds"]
    else:
        embeds = obj
    return embeds.to(device=device, dtype=dtype)
