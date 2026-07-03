from __future__ import annotations

from pathlib import Path

import torch


def build_export_text_embedding_command(path: str | Path, base_model: str, prompt: str) -> str:
    return (
        "uv run python -m scripts.export_text_embedding "
        f"--base-model {base_model} "
        f'--prompt "{prompt}" '
        f"--output {Path(path).as_posix()}"
    )


def save_text_embedding(path: str | Path, prompt_embeds: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompt_embeds": prompt_embeds.detach().cpu()}, path)


def load_text_embedding(
    path: str | Path,
    device: torch.device | str,
    dtype: torch.dtype,
    base_model: str = "stabilityai/sd-turbo",
    prompt: str = "a person wearing the fixed product",
) -> torch.Tensor:
    path = Path(path)
    if not path.exists():
        command = build_export_text_embedding_command(path, base_model, prompt)
        raise FileNotFoundError(f"Missing text embedding: {path}. Export it first with: {command}")

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        embeds = obj["prompt_embeds"]
    else:
        embeds = obj
    return embeds.to(device=device, dtype=dtype)
