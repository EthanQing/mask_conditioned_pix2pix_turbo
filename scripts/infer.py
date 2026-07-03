from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize

from models.mask_conditioned_turbo import MaskConditionedTurbo
from models.text_embedding import load_text_embedding
from scripts.validate import tensor_to_pil


def load_inputs(source_path: Path, mask_path: Path, width: int, height: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    source = Image.open(source_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    source_t = pil_to_tensor(resize(source, (height, width), InterpolationMode.BICUBIC, antialias=True)).float()[None] / 255.0
    mask_t = (pil_to_tensor(resize(mask, (height, width), InterpolationMode.NEAREST)).float()[None] >= 127.5).float()
    return source_t.to(device=device, dtype=dtype), mask_t.to(device=device, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/infer.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if cfg.get("dtype") == "fp16" and device.type == "cuda" else torch.float32
    source, mask = load_inputs(args.source, args.mask, cfg["width"], cfg["height"], device, dtype)
    model = MaskConditionedTurbo(cfg["model"]["base_model"], gray_value=cfg["model"]["gray_value"], torch_dtype=dtype, device=device)
    model.load_unet_checkpoint(args.checkpoint)
    model.to(device=device, dtype=dtype).eval()
    if cfg.get("enable_vae_slicing", True):
        model.vae.enable_slicing()
    if cfg.get("enable_vae_tiling", False):
        model.vae.enable_tiling()
    model.enable_memory_efficient_attention(bool(cfg.get("enable_xformers", True)))
    if cfg.get("enable_torch_compile", False):
        model.unet = torch.compile(model.unet)
    text_embedding = load_text_embedding(cfg["model"]["text_embedding_path"], device, dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(dtype == torch.float16 and device.type == "cuda")):
        outputs = model(source=source, mask=mask, fixed_text_embedding=text_embedding)
    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(outputs["final_pred"][0]).save(args.output)
    tensor_to_pil(outputs["raw_pred"][0]).save(out_dir / "raw_pred.png")
    tensor_to_pil(outputs["final_pred"][0]).save(out_dir / "final.png")
    tensor_to_pil(outputs["soft_mask"][0]).save(out_dir / "soft_mask.png")
    tensor_to_pil(outputs["agnostic"][0]).save(out_dir / "agnostic.png")
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    print(f"Done. device={device} peak_vram_gb={peak:.2f}")


if __name__ == "__main__":
    main()
