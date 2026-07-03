from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from models.mask_conditioned_turbo import MaskConditionedTurbo
from models.text_embedding import load_text_embedding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/infer.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")
    dtype = torch.float16 if device.type == "cuda" and cfg.get("dtype") == "fp16" else torch.float32
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
    source = torch.rand(1, 3, cfg["height"], cfg["width"], device=device, dtype=dtype)
    mask = torch.zeros(1, 1, cfg["height"], cfg["width"], device=device, dtype=dtype)
    mask[:, :, cfg["height"] // 4 : cfg["height"] * 3 // 4, cfg["width"] // 4 : cfg["width"] * 3 // 4] = 1
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(dtype == torch.float16 and device.type == "cuda")):
        _ = model(source=source, mask=mask, fixed_text_embedding=text_embedding)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    print(f"gpu={torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"resolution={cfg['width']}x{cfg['height']} time_sec={elapsed:.3f} peak_vram_gb={peak:.2f}")
    print(f"xformers={cfg.get('enable_xformers', True)} torch_compile={cfg.get('enable_torch_compile', False)} vae_slicing={cfg.get('enable_vae_slicing', True)} vae_tiling={cfg.get('enable_vae_tiling', False)}")


if __name__ == "__main__":
    main()
