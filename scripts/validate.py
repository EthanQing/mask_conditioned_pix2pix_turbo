from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from data.dataset import PairedGarmentDataset
from models.mask_conditioned_turbo import MaskConditionedTurbo
from models.text_embedding import load_text_embedding


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(0, 1)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr)


def save_visualization(batch: dict[str, Any], outputs: dict[str, torch.Tensor], target: torch.Tensor, out_dir: Path, step: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    source = batch["source"][:1]
    mask = batch["mask"][:1].repeat(1, 3, 1, 1)
    agnostic = outputs["agnostic"][:1]
    raw = outputs["raw_pred"][:1]
    final = outputs["final_pred"][:1]
    tgt = target[:1]
    diff = (final - tgt).abs()
    grid = make_grid(torch.cat([source, mask, agnostic, raw, final, tgt, diff], dim=0), nrow=7)
    sample_id = batch["sample_id"][0] if isinstance(batch["sample_id"], list) else str(batch["sample_id"])
    path = out_dir / f"step_{step:08d}_{sample_id}.png"
    tensor_to_pil(grid).save(path)
    return path


@torch.no_grad()
def run_validation(model, dataloader, text_embedding: torch.Tensor, device: torch.device, out_dir: Path, step: int, max_batches: int = 1) -> None:
    was_training = model.training
    model.eval()
    dtype = text_embedding.dtype if text_embedding.is_floating_point() else torch.float32
    for idx, batch in enumerate(dataloader):
        if idx >= max_batches:
            break
        source = batch["source"].to(device=device, dtype=dtype)
        mask = batch["mask"].to(device=device, dtype=dtype)
        target = batch["target"].to(device=device, dtype=dtype)
        with torch.amp.autocast("cuda", enabled=(dtype == torch.float16 and device.type == "cuda")):
            outputs = model(source=source, mask=mask, fixed_text_embedding=text_embedding)
        save_visualization(batch, outputs, target, out_dir, step)
    if was_training:
        model.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" and cfg["training"].get("mixed_precision") == "fp16" else torch.float32
    res = cfg["model"]["resolution"]
    ds = PairedGarmentDataset(
        cfg["data"]["root"],
        cfg["data"]["metadata"],
        split=args.split,
        width=res["width"],
        height=res["height"],
        gray_value=float(cfg["model"].get("gray_value", 0.5)),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    model = MaskConditionedTurbo(cfg["model"]["base_model"], gray_value=cfg["model"]["gray_value"], torch_dtype=dtype, device=device)
    model.load_unet_checkpoint(args.checkpoint)
    model.to(device=device, dtype=dtype)
    text_embedding = load_text_embedding(cfg["model"]["text_embedding_path"], device, dtype)
    run_validation(model, loader, text_embedding, device, Path(cfg["logging"]["visualization_dir"]), args.step, max_batches=args.max_batches)
    print(f"Saved validation grids to {cfg['logging']['visualization_dir']}")


if __name__ == "__main__":
    main()
