from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import PairedGarmentDataset
from losses.boundary import boundary_l1
from losses.masked_losses import masked_charbonnier, masked_l1
from losses.perceptual import MaskCropLPIPS
from models.lora import apply_attention_lora, keep_lora_trainable_params_fp32, load_lora_state_dict, lora_state_dict
from models.mask_conditioned_turbo import MaskConditionedTurbo
from models.text_embedding import load_text_embedding
from scripts.validate import run_validation

LOGGER = logging.getLogger("train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(params, cfg: dict):
    params = list(params)
    if not params:
        raise ValueError("No trainable parameters were selected.")
    if cfg.get("use_8bit_adam", False):
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        except Exception as exc:
            LOGGER.warning("bitsandbytes 8-bit AdamW requested but unavailable; falling back to torch AdamW: %s", exc)
    return torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])


def configure_trainable_unet(model: MaskConditionedTurbo, cfg: dict) -> tuple[list[torch.nn.Parameter], bool]:
    model_cfg = cfg["model"]
    train_full_unet = bool(model_cfg.get("train_full_unet", False))
    model.unet.requires_grad_(False)

    if train_full_unet:
        model.unet.requires_grad_(True)
        params = [p for p in model.unet.parameters() if p.requires_grad]
        LOGGER.info("Training full UNet parameters: %.2fM", sum(p.numel() for p in params) / 1_000_000)
        return params, False

    lora_rank = int(model_cfg.get("lora_rank", 0) or 0)
    lora_count = 0
    if lora_rank > 0:
        lora_alpha = float(model_cfg.get("lora_alpha", lora_rank))
        lora_count = apply_attention_lora(model.unet, rank=lora_rank, alpha=lora_alpha)
        keep_lora_trainable_params_fp32(model.unet)

    if bool(model_cfg.get("train_conv_in", True)):
        model.unet.conv_in.to(dtype=torch.float32)
        model.unet.conv_in.requires_grad_(True)

    params = [p for p in model.unet.parameters() if p.requires_grad]
    LOGGER.info(
        "Training LoRA/adapter parameters: %.2fM across %d LoRA layers; conv_in_trainable=%s",
        sum(p.numel() for p in params) / 1_000_000,
        lora_count,
        any(p.requires_grad for p in model.unet.conv_in.parameters()),
    )
    return params, True


def cleanup_checkpoints(output_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(p for p in output_dir.glob("step_*") if p.is_dir())
    for old in checkpoints[:-keep_last]:
        for child in sorted(old.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        old.rmdir()


def save_training_checkpoint(
    model: MaskConditionedTurbo,
    optimizer: torch.optim.Optimizer,
    path: Path,
    global_step: int,
    save_adapter_state: bool,
) -> None:
    model.save_checkpoint(path)
    if save_adapter_state:
        torch.save({"trainable": lora_state_dict(model.unet), "global_step": global_step}, path / "adapter_state.pt")
    torch.save({"optimizer": optimizer.state_dict(), "global_step": global_step}, path / "training_state.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(int(cfg["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if cfg["training"]["mixed_precision"] == "fp16" and device.type == "cuda" else torch.float32
    LOGGER.info("Using device=%s dtype=%s", device, dtype)
    text_embedding = load_text_embedding(
        cfg["model"]["text_embedding_path"],
        device,
        dtype,
        base_model=cfg["model"]["base_model"],
        prompt=cfg["model"].get("fixed_prompt", "a person wearing the fixed product"),
    )

    data_cfg = cfg["data"]
    res = cfg["model"]["resolution"]
    train_ds = PairedGarmentDataset(
        data_cfg["root"],
        data_cfg["metadata"],
        "train",
        width=res["width"],
        height=res["height"],
        horizontal_flip_prob=float(data_cfg.get("horizontal_flip_prob", 0.0)),
        gray_value=float(cfg["model"].get("gray_value", 0.5)),
        color_jitter=data_cfg.get("color_jitter"),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=data_cfg["num_workers"], pin_memory=device.type == "cuda")
    try:
        val_ds = PairedGarmentDataset(data_cfg["root"], data_cfg["metadata"], "val", width=res["width"], height=res["height"])
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    except ValueError:
        LOGGER.warning("No validation split found; local image visualizations during training will be skipped.")
        val_loader = None

    model = MaskConditionedTurbo(cfg["model"]["base_model"], gray_value=cfg["model"]["gray_value"], torch_dtype=dtype, device=device)
    resume_from = cfg["training"].get("resume_from")
    resume_path = Path(resume_from) if resume_from else None
    adapter_state_path = resume_path / "adapter_state.pt" if resume_path is not None else None
    if resume_path is not None and not adapter_state_path.exists():
        model.load_unet_checkpoint(resume_path)
        model.to(device=device, dtype=dtype)

    trainable_params, save_adapter_state = configure_trainable_unet(model, cfg)
    if adapter_state_path is not None and adapter_state_path.exists():
        load_lora_state_dict(model.unet, adapter_state_path)

    if cfg["training"].get("gradient_checkpointing", False):
        model.unet.enable_gradient_checkpointing()
    if cfg["training"].get("enable_vae_slicing", True):
        model.vae.enable_slicing()
    if cfg["training"].get("enable_vae_tiling", False):
        model.vae.enable_tiling()
    model.enable_memory_efficient_attention(bool(cfg["training"].get("enable_xformers", True)))
    model.train()
    model.vae.eval()

    optimizer = make_optimizer(trainable_params, cfg["training"])
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and device.type == "cuda"))
    lpips_weight = float(cfg["loss"].get("lpips_weight", 0.0))
    lpips_loss = (
        MaskCropLPIPS(device, resize=cfg["loss"]["lpips_resize"], margin_ratio=cfg["loss"]["lpips_crop_margin_ratio"])
        if lpips_weight > 0
        else None
    )
    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path("outputs/tensorboard")))
    global_step = 0
    if resume_path is not None:
        ckpt = torch.load(resume_path / "training_state.pt", map_location="cpu")
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = int(ckpt["global_step"])

    accum = int(cfg["training"]["gradient_accumulation_steps"])
    pbar = tqdm(total=cfg["training"]["max_steps"], initial=global_step)
    optimizer.zero_grad(set_to_none=True)
    while global_step < cfg["training"]["max_steps"]:
        for micro_step, batch in enumerate(train_loader, start=1):
            source = batch["source"].to(device=device, dtype=dtype)
            target = batch["target"].to(device=device, dtype=dtype)
            mask = batch["mask"].to(device=device, dtype=dtype)
            try:
                with torch.amp.autocast("cuda", enabled=(dtype == torch.float16 and device.type == "cuda")):
                    outputs = model(source=source, mask=mask, fixed_text_embedding=text_embedding)
                    raw = outputs["raw_pred"]
                    loss_inside = masked_charbonnier(raw, target, mask)
                    loss_outside = masked_l1(raw, source, 1.0 - mask)
                    loss_lpips = (
                        lpips_loss(raw.float(), target.float(), mask.float())
                        if lpips_loss is not None
                        else torch.zeros((), device=device, dtype=raw.dtype)
                    )
                    loss_boundary = boundary_l1(raw, target, mask, radius=cfg["loss"]["boundary_radius"])
                    loss = (
                        cfg["loss"]["inside_weight"] * loss_inside
                        + cfg["loss"]["outside_weight"] * loss_outside
                        + lpips_weight * loss_lpips
                        + cfg["loss"]["boundary_weight"] * loss_boundary
                    ) / accum
                scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError("CUDA OOM: lower resolution, keep batch_size=1, increase gradient accumulation, and keep full UNet training disabled.") from exc

            if micro_step % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg["training"]["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                pbar.update(1)
                writer.add_scalar("loss/total", float(loss.detach().cpu()) * accum, global_step)
                writer.add_scalar("loss/inside", float(loss_inside.detach().cpu()), global_step)
                writer.add_scalar("loss/outside", float(loss_outside.detach().cpu()), global_step)
                writer.add_scalar("loss/lpips", float(loss_lpips.detach().cpu()), global_step)
                writer.add_scalar("loss/boundary", float(loss_boundary.detach().cpu()), global_step)
                if val_loader is not None and global_step % cfg["logging"]["image_log_every_steps"] == 0:
                    run_validation(model, val_loader, text_embedding, device, Path(cfg["logging"]["visualization_dir"]), global_step)
                if global_step % cfg["logging"]["save_every_steps"] == 0:
                    ckpt_dir = output_dir / f"step_{global_step:08d}"
                    save_training_checkpoint(model, optimizer, ckpt_dir, global_step, save_adapter_state)
                    cleanup_checkpoints(output_dir, int(cfg["training"].get("keep_last_checkpoints", 3)))
                if global_step >= cfg["training"]["max_steps"]:
                    break
    save_training_checkpoint(model, optimizer, output_dir / "last", global_step, save_adapter_state)
    writer.close()


if __name__ == "__main__":
    main()
