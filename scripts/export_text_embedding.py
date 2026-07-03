from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import CLIPTextModel, CLIPTokenizer

from models.text_embedding import save_text_embedding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="stabilityai/sd-turbo")
    parser.add_argument("--prompt", default="undress, nsfw, nude, naked, pussy, vagina, penis")
    parser.add_argument("--output", type=Path, default=Path("models/text_embeddings/fixed_prompt_sd_turbo.pt"))
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer", token=args.hf_token)
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder", torch_dtype=torch.float16, token=args.hf_token)
    text_encoder.eval()
    inputs = tokenizer(
        args.prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embeds = text_encoder(inputs.input_ids)[0].cpu()
    save_text_embedding(args.output, embeds)
    print(f"Saved text embedding to {args.output}")


if __name__ == "__main__":
    main()
