from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import HfApi, snapshot_download


def inspect_model(repo_id: str, token: str | None) -> dict[str, object]:
    info = HfApi(token=token).model_info(repo_id)
    siblings = [s.rfilename for s in info.siblings]
    return {
        "repo_id": repo_id,
        "sha": info.sha,
        "license": (info.card_data or {}).get("license") if isinstance(info.card_data, dict) else getattr(info.card_data, "license", None),
        "tags": info.tags,
        "downloads": info.downloads,
        "has_diffusers_layout": any(name.startswith("unet/") for name in siblings) and any(name.startswith("vae/") for name in siblings),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="stabilityai/sd-turbo")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--allow-pattern", action="append", default=["*.json", "*.txt", "*.md", "*.safetensors", "*.bin", "scheduler/*", "unet/*", "vae/*", "tokenizer/*", "text_encoder/*"])
    parser.add_argument("--metadata-output", type=Path, default=Path("outputs/model_metadata/sd_turbo.json"))
    args = parser.parse_args()

    metadata = inspect_model(args.repo_id, args.hf_token)
    if not metadata["has_diffusers_layout"]:
        raise RuntimeError(f"{args.repo_id} does not look like a Diffusers SD-style repository")
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    local_dir = snapshot_download(
        repo_id=args.repo_id,
        cache_dir=args.cache_dir,
        token=args.hf_token,
        allow_patterns=args.allow_pattern,
    )
    print(f"Downloaded {args.repo_id} to {local_dir}")
    print(f"Wrote metadata to {args.metadata_output}")
    print(f"License field: {metadata.get('license') or 'see model card/LICENSE.md'}")


if __name__ == "__main__":
    main()
