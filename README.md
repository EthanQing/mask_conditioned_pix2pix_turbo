# Mask-conditioned Pix2Pix-Turbo Fixed Product Try-on

This project trains and serves a fixed single-product virtual try-on model. Inputs are a person image and an annotated edit mask; output is the same person wearing the fixed product. The model follows a Mask-conditioned Pix2Pix-Turbo style path: masked agnostic image, VAE latent encoding, concatenated latent mask, one-step SD-Turbo UNet, VAE decode, and soft-mask compositing back over the source.

## Dataset Format

```text
dataset/
├── images/{train,val,test}/000001.jpg
├── targets/{train,val,test}/000001.jpg
├── masks/{train,val,test}/000001.png
└── metadata.csv
```

Masks use `0` for keep and `255` for edit. `metadata.csv` columns are:

```csv
id,split,source_path,target_path,mask_path
000001,train,images/train/000001.jpg,targets/train/000001.jpg,masks/train/000001.png
```

## Install

```bash
pip install -r requirements.txt
```

For CUDA, install the PyTorch wheel matching your driver first if needed. `xformers` is optional and skipped on Windows by default.

## Model Source And License

Default base model: [`stabilityai/sd-turbo`](https://huggingface.co/stabilityai/sd-turbo).

- Version/source: Hugging Face Diffusers repository, loaded by `from_pretrained("stabilityai/sd-turbo")`.
- License: [Stability AI Community License](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.md), last updated July 5, 2024.
- Commercial use: the model card says commercial use must follow Stability AI licensing terms. The Community License allows research, non-commercial, and limited commercial use under its conditions, including an annual revenue threshold. Review the license before production use.
- Use in this project: VAE, UNet, scheduler, and one fixed prompt embedding. Inference does not load tokenizer or text encoder after embedding export.
- Replacement policy: if this model cannot be downloaded or its license is unsuitable, choose a same-family turbo/distilled image model with a clearer license and document the reason here before training.

No pretrained weights are committed to git; Hugging Face cache/local model cache is used.

## Preprocess

Download and inspect the base model before training:

```bash
uv run python -m scripts.download_base_model \
  --repo-id stabilityai/sd-turbo \
  --metadata-output outputs/model_metadata/sd_turbo.json
```

Split left/right stitched pair images:

```bash
uv run python -m scripts.split_pairs --input original_pairs --dataset-root dataset --split train --separator-width 0
```

Build metadata:

```bash
uv run python -m scripts.build_metadata --root dataset --output dataset/metadata.csv
```

Export the fixed text embedding once:

```bash
uv run python -m scripts.export_text_embedding \
  --base-model stabilityai/sd-turbo \
  --prompt "a person wearing the fixed product" \
  --output models/text_embeddings/fixed_prompt_sd_turbo.pt
```

If the model is gated or your environment needs authentication, set `HF_TOKEN` or pass `--hf-token`.

## Training

```bash
uv run python -m scripts.train --config configs/train.yaml
```

`training.max_steps` counts optimizer updates, not micro-batches. With the default batch size `1` and gradient accumulation `8`, each step consumes eight samples.

Resume:

```bash
uv run python -m scripts.train --config configs/train.yaml
```

Set `training.resume_from` in `configs/train.yaml` to a checkpoint directory such as `checkpoints/step_00001000`.

Run validation visualizations without training:

```bash
uv run python -m scripts.validate \
  --config configs/train.yaml \
  --checkpoint checkpoints/last \
  --split val \
  --max-batches 8
```

## Inference

```bash
uv run python -m scripts.infer \
  --config configs/infer.yaml \
  --checkpoint checkpoints/last \
  --source path/to/source.jpg \
  --mask path/to/mask.png \
  --output outputs/result.png
```

Outputs written next to `--output`: `raw_pred.png`, `final.png`, `soft_mask.png`, and `agnostic.png`.

## VRAM Benchmark

```bash
uv run python -m scripts.benchmark_vram --config configs/infer.yaml --checkpoint checkpoints/last
```

The script reports GPU name, PyTorch/CUDA versions, resolution, latency, peak VRAM, xFormers, torch.compile, VAE slicing, and VAE tiling.

## RTX 3070 8GB Notes

Default deployment is FP16, batch size 1, `512x768`, one step, guidance scale `1.0`, VAE slicing on, VAE tiling off, and no CPU offload. If 512x768 is unstable: enable VAE slicing, then VAE tiling, verify tokenizer/text encoder/teacher/ControlNet are not loaded, and only then reduce to `384x512`.

## FAQ

- Does inference need a product image? No. The product is fixed and learned during training.
- Does inference load text encoder? No. Use `scripts/export_text_embedding.py` before training/inference.
- Does this use ControlNet? No.
- Why is outside loss against source? Non-edit regions should preserve identity, face, hair, and background.
- Can masks be empty? Utilities and tests handle empty/full masks, but useful training samples should contain edit regions.

## Known Limits

- First version does not implement VAE skip zero-convs; the config reserves `train_vae_skip`.
- No GAN loss is included.
- SD-Turbo license may not fit every commercial setting; review Stability terms before deployment.
- The small unit tests avoid downloading real SD-Turbo weights; run an end-to-end training/inference check after exporting embeddings and downloading the base model.
