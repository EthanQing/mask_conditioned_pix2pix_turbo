# Changelog

## 2026-07-03

- Added a clear error for missing fixed text embeddings, including the exact export command to run before training, validation, inference, or VRAM benchmarking.
- Unified the default fixed prompt across training, inference, and text embedding export.
- Updated README commands to use `uv run python -m scripts...` and corrected the documented default gradient accumulation value.
