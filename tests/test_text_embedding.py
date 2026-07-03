from __future__ import annotations

import pytest
import torch

from models.text_embedding import load_text_embedding


def test_missing_text_embedding_error_includes_export_command(tmp_path) -> None:
    missing = tmp_path / "fixed_prompt.pt"

    with pytest.raises(FileNotFoundError) as exc:
        load_text_embedding(
            missing,
            torch.device("cpu"),
            torch.float32,
            base_model="stabilityai/sd-turbo",
            prompt="undress, nsfw, nude, naked",
        )

    message = str(exc.value)
    assert str(missing) in message
    assert "uv run python -m scripts.export_text_embedding" in message
    assert "--base-model stabilityai/sd-turbo" in message
    assert "--output" in message
