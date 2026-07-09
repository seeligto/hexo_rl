"""T-MISSTAMP: mis-stamped ckpt -> load_model_with_encoding(..., decode_override='v6_live2_ls') raises."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch


CKPT_PATH = Path("/home/timmy/Work/Hexo/hexo_rl/checkpoints/run2_retro/checkpoint_00050000.pt")


def test_misstamp_raises_on_declared_encoding_mismatch(tmp_path):
    """Corrupt the metadata['encoding_name'] stamp to v6w25 -> declared_encoding check raises."""
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    # Corrupt the stamp: set metadata['encoding_name'] to wrong encoding
    ck.setdefault("metadata", {})["encoding_name"] = "v6w25"
    # Also corrupt config to avoid multi-source agreement
    if isinstance(ck.get("config", {}).get("encoding"), dict):
        ck["config"]["encoding"] = {"version": "v6w25"}
    elif "encoding" in ck.get("config", {}):
        ck["config"]["encoding"] = "v6w25"

    bad_path = tmp_path / "bad_stamp.pt"
    torch.save(ck, bad_path)

    import sys
    sys.path.insert(0, "/home/timmy/Work/Hexo/hexo_rl")
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding

    device = torch.device("cpu")
    with pytest.raises(Exception):
        # declared_encoding="v6_live2_ls" but stamp says "v6w25" -> must raise
        load_model_with_encoding(str(bad_path), device, declared_encoding="v6_live2_ls")


def test_clean_ckpt_loads_with_declared_encoding(tmp_path):
    """A correctly-stamped checkpoint loads without error under declared_encoding='v6_live2_ls'."""
    import sys
    sys.path.insert(0, "/home/timmy/Work/Hexo/hexo_rl")
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding

    device = torch.device("cpu")
    # Should not raise
    model, spec, label = load_model_with_encoding(
        str(CKPT_PATH), device, declared_encoding="v6_live2_ls"
    )
    assert model is not None
    assert label == "v6_live2_ls"
