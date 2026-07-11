"""Task 4 — value_head_type + n_value_bins config plumbing.

Verifies:
- MODEL_HPARAM_DEFAULTS has the new keys with correct defaults.
- Building HexTacToeNet with value_head_type=dist65 via the config defaults gives a
  dist net with 65-bin classifier.
- An invalid value_head_type raises ValueError.
- configs/model.yaml has the key (defaults to scalar).
"""
from __future__ import annotations

import pytest
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.model_defaults import MODEL_HPARAM_DEFAULTS


def test_model_defaults_has_value_head_type():
    assert "value_head_type" in MODEL_HPARAM_DEFAULTS
    assert MODEL_HPARAM_DEFAULTS["value_head_type"] == "scalar"


def test_model_defaults_has_n_value_bins():
    assert "n_value_bins" in MODEL_HPARAM_DEFAULTS
    assert MODEL_HPARAM_DEFAULTS["n_value_bins"] == 65


def test_build_dist65_from_config_overrides():
    """Simulate what pretrain_cli / lifecycle does: read defaults, override, build."""
    cfg = dict(MODEL_HPARAM_DEFAULTS)
    cfg["value_head_type"] = "dist65"
    # Use a small net for speed
    net = HexTacToeNet(
        filters=32, res_blocks=1, encoding="v6_live2_ls",
        value_head_type=cfg["value_head_type"],
        n_value_bins=cfg["n_value_bins"],
    )
    assert net.value_head_type == "dist65"
    assert net.value_fc2_bins is not None
    assert net.value_fc2_bins.out_features == 65


def test_scalar_default_has_no_bins_layer():
    """Default config (scalar) must NOT build value_fc2_bins."""
    cfg = dict(MODEL_HPARAM_DEFAULTS)
    net = HexTacToeNet(
        filters=32, res_blocks=1, encoding="v6_live2_ls",
        value_head_type=cfg["value_head_type"],
    )
    assert net.value_head_type == "scalar"
    assert not hasattr(net, "value_fc2_bins") or net.value_fc2_bins is None


def test_invalid_value_head_type_raises():
    with pytest.raises(ValueError, match="value_head_type"):
        HexTacToeNet(
            filters=32, res_blocks=1, encoding="v6_live2_ls",
            value_head_type="bogus",
        )


def test_model_yaml_has_value_head_type_default():
    """configs/model.yaml must contain value_head_type defaulting to scalar."""
    from pathlib import Path
    import yaml
    model_yaml = Path(__file__).parents[3] / "configs" / "model.yaml"
    cfg = yaml.safe_load(model_yaml.read_text())
    assert "value_head_type" in cfg, "configs/model.yaml missing value_head_type"
    assert cfg["value_head_type"] == "scalar"
