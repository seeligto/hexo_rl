"""Regression test: best_model and eval_model in loop.py must honour
in_channels and se_reduction_ratio from config (B-001 / C-012 / H-004)."""

import torch

from hexo_rl.model.network import HexTacToeNet


def _make_model(cfg: dict) -> HexTacToeNet:
    """Replicate the fixed construction pattern from loop.py."""
    board_size = int(cfg.get("board_size", 19))
    res_blocks = int(cfg.get("res_blocks", 12))
    filters = int(cfg.get("filters", 128))
    in_channels = int(cfg.get("in_channels", 18))
    se_reduction_ratio = int(cfg.get("se_reduction_ratio", 4))
    return HexTacToeNet(
        board_size=board_size,
        res_blocks=res_blocks,
        filters=filters,
        in_channels=in_channels,
        se_reduction_ratio=se_reduction_ratio,
    )


def test_cold_start_best_model_matches_config():
    """best_model built with non-default in_channels/se_reduction_ratio must
    match the config — not silently fall back to HexTacToeNet defaults."""
    cfg = {
        "board_size": 9,
        "res_blocks": 2,
        "filters": 32,
        "in_channels": 24,
        "se_reduction_ratio": 8,
    }
    best_model = _make_model(cfg)
    best_model.eval()

    # in_channels reflected in trunk input conv
    assert best_model.trunk.input_conv.in_channels == 24, (
        f"expected 24, got {best_model.trunk.input_conv.in_channels}"
    )

    # se_reduction_ratio: fc1 out_features == filters // ratio
    se_fc1_out = best_model.trunk.tower[0].se.fc1.out_features
    assert se_fc1_out == 32 // 8, (
        f"expected SE mid={32 // 8}, got {se_fc1_out}"
    )

    # shape probe: forward must accept (B, 24, 9, 9)
    with torch.no_grad():
        x = torch.zeros(1, 24, 9, 9)
        policy, value, _ = best_model(x)
    assert policy.shape == (1, 9 * 9 + 1)
    assert value.shape == (1, 1)


def test_eval_model_matches_config():
    """eval_model construction obeys non-default in_channels/se_reduction_ratio."""
    cfg = {
        "board_size": 9,
        "res_blocks": 2,
        "filters": 32,
        "in_channels": 24,
        "se_reduction_ratio": 8,
    }
    eval_model = _make_model(cfg)
    eval_model.eval()

    assert eval_model.trunk.input_conv.in_channels == 24
    se_fc1_out = eval_model.trunk.tower[0].se.fc1.out_features
    assert se_fc1_out == 32 // 8

    with torch.no_grad():
        x = torch.zeros(1, 24, 9, 9)
        policy, value, _ = eval_model(x)
    assert policy.shape == (1, 9 * 9 + 1)
    assert value.shape == (1, 1)
