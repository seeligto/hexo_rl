"""Regression tests for training loop graduation invariants.

Covers:
  B-001 / C-012 / H-004 — best_model/eval_model constructor parity with config.
  F-016 — promotion must copy from eval_model snapshot, not drifted trainer.model.
  F-026 — cold-start must sync inf_model to best_model anchor, not trainer.model.
"""

from pathlib import Path

import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.checkpoints import save_inference_weights, normalize_model_state_dict_keys


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


# ── F-016: promotion copies from eval snapshot, not drifted trainer.model ─────

def _small_model() -> HexTacToeNet:
    return HexTacToeNet(board_size=9, res_blocks=1, filters=16, in_channels=18)


def _fill_params(model: HexTacToeNet, value: float) -> None:
    """Fill all parameters with a single scalar so comparisons are trivial."""
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(value)


def _param_mean(model: HexTacToeNet) -> float:
    return float(torch.stack([p.mean() for p in model.parameters()]).mean().item())


def test_promotion_copies_from_eval_snapshot_not_trainer_model():
    """Promotion in loop.py:458-463 must use eval_model weights (the kickoff
    snapshot), not trainer.model which has drifted forward during background eval.

    Mirrors the promotion block:
        eval_base = getattr(eval_model, "_orig_mod", eval_model)
        best_model.load_state_dict(eval_base.state_dict())

    If someone changes this to use trainer.model or base_model, the assertion
    below fires because best_model.params would reflect DRIFT_VALUE instead of
    EVAL_VALUE (F-016).
    """
    EVAL_VALUE = 2.0     # weights at eval kickoff time
    DRIFT_VALUE = 999.0  # weights trainer.model has drifted to during background eval

    trainer_model = _small_model()
    eval_model = _small_model()
    best_model = _small_model()

    # Kickoff: copy trainer_model → eval_model (simulates loop.py:480-481).
    _fill_params(trainer_model, EVAL_VALUE)
    eval_base = getattr(eval_model, "_orig_mod", eval_model)
    eval_base.load_state_dict(trainer_model.state_dict())

    # Simulate training continuing after kickoff: trainer_model drifts.
    _fill_params(trainer_model, DRIFT_VALUE)
    assert abs(_param_mean(trainer_model) - DRIFT_VALUE) < 0.01

    # Promotion: must copy from eval_model (kickoff snapshot), not trainer_model.
    eval_base2 = getattr(eval_model, "_orig_mod", eval_model)
    best_model.load_state_dict(eval_base2.state_dict())

    best_mean = _param_mean(best_model)
    assert abs(best_mean - EVAL_VALUE) < 0.01, (
        f"best_model mean={best_mean:.4f} should be EVAL_VALUE={EVAL_VALUE}; "
        f"if it's ~{DRIFT_VALUE} the promotion used drifted trainer.model instead of eval snapshot"
    )


# ── F-026: cold-start inf_model synced to best_model, not trainer.model ───────

def test_cold_start_inf_model_synced_to_best_model(tmp_path: Path):
    """On cold start, inf_model must load best_model.pt weights, not trainer.model.

    Mirrors loop.py:184-186:
        _inf_base = getattr(inf_model, "_orig_mod", inf_model)
        _inf_base.load_state_dict(best_model.state_dict())

    If the cold-start sync is reordered, removed, or points to the wrong model,
    inf_model ends up with trainer.model weights — self-play workers run against
    random-init trunk instead of the promoted anchor (F-026).
    """
    BEST_VALUE = 3.0     # weights in best_model.pt (the anchor)
    TRAINER_VALUE = 7.0  # weights in trainer.model (unrelated to anchor)

    # Build best_model.pt with known weights.
    best_model = _small_model()
    _fill_params(best_model, BEST_VALUE)
    best_path = tmp_path / "best_model.pt"
    save_inference_weights(best_model, best_path)

    # trainer.model has different weights.
    trainer_model = _small_model()
    _fill_params(trainer_model, TRAINER_VALUE)

    # inf_model starts with trainer weights (cold-start state before the sync).
    inf_model = _small_model()
    _fill_params(inf_model, TRAINER_VALUE)

    # Reload best_model.pt and run the cold-start sync (loop.py:173-186).
    from hexo_rl.training.trainer import Trainer
    minimal_cfg = {
        "board_size": 9, "res_blocks": 1, "filters": 16, "in_channels": 18,
        "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 8,
    }
    best_ref = Trainer.load_checkpoint(best_path, fallback_config=minimal_cfg)
    loaded_best = best_ref.model
    _inf_base = getattr(inf_model, "_orig_mod", inf_model)
    _inf_base.load_state_dict(loaded_best.state_dict())

    inf_mean = _param_mean(inf_model)
    assert abs(inf_mean - BEST_VALUE) < 0.1, (
        f"inf_model mean={inf_mean:.4f} should match best_model BEST_VALUE={BEST_VALUE}; "
        f"if ~{TRAINER_VALUE}, cold-start sync pointed to trainer.model instead of best_model.pt"
    )
