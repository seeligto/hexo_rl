"""§S181-AUDIT Wave 2 — EMA-of-weights unit tests.

Verify build_ema_model wraps the model, EMA update converges to a
running mean of the supplied weight stream, and state_dict structure
matches the raw model so downstream load paths work unchanged.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.ema import (
    DEFAULT_DECAY,
    DEFAULT_UPDATE_EVERY,
    _ema_avg_fn,
    build_ema_model,
    resolve_ema_config,
)
from hexo_rl.training.trainer import Trainer


_FAST_CFG = {
    "board_size":          19,
    "res_blocks":          2,
    "filters":             32,
    "batch_size":          8,
    "lr":                  2e-3,
    "weight_decay":        1e-4,
    "checkpoint_interval": 5,
    "log_interval":        1,
    "torch_compile":       False,
}


# ── avg_fn ──────────────────────────────────────────────────────────────────
def test_ema_avg_fn_converges_to_running_mean():
    """Repeated EMA updates with constant input must approach that input."""
    decay = 0.9
    avg_fn = _ema_avg_fn(decay)
    avg = torch.zeros(4)
    target = torch.ones(4) * 5.0
    for _ in range(200):
        avg = avg_fn(avg, target, torch.tensor(1))
    assert torch.allclose(avg, target, atol=1e-3), (
        f"EMA with decay=0.9 did not converge after 200 steps: avg={avg}"
    )


def test_ema_avg_fn_rejects_invalid_decay():
    with pytest.raises(ValueError):
        _ema_avg_fn(1.0)
    with pytest.raises(ValueError):
        _ema_avg_fn(-0.1)
    # boundary: 0.0 is allowed (degenerate: EMA == current).
    fn = _ema_avg_fn(0.0)
    out = fn(torch.zeros(2), torch.ones(2), torch.tensor(1))
    assert torch.allclose(out, torch.ones(2)), "decay=0 should mean EMA tracks current"


# ── build_ema_model ─────────────────────────────────────────────────────────
def test_build_ema_model_returns_state_dict_matching_raw():
    """EMA shadow state_dict structure matches the raw model's."""
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    ema = build_ema_model(model, decay=0.999)
    raw_keys = set(model.state_dict().keys())
    ema_keys = set(ema.state_dict().keys())
    assert raw_keys == ema_keys, (
        f"EMA state_dict keys mismatch raw model "
        f"(only-in-raw={raw_keys - ema_keys}, only-in-ema={ema_keys - raw_keys})"
    )
    # Module view (compatibility shim) returns the same keys.
    assert set(ema.module.state_dict().keys()) == raw_keys


def test_build_ema_model_drifts_slower_than_raw():
    """After perturbing raw weights, EMA mirrors with a one-step lag bounded by decay."""
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    ema = build_ema_model(model, decay=0.9)

    # Pick a known floating-point weight from both surfaces.
    target_key = "trunk.input_conv.weight"
    assert target_key in model.state_dict()
    initial_raw = model.state_dict()[target_key].detach().clone()
    initial_ema = ema.state_dict()[target_key].detach().clone()
    assert torch.allclose(initial_raw, initial_ema), "EMA starts as a copy of raw"

    # Sharp perturbation on the raw model's tensor (in-place).
    raw_tensor = dict(model.named_parameters())[target_key]
    with torch.no_grad():
        raw_tensor.add_(torch.full_like(raw_tensor, 1.0))

    # One EMA update mixes (1-decay)=0.1 of the new value in.
    ema.update_parameters(model)

    raw_after = model.state_dict()[target_key]
    ema_after = ema.state_dict()[target_key]
    raw_delta = (raw_after - initial_raw).abs().mean()
    ema_delta = (ema_after - initial_ema).abs().mean()
    assert raw_delta > ema_delta, (
        f"EMA must lag raw after a perturbation: raw_delta={raw_delta:.4f} "
        f"vs ema_delta={ema_delta:.4f}"
    )
    # With decay=0.9 + perturbation of +1, EMA should move ~0.1 toward it.
    assert 0.05 <= ema_delta.item() <= 0.20, (
        f"EMA mean delta {ema_delta.item():.4f} outside expected (1-decay) band"
    )


def test_ema_load_into_overwrites_target_model():
    """`load_into` populates the target model's state_dict from EMA shadow."""
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    ema = build_ema_model(model, decay=0.5)

    target_key = "trunk.input_conv.weight"
    # Drift EMA away from raw by perturbing raw + updating EMA a few times.
    raw_tensor = dict(model.named_parameters())[target_key]
    with torch.no_grad():
        raw_tensor.add_(torch.full_like(raw_tensor, 2.0))
    for _ in range(5):
        ema.update_parameters(model)

    # Build a sibling model; load EMA into it; its state should equal EMA shadow.
    sibling = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    ema.load_into(sibling)
    for k, v in ema.state_dict().items():
        assert torch.allclose(sibling.state_dict()[k], v), (
            f"load_into did not propagate EMA state for key {k}"
        )


# ── resolve_ema_config ──────────────────────────────────────────────────────
def test_resolve_ema_config_default_disabled():
    enabled, decay, every = resolve_ema_config({})
    assert enabled is False
    assert decay == DEFAULT_DECAY
    assert every == DEFAULT_UPDATE_EVERY


def test_resolve_ema_config_nested_form():
    cfg = {"ema": {"enabled": True, "decay": 0.995, "update_every": 5}}
    enabled, decay, every = resolve_ema_config(cfg)
    assert enabled is True
    assert decay == 0.995
    assert every == 5


def test_resolve_ema_config_flat_form_backcompat():
    cfg = {"ema_enabled": True, "ema_decay": 0.998, "ema_update_every": 20}
    enabled, decay, every = resolve_ema_config(cfg)
    assert enabled is True
    assert decay == 0.998
    assert every == 20


def test_resolve_ema_config_rejects_zero_update_every():
    with pytest.raises(ValueError):
        resolve_ema_config({"ema": {"enabled": True, "update_every": 0}})


# ── trainer wiring ──────────────────────────────────────────────────────────
def test_trainer_skips_ema_when_disabled(tmp_path: Path):
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, _FAST_CFG, checkpoint_dir=tmp_path)
    assert trainer.ema_model is None
    # inference_state_dict must still return the raw model's state
    raw_keys = set(model.state_dict().keys())
    inf_keys = set(trainer.inference_state_dict().keys())
    assert raw_keys == inf_keys


def test_trainer_builds_ema_when_enabled(tmp_path: Path):
    cfg = dict(_FAST_CFG)
    cfg["ema"] = {"enabled": True, "decay": 0.99, "update_every": 1}
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    assert trainer.ema_model is not None
    assert trainer.ema_update_every == 1
    # inference_state_dict returns EMA module state — keys identical, values
    # at step 0 are identical (EMA was deep-copied from model).
    inf_state = trainer.inference_state_dict()
    raw_state = trainer.model.state_dict()
    assert set(inf_state.keys()) == set(raw_state.keys())
    for k in inf_state:
        assert torch.allclose(inf_state[k], raw_state[k]), (
            f"EMA at step 0 must match raw model parameter {k}"
        )


def test_trainer_ema_diverges_from_raw_after_optim_steps(tmp_path: Path):
    """Run a few cheap training steps; EMA params should drift slower than raw."""
    cfg = dict(_FAST_CFG)
    cfg["ema"] = {"enabled": True, "decay": 0.9, "update_every": 1}
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    raw_param = next(iter(trainer.model.parameters()))
    initial_raw = raw_param.detach().clone()

    # Synthesize gradients + step the optimizer 20× without going through the
    # full training batch pipeline. Hand-rolled so the test stays milliseconds.
    with torch.no_grad():
        raw_param.add_(torch.full_like(raw_param, 0.5))
    for _ in range(20):
        # Drive the optimizer directly so we don't need a batch.
        for p in trainer.model.parameters():
            p.grad = torch.full_like(p, 0.01)
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        trainer.step += 1
        if trainer.ema_model is not None and trainer.step % trainer.ema_update_every == 0:
            trainer.ema_model.update_parameters(
                getattr(trainer.model, "_orig_mod", trainer.model)
            )

    ema_param = next(iter(trainer.ema_model.module.parameters()))
    raw_delta = (raw_param - initial_raw).abs().mean().item()
    ema_delta = (ema_param - initial_raw).abs().mean().item()
    assert ema_delta < raw_delta, (
        f"EMA drift {ema_delta:.4f} must be smaller than raw drift {raw_delta:.4f}"
    )
