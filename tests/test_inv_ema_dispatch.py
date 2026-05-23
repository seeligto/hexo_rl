"""§S181-AUDIT Wave 2 INV pin: EMA dispatch routes through Trainer.

Regression guard. When `training.ema.enabled = true`, every site that
synchronises weights from the trainer to a downstream consumer
(self-play InferenceServer, eval model, fresh best-model anchor) must
go through `Trainer.inference_state_dict()` so the EMA module's state
is what self-play sees — not the raw training-step weights.

If any caller bypasses the accessor and reads `trainer.model.state_dict()`
directly while EMA is on, this test breaks immediately and the bisect
surface is clear. The structural invariants this guards:

  1. `Trainer.inference_state_dict` exists and returns EMA module state
     when EMA is enabled (raw state when disabled).
  2. EMA-enabled inference state diverges from raw training-model state
     after a few optimizer steps (otherwise the dispatch routing is
     silently a no-op).
  3. None of the known dispatch call sites still reference the raw
     model's state_dict for inference/eval/promotion when the accessor
     exists (grep guard).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hexo_rl.model.network import HexTacToeNet
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


def test_trainer_exposes_inference_state_dict_accessor(tmp_path: Path):
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, _FAST_CFG, checkpoint_dir=tmp_path)
    assert hasattr(trainer, "inference_state_dict"), (
        "Trainer.inference_state_dict must exist — §S181-AUDIT Wave 2 dispatch "
        "centralises raw-vs-EMA routing through this accessor."
    )
    state = trainer.inference_state_dict()
    assert isinstance(state, dict) and len(state) > 0


def test_inference_state_dict_returns_raw_when_ema_disabled(tmp_path: Path):
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, _FAST_CFG, checkpoint_dir=tmp_path)
    raw = trainer.model.state_dict()
    inf = trainer.inference_state_dict()
    for k in raw:
        assert torch.allclose(raw[k], inf[k]), (
            f"With EMA disabled, inference_state_dict must mirror raw model "
            f"parameter {k}"
        )


def test_inference_state_dict_diverges_from_raw_when_ema_enabled(tmp_path: Path):
    """After optimizer steps, inference_state_dict (EMA) must differ from raw."""
    cfg = dict(_FAST_CFG)
    cfg["ema"] = {"enabled": True, "decay": 0.9, "update_every": 1}
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    assert trainer.ema_model is not None

    raw_param = next(iter(trainer.model.parameters()))
    initial = raw_param.detach().clone()
    # Synthesize a few optimizer steps with non-trivial gradients.
    for _ in range(15):
        for p in trainer.model.parameters():
            p.grad = torch.full_like(p, 0.02)
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        trainer.step += 1
        if trainer.step % trainer.ema_update_every == 0:
            _base = getattr(trainer.model, "_orig_mod", trainer.model)
            trainer.ema_model.update_parameters(_base)

    raw = trainer.model.state_dict()
    inf = trainer.inference_state_dict()
    # At least one parameter must differ — otherwise the EMA routing is inert.
    any_differ = any(
        not torch.allclose(raw[k], inf[k], atol=1e-7)
        for k in raw if raw[k].dtype.is_floating_point
    )
    assert any_differ, (
        "With EMA enabled and optimizer steps applied, inference_state_dict "
        "must diverge from raw model state. Routing through the accessor is "
        "load-bearing; if this fails, EMA is silently a no-op."
    )

    # And the EMA must have moved less than raw (i.e. it still tracks the
    # initial point more closely than raw does).
    raw_delta = (raw_param - initial).abs().mean().item()
    ema_delta = (
        trainer.ema_model.module.state_dict()[
            next(iter(trainer.model.state_dict().keys()))
        ] - initial
    ).abs().mean().item()
    assert ema_delta < raw_delta, (
        f"EMA delta {ema_delta:.4f} must be smaller than raw delta {raw_delta:.4f}"
    )


def test_dispatch_callers_use_inference_state_dict_accessor():
    """Grep-guard: known dispatch sites must reference inference_state_dict.

    Update the SITES list when adding a new dispatch caller. The point is to
    fail loudly if someone reverts the routing back to a raw state_dict call.
    """
    repo = Path(__file__).resolve().parents[1]
    sites = {
        repo / "hexo_rl" / "training" / "lifecycle.py": "build_inference_model",
        repo / "hexo_rl" / "training" / "step_coordinator.py": "eval kickoff",
        repo / "hexo_rl" / "training" / "anchor.py": "fresh anchor init",
    }
    missing = []
    for path, label in sites.items():
        text = path.read_text()
        if "inference_state_dict" not in text:
            missing.append(f"{path.name} ({label})")
    assert not missing, (
        "These dispatch sites no longer route through "
        "Trainer.inference_state_dict — EMA weights will not reach self-play "
        f"/ eval / promotion: {missing}"
    )
