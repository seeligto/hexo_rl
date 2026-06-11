"""W2 (§D-LOOPFIX) — incumbent pinning.

``resolve_anchor`` asserts the LOADED anchor's state-dict sha256 against the run
config's pinned expectation (``eval_pipeline.gating.expected_anchor_sha256``) and
logs an identity line (sha + path + step + run_id). A planted foreign incumbent —
e.g. the silent ``.bak`` restore that installed golong@50k-PEAK as the A/B's
anchor + self-play generator (promogate W2) — fails the preflight loudly instead
of running 25k games against the wrong weights.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.anchor import resolve_anchor, state_dict_sha256


def _model(seed: int) -> HexTacToeNet:
    torch.manual_seed(seed)
    return HexTacToeNet(board_size=5, res_blocks=1, filters=16)


def _ext_cfg(best_model_path, pin=None):
    gating = {"best_model_path": str(best_model_path)}
    if pin is not None:
        gating["expected_anchor_sha256"] = pin
    return {"eval_pipeline": {"gating": gating}}


def _resolve(ext, tmp_path, anchor_model, run_id="r1"):
    inf_model = MagicMock(spec=["in_channels", "load_state_dict"])
    inf_model.in_channels = 8
    trainer = MagicMock(spec=["model", "step"])
    trainer.step = 0
    best_ref = MagicMock(spec=["model", "step"])
    best_ref.model = anchor_model
    best_ref.step = 50000
    with patch("hexo_rl.training.anchor.load_best_model_resilient", return_value=best_ref), \
         patch("hexo_rl.training.anchor.save_best_model_atomic"):
        return resolve_anchor(
            eval_pipeline=object(), eval_ext_config=ext, inf_model=inf_model,
            trainer=trainer, args=MagicMock(checkpoint_dir=str(tmp_path)),
            config={}, device=torch.device("cpu"),
            board_size=5, res_blocks=1, filters=16,
            in_channels=8, input_channels=None, se_reduction_ratio=4, run_id=run_id,
        )


def test_state_dict_sha256_distinguishes_weights():
    a = _model(1)
    b = _model(2)
    b.load_state_dict(a.state_dict())  # b now byte-identical to a
    assert state_dict_sha256(a.state_dict()) == state_dict_sha256(b.state_dict())
    c = _model(2)  # different init → different weights
    assert state_dict_sha256(c.state_dict()) != state_dict_sha256(a.state_dict())


def test_state_dict_sha256_invariant_to_compile_prefix():
    """A torch.compiled checkpoint stores ``_orig_mod.`` / ``module.`` key prefixes;
    resolve_anchor hashes the UNWRAPPED model.state_dict() while scripts/anchor_sha256.py
    hashes the raw extract_model_state. The hash must be canonical w.r.t. compile
    wrapping or a compiled-checkpoint pin spuriously hard-fails the launch."""
    a = _model(3)
    plain = a.state_dict()
    wrapped = {f"_orig_mod.{k}": v for k, v in plain.items()}
    nested = {f"module._orig_mod.{k}": v for k, v in plain.items()}
    assert state_dict_sha256(wrapped) == state_dict_sha256(plain)
    assert state_dict_sha256(nested) == state_dict_sha256(plain)


def test_pin_match_passes(tmp_path):
    anchor = _model(7)
    (tmp_path / "best_model.pt").write_bytes(b"x")  # exists → skip fallback-persist
    pin = state_dict_sha256(anchor.state_dict())
    state = _resolve(_ext_cfg(tmp_path / "best_model.pt", pin), tmp_path, anchor)
    assert state.best_model is anchor


def test_planted_wrong_incumbent_fails_loudly(tmp_path):
    anchor = _model(7)        # the LOADED incumbent (stale golong stand-in)
    intended = _model(99)     # the run's intended bootstrap
    (tmp_path / "best_model.pt").write_bytes(b"x")
    pin = state_dict_sha256(intended.state_dict())
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        _resolve(_ext_cfg(tmp_path / "best_model.pt", pin), tmp_path, anchor)


def test_no_pin_does_not_raise(tmp_path):
    anchor = _model(7)
    (tmp_path / "best_model.pt").write_bytes(b"x")
    state = _resolve(_ext_cfg(tmp_path / "best_model.pt"), tmp_path, anchor)
    assert state.best_model is anchor
