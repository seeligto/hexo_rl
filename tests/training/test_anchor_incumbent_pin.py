"""W2 (§D-LOOPFIX) — incumbent pinning.

``resolve_anchor`` asserts the LOADED anchor's state-dict sha256 against the run
config's pinned expectation (``eval_pipeline.gating.expected_anchor_sha256``) and
logs an identity line (sha + path + step + run_id). A planted foreign incumbent —
e.g. the silent ``.bak`` restore that installed golong@50k-PEAK as the A/B's
anchor + self-play generator (promogate W2) — fails the preflight loudly instead
of running 25k games against the wrong weights.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.anchor import (
    checkpoint_state_sha256,
    resolve_anchor,
    state_dict_sha256,
)


def _model(seed: int) -> HexTacToeNet:
    torch.manual_seed(seed)
    return HexTacToeNet(board_size=5, res_blocks=1, filters=16)


def _ext_cfg(best_model_path, pin=None):
    gating = {"best_model_path": str(best_model_path)}
    if pin is not None:
        gating["expected_anchor_sha256"] = pin
    return {"eval_pipeline": {"gating": gating}}


def _resolve(ext, tmp_path, anchor_model, run_id="r1", source_path=None):
    inf_model = MagicMock(spec=["in_channels", "load_state_dict"])
    inf_model.in_channels = 8
    trainer = MagicMock(spec=["model", "step"])
    trainer.step = 0
    best_ref = MagicMock(spec=["model", "step"])
    best_ref.model = anchor_model
    best_ref.step = 50000
    # load_best_model_resilient returns (trainer, source_path) — resolve_anchor
    # hashes the STORED weights at source_path for the W2 pin (§D-RERUNPREP F1).
    src = Path(source_path) if source_path is not None else (tmp_path / "best_model.pt")
    with patch("hexo_rl.training.anchor.load_best_model_resilient",
               return_value=(best_ref, src)), \
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


def test_checkpoint_state_sha256_hashes_stored_weights_not_live_dtype(tmp_path):
    """§D-RERUNPREP F1 regression: the W2 identity must hash the STORED (fp32)
    weights so it equals the anchor_sha256.py pin regardless of the runtime
    model's dtype. The Phase-3 GPU smoke caught resolve_anchor hashing the LIVE
    model, which on CUDA is fp16 → a byte-copy of the pinned bootstrap
    false-failed the pin (4198d5cb… ≠ aba28e10…)."""
    anchor = _model(7)
    ckpt = tmp_path / "anchor.pt"
    torch.save({"model_state": anchor.state_dict()}, ckpt)
    # canonical source hash == hash of the stored fp32 weights (== anchor_sha256.py)
    assert checkpoint_state_sha256(ckpt) == state_dict_sha256(anchor.state_dict())
    # …and is NOT the (lossy) hash of an fp16 cast — the divergence the bug rode in on
    fp16 = {
        k: (v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
        for k, v in anchor.state_dict().items()
    }
    assert checkpoint_state_sha256(ckpt) != state_dict_sha256(fp16)


def test_pin_match_passes(tmp_path):
    anchor = _model(7)
    ckpt = tmp_path / "best_model.pt"
    torch.save({"model_state": anchor.state_dict()}, ckpt)
    pin = checkpoint_state_sha256(ckpt)
    state = _resolve(_ext_cfg(ckpt, pin), tmp_path, anchor, source_path=ckpt)
    assert state.best_model is anchor


def test_pin_matches_source_weights_when_runtime_anchor_is_fp16(tmp_path):
    """§D-RERUNPREP F1 regression: the pin is computed from the fp32 checkpoint
    (anchor_sha256.py); on CUDA the resolved anchor model is fp16. resolve_anchor
    must verify the pin against the SOURCE weights, not the live fp16 model — else
    a CORRECT incumbent (a byte-copy of the pinned bootstrap) false-fails the
    launch, as the Phase-3 GPU smoke caught (4198d5cb… ≠ aba28e10…)."""
    anchor = _model(7)
    ckpt = tmp_path / "best_model.pt"
    torch.save({"model_state": anchor.state_dict()}, ckpt)
    pin = checkpoint_state_sha256(ckpt)          # production pin: fp32 file sha
    fp16_live = _model(7).half()                  # the resolved anchor as fp16 (CUDA)
    state = _resolve(_ext_cfg(ckpt, pin), tmp_path, fp16_live, source_path=ckpt)
    assert state.best_model is fp16_live          # no RuntimeError — correct incumbent accepted


def test_planted_wrong_incumbent_fails_loudly(tmp_path):
    wrong = _model(7)         # the LOADED incumbent on disk (stale golong stand-in)
    intended = _model(99)     # the run's intended bootstrap
    ckpt = tmp_path / "best_model.pt"
    torch.save({"model_state": wrong.state_dict()}, ckpt)   # source = wrong weights
    pin = state_dict_sha256(intended.state_dict())          # pin = intended weights
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        _resolve(_ext_cfg(ckpt, pin), tmp_path, wrong, source_path=ckpt)


def test_fresh_init_verifies_pin_against_checkpoint_source(tmp_path):
    """§D-RERUNPREP F1 (W2-VACUOUS): the runbook's preflight `rm best_model.pt`
    routes EVERY launch through the fresh-init branch, where the existing-anchor
    pin check never ran — so the pin gave the real launch ZERO protection. The
    fresh-init path must verify the pin against the --checkpoint the anchor is
    seeded from."""
    from hexo_rl.training.anchor import verify_launch_anchor_pin

    intended = _model(7)
    ckpt = tmp_path / "bootstrap.pt"
    torch.save({"model_state": intended.state_dict()}, ckpt)
    pin = checkpoint_state_sha256(ckpt)
    # correct --checkpoint matches the pin → no raise
    verify_launch_anchor_pin(
        eval_ext_config=_ext_cfg(tmp_path / "best_model.pt", pin),
        checkpoint_path=ckpt, trainer_step=0, run_id="r1",
    )
    # WRONG --checkpoint (different weights) → hard-fail
    wrong = tmp_path / "wrong.pt"
    torch.save({"model_state": _model(99).state_dict()}, wrong)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        verify_launch_anchor_pin(
            eval_ext_config=_ext_cfg(tmp_path / "best_model.pt", pin),
            checkpoint_path=wrong, trainer_step=0, run_id="r1",
        )
    # no pin configured → never raises (even with no checkpoint source)
    verify_launch_anchor_pin(
        eval_ext_config=_ext_cfg(tmp_path / "best_model.pt"),
        checkpoint_path=None, trainer_step=0, run_id="r1",
    )
    # pin SET but no verifiable --checkpoint → fail CLOSED (the launch incumbent
    # would be unverified, which is the very W2 hole §D-RERUNPREP F1 set out to close).
    with pytest.raises(RuntimeError, match="UNVERIFIED|no readable"):
        verify_launch_anchor_pin(
            eval_ext_config=_ext_cfg(tmp_path / "best_model.pt", pin),
            checkpoint_path=None, trainer_step=0, run_id="r1",
        )


def test_no_pin_does_not_raise(tmp_path):
    anchor = _model(7)
    ckpt = tmp_path / "best_model.pt"
    torch.save({"model_state": anchor.state_dict()}, ckpt)
    state = _resolve(_ext_cfg(ckpt), tmp_path, anchor, source_path=ckpt)
    assert state.best_model is anchor
