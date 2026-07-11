"""Finding 2 — E1 promotion-gate worker encoding gate (D-EVALGATE HOLE 3, subprocess path).

The subprocess promotion-gate worker (``hexo_rl.eval.promotion_gate_worker``) loads candidate + best
checkpoints for the LIVE promotion decision. Pre-fix ``_load_model`` called
``load_model_with_encoding(ckpt, device)`` with NO ``declared_encoding`` — reintroducing the exact
D-EVALGATE "HOLE 3" the in-thread path (``eval_pipeline._load_anchor_model``) already closes: a
stale/foreign checkpoint (the documented ``vast-stale-checkpoint-name-collision`` risk) would silently
CROSS-DECODE into the promotion decision. Latent today (gated behind
``promotion_gate_subprocess_isolation: bool = False``), LIVE the instant run3 flips the flag.

Fix: thread ``declared_encoding=full_config["encoding"]`` into ``_load_model`` for BOTH candidate and
best, matching the in-thread ``_load_anchor_model`` assertion form — a stamp that disagrees with the
run's declared encoding RAISES ``DeclaredEncodingMismatchError`` rather than silently shape-inferring.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hexo_rl.eval.checkpoint_loader import DeclaredEncodingMismatchError
from hexo_rl.eval.promotion_gate_worker import _load_model
from hexo_rl.model.network import HexTacToeNet

_DEVICE = torch.device("cpu")


def _v6_live2_model() -> HexTacToeNet:
    # v6_live2 and v6_live2_ls are state-dict-shape-IDENTICAL (window fields only) so shape
    # inference cannot disambiguate them — the crux of the cross-decode hole.
    return HexTacToeNet(
        board_size=19, in_channels=4, filters=16, res_blocks=2, encoding="v6_live2",
    )


def _save_full_ckpt_stamped(path: Path, encoding_name: str) -> Path:
    """Full checkpoint carrying metadata['encoding_name'] — the real d1m artifact shape."""
    torch.save(
        {"model_state": _v6_live2_model().state_dict(),
         "metadata": {"encoding_name": encoding_name, "schema_version": 1}},
        path,
    )
    return path


def test_load_model_raises_on_stamp_declared_mismatch(tmp_path):
    """THE pin: a checkpoint stamped v6_live2 loaded under the run's declared v6_live2_ls RAISES —
    no silent cross-decode into the promotion decision."""
    ckpt = _save_full_ckpt_stamped(tmp_path / "checkpoint_stamped.pt", "v6_live2")
    with pytest.raises(DeclaredEncodingMismatchError) as excinfo:
        _load_model(str(ckpt), _DEVICE, declared_encoding="v6_live2_ls")
    msg = str(excinfo.value)
    assert "v6_live2" in msg  # the mismatch names both sides


def test_load_model_loads_when_stamp_agrees_with_declared(tmp_path):
    """Agreement path: stamp == declared → loads (the gate is an assertion, not a rejection)."""
    ckpt = _save_full_ckpt_stamped(tmp_path / "checkpoint_ok.pt", "v6_live2_ls")
    model = _load_model(str(ckpt), _DEVICE, declared_encoding="v6_live2_ls")
    assert isinstance(model, HexTacToeNet)


def test_load_model_none_declared_preserves_prior_behaviour(tmp_path):
    """Backward-compat: declared_encoding=None (the pre-fix default) still shape/stamp-loads — the
    fix only ADDS the assertion when the run threads its declared encoding."""
    ckpt = _save_full_ckpt_stamped(tmp_path / "checkpoint_bare.pt", "v6_live2")
    model = _load_model(str(ckpt), _DEVICE)  # no declared_encoding
    assert isinstance(model, HexTacToeNet)


# ── run_worker-level wiring: BOTH candidate and best are gated via full_config["encoding"] ────
def _write_config(path: Path, encoding) -> Path:
    import json
    payload = {"eval_pipeline": {"enabled": True}}
    if encoding is not None:
        payload["encoding"] = encoding
    path.write_text(json.dumps(payload))
    return path


def test_run_worker_gates_best_checkpoint_on_run_encoding(tmp_path):
    """The --best (promotion-lineage best_model) checkpoint is gated too: a best whose stamp
    disagrees with the run's declared encoding RAISES before any eval — the candidate is valid so
    only the best trips the gate (proves best is wired, not just candidate)."""
    from hexo_rl.eval.promotion_gate_worker import run_worker

    good = _save_full_ckpt_stamped(tmp_path / "checkpoint_cand.pt", "v6_live2_ls")
    stale_best = _save_full_ckpt_stamped(tmp_path / "checkpoint_best.pt", "v6_live2")
    cfg = _write_config(tmp_path / "cfg.json", "v6_live2_ls")
    with pytest.raises(DeclaredEncodingMismatchError):
        run_worker([
            "--candidate", str(good), "--best", str(stale_best),
            "--config", str(cfg), "--step", "1",
            "--result", str(tmp_path / "r.jsonl"),
        ])


def test_run_worker_missing_encoding_key_is_backward_compat(tmp_path):
    """A config with NO top-level encoding key → declared_encoding None → NO gate raise on load
    (pre-fix shape/stamp inference). Load succeeds; eval may fail later on the stub config, but the
    ENCODING gate must not fire (byte-pure backward-compat for the pathological unstamped-config case)."""
    from hexo_rl.eval.promotion_gate_worker import run_worker

    stamped = _save_full_ckpt_stamped(tmp_path / "checkpoint_cand.pt", "v6_live2")
    cfg = _write_config(tmp_path / "cfg.json", None)  # NO encoding key
    # The gate must NOT raise DeclaredEncodingMismatchError; any downstream eval error is a different
    # class (the stub eval_pipeline config), which is fine — we only assert the gate stayed silent.
    with pytest.raises(Exception) as excinfo:
        run_worker([
            "--candidate", str(stamped), "--best", "-",
            "--config", str(cfg), "--step", "1",
            "--result", str(tmp_path / "r.jsonl"),
        ])
    assert not isinstance(excinfo.value, DeclaredEncodingMismatchError), (
        "missing-encoding-key config must not trigger the declared-vs-stamp gate (backward-compat)"
    )
