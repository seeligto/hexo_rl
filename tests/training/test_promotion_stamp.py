"""W3 (§D-LOOPFIX) — promotion save stamps step + run_id so a promoted anchor is
log- and filename-distinguishable from the bootstrap, and its step survives reload.

Pre-fix (promogate W3): ``save_best_model_atomic`` wrote a bare ``state_dict``;
``Trainer.load_checkpoint``'s ``is_full_ckpt`` gate (requires
optimizer/scaler/config/step all present) left ``trainer.step = 0`` for the partial
anchor payload → every promoted golong anchor logged ``best_model_loaded step=0``,
indistinguishable from a freshly-copied bootstrap (mis-routed two investigations).
"""
from __future__ import annotations

import json

import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.anchor import save_best_model_atomic
from hexo_rl.training.trainer import Trainer

# Real arch (board_size 19, 8-plane v6) so the encoding-aware loader's
# board_size propagation is consistent with the saved state_dict.
_CFG = {
    "board_size": 19, "res_blocks": 2, "filters": 32, "batch_size": 8,
    "lr": 1e-3, "weight_decay": 1e-4, "checkpoint_interval": 5,
    "log_interval": 1, "torch_compile": False,
}


def _model() -> HexTacToeNet:
    return HexTacToeNet(board_size=19, res_blocks=2, filters=32)


def test_save_stamps_step_run_id_promoted(tmp_path):
    path = tmp_path / "best_model.pt"
    save_best_model_atomic(_model(), path, step=25000, run_id="e928c854", encoding="v6")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    assert raw["step"] == 25000
    assert raw["run_id"] == "e928c854"
    assert raw["promoted"] is True
    assert raw["metadata"]["encoding_name"] == "v6"
    # model weights still recoverable under a wrapper key
    assert any(k.endswith("trunk.input_conv.weight") for k in raw["model_state"])


def test_save_without_step_is_bare_state_dict_backcompat(tmp_path):
    path = tmp_path / "best_model.pt"
    save_best_model_atomic(_model(), path)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert "step" not in raw and "model_state" not in raw
    assert "trunk.input_conv.weight" in raw


def test_load_checkpoint_recovers_stamped_step(tmp_path):
    """The W3 regression: a stamped partial anchor's step survives reload (was 0)."""
    path = tmp_path / "best_model.pt"
    save_best_model_atomic(_model(), path, step=25000, run_id="e928c854", encoding="v6")
    ref = Trainer.load_checkpoint(path, checkpoint_dir=tmp_path, fallback_config=_CFG)
    assert ref.step == 25000


def test_save_writes_provenance_sidecar(tmp_path):
    path = tmp_path / "best_model.pt"
    save_best_model_atomic(_model(), path, step=25000, run_id="e928c854", encoding="v6")
    sidecar = path.with_name(path.name + ".provenance.json")
    assert sidecar.exists()
    prov = json.loads(sidecar.read_text())
    assert prov["step"] == 25000
    assert prov["run_id"] == "e928c854"
    assert prov["promoted"] is True
    assert prov["encoding"] == "v6"
