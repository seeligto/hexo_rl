"""Unit tests for resolve_anchor branches (Q-§159b §B item 15)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.anchor import resolve_anchor


def _ext_cfg(best_model_path: str) -> dict:
    return {"eval_pipeline": {"gating": {"best_model_path": best_model_path}}}


def test_resolve_anchor_eval_pipeline_none(tmp_path):
    state = resolve_anchor(
        eval_pipeline=None,
        eval_ext_config=_ext_cfg(str(tmp_path / "best_model.pt")),
        inf_model=MagicMock(),
        trainer=MagicMock(),
        args=MagicMock(checkpoint_dir=str(tmp_path)),
        config={},
        device=torch.device("cpu"),
        board_size=5, res_blocks=1, filters=16,
        in_channels=8, input_channels=None, se_reduction_ratio=4,
    )
    assert state.best_model is None
    assert state.best_model_step is None
    assert isinstance(state.best_model_path, Path)


@patch("hexo_rl.training.anchor.load_best_model_resilient", return_value=None)
@patch("hexo_rl.training.anchor.save_best_model_atomic")
def test_resolve_anchor_fresh_init_no_candidates(mock_save, mock_load, tmp_path):
    device = torch.device("cpu")
    real_model = HexTacToeNet(board_size=5, res_blocks=1, filters=16).to(device)
    # §S181-AUDIT Wave 2 — anchor fresh-init now routes the source
    # state through Trainer.inference_state_dict so EMA weights flow
    # consistently with the rest of the dispatch surface.
    trainer = MagicMock(spec=["model", "step", "inference_state_dict"])
    trainer.model = real_model
    trainer.step = 7
    trainer.inference_state_dict.return_value = real_model.state_dict()

    state = resolve_anchor(
        eval_pipeline=object(),  # not None → eval branch
        eval_ext_config=_ext_cfg(str(tmp_path / "best_model.pt")),
        inf_model=MagicMock(spec=["in_channels", "load_state_dict"]),
        trainer=trainer,
        args=MagicMock(checkpoint_dir=str(tmp_path)),
        config={},
        device=device,
        board_size=5, res_blocks=1, filters=16,
        in_channels=8, input_channels=None, se_reduction_ratio=4,
    )
    assert isinstance(state.best_model, HexTacToeNet)
    assert state.best_model_step == 7
    mock_save.assert_called_once()


@patch("hexo_rl.training.anchor.load_best_model_resilient")
@patch("hexo_rl.training.anchor.save_best_model_atomic")
def test_resolve_anchor_arch_mismatch_skips_sync(mock_save, mock_load, tmp_path):
    device = torch.device("cpu")
    # Anchor loaded with in_channels=18; inf_model has in_channels=8 → no sync.
    anchor_model = MagicMock(spec=["in_channels", "eval", "state_dict"])
    anchor_model.in_channels = 18
    best_ref = MagicMock(spec=["model", "step"])
    best_ref.model = anchor_model
    best_ref.step = 5
    # §D-RERUNPREP F1: the loader returns (trainer, source_path); resolve_anchor
    # hashes the STORED weights at source_path for the W2 identity (not the live
    # model), so a real checkpoint must exist there.
    ckpt = tmp_path / "best_model.pt"
    torch.save(
        {"model_state": HexTacToeNet(board_size=5, res_blocks=1, filters=16).state_dict()},
        ckpt,
    )
    mock_load.return_value = (best_ref, ckpt)

    inf_model = MagicMock(spec=["in_channels", "load_state_dict"])
    inf_model.in_channels = 8

    trainer = MagicMock(spec=["model", "step"])
    trainer.step = 5

    state = resolve_anchor(
        eval_pipeline=object(),
        eval_ext_config=_ext_cfg(str(tmp_path / "best_model.pt")),
        inf_model=inf_model,
        trainer=trainer,
        args=MagicMock(checkpoint_dir=str(tmp_path)),
        config={},
        device=device,
        board_size=5, res_blocks=1, filters=16,
        in_channels=8, input_channels=None, se_reduction_ratio=4,
    )
    inf_model.load_state_dict.assert_not_called()
    assert state.best_model is anchor_model
    assert state.best_model_step == 5


# ── D-FORENSIC F1 (2026-07-02) — anchor-path encoding-mismatch visibility ──


def _stamped_v6_live2_anchor(path):
    """Full-ckpt anchor stamped single-window v6_live2 (the d1m lineage shape)."""
    model = HexTacToeNet(
        board_size=19, in_channels=4, filters=16, res_blocks=2,
        encoding="v6_live2",
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {"in_channels": 4, "encoding": {"version": "v6_live2"}},
            "metadata": {"encoding_name": "v6_live2", "schema_version": 1},
        },
        path,
    )
    return path


_LS_DECLARING_CFG = {
    "batch_size": 8,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "torch_compile": False,
    "res_blocks": 2,
    "filters": 16,
    "in_channels": 4,
    "encoding": "v6_live2_ls",  # declared multi-window vs stamped v6_live2
}


def test_try_load_anchor_encoding_mismatch_raises_with_dedicated_event(tmp_path):
    """Encoding disagreement is a CONFIGURATION error, not corruption: it must
    RAISE (hard-fail the launch, matching the trainer path) after emitting its
    own ERROR event — NOT return None, which routes a valid anchor into the
    corruption-quarantine/fresh-init machinery (F1 review BLOCKER)."""
    import structlog.testing

    from hexo_rl.training.anchor import _try_load_anchor

    candidate = _stamped_v6_live2_anchor(tmp_path / "best_model.pt")
    with structlog.testing.capture_logs() as cap_logs:
        with pytest.raises(ValueError, match="Encoding version disagrees"):
            _try_load_anchor(
                candidate,
                checkpoint_dir=str(tmp_path),
                device=torch.device("cpu"),
                fallback_config=dict(_LS_DECLARING_CFG),
            )
    events = [e.get("event") for e in cap_logs]
    assert "anchor_encoding_mismatch" in events, f"events: {events}"


def test_resilient_load_encoding_mismatch_never_quarantines(tmp_path):
    """load_best_model_resilient must NOT quarantine (rename) or replace a
    VALID anchor whose only problem is an encoding disagreement — the F1
    review reproduced best_model.pt being renamed .corrupt-* and silently
    replaced by a fresh-init net. The ValueError must propagate instead."""
    from hexo_rl.training.anchor import load_best_model_resilient

    best = _stamped_v6_live2_anchor(tmp_path / "best_model.pt")
    original_bytes = best.read_bytes()
    with pytest.raises(ValueError, match="Encoding version disagrees"):
        load_best_model_resilient(
            best,
            checkpoint_dir=str(tmp_path),
            device=torch.device("cpu"),
            config=dict(_LS_DECLARING_CFG),
        )
    # File untouched at its original path — no quarantine, no overwrite.
    assert best.exists(), "best_model.pt was moved/quarantined"
    assert best.read_bytes() == original_bytes, "best_model.pt was rewritten"
    corpses = list(tmp_path.glob("best_model.pt.corrupt-*"))
    assert corpses == [], f"valid anchor quarantined: {corpses}"


# ── GNN-integration WP-4 (C4, contract node 11c) — graph fresh-init ────────


@patch("hexo_rl.training.anchor.load_best_model_resilient", return_value=None)
@patch("hexo_rl.training.anchor.save_best_model_atomic")
def test_resolve_anchor_fresh_init_graph_builds_gnn_net(mock_save, mock_load, tmp_path):
    """Pre-WP-4 this branch built a HexTacToeNet UNCONDITIONALLY — a
    representation=graph config anchored a CNN with no error (contract node
    11c, the SILENT-CORRUPT hole). `resolve_anchor` now resolves the
    encoding from `config` and dispatches through `build_net`."""
    from hexo_rl.model.gnn_net import GnnNet

    device = torch.device("cpu")
    real_model = GnnNet(hidden=16, num_layers=2, policy_hidden=8, value_hidden=4)
    trainer = MagicMock(spec=["model", "step", "inference_state_dict"])
    trainer.model = real_model
    trainer.step = 3
    trainer.inference_state_dict.return_value = real_model.state_dict()

    graph_config = {
        "encoding": "gnn_axis_v1",
        "value_head_type": "dist65",
        "gnn_hidden": 16,
        "gnn_num_layers": 2,
        "gnn_policy_hidden": 8,
        "gnn_value_hidden": 4,
    }

    state = resolve_anchor(
        eval_pipeline=object(),  # not None -> eval branch
        eval_ext_config=_ext_cfg(str(tmp_path / "best_model.pt")),
        inf_model=MagicMock(spec=["in_channels", "load_state_dict"]),
        trainer=trainer,
        args=MagicMock(checkpoint_dir=str(tmp_path)),
        config=graph_config,
        device=device,
        # grid-only geometry — ignored by the graph path (contract §"Load-
        # bearing audit finding": a graph net has no board_size/filters/
        # res_blocks meaning).
        board_size=19, res_blocks=1, filters=16,
        in_channels=8, input_channels=None, se_reduction_ratio=4,
        value_head_type="dist65", n_value_bins=65,
    )
    assert isinstance(state.best_model, GnnNet)
    assert state.best_model.representation.num_layers == 2
    assert state.best_model_step == 3
    mock_save.assert_called_once()


def test_resolve_anchor_fresh_init_empty_config_still_grid(tmp_path):
    """`config={}` (every pre-WP-4 caller) resolves to the v6 grid default —
    unchanged behavior, no regression from threading `config` into
    `build_net`."""
    device = torch.device("cpu")
    real_model = HexTacToeNet(board_size=5, res_blocks=1, filters=16).to(device)
    trainer = MagicMock(spec=["model", "step", "inference_state_dict"])
    trainer.model = real_model
    trainer.step = 1
    trainer.inference_state_dict.return_value = real_model.state_dict()

    with (
        patch("hexo_rl.training.anchor.load_best_model_resilient", return_value=None),
        patch("hexo_rl.training.anchor.save_best_model_atomic"),
    ):
        state = resolve_anchor(
            eval_pipeline=object(),
            eval_ext_config=_ext_cfg(str(tmp_path / "best_model.pt")),
            inf_model=MagicMock(spec=["in_channels", "load_state_dict"]),
            trainer=trainer,
            args=MagicMock(checkpoint_dir=str(tmp_path)),
            config={},
            device=device,
            board_size=5, res_blocks=1, filters=16,
            in_channels=8, input_channels=None, se_reduction_ratio=4,
        )
    assert isinstance(state.best_model, HexTacToeNet)


def test_resilient_load_foreign_bootstrap_mismatch_skips_not_raises(
    tmp_path, monkeypatch,
):
    """The hardcoded _BOOTSTRAP_ANCHOR_CANDIDATES are v6-family by
    construction — a fresh launch of any OTHER encoding on a host where
    such a bootstrap exists (and no best_model.pt yet) must SKIP the
    mismatched foreign candidate and fall through (→ fresh-init at the
    caller), not hard-crash the launch (red-team R3b regression). The
    raise is reserved for the same-lineage best_model.pt/.bak tiers."""
    from hexo_rl.training import anchor as anchor_mod

    # v6-shaped weights-only bootstrap (8-plane) — foreign to the declared
    # v6_live2_ls (4-plane) variant.
    boot_model = HexTacToeNet(
        board_size=19, in_channels=8, filters=16, res_blocks=2, encoding="v6",
    )
    boot = tmp_path / "bootstrap_model_v6.pt"
    torch.save(boot_model.state_dict(), boot)
    monkeypatch.setattr(
        anchor_mod, "_BOOTSTRAP_ANCHOR_CANDIDATES", (str(boot),),
    )

    out = anchor_mod.load_best_model_resilient(
        tmp_path / "best_model.pt",  # does not exist — fresh launch
        checkpoint_dir=str(tmp_path),
        device=torch.device("cpu"),
        config=dict(_LS_DECLARING_CFG),
    )
    assert out is None  # skipped the foreign candidate; caller fresh-inits
    assert boot.exists()  # and the bootstrap file was not touched
