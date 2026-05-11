"""Trainer.load_checkpoint encoding-reconciliation tests (§171 P3 unblock).

bootstrap_model_v6w25.pt is a weights-only payload (no `config` key). The
trainer must infer the encoding from the state-dict shape, surface it
into the in-memory config dict (so downstream selfplay sees the right
board_size / cluster_window_size / cluster_threshold), and refuse to
silently override an explicit config that disagrees.

Run with: .venv/bin/python -m pytest tests/test_trainer_encoding_load.py -xvs
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pytest
import torch

from hexo_rl.encoding import resolve_from_config
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.trainer import Trainer
from hexo_rl.utils.encoding import v6_spec, v6w25_spec


_FAST_RES_BLOCKS = 2
_FAST_FILTERS = 16


def _save_weights_only(model: HexTacToeNet, path: Path) -> Path:
    """Mirror bootstrap_model_v6w25.pt format: bare state_dict."""
    torch.save(model.state_dict(), path)
    return path


def _make_v6_ckpt(tmp_path: Path) -> Path:
    """v6: 8-channel 19×19 trunk, policy_fc out=362."""
    model = HexTacToeNet(
        board_size=19,
        in_channels=8,
        filters=_FAST_FILTERS,
        res_blocks=_FAST_RES_BLOCKS,
        encoding="v6",
    )
    return _save_weights_only(model, tmp_path / "bootstrap_model_v6.pt")


def _make_v6w25_ckpt(tmp_path: Path) -> Path:
    """v6w25: 8-channel 25×25 trunk, policy_fc out=626.

    Filename mirrors the production artifact so the disambiguator's
    filename heuristic is also exercised.
    """
    model = HexTacToeNet(
        board_size=25,
        in_channels=8,
        filters=_FAST_FILTERS,
        res_blocks=_FAST_RES_BLOCKS,
        encoding="v6w25",
    )
    return _save_weights_only(model, tmp_path / "bootstrap_model_v6w25.pt")


def _base_train_cfg() -> dict:
    """Minimal config the trainer needs to construct optimizer / scaler / scheduler."""
    return {
        "batch_size": 8,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "checkpoint_interval": 5,
        "log_interval": 1,
        "torch_compile": False,
        "res_blocks": _FAST_RES_BLOCKS,
        "filters": _FAST_FILTERS,
        "in_channels": 8,
    }


def test_load_v6w25_checkpoint_resolves_to_v6w25(tmp_path: Path) -> None:
    """v6w25 ckpt + v6w25 config → trainer.config gets v6w25 numerics."""
    ckpt_path = _make_v6w25_ckpt(tmp_path)
    cfg = {
        **_base_train_cfg(),
        "encoding": {"version": "v6w25"},
    }
    trainer = Trainer.load_checkpoint(
        ckpt_path,
        checkpoint_dir=tmp_path,
        fallback_config=cfg,
    )
    assert resolve_from_config(trainer.config).trunk_size == 25
    assert trainer.config["cluster_window_size"] == 25
    assert trainer.config["cluster_threshold"] == 8
    assert trainer.config["encoding"]["version"] == "v6w25"
    # Spec sanity — must agree with hexo_rl.utils.encoding's v6w25 helper
    # on the perception fields (canvas board_size differs by design and
    # is overridden to the trunk dimension; that's the whole point).
    spec = v6w25_spec()
    assert trainer.config["cluster_window_size"] == spec.cluster_window_size
    assert trainer.config["cluster_threshold"] == spec.cluster_threshold


def test_load_v6_checkpoint_resolves_to_v6(tmp_path: Path) -> None:
    """v6 ckpt + v6 config (or unset) → trainer.config keeps v6 numerics."""
    ckpt_path = _make_v6_ckpt(tmp_path)
    cfg = {
        **_base_train_cfg(),
        "encoding": {"version": "v6"},
    }
    trainer = Trainer.load_checkpoint(
        ckpt_path,
        checkpoint_dir=tmp_path,
        fallback_config=cfg,
    )
    assert resolve_from_config(trainer.config).trunk_size == 19
    assert trainer.config["cluster_window_size"] == 19
    assert trainer.config["cluster_threshold"] == 5
    assert trainer.config["encoding"]["version"] == "v6"
    spec = v6_spec()
    assert trainer.config["cluster_window_size"] == spec.cluster_window_size
    assert trainer.config["cluster_threshold"] == spec.cluster_threshold


def test_load_v6_checkpoint_with_no_encoding_section_backward_compat(tmp_path: Path) -> None:
    """v6 ckpt + minimal v6-default config (no `encoding` key, no pins) → loads cleanly.

    Backward-compat for legacy variants that pre-date the encoding section.
    """
    ckpt_path = _make_v6_ckpt(tmp_path)
    cfg = _base_train_cfg()  # no encoding section, no board_size pin
    trainer = Trainer.load_checkpoint(
        ckpt_path,
        checkpoint_dir=tmp_path,
        fallback_config=cfg,
    )
    assert resolve_from_config(trainer.config).trunk_size == 19
    assert trainer.config["encoding"]["version"] == "v6"


def test_config_encoding_mismatch_raises_valueerror(tmp_path: Path) -> None:
    """v6w25 ckpt + v6 config → ValueError naming both source values."""
    ckpt_path = _make_v6w25_ckpt(tmp_path)
    cfg = {
        **_base_train_cfg(),
        "encoding": {"version": "v6"},
    }
    with pytest.raises(ValueError) as excinfo:
        Trainer.load_checkpoint(
            ckpt_path,
            checkpoint_dir=tmp_path,
            fallback_config=cfg,
        )
    msg = str(excinfo.value)
    assert "v6" in msg
    assert "v6w25" in msg
    # Both sources must be named — the in-memory config and the checkpoint.
    assert "in-memory config" in msg or "config[" in msg
    assert "checkpoint" in msg or "state_dict" in msg or str(ckpt_path) in msg


def test_config_encoding_mismatch_via_board_size_pin_raises(tmp_path: Path) -> None:
    """v6w25 ckpt + config pinning board_size=19 (no encoding section) → ValueError."""
    ckpt_path = _make_v6w25_ckpt(tmp_path)
    cfg = {
        **_base_train_cfg(),
        "board_size": 19,  # explicit pin contradicts ckpt's effective 25
    }
    with pytest.raises(ValueError) as excinfo:
        Trainer.load_checkpoint(
            ckpt_path,
            checkpoint_dir=tmp_path,
            fallback_config=cfg,
        )
    msg = str(excinfo.value)
    assert "board_size" in msg
    assert "19" in msg
    assert "25" in msg


def test_config_propagates_resolved_encoding(tmp_path: Path) -> None:
    """After load, trainer.config carries every resolved encoding field."""
    ckpt_path = _make_v6w25_ckpt(tmp_path)
    cfg = {
        **_base_train_cfg(),
        "encoding": {"version": "v6w25"},
    }
    trainer = Trainer.load_checkpoint(
        ckpt_path,
        checkpoint_dir=tmp_path,
        fallback_config=cfg,
    )
    spec = v6w25_spec()
    # All fields the §171 P3 selfplay surfaces depend on must be present.
    assert resolve_from_config(trainer.config).trunk_size == 25  # model trunk dimension
    assert trainer.config["cluster_window_size"] == spec.cluster_window_size
    assert trainer.config["cluster_threshold"] == spec.cluster_threshold
    assert trainer.config["legal_move_radius"] == spec.legal_move_radius
    assert trainer.config["encoding"]["version"] == "v6w25"


# ── §172 A4.3 — metadata-preference path ──────────────────────────────────

def _make_v6_ckpt_with_metadata(tmp_path: Path, encoding_name: str = "v6") -> Path:
    """v6 weights-only ckpt + a `metadata` dict (§172 A5 schema preview).

    Mimics the artifact A5's migration script will stamp onto every
    existing ckpt: bare state_dict + a top-level `metadata` dict whose
    `encoding_name` keys into the registry.
    """
    model = HexTacToeNet(
        board_size=19,
        in_channels=8,
        filters=_FAST_FILTERS,
        res_blocks=_FAST_RES_BLOCKS,
        encoding="v6",
    )
    payload = {
        "model_state": model.state_dict(),
        "metadata": {
            "encoding_name": encoding_name,
            "schema_version": 1,
        },
    }
    path = tmp_path / "checkpoint_with_metadata.pt"
    torch.save(payload, path)
    return path


def test_load_v6_checkpoint_prefers_metadata_when_present(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Metadata-tagged ckpt routes through the registry path (no DeprecationWarning)."""
    import structlog.testing
    ckpt_path = _make_v6_ckpt_with_metadata(tmp_path, encoding_name="v6")
    cfg = {
        **_base_train_cfg(),
        "encoding": {"version": "v6"},
    }
    with structlog.testing.capture_logs() as cap_logs:
        trainer = Trainer.load_checkpoint(
            ckpt_path,
            checkpoint_dir=tmp_path,
            fallback_config=cfg,
        )
    assert resolve_from_config(trainer.config).trunk_size == 19
    assert trainer.config["encoding"]["version"] == "v6"
    # The metadata-found event must have fired and the resolved-encoding
    # event must record source="registry_metadata".
    events = [e.get("event") for e in cap_logs]
    assert "checkpoint_encoding_metadata_found" in events
    resolved_evts = [e for e in cap_logs if e.get("event") == "checkpoint_encoding_resolved"]
    assert resolved_evts, f"no resolved event in {events}"
    assert any(e.get("source") == "registry_metadata" for e in resolved_evts)


def test_load_v6_checkpoint_legacy_fallback_emits_deprecation_warning(
    tmp_path: Path,
) -> None:
    """Legacy ckpt with no metadata → DeprecationWarning + shape-inference path."""
    ckpt_path = _make_v6_ckpt(tmp_path)
    cfg = {
        **_base_train_cfg(),
        "encoding": {"version": "v6"},
    }
    with pytest.warns(DeprecationWarning, match="metadata\\['encoding_name'\\]"):
        trainer = Trainer.load_checkpoint(
            ckpt_path,
            checkpoint_dir=tmp_path,
            fallback_config=cfg,
        )
    assert resolve_from_config(trainer.config).trunk_size == 19
    assert trainer.config["encoding"]["version"] == "v6"


def test_load_v6w25_checkpoint_with_metadata_routes_via_registry(
    tmp_path: Path,
) -> None:
    """v6w25 ckpt with metadata['encoding_name']='v6w25' propagates v6w25 numerics."""
    # Build v6w25-shaped weights + metadata payload directly.
    model = HexTacToeNet(
        board_size=25,
        in_channels=8,
        filters=_FAST_FILTERS,
        res_blocks=_FAST_RES_BLOCKS,
        encoding="v6w25",
    )
    payload = {
        "model_state": model.state_dict(),
        "metadata": {
            "encoding_name": "v6w25",
            "schema_version": 1,
        },
    }
    path = tmp_path / "ckpt_v6w25_with_metadata.pt"
    torch.save(payload, path)
    cfg = {
        **_base_train_cfg(),
        "encoding": {"version": "v6w25"},
    }
    trainer = Trainer.load_checkpoint(
        path,
        checkpoint_dir=tmp_path,
        fallback_config=cfg,
    )
    assert resolve_from_config(trainer.config).trunk_size == 25
    assert trainer.config["cluster_window_size"] == 25
    assert trainer.config["cluster_threshold"] == 8
    assert trainer.config["encoding"]["version"] == "v6w25"


def test_load_checkpoint_metadata_with_non_string_encoding_name_falls_back(
    tmp_path: Path,
) -> None:
    """Bad metadata['encoding_name'] type → DeprecationWarning + shape inference."""
    model = HexTacToeNet(
        board_size=19,
        in_channels=8,
        filters=_FAST_FILTERS,
        res_blocks=_FAST_RES_BLOCKS,
        encoding="v6",
    )
    payload = {
        "model_state": model.state_dict(),
        "metadata": {
            "encoding_name": 6,  # wrong type
            "schema_version": 1,
        },
    }
    path = tmp_path / "ckpt_bad_metadata.pt"
    torch.save(payload, path)
    cfg = {
        **_base_train_cfg(),
        "encoding": {"version": "v6"},
    }
    with pytest.warns(DeprecationWarning, match="expected str"):
        trainer = Trainer.load_checkpoint(
            path,
            checkpoint_dir=tmp_path,
            fallback_config=cfg,
        )
    # Falls back through shape inference; v6 still resolves correctly.
    assert resolve_from_config(trainer.config).trunk_size == 19
    assert trainer.config["encoding"]["version"] == "v6"
