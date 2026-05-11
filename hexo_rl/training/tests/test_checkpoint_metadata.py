"""§172 A5.1 — checkpoint metadata writer/reader contract.

Round-trip:
  - save_full_checkpoint(encoding_name=...) writes a top-level `metadata`
    dict per design §8.
  - resolve_from_checkpoint reads metadata silently when present.
  - resolve_from_checkpoint falls back to shape inference + DeprecationWarning
    when the metadata block is absent.

Tests use synthetic state-dicts (correct shapes for v6 / v8) — no real
production checkpoints loaded.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict

import pytest
import torch

from hexo_rl.encoding.resolvers import resolve_from_checkpoint
from hexo_rl.training.checkpoints import (
    CHECKPOINT_METADATA_SCHEMA_VERSION,
    build_checkpoint_metadata,
    save_full_checkpoint,
)


# Minimal synthetic NN matching the shape probes in
# hexo_rl/encoding/compat.py: `trunk.0.weight` (in_channels) +
# `policy_fc.weight` (out_features).
class _SyntheticNet(torch.nn.Module):
    def __init__(self, n_planes: int, policy_logit_count: int) -> None:
        super().__init__()
        # trunk.0.weight matches compat._FIRST_CONV_KEYS[0]
        self.trunk = torch.nn.Sequential(
            torch.nn.Conv2d(n_planes, 4, kernel_size=3, padding=1, bias=False),
        )
        # policy_fc.weight matches compat._POLICY_FC_KEYS[0]
        self.policy_fc = torch.nn.Linear(4, policy_logit_count)

    def forward(self, x):  # pragma: no cover — never called in these tests
        return x


def _v6_net() -> _SyntheticNet:
    # v6: n_planes=8, policy_logit_count=362 (19*19 + 1 pass slot)
    return _SyntheticNet(n_planes=8, policy_logit_count=362)


def _v8_net() -> _SyntheticNet:
    # v8: n_planes=11, policy_logit_count=625 (25*25, no pass slot)
    return _SyntheticNet(n_planes=11, policy_logit_count=625)


def _make_optim_scaler(model: torch.nn.Module):
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    return optim, scaler


# ---- save path ---------------------------------------------------------


def test_build_checkpoint_metadata_keys_and_types() -> None:
    """Schema check on build_checkpoint_metadata helper."""
    meta = build_checkpoint_metadata(
        encoding_name="v6",
        train_config_path="configs/training.yaml",
        corpus_sha256="deadbeef" * 8,
        model_architecture="HexTacToeNet",
    )
    expected_keys = {
        "encoding_name", "commit_sha", "training_date",
        "train_config_path", "corpus_sha256", "model_architecture",
        "model_variant", "schema_version",
    }
    assert set(meta) == expected_keys
    assert meta["encoding_name"] == "v6"
    assert isinstance(meta["commit_sha"], str)
    assert meta["training_date"].endswith("Z")
    assert meta["train_config_path"] == "configs/training.yaml"
    assert meta["corpus_sha256"] == "deadbeef" * 8
    assert meta["model_architecture"] == "HexTacToeNet"
    assert meta["schema_version"] == CHECKPOINT_METADATA_SCHEMA_VERSION


def test_build_checkpoint_metadata_rejects_empty_encoding() -> None:
    with pytest.raises(ValueError, match="encoding_name"):
        build_checkpoint_metadata(encoding_name="")


def test_save_checkpoint_writes_metadata(tmp_path: Path) -> None:
    """save_full_checkpoint(encoding_name=...) stamps a metadata dict."""
    model = _v6_net()
    optim, scaler = _make_optim_scaler(model)
    ckpt = tmp_path / "checkpoint_00000000.pt"
    save_full_checkpoint(
        model, optim, scaler, scheduler=None,
        step=42, config={"board_size": 19}, path=ckpt,
        encoding_name="v6",
        train_config_path="configs/variants/foo.yaml",
        corpus_sha256="abcd" * 16,
    )
    assert ckpt.exists()
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert isinstance(raw, dict)
    assert raw["step"] == 42
    assert "model_state" in raw
    assert "metadata" in raw
    meta = raw["metadata"]
    assert meta["encoding_name"] == "v6"
    assert meta["train_config_path"] == "configs/variants/foo.yaml"
    assert meta["corpus_sha256"] == "abcd" * 16
    assert meta["schema_version"] == CHECKPOINT_METADATA_SCHEMA_VERSION
    assert isinstance(meta["commit_sha"], str)


def test_save_checkpoint_legacy_omits_metadata(tmp_path: Path) -> None:
    """No encoding_name kwarg → metadata block omitted (load path infers)."""
    model = _v6_net()
    optim, scaler = _make_optim_scaler(model)
    ckpt = tmp_path / "checkpoint_legacy.pt"
    save_full_checkpoint(
        model, optim, scaler, scheduler=None,
        step=0, config={}, path=ckpt,
    )
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "metadata" not in raw


# ---- load path ---------------------------------------------------------


def test_load_checkpoint_with_metadata_no_warning(tmp_path: Path) -> None:
    """resolve_from_checkpoint silent on stamped ckpt."""
    model = _v6_net()
    optim, scaler = _make_optim_scaler(model)
    ckpt = tmp_path / "checkpoint_stamped.pt"
    save_full_checkpoint(
        model, optim, scaler, scheduler=None,
        step=0, config={}, path=ckpt,
        encoding_name="v6",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = resolve_from_checkpoint(ckpt)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations == [], (
        f"unexpected DeprecationWarning(s) on stamped ckpt: "
        f"{[str(w.message) for w in deprecations]}"
    )
    assert spec.name == "v6"


def test_load_checkpoint_legacy_emits_warning(tmp_path: Path) -> None:
    """No metadata → infer + DeprecationWarning naming the path.

    Use a v6-named filename so the compat filename heuristic resolves
    deterministically (v6 + v7full share state-dict shape; the filename
    hint is what disambiguates them in real artifacts too).
    """
    model = _v6_net()
    ckpt = tmp_path / "legacy_v6_no_metadata.pt"
    torch.save({"model_state": model.state_dict()}, ckpt)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = resolve_from_checkpoint(ckpt)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) >= 1, "expected DeprecationWarning on legacy ckpt"
    msg = str(deprecations[0].message)
    assert str(ckpt) in msg, f"expected {ckpt!s} in warning message, got: {msg}"
    assert spec.name == "v6"


def test_load_checkpoint_legacy_v8_shape_inference(tmp_path: Path) -> None:
    """v8-shaped state-dict with no filename hint resolves via shape probe."""
    model = _v8_net()
    # Filename intentionally lacks 'v8' substring so we exercise shape
    # inference, not the filename heuristic.
    ckpt = tmp_path / "anonymous_legacy_ckpt.pt"
    torch.save({"model_state": model.state_dict()}, ckpt)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = resolve_from_checkpoint(ckpt)
    assert spec.n_planes == 11
    assert spec.policy_logit_count == 625
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) >= 1


def test_build_checkpoint_metadata_model_variant_default_none():
    from hexo_rl.training.checkpoints import build_checkpoint_metadata
    meta = build_checkpoint_metadata(encoding_name="v6w25")
    assert "model_variant" in meta
    assert meta["model_variant"] is None


def test_build_checkpoint_metadata_model_variant_explicit():
    from hexo_rl.training.checkpoints import build_checkpoint_metadata
    meta = build_checkpoint_metadata(
        encoding_name="v8",
        model_variant="B1_128x12_GPool6_10",
    )
    assert meta["model_variant"] == "B1_128x12_GPool6_10"
