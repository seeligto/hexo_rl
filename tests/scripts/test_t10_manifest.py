"""§173 T10 — model-variant --manifest priority test.

Covers manifest-resolved, filename-fallback, and shape-fallback paths.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

import importlib.util
import sys

_stamp_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "migrations" / "2026_05_09_stamp_artifact_metadata.py"
_spec = importlib.util.spec_from_file_location("stamp_mod", _stamp_path)
assert _spec is not None and _spec.loader is not None
stamp_mod = importlib.util.module_from_spec(_spec)
sys.modules["stamp_mod"] = stamp_mod
_spec.loader.exec_module(stamp_mod)


def _make_raw_ckpt(path: Path, *, n_planes: int, policy_logits: int) -> None:
    """Save a minimal raw state-dict .pt that probes uniquely to a single encoding."""
    sd = {
        "trunk.input_conv.weight": torch.zeros(128, n_planes, 3, 3),
        "policy_fc.weight": torch.zeros(policy_logits, 722),
    }
    torch.save(sd, path)


def _load_meta(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("metadata") if isinstance(ck, dict) else None


def test_manifest_priority() -> None:
    """Manifest entry overrides filename/shape for an ambiguous checkpoint."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ckpt = root / "best_model.pt"  # ambiguous filename + shape (v6 family)
        _make_raw_ckpt(ckpt, n_planes=8, policy_logits=362)

        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({str(ckpt): "v7full"}))

        args = stamp_mod.build_parser().parse_args(
            ["model-variant", "--dir", str(root), "--manifest", str(manifest)]
        )
        rc = stamp_mod._cmd_model_variant(args)
        assert rc == 0

        meta = _load_meta(ckpt)
        assert meta is not None
        assert meta["encoding_name"] == "v7full"
        assert meta["model_variant"] is None


def test_filename_fallback() -> None:
    """Filename glob resolves when no manifest entry exists."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ckpt = root / "bootstrap_model_v6w25.pt"
        # Deliberately wrong shape — filename should still win.
        _make_raw_ckpt(ckpt, n_planes=8, policy_logits=362)

        args = stamp_mod.build_parser().parse_args(
            ["model-variant", "--dir", str(root)]
        )
        # Without --manifest, un-stamped ckpts are skipped.
        rc = stamp_mod._cmd_model_variant(args)
        assert rc == 0
        assert _load_meta(ckpt) is None

        # With an empty manifest, same skip.
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({}))
        args2 = stamp_mod.build_parser().parse_args(
            ["model-variant", "--dir", str(root), "--manifest", str(manifest)]
        )
        rc2 = stamp_mod._cmd_model_variant(args2)
        assert rc2 == 0
        meta = _load_meta(ckpt)
        assert meta is not None
        assert meta["encoding_name"] == "v6w25"


def test_shape_fallback() -> None:
    """Shape probe resolves when no manifest entry and no filename match."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ckpt = root / "unknown.pt"
        # v6w25 has a unique (8, 626) signature among current encodings.
        _make_raw_ckpt(ckpt, n_planes=8, policy_logits=626)

        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({}))

        args = stamp_mod.build_parser().parse_args(
            ["model-variant", "--dir", str(root), "--manifest", str(manifest)]
        )
        rc = stamp_mod._cmd_model_variant(args)
        assert rc == 0

        meta = _load_meta(ckpt)
        assert meta is not None
        assert meta["encoding_name"] == "v6w25"


def test_dead_dir_skip() -> None:
    """Files under dead-dir patterns are skipped, not stamped."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Dead-dir patterns assume a `checkpoints/` prefix (real repo layout).
        dead = root / "checkpoints" / "broken" / "model.pt"
        dead.parent.mkdir(parents=True)
        _make_raw_ckpt(dead, n_planes=8, policy_logits=362)

        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({str(dead): "v7full"}))

        args = stamp_mod.build_parser().parse_args(
            ["model-variant", "--dir", str(root), "--manifest", str(manifest)]
        )
        rc = stamp_mod._cmd_model_variant(args)
        assert rc == 0

        assert _load_meta(dead) is None


def test_manifest_rejects_unknown_encoding() -> None:
    """Manifest with an unregistered encoding name exits with code 2."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"foo.pt": "not_a_real_encoding"}))

        with pytest.raises(SystemExit):
            stamp_mod._load_manifest_json(manifest)
