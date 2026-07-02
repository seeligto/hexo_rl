"""Tests for scripts/make_ws3v3_warmstart.py.

Uses tiny fake checkpoint dicts (small tensors, no real model) — the strip
logic is key-set surgery, not shape-dependent.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from make_ws3v3_warmstart import (  # noqa: E402
    strip_checkpoint,
    restamp_encoding,
    _wire_signature,
    STRIPPED_KEYS,
    FORBIDDEN_KEYS,
    DEFAULT_ENCODING_NAME,
    main as _main,
)
import hexo_rl.encoding as enc  # noqa: E402


def _fake_full_checkpoint() -> dict:
    return {
        "step": 200000,
        "model_state": {"trunk.input_conv.weight": torch.zeros(2, 2)},
        "optimizer_state": {"state": {}, "param_groups": []},
        "scaler_state": {"scale": 65536.0},
        "scheduler_state": {"last_epoch": 200000},
        "config": {"lr": 2e-3, "eta_min": 5e-4, "total_steps": 1_000_000},
        "metadata": {"encoding_name": "v6_live2", "commit_sha": "deadbeef"},
    }


def test_strip_checkpoint_keeps_exactly_the_contract_keys() -> None:
    src = _fake_full_checkpoint()
    stripped = strip_checkpoint(src)

    assert set(stripped.keys()) == set(STRIPPED_KEYS)
    for k in FORBIDDEN_KEYS:
        assert k not in stripped
    assert "scheduler_state" not in stripped  # not in the contract either

    assert stripped["step"] == 200000
    assert stripped["metadata"] == src["metadata"]  # verbatim
    assert "trunk.input_conv.weight" in stripped["model_state"]


def test_strip_checkpoint_missing_key_raises() -> None:
    src = _fake_full_checkpoint()
    del src["metadata"]
    with pytest.raises(ValueError, match="missing required key"):
        strip_checkpoint(src)


def test_strip_checkpoint_does_not_mutate_source() -> None:
    src = _fake_full_checkpoint()
    stripped = strip_checkpoint(src)
    stripped["step"] = -1
    assert src["step"] == 200000  # shallow copy, new dict — source untouched


def test_cli_end_to_end_strips_saves_and_reloads(tmp_path: Path) -> None:
    """Drives main(): write a fake full checkpoint, run the CLI strip, reload
    the saved file and assert the on-disk payload matches the contract."""
    src_path = tmp_path / "checkpoint_00200000.pt"
    out_path = tmp_path / "ws3v3_warmstart_200k.pt"
    torch.save(_fake_full_checkpoint(), src_path)

    argv = sys.argv
    sys.argv = ["make_ws3v3_warmstart.py", "--in", str(src_path), "--out", str(out_path)]
    try:
        # main() only calls sys.exit() on error paths; the success path falls
        # through to an implicit None return (no SystemExit raised at all).
        _main()
    except SystemExit as exc:
        pytest.fail(f"main() exited unexpectedly with code {exc.code}")
    finally:
        sys.argv = argv

    assert out_path.exists()
    reloaded = torch.load(out_path, map_location="cpu", weights_only=False)
    assert set(reloaded.keys()) == set(STRIPPED_KEYS)
    for k in FORBIDDEN_KEYS:
        assert k not in reloaded
    assert reloaded["step"] == 200000
    # FIX1 — default --encoding-name re-stamps v6_live2 -> v6_live2_ls (the
    # source fixture's stale single-window stamp -> the multi-window
    # legal-set encoding the D-WS3V3 variants declare).
    assert reloaded["metadata"]["encoding_name"] == DEFAULT_ENCODING_NAME == "v6_live2_ls"
    assert reloaded["metadata"]["commit_sha"] == "deadbeef"  # other metadata keys untouched


# ── FIX1(a) — encoding re-stamp + wire-signature validation ─────────────────

def test_restamp_encoding_v6_live2_to_ls_preserves_wire_signature() -> None:
    meta = {"encoding_name": "v6_live2", "commit_sha": "deadbeef"}
    new_meta = restamp_encoding(meta, "v6_live2_ls")
    assert new_meta["encoding_name"] == "v6_live2_ls"
    assert new_meta["commit_sha"] == "deadbeef"
    assert meta["encoding_name"] == "v6_live2"  # source untouched (new dict returned)

    orig_spec = enc.lookup("v6_live2")
    new_spec = enc.lookup("v6_live2_ls")
    assert _wire_signature(orig_spec) == _wire_signature(new_spec)


def test_restamp_encoding_rejects_wire_signature_mismatch() -> None:
    """v6 has n_planes=8 (vs v6_live2's 4) — a genuinely different wire
    signature. The re-stamp must refuse loudly, not silently override."""
    meta = {"encoding_name": "v6_live2", "commit_sha": "deadbeef"}
    orig_sig = _wire_signature(enc.lookup("v6_live2"))
    v6_sig = _wire_signature(enc.lookup("v6"))
    assert orig_sig != v6_sig, "test fixture assumption broken: v6/v6_live2 now share a wire signature"

    with pytest.raises(ValueError, match="wire_signature"):
        restamp_encoding(meta, "v6")


def test_restamp_encoding_missing_original_name_raises() -> None:
    with pytest.raises(ValueError, match="encoding_name"):
        restamp_encoding({"commit_sha": "deadbeef"}, "v6_live2_ls")


def test_restamp_encoding_unresolvable_original_raises() -> None:
    with pytest.raises(ValueError, match="does not resolve"):
        restamp_encoding({"encoding_name": "not_a_real_encoding"}, "v6_live2_ls")


def test_restamp_encoding_unresolvable_override_raises() -> None:
    with pytest.raises(ValueError, match="does not resolve"):
        restamp_encoding({"encoding_name": "v6_live2"}, "not_a_real_encoding")


def test_cli_encoding_name_empty_string_keeps_verbatim(tmp_path: Path) -> None:
    src_path = tmp_path / "checkpoint_00200000.pt"
    out_path = tmp_path / "ws3v3_warmstart_200k.pt"
    torch.save(_fake_full_checkpoint(), src_path)

    argv = sys.argv
    sys.argv = ["make_ws3v3_warmstart.py", "--in", str(src_path), "--out", str(out_path), "--encoding-name", ""]
    try:
        _main()
    except SystemExit as exc:
        pytest.fail(f"main() exited unexpectedly with code {exc.code}")
    finally:
        sys.argv = argv

    reloaded = torch.load(out_path, map_location="cpu", weights_only=False)
    assert reloaded["metadata"]["encoding_name"] == "v6_live2"  # unchanged


def test_cli_encoding_name_mismatch_exits_loudly(tmp_path: Path, capsys) -> None:
    src_path = tmp_path / "checkpoint_00200000.pt"
    out_path = tmp_path / "ws3v3_warmstart_200k.pt"
    torch.save(_fake_full_checkpoint(), src_path)

    argv = sys.argv
    sys.argv = ["make_ws3v3_warmstart.py", "--in", str(src_path), "--out", str(out_path), "--encoding-name", "v6"]
    try:
        with pytest.raises(SystemExit) as exc:
            _main()
        assert exc.value.code == 2
    finally:
        sys.argv = argv

    assert not out_path.exists()  # refused before any write
    err = capsys.readouterr().err
    assert "wire_signature" in err


def test_cli_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    src_path = tmp_path / "checkpoint_00200000.pt"
    out_path = tmp_path / "ws3v3_warmstart_200k.pt"
    torch.save(_fake_full_checkpoint(), src_path)
    out_path.write_bytes(b"pre-existing")

    argv = sys.argv
    sys.argv = ["make_ws3v3_warmstart.py", "--in", str(src_path), "--out", str(out_path)]
    try:
        with pytest.raises(SystemExit) as exc:
            _main()
        assert exc.value.code == 2
    finally:
        sys.argv = argv

    assert out_path.read_bytes() == b"pre-existing"  # untouched


def test_cli_force_overwrites(tmp_path: Path) -> None:
    src_path = tmp_path / "checkpoint_00200000.pt"
    out_path = tmp_path / "ws3v3_warmstart_200k.pt"
    torch.save(_fake_full_checkpoint(), src_path)
    out_path.write_bytes(b"pre-existing")

    argv = sys.argv
    sys.argv = ["make_ws3v3_warmstart.py", "--in", str(src_path), "--out", str(out_path), "--force"]
    try:
        _main()
    except SystemExit as exc:
        pytest.fail(f"main() exited unexpectedly with code {exc.code}")
    finally:
        sys.argv = argv

    reloaded = torch.load(out_path, map_location="cpu", weights_only=False)
    assert set(reloaded.keys()) == set(STRIPPED_KEYS)
