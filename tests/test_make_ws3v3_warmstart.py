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
    extract_model_state,
    synthesize_payload,
    _wire_signature,
    STRIPPED_KEYS,
    FORBIDDEN_KEYS,
    DEFAULT_ENCODING_NAME,
    DEFAULT_SOURCE_ENCODING,
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


# ── D-RUN2 B1 — --synthesize-metadata (bare/unstamped input) ─────────────────

def _fake_bare_state_dict() -> dict:
    # Mirrors the real bootstrap_model_v6_live2_8300.pt shape: a plain
    # tensor-only mapping, no wrapper keys at all.
    return {
        "trunk.input_conv.weight": torch.zeros(2, 2),
        "value_head.fc.weight": torch.ones(1, 2),
    }


def test_extract_model_state_detects_all_three_layouts() -> None:
    bare = _fake_bare_state_dict()
    ms, layout = extract_model_state(bare)
    assert ms is bare and "bare" in layout

    wrapped_ms = {"model_state": bare}
    ms, layout = extract_model_state(wrapped_ms)
    assert ms is bare and "model_state" in layout

    wrapped_sd = {"state_dict": bare}
    ms, layout = extract_model_state(wrapped_sd)
    assert ms is bare and "state_dict" in layout


def test_extract_model_state_refuses_unknown_schema() -> None:
    # Bookkeeping key present but no model_state/state_dict → refuse, don't guess.
    with pytest.raises(ValueError, match="refusing to guess"):
        extract_model_state({"step": 3, "weights_maybe": _fake_bare_state_dict()})
    # Non-tensor values in a would-be-bare dict → refuse.
    with pytest.raises(ValueError, match="refusing to guess"):
        extract_model_state({"trunk.w": torch.zeros(2), "note": "hi"})
    # Empty dict → refuse.
    with pytest.raises(ValueError, match="refusing to guess"):
        extract_model_state({})


def test_synthesize_payload_bare_input_contract() -> None:
    bare = _fake_bare_state_dict()
    payload = synthesize_payload(bare, DEFAULT_SOURCE_ENCODING, "v6_live2_ls")
    assert set(payload.keys()) == set(STRIPPED_KEYS)
    for k in FORBIDDEN_KEYS:
        assert k not in payload
    assert payload["step"] == 0
    assert payload["metadata"]["encoding_name"] == "v6_live2_ls"
    assert payload["model_state"] is bare
    assert "metadata" not in bare  # source dict untouched


def test_synthesize_payload_refuses_wire_signature_mismatch() -> None:
    # v6 (n_planes=8) does not share v6_live2_ls's wire signature — the
    # synthesis path must go through the same loud gate as the re-stamp path.
    assert _wire_signature(enc.lookup("v6")) != _wire_signature(enc.lookup("v6_live2_ls"))
    with pytest.raises(ValueError, match="wire_signature"):
        synthesize_payload(_fake_bare_state_dict(), "v6", "v6_live2_ls")
    with pytest.raises(ValueError, match="wire_signature"):
        synthesize_payload(_fake_bare_state_dict(), "v6_live2", "v6")


def test_synthesize_payload_refuses_already_stamped_input() -> None:
    with pytest.raises(ValueError, match="already carries"):
        synthesize_payload(_fake_full_checkpoint(), "v6_live2", "v6_live2_ls")


def test_cli_bare_input_without_flag_still_fails_loudly(tmp_path: Path) -> None:
    """B1 crash path, unchanged behavior: a bare state_dict fed to the
    default strip path must fail loudly (missing model_state/metadata/step),
    never silently produce a truncated warm-start."""
    src_path = tmp_path / "bootstrap_bare.pt"
    out_path = tmp_path / "out.pt"
    torch.save(_fake_bare_state_dict(), src_path)

    argv = sys.argv
    sys.argv = ["make_ws3v3_warmstart.py", "--in", str(src_path), "--out", str(out_path)]
    try:
        with pytest.raises(ValueError, match="missing required key"):
            _main()
    finally:
        sys.argv = argv
    assert not out_path.exists()


def test_cli_synthesize_metadata_end_to_end(tmp_path: Path) -> None:
    """Bare input + --synthesize-metadata → on-disk payload is EXACTLY
    {model_state, metadata, step}, encoding_name v6_live2_ls, step 0 —
    and the SHA-pinned source file is byte-unchanged."""
    src_path = tmp_path / "bootstrap_bare.pt"
    out_path = tmp_path / "run2_bootstrap.pt"
    torch.save(_fake_bare_state_dict(), src_path)
    src_bytes_before = src_path.read_bytes()

    argv = sys.argv
    sys.argv = [
        "make_ws3v3_warmstart.py", "--synthesize-metadata",
        "--in", str(src_path), "--out", str(out_path),
    ]
    try:
        _main()
    except SystemExit as exc:
        pytest.fail(f"main() exited unexpectedly with code {exc.code}")
    finally:
        sys.argv = argv

    assert src_path.read_bytes() == src_bytes_before  # original never mutated

    reloaded = torch.load(out_path, map_location="cpu", weights_only=False)
    assert set(reloaded.keys()) == set(STRIPPED_KEYS)
    for k in FORBIDDEN_KEYS:
        assert k not in reloaded
    assert reloaded["step"] == 0
    assert reloaded["metadata"]["encoding_name"] == DEFAULT_ENCODING_NAME == "v6_live2_ls"
    assert reloaded["metadata"]["synthesized_metadata"] is True
    assert reloaded["metadata"]["synthesized_from"] == src_path.name
    assert "trunk.input_conv.weight" in reloaded["model_state"]


def test_cli_synthesize_metadata_wire_mismatch_exits_loudly(tmp_path: Path, capsys) -> None:
    src_path = tmp_path / "bootstrap_bare.pt"
    out_path = tmp_path / "run2_bootstrap.pt"
    torch.save(_fake_bare_state_dict(), src_path)
    src_bytes_before = src_path.read_bytes()

    argv = sys.argv
    sys.argv = [
        "make_ws3v3_warmstart.py", "--synthesize-metadata",
        "--in", str(src_path), "--out", str(out_path),
        "--source-encoding", "v6",  # n_planes=8 — genuinely different wire signature
    ]
    try:
        with pytest.raises(SystemExit) as exc:
            _main()
        assert exc.value.code == 2
    finally:
        sys.argv = argv

    assert not out_path.exists()  # refused before any write
    assert src_path.read_bytes() == src_bytes_before
    assert "wire_signature" in capsys.readouterr().err


def test_cli_synthesize_metadata_requires_encoding_name(tmp_path: Path, capsys) -> None:
    src_path = tmp_path / "bootstrap_bare.pt"
    out_path = tmp_path / "run2_bootstrap.pt"
    torch.save(_fake_bare_state_dict(), src_path)

    argv = sys.argv
    sys.argv = [
        "make_ws3v3_warmstart.py", "--synthesize-metadata",
        "--in", str(src_path), "--out", str(out_path), "--encoding-name", "",
    ]
    try:
        with pytest.raises(SystemExit) as exc:
            _main()
        assert exc.value.code == 2
    finally:
        sys.argv = argv
    assert not out_path.exists()
    assert "non-empty --encoding-name" in capsys.readouterr().err


def test_cli_refuses_out_equal_in(tmp_path: Path, capsys) -> None:
    """Requirement 4: the SHA-pinned source is NEVER mutated — --out == --in
    is refused outright, even with --force."""
    src_path = tmp_path / "bootstrap_bare.pt"
    torch.save(_fake_bare_state_dict(), src_path)
    src_bytes_before = src_path.read_bytes()

    argv = sys.argv
    sys.argv = [
        "make_ws3v3_warmstart.py", "--synthesize-metadata",
        "--in", str(src_path), "--out", str(src_path), "--force",
    ]
    try:
        with pytest.raises(SystemExit) as exc:
            _main()
        assert exc.value.code == 2
    finally:
        sys.argv = argv
    assert src_path.read_bytes() == src_bytes_before
    assert "never mutated" in capsys.readouterr().err
