"""Tests for hexo_rl.encoding.audit — §172 A5 audit CLI.

Builds synthetic checkpoint + corpus + variant + source-tree fixtures
under tmp_path, runs `audit.main(...)`, asserts exit code + section
contents. Tests must NOT touch real `checkpoints/` or `data/` —
isolation via tmp_path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from hexo_rl.encoding import lookup
from hexo_rl.encoding.audit import audit, main


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _v6_state_dict() -> dict[str, torch.Tensor]:
    """Synthetic state dict matching v6 (n_planes=8, policy_logits=362)."""
    return {
        "trunk.0.weight": torch.zeros(64, 8, 3, 3),
        "policy_fc.weight": torch.zeros(362, 64),
    }


def _make_legacy_ckpt(path: Path) -> None:
    """Write a state-dict-only checkpoint (no metadata key)."""
    torch.save(_v6_state_dict(), path)


def _make_stamped_ckpt(path: Path, encoding_name: str = "v6") -> None:
    """Write a checkpoint with metadata + model_state."""
    ckpt = {
        "model_state": _v6_state_dict(),
        "metadata": {
            "encoding_name": encoding_name,
            "schema_version": 1,
        },
    }
    torch.save(ckpt, path)


def _make_corpus_with_sidecar(
    npz_path: Path, encoding_name: str = "v6"
) -> None:
    """Write a tiny .npz + matching sidecar with correct sha256."""
    import numpy as np

    np.savez(npz_path, x=np.zeros((2, 3), dtype=np.float32))
    sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    sidecar = npz_path.with_suffix(npz_path.suffix + ".metadata.json")
    sidecar.write_text(
        json.dumps(
            {
                "encoding_name": encoding_name,
                "sha256": sha,
                "n_positions": 2,
                "schema_version": 1,
            }
        )
    )


def _make_variant(yaml_path: Path, encoding_name: str = "v6") -> None:
    """Write a clean variant yaml that resolves cleanly."""
    spec = lookup(encoding_name)
    cfg = {
        "encoding": encoding_name,
        "board_size": spec.board_size,
        "n_planes": spec.n_planes,
    }
    yaml_path.write_text(yaml.safe_dump(cfg))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_audit_clean_returns_0(tmp_path: Path):
    """Single stamped ckpt + sidecar-stamped corpus + clean variant ⇒ exit 0
    (or 1 if hardcode-literal grep finds hits in the real source tree).

    Test isolates source tree by pointing repo_root at a clean tmp dir."""
    ckpts = tmp_path / "checkpoints"
    corpora = tmp_path / "data"
    variants = tmp_path / "variants"
    src_root = tmp_path / "src_root"  # empty source tree → no literal hits
    (src_root / "engine" / "src").mkdir(parents=True)
    (src_root / "hexo_rl").mkdir(parents=True)
    ckpts.mkdir()
    corpora.mkdir()
    variants.mkdir()

    _make_stamped_ckpt(ckpts / "bootstrap_v6.pt", encoding_name="v6")
    _make_corpus_with_sidecar(corpora / "bootstrap_corpus_v6.npz", encoding_name="v6")
    _make_variant(variants / "clean_v6.yaml", encoding_name="v6")

    rep = audit(ckpts, corpora, variants, repo_root=src_root)
    assert rep.exit_code() == 0, f"expected clean audit; report:\n{rep}"
    assert any(
        f.section == "§2" and "v6" in f.message for f in rep.findings
    )


def test_audit_legacy_ckpt_warns(tmp_path: Path):
    """State-dict-only ckpt (no metadata) → LEGACY warn → exit 1."""
    ckpts = tmp_path / "checkpoints"
    corpora = tmp_path / "data"
    variants = tmp_path / "variants"
    src_root = tmp_path / "src_root"
    (src_root / "engine" / "src").mkdir(parents=True)
    (src_root / "hexo_rl").mkdir(parents=True)
    ckpts.mkdir()
    corpora.mkdir()
    variants.mkdir()

    # Filename contains "v6" so inference works → status=LEGACY (not UNKNOWN).
    _make_legacy_ckpt(ckpts / "old_v6_ckpt.pt")
    _make_corpus_with_sidecar(corpora / "bootstrap_corpus_v6.npz", encoding_name="v6")
    _make_variant(variants / "clean_v6.yaml", encoding_name="v6")

    rep = audit(ckpts, corpora, variants, repo_root=src_root)
    assert rep.exit_code() == 1, f"expected exit 1 (warn); got {rep.exit_code()}\n{rep}"
    assert any(
        f.section == "§2" and "old_v6_ckpt" in f.message and "no metadata" in f.message
        for f in rep.findings
    )
    # Section row must say LEGACY.
    sect = rep.sections["§2"]
    assert any("LEGACY" in row for row in sect.rows[0])


def test_audit_hardcode_hit_errors_under_strict(tmp_path: Path):
    """Synthetic source file with `BOARD_SIZE = 19` + no allowlist marker:
    - default → exit 1 (warn)
    - --strict → exit 2 (error)"""
    ckpts = tmp_path / "checkpoints"
    corpora = tmp_path / "data"
    variants = tmp_path / "variants"
    src_root = tmp_path / "src_root"
    (src_root / "engine" / "src").mkdir(parents=True)
    hxrl_root = src_root / "hexo_rl"
    hxrl_root.mkdir(parents=True)
    ckpts.mkdir()
    corpora.mkdir()
    variants.mkdir()

    # Stamp clean ckpt + corpus so §2/§3 don't add stray warnings.
    _make_stamped_ckpt(ckpts / "bootstrap_v6.pt", encoding_name="v6")
    _make_corpus_with_sidecar(corpora / "bootstrap_corpus_v6.npz", encoding_name="v6")
    _make_variant(variants / "clean_v6.yaml", encoding_name="v6")

    # The bare literal — not allowlisted.
    (hxrl_root / "foo.py").write_text("BOARD_SIZE = 19\n")

    # Default → warn → exit 1.
    rep = audit(ckpts, corpora, variants, strict=False, repo_root=src_root)
    assert rep.exit_code() == 1, f"want exit 1 default; got {rep.exit_code()}\n{rep}"
    assert any(f.section == "§5" and f.severity == "warn" for f in rep.findings)

    # Strict → error → exit 2.
    rep_strict = audit(ckpts, corpora, variants, strict=True, repo_root=src_root)
    assert rep_strict.exit_code() == 2, (
        f"want exit 2 strict; got {rep_strict.exit_code()}\n{rep_strict}"
    )
    assert any(
        f.section == "§5" and f.severity == "error" for f in rep_strict.findings
    )


def test_audit_main_cli_exit_code(tmp_path: Path, capsys):
    """`main(argv)` runs end-to-end and returns proper exit code; prints report."""
    ckpts = tmp_path / "checkpoints"
    corpora = tmp_path / "data"
    variants = tmp_path / "variants"
    src_root = tmp_path / "src_root"
    (src_root / "engine" / "src").mkdir(parents=True)
    (src_root / "hexo_rl").mkdir(parents=True)
    ckpts.mkdir()
    corpora.mkdir()
    variants.mkdir()

    _make_stamped_ckpt(ckpts / "bootstrap_v6.pt", encoding_name="v6")
    _make_corpus_with_sidecar(corpora / "bootstrap_corpus_v6.npz", encoding_name="v6")
    _make_variant(variants / "clean_v6.yaml", encoding_name="v6")

    code = main(
        [
            "audit",
            "--checkpoints-dir",
            str(ckpts),
            "--corpora-dir",
            str(corpora),
            "--variants-dir",
            str(variants),
            "--repo-root",
            str(src_root),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "encoding audit" in out
    assert "§1 Registered encodings" in out
    assert "§2 Checkpoints" in out
    assert "§3 Corpora" in out
    assert "§4 Variants" in out
    assert "§5 Hardcoded literals" in out
