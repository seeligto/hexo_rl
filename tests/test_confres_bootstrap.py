"""CONFRES P3 — single validated resolver for the training bootstrap/resume checkpoint.

The bug this closes: `make train` with the (stale) Makefile `BOOTSTRAP` default passes a
non-existent path to `--checkpoint`; today it fails LATE inside `torch.load` with an
uninformative `FileNotFoundError` and no early guard. `resolve_bootstrap` validates the
resolved path at LAUNCH, naming the path + the override knob.
"""
from __future__ import annotations

import pytest

from hexo_rl.config.resolve.bootstrap import (
    BootstrapNotFoundError,
    ResolvedBootstrap,
    resolve_bootstrap,
)


def test_none_checkpoint_is_a_fresh_run():
    """No --checkpoint → a fresh run (no bootstrap); must NOT raise."""
    resolved = resolve_bootstrap(None)
    assert resolved == ResolvedBootstrap(path=None, source="none")


def test_existing_path_resolves_with_cli_source():
    resolved = resolve_bootstrap("checkpoints/exists.pt", exists=lambda p: True)
    assert resolved == ResolvedBootstrap(path="checkpoints/exists.pt", source="cli")


def test_missing_path_raises_at_launch_naming_the_path():
    with pytest.raises(BootstrapNotFoundError) as exc:
        resolve_bootstrap("checkpoints/bootstrap_model_v6.pt", exists=lambda p: False)
    msg = str(exc.value)
    assert "checkpoints/bootstrap_model_v6.pt" in msg
    assert "BOOTSTRAP" in msg  # names the override knob so the operator can fix it


def test_error_is_a_filenotfounderror_subclass():
    """Existing `except FileNotFoundError` handlers still catch it — just earlier + clearer."""
    assert issubclass(BootstrapNotFoundError, FileNotFoundError)


def test_missing_path_error_fires_before_any_load():
    """The guard is the resolver itself — no torch import / no load needed to raise."""
    calls = {"n": 0}

    def _exists(_p):
        calls["n"] += 1
        return False

    with pytest.raises(BootstrapNotFoundError):
        resolve_bootstrap("nope.pt", exists=_exists)
    assert calls["n"] == 1  # existence checked exactly once, then raised
