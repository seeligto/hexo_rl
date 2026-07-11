"""CONFRES 6c grep-gate + migration pins (design §8).

Two guarantees:

1. The three buffer-sizing seams (ReplayBuffer / RecentBuffer / batch buffers) route through the
   ONE encoding→spec authority ``resolve.encoding.window_set``, NOT a raw pre-checkpoint spec or a
   v6-shaped literal fallback.
2. The ``allocate_batch_buffers_for_config`` v6-literal warn-only fallback is GONE — an
   unresolvable encoding is now a HARD-ERROR (a v6-geometry buffer against a non-v6 net was a
   silent-corruption class).

The grep is scoped to the migrated seams (not every ``resolve_from_config`` site — those are the
SINGLE-RESOLVER registry-canonical resolver, adopted-as-shared-rule, out of the 6c migration set).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1] / "hexo_rl" / "training" / "orchestrator.py"


def _src() -> str:
    return _ORCH.read_text()


def test_replay_buffer_sizes_from_window_set():
    src = _src()
    # The ReplayBuffer line must build its encoding via window_set(...).name, not a raw
    # normalize_encoding_name(config.get("encoding")).
    m = re.search(r"buffer = ReplayBuffer\(capacity=capacity, encoding=([^)]+)\)", src)
    assert m, "ReplayBuffer construction line not found"
    assert "window_set" in m.group(1), f"ReplayBuffer not sized from window_set: {m.group(1)!r}"


def test_recent_buffer_reresolves_from_combined_config():
    src = _src()
    # init_recent_buffer must re-resolve the spec from combined_config via window_set.
    body = src[src.index("def init_recent_buffer"):src.index("def allocate_batch_buffers_for_config")]
    assert "window_set" in body, "init_recent_buffer does not re-resolve via window_set"
    assert "combined_config" in body, "init_recent_buffer does not consult combined_config"


def test_batch_buffer_alloc_hard_errors_no_v6_literal_fallback():
    src = _src()
    body = src[src.index("def allocate_batch_buffers_for_config"):src.index("def read_mixing_params")]
    assert "window_set" in body, "batch buffer alloc does not use window_set"
    # The v6-literal warn-only fallback must be gone.
    assert "buffer_alloc_registry_resolve_failed" not in body, "v6-literal warn fallback still present"
    assert "_n_planes_spec = 8" not in body, "v6-literal n_planes=8 fallback still present"


def test_batch_buffer_alloc_raises_on_unknown_encoding():
    """An unresolvable encoding hard-errors (not a silent v6 fallback)."""
    import logging

    from hexo_rl.encoding.registry import EncodingRegistryError
    from hexo_rl.training.orchestrator import allocate_batch_buffers_for_config

    log = logging.getLogger("test")
    with pytest.raises(EncodingRegistryError):
        allocate_batch_buffers_for_config(
            {"batch_size": 8}, {}, {"encoding": "this_encoding_does_not_exist"}, log,
        )
