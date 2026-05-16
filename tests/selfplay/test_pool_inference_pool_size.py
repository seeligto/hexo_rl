"""§P55 / Wave 5a operator follow-up — `inference_pool_size` kwarg wiring.

The Rust `SelfPlayRunner.new` PyO3 signature accepts an `inference_pool_size`
kwarg (cycle 2 Wave 4 Batch C, commit 408a5c5). Cycle 1 hard-coded the
InferenceBatcher feature-buffer pool at 512 prefill / 1024 channel capacity;
high-K v6w25 16-worker runs need a larger pool (per the §P55 commit body,
recommended size is `n_workers * leaf_batch_size * K_max * 2`).

Pre-Wave-5a the Python `WorkerPool.__init__` did NOT forward this kwarg, so
operator opt-in was impossible from YAML. Wave 5a Batch B closes the gap:
`WorkerPool.__init__` reads `selfplay.inference_pool_size` from config and
threads it into the `SelfPlayRunner(...)` ctor.

Tests below assert:
  1. Default behavior (no `inference_pool_size` key in config) → kwarg passed
     as `None` (cycle-1-equivalent).
  2. Explicit `inference_pool_size = N` in config → kwarg passed as `int(N)`.
  3. Non-int values get coerced to int.

Strategy: monkey-patch `hexo_rl.selfplay.pool.SelfPlayRunnerConfig` to a
recorder that captures the kwargs of its call (cycle 3 Wave 7 Batch A — kwargs
now ride on `SelfPlayRunnerConfig` rather than `SelfPlayRunner`) and
`hexo_rl.selfplay.pool.SelfPlayRunner` to a passthrough mock that returns a
runner-like instance. We don't need the runner to actually start — we only
assert the kwarg propagation.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest


def _make_recorder() -> Any:
    """Build a recorder pair (config_recorder, runner_mock) that mimics the
    SelfPlayRunner construction path enough for WorkerPool.__init__ to finish
    (it accesses `.batcher` after construct).

    cycle 3 Wave 7 Batch A: kwargs land on `SelfPlayRunnerConfig` first; the
    recorder is set as `pool_mod.SelfPlayRunnerConfig`. `pool_mod.SelfPlayRunner`
    receives the config instance positionally and returns a `.batcher`-enabled
    MagicMock so InferenceServer ctor doesn't blow up on attribute access.

    The recorder stashes the kwargs on `recorder.last_kwargs`."""
    recorder = MagicMock()

    def _config_ctor(*args, **kwargs):
        recorder.last_kwargs = kwargs
        # Return a sentinel config object — passed positionally to SelfPlayRunner
        # mock; tests don't introspect it (kwargs were already captured above).
        return MagicMock(name="SelfPlayRunnerConfig_instance", _captured=kwargs)

    recorder.side_effect = _config_ctor

    def _runner_ctor(config, *args, **kwargs):
        # `config` is the sentinel from _config_ctor; pull captured kwargs
        # off it so the `.batcher.feature_len` / `.policy_len_py` accessors
        # below work on the runner-like instance.
        captured = getattr(config, "_captured", {})
        instance = MagicMock()
        instance.batcher = MagicMock()
        instance.batcher.policy_len_py = captured.get("policy_len", 362)
        instance.batcher.feature_len = MagicMock(return_value=captured.get("feature_len", 2888))
        return instance

    runner_mock = MagicMock()
    runner_mock.side_effect = _runner_ctor
    recorder.runner_mock = runner_mock
    return recorder


def _minimal_cfg(extra_sp: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Minimal config dict that satisfies the WorkerPool construct path up
    to (and including) the SelfPlayRunner(...) call. `playout_cap.fast_sims`
    is required (loud-fail otherwise per §100)."""
    sp: Dict[str, Any] = {
        "n_workers": 1,
        "playout_cap": {"fast_sims": 1, "fast_prob": 0.0, "standard_sims": 1},
    }
    if extra_sp:
        sp.update(extra_sp)
    return {
        "encoding": "v6",
        "mcts": {"n_simulations": 1, "c_puct": 1.0, "fpu_reduction": 0.25},
        "selfplay": sp,
        "training": {"draw_value": -0.1},
    }


def _make_fake_model() -> Any:
    """Minimal stub matching the surface WorkerPool reads off `model`."""
    m = SimpleNamespace()
    m.board_size = 19
    return m


# --------------------------------------------------------------------------- #
# Default = None (cycle-1 behavior preserved)
# --------------------------------------------------------------------------- #
def test_inference_pool_size_default_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No `selfplay.inference_pool_size` key → SelfPlayRunner receives
    `inference_pool_size=None`. Preserves cycle-1 behavior (fixed 512 prefill
    / 1024 channel capacity inside InferenceBatcher)."""
    from hexo_rl.selfplay import pool as pool_mod

    recorder = _make_recorder()
    monkeypatch.setattr(pool_mod, "SelfPlayRunnerConfig", recorder)
    monkeypatch.setattr(pool_mod, "SelfPlayRunner", recorder.runner_mock)
    # InferenceServer ctor reads model attrs; stub it too so it doesn't fail
    # on the MagicMock model.
    monkeypatch.setattr(pool_mod, "InferenceServer", MagicMock())

    cfg = _minimal_cfg()
    model = _make_fake_model()
    # ReplayBuffer arg satisfied by MagicMock; the constructor only stores it.
    pool_mod.WorkerPool(model, cfg, "cpu", MagicMock(), n_workers=1)

    assert "inference_pool_size" in recorder.last_kwargs
    assert recorder.last_kwargs["inference_pool_size"] is None


# --------------------------------------------------------------------------- #
# Explicit opt-in → int forwarded
# --------------------------------------------------------------------------- #
def test_inference_pool_size_explicit_int_threads_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """`selfplay.inference_pool_size = 8192` in config → SelfPlayRunner
    receives `inference_pool_size=8192` (the int form)."""
    from hexo_rl.selfplay import pool as pool_mod

    recorder = _make_recorder()
    monkeypatch.setattr(pool_mod, "SelfPlayRunnerConfig", recorder)
    monkeypatch.setattr(pool_mod, "SelfPlayRunner", recorder.runner_mock)
    monkeypatch.setattr(pool_mod, "InferenceServer", MagicMock())

    cfg = _minimal_cfg({"inference_pool_size": 8192})
    model = _make_fake_model()
    pool_mod.WorkerPool(model, cfg, "cpu", MagicMock(), n_workers=1)

    assert recorder.last_kwargs["inference_pool_size"] == 8192
    assert isinstance(recorder.last_kwargs["inference_pool_size"], int)


# --------------------------------------------------------------------------- #
# Numeric string coerced to int
# --------------------------------------------------------------------------- #
def test_inference_pool_size_string_coerced_to_int(monkeypatch: pytest.MonkeyPatch) -> None:
    """YAML loaders sometimes leave numerics as strings — operator-passed
    `inference_pool_size = "4096"` must coerce to int(4096), not pass a
    raw string that PyO3 would reject."""
    from hexo_rl.selfplay import pool as pool_mod

    recorder = _make_recorder()
    monkeypatch.setattr(pool_mod, "SelfPlayRunnerConfig", recorder)
    monkeypatch.setattr(pool_mod, "SelfPlayRunner", recorder.runner_mock)
    monkeypatch.setattr(pool_mod, "InferenceServer", MagicMock())

    cfg = _minimal_cfg({"inference_pool_size": "4096"})
    model = _make_fake_model()
    pool_mod.WorkerPool(model, cfg, "cpu", MagicMock(), n_workers=1)

    assert recorder.last_kwargs["inference_pool_size"] == 4096
    assert isinstance(recorder.last_kwargs["inference_pool_size"], int)
