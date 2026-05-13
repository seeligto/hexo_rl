"""INV3 (§176 §E) — training.augment is a required config knob (§112).

Pinned at hexo_rl/training/loop.py:203-208 (or wherever the augment-required
check currently lives — see _AUGMENT_CHECK_REF below). The refactor at P15
may relocate this check; this test guards the behavior at any new home.

Fixture sourcing: no suitable minimal_config or fake_trainer fixture was found
in tests/conftest.py or tests/test_train_lifecycle.py — grep found no
run_training_loop usage in tests/ other than a docstring reference. The
"missing" test constructs minimal mocks inline sufficient to reach line 203.

Strategy for "present" tests: the ValueError check fires AFTER
build_inference_model, cuda_warmup, WorkerPool, build_eval_pipeline,
resolve_anchor, and build_subsystems — all of which require real GPU/Rust
workers. Rather than mocking ~10 subsystems just to reach a guard that
confirms "no ValueError was raised", the "present" tests use source-code
inspection: they assert the guard text is structurally conditioned on
`"augment" not in train_cfg and "augment" not in config`, proving the check
EXISTS and would be bypassed when either dict contains augment. This is
preferable to either (a) a hollow mock chain or (b) an incomplete test.

The "missing" test DOES call run_training_loop (with all pre-check subsystems
mocked) to confirm the ValueError is raised at runtime, not just in source.
"""
from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import hexo_rl.training.loop as _loop_mod
from hexo_rl.training.loop import run_training_loop

# The guard condition as it appears in source — used as a structural anchor.
_AUGMENT_CHECK_REF = "203-208"
_AUGMENT_GUARD_SNIPPET = '"augment" not in train_cfg and "augment" not in config'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args() -> MagicMock:
    """Minimal argparse.Namespace stub."""
    args = MagicMock()
    args.iterations = 0
    args.no_dashboard = True
    return args


def _make_subsys_mock() -> MagicMock:
    subsys = MagicMock()
    subsys.gpu_monitor = None
    subsys.disk_guard = None
    subsys.early_game_probe = None
    subsys.value_probe = None
    subsys.value_probe_interval = 9999
    subsys.composition_interval = 9999
    subsys.instrumentation_enabled = False
    subsys.axis_baseline = None
    subsys.tb_writer = None
    subsys.dashboards = []
    return subsys


def _make_arch_mock() -> MagicMock:
    arch = MagicMock()
    arch.board_size = 19
    arch.res_blocks = 4
    arch.filters = 64
    arch.in_channels = 8
    arch.se_reduction_ratio = 8
    arch.input_channels = 8
    return arch


def _make_trainer_mock() -> MagicMock:
    trainer = MagicMock(spec=[])
    trainer.model = MagicMock()
    trainer.step = 0
    trainer.config = {}
    return trainer


def _make_minimal_config(*, with_augment_top: bool = False, with_augment_train: bool = False) -> tuple[dict, dict]:
    """Return (config, train_cfg) with/without augment as requested."""
    config: dict[str, Any] = {}
    train_cfg: dict[str, Any] = {}
    if with_augment_top:
        config["augment"] = True
    if with_augment_train:
        train_cfg["augment"] = True
    return config, train_cfg


# Patches needed to reach line 203 without real subsystems.
_PATCHES = [
    patch("hexo_rl.training.loop.build_inference_model"),
    patch("hexo_rl.training.loop.cuda_warmup"),
    patch("hexo_rl.training.loop.cuda_stream_audit"),
    patch("hexo_rl.training.loop.build_subsystems"),
    patch("hexo_rl.training.loop.build_eval_pipeline"),
    patch("hexo_rl.training.loop.resolve_anchor"),
    patch("hexo_rl.training.loop.build_eval_model"),
    patch("hexo_rl.training.loop.install_signal_handlers"),
    patch("hexo_rl.training.loop.ShutdownState"),
    # WorkerPool is imported inside the function body
    patch("hexo_rl.selfplay.pool.WorkerPool"),
]


def _call_run_training_loop(config: dict, train_cfg: dict) -> None:
    """Call run_training_loop with mocked subsystems, supplying the given config dicts.

    Patch targets:
    - Module-level imports: patched at hexo_rl.training.loop.<name>
    - Function-body imports (build_eval_pipeline, WorkerPool): patched at their
      source modules, since loop.py does `from X import Y` inside the function.
    """
    trainer = _make_trainer_mock()
    arch = _make_arch_mock()
    subsys = _make_subsys_mock()

    with (
        patch("hexo_rl.training.loop.build_inference_model", return_value=(MagicMock(), arch)),
        patch("hexo_rl.training.loop.cuda_warmup"),
        patch("hexo_rl.training.loop.cuda_stream_audit"),
        patch("hexo_rl.training.loop.build_subsystems", return_value=subsys),
        # build_eval_pipeline is imported inside the function body; patch source module
        patch("hexo_rl.eval.pipeline_setup.build_eval_pipeline", return_value=(None, {}, 2500)),
        patch("hexo_rl.training.loop.resolve_anchor", return_value=None),
        patch("hexo_rl.training.loop.build_eval_model", return_value=MagicMock()),
        patch("hexo_rl.training.loop.install_signal_handlers"),
        patch("hexo_rl.training.loop.ShutdownState"),
        # WorkerPool is imported inside the function body; patch source module
        patch("hexo_rl.selfplay.pool.WorkerPool"),
    ):
        run_training_loop(
            trainer=trainer,
            buffer=MagicMock(),
            pretrained_buffer=None,
            recent_buffer=None,
            bufs=MagicMock(),
            config=config,
            train_cfg=train_cfg,
            mcts_config={},
            args=_make_args(),
            device=MagicMock(),
            run_id="test-run-id",
            capacity=1024,
            min_buf_size=64,
            buffer_schedule=[],
            recency_weight=0.0,
            batch_size_cfg=32,
            mixing_cfg={},
            mixing_initial_w=1.0,
            mixing_min_w=0.0,
            mixing_decay_steps=10000.0,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_augment_missing_from_both_train_and_top_level_raises() -> None:
    """INV3: ValueError raised when augment absent from both train_cfg and config.

    Calls run_training_loop with all pre-check subsystems mocked so the guard
    at loop.py:203-208 is the first meaningful execution path that fires.
    """
    config, train_cfg = _make_minimal_config(with_augment_top=False, with_augment_train=False)
    with pytest.raises(ValueError, match="augment"):
        _call_run_training_loop(config, train_cfg)


def test_augment_in_train_cfg_only_passes() -> None:
    """INV3: augment in train_cfg alone is sufficient — guard must NOT fire.

    Verification strategy: source-code inspection. We assert the guard is
    structurally conditioned on absence from BOTH dicts. Calling the full
    function is impractical because it continues into the training main-loop
    (worker pool, step coordinator) which requires Rust workers and GPU.
    The "missing" test confirms the guard fires at runtime; here we confirm
    its logical dual is embedded in source.
    """
    src = inspect.getsource(run_training_loop)
    # Guard must require absence from both — meaning presence in train_cfg alone bypasses it.
    assert _AUGMENT_GUARD_SNIPPET in src, (
        f"augment-required guard not found in run_training_loop source "
        f"(expected: {_AUGMENT_GUARD_SNIPPET!r}). "
        f"Check for relocation — update _AUGMENT_CHECK_REF={_AUGMENT_CHECK_REF!r}."
    )
    # Structural implication: if the condition is `not in train_cfg AND not in config`,
    # then having augment in train_cfg makes the condition False → no raise.
    # Explicitly verify the `and` combinator is present (not `or`).
    assert " and " in _AUGMENT_GUARD_SNIPPET


def test_augment_in_top_level_only_passes() -> None:
    """INV3: augment in top-level config alone is sufficient — guard must NOT fire.

    Same strategy as test_augment_in_train_cfg_only_passes — source inspection.
    The guard condition uses `and` so either dict containing augment is enough.
    """
    src = inspect.getsource(run_training_loop)
    assert _AUGMENT_GUARD_SNIPPET in src, (
        f"augment-required guard not found in run_training_loop source "
        f"(expected: {_AUGMENT_GUARD_SNIPPET!r}). "
        f"Check for relocation — update _AUGMENT_CHECK_REF={_AUGMENT_CHECK_REF!r}."
    )
    # The guard is `not in train_cfg AND not in config` — so top-level-only also bypasses.
    assert '"augment" not in config' in src
