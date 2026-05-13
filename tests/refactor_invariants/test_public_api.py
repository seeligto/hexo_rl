"""INV13 + INV14 (§176 §E) — public API stability pins.

§88 + §159 close-outs locked these contracts:
- scripts.train calls run_training_loop with N positional/keyword args
- 20+ files import Trainer from hexo_rl.training.trainer

Any refactor changing signature requires updating this test (deliberate).
Test failure = silent API break.
"""
import inspect

from hexo_rl.training.loop import run_training_loop


# Captured at HEAD ea881d8. Update only when changing the public API on purpose.
_EXPECTED_PARAMS: tuple[str, ...] = (
    "trainer",
    "buffer",
    "pretrained_buffer",
    "recent_buffer",
    "bufs",
    "config",
    "train_cfg",
    "mcts_config",
    "args",
    "device",
    "run_id",
    "capacity",
    "min_buf_size",
    "buffer_schedule",
    "recency_weight",
    "batch_size_cfg",
    "mixing_cfg",
    "mixing_initial_w",
    "mixing_min_w",
    "mixing_decay_steps",
)


def test_run_training_loop_signature_frozen() -> None:
    sig = inspect.signature(run_training_loop)
    actual = tuple(sig.parameters.keys())
    assert actual == _EXPECTED_PARAMS, (
        "run_training_loop signature changed — was a deliberate API "
        "update? Update _EXPECTED_PARAMS and CLAUDE.md.\n"
        f"  expected: {_EXPECTED_PARAMS}\n"
        f"  actual:   {actual}"
    )


def test_trainer_importable_from_canonical_path() -> None:
    """from hexo_rl.training.trainer import Trainer must work."""
    from hexo_rl.training.trainer import Trainer
    assert Trainer is not None
