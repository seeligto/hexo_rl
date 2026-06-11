"""§D-WALLCAUSATION — the replay recorder's checkpoint_step is refreshed at the
ONLY self-play weight-sync point: promotion (drain_pending_eval).  Tagging at the
sync point (not per train step) attributes each recorded game to the weights that
actually generated it (self-play runs the inference model, swapped only on promotion).
"""
from __future__ import annotations

from unittest.mock import Mock, patch

from hexo_rl.training import eval_drain


def _dead_thread() -> Mock:
    t = Mock()
    t.is_alive = Mock(return_value=False)
    return t


def test_promotion_refreshes_recorder_checkpoint_step(tmp_path):
    pool = Mock()
    eval_model = Mock()
    eval_model._orig_mod = eval_model
    eval_model.state_dict = Mock(return_value={})
    best_model = Mock()
    eval_result = [{"promoted": True, "step": 12345, "wr_best": 0.6}]

    with patch("hexo_rl.training.eval_drain.save_best_model_atomic"), \
         patch("hexo_rl.training.eval_drain.emit_event"):
        _thread, new_step = eval_drain.drain_pending_eval(
            _dead_thread(), eval_result, eval_model, best_model,
            tmp_path / "best.pt", best_model_step=0, pool=pool, train_step=99999,
        )

    assert new_step == 12345
    pool.sync_inference_weights.assert_called_once()
    # the recorder is re-tagged with the PROMOTED step, not the (later) train step
    pool.update_checkpoint_step.assert_called_once_with(12345)


def test_promotion_stamps_step_and_run_id(tmp_path):
    """§D-LOOPFIX W3 — the promotion save carries the eval step + run_id +
    encoding so the written anchor is log/filename-distinguishable from bootstrap."""
    pool = Mock()
    eval_model = Mock()
    eval_model._orig_mod = eval_model
    eval_model.state_dict = Mock(return_value={})
    best_model = Mock()
    eval_result = [{"promoted": True, "step": 25000, "wr_best": 0.6}]

    with patch("hexo_rl.training.eval_drain.save_best_model_atomic") as mock_save, \
         patch("hexo_rl.training.eval_drain.emit_event"):
        eval_drain.drain_pending_eval(
            _dead_thread(), eval_result, eval_model, best_model,
            tmp_path / "best.pt", best_model_step=0, pool=pool, train_step=26000,
            run_id="e928c854", encoding="v6_live2",
        )

    _args, kwargs = mock_save.call_args
    assert kwargs["step"] == 25000
    assert kwargs["run_id"] == "e928c854"
    assert kwargs["encoding"] == "v6_live2"


def test_no_promotion_leaves_checkpoint_step_untouched(tmp_path):
    pool = Mock()
    eval_result = [{"promoted": False, "step": 12345}]
    with patch("hexo_rl.training.eval_drain.emit_event"):
        eval_drain.drain_pending_eval(
            _dead_thread(), eval_result, None, None,
            tmp_path / "best.pt", best_model_step=7, pool=pool, train_step=99999,
        )
    pool.update_checkpoint_step.assert_not_called()
    pool.sync_inference_weights.assert_not_called()
