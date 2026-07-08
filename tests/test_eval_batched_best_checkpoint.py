"""Eval-cost lever 1 — opt-in batched best_checkpoint round.

``opponents.best_checkpoint.batched`` routes the model-vs-model round through the
cross-game batched path. Default OFF → serial path unchanged. These tests pin the
dispatch + the config-flag wiring without constructing a real net (the heavy leaf calls
are mocked).
"""
from __future__ import annotations

from unittest.mock import Mock, patch

import torch

from hexo_rl.eval.evaluator import Evaluator

_ENC = "v6_live2_ls"


def _evaluator() -> Evaluator:
    return Evaluator(object(), torch.device("cpu"), {"encoding": _ENC, "evaluation": {}})


class TestEvaluateVsModelDispatch:
    # build_model_bot needs a real net; the serial path builds one eagerly. These tests
    # assert dispatch only, so stub it out.
    def test_batched_true_routes_to_evaluate_batched(self) -> None:
        ev = _evaluator()
        ev.evaluate = Mock(name="evaluate")
        ev.evaluate_batched = Mock(name="evaluate_batched", return_value="BATCHED")
        with patch("hexo_rl.eval.evaluator.build_model_bot", Mock()):
            out = ev.evaluate_vs_model(object(), n_games=8, model_sims=16, batched=True)
        assert out == "BATCHED"
        ev.evaluate_batched.assert_called_once()
        ev.evaluate.assert_not_called()
        # An opponent_factory (callable) is passed, not a pre-built bot.
        assert callable(ev.evaluate_batched.call_args.args[0])

    def test_batched_false_routes_to_serial_evaluate(self) -> None:
        ev = _evaluator()
        ev.evaluate = Mock(name="evaluate", return_value="SERIAL")
        ev.evaluate_batched = Mock(name="evaluate_batched")
        with patch("hexo_rl.eval.evaluator.build_model_bot", Mock()):
            out = ev.evaluate_vs_model(object(), n_games=8, model_sims=16, batched=False)
        assert out == "SERIAL"
        ev.evaluate.assert_called_once()
        ev.evaluate_batched.assert_not_called()

    def test_default_is_serial(self) -> None:
        ev = _evaluator()
        ev.evaluate = Mock(return_value="SERIAL")
        ev.evaluate_batched = Mock()
        with patch("hexo_rl.eval.evaluator.build_model_bot", Mock()):
            ev.evaluate_vs_model(object(), n_games=8, model_sims=16)
        ev.evaluate_batched.assert_not_called()


class TestRunBestConfigFlag:
    """_run_best reads opponents.best_checkpoint.batched and forwards it."""

    def _ctx(self, best_cfg):
        from hexo_rl.eval.opponent_runners import _RunnerContext

        pipeline = Mock()
        pipeline.best_cfg = best_cfg
        pipeline.run_id = 1
        pipeline.db = Mock()
        pipeline.db.get_or_create_player = Mock(return_value=7)
        evaluator = Mock()
        evaluator.evaluate_vs_model = Mock(return_value=Mock(
            win_rate=0.6, win_count=6, draw_count=1, colony_wins=0,
        ))
        return _RunnerContext(
            pipeline=pipeline, evaluator=evaluator, train_step=100, ckpt_pid=1,
            ckpt_name="ckpt_100", best_model=object(), best_model_step=50,
            results={"step": 100}, should_run=lambda *_: True, current_radius=None,
        )

    def test_flag_true_forwarded(self) -> None:
        from hexo_rl.eval.opponent_runners import _run_best

        ctx = self._ctx({"enabled": True, "n_games": 10, "batched": True})
        _run_best(ctx)
        assert ctx.evaluator.evaluate_vs_model.call_args.kwargs["batched"] is True

    def test_flag_defaults_false(self) -> None:
        from hexo_rl.eval.opponent_runners import _run_best

        ctx = self._ctx({"enabled": True, "n_games": 10})
        _run_best(ctx)
        assert ctx.evaluator.evaluate_vs_model.call_args.kwargs["batched"] is False
