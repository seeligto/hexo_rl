"""CONFRES P5 — one default authority for eval opponent n_sims (model_sims).

Pre-CONFRES: the eval pipeline injected 96/128 while a directly-constructed ``Evaluator`` fell
back to a DIVERGENT 100/200 — dead defaults no production path or test actually depended on (the
'intentional split' in defaults.py was drift). Collapsed to one authority in resolve/nsims.py.
"""
from __future__ import annotations

import pytest

from hexo_rl.config.resolve.nsims import EVAL_MODEL_SIMS_DEFAULT, resolve_eval_model_sims


def test_random_default_is_96():
    assert resolve_eval_model_sims("random", None) == 96


def test_sealbot_default_is_128():
    assert resolve_eval_model_sims("sealbot", None) == 128


def test_config_value_wins_over_default():
    assert resolve_eval_model_sims("random", 64) == 64
    assert resolve_eval_model_sims("sealbot", 32) == 32


def test_unknown_opponent_raises():
    with pytest.raises(ValueError):
        resolve_eval_model_sims("mystery", None)


def test_default_map_is_the_single_authority():
    assert EVAL_MODEL_SIMS_DEFAULT == {"random": 96, "sealbot": 128}


def test_direct_evaluator_uses_pipeline_default_not_legacy_100_200():
    """The P5 collapse: a directly-constructed Evaluator (no pipeline merge, empty eval config)
    sees the SAME model_sims the pipeline injects (96/128), NOT the old divergent 100/200."""
    from hexo_rl.eval.evaluator import Evaluator

    class _DummyModel:  # Evaluator.__init__ only reads config + getattr(model, "_orig_mod", model)
        pass

    ev = Evaluator(_DummyModel(), "cpu", {})
    assert ev.random_model_sims == 96
    assert ev.sealbot_model_sims == 128
