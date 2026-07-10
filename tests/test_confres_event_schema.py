"""CONFRES S2 — regime-stable event schema for the PUCT-descent-specific cluster stats.

Pre-CONFRES, ``iteration_complete`` DROPPED ``mcts_root_concentration`` + the §107 I2 cluster
trio under Gumbel-root (they are meaningless there), silently changing the emitted schema by
regime — a consumer wired to those keys flatlined on a regime switch. Fix: always key them,
as their value under PUCT and as ``null`` under Gumbel (design §6 S2).
"""
from __future__ import annotations

from types import SimpleNamespace

from hexo_rl.training.events import (
    _REGIME_GATED_CLUSTER_STAT_KEYS,
    regime_gated_cluster_stats,
)


def _fake_rstats():
    return SimpleNamespace(
        mcts_mean_root_concentration=0.42,
        cluster_value_std_mean=0.1,
        cluster_policy_disagreement_mean=0.2,
        cluster_variance_sample_count=5,
    )


def test_puct_regime_emits_values():
    out = regime_gated_cluster_stats(_fake_rstats(), puct_regime=True)
    assert out == {
        "mcts_root_concentration": 0.42,
        "cluster_value_std_mean": 0.1,
        "cluster_policy_disagreement_mean": 0.2,
        "cluster_variance_sample_count": 5,
    }


def test_gumbel_regime_emits_null_not_dropped():
    out = regime_gated_cluster_stats(_fake_rstats(), puct_regime=False)
    assert out == {k: None for k in _REGIME_GATED_CLUSTER_STAT_KEYS}
    # the KEYS must be present (null), never absent — that is the whole S2 fix
    assert set(out) == set(_REGIME_GATED_CLUSTER_STAT_KEYS)


def test_schema_stable_same_keys_both_regimes():
    puct = regime_gated_cluster_stats(_fake_rstats(), puct_regime=True)
    gumbel = regime_gated_cluster_stats(_fake_rstats(), puct_regime=False)
    assert set(puct) == set(gumbel) == set(_REGIME_GATED_CLUSTER_STAT_KEYS)
