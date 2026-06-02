"""O1 forced-win → one-hot POLICY target — Python-side survival + config pins.

Two guarantees the Rust side cannot self-check:

1. ``prune_policy_targets`` (the only post-buffer transform that touches the
   policy target before ``compute_policy_loss``) must NOT strip the O1 one-hot.
   A pure one-hot is the row max → it survives by construction; a near-one-hot
   blend must stay dominant on the winning action. This is the Python end of the
   "the forced-win target is never silently dropped" chain (the Rust end —
   aggregate + rotate survival — is pinned in ``records.rs``/``rotate`` + the
   ``test_one_hot_survives_aggregate_to_local`` unit test).

2. The 3 O1 config knobs are settable attributes on ``SelfPlayRunnerConfig``
   (they ride as ``#[pyo3(get, set)]`` rather than ctor kwargs so the
   38-positional Rust ctor surface — INV19 — is untouched). ``pool.py`` sets
   them from ``configs/selfplay.yaml``.
"""

import pytest
import torch

from hexo_rl.training.trainer import prune_policy_targets


def test_prune_preserves_pure_one_hot():
    """A pure one-hot (O1 weight=1.0) survives prune_policy_targets unchanged."""
    pi = torch.zeros(1, 8)
    pi[0, 3] = 1.0
    pruned = prune_policy_targets(pi.clone(), threshold_frac=0.02)
    assert torch.argmax(pruned[0]).item() == 3
    assert pruned[0, 3].item() == pytest.approx(1.0)
    assert pruned[0].sum().item() == pytest.approx(1.0)
    # Every non-winning slot stays zero — the hard target is intact.
    assert (pruned[0] > 0).sum().item() == 1


def test_prune_keeps_blend_dominant():
    """A near-one-hot blend (O1 weight<1.0) stays dominant on the winning move
    after prune (prune zeroes weak exploration mass + renormalizes — it can only
    sharpen toward the one-hot, never away from it)."""
    # weight=0.5 blend of a uniform-8 improved policy: winning action 0.5625,
    # the other 7 each 0.0625.
    pi = torch.full((1, 8), 0.0625)
    pi[0, 0] = 0.5625
    assert pi[0].sum().item() == pytest.approx(1.0)
    pruned = prune_policy_targets(pi.clone(), threshold_frac=0.02)
    assert torch.argmax(pruned[0]).item() == 0
    assert pruned[0, 0].item() >= pi[0, 0].item(), "prune must not weaken the winning action"
    assert pruned[0].sum().item() == pytest.approx(1.0)


def test_prune_zero_frac_is_identity():
    """threshold_frac<=0 short-circuits — a disabled prune never mutates O1 targets."""
    pi = torch.zeros(1, 8)
    pi[0, 5] = 1.0
    pruned = prune_policy_targets(pi.clone(), threshold_frac=0.0)
    assert torch.equal(pruned, pi)


def test_config_forced_win_attrs_settable():
    """The 3 O1 knobs are get/set attributes on SelfPlayRunnerConfig, default OFF."""
    engine = pytest.importorskip("engine", reason="engine extension not built")
    cfg = engine.SelfPlayRunnerConfig()
    # Defaults (back-compat / single-variable): OFF, depth 2, pure one-hot.
    assert cfg.forced_win_policy_enabled is False
    assert cfg.forced_win_policy_depth == 2
    assert cfg.forced_win_policy_weight == pytest.approx(1.0)
    # Settable (pool.py drives these from YAML).
    cfg.forced_win_policy_enabled = True
    cfg.forced_win_policy_depth = 1
    cfg.forced_win_policy_weight = 0.8
    assert cfg.forced_win_policy_enabled is True
    assert cfg.forced_win_policy_depth == 1
    assert cfg.forced_win_policy_weight == pytest.approx(0.8)
