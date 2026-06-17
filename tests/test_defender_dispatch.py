"""Single-source dispatch guard — legal-set encodings must NOT play through the
single-window ModelPlayer (it drops off-window legal moves, evaluator.py:111-113,
and FALSE-CLEARS / mis-routes the multi-window model). ``defender_kind`` routes
legal-set → KClusterMCTSBot (no-drop), single-window → ModelPlayer, with an
explicit force for the drop-vs-no-drop counterfactual.

Pure-logic: no model / torch construction at call time. This is the consolidated
home of the dispatch ``a71ae28`` introduced inline in ``scripts/exploit_probe.py``;
every eval instrument now routes through it.
"""
from __future__ import annotations

from hexo_rl.eval.defender_dispatch import (
    LEGAL_SET_POLICY_POOL,
    defender_kind,
    needs_no_drop_bot,
    policy_pool_of,
)


def test_auto_routes_legalset_to_kcluster():
    # v6_live2_ls (the multi-window TREATMENT) MUST use the no-drop bot.
    assert defender_kind("legal_set_scatter_max", "auto") == "kcluster"
    assert defender_kind(LEGAL_SET_POLICY_POOL) == "kcluster"  # auto is the default


def test_auto_keeps_singlewindow_on_modelplayer():
    assert defender_kind("none", "auto") == "modelplayer"
    assert defender_kind("", "auto") == "modelplayer"


def test_explicit_force_beats_policy_pool():
    # counterfactual legs: force the drop baseline on legal-set, or no-drop on single-window.
    assert defender_kind("legal_set_scatter_max", "modelplayer") == "modelplayer"
    assert defender_kind("none", "kcluster") == "kcluster"


def test_accepts_spec_object_not_just_string():
    class _Spec:
        policy_pool = "legal_set_scatter_max"

    class _Single:
        policy_pool = "none"

    assert needs_no_drop_bot(_Spec()) is True
    assert needs_no_drop_bot(_Single()) is False
    assert defender_kind(_Spec()) == "kcluster"
    assert defender_kind(_Single()) == "modelplayer"


def test_policy_pool_of_handles_missing_and_none():
    class _NoAttr:
        pass

    assert policy_pool_of(_NoAttr()) == "none"
    assert policy_pool_of(None) == "none"  # bare None spec → treated as single-window
    assert policy_pool_of("") == "none"


def test_registry_v6_live2_ls_is_legal_set_and_v6_live2_is_not():
    # Guards the dispatch against a registry drift that would silently re-route.
    from hexo_rl.encoding import lookup

    assert needs_no_drop_bot(lookup("v6_live2_ls")) is True
    assert needs_no_drop_bot(lookup("v6_live2")) is False


# ── integration: build_model_bot constructs the right concrete bot ──────────────
# Locks the wiring (kept_plane_indices, the model.encoding stamp, the lazy imports),
# not just the string decision above.

def test_build_model_bot_constructs_kcluster_for_legalset():
    import torch

    from hexo_rl.encoding import lookup
    from hexo_rl.eval.defender_dispatch import build_model_bot
    from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot
    from hexo_rl.model.network import HexTacToeNet

    device = torch.device("cpu")
    model = HexTacToeNet(board_size=19, in_channels=4, filters=8, res_blocks=1,
                         encoding="v6_live2")  # detects as the arch family
    bot = build_model_bot(model, lookup("v6_live2_ls"), device, n_sims=2,
                          encoding_label="v6_live2_ls")
    assert isinstance(bot, KClusterMCTSBot)
    # the override label is stamped so the bot's own cluster-geometry lookup is correct
    assert model.encoding == "v6_live2_ls"


def test_build_model_bot_constructs_modelplayer_for_singlewindow():
    import torch

    from hexo_rl.encoding import lookup
    from hexo_rl.eval.defender_dispatch import build_model_bot
    from hexo_rl.eval.evaluator import ModelPlayer
    from hexo_rl.model.network import HexTacToeNet

    device = torch.device("cpu")
    model = HexTacToeNet(board_size=19, in_channels=4, filters=8, res_blocks=1,
                         encoding="v6_live2")
    bot = build_model_bot(model, lookup("v6_live2"), device, n_sims=2,
                          encoding_label="v6_live2")
    assert isinstance(bot, ModelPlayer)
