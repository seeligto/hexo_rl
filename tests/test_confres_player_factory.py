"""CONFRES batch 7 — build_player factory + planner dispatch pin (design §4, B1, B1a, N5).

Pins the RESOLVED DISPATCH RESULT per (context, subcontext): which bot_impl each play seam runs.
A ``policy_pool`` edit or a ``mode=`` force FIRES these. Also pins the N5 explicit-label channel
(KClusterMCTSBot no longer depends on ``model.encoding``) and the B1a batched carve-out.
"""
from __future__ import annotations

import pytest

from hexo_rl.config.resolve.planner import (
    BOT_DEPLOY_GUMBEL,
    BOT_KCLUSTER,
    BOT_MODEL_PLAYER,
    resolve_deploy_planner,
    resolve_eval_planner,
)
from hexo_rl.eval.player_factory import (
    BatchedLegalSetRefusedError,
    assert_batched_dispatch_ok,
    build_player,
)


# ── planner dispatch pin (the resolved bot_impl per context/subcontext) ───────
@pytest.mark.parametrize(
    "encoding,expected",
    [
        ("v6", BOT_MODEL_PLAYER),
        ("v6_live2", BOT_MODEL_PLAYER),
        ("v6w25", BOT_MODEL_PLAYER),
        ("v8", BOT_MODEL_PLAYER),
        ("v6_live2_ls", BOT_KCLUSTER),  # legal-set → no-drop KCluster (the load-bearing case)
    ],
)
def test_eval_dispatch_pinned_per_encoding(encoding, expected):
    spec = resolve_eval_planner(encoding, "best", n_sims=64)
    assert spec.bot_impl == expected
    assert spec.legal_set == (expected == BOT_KCLUSTER)
    assert spec.root == "puct" and spec.fpu == 0.0 and spec.virtual_loss is False


def test_eval_dispatch_mode_force_fires():
    # Forcing modelplayer on a legal-set encoding (the drop-baseline counterfactual) MUST change
    # the resolved dispatch — a regression that silently ignored mode would fail here.
    forced = resolve_eval_planner("v6_live2_ls", "best", n_sims=64, mode="modelplayer")
    assert forced.bot_impl == BOT_MODEL_PLAYER
    forced_k = resolve_eval_planner("v6", "best", n_sims=64, mode="kcluster")
    assert forced_k.bot_impl == BOT_KCLUSTER


def test_deploy_dispatch_pinned():
    spec = resolve_deploy_planner("v6_live2_ls", n_sims=150)
    assert spec.bot_impl == BOT_DEPLOY_GUMBEL
    assert spec.root == "gumbel" and spec.gumbel_scale == 0.0
    assert spec.legal_set is True  # legal-set encoding → multi-window deploy decode
    assert resolve_deploy_planner("v6", n_sims=150).legal_set is False


# ── build_player construction (distinct algorithms, one entry) ────────────────
def test_build_player_requires_model_for_eval():
    spec = resolve_eval_planner("v6", "best", n_sims=8)
    with pytest.raises(ValueError, match="requires model"):
        build_player(spec, encoding_label="v6")


def test_build_player_requires_engine_for_deploy():
    spec = resolve_deploy_planner("v6", n_sims=8)
    with pytest.raises(ValueError, match="requires engine"):
        build_player(spec, encoding_label="v6")


def test_build_player_dispatch_mismatch_guard():
    # A PlannerSpec whose bot_impl contradicts the encoding's policy_pool dispatch → hard-error
    # (the resolver and defender_dispatch drifted). Fabricate a mismatch.
    import dataclasses

    spec = resolve_eval_planner("v6_live2_ls", "best", n_sims=8)  # bot_impl=KCluster
    bad = dataclasses.replace(spec, bot_impl=BOT_MODEL_PLAYER)  # lie
    with pytest.raises(ValueError, match="dispatch mismatch"):
        build_player(bad, encoding_label="v6_live2_ls", model=object(), device="cpu")


# ── B1a batched carve-out ─────────────────────────────────────────────────────
def test_batched_refused_on_legal_set():
    with pytest.raises(BatchedLegalSetRefusedError):
        assert_batched_dispatch_ok("v6_live2_ls")


def test_batched_allowed_on_flat_encoding():
    for enc in ("v6", "v6_live2", "v6w25", "v8"):
        assert_batched_dispatch_ok(enc)  # no raise


# ── N5 explicit-label channel ─────────────────────────────────────────────────
def test_kcluster_reads_explicit_label_not_model_attr():
    """KClusterMCTSBot resolves its geometry from the explicit ``encoding=`` param even when
    ``model.encoding`` says something else (the last-writer-wins bug the N5 fix closes)."""
    import torch

    from hexo_rl.encoding import lookup
    from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot
    from hexo_rl.model.network import HexTacToeNet

    spec = lookup("v6_live2_ls")
    model = HexTacToeNet(
        board_size=spec.trunk_size, in_channels=spec.n_planes, filters=8, res_blocks=1,
    )
    # Poison the attribute with a WRONG label; explicit encoding must win.
    model.encoding = "v8"  # v8 has no pass slot → would raise if the attr were used
    bot = KClusterMCTSBot(
        model, torch.device("cpu"), n_sims=1,
        kept_plane_indices=list(spec.kept_plane_indices),
        encoding="v6_live2_ls",
    )
    assert bot is not None  # constructed against the explicit legal-set label, not model.encoding


# ── offline-seam routing pins (CONFRES batch 7) ───────────────────────────────
def _small_model(encoding: str):
    from hexo_rl.encoding import lookup
    from hexo_rl.model.network import HexTacToeNet

    spec = lookup(encoding)
    return HexTacToeNet(
        board_size=spec.trunk_size, in_channels=spec.n_planes, filters=8, res_blocks=1,
    ), spec


def test_make_head_bot_routes_through_factory():
    """scripts/evalfair/core.make_head_bot builds its deploy head via build_player — the
    resulting DeployHeadBot dispatches legal_set from the encoding's policy_pool."""
    import torch

    from hexo_rl.eval.defender_dispatch import needs_no_drop_bot
    from hexo_rl.eval.deploy_strength_eval import DeployHeadBot, _build_engine_for_model
    from scripts.evalfair.core import ArmSpec, make_head_bot

    model, spec = _small_model("v6_live2_ls")
    eng = _build_engine_for_model(model, "v6_live2_ls", torch.device("cpu"))
    knobs = {"gumbel_m": 16, "c_visit": 50, "c_scale": 1.0, "n_sims_full": 8, "c_puct": 1.5}
    bot = make_head_bot(eng, knobs, ArmSpec(label="head"), needs_no_drop_bot(spec), "v6_live2_ls")
    assert isinstance(bot, DeployHeadBot)
    assert bot._legal_set is True  # legal-set encoding → multi-window deploy decode
    assert bot._label == "head"    # repr label preserved byte-pure


def test_make_head_bot_legal_set_drift_asserts():
    """A caller passing a legal_set that disagrees with the encoding's policy_pool dispatch is a
    drift bug — make_head_bot asserts they agree."""
    import torch

    from hexo_rl.eval.deploy_strength_eval import _build_engine_for_model
    from scripts.evalfair.core import ArmSpec, make_head_bot

    model, _ = _small_model("v6")  # v6 → legal_set should be False
    eng = _build_engine_for_model(model, "v6", torch.device("cpu"))
    knobs = {"gumbel_m": 16, "c_visit": 50, "c_scale": 1.0, "n_sims_full": 8, "c_puct": 1.5}
    with pytest.raises(AssertionError, match="legal_set dispatch drift"):
        make_head_bot(eng, knobs, ArmSpec(label="head"), True, "v6")  # lie: legal_set=True on v6


def test_round_robin_cached_bot_routes_through_factory(tmp_path):
    """hexo_rl/eval/round_robin._CachedModelBot builds its player via build_player; a v6 ckpt
    dispatches ModelPlayer, a v6_live2_ls decode dispatches KClusterMCTSBot."""
    import torch

    from hexo_rl.eval.evaluator import ModelPlayer
    from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot
    from hexo_rl.eval.round_robin import _CachedModelBot

    # Save a tiny v6 checkpoint and load it through the cached bot.
    model, spec = _small_model("v6")
    ckpt = tmp_path / "tiny_v6.pt"
    torch.save({"model_state": model.state_dict(),
                "metadata": {"encoding_name": "v6"}}, ckpt)
    _CachedModelBot._NET_CACHE.clear()
    bot = _CachedModelBot(str(ckpt), n_sims=1, temperature=0.0, device=torch.device("cpu"))
    assert isinstance(bot._player, ModelPlayer)  # v6 flat → single-window ModelPlayer

    # v6_live2_ls decode of the same shape-identical arch → KClusterMCTSBot (no-drop).
    model2, spec2 = _small_model("v6_live2_ls")
    ckpt2 = tmp_path / "tiny_ls.pt"
    torch.save({"model_state": model2.state_dict(),
                "metadata": {"encoding_name": "v6_live2_ls"}}, ckpt2)
    _CachedModelBot._NET_CACHE.clear()
    bot2 = _CachedModelBot(str(ckpt2), n_sims=1, temperature=0.0, device=torch.device("cpu"))
    assert isinstance(bot2._player, KClusterMCTSBot)  # legal-set → no-drop KCluster
