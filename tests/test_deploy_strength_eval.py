"""D-LOCALIZE P4 (TRACK B) — deploy-matched in-loop strength gate.

Pins the load-bearing properties the §D-LADDER instrument-mismatch finding requires:
  1. g=0 root noise → DETERMINISTIC SH winner (the deploy strength head; mctx
     gumbel_scale=0). A seed change must NOT change the played move.
  2. Missing deploy knobs HARD-ERROR (no silent PUCT/temp/64-sim fallback).
  3. The adaptive screen->confirm band CANNOT false-negative a true candidate: a screen
     WR >= screen_confirm_lo escalates to a full powered confirm; only a clear-reject
     (WR < lo) skips. The default lo sits a screen Wilson half-width below the bar.
  4. The pipeline gate honours deploy_strength_promoted over the legacy wr_best Wilson
     gate when the deploy_strength opponent is enabled.
"""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from hexo_rl.encoding import lookup as _lookup
from hexo_rl.eval.deploy_strength_eval import (
    EVAL_GUMBEL_SCALE,
    DeployStrengthConfig,
    DeployStrengthResult,
    extract_deploy_knobs,
)


def _tiny_model():
    spec = _lookup("v6")
    from hexo_rl.model.network import HexTacToeNet

    m = HexTacToeNet(in_channels=spec.n_planes, board_size=spec.board_size,
                     res_blocks=2, filters=32)
    m.eval()
    return m, spec


# ── (1) g=0 determinism — the deploy strength head ─────────────────────────────


def test_eval_gumbel_scale_is_zero() -> None:
    """The in-loop strength head zeroes the Gumbel root noise (deploy-eval g=0)."""
    assert EVAL_GUMBEL_SCALE == 0.0


def test_g0_played_move_is_seed_invariant() -> None:
    """g=0 → deterministic SH winner. Two different RNG seeds must agree (root noise
    zeroed), whereas the candidate-selection/score path would otherwise be seed-driven."""
    from engine import Board
    from hexo_rl.eval.gumbel_search_py import run_gumbel_on_board
    from hexo_rl.selfplay.inference import LocalInferenceEngine

    model, spec = _tiny_model()
    eng = LocalInferenceEngine(model, torch.device("cpu"), encoding_spec=spec)

    def play(seed: int):
        b = Board.with_encoding_name("v6")
        for _ in range(3):
            q, r = b.legal_moves()[0]
            b.apply_move(q, r)
        out = run_gumbel_on_board(eng, b, n_sims=48, m=16,
                                  gumbel_scale=0.0, rng=np.random.default_rng(seed))
        return out["played_move"]

    assert play(1) == play(987654), "g=0 SH winner must be seed-invariant (deterministic)"


# ── (2) hard-error on missing deploy knobs (no silent fallback) ────────────────


def test_extract_deploy_knobs_hard_errors_on_gap() -> None:
    with pytest.raises(KeyError):
        extract_deploy_knobs({})
    with pytest.raises(KeyError):
        extract_deploy_knobs({"selfplay": {"gumbel_m": 16}})  # missing the rest


def test_extract_deploy_knobs_reads_run_config() -> None:
    cfg = {
        "selfplay": {"gumbel_m": 16, "c_visit": 50.0, "c_scale": 1.0,
                     "playout_cap": {"n_sims_full": 150}},
        "mcts": {"c_puct": 1.5},
    }
    k = extract_deploy_knobs(cfg)
    assert k["gumbel_m"] == 16 and k["n_sims_full"] == 150 and k["c_puct"] == 1.5


# ── (2b) §D-PFIT WS1: deploy head decodes in the encoding's TRAINED action space ──


def _make_deploy_evaluator(enc: str):
    """Minimal DeployStrengthEvaluator on CPU (no forward pass needed — the test only
    inspects the bots' decode flag)."""
    from hexo_rl.model.network import HexTacToeNet
    from hexo_rl.eval.deploy_strength_eval import DeployStrengthEvaluator

    model = HexTacToeNet(encoding=enc).eval()
    config = {
        "encoding": enc,
        "selfplay": {"gumbel_m": 16, "c_visit": 50.0, "c_scale": 1.0,
                     "playout_cap": {"n_sims_full": 150}},
        "mcts": {"c_puct": 1.5},
    }
    ev = DeployStrengthEvaluator(model, torch.device("cpu"), config, {}, 0.55)
    return ev, model


def test_deploy_strength_legal_set_derived_from_encoding() -> None:
    """The promotion-gate deploy head must decode in the SAME action space the encoding
    trains under: MULTI-WINDOW no-drop for legal_set_scatter_max encodings (v6_live2_ls),
    single-window for the rest (v6_live2). Else the strength gate handicaps the off-window
    defense the net was trained to express (train↔deploy decode mismatch, §D-PFIT WS1)."""
    ev_ls, m_ls = _make_deploy_evaluator("v6_live2_ls")
    assert ev_ls._cand._legal_set is True
    assert ev_ls._best_bot(m_ls)._legal_set is True, "best anchor must flip symmetrically"

    ev_sw, _ = _make_deploy_evaluator("v6_live2")
    assert ev_sw._cand._legal_set is False, "single-window encoding must stay legal_set=False"


# ── (3) pre-registered screen->confirm band ────────────────────────────────────


def test_screen_confirm_lo_is_one_half_width_below_bar() -> None:
    """The default escalate floor = promotion_winrate − one screen Wilson half-width, so a
    true candidate at the bar (scattering down by sampling noise) is still escalated —
    the band cannot false-negative it."""
    cfg = DeployStrengthConfig.from_cfg({"screen_n": 80, "confirm_n": 200}, promotion_winrate=0.55)
    hw = 1.96 * math.sqrt(0.25 / 80)
    assert cfg.screen_confirm_lo == pytest.approx(round(0.55 - hw, 3))
    assert cfg.screen_confirm_lo < 0.55, "escalate floor must sit BELOW the bar"
    assert cfg.screen_confirm_hi == 1.0, "escalate region extends to 1.0 (every eligible WR confirms)"


def test_band_escalates_every_promotion_eligible_screen() -> None:
    """A candidate whose true WR is exactly the bar produces screen point estimates that
    scatter around it; the band's lower edge is a half-width below, so the realized
    scatter stays inside the escalate region (cannot drop a true candidate)."""
    cfg = DeployStrengthConfig.from_cfg({"screen_n": 80}, promotion_winrate=0.55)
    # WR exactly at the bar, and one half-width below — both must escalate.
    hw = 1.96 * math.sqrt(0.25 / 80)
    assert 0.55 >= cfg.screen_confirm_lo
    assert (0.55 - hw) >= cfg.screen_confirm_lo - 1e-9


# ── (4) pipeline gate honours the deploy_strength decision ─────────────────────


def _pipeline_cfg(tmp_path: Path, deploy_promoted: bool) -> dict:
    return {
        "eval_pipeline": {
            "enabled": True,
            "eval_interval": 100,
            "report_dir": str(tmp_path / "eval"),
            "db_path": str(tmp_path / "eval" / "results.db"),
            "ratings_plot_path": str(tmp_path / "eval" / "rc.png"),
            "opponents": {
                "best_checkpoint": {"enabled": True, "n_games": 10, "model_sims": 8, "opponent_sims": 8},
                "sealbot": {"enabled": False},
                "random": {"enabled": False},
                "bootstrap_anchor": {"enabled": False},
                "deploy_strength": {"enabled": True, "stride": 1},
            },
            "gating": {"promotion_winrate": 0.55, "best_model_path": str(tmp_path / "best.pt"),
                       "bootstrap_floor": {"enabled": False}},
            "bradley_terry": {"anchor_player": "checkpoint_0", "regularization": 1e-6},
        }
    }


@patch("hexo_rl.eval.deploy_strength_eval.DeployStrengthEvaluator")
@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_pipeline_promotes_on_deploy_strength_true_even_if_wr_best_low(
    mock_eval_cls: MagicMock, mock_dse_cls: MagicMock, tmp_path: Path
) -> None:
    """deploy_strength confirmed+promoted=True must promote even when the legacy wr_best
    (0.3) would block — the deploy gate REPLACES the Wilson gate."""
    from hexo_rl.eval.eval_pipeline import EvalPipeline
    from hexo_rl.eval.evaluator import EvalResult

    mock_eval = MagicMock()
    mock_eval.evaluate_vs_model.return_value = EvalResult(
        win_rate=0.3, win_count=3, n_games=10, colony_wins=0,
    )
    mock_eval.model = MagicMock()
    mock_eval.device = torch.device("cpu")
    mock_eval.config = {"encoding": "v6"}
    mock_eval_cls.return_value = mock_eval

    mock_dse = MagicMock()
    mock_dse.run.return_value = DeployStrengthResult(
        wr_screen=0.7, wr_confirm=0.66, confirmed=True, promoted=True,
        elo_vs_best=120.0, ci_lo_boot=15.0, ci_hi_boot=240.0, n_games=280,
        copy_multiplier=1.0, distinct_per_pair_min=140, head_fired_frac=1.0,
        sealbot_wr=0.6, reason="PROMOTE",
    )
    mock_dse_cls.return_value = mock_dse

    pipeline = EvalPipeline(_pipeline_cfg(tmp_path, True), torch.device("cpu"))
    result = pipeline.run_evaluation(MagicMock(), 1000, MagicMock())
    assert result["deploy_strength_promoted"] is True
    assert result["promoted"] is True, "deploy_strength promote must override low wr_best"


@patch("hexo_rl.eval.deploy_strength_eval.DeployStrengthEvaluator")
@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_pipeline_blocks_on_deploy_strength_false_even_if_wr_best_high(
    mock_eval_cls: MagicMock, mock_dse_cls: MagicMock, tmp_path: Path
) -> None:
    """deploy_strength promoted=False must BLOCK even when the legacy wr_best (0.9) would
    pass — the deploy bootstrap-CI gate is authoritative."""
    from hexo_rl.eval.eval_pipeline import EvalPipeline
    from hexo_rl.eval.evaluator import EvalResult

    mock_eval = MagicMock()
    mock_eval.evaluate_vs_model.return_value = EvalResult(
        win_rate=0.9, win_count=9, n_games=10, colony_wins=0,
    )
    mock_eval.model = MagicMock()
    mock_eval.device = torch.device("cpu")
    mock_eval.config = {"encoding": "v6"}
    mock_eval_cls.return_value = mock_eval

    mock_dse = MagicMock()
    mock_dse.run.return_value = DeployStrengthResult(
        wr_screen=0.58, wr_confirm=0.56, confirmed=True, promoted=False,
        elo_vs_best=20.0, ci_lo_boot=-30.0, ci_hi_boot=80.0, n_games=280,
        copy_multiplier=1.0, distinct_per_pair_min=140, head_fired_frac=1.0,
        sealbot_wr=0.5, reason="BLOCKED: CI straddles 0",
    )
    mock_dse_cls.return_value = mock_dse

    pipeline = EvalPipeline(_pipeline_cfg(tmp_path, False), torch.device("cpu"))
    result = pipeline.run_evaluation(MagicMock(), 1000, MagicMock())
    assert result["deploy_strength_promoted"] is False
    assert result["promoted"] is False, "deploy_strength block must override high wr_best"


@patch("hexo_rl.eval.deploy_strength_eval.DeployStrengthEvaluator")
@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_pipeline_blocks_when_deploy_strength_crashes_no_puct_fallback(
    mock_eval_cls: MagicMock, mock_dse_cls: MagicMock, tmp_path: Path
) -> None:
    """FAIL-SAFE: deploy_strength enabled but the runner raised (per-opponent isolation
    swallows it → no deploy_strength_* keys). The gate must BLOCK, NOT silently fall back
    to the legacy wr_best=0.9 Wilson gate (the §D-LADDER PUCT/temp/64-sim head)."""
    from hexo_rl.eval.eval_pipeline import EvalPipeline
    from hexo_rl.eval.evaluator import EvalResult

    mock_eval = MagicMock()
    mock_eval.evaluate_vs_model.return_value = EvalResult(
        win_rate=0.9, win_count=9, n_games=10, colony_wins=0,
    )
    mock_eval.model = MagicMock()
    mock_eval.device = torch.device("cpu")
    mock_eval.config = {"encoding": "v6"}
    mock_eval_cls.return_value = mock_eval

    mock_dse = MagicMock()
    mock_dse.run.side_effect = RuntimeError("deploy head exploded")
    mock_dse_cls.return_value = mock_dse

    pipeline = EvalPipeline(_pipeline_cfg(tmp_path, False), torch.device("cpu"))
    result = pipeline.run_evaluation(MagicMock(), 1000, MagicMock())
    assert "deploy_strength_promoted" not in result
    assert result["promoted"] is False, (
        "crashed deploy gate must block, never fall back to the PUCT wr_best gate"
    )


# ── (5) SealBot external-bar arm is droppable (sealbot_games=0) ────────────────
# The arm is NOT in the promotion gate (sealbot_wr reported-only) and is the
# dominant deploy-round cost (SealBot depth-5 ~4s/move). D-PRELAUNCH run2 drops it.


def test_sealbot_games_defaults_to_confirm_n_and_honours_override() -> None:
    d_default = DeployStrengthConfig.from_cfg({"confirm_n": 200}, promotion_winrate=0.55)
    assert d_default.sealbot_games == 200, "default back-compat = confirm_n"
    d_zero = DeployStrengthConfig.from_cfg(
        {"confirm_n": 100, "sealbot_games": 0}, promotion_winrate=0.55)
    assert d_zero.sealbot_games == 0, "explicit 0 drops the arm"
    d_small = DeployStrengthConfig.from_cfg(
        {"confirm_n": 100, "sealbot_games": 50}, promotion_winrate=0.55)
    assert d_small.sealbot_games == 50, "decoupled from confirm_n"


def _run_evaluator_recording_pairs(sealbot_games: int):
    """Run DeployStrengthEvaluator.run with _play_pair + aggregation stubbed so no
    real games play. Returns (result, list of (label_a,label_b,n_games) played)."""
    from unittest.mock import patch
    import hexo_rl.eval.deploy_strength_eval as dse

    model, spec = _tiny_model()
    config = {
        "encoding": "v6",
        "selfplay": {"gumbel_m": 16, "c_visit": 50.0, "c_scale": 1.0,
                     "playout_cap": {"n_sims_full": 32}},
        "mcts": {"c_puct": 1.5},
    }
    deploy_cfg = {"screen_n": 8, "confirm_n": 8, "sealbot_games": sealbot_games,
                  "screen_confirm_lo": 0.0}  # force escalation
    calls: list = []

    def fake_play_pair(bot_a, bot_b, label_a, label_b, enc, n_games, *a, **k):
        calls.append((label_a, label_b, n_games))
        # winner alternates so cand wins enough to escalate + clear
        return [{"p1": label_a if gi % 2 == 0 else label_b,
                 "p2": label_b if gi % 2 == 0 else label_a,
                 "winner": "p1" if gi % 2 == 0 else "p2",  # cand wins each
                 "plies": 4, "moves": [[0, 0]], "head_fired": True}
                for gi in range(n_games)]

    with patch.object(dse, "_play_pair", side_effect=fake_play_pair), \
         patch.object(dse, "aggregate_games",
                      return_value={"rungs": [{"label": "cand", "elo": 10.0},
                                              {"label": "best", "elo": 0.0}]}), \
         patch.object(dse, "bootstrap_ratings_ci", return_value={"cand": (1.0, 20.0)}), \
         patch.object(dse, "effective_n_guard",
                      return_value={"low_power_warning": False, "copy_multiplier": 1.0}), \
         patch.object(dse, "distinct_per_pair", return_value={("best", "cand"): 8}):
        ev = dse.DeployStrengthEvaluator(model, torch.device("cpu"), config,
                                         deploy_cfg=deploy_cfg, promotion_winrate=0.55)
        res = ev.run(model)
    return res, calls


def test_run_skips_sealbot_arm_when_zero() -> None:
    res, calls = _run_evaluator_recording_pairs(sealbot_games=0)
    seal_calls = [c for c in calls if "sealbot" in (c[0], c[1])]
    assert seal_calls == [], "sealbot pair must NOT be played when sealbot_games=0"
    assert res.sealbot_wr is None, "sealbot_wr must be None when arm dropped"
    # promotion path still runs on the vs-best games (screen + confirm played)
    assert res.confirmed is True


def test_run_plays_sealbot_arm_when_positive() -> None:
    res, calls = _run_evaluator_recording_pairs(sealbot_games=6)
    seal_calls = [c for c in calls if "sealbot" in (c[0], c[1])]
    assert len(seal_calls) == 1 and seal_calls[0][2] == 6, "sealbot arm plays sealbot_games"
    assert res.sealbot_wr is not None
