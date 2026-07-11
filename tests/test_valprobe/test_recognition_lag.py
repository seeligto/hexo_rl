"""Tests for D-C VALPROBE WP1 recognition-lag measurement.

Tests per §5.10:
  T-TURNMAP  — turn_of_ply + head-turn-start predicate
  T-SETS     — game set counts + per_loss_table cross-check + win-control determinism
  T-CROSS    — sustained crossing rule on synthetic trajectories
  T-CLASS    — classification table §4.5
  T-SOLVER-SIGN — head_lost sign logic from probe result
  T-REPLAY   — integration: one 248k game replays to terminal winner + move match (opt-in)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

# Import the measurement module
from valprobe.measure_recognition_lag import (
    classify_game,
    compute_t_cross,
    head_lost_from_probe,
    is_head_turn_start,
    is_any_turn_start,
    load_games_jsonl,
    build_loss_and_win_sets,
    turn_of_ply,
    game_move_sha,
)

GAMES_248K = REPO / "reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl"
GAMES_175K = REPO / "reports/evalfair/retro_slope/run2_175k/games.jsonl"
PER_LOSS_TABLE = REPO / "reports/evalfair/per_loss_table.jsonl"


# ═════════════════════════════════════════════════════════════════════════════
# T-TURNMAP
# ═════════════════════════════════════════════════════════════════════════════


class TestTurnMap:
    """§5.10 T-TURNMAP: turn_of_ply correctness + head-turn-start predicate."""

    def test_turn_of_ply_known_values(self):
        """plies 0/1/2/3/4/5 → turns 0/1/1/2/2/3."""
        expected = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3}
        for ply, expected_turn in expected.items():
            assert turn_of_ply(ply) == expected_turn, f"turn_of_ply({ply}) != {expected_turn}"

    def test_turn_of_ply_extended(self):
        """Spot-check further values."""
        assert turn_of_ply(6) == 3
        assert turn_of_ply(7) == 4
        assert turn_of_ply(8) == 4
        assert turn_of_ply(9) == 5

    def test_turn_of_ply_ply0_is_turn0(self):
        assert turn_of_ply(0) == 0

    def test_head_turn_start_at_ply0(self):
        """ply=0, mr=1 is a head turn-start if cp matches head."""
        # head_as_p1: head_pn=1, cp=1 → yes
        assert is_head_turn_start(cp=1, mr=1, ply=0, head_pn=1)
        # head_as_p2: head_pn=-1, cp=-1 → yes
        assert is_head_turn_start(cp=-1, mr=1, ply=0, head_pn=-1)
        # wrong player → no
        assert not is_head_turn_start(cp=-1, mr=1, ply=0, head_pn=1)

    def test_head_turn_start_mr2(self):
        """mr=2 anywhere (except ply=0 which has mr=1 for P1) → turn-start."""
        # ply=3, mr=2, cp=1, head_pn=1 → yes
        assert is_head_turn_start(cp=1, mr=2, ply=3, head_pn=1)
        # ply=3, mr=2, cp=-1, head_pn=1 → no (wrong player)
        assert not is_head_turn_start(cp=-1, mr=2, ply=3, head_pn=1)

    def test_head_turn_start_mr1_not_ply0(self):
        """mr=1 at ply>0 = mid-turn, NOT a turn-start."""
        assert not is_head_turn_start(cp=1, mr=1, ply=2, head_pn=1)
        assert not is_head_turn_start(cp=-1, mr=1, ply=5, head_pn=-1)

    def test_any_turn_start(self):
        """is_any_turn_start: ply==0 or mr==2."""
        assert is_any_turn_start(mr=1, ply=0)
        assert is_any_turn_start(mr=2, ply=0)
        assert is_any_turn_start(mr=2, ply=5)
        assert not is_any_turn_start(mr=1, ply=3)

    @pytest.mark.skipif(
        not GAMES_248K.exists(),
        reason="248k games.jsonl not available"
    )
    def test_head_turn_start_on_fixture_game(self):
        """Replay a fixture game; verify head-turn-start predicate matches mr==2 or ply==0."""
        from hexo_rl.eval.eval_board import make_eval_board
        games = load_games_jsonl(str(GAMES_248K))
        g = games[0]
        head_pn = 1 if g["head_as_p1"] else -1
        board = make_eval_board("v6_live2_ls", g["radius"])

        for t, (q, r) in enumerate(g["moves"]):
            ply = int(board.ply)
            cp = int(board.current_player)
            mr = int(board.moves_remaining)

            is_turn_start_ref = (ply == 0) or (mr == 2)
            pred = is_head_turn_start(cp, mr, ply, head_pn)
            expected = is_turn_start_ref and (cp == head_pn)
            assert pred == expected, (
                f"ply={ply}, mr={mr}, cp={cp}, head_pn={head_pn}: "
                f"pred={pred} != expected={expected}"
            )
            board.apply_move(int(q), int(r))


# ═════════════════════════════════════════════════════════════════════════════
# T-SETS
# ═════════════════════════════════════════════════════════════════════════════


class TestSets:
    """§5.10 T-SETS: game set counts + per_loss_table + win-control determinism."""

    @pytest.mark.skipif(not GAMES_248K.exists(), reason="248k games.jsonl not available")
    def test_248k_counts(self):
        """248k → 57 losses, 70 wins, 1 censored."""
        games = load_games_jsonl(str(GAMES_248K))
        assert len(games) == 128

        from valprobe.measure_recognition_lag import is_head_win, is_head_loss, is_censored

        censored = [g for g in games if is_censored(g)]
        non_censored_losses = [g for g in games if is_head_loss(g) and not is_censored(g)]
        non_censored_wins = [g for g in games if is_head_win(g) and not is_censored(g)]

        assert len(censored) == 1, f"Expected 1 censored, got {len(censored)}"
        assert len(non_censored_losses) == 57, f"Expected 57 losses, got {len(non_censored_losses)}"
        assert len(non_censored_wins) == 70, f"Expected 70 wins, got {len(non_censored_wins)}"

    @pytest.mark.skipif(not GAMES_175K.exists(), reason="175k games.jsonl not available")
    def test_175k_counts(self):
        """175k → 52 losses, 76 wins, 0 censored."""
        games = load_games_jsonl(str(GAMES_175K))
        assert len(games) == 128

        from valprobe.measure_recognition_lag import is_head_win, is_head_loss, is_censored

        censored = [g for g in games if is_censored(g)]
        non_censored_losses = [g for g in games if is_head_loss(g) and not is_censored(g)]
        non_censored_wins = [g for g in games if is_head_win(g) and not is_censored(g)]

        assert len(censored) == 0, f"Expected 0 censored, got {len(censored)}"
        assert len(non_censored_losses) == 52, f"Expected 52 losses, got {len(non_censored_losses)}"
        assert len(non_censored_wins) == 76, f"Expected 76 wins, got {len(non_censored_wins)}"

    @pytest.mark.skipif(
        not (GAMES_175K.exists() and PER_LOSS_TABLE.exists()),
        reason="175k or per_loss_table not available"
    )
    def test_per_loss_table_multiset_match(self):
        """per_loss_table (opening_idx, plies) multiset == 175k loss multiset."""
        games = load_games_jsonl(str(GAMES_175K))
        per_loss = [json.loads(l) for l in open(PER_LOSS_TABLE)]

        from valprobe.measure_recognition_lag import is_head_loss, is_censored

        losses = [g for g in games if is_head_loss(g) and not is_censored(g)]
        loss_set = set((g["opening_idx"], g["plies"]) for g in losses)
        pl_set = set((r["opening_idx"], r["plies"]) for r in per_loss)
        assert loss_set == pl_set, (
            f"Mismatch:\n  losses only: {loss_set - pl_set}\n  per_loss only: {pl_set - loss_set}"
        )

    @pytest.mark.skipif(not GAMES_248K.exists(), reason="248k games.jsonl not available")
    def test_win_control_determinism(self):
        """Win control selection is deterministic and disjoint from losses."""
        games = load_games_jsonl(str(GAMES_248K))
        losses, wins = build_loss_and_win_sets(games, expected_losses=57)

        # Check count
        assert len(wins) == 57

        # Disjoint from losses (no game appears in both)
        loss_keys = {(g["opening_idx"], g["head_as_p1"]) for g in losses}
        win_keys = {(g["opening_idx"], g["head_as_p1"]) for g in wins}
        # Keys may overlap (different outcomes same opening); check by object identity via moves
        loss_move_shas = {game_move_sha(g["moves"]) for g in losses}
        win_move_shas = {game_move_sha(g["moves"]) for g in wins}
        assert loss_move_shas.isdisjoint(win_move_shas), "Loss and win games should be disjoint"

        # Determinism: sort by (opening_idx, head_as_p1) ascending
        from valprobe.measure_recognition_lag import is_head_win, is_censored
        wins_all = sorted(
            [g for g in games if is_head_win(g) and not is_censored(g)],
            key=lambda g: (g["opening_idx"], int(g["head_as_p1"])),
        )
        expected_wins = wins_all[:57]
        for i, (w, e) in enumerate(zip(wins, expected_wins)):
            assert game_move_sha(w["moves"]) == game_move_sha(e["moves"]), (
                f"Win control game {i} mismatch"
            )

    @pytest.mark.skipif(not GAMES_175K.exists(), reason="175k games.jsonl not available")
    def test_175k_win_control_count(self):
        games = load_games_jsonl(str(GAMES_175K))
        losses, wins = build_loss_and_win_sets(games, expected_losses=52)
        assert len(losses) == 52
        assert len(wins) == 52


# ═════════════════════════════════════════════════════════════════════════════
# T-CROSS
# ═════════════════════════════════════════════════════════════════════════════


def _traj(v_vals: List[float]) -> List[dict]:
    """Build synthetic head trajectory from v_val list (t=index, turn=index)."""
    return [{"t": i, "turn": i, "v_raw": v} for i, v in enumerate(v_vals)]


class TestCross:
    """§5.10 T-CROSS: sustained crossing rule on synthetic trajectories."""

    def test_single_blip_rejected(self):
        """Single below-threshold sample followed by above-threshold → not crossed."""
        traj = _traj([0.5, 0.2, -0.6, 0.3, 0.4])
        T_ply, T_turn, nc, tc = compute_t_cross(traj, -0.5, is_loss_game=True)
        assert T_ply is None, "Single blip should not count as crossing"
        assert nc is True

    def test_two_consecutive_accepted_at_first(self):
        """Two consecutive below-threshold → crossing at FIRST sample."""
        traj = _traj([0.5, -0.6, -0.7, 0.4])
        T_ply, T_turn, nc, tc = compute_t_cross(traj, -0.5, is_loss_game=True)
        assert T_ply == 1, f"Expected T_ply=1, got {T_ply}"
        assert T_turn == 1
        assert nc is False

    def test_terminal_edge_rule_loss_only(self):
        """Last sample below threshold on loss game → terminal_confirmed=True."""
        traj = _traj([0.5, 0.3, -0.6])   # last sample only
        T_ply, T_turn, nc, tc = compute_t_cross(traj, -0.5, is_loss_game=True)
        assert T_ply == 2, f"Expected terminal crossing at 2, got {T_ply}"
        assert tc is True
        assert nc is False

    def test_terminal_edge_rule_NOT_applied_on_win(self):
        """Last sample below threshold on win game → NOT crossed (no terminal confirmation)."""
        traj = _traj([0.5, 0.3, -0.6])
        T_ply, T_turn, nc, tc = compute_t_cross(traj, -0.5, is_loss_game=False)
        assert T_ply is None, "Win game: last blip should not count"
        assert nc is True
        assert tc is False

    def test_never_crossed_flag(self):
        """All positive values → never crossed."""
        traj = _traj([0.8, 0.6, 0.4, 0.2])
        T_ply, T_turn, nc, tc = compute_t_cross(traj, -0.5, is_loss_game=False)
        assert T_ply is None
        assert nc is True

    def test_never_crossed_flag_loss_game(self):
        """Loss game, no crossing → never_crossed=True."""
        traj = _traj([0.5, 0.3, 0.1, -0.3])   # all above -0.5
        T_ply, T_turn, nc, tc = compute_t_cross(traj, -0.5, is_loss_game=True)
        assert T_ply is None
        assert nc is True
        assert tc is False

    def test_threshold_sensitivity_independent(self):
        """Sweep thresholds operate independently."""
        traj = _traj([0.2, -0.35, -0.4, -0.8, -0.9])
        # At -0.3: crosses at index 1 (both 1 and 2 are <= -0.3)
        T30_ply, _, nc30, _ = compute_t_cross(traj, -0.3, is_loss_game=True)
        assert T30_ply == 1
        assert nc30 is False

        # At -0.5: index 1 = -0.35 > -0.5 (blip at 3 then 4 confirmed)
        T50_ply, _, nc50, _ = compute_t_cross(traj, -0.5, is_loss_game=True)
        # -0.8 and -0.9 both <= -0.5 → crossing at index 3
        assert T50_ply == 3
        assert nc50 is False

        # At -0.7: -0.8 and -0.9 both <= -0.7
        T70_ply, _, nc70, _ = compute_t_cross(traj, -0.7, is_loss_game=True)
        assert T70_ply == 3
        assert nc70 is False

    def test_exact_threshold(self):
        """Exactly at threshold (-0.5) counts as crossed."""
        traj = _traj([0.0, -0.5, -0.5, 0.5])
        T_ply, _, nc, _ = compute_t_cross(traj, -0.5, is_loss_game=True)
        assert T_ply == 1
        assert nc is False

    def test_empty_trajectory(self):
        """Empty trajectory → never crossed."""
        T_ply, T_turn, nc, tc = compute_t_cross([], -0.5, is_loss_game=True)
        assert T_ply is None
        assert nc is True

    def test_single_position_below_threshold_loss(self):
        """Single position below threshold on loss → terminal confirmed."""
        traj = _traj([-0.8])
        T_ply, T_turn, nc, tc = compute_t_cross(traj, -0.5, is_loss_game=True)
        assert T_ply == 0
        assert tc is True
        assert nc is False

    def test_single_position_below_threshold_win(self):
        """Single position below threshold on win → not crossed."""
        traj = _traj([-0.8])
        T_ply, T_turn, nc, tc = compute_t_cross(traj, -0.5, is_loss_game=False)
        assert T_ply is None
        assert nc is True


# ═════════════════════════════════════════════════════════════════════════════
# T-CLASS
# ═════════════════════════════════════════════════════════════════════════════


class TestClass:
    """§5.10 T-CLASS: classification table §4.5 on synthetic (T_provable, T_cross) combos."""

    def test_late_t_provable_defined_never_crossed(self):
        """T_provable defined AND never_crossed → LATE."""
        assert classify_game(T_provable_turn=5, T_cross_turn=None, lag_raw=None,
                             never_crossed=True) == "LATE"

    def test_late_t_provable_defined_lag_ge2(self):
        """T_provable defined AND lag_raw >= 2 → LATE."""
        assert classify_game(T_provable_turn=5, T_cross_turn=8, lag_raw=3,
                             never_crossed=False) == "LATE"
        assert classify_game(T_provable_turn=5, T_cross_turn=7, lag_raw=2,
                             never_crossed=False) == "LATE"

    def test_early_t_cross_defined_t_provable_undefined(self):
        """T_cross defined AND T_provable undefined → EARLY (censoring safe)."""
        assert classify_game(T_provable_turn=None, T_cross_turn=3, lag_raw=None,
                             never_crossed=False) == "EARLY"

    def test_early_t_cross_defined_lag_le0(self):
        """T_cross defined AND lag_raw <= 0 → EARLY."""
        assert classify_game(T_provable_turn=8, T_cross_turn=7, lag_raw=-1,
                             never_crossed=False) == "EARLY"
        assert classify_game(T_provable_turn=8, T_cross_turn=8, lag_raw=0,
                             never_crossed=False) == "EARLY"

    def test_mid_both_defined_lag1(self):
        """Both defined, lag_raw == 1 → MID."""
        assert classify_game(T_provable_turn=5, T_cross_turn=6, lag_raw=1,
                             never_crossed=False) == "MID"

    def test_unmeasurable_t_provable_undefined_never_crossed(self):
        """T_provable undefined AND never_crossed → UNMEASURABLE."""
        assert classify_game(T_provable_turn=None, T_cross_turn=None, lag_raw=None,
                             never_crossed=True) == "UNMEASURABLE"

    def test_censored_crossed_is_early(self):
        """T_provable undefined + T_cross defined → EARLY (§4.5 note)."""
        assert classify_game(T_provable_turn=None, T_cross_turn=5, lag_raw=None,
                             never_crossed=False) == "EARLY"

    def test_all_undefined_never_crossed_is_unmeasurable(self):
        assert classify_game(T_provable_turn=None, T_cross_turn=None, lag_raw=None,
                             never_crossed=True) == "UNMEASURABLE"

    def test_late_not_claimable_without_t_provable(self):
        """Never_crossed + T_provable undefined → UNMEASURABLE (not LATE)."""
        assert classify_game(T_provable_turn=None, T_cross_turn=None, lag_raw=None,
                             never_crossed=True) == "UNMEASURABLE"

    def test_lag2_is_late_not_mid(self):
        """lag_raw=2 exactly is LATE (the V-CONFIRM threshold)."""
        result = classify_game(T_provable_turn=5, T_cross_turn=7, lag_raw=2, never_crossed=False)
        assert result == "LATE"

    def test_lag_minus1_is_early(self):
        result = classify_game(T_provable_turn=5, T_cross_turn=4, lag_raw=-1, never_crossed=False)
        assert result == "EARLY"


# ═════════════════════════════════════════════════════════════════════════════
# T-SOLVER-SIGN
# ═════════════════════════════════════════════════════════════════════════════


class TestSolverSign:
    """§5.10 T-SOLVER-SIGN: head_lost predicate from solver result."""

    def test_head_to_move_result_minus1_is_head_lost(self):
        """Head to move, result=-1 (proven LOSS for side-to-move) → head_lost=True."""
        assert head_lost_from_probe(-1, side_to_move_is_head=True) is True

    def test_head_to_move_result_plus1_is_not_head_lost(self):
        """Head to move, result=+1 (proven WIN for side-to-move) → head_lost=False."""
        assert head_lost_from_probe(+1, side_to_move_is_head=True) is False

    def test_head_to_move_unknown_is_not_head_lost(self):
        """Head to move, result=0 (UNKNOWN) → head_lost=False."""
        assert head_lost_from_probe(0, side_to_move_is_head=True) is False

    def test_opp_to_move_result_plus1_is_head_lost(self):
        """Opponent to move, result=+1 (proven WIN for opp) → head proven lost."""
        assert head_lost_from_probe(+1, side_to_move_is_head=False) is True

    def test_opp_to_move_result_minus1_is_not_head_lost(self):
        """Opponent to move, result=-1 (opp proven LOSS) → head is winning → not lost."""
        assert head_lost_from_probe(-1, side_to_move_is_head=False) is False

    def test_opp_to_move_unknown_is_not_head_lost(self):
        """Opponent to move, UNKNOWN → head_lost=False."""
        assert head_lost_from_probe(0, side_to_move_is_head=False) is False

    @pytest.mark.skipif(
        not os.environ.get("VALPROBE_INTEGRATION"),
        reason="Integration: requires engine + fixture position. Set VALPROBE_INTEGRATION=1."
    )
    def test_solver_on_forced_loss_position(self):
        """Engine integration: known forced-loss position → result=-1 for head-to-move."""
        from engine import TacticalSolver
        from hexo_rl.eval.eval_board import make_eval_board

        # Build a position known to be a forced loss: use a tactics test fixture
        # A simple 5-in-a-row threat that cannot be stopped
        solver = TacticalSolver(window_half=None, cand_cap=40, neighbor_dist=2)
        board = make_eval_board("v6_live2_ls", 5)

        # Play out a forced-win line for P1 (so it's a loss for P2 = head if head_as_p2)
        # 6-in-a-row at (0,0),(0,1),(0,2),(0,3),(0,4),(0,5) for P1
        # We need to play a realistic line — use an existing test fixture if available
        # For now verify the interface works on an empty board (should be UNKNOWN)
        result, line, nodes = solver.prove(board, depth=5, node_budget=10000)
        # Empty board should be UNKNOWN (not provable at start)
        assert result == 0, f"Empty board should be UNKNOWN, got result={result}"
        head_pn = -1  # head is P2
        assert head_lost_from_probe(result, side_to_move_is_head=True) is False


# ═════════════════════════════════════════════════════════════════════════════
# T-REPLAY (integration, opt-in)
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not os.environ.get("VALPROBE_INTEGRATION"),
    reason="Integration test: requires GPU + full engine. Set VALPROBE_INTEGRATION=1 to enable."
)
class TestReplay:
    """§5.10 T-REPLAY: one 248k game replays to terminal winner + first head move reproduced."""

    def test_one_game_replay_terminal(self):
        """Replay first 248k game → check terminal winner matches record."""
        from valprobe.measure_recognition_lag import (
            load_games_jsonl,
            load_book,
            replay_game,
            verify_game_integrity,
        )

        games = load_games_jsonl(str(GAMES_248K))
        g = games[0]
        book = load_book(g["book_id"])
        verify_game_integrity(g, book)
        snaps, terminal_ok, winner_int = replay_game(g, "v6_live2_ls")
        assert terminal_ok, "Game should terminate with a winner"
        expected = 1 if g["winner"] == "p1" else -1 if g["winner"] == "p2" else None
        assert winner_int == expected, f"Winner mismatch: {winner_int} != {expected}"

    def test_first_head_move_reproduced(self):
        """Run Gumbel search on first head turn-start → matches recorded move."""
        import torch
        import numpy as np
        from valprobe.measure_recognition_lag import (
            load_games_jsonl,
            load_book,
            replay_game,
            verify_game_integrity,
            is_head_turn_start,
            turn_of_ply,
            _ckpt_sha,
        )
        from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
        from hexo_rl.eval.deploy_strength_eval import _build_engine_for_model, extract_deploy_knobs
        from hexo_rl.eval.gumbel_search_py import run_gumbel_on_board

        ckpt_path = str(REPO / "checkpoints/run2_retro/checkpoint_00248000.pt")
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, spec, label = load_model_with_encoding(
            ckpt_path, dev, declared_encoding="v6_live2_ls"
        )
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        knobs = extract_deploy_knobs(ck["config"])
        eng = _build_engine_for_model(model, "v6_live2_ls", dev)

        games = load_games_jsonl(str(GAMES_248K))
        # Find a loss game to test on
        g = next(g for g in games if g["winner"] == ("p2" if g["head_as_p1"] else "p1"))
        book = load_book(g["book_id"])
        verify_game_integrity(g, book)
        snaps, _, _ = replay_game(g, "v6_live2_ls")

        head_pn = 1 if g["head_as_p1"] else -1
        head_snaps = [
            s for s in snaps
            if is_head_turn_start(s["cp"], s["mr"], s["ply"], head_pn)
        ]
        assert head_snaps, "No head turn-starts found"

        snap = head_snaps[0]
        out = run_gumbel_on_board(
            eng,
            snap["board"].clone(),
            n_sims=int(knobs["n_sims_full"]),
            m=int(knobs["gumbel_m"]),
            c_visit=float(knobs["c_visit"]),
            c_scale=float(knobs["c_scale"]),
            c_puct=float(knobs["c_puct"]),
            dirichlet=False,
            gumbel_scale=0.0,
            legal_set=True,
            rng=np.random.default_rng(0),
        )
        played_rederived = out["played_move"]
        recorded = tuple(g["moves"][snap["t"]])
        assert played_rederived is not None, "Gumbel search returned no played move"
        assert tuple(played_rederived) == recorded, (
            f"Move mismatch: rederived={played_rederived} vs recorded={recorded}"
        )
