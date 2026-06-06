"""Off-window exploit measurement (D-EXPLOIT) — importable core shared by the
standalone probe (scripts/exploit_probe.py) and the in-pipeline exploitability eval
opponent (Evaluator.evaluate_vs_offwindow_adversary).

EVAL-PATH ONLY (pinned by tests/test_offwindow_adversary_eval_path_only.py). Measures how
often an off-window adversary forces a win the model has no logit to block (off-window at
the model's last decision) vs a frozen ModelPlayer running its own MCTS = genuine resistance.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np

from engine import Board
from hexo_rl.bots.offwindow_adversary_bot import OffWindowAdversaryBot
from hexo_rl.bots.offwindow_geom import HEX_AXES
from hexo_rl.diagnostics.forced_win_detector import cheb, is_off_window, window_center
from hexo_rl.env.game_state import GameState

WIN_LEN = 6


def _threat_player(side: int) -> int:
    return 0 if side == 1 else 1


def oneturn_win_cells(board: Any, side: int) -> list[tuple[int, int]]:
    """Cells where ``side`` can complete a win within ONE turn (2 stones): one-stone
    6-completions (engine level-(WIN_LEN-1) threats, incl. gap-fills) PLUS the immediate
    ends of contiguous runs >= WIN_LEN-2 reachable to WIN_LEN. The set the opponent must
    defend; used to test whether the model's only defense was off-window."""
    tp = _threat_player(side)
    cells: set[tuple[int, int]] = set()
    for (q, r, lvl, p) in board.get_threats():
        if p == tp and lvl == WIN_LEN - 1:
            cells.add((int(q), int(r)))
    mine = {(int(q), int(r)) for (q, r, p) in board.get_stones() if p == side}
    for u in HEX_AXES:
        for s in mine:
            if (s[0] - u[0], s[1] - u[1]) in mine:
                continue
            q, r = s
            length = 1
            while (q + u[0], r + u[1]) in mine:
                q += u[0]
                r += u[1]
                length += 1
            if length < WIN_LEN - 2:
                continue
            plus = (q + u[0], r + u[1])
            minus = (s[0] - u[0], s[1] - u[1])
            plus_open = board.get(*plus) == 0
            minus_open = board.get(*minus) == 0
            need = WIN_LEN - length
            for end, du, dv, other_open in (
                (plus, u[0], u[1], minus_open),
                (minus, -u[0], -u[1], plus_open),
            ):
                if board.get(int(end[0]), int(end[1])) != 0:
                    continue
                cq, cr = end
                empties = 0
                while board.get(cq, cr) == 0 and empties < need:
                    empties += 1
                    cq += du
                    cr += dv
                if empties >= need or (length >= WIN_LEN - 2 and other_open):
                    cells.add((int(end[0]), int(end[1])))
    return sorted(cells)


def play_game(model_bot, adversary, board_enc, adv_side, spec, sims, max_plies, opening_plies, rng):
    """One adversary-vs-model game. Returns a per-game record dict."""
    board = Board.with_encoding_name(board_enc)
    state = GameState.from_board(board)
    model_bot.reset()
    adversary.reset()
    model_side = -adv_side

    model_last_snapshot = None
    last_move = None
    last_mover = None
    max_offwin_threat_cheb = -1
    any_offwin_forcing_position = False

    ply = 0
    while not board.check_win() and board.legal_move_count() > 0 and ply < max_plies:
        cp = board.current_player
        if ply < opening_plies:
            q, r = rng.choice(board.legal_moves())
        elif cp == adv_side:
            q, r = adversary.get_move(state, board)
        else:
            model_last_snapshot = board.clone()
            threats = oneturn_win_cells(board, adv_side)
            if threats:
                ctr = window_center([(s[0], s[1]) for s in board.get_stones()])
                off = [c for c in threats if is_off_window(board, c, spec)]
                if off:
                    any_offwin_forcing_position = True
                    max_offwin_threat_cheb = max(
                        max_offwin_threat_cheb, max(cheb(c, ctr) for c in off)
                    )
            q, r = model_bot.get_move(state, board)
        state = state.apply_move(board, q, r)
        last_move = (int(q), int(r))
        last_mover = cp
        ply += 1

    winner = board.winner()
    adv_won = winner == adv_side
    win_cell_off_window = False
    win_cell_cheb = None
    model_had_inwindow_block = None
    if adv_won and last_mover == adv_side and last_move is not None and model_last_snapshot is not None:
        snap = model_last_snapshot
        win_cell_off_window = bool(is_off_window(snap, last_move, spec))
        ctr = window_center([(s[0], s[1]) for s in snap.get_stones()])
        win_cell_cheb = int(cheb(last_move, ctr))
        threats = oneturn_win_cells(snap, adv_side)
        if threats:
            model_had_inwindow_block = any(not is_off_window(snap, c, spec) for c in threats)

    return {
        "adv_side": int(adv_side),
        "model_side": int(model_side),
        "winner": (int(winner) if winner is not None else None),
        "adversary_won": bool(adv_won),
        "plies": int(ply),
        "off_window_win": bool(adv_won and win_cell_off_window),
        "strict_off_window_forced": bool(
            adv_won and win_cell_off_window and model_had_inwindow_block is False
        ),
        "win_cell": list(last_move) if last_move is not None else None,
        "win_cell_cheb": win_cell_cheb,
        "win_cell_off_window": bool(win_cell_off_window),
        "model_had_inwindow_block": model_had_inwindow_block,
        "any_offwindow_forcing_position": bool(any_offwin_forcing_position),
        "max_offwindow_threat_cheb": (int(max_offwin_threat_cheb) if max_offwin_threat_cheb >= 0 else None),
    }


def summarize(arm: str, recs: list[dict]) -> dict:
    n = len(recs)
    if n == 0:
        return {"arm": arm, "n_games": 0}
    adv_wins = sum(r["adversary_won"] for r in recs)
    ow = sum(r["off_window_win"] for r in recs)
    strict = sum(r["strict_off_window_forced"] for r in recs)
    draws = sum(r["winner"] is None for r in recs)
    chebs = [r["win_cell_cheb"] for r in recs if r["off_window_win"] and r["win_cell_cheb"] is not None]
    forcing = sum(r["any_offwindow_forcing_position"] for r in recs)
    return {
        "arm": arm,
        "n_games": n,
        "adversary_win_rate": round(adv_wins / n, 4),
        "off_window_forced_win_rate": round(ow / n, 4),
        "strict_off_window_forced_rate": round(strict / n, 4),
        "draw_rate": round(draws / n, 4),
        "any_offwindow_forcing_position_rate": round(forcing / n, 4),
        "off_window_win_cheb_min": (min(chebs) if chebs else None),
        "off_window_win_cheb_max": (max(chebs) if chebs else None),
        "off_window_win_cheb_mean": (round(sum(chebs) / len(chebs), 2) if chebs else None),
        "mean_plies": round(sum(r["plies"] for r in recs) / n, 1),
    }


def run_adversary_games(model_bot, encoding, spec, arm, n_games, sims, *,
                        max_plies=150, opening_plies=6, seed_base=0):
    """Run ``n_games`` adversary(arm)-vs-model games (alternating colors + axes) and
    return (summary, per-game records). ``model_bot`` is a ready ModelPlayer."""
    recs = []
    for gi in range(n_games):
        seed = seed_base + gi
        rng = random.Random(seed)
        np.random.seed(seed)
        random.seed(seed)
        adv_side = 1 if gi % 2 == 0 else -1
        axis = HEX_AXES[gi % len(HEX_AXES)]
        adversary = OffWindowAdversaryBot(arm=arm, encoding=encoding, axis=axis, seed=seed)
        rec = play_game(model_bot, adversary, encoding, adv_side, spec, sims,
                        max_plies, opening_plies, rng)
        rec.update({"arm": arm, "game": gi, "seed": seed, "axis": list(axis)})
        recs.append(rec)
    return summarize(arm, recs), recs
