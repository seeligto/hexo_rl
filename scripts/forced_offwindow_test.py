#!/usr/bin/env python
"""Forced-position off-window test — discriminate EXPOSURE vs DEFENSE.

The Phase-3 coherence "Gumbel defends / PUCT forceable" was a seed-parity EXPOSURE confound:
Gumbel's decisive play never REACHES the 17 odd-seed (-4,3) off-window-forcing positions PUCT
loses. This forces the question: replay each of PUCT's 17 losses, and at the off-window-forcing
ONSET swap the defender to Gumbel — does Gumbel make the in-window block PUCT missed (DEFENSE
edge survives) or also lose off-window (pure EXPOSURE, no defense difference)?

control = PUCT all the way (must reproduce the 17 losses → validates determinism).
test    = PUCT builds to the forcing onset, Gumbel defends from there.
"""
from __future__ import annotations
import json, random, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
from engine import Board
from hexo_rl.bots.offwindow_geom import HEX_AXES
from hexo_rl.diagnostics.forced_win_detector import cheb, is_off_window, window_center
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.offwindow_probe import oneturn_win_cells
from hexo_rl.bots.offwindow_adversary_bot import OffWindowAdversaryBot
from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.encoding import lookup as _lookup, normalize_encoding_name as _norm

ENC = "v6_live2_ls"
SIMS = 128
C_PUCT = 1.5
spec = _lookup(_norm(ENC))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_bot(ckpt):
    model, _spec, _label = load_model_with_encoding(Path(ckpt), dev)  # auto-detects v6_live2 family
    try: model.encoding = ENC   # override to the legal-set action space (state-dict-identical)
    except Exception: pass
    return KClusterMCTSBot(model, dev, n_sims=SIMS, c_puct=C_PUCT, temperature=0.0,
                           kept_plane_indices=list(spec.kept_plane_indices))


def play_swap(puct_bot, gumbel_bot, adversary, adv_side, max_plies, opening_plies, rng, swap=True):
    """play_game with the defender swapped PUCT->Gumbel at the first off-window-forcing onset.
    swap=False = PUCT all the way (control). Returns (off_window_win, plies, swapped_at)."""
    board = Board.with_encoding_name(ENC); state = GameState.from_board(board)
    puct_bot.reset(); gumbel_bot.reset(); adversary.reset()
    defender = puct_bot; swapped_at = None
    snap = None; last_move = None; last_mover = None; ply = 0
    while not board.check_win() and board.legal_move_count() > 0 and ply < max_plies:
        cp = board.current_player
        if ply < opening_plies:
            q, r = rng.choice(board.legal_moves())
        elif cp == adv_side:
            q, r = adversary.get_move(state, board)
        else:
            snap = board.clone()
            threats = oneturn_win_cells(board, adv_side)
            off = [c for c in threats if is_off_window(board, c, spec)] if threats else []
            if off and swap and swapped_at is None:        # off-window forcing onset → swap defender
                defender = gumbel_bot; swapped_at = ply
            q, r = defender.get_move(state, board)
        state = state.apply_move(board, q, r); last_move = (int(q), int(r)); last_mover = cp; ply += 1
    winner = board.winner(); adv_won = winner == adv_side; off_win = False
    if adv_won and last_mover == adv_side and last_move is not None and snap is not None:
        off_win = bool(is_off_window(snap, last_move, spec))
    return bool(adv_won and off_win), ply, swapped_at


def main():
    # the 17 PUCT off-window-loss games (gi/seed) from the coherence data
    txt = (REPO / "reports/p3_coherence/puct.json").read_text()
    dec = json.JSONDecoder(); i = 0; recs = []
    while i < len(txt):
        while i < len(txt) and txt[i] in " \n\t\r": i += 1
        if i >= len(txt): break
        o, j = dec.raw_decode(txt, i); recs.append(o); i = j
    losses = [r for r in recs if r.get("arm") == "exploit" and r.get("off_window_win")]
    gis = sorted(r["game"] for r in losses)
    print(f"PUCT off-window losses to replay: {len(gis)} (seeds {[7000+g for g in gis[:5]]}...)", flush=True)

    puct_bot = load_bot("checkpoints/p3_eval/checkpoint_00015001.pt")   # PUCT-15k
    gumbel_bot = load_bot("checkpoints/p3_eval/checkpoint_00015000.pt") # Gumbel-15k

    SEED_BASE, OPENING, MAXP = 7000, 0, 150
    ctrl_off = 0; test_off = 0; test_held = 0; swap_plies = []
    print(f"\n{'gi':>3} {'seed':>5} | PUCT-control | Gumbel-swapped (swap_ply)", flush=True)
    for gi in gis:
        seed = SEED_BASE + gi
        adv_side = 1 if gi % 2 == 0 else -1
        axis = HEX_AXES[gi % len(HEX_AXES)]
        # control: PUCT all the way
        np.random.seed(seed); random.seed(seed); rng = random.Random(seed)
        adv = OffWindowAdversaryBot(arm="exploit", encoding=ENC, axis=axis, seed=seed)
        c_off, _, _ = play_swap(puct_bot, gumbel_bot, adv, adv_side, MAXP, OPENING, rng, swap=False)
        # test: swap to Gumbel at forcing onset
        np.random.seed(seed); random.seed(seed); rng = random.Random(seed)
        adv = OffWindowAdversaryBot(arm="exploit", encoding=ENC, axis=axis, seed=seed)
        t_off, _, sp = play_swap(puct_bot, gumbel_bot, adv, adv_side, MAXP, OPENING, rng, swap=True)
        ctrl_off += c_off; test_off += t_off; test_held += (not t_off); swap_plies.append(sp)
        print(f"{gi:>3} {seed:>5} | {'LOSS' if c_off else 'held':>11} | "
              f"{'LOSS' if t_off else 'HELD':>5} (swap@{sp})", flush=True)
    n = len(gis)
    print(f"\n=== VERDICT ===", flush=True)
    print(f"PUCT-control off-window losses: {ctrl_off}/{n} (should be {n} → validates determinism)", flush=True)
    print(f"Gumbel-swapped off-window losses: {test_off}/{n}  | HELD: {test_held}/{n}", flush=True)
    if ctrl_off < n:
        print("  ⚠ control did not fully reproduce — seed/opening mismatch, interpret with care", flush=True)
    if test_off >= ctrl_off * 0.8:
        print("  → EXPOSURE confirmed: Gumbel ALSO loses these positions; no defense edge. The off-window\n"
              "    'inversion' was purely which positions each arm reaches.", flush=True)
    elif test_held >= n * 0.8:
        print("  → DEFENSE edge SURVIVES: Gumbel makes the in-window block PUCT missed on these positions.", flush=True)
    else:
        print("  → MIXED: partial defense edge.", flush=True)
    (REPO / "reports/p3_coherence/forced_test.json").write_text(json.dumps(
        {"n": n, "puct_control_off": ctrl_off, "gumbel_swapped_off": test_off,
         "gumbel_held": test_held, "swap_plies": swap_plies}, indent=2))
    print("\nwrote reports/p3_coherence/forced_test.json", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
