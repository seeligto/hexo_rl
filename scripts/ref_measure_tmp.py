"""Reference rate measurement on D-VETO V2 arm-A losses."""
import sys, json, time
from pathlib import Path

REPO = Path('/home/timmy/Work/Hexo/hexo_rl')
sys.path.insert(0, str(REPO / 'vendor' / 'bots' / 'sealbot' / 'best'))
sys.path.insert(0, str(REPO / 'vendor' / 'bots' / 'sealbot'))

import warnings; warnings.filterwarnings('ignore')

from minimax_cpp import MinimaxBot as _MinimaxBot
from game import Player as SealPlayer
from hexo_rl.eval.eval_board import make_eval_board

WIN_THRESHOLD = 99_999_000
ENCODING = 'v6_live2_ls'
RADIUS = 4

sealbot = _MinimaxBot(time_limit=60.0)
sealbot.max_depth = 7

class MockGame:
    def __init__(self, bd, cp, ml, mc):
        self.board = bd
        self.current_player = SealPlayer.A if cp == 1 else SealPlayer.B
        self.moves_left_in_turn = ml
        self.move_count = mc

def probe_d7(snap):
    bd = {}
    for q, r, p in snap['board'].get_stones():
        bd[(q,r)] = SealPlayer.A if p == 1 else SealPlayer.B
    mg = MockGame(bd, snap['cp'], snap['mr'], len(bd))
    sealbot.get_move(mg)
    return getattr(sealbot, 'last_score', None)

def is_proven_loss(snap):
    sc = probe_d7(snap)
    return sc is not None and sc <= -WIN_THRESHOLD, sc

def build_snaps(moves):
    board = make_eval_board(ENCODING, RADIUS)
    snaps = []
    for q, r in moves:
        snaps.append({'ply': board.ply, 'cp': board.current_player, 'mr': board.moves_remaining, 'board': board.clone()})
        board.apply_move(q, r)
    return snaps

def get_hd_indices(snaps, head_pn):
    return [i for i, s in enumerate(snaps) if s['cp'] == head_pn and (s['mr'] == 2 or (s['ply'] == 0 and s['mr'] == 1))]

def find_t_star(snaps, hdx):
    if not hdx: return None, 0
    last_loss, _ = is_proven_loss(snaps[hdx[-1]])
    n = 1
    if not last_loss: return None, n
    t_star_pos = len(hdx) - 1
    for pos in range(len(hdx)-2, -1, -1):
        il, _ = is_proven_loss(snaps[hdx[pos]])
        n += 1
        if not il:
            t_star_pos = pos + 1
            break
        t_star_pos = pos
    return hdx[t_star_pos], n

# Load D-VETO V2 arm A games
d = json.load(open(REPO / 'scripts/arena/results/dveto_v2_nautilus150.json'))
games = d['games']['nautilus150_vs_sealbotd5']
losses = [g for g in games if not (
    (g['p1'] == 'nautilus150' and g['winner'] == 'p1') or
    (g['p2'] == 'nautilus150' and g['winner'] == 'p2')
)]
print(f'D-VETO V2 nautilus150 losses: {len(losses)}', flush=True)

results = []
for gi, g in enumerate(losses):
    moves_raw = sorted(g['record']['moves'], key=lambda m: m['moveNumber'])
    moves = [(m['x'], m['y']) for m in moves_raw]
    head_pn = 1 if g['p1'] == 'nautilus150' else -1
    opp_pn = -head_pn
    snaps = build_snaps(moves)
    hdx = get_hd_indices(snaps, head_pn)
    print(f'  [{gi+1}/{len(losses)}] plies={len(moves)} head_turns={len(hdx)}', end='', flush=True)
    t0 = time.time()
    t_star, n_pr = find_t_star(snaps, hdx)
    if t_star is None:
        print(f'  → no_t_star ({time.time()-t0:.1f}s)', flush=True)
        results.append({'walk_in': False, 'class': 'no_t_star'})
        continue
    t_star_pos = hdx.index(t_star)
    any_wi = False
    for n in [2, 3]:
        tp = t_star_pos - n
        if tp < 0: continue
        al, sc = is_proven_loss(snaps[hdx[tp]])
        n_pr += 1
        is_pw = (sc is not None and sc >= WIN_THRESHOLD)
        wm = snaps[hdx[tp]]['board'].winning_moves(opp_pn)
        has_dt = len(wm) >= 2
        wi = (not al) and (not is_pw) and has_dt
        if wi: any_wi = True
        print(f'  N={n}: al={al} pw={is_pw} dt={len(wm)} wi={wi}', end='', flush=True)
    results.append({'walk_in': any_wi, 't_star_ply': snaps[t_star]['ply']})
    print(f'  → walk_in={any_wi} ({time.time()-t0:.1f}s)', flush=True)

n_wi = sum(1 for r in results if r['walk_in'])
n_total = len(results)
print(f'\nREFERENCE RESULT: {n_wi}/{n_total} = {n_wi/n_total:.3f} walk-in rate', flush=True)

# Save result
import json as js
with open(REPO / 'reports/evalfair/ref_measure_dveto_v2.json', 'w') as f:
    js.dump({'n_wi': n_wi, 'n_losses': n_total, 'rate': n_wi/n_total, 'per_game': results}, f, indent=2)
print(f'Saved to reports/evalfair/ref_measure_dveto_v2.json', flush=True)
