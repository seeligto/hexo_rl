"""Reference rate (threat_moves) on D-VETO V2 arm-A losses."""
import sys, json, time
from pathlib import Path

REPO = Path('/home/timmy/Work/Hexo/hexo_rl')
sys.path.insert(0, str(REPO / 'vendor' / 'bots' / 'sealbot' / 'best'))
sys.path.insert(0, str(REPO / 'vendor' / 'bots' / 'sealbot'))
import warnings; warnings.filterwarnings('ignore')

from minimax_cpp import MinimaxBot
from game import Player as SealPlayer
from hexo_rl.eval.eval_board import make_eval_board

WIN_THRESHOLD = 99_999_000
sealbot = MinimaxBot(time_limit=60.0)
sealbot.max_depth = 7

class MG:
    def __init__(self, bd, cp, ml, mc):
        self.board = bd
        self.current_player = SealPlayer.A if cp == 1 else SealPlayer.B
        self.moves_left_in_turn = ml
        self.move_count = mc

def probe(snap):
    bd = {}
    for q, r, p in snap['board'].get_stones():
        bd[(q,r)] = SealPlayer.A if p == 1 else SealPlayer.B
    sealbot.get_move(MG(bd, snap['cp'], snap['mr'], len(bd)))
    return getattr(sealbot, 'last_score', None)

def build(moves):
    b = make_eval_board('v6_live2_ls', 4)
    snaps = []
    for q, r in moves:
        snaps.append({'ply': b.ply, 'cp': b.current_player, 'mr': b.moves_remaining, 'board': b.clone()})
        b.apply_move(q, r)
    return snaps

def hdx(snaps, hp):
    return [i for i, s in enumerate(snaps) if s['cp'] == hp and (s['mr'] == 2 or (s['ply'] == 0 and s['mr'] == 1))]

def t_star(snaps, hd):
    if not hd: return None
    sc = probe(snaps[hd[-1]])
    if sc is None or sc > -WIN_THRESHOLD: return None
    tsp = len(hd) - 1
    for pos in range(len(hd)-2, -1, -1):
        sc2 = probe(snaps[hd[pos]])
        if sc2 is None or sc2 > -WIN_THRESHOLD:
            tsp = pos + 1; break
        tsp = pos
    return hd[tsp]

d = json.load(open(REPO / 'scripts/arena/results/dveto_v2_nautilus150.json'))
games = d['games']['nautilus150_vs_sealbotd5']
losses = [g for g in games if not (
    (g['p1']=='nautilus150' and g['winner']=='p1') or
    (g['p2']=='nautilus150' and g['winner']=='p2'))]
print(f'D-VETO V2 losses: {len(losses)}', flush=True)

results = []
for gi, g in enumerate(losses):
    moves = [(m['x'], m['y']) for m in sorted(g['record']['moves'], key=lambda m: m['moveNumber'])]
    hp = 1 if g['p1'] == 'nautilus150' else -1
    op = -hp
    snaps = build(moves)
    hd = hdx(snaps, hp)
    print(f'  [{gi+1}/{len(losses)}] plies={len(moves)} head_turns={len(hd)}', end='', flush=True)
    t0 = time.time()
    ts = t_star(snaps, hd)
    if ts is None:
        print(f'  → no_t_star ({time.time()-t0:.1f}s)', flush=True)
        results.append({'walk_in_wm': False, 'walk_in_tm': False, 'class': 'no_t_star'})
        continue
    tsp = hd.index(ts)
    wi_wm = wi_tm = False
    for n in [2, 3]:
        tp = tsp - n
        if tp < 0: continue
        sidx = hd[tp]
        sc = probe(snaps[sidx])
        al = (sc is not None and sc <= -WIN_THRESHOLD)
        pw = (sc is not None and sc >= WIN_THRESHOLD)
        wm = len(snaps[sidx]['board'].winning_moves(op))
        tm = len(snaps[sidx]['board'].threat_moves(op))
        if not al and not pw and wm >= 2: wi_wm = True
        if not al and not pw and tm >= 2: wi_tm = True
        print(f'  N{n}:sc={int(sc) if sc else "?"} al={al} pw={pw} wm={wm} tm={tm}', end='', flush=True)
    results.append({'walk_in_wm': wi_wm, 'walk_in_tm': wi_tm, 't_star_ply': snaps[ts]['ply']})
    print(f'  → wm={wi_wm} tm={wi_tm} ({time.time()-t0:.1f}s)', flush=True)

n_wm = sum(1 for r in results if r['walk_in_wm'])
n_tm = sum(1 for r in results if r['walk_in_tm'])
N = len(results)
print(f'\nREFERENCE (D-VETO V2): wm={n_wm}/{N}={n_wm/N:.3f} tm={n_tm}/{N}={n_tm/N:.3f}', flush=True)
with open(REPO / 'reports/evalfair/ref_measure_dveto_v2_tm.json', 'w') as f:
    json.dump({'n_wm': n_wm, 'n_tm': n_tm, 'n': N, 'rate_wm': n_wm/N, 'rate_tm': n_tm/N, 'per': results}, f)
print('Saved.', flush=True)
