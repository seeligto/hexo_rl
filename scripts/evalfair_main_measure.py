"""Fair-book formation walk-in measurement."""
import sys, json, time, hashlib, random
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

def hd_indices(snaps, head_pn):
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

def head_pn(g):
    return 1 if g['head_as_p1'] else -1

def is_head_loss(g):
    return g['winner'] != ('p1' if g['head_as_p1'] else 'p2')

# Load fair-book losses
games = [json.loads(l) for l in open(REPO / 'reports/watchguard/verdict2/games.jsonl')]
fair_losses = [g for g in games if g['arm'] == 'fair' and is_head_loss(g)]

# Deduplicate
seen = set()
unique_losses = []
for g in fair_losses:
    h = hashlib.sha256(json.dumps(g['moves']).encode()).hexdigest()
    if h not in seen:
        seen.add(h)
        unique_losses.append(g)
eff_n = len(unique_losses)
print(f'Fair losses: {len(fair_losses)}, unique (eff_n): {eff_n}', flush=True)

results = []
for gi, g in enumerate(unique_losses):
    hpn = head_pn(g)
    opn = -hpn
    snaps = build_snaps(g['moves'])
    hdx = hd_indices(snaps, hpn)
    print(f'  [{gi+1}/{eff_n}] open={g["opening_idx"]:3d} plies={g["plies"]:4d} head_turns={len(hdx):3d}', end='', flush=True)
    t0 = time.time()
    t_star, n_pr = find_t_star(snaps, hdx)
    if t_star is None:
        print(f'  → no_t_star ({time.time()-t0:.1f}s)', flush=True)
        results.append({'opening_idx': g['opening_idx'], 'plies': g['plies'], 't_star': None, 'walk_in': False,
                        'class': 'no_proven_loss_at_d7', 'N2_dt': None, 'N3_dt': None, 'N2_sc': None, 'N3_sc': None,
                        'n_probes': n_pr})
        continue
    t_star_pos = hdx.index(t_star)
    t_star_ply = snaps[t_star]['ply']
    rec = {'opening_idx': g['opening_idx'], 'plies': g['plies'], 't_star_snap': t_star,
           't_star_ply': t_star_ply, 'n_probes': n_pr}
    any_wi = False
    for n in [2, 3]:
        tp = t_star_pos - n
        if tp < 0:
            rec[f'N{n}_snap'] = None; rec[f'N{n}_dt'] = None; rec[f'N{n}_sc'] = None; rec[f'N{n}_walk_in'] = None
            continue
        sample_idx = hdx[tp]
        al, sc = is_proven_loss(snaps[sample_idx])
        n_pr += 1
        is_pw = (sc is not None and sc >= WIN_THRESHOLD)
        # winning_moves (immediate completions) AND threat_moves (one-step threat creating moves)
        wm = snaps[sample_idx]['board'].winning_moves(opn)
        tm = snaps[sample_idx]['board'].threat_moves(opn)
        has_dt_wm = len(wm) >= 2
        has_dt_tm = len(tm) >= 2
        # Also check opponent's turn AFTER head plays 2 stones
        opp_after_snap_idx = sample_idx + 2  # opponent's turn start after head plays 2 stones
        wm_after = tm_after = []
        if opp_after_snap_idx < len(snaps) and snaps[opp_after_snap_idx]['cp'] == opn:
            wm_after = snaps[opp_after_snap_idx]['board'].winning_moves(opn)
            tm_after = snaps[opp_after_snap_idx]['board'].threat_moves(opn)
        rec[f'N{n}_snap'] = sample_idx
        rec[f'N{n}_ply'] = snaps[sample_idx]['ply']
        rec[f'N{n}_sc'] = float(sc) if sc is not None else None
        rec[f'N{n}_al'] = al
        rec[f'N{n}_pw'] = is_pw
        rec[f'N{n}_wm'] = len(wm)
        rec[f'N{n}_tm'] = len(tm)
        rec[f'N{n}_wm_after'] = len(wm_after)
        rec[f'N{n}_tm_after'] = len(tm_after)
        rec[f'N{n}_dt'] = has_dt_wm or has_dt_tm  # either winning_moves or threat_moves
        wi_wm = (not al) and (not is_pw) and has_dt_wm
        wi_tm = (not al) and (not is_pw) and has_dt_tm
        wi_after = (not al) and (not is_pw) and len(tm_after) >= 2
        wi = wi_wm  # per the pre-registered definition (winning_moves k=0)
        rec[f'N{n}_walk_in'] = wi
        rec[f'N{n}_walk_in_tm'] = wi_tm  # additional: threat_moves version
        rec[f'N{n}_walk_in_after'] = wi_after  # additional: threat_moves after head plays
        if wi: any_wi = True
        print(f'  N{n}:sc={int(sc) if sc else "N/A"} al={al} pw={is_pw} wm={len(wm)} tm={len(tm)} wm_a={len(wm_after)} tm_a={len(tm_after)} wi={wi}', end='', flush=True)
    rec['walk_in'] = any_wi
    rec['n_probes'] = n_pr
    results.append(rec)
    elapsed = time.time()-t0
    print(f'  → walk_in={any_wi} t*_ply={t_star_ply} ({elapsed:.1f}s)', flush=True)

# Save durable table
with open(REPO / 'reports/evalfair/per_loss_table.jsonl', 'w') as f:
    for r in results: f.write(json.dumps(r) + '\n')

# Summary stats
n_wi = sum(1 for r in results if r.get('walk_in'))
n_no_t = sum(1 for r in results if r.get('class') == 'no_proven_loss_at_d7')
n_meas = len(results)

# Also check alternative definitions
n_wi_tm = sum(1 for r in results if any(r.get(f'N{n}_walk_in_tm') for n in [2,3]))
n_wi_after = sum(1 for r in results if any(r.get(f'N{n}_walk_in_after') for n in [2,3]))

print(f'\n=== RESULTS ===')
print(f'Walk-in rate (wm k=0): {n_wi}/{n_meas} = {n_wi/n_meas:.3f}', flush=True)
print(f'Walk-in rate (tm k=0): {n_wi_tm}/{n_meas} = {n_wi_tm/n_meas:.3f}', flush=True)
print(f'Walk-in rate (tm_after k=0): {n_wi_after}/{n_meas} = {n_wi_after/n_meas:.3f}', flush=True)
print(f'No proven loss at d7: {n_no_t}/{n_meas}', flush=True)
print(f'eff_n={eff_n}', flush=True)
