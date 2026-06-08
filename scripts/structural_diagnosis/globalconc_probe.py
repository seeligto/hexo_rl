#!/usr/bin/env python3
"""§D-GLOBALCONC — mid-game GLOBAL (whole-board, build-up) concentration discriminator.

THE GATE for the §D-OVERSPREAD follow-on. §D-OVERSPREAD found every concentration signal
present/rising — but ALL of it was measured at LOCAL TACTICAL positions (won in-window
forced-win snapshots; fork-approach pools). The over-spread it explains (own ncomp 14->22) is
a GLOBAL STRATEGIC build-up property. Different scales. This probe asks whether the
concentration signal exists at the GLOBAL / build-up scale — the genuinely-missing measurement.

Pre-registration (locked BEFORE running): investigation/globalconc_2026-06-08/PREREGISTRATION.md

POOL (the NEW regime): turn-start BUILD-UP snapshots (NOT forced-win turn-starts, NOT
fork-approach) — skip any snapshot where the mover OR opponent has an immediate winning_turn.
ply-band SWEPT (rolling). Bucketed by source checkpoint_step {30k,53k,87.5k}.

MEASURE 1 — VALUE: AUC_globalconc = P(value ranks CONCENTRATED build-up > SCATTERED) at
  matched stones AND matched eventual outcome (STRATIFIED median-split Mann-Whitney, pooled).
  PLUS the naive (unstratified) AUC to expose the stone-confound (the D1 trap).
MEASURE 2 — POLICY PRIOR (the untested node): AUC_prior_conc = P(prior(concentrating move) >
  prior(new-front move)) within-position (inherently stone-matched) + concentration LIFT.

VERDICTS (pre-registered, 0.60 bar): GLOBAL-SIGNAL-ABSENT (<=0.60 esp. falling) /
  GLOBAL-SIGNAL-PRESENT (>0.60 flat-rising) / MIXED (value vs policy-prior split).

EVAL-ONLY. Read-only on banked replays + checkpoints. Zero geometry literals (everything from
the encoding spec + detector + turn_wins primitive). New untracked script.

Run:
  cd /home/timmy/Work/hexo_rl && PYTHONPATH=. .venv/bin/python \
    scripts/structural_diagnosis/globalconc_probe.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from engine import Board
from hexo_rl.diagnostics.forced_win_detector import HEX_AXES, cheb
from hexo_rl.encoding import lookup, normalize_encoding_name
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.k_cluster_mcts_bot import _aggregate_priors
from hexo_rl.training.checkpoints import load_inference_model

# turn-correct primitive (NOT depth-1) — the build-up exclusion uses the turn-correct win set
sys.path.insert(0, str(Path(__file__).resolve().parent))
from turn_wins import winning_turn_cells  # noqa: E402

_NB = HEX_AXES + [(-q, -r) for (q, r) in HEX_AXES]   # 6-neighbour hex adjacency (coherence_overspread)


# ---- GLOBAL concentration labels (whole-board scale) ------------------------------------
def _components_sizes(cells):
    seen, out = set(), []
    for s in cells:
        if s in seen:
            continue
        sz = 0
        q = deque([s]); seen.add(s)
        while q:
            c = q.popleft(); sz += 1
            for dq, dr in _NB:
                nx = (c[0] + dq, c[1] + dr)
                if nx in cells and nx not in seen:
                    seen.add(nx); q.append(nx)
        out.append(sz)
    return out


def _fronts_cheb(cells, thresh):
    """Number of distinct fronts: stones within chebyshev <= thresh merged (union-find).

    Robustness variant of the hex-adjacency ncomp at a relaxed clustering threshold (the D2
    FRONTIER_BAND lesson — is 'number of fronts' robust to the clustering radius?)."""
    cs = list(cells)
    n = len(cs)
    if n == 0:
        return 0
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if cheb(cs[i], cs[j]) <= thresh:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    return len({find(k) for k in range(n)})


def _components_labeled(cells):
    """Return (label: cell->comp_id, sizes: list, largest_id). Hex adjacency."""
    label = {}
    sizes = []
    cid = 0
    for s in cells:
        if s in label:
            continue
        sz = 0
        q = deque([s]); label[s] = cid
        while q:
            c = q.popleft(); sz += 1
            for dq, dr in _NB:
                nx = (c[0] + dq, c[1] + dr)
                if nx in cells and nx not in label:
                    label[nx] = cid; q.append(nx)
        sizes.append(sz); cid += 1
    largest = int(np.argmax(sizes)) if sizes else -1
    return label, sizes, largest


def _terminal_winner(mv, name):
    b = Board.with_encoding_name(name)
    for (q, r) in mv:
        try:
            b.apply_move(q, r)
        except Exception:
            return None
        if b.check_win():
            return int(b.winner())
    return None


def _other(side):
    return -side


# ---- pool build -------------------------------------------------------------------------
def build_pool(files, name, spec, want_steps, ply_lo, ply_hi, max_pos, seed):
    """Build-up turn-start snapshots across the WIDE band; sub-bands filter on stored ply."""
    kept = list(spec.kept_plane_indices)
    S, P = spec.trunk_size, spec.n_source_planes
    rows = []
    gid = 0
    for fn in files:
        try:
            fh = open(fn)
        except FileNotFoundError:
            print(f"  (skip missing {fn})", file=sys.stderr); continue
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("game_length", 0) <= 0:
                continue
            src = int(d.get("checkpoint_step", 0))
            if src not in want_steps:
                continue
            mv = [(int(q), int(r)) for (q, r) in d["moves"]]
            winner = _terminal_winner(mv, name)
            this_gid = gid; gid += 1
            b = Board.with_encoding_name(name)
            i, n = 0, len(mv)
            while i < n:
                cp = int(b.current_player)
                ply = int(b.ply)
                if ply_lo <= ply < ply_hi:
                    snap = b.clone()
                    # build-up only: neither side has an immediate this-turn forced win
                    if not winning_turn_cells(snap, cp) and not winning_turn_cells(snap, _other(cp)):
                        stones = [(int(q), int(r), int(pp)) for (q, r, pp) in snap.get_stones()]
                        mine = {(q, r) for (q, r, pp) in stones if pp == cp}
                        if len(mine) >= 4:          # need >=4 own stones for a meaningful structure
                            szs = _components_sizes(mine)
                            ncomp = len(szs)
                            largest_frac = max(szs) / len(mine)
                            ncomp_cheb2 = _fronts_cheb(mine, 2)
                            flat = np.asarray(snap.to_tensor(), dtype=np.float32).reshape(P, S, S)
                            if winner is None:
                                oc = -1
                            elif winner == cp:
                                oc = 1
                            else:
                                oc = 0
                            rows.append({
                                "wire": flat[kept], "prefix": mv[:i], "side": cp,
                                "src_step": src, "ply": ply, "gid": this_gid,
                                "stones": len(mine), "ncomp": ncomp,
                                "ncomp_cheb2": ncomp_cheb2, "largest_frac": largest_frac,
                                "outcome": oc,
                            })
                # advance one turn
                while i < n:
                    q, r = mv[i]
                    try:
                        b.apply_move(q, r)
                    except Exception:
                        i = n; break
                    i += 1
                    if b.check_win():
                        break
                    if int(b.current_player) != cp:
                        break
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_pos and len(rows) > max_pos:
        rows = rows[:max_pos]
    return rows


# ---- AUC primitives ---------------------------------------------------------------------
def _mwu(pos, neg):
    """Mann-Whitney U statistic count (concordant + 0.5 ties) and the pair count (nw*nl)."""
    nw, nl = len(pos), len(neg)
    if nw == 0 or nl == 0:
        return 0.0, 0
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    U = ranks[:nw].sum() - nw * (nw + 1) / 2.0     # # of (pos>neg) + 0.5 ties
    return float(U), nw * nl


def stratified_auc(values, concflag, strata_key, valid):
    """Pooled stratified median-split AUC: P(value(concentrated) > value(scattered) | stratum).

    concflag: per-row int in {+1 concentrated, -1 scattered, 0 excluded (median tie)}.
    strata_key: per-row hashable stratum (stone_band, outcome). valid: per-row bool mask.
    """
    num = 0.0; den = 0
    groups = defaultdict(lambda: ([], []))   # stratum -> (conc_vals, scat_vals)
    for k in range(len(values)):
        if not valid[k] or concflag[k] == 0:
            continue
        cv, sv = groups[strata_key[k]]
        (cv if concflag[k] > 0 else sv).append(values[k])
    for (cv, sv) in groups.values():
        U, w = _mwu(np.asarray(cv), np.asarray(sv))
        num += U; den += w
    return (num / den) if den else float("nan"), den


def make_concflag(ncomp, strata_key, valid):
    """Within each stratum, concentrated = ncomp <= stratum median, scattered = ncomp > median.

    FEWER components = more concentrated. Rows at the exact median when it splits unevenly are
    handled by the <= / > rule (ties on the median go to 'concentrated'); a stratum whose ncomp
    is constant produces all-concentrated (no scattered) -> contributes 0 pairs (den 0)."""
    by = defaultdict(list)
    for k in range(len(ncomp)):
        if valid[k]:
            by[strata_key[k]].append(k)
    flag = np.zeros(len(ncomp), dtype=np.int8)
    for ks in by.values():
        vals = np.array([ncomp[k] for k in ks])
        med = np.median(vals)
        for k in ks:
            flag[k] = 1 if ncomp[k] <= med else -1
    return flag


def make_concflag_global(scalar, valid, higher_is_concentrated):
    """Single-stratum (naive) median split on a concentration scalar — NO stone/outcome match."""
    idx = [k for k in range(len(scalar)) if valid[k]]
    med = np.median([scalar[k] for k in idx]) if idx else 0.0
    flag = np.zeros(len(scalar), dtype=np.int8)
    for k in idx:
        if higher_is_concentrated:
            flag[k] = 1 if scalar[k] >= med else -1
        else:
            flag[k] = 1 if scalar[k] <= med else -1
    return flag


# ---- policy-prior concentration -----------------------------------------------------------
def rebuild(name, prefix):
    b = Board.with_encoding_name(name)
    st = GameState.from_board(b)
    for (q, r) in prefix:
        st = st.apply_move(b, q, r)
    return b, st


def precompute_policy_cache(name, kept, in_channels, rows):
    """Rebuild each row ONCE (the expensive prefix replay), cache the model-input tensor +
    centers + legal + the BOARD-ONLY concentration masks (checkpoint-independent).

    Per-legal-move classification via the labelled own-component map (O(legal*6), exact):
      dncomp = 1 - k, where k = #distinct own components adjacent to the cell
               (k=0 -> +1 NEW FRONT; k=1 -> 0 extend; k>=2 -> merge/bridge, negative).
      touches_largest = the largest own component is among those k.
      CONCENTRATING = touches_largest AND dncomp <= 0 (grows/merges the dominant mass).
      SPREADING     = (not touches_largest)            (isolated new front, or grows a
                       PERIPHERAL/secondary front) -> the over-spread-relevant 'scatter' move.
      adj/isolated  = the trivial floor (adjacent to ANY own stone vs empty space).
    """
    cache = []
    for row in rows:
        b, st = rebuild(name, row["prefix"])
        legal = list(b.legal_moves())
        if len(legal) < 2:
            continue
        cp = row["side"]
        mine = {(int(q), int(r)) for (q, r, pp) in b.get_stones() if int(pp) == cp}
        label, _sizes, largest = _components_labeled(mine)
        dncomp = np.empty(len(legal), dtype=np.float64)
        touch_large = np.zeros(len(legal), dtype=bool)
        adj = np.zeros(len(legal), dtype=bool)
        for i, (q, r) in enumerate(legal):
            comps = set()
            for dq, dr in _NB:
                nb = (q + dq, r + dr)
                if nb in label:
                    comps.add(label[nb])
            k = len(comps)
            dncomp[i] = 1 - k
            touch_large[i] = largest in comps
            adj[i] = k > 0
        conc = touch_large & (dncomp <= 0)
        spread = ~touch_large                    # isolated OR only-peripheral fronts
        periph = adj & (~touch_large)            # ADJACENT to own force but NOT the largest =
                                                 # grows a secondary/peripheral front (the
                                                 # over-spread-relevant 'scatter' move, controlled
                                                 # for the trivial adjacency floor)
        tensor, centers = st.to_tensor()
        if tensor.shape[1] != in_channels:
            tensor = tensor[:, kept]
        cache.append({
            "tensor": np.ascontiguousarray(tensor, dtype=np.float32),
            "centers": list(centers), "legal": legal,
            "view_size": int(tensor.shape[-1]),
            "dncomp": dncomp, "conc": conc, "spread": spread, "adj": adj,
            "main": touch_large, "periph": periph,
            "gid": row["gid"],
        })
    return cache


def score_policy(model, device, cache):
    """Per checkpoint: forward each cached row, aggregate prior, combine with board-only masks.

    Returns aligned lists (over rows with >=1 of each compared type):
      auc_conc       = P(prior(CONCENTRATING) > prior(SPREADING=isolated|peripheral)) — DOMINATED
                       by the adjacency floor (isolated moves carry ~0 prior), so it ~= auc_adj.
      auc_main_periph= P(prior(MAIN-mass-adjacent) > prior(PERIPHERAL-adjacent)) — BOTH adjacent,
                       so the trivial adjacency is controlled OUT: this is the genuinely
                       over-spread-relevant concentration signal (does the prior prefer growing the
                       DOMINANT mass over a secondary front?). (§D-GLOBALCONC red-team correction.)
      drift          = E_prior[dncomp] - E_uniform[dncomp]  (>0 = prior FRAGMENTS more than chance)
      auc_adj        = P(prior(adjacent) > prior(isolated))                   (trivial floor)
      lift_conc      = prior mass on CONCENTRATING / uniform share
    """
    auc_conc, auc_main_periph, drift, auc_adj, lift_conc, gids = [], [], [], [], [], []
    for c in cache:
        x = torch.from_numpy(c["tensor"]).float().to(device)
        with torch.no_grad():
            log_p, _v, _ = model(x)
        priors = np.asarray(_aggregate_priors(log_p.cpu().numpy(), c["centers"], c["legal"],
                                              c["view_size"]), dtype=np.float64)
        conc, spread, adj, dnc = c["conc"], c["spread"], c["adj"], c["dncomp"]
        main, periph = c["main"], c["periph"]
        gids.append(c["gid"])
        # over-spread-relevant AUC (note: dominated by the adjacency floor)
        if conc.any() and spread.any():
            U, w = _mwu(priors[conc], priors[spread])
            auc_conc.append(U / w if w else np.nan)
            nconc = int(conc.sum())
            lift_conc.append(float(priors[conc].sum()) / (nconc / len(c["legal"])))
        else:
            auc_conc.append(np.nan); lift_conc.append(np.nan)
        # THE clean concentration signal: main-mass-adjacent vs peripheral-adjacent (adjacency controlled)
        if main.any() and periph.any():
            U, w = _mwu(priors[main], priors[periph])
            auc_main_periph.append(U / w if w else np.nan)
        else:
            auc_main_periph.append(np.nan)
        # fragmentation drift: prior vs uniform expected dncomp
        drift.append(float((priors * dnc).sum() - dnc.mean()))
        # trivial floor
        if adj.any() and (~adj).any():
            U, w = _mwu(priors[adj], priors[~adj])
            auc_adj.append(U / w if w else np.nan)
        else:
            auc_adj.append(np.nan)
    return auc_conc, auc_main_periph, drift, auc_adj, lift_conc, gids


# ---- bootstrap over games -----------------------------------------------------------------
def boot_over_games(gids, stat_fn, nboot, rng):
    """Bootstrap CI by resampling GAME ids (the independent unit) with replacement."""
    by_g = defaultdict(list)
    for idx, g in enumerate(gids):
        by_g[g].append(idx)
    glist = list(by_g)
    if not glist:
        return (float("nan"), float("nan"))
    out = []
    for _ in range(nboot):
        pick = rng.integers(0, len(glist), len(glist))
        sel = []
        for gi in pick:
            sel.extend(by_g[glist[gi]])
        out.append(stat_fn(sel))
    out = np.asarray([o for o in out if o == o], dtype=np.float64)
    if len(out) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def discover_ckpts(dirs):
    found = {}
    for dd in dirs:
        for cp in Path(dd).glob("checkpoint_*.pt"):
            tail = cp.stem.replace("checkpoint_", "")
            digits = tail.split("_")[0]
            if not digits.isdigit():
                continue
            step = int(digits.lstrip("0") or "0")
            found.setdefault(step, cp)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", default="v6_live2")
    ap.add_argument("--replays", nargs="+",
                    default=sorted(str(p) for p in Path(
                        "investigation/coherence_2026-06-08/replays").glob("games_2026-06-0*.jsonl")))
    ap.add_argument("--ckpt-dirs", nargs="+",
                    default=["investigation/coherence_2026-06-08/checkpoints",
                             "investigation/fragility_2026-06-07/checkpoints"])
    ap.add_argument("--pool-steps", type=int, nargs="+", default=[30000, 53000, 87500])
    ap.add_argument("--ply-lo", type=int, default=20, help="WIDE band low (sub-bands filter within)")
    ap.add_argument("--ply-hi", type=int, default=65, help="WIDE band high")
    ap.add_argument("--bands", type=str,
                    default="20-35,25-40,30-45,35-50,40-55,45-60,50-65,30-60",
                    help="comma list of lo-hi ply sub-bands to sweep")
    ap.add_argument("--max-pos", type=int, default=9000)
    ap.add_argument("--policy-max", type=int, default=1500, help="cap rows for the policy-prior arm")
    ap.add_argument("--stone-band", type=int, default=4, help="stone-count stratum width")
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--out", default="investigation/globalconc_2026-06-08/globalconc_probe.json")
    args = ap.parse_args()

    name = normalize_encoding_name(args.encoding)
    spec = lookup(name)
    kept = list(spec.kept_plane_indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bands = [(int(a), int(b)) for a, b in (s.split("-") for s in args.bands.split(","))]

    print(f"[cfg] encoding={name} trunk={spec.trunk_size} kept={kept} "
          f"policy_logit_count={spec.policy_logit_count} device={device.type}", flush=True)
    print(f"[pool] building BUILD-UP pool ply[{args.ply_lo},{args.ply_hi}) "
          f"steps={args.pool_steps} (excl. immediate-forced-win turn-starts) ...", flush=True)
    t0 = time.time()
    rows = build_pool(args.replays, name, spec, set(args.pool_steps),
                      args.ply_lo, args.ply_hi, args.max_pos, args.seed)
    if not rows:
        print("NO build-up positions"); return 1
    print(f"[pool] n={len(rows)}  by_src={dict(sorted(Counter(r['src_step'] for r in rows).items()))}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    sc = np.array([r["stones"] for r in rows])
    nc = np.array([r["ncomp"] for r in rows], dtype=np.float64)
    print(f"[pool] stones mean={sc.mean():.1f} [{sc.min()}..{sc.max()}]  "
          f"ncomp mean={nc.mean():.2f}  corr(ncomp,stones)={np.corrcoef(nc, sc)[0,1]:.3f} "
          f"(the D1 confound to defeat by stone-matching)", flush=True)

    ncomp = np.array([r["ncomp"] for r in rows], dtype=np.float64)
    ncheb2 = np.array([r["ncomp_cheb2"] for r in rows], dtype=np.float64)
    lfrac = np.array([r["largest_frac"] for r in rows], dtype=np.float64)
    stones = np.array([r["stones"] for r in rows])
    outc = np.array([r["outcome"] for r in rows])
    plies = np.array([r["ply"] for r in rows])
    gids = np.array([r["gid"] for r in rows])

    X = np.stack([r["wire"] for r in rows]).astype(np.float32)

    ckpts = discover_ckpts(args.ckpt_dirs)
    steps = sorted(ckpts)
    print(f"[ckpts] {steps} (n={len(steps)})", flush=True)
    rng = np.random.default_rng(args.seed)

    def vforward(model):
        v = np.empty(len(X), dtype=np.float64)
        with torch.no_grad():
            for b0 in range(0, len(X), 512):
                xb = torch.from_numpy(X[b0:b0 + 512]).to(device)
                v[b0:b0 + len(xb)] = model(xb)[1].float().cpu().numpy().reshape(-1)
        return v

    # policy-prior subsample (fixed)
    pol_idx = list(range(len(rows)))
    random.Random(args.seed + 1).shuffle(pol_idx)
    pol_idx = sorted(pol_idx[:args.policy_max])
    pol_rows = [rows[i] for i in pol_idx]
    print(f"[policy] arm on {len(pol_rows)} subsampled rows", flush=True)

    out = {"encoding": name, "pool_n": len(rows), "pool_steps": args.pool_steps,
           "wide_band": [args.ply_lo, args.ply_hi], "bands": [list(b) for b in bands],
           "stone_band_width": args.stone_band,
           "by_src": dict(sorted(Counter(r["src_step"] for r in rows).items())),
           "corr_ncomp_stones": float(np.corrcoef(ncomp, stones)[0, 1]),
           "verdict_rule": "ABSENT if AUC<=0.60 (esp falling); PRESENT if >0.60 flat/rising; "
                           "MIXED if value vs policy-prior split. 0.60 bar (== D1).",
           "value": {b2s(b): [] for b in bands}, "policy": []}

    # ---------------- VALUE ARM (all checkpoints, all bands) ----------------
    print(f"\n{'='*120}\nMEASURE 1 — VALUE AUC_globalconc (stratified stones+outcome) per band per checkpoint\n{'='*120}", flush=True)
    band_strata = {}
    band_valid = {}
    band_flag_ncomp = {}
    band_naive_flag = {}
    band_flag_lfrac = {}
    band_flag_cheb2 = {}
    for (lo, hi) in bands:
        m = (plies >= lo) & (plies < hi)
        valid = m & np.isin(outc, [0, 1])     # value AUC on won/lost only (drop draws)
        skey = [None] * len(rows)
        for k in range(len(rows)):
            if valid[k]:
                skey[k] = (int(stones[k]) // args.stone_band, int(outc[k]))
        band_strata[(lo, hi)] = skey
        band_valid[(lo, hi)] = valid
        band_flag_ncomp[(lo, hi)] = make_concflag(ncomp, skey, valid)
        band_flag_lfrac[(lo, hi)] = make_concflag(-lfrac, skey, valid)   # higher frac = concentrated -> flip sign for "low=conc"
        band_flag_cheb2[(lo, hi)] = make_concflag(ncheb2, skey, valid)
        band_naive_flag[(lo, hi)] = make_concflag_global(ncomp, m, higher_is_concentrated=False)

    for st in steps:
        model, _sp, _lab = load_inference_model(ckpts[st], {}, device=device)
        model = model.float().eval()
        v = vforward(model)
        for (lo, hi) in bands:
            skey = band_strata[(lo, hi)]; valid = band_valid[(lo, hi)]
            mband = (plies >= lo) & (plies < hi)
            flag = band_flag_ncomp[(lo, hi)]
            auc_strat, den = stratified_auc(v, flag, skey, valid)
            auc_lfrac, _ = stratified_auc(v, band_flag_lfrac[(lo, hi)], skey, valid)
            auc_cheb2, _ = stratified_auc(v, band_flag_cheb2[(lo, hi)], skey, valid)
            # outcome-split transparency: is value blindness a won/lost sign-flip that cancels?
            valid_won = valid & (outc == 1)
            valid_lost = valid & (outc == 0)
            auc_won, _ = stratified_auc(v, flag, skey, valid_won)
            auc_lost, _ = stratified_auc(v, flag, skey, valid_lost)
            # naive (no matching) — single stratum
            naive_flag = band_naive_flag[(lo, hi)]
            naive_key = ["all"] * len(rows)
            auc_naive, _ = stratified_auc(v, naive_flag, naive_key, mband)
            # bootstrap over games within the band-valid rows
            ci = _boot_value(v, flag, skey, valid, gids, args.nboot, rng)
            n_band = int(mband.sum()); n_used = int(valid.sum())
            out["value"][b2s((lo, hi))].append({
                "step": st, "n_band": n_band, "n_used": n_used, "n_pairs": int(den),
                "auc_strat_ncomp": auc_strat, "auc_strat_ci": list(ci),
                "auc_won_only": auc_won, "auc_lost_only": auc_lost,
                "auc_strat_largest_frac": auc_lfrac, "auc_strat_cheb2_fronts": auc_cheb2,
                "auc_naive_ncomp": auc_naive,
            })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  [value] ckpt {st} done", flush=True)

    # print value tables per band
    for (lo, hi) in bands:
        recs = out["value"][b2s((lo, hi))]
        print(f"\n-- band ply[{lo},{hi})  (n_used avg={np.mean([r['n_used'] for r in recs]):.0f}) --")
        print(f"{'step':>7} | {'AUC_strat(ncomp)':>18} [95%CI] | {'AUC(lfrac)':>10} | "
              f"{'AUC(cheb2)':>10} | {'AUC_naive':>9} | {'npairs':>7}")
        for r in recs:
            ci = r["auc_strat_ci"]
            print(f"{r['step']:>7} | {r['auc_strat_ncomp']:>18.3f} [{ci[0]:.3f},{ci[1]:.3f}] | "
                  f"{r['auc_strat_largest_frac']:>10.3f} | {r['auc_strat_cheb2_fronts']:>10.3f} | "
                  f"{r['auc_naive_ncomp']:>9.3f} | {r['n_pairs']:>7}")
        sv = np.array([r["step"] for r in recs], dtype=np.float64)
        av = np.array([r["auc_strat_ncomp"] for r in recs])
        slope = float(np.polyfit(sv, av, 1)[0]) * 1000.0
        print(f"   arc: AUC_strat {recs[0]['auc_strat_ncomp']:.3f} -> {recs[-1]['auc_strat_ncomp']:.3f}  "
              f"mean={av.mean():.3f} slope/1k={slope:+.4f}  "
              f"naive mean={np.mean([r['auc_naive_ncomp'] for r in recs]):.3f}")
        out.setdefault("value_arc", {})[b2s((lo, hi))] = {
            "auc_from": recs[0]["auc_strat_ncomp"], "auc_to": recs[-1]["auc_strat_ncomp"],
            "auc_mean": float(av.mean()), "slope_per_1k": slope,
            "auc_naive_mean": float(np.mean([r["auc_naive_ncomp"] for r in recs]))}

    # ---------------- POLICY-PRIOR ARM (all checkpoints, wide-band subsample) ----------------
    print(f"\n{'='*120}\nMEASURE 2 — POLICY-PRIOR: AUC_conc P(prior(consolidate-main)>prior(spread-peripheral)) "
          f"+ fragmentation DRIFT + trivial floor\n{'='*120}", flush=True)
    print(f"[policy] precomputing board-only masks (rebuild once) for {len(pol_rows)} rows ...", flush=True)
    tpc = time.time()
    pcache = precompute_policy_cache(name, kept, int(spec.n_planes), pol_rows)
    print(f"[policy] cache n={len(pcache)} ({time.time()-tpc:.0f}s)", flush=True)

    def nanmean(a):
        a = np.asarray([x for x in a if x == x], dtype=np.float64)
        return float(a.mean()) if len(a) else float("nan")

    print(f"{'step':>7} | {'AUC_main_v_periph':>17} [95%CI] | {'AUC_conc(vs all)':>16} | "
          f"{'adj_floor':>9} | {'frag_drift':>10} | {'lift':>6} | {'n':>5}")
    print("  (AUC_main_v_periph = THE over-spread-relevant signal, adjacency controlled; "
          "AUC_conc ~= adj_floor = trivial)")
    for st in steps:
        model, _sp, _lab = load_inference_model(ckpts[st], {}, device=device)
        model = model.float().eval()
        ac, amp, dr, aa, lf, pgids = score_policy(model, device, pcache)
        m_ac, m_amp, m_dr, m_aa, m_lf = nanmean(ac), nanmean(amp), nanmean(dr), nanmean(aa), nanmean(lf)
        ci_amp = boot_over_games(pgids, lambda sel: nanmean([amp[i] for i in sel]), args.nboot, rng)
        ci_dr = boot_over_games(pgids, lambda sel: nanmean([dr[i] for i in sel]), args.nboot, rng)
        out["policy"].append({"step": st, "auc_main_vs_periph": m_amp, "auc_main_periph_ci": list(ci_amp),
                              "auc_conc": m_ac, "frag_drift": m_dr, "drift_ci": list(ci_dr),
                              "auc_adj_floor": m_aa, "lift": m_lf,
                              "n": int(sum(1 for x in amp if x == x))})
        print(f"{st:>7} | {m_amp:>17.3f} [{ci_amp[0]:.3f},{ci_amp[1]:.3f}] | {m_ac:>16.3f} | "
              f"{m_aa:>9.3f} | {m_dr:>+10.4f} | {m_lf:>6.2f} | {out['policy'][-1]['n']:>5}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pol = out["policy"]
    sv = np.array([r["step"] for r in pol], dtype=np.float64)
    pamp = np.array([r["auc_main_vs_periph"] for r in pol])
    pa = np.array([r["auc_conc"] for r in pol])
    pd_ = np.array([r["frag_drift"] for r in pol])
    ampslope = float(np.polyfit(sv, pamp, 1)[0]) * 1000.0
    out["policy_arc"] = {
        "auc_main_periph_from": pol[0]["auc_main_vs_periph"], "auc_main_periph_to": pol[-1]["auc_main_vs_periph"],
        "auc_main_periph_mean": float(pamp.mean()), "auc_main_periph_slope_per_1k": ampslope,
        "auc_conc_floor_mean": float(pa.mean()),
        "frag_drift_mean": float(pd_.mean()),
        "auc_adj_floor_mean": float(np.mean([r["auc_adj_floor"] for r in pol])),
        "lift_mean": float(np.mean([r["lift"] for r in pol]))}
    print(f"\nPOLICY arc (THE signal = main-vs-peripheral, adjacency controlled): "
          f"{pol[0]['auc_main_vs_periph']:.3f} -> {pol[-1]['auc_main_vs_periph']:.3f} "
          f"mean={pamp.mean():.3f} slope/1k={ampslope:+.4f}")
    print(f"  (trivial floors: auc_conc-vs-all {pa.mean():.3f} ~= adj_floor "
          f"{out['policy_arc']['auc_adj_floor_mean']:.3f}; frag_drift {pd_.mean():+.4f})")

    # ---------------- MECHANICAL VERDICT (note makes the final call) ----------------
    # value verdict on the PRIMARY mid-game band [30,60) if present else first band
    prim = (30, 60) if (30, 60) in bands else bands[0]
    varc = out["value_arc"][b2s(prim)]
    v_present = varc["auc_mean"] > 0.60
    p_present = out["policy_arc"]["auc_main_periph_mean"] > 0.60   # THE signal, adjacency controlled
    if v_present and p_present:
        verdict = "GLOBAL-SIGNAL-PRESENT (value AND policy main-vs-periph > 0.60)"
    elif (not v_present) and (not p_present):
        verdict = ("GLOBAL-SIGNAL-ABSENT / near-absent (value AND policy main-vs-periph <= 0.60) "
                   "— no clean global concentration node on either head")
    else:
        verdict = (f"MIXED (value {'sees' if v_present else 'blind/absent'} / "
                   f"policy main-vs-periph {'sees' if p_present else 'borderline-absent'})")
    out["mechanical_verdict"] = verdict
    out["primary_band"] = list(prim)
    print(f"\n{'='*120}")
    print(f"MECHANICAL VERDICT (primary band ply{list(prim)}): {verdict}")
    print(f"  value  AUC_strat mean={varc['auc_mean']:.3f} (naive={varc['auc_naive_mean']:.3f}) slope/1k={varc['slope_per_1k']:+.4f}")
    print(f"  policy main-vs-periph mean={out['policy_arc']['auc_main_periph_mean']:.3f} "
          f"slope/1k={out['policy_arc']['auc_main_periph_slope_per_1k']:+.4f} "
          f"(trivial floors: conc-vs-all {out['policy_arc']['auc_conc_floor_mean']:.3f} ~= adj {out['policy_arc']['auc_adj_floor_mean']:.3f})")
    print(f"{'='*120}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[out] {args.out}")
    return 0


def _boot_value(v, flag, skey, valid, gids, nboot, rng):
    """Bootstrap the stratified value AUC by resampling games among the band-valid rows."""
    idxs = np.where(valid)[0]
    if len(idxs) == 0:
        return (float("nan"), float("nan"))
    by_g = defaultdict(list)
    for k in idxs:
        by_g[int(gids[k])].append(k)
    glist = list(by_g)
    out = []
    for _ in range(nboot):
        pick = rng.integers(0, len(glist), len(glist))
        sel = []
        for gi in pick:
            sel.extend(by_g[glist[gi]])
        sub_v = v[sel]; sub_flag = flag[sel]
        sub_key = [skey[i] for i in sel]; sub_valid = np.ones(len(sel), dtype=bool)
        a, _d = stratified_auc(sub_v, sub_flag, sub_key, sub_valid)
        if a == a:
            out.append(a)
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def b2s(b):
    return f"{b[0]}-{b[1]}"


if __name__ == "__main__":
    sys.exit(main())
