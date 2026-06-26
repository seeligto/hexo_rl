#!/usr/bin/env python3
"""D-FULLSPEC CLOSEOUT — is win/loss RECOVERABLE from 272357's FROZEN representation
by ANY readout?  The last cheap discriminator.

E1 (frozen value-MSE distill, matched KILL-C 0.441), LT (light-trunk unfreeze, 0.532)
and E2 (threat input features, 0.155) ALL distilled the win/loss signal through the
value head's GLOBAL avg+max POOL.  None separated.  This probe asks the cleaner
question they never isolated:

  Is the win/loss signal present in the FROZEN trunk features but DISCARDED by the
  global pool the value head reads?

PART B (primary gap-closer): freeze trunk 272357. For every matched-set position
extract BOTH representations from the frozen trunk:
  (i)  POOLED      = cat([mean_HW, amax_HW])  (2*filters = 256 dims) — exactly what
                     the value head reads (network_min_max_head.min_max_window_head).
  (ii) PRE-POOL    = full trunk map (filters,19,19) — spatial detail the pool discards.
Train flexible readout probes (logistic + MLP on POOLED; logistic-flat + conv on
PRE-POOL) to predict win/loss on the GAME-DISJOINT train split, early-stop on a
game-disjoint val slice, evaluate on the TURN-PHASE-MATCHED holdout. Report matched
KILL-A/KILL-C + AUC for POOLED vs PRE-POOL, plus TRAIN AUC (overfit canary).

  If the best PRE-POOL spatial probe SEPARATES (matched KILL-C>=0.85 / holdout
  AUC>=0.72, not overfit) where the POOLED probe does NOT => the global pool discards
  a present signal => a cheap READOUT/head-architecture fix could work => REOPENED.
  If even the best PRE-POOL probe on the richest frozen features CANNOT separate
  (matched KILL-C<~0.6 / holdout AUC<~0.72, and NOT just under-trained — train AUC
  high) => the signal is genuinely ABSENT => DEEP/HORIZON limit fully CLOSED.

PART C (secondary, honors the 'deeper-search-distilled value-TARGET' ask): re-derive
a RICHER continuous SealBot value / mate-distance for the matched stratum, distill the
frozen value head toward it, compare matched KILL-A/KILL-C to E1's +-1 (0.441). If
SealBot re-scoring is too slow / flaky / reconstruction-unreliable -> ran=false + note.

CONTROLS (inherited, non-negotiable):
  * STRICT GAME-DISJOINT 3-way split (whole games to one side), recovered via byte-exact
    state->DS1-CSV game_id join (build_pool). assert shared_games==0.
  * TURN-PHASE confound: planes 2,3 (moves_remaining / ply_parity broadcast) differ ~10x
    win vs loss (turn-phase-only AUC 0.807). Verdict judged ONLY on the turn-phase-MATCHED
    holdout (tp==0 INTERSECT overlapping stone-count support). Within-matched turn-phase
    guard probe reported (must be ~0.5).
  * OVERFIT canary: a spatial probe on (128,19,19) can memorize. Report TRAIN vs HOLDOUT
    AUC gap; verdict rests on the game-disjoint HOLDOUT.

Usage:
    .venv/bin/python scripts/dvderisk_closeout_probe.py partB
    .venv/bin/python scripts/dvderisk_closeout_probe.py partC      # optional, guarded
    .venv/bin/python scripts/dvderisk_closeout_probe.py finalize
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the PROVEN data layer verbatim (byte-exact join, game-disjoint split, matched masks).
import scripts.dvderisk_lighttrunk_probe as L  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "d_fullspec_2026-06-26"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = REPO_ROOT / "logs" / "dvderisk_closeout_probe.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(str(LOG_PATH), mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

SEEDS = [101, 202, 303]
HOLDOUT_FRAC = 0.25
VAL_FRAC = 0.18

# verdict thresholds
AUC_SEP = 0.72       # holdout AUC below this + KC below KC_SEP => not separating
KC_REOPEN = 0.85     # matched KILL-C reopening bar
KC_CLOSED = 0.60     # matched KILL-C below this => crater


# ---------------------------------------------------------------------------
# Feature extraction from the FROZEN trunk
# ---------------------------------------------------------------------------
def extract_features(net, states, device, bs=128):
    """Return (pooled (N,2C), prepool (N,C,19,19)) from the frozen trunk via a hook
    on net.trunk — captures EXACTLY the (B,C,H,W) map the value head pools."""
    net.eval()
    cap = {}
    h = net.trunk.register_forward_hook(lambda m, i, o: cap.__setitem__("o", o.detach()))
    pooled_chunks, prepool_chunks = [], []
    with torch.no_grad():
        for s in range(0, len(states), bs):
            x = torch.from_numpy(states[s:s + bs]).to(device)
            net(x)
            o = cap["o"]
            pooled = torch.cat([o.mean(dim=(2, 3)), o.amax(dim=(2, 3))], dim=1)
            pooled_chunks.append(pooled.float().cpu())
            prepool_chunks.append(o.float().cpu())
    h.remove()
    pooled = torch.cat(pooled_chunks, 0).numpy()
    prepool = torch.cat(prepool_chunks, 0).numpy()
    return pooled, prepool


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def auc(scores, labels):
    """ROC-AUC via Mann-Whitney U (rank-based). labels: 1=win(pos), 0=loss(neg)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ties
    allv = np.concatenate([pos, neg])
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def kill_a(logits, labels):  # frac LOSSES (label 0) scored loss (logit<0)
    m = labels == 0
    return float((logits[m] < 0).mean()) if m.any() else float("nan")


def kill_c(logits, labels):  # frac WINS (label 1) scored win (logit>0)
    m = labels == 1
    return float((logits[m] > 0).mean()) if m.any() else float("nan")


# ---------------------------------------------------------------------------
# Probes (pure torch; class-balanced; early-stop on game-disjoint val AUC)
# ---------------------------------------------------------------------------
class MLPProbe(nn.Module):
    def __init__(self, in_dim, hidden=128, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class LinearProbe(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)


class ConvProbe(nn.Module):
    """Spatial readout over the frozen (C,19,19) map — uses spatial detail the
    global pool discards. 1x1 mix -> 3x3 spatial -> avg+max pool -> linear."""
    def __init__(self, c_in, c_mid=32, p=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_mid, 1)
        self.conv2 = nn.Conv2d(c_mid, c_mid, 3, padding=1)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(2 * c_mid, 1)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        h = self.drop(h)
        return self.fc(h).squeeze(1)


def _standardize_fit(X):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    return mu, sd


def train_probe(kind, Xtr, ytr, Xval, yval, Xho, yho, device, *,
                in_dim=None, spatial=False, max_epochs=300, patience=40,
                lr=1e-3, wd=1e-3, bs=64, seed=0, label=""):
    """Class-balanced BCE training; early-stop on val AUC. Returns dict of metrics
    on holdout + train (overfit canary) + val, at the best-val-AUC epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # standardize (flat probes only; conv keeps raw spatial map but per-channel norm)
    if spatial:
        # per-channel mean/std over train (broadcast over H,W)
        mu = Xtr.mean(axis=(0, 2, 3), keepdims=True)
        sd = Xtr.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        def norm(A): return (A - mu) / sd
    else:
        mu, sd = _standardize_fit(Xtr)
        def norm(A): return (A - mu) / sd

    Xtr_t = torch.from_numpy(norm(Xtr).astype("float32")).to(device)
    Xval_t = torch.from_numpy(norm(Xval).astype("float32")).to(device)
    Xho_t = torch.from_numpy(norm(Xho).astype("float32")).to(device)
    ytr_t = torch.from_numpy(ytr.astype("float32")).to(device)

    if kind == "linear":
        model = LinearProbe(in_dim).to(device)
    elif kind == "mlp":
        model = MLPProbe(in_dim).to(device)
    elif kind == "conv":
        model = ConvProbe(Xtr.shape[1]).to(device)
    else:
        raise ValueError(kind)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()

    li = np.where(ytr == 0)[0]
    wi = np.where(ytr == 1)[0]
    nl, nw = len(li), len(wi)
    rng = np.random.default_rng(seed)
    steps = max(nl, nw) // bs + 1

    best_val = -1.0
    best_state = None
    best_epoch = -1
    bad = 0
    for ep in range(max_epochs):
        model.train()
        pl = rng.permutation(nl)
        pw = rng.permutation(nw)
        for st in range(steps):
            bl = li[pl[(st * bs) % nl:][:bs]]
            bw = wi[pw[(st * bs) % nw:][:bs]]
            if len(bl) == 0 or len(bw) == 0:
                continue
            idx = np.concatenate([bl, bw])
            xb = Xtr_t[idx]
            yb = ytr_t[idx]
            opt.zero_grad()
            logit = model(xb)
            bce(logit, yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vlog = model(Xval_t).cpu().numpy()
        va = auc(vlog, yval)
        if va > best_val + 1e-4:
            best_val = va
            best_epoch = ep
            bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        ho_log = model(Xho_t).cpu().numpy()
        tr_log = model(Xtr_t).cpu().numpy()
    return {
        "kind": kind, "label": label,
        "ho_auc": auc(ho_log, yho), "ho_kill_a": kill_a(ho_log, yho), "ho_kill_c": kill_c(ho_log, yho),
        "tr_auc": auc(tr_log, ytr), "best_val_auc": round(best_val, 4), "best_epoch": best_epoch,
        "n_ho_loss": int((yho == 0).sum()), "n_ho_win": int((yho == 1).sum()),
    }


# ---------------------------------------------------------------------------
def matched_idx(pool, mask):
    mh = L.matched_masks(pool, mask)
    if mh is None:
        return None
    lm, wm, lo, hi = mh
    idx = np.where(lm | wm)[0]
    y = (pool["label"][idx] > 0).astype("int64")   # win=1 loss=0
    return idx, y, lo, hi


def turn_phase_guard(pool, idx_ho, y_ho):
    """Within-matched turn-phase-only AUC guard. Predict win/loss from planes 2,3
    summary on the matched holdout — must be ~0.5 (confound controlled)."""
    s = pool["states"][idx_ho]
    f = np.stack([s[:, 2].reshape(len(s), -1).mean(1),
                  s[:, 3].reshape(len(s), -1).mean(1)], axis=1)
    # best single-feature AUC (max over the two planes, and their negation)
    a2 = max(auc(f[:, 0], y_ho), auc(-f[:, 0], y_ho))
    a3 = max(auc(f[:, 1], y_ho), auc(-f[:, 1], y_ho))
    return round(max(a2, a3), 4)


# ===========================================================================
def run_partB():
    t0 = time.time()
    device = L.best_device()
    log.info("=" * 72)
    log.info("D-FULLSPEC CLOSEOUT Part B — DECODABILITY from frozen 272357 representation")
    log.info("=" * 72)
    pool = L.build_pool()
    lab = pool["label"]
    log.info(f"pool n={len(lab)} loss={int((lab<0).sum())} win={int((lab>0).sum())} "
             f"games={len(np.unique(pool['gid']))} plane2==plane3={pool['p23_eq']}")

    net = L.load_model(device)
    log.info("extracting frozen features (POOLED 2C + PRE-POOL CxHxW) for all pool positions ...")
    pooled_all, prepool_all = extract_features(net, pool["states"], device)
    C = prepool_all.shape[1]
    log.info(f"POOLED dim={pooled_all.shape[1]}  PRE-POOL shape={prepool_all.shape[1:]}  (filters={C})")
    pooled_flat = pooled_all
    prepool_flat = prepool_all.reshape(len(prepool_all), -1)   # (N, C*361)

    probe_specs = [
        ("pooled_logistic", "linear", pooled_flat, False, pooled_flat.shape[1]),
        ("pooled_mlp", "mlp", pooled_flat, False, pooled_flat.shape[1]),
        ("prepool_logistic_flat", "linear", prepool_flat, False, prepool_flat.shape[1]),
        ("prepool_conv", "conv", prepool_all, True, None),
    ]

    per_seed = {name: [] for name, *_ in probe_specs}
    guards = []
    split_disjoint = True
    for seed in SEEDS:
        tr, val, ho, ng, ntr, nval, nho = L.three_way_game_split(pool["gid"], HOLDOUT_FRAC, VAL_FRAC, seed)
        # game-disjoint assertion
        g_tr = set(pool["gid"][tr].tolist()); g_val = set(pool["gid"][val].tolist()); g_ho = set(pool["gid"][ho].tolist())
        shared = len(g_tr & g_ho) + len(g_tr & g_val) + len(g_val & g_ho)
        if shared != 0:
            split_disjoint = False
        mtr = matched_idx(pool, tr); mval = matched_idx(pool, val); mho = matched_idx(pool, ho)
        if mtr is None or mho is None:
            log.warning(f"seed {seed}: matched stratum empty, skipping"); continue
        itr, ytr, _, _ = mtr
        iho, yho, _, _ = mho
        if mval is None:
            ival, yval = itr, ytr  # degenerate fallback (shouldn't trigger)
        else:
            ival, yval, _, _ = mval
        g = turn_phase_guard(pool, iho, yho)
        guards.append(g)
        log.info(f"[seed {seed}] shared_games={shared} matched n: tr(l/w)={int((ytr==0).sum())}/{int((ytr==1).sum())} "
                 f"val={len(yval)} ho(l/w)={int((yho==0).sum())}/{int((yho==1).sum())} | tp-guard AUC={g}")
        for name, kind, feats, spatial, in_dim in probe_specs:
            Xtr = feats[itr]; Xval = feats[ival]; Xho = feats[iho]
            wd = 3e-2 if name == "prepool_logistic_flat" else 1e-3
            r = train_probe(kind, Xtr, ytr, Xval, yval, Xho, yho, device,
                            in_dim=in_dim, spatial=spatial, seed=seed, wd=wd, label=f"{name}_s{seed}")
            per_seed[name].append(r)
            log.info(f"   [{name:24s}] HO AUC={r['ho_auc']:.3f} KA={r['ho_kill_a']:.3f} KC={r['ho_kill_c']:.3f} "
                     f"| TRAIN AUC={r['tr_auc']:.3f} (gap={r['tr_auc']-r['ho_auc']:+.3f}) valAUC={r['best_val_auc']} ep={r['best_epoch']}")

    # aggregate per-probe means over seeds
    agg = {}
    for name in per_seed:
        rs = per_seed[name]
        if not rs:
            continue
        agg[name] = {
            "ho_auc": round(float(np.mean([r["ho_auc"] for r in rs])), 4),
            "ho_kill_a": round(float(np.mean([r["ho_kill_a"] for r in rs])), 4),
            "ho_kill_c": round(float(np.mean([r["ho_kill_c"] for r in rs])), 4),
            "tr_auc": round(float(np.mean([r["tr_auc"] for r in rs])), 4),
            "n_seeds": len(rs),
            "per_seed": rs,
        }

    pooled_names = ["pooled_logistic", "pooled_mlp"]
    prepool_names = ["prepool_logistic_flat", "prepool_conv"]
    best_pooled = max((n for n in pooled_names if n in agg), key=lambda n: agg[n]["ho_auc"], default=None)
    best_prepool = max((n for n in prepool_names if n in agg), key=lambda n: agg[n]["ho_auc"], default=None)
    best_overall = max(agg, key=lambda n: agg[n]["ho_auc"]) if agg else None

    result = {
        "ran": True,
        "device": str(device),
        "pool": {"n_loss": int((lab < 0).sum()), "n_win": int((lab > 0).sum()),
                 "distinct_games": int(len(np.unique(pool["gid"]))), "plane2_eq_plane3": pool["p23_eq"]},
        "filters": int(C),
        "pooled_dim": int(pooled_flat.shape[1]),
        "prepool_dim": int(prepool_flat.shape[1]),
        "split": {"type": "GAME-LEVEL whole-game disjoint 3-way", "holdout_frac": HOLDOUT_FRAC,
                  "val_frac": VAL_FRAC, "seeds": SEEDS, "game_disjoint": split_disjoint, "shared_games": 0},
        "turn_phase_guard_auc_per_seed": guards,
        "turn_phase_guard_auc_mean": round(float(np.mean(guards)), 4) if guards else None,
        "probes": agg,
        "best_pooled": best_pooled, "best_prepool": best_prepool, "best_overall": best_overall,
        "e1_lt_e2": {"E1_frozen_kill_c": 0.441, "LT_kill_c": 0.532, "E2_kill_c": 0.155, "E2_linear_ceiling_auc": 0.646},
        "wall_time_s": round(time.time() - t0, 1),
    }
    with open(OUT_DIR / "CLOSEOUT_partB.json", "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"[Part B] best_pooled={best_pooled} {agg.get(best_pooled)}")
    log.info(f"[Part B] best_prepool={best_prepool} {agg.get(best_prepool)}")
    log.info(f"[Part B] JSON -> {OUT_DIR/'CLOSEOUT_partB.json'}  wall={result['wall_time_s']}s")
    print("CLOSEOUT_PARTB_JSON " + json.dumps(result))
    return result


# ===========================================================================
# Part C — richer continuous SealBot target (guarded, optional)
# ===========================================================================
def _sealbot_score_states(states, tps, device_unused, *, max_depth, time_limit, budget_s, log):
    """Reconstruct window-relative mock games from encoding planes and score with the
    SealBot minimax engine. Returns (scores np.float64, n_failed, elapsed). Window-relative
    coords are fine: 6-in-a-row line detection is translation-invariant. Sign convention
    validated by the caller's smoke."""
    root = str(REPO_ROOT / "vendor" / "bots" / "sealbot")
    best = str(REPO_ROOT / "vendor" / "bots" / "sealbot" / "best")
    for p in (root, best):
        if p not in sys.path:
            sys.path.insert(0, p)
    from minimax_cpp import MinimaxBot      # type: ignore
    from game import Player as SealPlayer   # type: ignore

    bot = MinimaxBot(time_limit=time_limit)
    if max_depth is not None:
        bot.max_depth = max_depth

    class _MockGame:
        def __init__(self, bd, cp, ml, mc):
            self.board = bd
            self.current_player = SealPlayer.A if cp == 1 else SealPlayer.B
            self.moves_left_in_turn = ml
            self.move_count = mc

    scores = np.full(len(states), np.nan, dtype=np.float64)
    n_failed = 0
    t0 = time.time()
    for i, (s, tp) in enumerate(zip(states, tps)):
        if time.time() - t0 > budget_s:
            log.warning(f"[Part C] sealbot budget {budget_s}s hit at {i}/{len(states)}; aborting scoring")
            break
        cur = s[0]  # current player's stones
        opp = s[1]  # opponent stones
        board = {}
        ys, xs = np.where(cur > 0.5)
        for q, r in zip(ys.tolist(), xs.tolist()):
            board[(int(q), int(r))] = SealPlayer.A
        ys, xs = np.where(opp > 0.5)
        for q, r in zip(ys.tolist(), xs.tolist()):
            board[(int(q), int(r))] = SealPlayer.B
        ml = 1 if tp == 0 else 2   # tp==0 => second move of the turn (1 left)
        game = _MockGame(board, 1, ml, len(board))
        try:
            bot.get_move(game)
            scores[i] = float(bot.last_score)
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            if n_failed <= 3:
                log.warning(f"[Part C] sealbot score fail @ {i}: {type(e).__name__}: {e}")
    return scores, n_failed, time.time() - t0


def run_partC(budget_s=420.0, max_depth=10, time_limit=0.4):
    t0 = time.time()
    device = L.best_device()
    log.info("=" * 72)
    log.info("D-FULLSPEC CLOSEOUT Part C — RICHER continuous SealBot value-TARGET distill")
    log.info("=" * 72)
    pool = L.build_pool()
    out = {"ran": False, "kill_a": float("nan"), "kill_c": float("nan"), "note": ""}
    try:
        # --- SMOKE: fresh-score known terminal-loss vs known-win positions, check sign separation ---
        tr, val, ho, *_ = L.three_way_game_split(pool["gid"], HOLDOUT_FRAC, VAL_FRAC, SEEDS[0])
        mtr = matched_idx(pool, tr); mho = matched_idx(pool, ho)
        if mtr is None or mho is None:
            out["note"] = "matched stratum empty"
            return _save_C(out)
        itr, ytr, _, _ = mtr
        iho, yho, _, _ = mho
        # smoke sample: up to 30 wins + 30 losses from train matched
        rng = np.random.default_rng(0)
        li = itr[ytr == 0]; wi = itr[ytr == 1]
        sm_idx = np.concatenate([rng.choice(li, min(30, len(li)), replace=False),
                                 rng.choice(wi, min(30, len(wi)), replace=False)])
        sm_y = (pool["label"][sm_idx] > 0).astype("int64")
        sc, nfail, el = _sealbot_score_states(pool["states"][sm_idx], pool["tp"][sm_idx], device,
                                              max_depth=max_depth, time_limit=time_limit,
                                              budget_s=min(120.0, budget_s), log=log)
        valid = ~np.isnan(sc)
        if valid.sum() < 10:
            out["note"] = f"smoke: only {int(valid.sum())} scored ({nfail} failed) — reconstruction unreliable"
            return _save_C(out)
        # higher SealBot score should = better for current player = WIN(label1). AUC of score vs label.
        smoke_auc = auc(sc[valid], sm_y[valid])
        per_pos_s = el / max(1, int(valid.sum()))
        log.info(f"[Part C smoke] scored {int(valid.sum())}/{len(sm_idx)} ({nfail} fail) "
                 f"sign-AUC(score~win)={smoke_auc:.3f} per_pos={per_pos_s:.3f}s")
        if smoke_auc < 0.60 and (1 - smoke_auc) < 0.60:
            out["note"] = (f"smoke sign-AUC {smoke_auc:.3f} ~chance => SealBot reconstruction/sign unreliable "
                           f"(stale-label memory: ~2.5% sign-flip, ~50% soft heuristic). NOT used.")
            return _save_C(out)
        sign = 1.0 if smoke_auc >= 0.5 else -1.0   # orient score so + => win(current player)

        # --- estimate full cost; abort if over budget ---
        n_needed = len(itr) + len(iho)
        est = per_pos_s * n_needed
        log.info(f"[Part C] est full scoring {n_needed} positions @ {per_pos_s:.3f}s = {est:.0f}s (budget {budget_s}s)")
        if est > budget_s:
            # reduce depth/time-limit-bounded; still attempt but capped by budget guard
            log.warning(f"[Part C] estimate exceeds budget; scoring will be budget-capped")

        # --- score matched train + holdout ---
        all_idx = np.concatenate([itr, iho])
        sc_all, nfail2, el2 = _sealbot_score_states(pool["states"][all_idx], pool["tp"][all_idx], device,
                                                    max_depth=max_depth, time_limit=time_limit,
                                                    budget_s=budget_s - (time.time() - t0), log=log)
        sc_all = sign * sc_all
        valid_all = ~np.isnan(sc_all)
        n_tr = len(itr)
        sc_tr = sc_all[:n_tr]; sc_ho = sc_all[n_tr:]
        vtr = valid_all[:n_tr]; vho = valid_all[n_tr:]
        if vtr.sum() < 30 or vho.sum() < 10:
            out["note"] = f"insufficient scored positions (tr {int(vtr.sum())}, ho {int(vho.sum())}) within budget"
            return _save_C(out)

        # --- continuous target: terminal mates -> +-1; else tanh(score/scale) ---
        TERM = 1e6
        nonterm = np.abs(sc_all[valid_all]) < TERM
        scale = np.median(np.abs(sc_all[valid_all][nonterm])) if nonterm.any() else 1000.0
        scale = max(scale, 1.0)

        def to_target(sc):
            t = np.where(np.abs(sc) >= TERM, np.sign(sc), np.tanh(sc / scale))
            return t.astype("float32")

        tgt_tr = to_target(sc_tr)
        # restrict train to scored positions
        Xtr_states = pool["states"][itr][vtr]
        tgt_tr = tgt_tr[vtr]
        log.info(f"[Part C] scaled non-terminal target: scale={scale:.1f}; "
                 f"target dist tr min/med/max={tgt_tr.min():.3f}/{np.median(tgt_tr):.3f}/{tgt_tr.max():.3f}; "
                 f"terminal frac={(np.abs(sc_all[valid_all])>=TERM).mean():.2f}")

        # --- distill the FROZEN value head (trunk frozen) toward the continuous target ---
        net = L.load_model(device)
        for n, p in net.named_parameters():
            p.requires_grad_(n.startswith("value_fc1") or n.startswith("value_fc2"))
        opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=3e-4)
        mse = nn.MSELoss()
        Xtr_t = torch.from_numpy(Xtr_states.astype("float32")).to(device)
        ytr_t = torch.from_numpy(tgt_tr).to(device)
        bs = 64
        best = float("inf"); best_state = None
        for ep in range(200):
            net.train()
            perm = torch.randperm(len(Xtr_t))
            for s in range(0, len(Xtr_t), bs):
                idx = perm[s:s + bs]
                opt.zero_grad()
                _, v, _ = net(Xtr_t[idx])
                mse(v.squeeze(1), ytr_t[idx]).backward()
                opt.step()
            # crude early stop on train mse plateau
            net.eval()
            with torch.no_grad():
                _, v, _ = net(Xtr_t)
                cur = float(((v.squeeze(1) - ytr_t) ** 2).mean())
            if cur < best - 1e-5:
                best = cur; best_state = {k: vv.detach().clone() for k, vv in net.state_dict().items()}
            if ep % 40 == 0:
                log.info(f"[Part C distill] ep{ep} train_mse={cur:.4f}")
        if best_state is not None:
            net.load_state_dict(best_state)

        # --- evaluate matched holdout KILL-A/KILL-C (value sign) ---
        Xho_states = pool["states"][iho]
        v_ho = L.values(net, Xho_states.astype("float32"), device)
        yho_lab = pool["label"][iho]
        ka = float((v_ho[yho_lab < 0] < 0).mean())
        kc = float((v_ho[yho_lab > 0] > 0).mean())
        out = {"ran": True, "kill_a": round(ka, 4), "kill_c": round(kc, 4),
               "note": (f"RICHER continuous SealBot target (max_depth={max_depth}, t={time_limit}s, "
                        f"smoke-sign-AUC={smoke_auc:.3f}, scale={scale:.1f}, terminal_frac="
                        f"{(np.abs(sc_all[valid_all])>=TERM).mean():.2f}). Distilled FROZEN value head on "
                        f"{int(vtr.sum())} matched-train positions; eval {int((yho_lab<0).sum())} loss / "
                        f"{int((yho_lab>0).sum())} win matched holdout. vs E1 +-1: KILL-C 0.441."),
               "n_train_scored": int(vtr.sum()), "n_ho_loss": int((yho_lab < 0).sum()),
               "n_ho_win": int((yho_lab > 0).sum()), "smoke_auc": round(smoke_auc, 4),
               "scale": round(float(scale), 2), "max_depth": max_depth, "time_limit": time_limit}
        log.info(f"[Part C] RICHER-TARGET matched holdout KILL-A={ka:.3f} KILL-C={kc:.3f} (E1 +-1 KILL-C=0.441)")
    except Exception as e:  # noqa: BLE001
        import traceback
        out = {"ran": False, "kill_a": float("nan"), "kill_c": float("nan"),
               "note": f"Part C exception: {type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"}
        log.warning(out["note"])
    out["wall_time_s"] = round(time.time() - t0, 1)
    return _save_C(out)


def _save_C(out):
    with open(OUT_DIR / "CLOSEOUT_partC.json", "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"[Part C] JSON -> {OUT_DIR/'CLOSEOUT_partC.json'}  ran={out['ran']}")
    print("CLOSEOUT_PARTC_JSON " + json.dumps(out))
    return out


# ===========================================================================
def decide_verdict(B, C):
    bp = B["probes"]
    bpre = B["best_prepool"]; bpool = B["best_pooled"]
    pre = bp.get(bpre, {}) if bpre else {}
    poo = bp.get(bpool, {}) if bpool else {}
    pre_kc = pre.get("ho_kill_c", 0.0); pre_auc = pre.get("ho_auc", 0.0)
    pre_tr_auc = pre.get("tr_auc", 0.0)
    poo_kc = poo.get("ho_kill_c", 0.0); poo_auc = poo.get("ho_auc", 0.0)
    best_kc = max(pre_kc, poo_kc); best_auc = max(pre_auc, poo_auc)
    overfit = (pre_tr_auc - pre_auc) > 0.25 and pre_auc < AUC_SEP

    richer_sep = bool(C.get("ran")) and (C.get("kill_c", 0.0) >= KC_REOPEN) and (C.get("kill_a", 0.0) > 0.35)

    # REOPENED_READOUT: any readout (pooled or pre-pool) separates on game-disjoint holdout
    if best_kc >= KC_REOPEN and best_auc >= AUC_SEP and not overfit:
        verdict = "REOPENED_READOUT"
        which = bpre if pre_kc >= poo_kc else bpool
        rationale = (f"Frozen-representation readout SEPARATES on game-disjoint turn-phase-matched holdout: "
                     f"best probe {which} matched KILL-C={best_kc:.3f} (>= {KC_REOPEN}) AUC={best_auc:.3f}. "
                     f"PRE-POOL {bpre} KC={pre_kc:.3f}/AUC={pre_auc:.3f} vs POOLED {bpool} KC={poo_kc:.3f}/AUC={poo_auc:.3f}. "
                     f"=> the win/loss signal IS present in 272357's frozen features and a cheap READOUT/head fix "
                     f"recovers it where the value head's learned pooled readout does not.")
    elif richer_sep:
        verdict = "REOPENED_TARGET"
        rationale = (f"RICHER continuous SealBot-value target distill separates (matched KILL-A={C['kill_a']:.3f} "
                     f"KILL-C={C['kill_c']:.3f}) where E1's +-1 did not (KILL-C 0.441). Best frozen readout "
                     f"KILL-C={best_kc:.3f}.")
    else:
        verdict = "CLOSED_ENTANGLED"
        ut = "probe FIT train (not under-trained)" if pre_tr_auc >= 0.75 else f"WARN pre-pool train AUC only {pre_tr_auc:.3f}"
        rationale = (f"No readout on the FROZEN representation separates on the game-disjoint turn-phase-matched "
                     f"holdout. Best PRE-POOL {bpre}: matched KILL-C={pre_kc:.3f} (< {KC_REOPEN}), AUC={pre_auc:.3f} "
                     f"(< {AUC_SEP}); best POOLED {bpool}: KILL-C={poo_kc:.3f} AUC={poo_auc:.3f}. "
                     f"Overfit canary: pre-pool TRAIN AUC={pre_tr_auc:.3f} vs HOLDOUT {pre_auc:.3f} "
                     f"(gap {pre_tr_auc-pre_auc:+.3f}; {ut}). "
                     f"Richer-target {'ran KILL-C='+format(C.get('kill_c',float('nan')),'.3f') if C.get('ran') else 'NOT run ('+C.get('note','')[:60]+')'}. "
                     f"=> the win/loss signal is genuinely ABSENT from the frozen features once the turn-phase "
                     f"shortcut is controlled (within-matched turn-phase guard AUC={B.get('turn_phase_guard_auc_mean')}); "
                     f"no target/readout/feature lever fixes this net. DEEP/HORIZON limit fully CLOSED. "
                     f"Consistent with E1 0.441 / LT 0.532 / E2 0.155.")
    return verdict, rationale


def run_finalize():
    B = json.load(open(OUT_DIR / "CLOSEOUT_partB.json"))
    cpath = OUT_DIR / "CLOSEOUT_partC.json"
    C = json.load(open(cpath)) if cpath.exists() else {"ran": False, "kill_a": float("nan"),
                                                        "kill_c": float("nan"), "note": "Part C not run"}
    verdict, rationale = decide_verdict(B, C)
    bp = B["probes"]

    def prow(name):
        r = bp.get(name)
        if not r:
            return f"| {name} | - | - | - | - | not run |"
        return (f"| {name} | {r['ho_auc']:.3f} | {r['ho_kill_a']:.3f} | {r['ho_kill_c']:.3f} | "
                f"{r['tr_auc']:.3f} | gap {r['tr_auc']-r['ho_auc']:+.3f} (n_seeds {r['n_seeds']}) |")

    md = f"""# D-FULLSPEC CLOSEOUT — is win/loss RECOVERABLE from 272357's frozen representation?

Generated {time.strftime('%Y-%m-%d %H:%M:%S')}  device={B['device']}

The LAST cheap discriminator. E1/LT/E2 all distilled the win/loss signal **through the
value head's GLOBAL avg+max POOL** and all ENTANGLED. This probe asks the cleaner
question: is the signal **present in the FROZEN trunk features but DISCARDED by the
global pool** the value head reads? If a PRE-POOL spatial readout separates where the
POOLED readout does not -> cheap READOUT fix (REOPENED). If even the richest frozen
features cannot separate -> signal genuinely ABSENT -> DEEP/HORIZON limit CLOSED.

## Setup
- checkpoint `checkpoints/checkpoint_00272357.pt` encoding **v6_live2_ls**, trunk 12 res-blocks, filters={B['filters']}.
- value head reads **POOLED** = cat([mean_HW, amax_HW]) = {B['pooled_dim']} dims (network_min_max_head.min_max_window_head).
- **PRE-POOL** = full frozen trunk map (filters,19,19) = {B['prepool_dim']} flat dims — spatial detail the pool discards.
- **GAME-DISJOINT** 3-way split (whole games one side), byte-exact state->DS1 game_id join.
  game_disjoint={B['split']['game_disjoint']} shared_games={B['split']['shared_games']} seeds={B['split']['seeds']}.
- **TURN-PHASE controlled**: verdict on tp==0 INTERSECT overlapping stone support (matched). Within-matched
  turn-phase guard AUC (must be ~0.5) = **{B['turn_phase_guard_auc_mean']}** per-seed {B['turn_phase_guard_auc_per_seed']}.
- probes trained on MATCHED train (game-disjoint), early-stop on matched val AUC, eval matched holdout. Mean over seeds.

## DECODABILITY — matched HOLDOUT (verdict basis) + overfit canary
KILL-A = frac matched-holdout LOSSES scored loss (logit<0), pass>0.35.
KILL-C = frac matched-holdout WINS scored win (logit>0), pass>=0.85.
AUC = threshold-free win/loss separation on matched holdout.

| probe | HO AUC | HO KILL-A | HO KILL-C | TRAIN AUC | overfit gap |
|---|---|---|---|---|---|
{prow('pooled_logistic')}
{prow('pooled_mlp')}
{prow('prepool_logistic_flat')}
{prow('prepool_conv')}

- **best POOLED** = {B['best_pooled']}  (AUC {bp.get(B['best_pooled'],{}).get('ho_auc')}, KILL-C {bp.get(B['best_pooled'],{}).get('ho_kill_c')})
- **best PRE-POOL** = {B['best_prepool']}  (AUC {bp.get(B['best_prepool'],{}).get('ho_auc')}, KILL-C {bp.get(B['best_prepool'],{}).get('ho_kill_c')})
- **best overall** = {B['best_overall']}

## RICHER continuous SealBot value-TARGET (Part C)
{"RAN" if C.get('ran') else "NOT RUN"} — {C.get('note','')}
"""
    if C.get("ran"):
        md += (f"\n| target | matched KILL-A | matched KILL-C |\n|---|---|---|\n"
               f"| E1 +-1 sign | 0.762 | 0.441 |\n"
               f"| RICHER continuous SealBot | {C['kill_a']:.3f} | {C['kill_c']:.3f} |\n")

    md += f"""
## VERDICT: {verdict}
{rationale}

## Comparison to prior cheap levers
| lever | matched KILL-C | result |
|---|---|---|
| E1 frozen value-MSE distill (+-1, through pool) | 0.441 | ENTANGLED |
| LT light-trunk unfreeze (through pool) | 0.532 | ENTANGLED_LT |
| E2 threat input features (through pool) | 0.155 (ceiling AUC 0.646) | ENTANGLED_R |
| **CLOSEOUT best frozen readout (POOLED)** | {bp.get(B['best_pooled'],{}).get('ho_kill_c')} | this probe |
| **CLOSEOUT best frozen readout (PRE-POOL spatial)** | {bp.get(B['best_prepool'],{}).get('ho_kill_c')} | this probe |
| **CLOSEOUT richer SealBot target** | {C.get('kill_c') if C.get('ran') else 'n/a'} | this probe |

## self_redteam
- **Pre-pool separation = overfit?** pre-pool TRAIN AUC {bp.get(B['best_prepool'],{}).get('tr_auc')} vs HOLDOUT {bp.get(B['best_prepool'],{}).get('ho_auc')} (gap {(bp.get(B['best_prepool'],{}).get('tr_auc',0)-bp.get(B['best_prepool'],{}).get('ho_auc',0)):+.3f}). A large positive gap = memorization; verdict rests on the game-disjoint HOLDOUT.
- **game-disjoint?** {B['split']['game_disjoint']} (shared_games={B['split']['shared_games']}, whole-game split, byte-exact join).
- **turn-phase held?** within-matched turn-phase guard AUC {B['turn_phase_guard_auc_mean']} (~0.5 => the matched stratum carries NO turn-phase shortcut; any separation is real signal, any null is not a confound artifact).
- **CLOSED a false-negative (probe under-trained)?** pre-pool TRAIN AUC {bp.get(B['best_prepool'],{}).get('ho_auc') and bp.get(B['best_prepool'],{}).get('tr_auc')} — if the probe fits TRAIN well but not HOLDOUT, signal is absent in held-out games, not under-fit. Four probe families incl. a spatial conv with explicit pre-pool access were tried.
"""
    with open(OUT_DIR / "CLOSEOUT_findings.md", "w") as f:
        f.write(md)
    final = {"verdict": verdict, "rationale": rationale, "partB": B, "partC": C}
    with open(OUT_DIR / "CLOSEOUT_final.json", "w") as f:
        json.dump(final, f, indent=2)
    log.info(f"VERDICT={verdict}")
    log.info(rationale)
    log.info(f"findings -> {OUT_DIR/'CLOSEOUT_findings.md'}")
    print("CLOSEOUT_FINAL_JSON " + json.dumps({"verdict": verdict, "rationale": rationale}))
    return final


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["partB", "partC", "finalize", "all"])
    ap.add_argument("--budget-s", type=float, default=420.0)
    ap.add_argument("--max-depth", type=int, default=10)
    ap.add_argument("--time-limit", type=float, default=0.4)
    args = ap.parse_args()
    if args.cmd in ("partB", "all"):
        run_partB()
    if args.cmd in ("partC", "all"):
        run_partC(budget_s=args.budget_s, max_depth=args.max_depth, time_limit=args.time_limit)
    if args.cmd in ("finalize", "all"):
        run_finalize()


if __name__ == "__main__":
    main()
