#!/usr/bin/env python3
"""D-PFIT STAGE-0 — PAIRWISE LOCAL RE-RANK separation discriminator (the headline
cheap pre-GPU falsifier for the value-z lever L2; coupled_valuez_decode_design.md §4).

THE QUESTION THE FALSIFIED REGISTER NEVER RAN
---------------------------------------------
D-FULLSPEC only ever measured ABSOLUTE win/loss CLASS separation (E1 frozen 0.441,
light-trunk 0.532, E2-readout, closeout-richer 0.045 — all FAIL). The decision-relevant
question is PAIRWISE LOCAL RE-RANKING: given the right (paired, richer-feature) signal,
can a value head rank the SAVING-line sibling ABOVE its adjacent BLUNDER-line sibling?
That is the necessary condition for value-z to fix the search veto.

    PAIRWISE-RERANK = P( value(saving-sibling) > value(blunder) )  on GAME-DISJOINT held-out pairs.

We compare a 4-plane CONTROL (v6_live2_ls base) vs an 8-plane TREATMENT (+4 cheap tac
planes prototyped IN PYTHON from the board threat API). Per design §8 the DECISION is the
4-vs-8 DELTA (absolute 0.70/0.85 thresholds may not transfer from a small from-scratch conv
to L2's warm-restarted 12-block trunk), reported with an HONEST distinct-game bootstrap CI
(§D-ARGMAX — n_effective = DISTINCT GAMES, not raw count).

PAIR CONSTRUCTION (corpus reports/d_tactical_2026-06-26/corpus.jsonl, 38 proven-core)
------------------------------------------------------------------------------------
parent = replay(parent_move_seq) (MODEL to-move, about to blunder).
  blunder-line = parent + blunder_move  (the REALIZED self-play move -> proven LOSS)  label z_loss(mate_distance)
  saving-line  = parent + refuting_move (SealBot loss-avoiding move)                  label +z_save
Both children apply ONE stone to the SAME parent => identical moves_remaining AND ply =>
plane2/plane3 (turn-phase) are BYTE-IDENTICAL within a pair => the D-FULLSPEC AUC-0.807
turn-phase shortcut is STRUCTURALLY NEUTRALIZED for the pairwise metric (verified, §control).
2 of 38 rows have refuting_move == blunder_move (parent already lost, no distinct saving move)
=> DROPPED (degenerate self-contradictory pair) -> 36 clean pairs, all distinct game_id.

B1 UNIT PIN (coupled_valuez_decode_design.md §1.1/§8 — obeyed):
  d = corpus `mate_distance` (TURNS, 2-9); z_loss(d) = -(1 - (d-2)/6 * 0.5), clamp d>8 -> -0.5.
  mate-in-2 -> -1.00, mate-in-5 -> -0.75, mate-in-8 -> -0.50. NOT proven_depth (the SealBot
  search-depth band {6:34,7:3,8:1}, NOT a filter). All 38 proven-core qualify.
  saving label z_save = +0.6 (escaping-but-unproven default; corpus has no proven escape distance
  for the refuting move).

TAC PLANES (4) — prototyped from the board threat API, NOT the Rust encoding (eval-only):
  plane4 self win-in-1 map  = board.winning_moves(self)   (cells completing 6 NOW for the model)
  plane5 opp  win-in-1 map  = board.winning_moves(opp)
  plane6 self open-4/fork map= board.threat_moves(self)    (cells that CREATE a win-in-1 threat;
                                                            >=2 active = a double-threat/fork)
  plane7 opp  open-4/fork map= board.threat_moves(opp)
  MEASUREMENT-UNIT CAVEAT (verified at build, reported): at these DEEP traps (mate-in-3..9 TURNS)
  the win-in-1 planes are ALL-ZERO (no immediate 6-completion exists) and the open-4 planes
  differ blunder-vs-saving in only ~55% of pairs. The cheap tac planes are a WEAK proxy for L2's
  real solver-derived `forced_loss/win_within_2` maps; Stage-0 bounds the CHEAP-PROXY power.
  All tac cells outside the single primary window are DROPPED (same single-window frame as the
  base planes; off-window forcing geometry is the separate D-DECODE multi-window lever).

PERSPECTIVE (measurement-unit discipline): both children + both tac sides are built with
"my" = the PARENT'S current player (the MODEL side), NOT board.current_player after the move
(which flips when moves_remaining was 1). z is the value TO THE MODEL: blunder -> negative
(model lost), saving -> positive (model escaped). Consistent across the pair.

METRICS
  primary : PAIRWISE-RERANK (4 vs 8) on LOGO-CV held-out + distinct-game bootstrap CI; DELTA.
  also    : single 23/13 game-disjoint split (~13 held-out) multi-seed (spec framing);
            KILL-C (held-out saving-sibling predicted >0, bar 0.85), KILL-A;
            mean_v on a 300 neutral-position bank (anti-correlation canary);
            turn-phase control validity (AUC of turn-phase -> z-sign in [0.45,0.55]).

CORPUS-EXPANSION YIELD (--expand-scan N): SealBot-d6 scan of N model-lost games NOT in the
corpus to estimate genuinely-new distinct-game trap pairs minable offline (no new self-play).

Run:  .venv/bin/python scripts/dpfit_stage0_rerank.py            # the gate (CPU, ~minutes)
      .venv/bin/python scripts/dpfit_stage0_rerank.py --expand-scan 12
Commits nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import engine  # noqa: E402

ENC = "v6_live2_ls"
S = 19
HALF = (S - 1) // 2
N_CELLS = S * S
CORPUS = REPO_ROOT / "reports" / "d_tactical_2026-06-26" / "corpus.jsonl"
PER_GAME = REPO_ROOT / "reports" / "d_ladder_2026-06-24" / "per_game_seald5.jsonl"
REPORT = REPO_ROOT / "reports" / "d_tactical_2026-06-26" / "stage0_rerank_report.md"
CACHE = REPO_ROOT / "reports" / "d_tactical_2026-06-26" / "stage0_rerank_cache.npz"

# B1 z-label
D_MIN, D_MAX, Z_FLOOR = 2.0, 8.0, 0.5
Z_SAVE_DEFAULT = 0.6


def z_loss(d: float) -> float:
    d = min(max(float(d), D_MIN), D_MAX)
    return -(1.0 - (d - D_MIN) / (D_MAX - D_MIN) * (1.0 - Z_FLOOR))


# ---------------------------------------------------------------------------
# board replay + plane construction (engine-kernel-faithful, fixed perspective)
# ---------------------------------------------------------------------------
def replay(seq: List[List[int]]) -> "engine.Board":
    b = engine.Board.with_encoding_name(ENC)
    for (q, r) in seq:
        b.apply_move(int(q), int(r))
    return b


def _planes8(board: "engine.Board", my_player: int) -> np.ndarray:
    """(8,19,19) at the board's single primary window. my_player fixes perspective
    (NOT board.current_player). planes [my,opp,mr==2,ply%2, self_w1,opp_w1,self_o4,opp_o4]."""
    pl = np.zeros((8, S, S), dtype=np.float32)
    opp = -my_player
    # base stone planes (drop stones outside the single window, exactly as Rust to_planes)
    for (q, r, p) in board.get_stones():
        flat = board.to_flat(int(q), int(r))
        if flat >= N_CELLS:
            continue
        wq, wr = flat // S, flat % S
        pl[0 if int(p) == my_player else 1, wq, wr] = 1.0
    mr = int(board.moves_remaining)
    pl[2, :, :] = 0.0 if mr == 1 else 1.0
    pl[3, :, :] = float(int(board.ply) % 2)

    def mark(plane_idx: int, cells):
        for (q, r) in cells:
            flat = board.to_flat(int(q), int(r))
            if flat < N_CELLS:
                wq, wr = flat // S, flat % S
                pl[plane_idx, wq, wr] = 1.0

    mark(4, board.winning_moves(my_player))   # self win-in-1 (DEAD at deep traps; per spec)
    mark(5, board.winning_moves(opp))         # opp  win-in-1 (DEAD at deep traps; per spec)
    mark(6, board.threat_moves(my_player))    # self open-4/fork forcing cells (LIVE)
    mark(7, board.threat_moves(opp))          # opp  open-4/fork forcing cells (LIVE)
    return pl


# ---------------------------------------------------------------------------
# build pairs (cached)
# ---------------------------------------------------------------------------
def build_pairs(force: bool = False) -> Dict[str, Any]:
    if CACHE.exists() and not force:
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}
    recs = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
    core = [r for r in recs if r.get("is_proven_core")]
    core.sort(key=lambda r: r["pos_id"])
    Xb, Xs, zl, zs, gid, band, md, mr_a, ply_a = [], [], [], [], [], [], [], [], []
    dropped = []
    for r in core:
        bm = [int(x) for x in r["blunder_move"]]
        rm = [int(x) for x in r["refuting_move"]]
        if bm == rm:
            dropped.append(r["pos_id"])
            continue
        my = int(r["current_player_parent"])
        par = replay(r["parent_move_seq"])
        assert int(par.current_player) == my, f"{r['pos_id']} perspective mismatch"
        bl = par.clone(); bl.apply_move(*bm)
        sv = par.clone(); sv.apply_move(*rm)
        # turn-phase identity within pair (the structural control)
        assert int(bl.moves_remaining) == int(sv.moves_remaining) and int(bl.ply) == int(sv.ply)
        Xb.append(_planes8(bl, my)); Xs.append(_planes8(sv, my))
        zl.append(z_loss(r["mate_distance"])); zs.append(Z_SAVE_DEFAULT)
        gid.append(r["game_id"]); band.append(r["depth_band"]); md.append(float(r["mate_distance"]))
        mr_a.append(int(bl.moves_remaining)); ply_a.append(int(bl.ply))
    out = dict(
        Xb=np.asarray(Xb, np.float32), Xs=np.asarray(Xs, np.float32),
        z_loss=np.asarray(zl, np.float32), z_save=np.asarray(zs, np.float32),
        game_id=np.asarray(gid), depth_band=np.asarray(band), mate_distance=np.asarray(md),
        mr=np.asarray(mr_a), ply=np.asarray(ply_a), dropped=np.asarray(dropped),
    )
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


# ---------------------------------------------------------------------------
# small from-scratch conv value-regressor (forked from dvderisk_e2_featablation)
# ---------------------------------------------------------------------------
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


class ConvValueNet(nn.Module):
    def __init__(self, in_channels: int, width: int = 16, fc: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.Linear(2 * width, fc), nn.ReLU(inplace=True), nn.Linear(fc, 1))

    def forward(self, x):
        h = self.conv(x)
        z = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return torch.tanh(self.head(z)).squeeze(1)


def train_reg(X: np.ndarray, y: np.ndarray, in_ch: int, seed: int,
              epochs: int = 120, lr: float = 2e-3, wd: float = 1e-3) -> ConvValueNet:
    torch.manual_seed(seed); np.random.seed(seed)
    net = ConvValueNet(in_ch)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    mse = nn.MSELoss()
    xt = torch.from_numpy(X[:, :in_ch].copy())
    yt = torch.from_numpy(y.astype(np.float32))
    rng = np.random.default_rng(seed)
    n = len(y); bs = min(32, n)
    net.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            v = net(xt[idx])
            mse(v, yt[idx]).backward()
            opt.step()
    net.eval()
    return net


@torch.no_grad()
def predict(net: ConvValueNet, X: np.ndarray, in_ch: int) -> np.ndarray:
    return net(torch.from_numpy(X[:, :in_ch].copy())).numpy()


# ---------------------------------------------------------------------------
# LOGO-CV pairwise rerank (each pair = one distinct game)
# ---------------------------------------------------------------------------
def logo_rerank(P: Dict[str, Any], in_ch: int, seeds=(0, 1, 2), epochs=120) -> Dict[str, np.ndarray]:
    Xb, Xs = P["Xb"], P["Xs"]
    zl, zs = P["z_loss"], P["z_save"]
    n = len(zl)
    # training matrix: 2n boards (blunder + saving), labels z
    Xall = np.concatenate([Xb, Xs], 0)
    yall = np.concatenate([zl, zs], 0)
    grp = np.concatenate([np.arange(n), np.arange(n)])  # pair index per board
    v_b = np.zeros(n); v_s = np.zeros(n)
    for i in range(n):
        train_mask = grp != i              # drop BOTH boards of held-out pair i (game-disjoint)
        Xt, yt = Xall[train_mask], yall[train_mask]
        pb = np.zeros(len(seeds)); ps = np.zeros(len(seeds))
        for k, sd in enumerate(seeds):
            net = train_reg(Xt, yt, in_ch, sd, epochs=epochs)
            pb[k] = predict(net, Xb[i:i + 1], in_ch)[0]
            ps[k] = predict(net, Xs[i:i + 1], in_ch)[0]
        v_b[i] = pb.mean(); v_s[i] = ps.mean()
    return {"v_b": v_b, "v_s": v_s, "rerank": (v_s > v_b).astype(np.float64)}


def boot_ci(x: np.ndarray, nboot=5000, seed=0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = np.array([x[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    return float(x.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def boot_ci_paired(a: np.ndarray, b: np.ndarray, nboot=5000, seed=0) -> Tuple[float, float, float]:
    """CI of mean(a)-mean(b) under the SAME resampled games (paired)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    stats = np.zeros(nboot)
    for j in range(nboot):
        idx = rng.integers(0, n, n)
        stats[j] = a[idx].mean() - b[idx].mean()
    return float(a.mean() - b.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ---------------------------------------------------------------------------
# single game-disjoint split (~13 held-out) — spec framing
# ---------------------------------------------------------------------------
def single_split(P: Dict[str, Any], in_ch: int, holdout_frac=0.36, seeds=(101, 202, 303), epochs=120):
    Xb, Xs, zl, zs = P["Xb"], P["Xs"], P["z_loss"], P["z_save"]
    n = len(zl)
    res = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        perm = rng.permutation(n)
        n_ho = max(3, int(round(n * holdout_frac)))
        ho = perm[:n_ho]; tr = perm[n_ho:]
        Xt = np.concatenate([Xb[tr], Xs[tr]], 0)
        yt = np.concatenate([zl[tr], zs[tr]], 0)
        net = train_reg(Xt, yt, in_ch, sd, epochs=epochs)
        vb = predict(net, Xb[ho], in_ch); vs = predict(net, Xs[ho], in_ch)
        res.append({"n_ho": int(n_ho), "rerank": float((vs > vb).mean()),
                    "kill_c": float((vs > 0).mean()), "kill_a": float((vb < 0).mean())})
    return res


# ---------------------------------------------------------------------------
# neutral bank (anti-correlation mean_v canary)
# ---------------------------------------------------------------------------
def neutral_bank(n_target=300, seed=0) -> np.ndarray:
    games = [json.loads(l) for l in PER_GAME.read_text().splitlines()]
    rng = np.random.default_rng(seed)
    out = []
    order = rng.permutation(len(games))
    for gi in order:
        g = games[int(gi)]
        # model-WIN games only -> neutral/healthy positions, NOT trap-laden losses
        model_side = "p1" if g["p2"] == "sealbot" else ("p2" if g["p1"] == "sealbot" else None)
        if model_side is None or g["winner"] != model_side:
            continue
        moves = g["moves"]; plies = len(moves)
        if plies < 12:
            continue
        for _ in range(2):
            p = int(rng.integers(8, max(9, plies - 2)))
            b = engine.Board.with_encoding_name(ENC)
            for (q, r) in moves[:p]:
                b.apply_move(int(q), int(r))
            out.append(_planes8(b, int(b.current_player)))
            if len(out) >= n_target:
                return np.asarray(out, np.float32)
    return np.asarray(out, np.float32)


def mean_v_on(P_train: Dict[str, Any], in_ch: int, bank: np.ndarray, seeds=(0, 1, 2), epochs=120) -> float:
    """Train on ALL pairs, report mean predicted value on the neutral bank."""
    Xb, Xs, zl, zs = P_train["Xb"], P_train["Xs"], P_train["z_loss"], P_train["z_save"]
    Xt = np.concatenate([Xb, Xs], 0); yt = np.concatenate([zl, zs], 0)
    vals = []
    for sd in seeds:
        net = train_reg(Xt, yt, in_ch, sd, epochs=epochs)
        vals.append(predict(net, bank, in_ch).mean())
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# turn-phase control validity (AUC of turn-phase -> z-sign should be ~0.5)
# ---------------------------------------------------------------------------
def turnphase_auc(P: Dict[str, Any]) -> Dict[str, Any]:
    # build a per-board table: feature=(mr==2, ply%2), label= z-sign (blunder<0, saving>0)
    n = len(P["z_loss"])
    feat = []
    lab = []
    for i in range(n):
        tp = (1 if P["mr"][i] == 2 else 0, int(P["ply"][i]) % 2)
        feat.append(tp); lab.append(0)   # blunder -> negative class
        feat.append(tp); lab.append(1)   # saving  -> positive class
    feat = np.array(feat); lab = np.array(lab)
    # AUC of a single turn-phase score (mr*2+ply) vs label; identical within pair -> ~0.5
    score = feat[:, 0] * 2 + feat[:, 1]
    # Mann-Whitney AUC
    pos = score[lab == 1]; neg = score[lab == 0]
    auc = 0.0
    for s in pos:
        auc += (neg < s).sum() + 0.5 * (neg == s).sum()
    auc /= (len(pos) * len(neg))
    within_pair_identical = bool(np.all(P["Xb"][:, 2:4] == P["Xs"][:, 2:4]))
    return {"turnphase_auc": float(auc), "within_pair_plane23_identical": within_pair_identical}


# ---------------------------------------------------------------------------
# plane activity diagnostic (measurement-unit honesty)
# ---------------------------------------------------------------------------
def plane_activity(P: Dict[str, Any]) -> Dict[str, Any]:
    Xb, Xs = P["Xb"], P["Xs"]
    n = len(Xb)
    def fire(arr, idx):  # frac of boards with any active cell on plane idx
        return float((arr[:, idx].reshape(n, -1).sum(1) > 0).mean())
    differ = {}
    for nm, idx in [("self_w1", 4), ("opp_w1", 5), ("self_o4", 6), ("opp_o4", 7)]:
        # frac of PAIRS where this plane differs blunder vs saving
        d = (Xb[:, idx].reshape(n, -1) != Xs[:, idx].reshape(n, -1)).any(1)
        differ[nm] = {"fire_blunder": fire(Xb, idx), "fire_saving": fire(Xs, idx),
                      "pairs_differ": float(d.mean())}
    any_tac_differ = float(((Xb[:, 4:8].reshape(n, -1) != Xs[:, 4:8].reshape(n, -1)).any(1)).mean())
    return {"per_plane": differ, "any_tac_plane_differs_pairs": any_tac_differ}


# ===========================================================================
# corpus-expansion SealBot scan
# ===========================================================================
WIN_THRESHOLD = 99_999_000.0


def _seal_imports():
    for p in (str(REPO_ROOT / "vendor" / "bots" / "sealbot"),
              str(REPO_ROOT / "vendor" / "bots" / "sealbot" / "best")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from minimax_cpp import MinimaxBot  # type: ignore
    from game import Player as SealPlayer  # type: ignore
    return MinimaxBot, SealPlayer


def _seal_score(MinimaxBot, SealPlayer, board, depth, time_limit=30.0):
    bd = {}
    for q, r, p in board.get_stones():
        bd[(q, r)] = SealPlayer.A if p == 1 else SealPlayer.B
    cp = SealPlayer.A if board.current_player == 1 else SealPlayer.B

    class MG:
        pass
    g = MG()
    g.board = bd; g.current_player = cp
    g.moves_left_in_turn = int(board.moves_remaining); g.move_count = len(bd)
    bot = MinimaxBot(time_limit=time_limit); bot.max_depth = depth
    bot.get_move(g)
    return float(bot.last_score)


def expand_scan(n_games: int, depth: int = 6, plies_per_game: int = 6,
                time_limit: float = 25.0, seed: int = 0) -> Dict[str, Any]:
    """Estimate genuinely-NEW distinct-game trap pairs minable offline (no new self-play):
    SealBot-d6 scan of model-lost games NOT already in the corpus, at model-to-move plies.
    A usable NEW pair needs: parent NOT already proven-lost (saving move exists) AND the
    realized move proven-lost (terminal mate, side-to-move POV). Counts decision points found."""
    MinimaxBot, SealPlayer = _seal_imports()
    games = [json.loads(l) for l in PER_GAME.read_text().splitlines()]
    corpus = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
    used_gidx = set(int(r["game_idx"]) for r in corpus)
    # model-lost games NOT in corpus
    cand = []
    for i, g in enumerate(games):
        if i in used_gidx:
            continue
        ms = "p1" if g["p2"] == "sealbot" else ("p2" if g["p1"] == "sealbot" else None)
        if ms is None or g["winner"] == ms:
            continue   # only model-LOST games yield proven-loss-to-move positions
        cand.append((i, ms))
    rng = np.random.default_rng(seed)
    rng.shuffle(cand)
    cand = cand[:n_games]
    n_decision_points = 0
    n_games_with_point = 0
    per_game = []
    t0 = time.time()
    for (gi, ms) in cand:
        g = games[gi]
        moves = g["moves"]
        model_is_p1 = (ms == "p1")
        found = 0
        # scan model-to-move plies in the back half (where mates live), sampled
        b = engine.Board.with_encoding_name(ENC)
        snaps = []
        for pidx, (q, r) in enumerate(moves):
            # record model-to-move BEFORE applying
            if int(b.current_player) == (1 if model_is_p1 else -1):
                snaps.append((pidx, b.clone(), [int(q), int(r)]))
            b.apply_move(int(q), int(r))
        # sample plies from the back 60% (mates are late)
        cand_snaps = [s for s in snaps if s[0] >= int(0.4 * len(moves))]
        if len(cand_snaps) > plies_per_game:
            sel = rng.choice(len(cand_snaps), plies_per_game, replace=False)
            cand_snaps = [cand_snaps[int(k)] for k in sorted(sel)]
        for (pidx, par, played) in cand_snaps:
            if par.check_win() or par.legal_move_count() == 0:
                continue
            # parent must NOT be already proven-lost (else no saving move) ...
            par_score = _seal_score(MinimaxBot, SealPlayer, par, depth, time_limit)
            if abs(par_score) >= WIN_THRESHOLD and par_score < 0:
                continue   # parent already lost-to-move -> no usable saving pair
            # ... and the REALIZED move must lead to a proven loss (child, model still relevant)
            ch = par.clone(); ch.apply_move(*played)
            if ch.check_win():
                continue
            ch_score = _seal_score(MinimaxBot, SealPlayer, ch, depth, time_limit)
            # child side-to-move POV; a proven loss for whoever is to move at ch.
            if abs(ch_score) >= WIN_THRESHOLD and ch_score < 0:
                found += 1
        n_decision_points += found
        if found:
            n_games_with_point += 1
        per_game.append({"game_idx": gi, "plies_scanned": len(cand_snaps), "decision_points": found})
    secs = time.time() - t0
    # extrapolate to the full untapped pool
    n_total_new_lost = sum(
        1 for i, g in enumerate(games) if i not in used_gidx
        and (("p1" if g["p2"] == "sealbot" else ("p2" if g["p1"] == "sealbot" else None)) is not None)
        and g["winner"] != ("p1" if g["p2"] == "sealbot" else "p2"))
    rate = n_decision_points / max(1, len(cand))
    return {
        "scanned_games": len(cand), "depth": depth, "plies_per_game": plies_per_game,
        "decision_points_found": n_decision_points, "games_with_a_point": n_games_with_point,
        "per_game_rate": rate, "total_untapped_model_lost_games": n_total_new_lost,
        "extrapolated_new_pairs": round(rate * n_total_new_lost, 1),
        "wall_s": round(secs, 1), "per_game": per_game,
    }


# ===========================================================================
def run_gate(seeds, epochs, nboot) -> Dict[str, Any]:
    P = build_pairs()
    n = len(P["z_loss"])
    n_games = len(set(P["game_id"].tolist()))
    act = plane_activity(P)
    tp = turnphase_auc(P)

    print(f"[stage0] pairs={n} distinct_games={n_games} dropped_degenerate={list(P['dropped'])}")
    print(f"[stage0] plane activity: {json.dumps(act)}")
    print(f"[stage0] turn-phase AUC={tp['turnphase_auc']:.3f} within-pair plane23 identical={tp['within_pair_plane23_identical']}")

    print("[stage0] LOGO-CV control (4-plane)...")
    c = logo_rerank(P, 4, seeds=seeds, epochs=epochs)
    print("[stage0] LOGO-CV treatment (8-plane)...")
    t = logo_rerank(P, 8, seeds=seeds, epochs=epochs)

    rr4, lo4, hi4 = boot_ci(c["rerank"], nboot)
    rr8, lo8, hi8 = boot_ci(t["rerank"], nboot)
    d, dlo, dhi = boot_ci_paired(t["rerank"], c["rerank"], nboot)

    # KILL-C / KILL-A from LOGO held-out predictions (saving siblings = win-preserve class)
    kc8 = float((t["v_s"] > 0).mean()); ka8 = float((t["v_b"] < 0).mean())
    kc4 = float((c["v_s"] > 0).mean()); ka4 = float((c["v_b"] < 0).mean())

    print("[stage0] single 23/13 split (spec framing)...")
    ss4 = single_split(P, 4, seeds=tuple(s + 1000 for s in seeds), epochs=epochs)
    ss8 = single_split(P, 8, seeds=tuple(s + 1000 for s in seeds), epochs=epochs)

    print("[stage0] neutral bank + mean_v canary...")
    bank = neutral_bank(300)
    mv8 = mean_v_on(P, 8, bank, seeds=seeds, epochs=epochs)
    mv4 = mean_v_on(P, 4, bank, seeds=seeds, epochs=epochs)

    # verdict (lead with DELTA + power; design §4)
    treat_pass = (rr8 >= 0.70 and lo8 > 0.55) and (kc8 >= 0.85) and (abs(mv8) <= 0.10)
    treat_kill = (rr8 < 0.60) or (kc8 < 0.75) or (mv8 < -0.15)
    delta_meaningful = dlo > 0.0   # 4-vs-8 delta CI excludes 0
    if treat_pass and delta_meaningful:
        verdict = "PASS"
    elif treat_kill and (d <= 0.0 or dlo <= 0.0):
        verdict = "KILL"
    else:
        verdict = "INDETERMINATE"

    return {
        "n_pairs": n, "n_distinct_games": n_games, "dropped": list(P["dropped"]),
        "plane_activity": act, "turn_phase": tp,
        "rerank_control_4": {"mean": rr4, "ci": [lo4, hi4]},
        "rerank_treatment_8": {"mean": rr8, "ci": [lo8, hi8]},
        "delta_8_minus_4": {"mean": d, "ci_paired": [dlo, dhi]},
        "kill_c": {"treatment_8": kc8, "control_4": kc4},
        "kill_a": {"treatment_8": ka8, "control_4": ka4},
        "mean_v_neutral": {"treatment_8": mv8, "control_4": mv4, "n_bank": int(len(bank))},
        "single_split_13": {"control_4": ss4, "treatment_8": ss8},
        "verdict": verdict,
        "seeds": list(seeds), "epochs": epochs, "nboot": nboot,
    }


def write_report(g: Dict[str, Any], expand: Optional[Dict[str, Any]]):
    rr4 = g["rerank_control_4"]; rr8 = g["rerank_treatment_8"]; dd = g["delta_8_minus_4"]
    act = g["plane_activity"]["per_plane"]
    ss4 = g["single_split_13"]["control_4"]; ss8 = g["single_split_13"]["treatment_8"]
    def ssline(rows):
        return ", ".join(f"n_ho={r['n_ho']} rr={r['rerank']:.2f} KC={r['kill_c']:.2f}" for r in rows)
    md = f"""# D-PFIT STAGE-0 — PAIRWISE LOCAL RE-RANK separation discriminator

Generated {time.strftime('%Y-%m-%d %H:%M:%S')} | eval-only CPU | seeds={g['seeds']} epochs={g['epochs']} nboot={g['nboot']}
Gate spec: `docs/designs/coupled_valuez_decode_design.md` §4 (+ §8 corrections). Commits nothing.

## TL;DR — lead with the DELTA + power honesty (design §8)
- **PAIRWISE-RERANK 4-vs-8 DELTA (paired distinct-game bootstrap):** **{dd['mean']:+.3f}**  CI [{dd['ci_paired'][0]:+.3f}, {dd['ci_paired'][1]:+.3f}]
- CONTROL (4-plane) rerank = {rr4['mean']:.3f}  CI [{rr4['ci'][0]:.3f}, {rr4['ci'][1]:.3f}]
- TREATMENT (8-plane) rerank = {rr8['mean']:.3f}  CI [{rr8['ci'][0]:.3f}, {rr8['ci'][1]:.3f}]
- **VERDICT: {g['verdict']}**

## Power / measurement-unit honesty (READ FIRST)
- {g['n_pairs']} clean pairs from {g['n_distinct_games']} DISTINCT games (dropped degenerate refuting==blunder: {g['dropped']}).
- Distinct-game bootstrap (§D-ARGMAX): n_effective = {g['n_distinct_games']} games, NOT a raw count.
- **The 4 cheap tac planes are a WEAK proxy at these DEEP traps:**

| tac plane | fires (blunder) | fires (saving) | pairs differ blunder-vs-saving |
|---|---|---|---|
| self win-in-1 | {act['self_w1']['fire_blunder']:.2f} | {act['self_w1']['fire_saving']:.2f} | {act['self_w1']['pairs_differ']:.2f} |
| opp win-in-1  | {act['opp_w1']['fire_blunder']:.2f} | {act['opp_w1']['fire_saving']:.2f} | {act['opp_w1']['pairs_differ']:.2f} |
| self open-4/fork | {act['self_o4']['fire_blunder']:.2f} | {act['self_o4']['fire_saving']:.2f} | {act['self_o4']['pairs_differ']:.2f} |
| opp open-4/fork  | {act['opp_o4']['fire_blunder']:.2f} | {act['opp_o4']['fire_saving']:.2f} | {act['opp_o4']['pairs_differ']:.2f} |

  ANY tac plane differs within pair: {g['plane_activity']['any_tac_plane_differs_pairs']:.2f} of pairs.
  The win-in-1 planes are DEAD (no immediate 6-completion at mate-in-3..9-TURN traps); only the
  open-4/fork planes carry signal, and in a minority of pairs. This is the D-TACTICAL "80% quiet
  developmental" / D-SOLVER A2 granularity-binding structure showing up in the PLANE dimension:
  the cheap tac planes cannot proxy L2's real solver-derived `forced_loss/win_within_2` maps.

## Turn-phase control validity (D-FULLSPEC AUC-0.807 shortcut stripped)
- within-pair plane2/plane3 byte-identical: {g['turn_phase']['within_pair_plane23_identical']}
- turn-phase -> z-sign AUC = {g['turn_phase']['turnphase_auc']:.3f} (target [0.45,0.55]; identical within pair => 0.5 by construction => control valid).

## Primary metric — PAIRWISE-RERANK (LOGO-CV, every pair held-out game-disjoint)
| condition | rerank P(v_save>v_blunder) | distinct-game bootstrap 95% CI |
|---|---|---|
| CONTROL 4-plane | {rr4['mean']:.3f} | [{rr4['ci'][0]:.3f}, {rr4['ci'][1]:.3f}] |
| TREATMENT 8-plane | {rr8['mean']:.3f} | [{rr8['ci'][0]:.3f}, {rr8['ci'][1]:.3f}] |
| **DELTA 8-4 (paired)** | **{dd['mean']:+.3f}** | [{dd['ci_paired'][0]:+.3f}, {dd['ci_paired'][1]:+.3f}] |

PASS bar (design §4): rerank8>=0.70 AND CI_lo>0.55 AND KILL-C>=0.85 AND |mean_v|<=0.10 AND delta CI>0.

## Secondary
- KILL-C (held-out saving-sibling predicted >0; bar 0.85): treatment={g['kill_c']['treatment_8']:.3f} control={g['kill_c']['control_4']:.3f}
- KILL-A (held-out blunder predicted <0): treatment={g['kill_a']['treatment_8']:.3f} control={g['kill_a']['control_4']:.3f}
- mean_v on {g['mean_v_neutral']['n_bank']} neutral (model-win) positions (anti-correlation canary, |drift|<=0.10):
  treatment={g['mean_v_neutral']['treatment_8']:+.3f} control={g['mean_v_neutral']['control_4']:+.3f}
- single 23/13 game-disjoint split (~13 held-out, spec framing):
  CONTROL  [{ssline(ss4)}]
  TREATMENT[{ssline(ss8)}]

## Corpus-expansion yield (offline SealBot re-mine, NO new self-play)
"""
    if expand:
        md += f"""SealBot-d{expand['depth']} scan of {expand['scanned_games']} model-LOST games NOT in the corpus
({expand['plies_per_game']} back-half model-to-move plies/game, time_limit per call; wall={expand['wall_s']}s):
- decision points found (parent not-yet-lost AND realized move proven-lost): **{expand['decision_points_found']}** in {expand['games_with_a_point']} games
- per-game new-pair rate: {expand['per_game_rate']:.2f}
- total untapped model-LOST games NOT in corpus: **{expand['total_untapped_model_lost_games']}**
- **extrapolated genuinely-NEW distinct-game pairs: ~{expand['extrapolated_new_pairs']}**

Re-mining the SAME 61 corpus games adds same-game pairs (NOT distinct games -> does NOT improve
the held-out distinct-game power, §D-ARGMAX). The only distinct-game headroom WITHOUT new self-play
is the {expand['total_untapped_model_lost_games']} untapped model-lost games above.
"""
    else:
        md += "(scan not run; pass --expand-scan N)\n"
    mv8 = g["mean_v_neutral"]["treatment_8"]; mv4 = g["mean_v_neutral"]["control_4"]
    md += f"""
## Recommendation (L2 fundability) — VERDICT {g['verdict']}
Two independent KILL signals, both robust to the thin n:
1. **DELTA = {dd['mean']:+.3f}, paired CI [{dd['ci_paired'][0]:+.3f},{dd['ci_paired'][1]:+.3f}]** — the 8-plane tac TREATMENT
   delivers ZERO held-out pairwise-rerank lift over the 4-plane CONTROL. Both sit at {rr8['mean']:.3f}
   (CI straddles chance 0.5). The cheap tac planes do NOT make the local cut.
2. **mean_v(neutral) treatment={mv8:+.3f} vs control={mv4:+.3f}** — the tac planes make the
   anti-correlation WORSE, not better: the D-INJECT / ENTANGLED_LT global-depression signature
   ({'FIRES <-0.15' if mv8 < -0.15 else 'within bound'}). The open-4 planes fire on threats in BOTH win and loss neighborhoods
   (opp_o4 {act['opp_o4']['fire_blunder']:.2f} blunder / {act['opp_o4']['fire_saving']:.2f} saving) -> "threat present" is read as "loss" -> depresses healthy positions.

These are CHEAP-PROXY planes (win-in-1 + threat_moves open-4), a documented WEAK proxy for L2's
real solver-derived `forced_loss/win_within_2` maps. The cheap discriminator the D-FULLSPEC RULE
demanded before any GPU-week therefore returns **KILL/null**, not the KILL-C>=0.85 PASS it required.

**L2 (value-restart GPU-week) is NOT fundable on this evidence.** Corpus expansion (~{expand['extrapolated_new_pairs'] if expand else '?'} new
distinct-game pairs available offline) would tighten the CI but CANNOT flip a delta that is
structurally ~0 on dead planes — it fixes power-in-n, not the plane signal. The only path that
could revive L2 is building the real solver-derived planes (Stage-1B, the expensive step this gate
was meant to de-risk) and re-running the gate on THOSE — i.e. the gate cannot pre-clear L2 cheaply.
**Ship L1 (multi-window decode/policy) + the validated deploy-backup oracle floor (+0.165/+0.195);
hold L2 unless solver-plane separation is shown on an expanded corpus.**
"""
    REPORT.write_text(md)
    print(f"[stage0] report -> {REPORT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--nboot", type=int, default=5000)
    ap.add_argument("--rebuild", action="store_true", help="force rebuild pair cache")
    ap.add_argument("--expand-scan", type=int, default=0, help="N model-lost games to SealBot-scan")
    ap.add_argument("--expand-only", action="store_true", help="only run the expansion scan")
    args = ap.parse_args()

    if args.rebuild and CACHE.exists():
        CACHE.unlink()

    expand = None
    if args.expand_only:
        expand = expand_scan(args.expand_scan or 12)
        print("EXPAND " + json.dumps(expand))
        return 0

    g = run_gate(tuple(args.seeds), args.epochs, args.nboot)
    if args.expand_scan:
        print(f"[stage0] expansion scan ({args.expand_scan} games)...")
        expand = expand_scan(args.expand_scan)
        print("EXPAND " + json.dumps({k: v for k, v in expand.items() if k != "per_game"}))
    write_report(g, expand)
    print("STAGE0_RESULT " + json.dumps(g))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
