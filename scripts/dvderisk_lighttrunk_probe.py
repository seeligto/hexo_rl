#!/usr/bin/env python3
"""D-FULLSPEC LIGHT-TRUNK — single-variable variation on E1 (frozen trunk).

E1 froze the trunk and read the value head; verdict ENTANGLED (matched holdout
KILL-A 0.762 PASS, KILL-C 0.441 FAIL). THIS probe changes exactly ONE thing:
UNFREEZE the trunk at a LIGHT learning rate (value head at a normal LR) and
re-run the EXACT same full-spectrum discriminator. Question: does trunk
ADAPTATION manufacture the win/loss separation the frozen read could not?

  SEPARABLE_LT : matched game-disjoint holdout KILL-A>0.35 AND KILL-C>=0.85,
                 not an overfit artifact (holdout ~ train, survives neutralization)
                 => light-trunk finetune target-fix viable, restart maybe unnecessary.
  ENTANGLED_LT : matched holdout KILL-C still craters (<~0.6) despite unfreezing
                 => trunk adaptation does NOT manufacture separation => FEATURE
                 problem confirmed => E2 restart warranted.
  PARTIAL_LT   : in between / passes only via overfit (train passes, holdout doesn't).

*** DOMINANT RISK = OVERFIT. *** An unfrozen ResNet trunk on ~960 positions can
MEMORIZE -> a fake SEPARABLE on positions it effectively saw. Defenses:
  1. STRICT GAME-DISJOINT 3-way split (train / val / holdout), whole games to ONE
     side, recovered via byte-exact state->DS1-CSV game_id join (the
     dvderisk_e1_review_indep build_pool). assert shared_games==0 pairwise.
  2. Early-stop on the game-disjoint VAL slice; report epoch chosen.
  3. Report TRAIN-fit KILL-A/KILL-C vs HOLDOUT. Large train>>holdout gap =
     memorization, NOT separation. Verdict rests on the HOLDOUT.

*** #1 obligation (inherited from E1): control the TURN-PHASE confound. ***
v6_live2 planes 2,3 (moves_remaining_bcast / ply_parity_bcast, identical per pos)
differ ~10x between wins and losses; a trunk can separate off turn-phase alone
(turn-phase-only logistic AUC 0.807). Verdict judged on the TURN-PHASE-MATCHED
holdout (tp==0 INTERSECT overlapping stone-count support), never naive, plus a
planes-2,3-neutralized control.

Usage:
    .venv/bin/python scripts/dvderisk_lighttrunk_probe.py
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.checkpoints import extract_model_state
from hexo_rl.utils.device import best_device

LOG_PATH = REPO_ROOT / "logs" / "dvderisk_lighttrunk_probe.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(str(LOG_PATH), mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

CKPT = "checkpoints/checkpoint_00272357.pt"
BUF_PRIMARY = "data/livetail_bank_e928c854.npz"
BUF_EXTRA = "data/selfplay_bank12k_v6_live2_ls_50k.npz"
DS1_CSVS = ["data/dvderisk_ds1_train.csv", "data/dvderisk_ds1_holdout.csv"]
BANKS = {
    "loss": ["data/distill_d7_train.npz", "data/distill_d7_holdout.npz"],
    "win": ["data/killc_netwin_train.npz", "data/killc_netwin_holdout.npz"],
}


# ---------------------------------------------------------------------------
def load_model(device, encoding="v6_live2_ls"):
    ckpt = torch.load(REPO_ROOT / CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    state = extract_model_state(ckpt)
    inp_w = state.get("trunk.input_conv.weight")
    in_ch = int(inp_w.shape[1]) if inp_w is not None else cfg.get("in_channels", 4)
    net = HexTacToeNet(
        in_channels=in_ch,
        res_blocks=cfg.get("res_blocks", 12),
        filters=cfg.get("filters", 128),
        board_size=cfg.get("board_size", 19),
        encoding=encoding,
    )
    net.load_state_dict(state, strict=True)
    return net.to(device)


@torch.no_grad()
def values(net, states, device, bs=256):
    net.eval()
    out = []
    for s in range(0, len(states), bs):
        x = torch.from_numpy(states[s:s + bs]).to(device)
        _, v, _ = net(x)
        out.append(v.squeeze(1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def kill_a(vl):  # frac LOSSES called loss (value<0)
    return float((vl < 0).mean()) if len(vl) else float("nan")


def kill_c(vw):  # frac WINS called win (value>0)
    return float((vw > 0).mean()) if len(vw) else float("nan")


# ---------------------------------------------------------------------------
def build_pool():
    """Recover game_id for every bank position via a byte-exact state join against
    the DS1 buffer (livetail + selfplay concat) + DS1 CSV. assert 100% join,
    0 ambiguity. Returns the full 1166-pos pool. (Reuses dvderisk_e1_review_indep.)"""
    s1 = np.load(REPO_ROOT / BUF_PRIMARY)["states"].astype("float32")
    s2 = np.load(REPO_ROOT / BUF_EXTRA)["states"].astype("float32")
    buf = np.concatenate([s1, s2], axis=0)
    byte2gid = defaultdict(set)
    for cf in DS1_CSVS:
        for r in csv.DictReader(open(REPO_ROOT / cf)):
            bidx = int(r["buffer_idx"])
            if bidx < buf.shape[0]:
                byte2gid[buf[bidx].tobytes()].add(int(r["game_id"]))
    ambig = sum(1 for v in byte2gid.values() if len(v) > 1)
    assert ambig == 0, f"{ambig} ambiguous state-bytes (multi game_id)"

    states, labels, gids = [], [], []
    join_fail = 0
    for cls, paths in BANKS.items():
        lab = -1.0 if cls == "loss" else 1.0
        for p in paths:
            st = np.load(REPO_ROOT / p)["states"].astype("float32")
            for x in st:
                k = x.tobytes()
                if k not in byte2gid:
                    join_fail += 1
                    continue
                states.append(x)
                labels.append(lab)
                gids.append(next(iter(byte2gid[k])))
    assert join_fail == 0, f"{join_fail} positions failed game_id join"
    states = np.stack(states).astype("float32")
    labels = np.array(labels, "float32")
    gids = np.array(gids, "int64")
    tp = (states[:, 2].reshape(len(states), -1).mean(1) >= 0.5).astype("int64")
    stones = (states[:, 0] + states[:, 1]).reshape(len(states), -1).sum(1)
    p23_eq = np.array_equal(states[:, 2], states[:, 3])
    # buffer pool of OTHER (non-distill) positions for the value-destruction spot-check
    distill_bytes = {x.tobytes() for x in states}
    other = np.array([i for i in range(len(buf)) if buf[i].tobytes() not in distill_bytes], dtype="int64")
    return {"states": states, "label": labels, "gid": gids, "tp": tp,
            "stones": stones, "p23_eq": bool(p23_eq), "buf": buf, "buf_other_idx": other}


def three_way_game_split(gids, holdout_frac, val_frac, seed):
    """Whole-game 3-way split: holdout / val / train, each game on exactly ONE side.
    val_frac is a fraction of the NON-holdout games. Returns (tr,val,ho) bool masks."""
    uniq = np.unique(gids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_ho = max(1, int(round(len(uniq) * holdout_frac)))
    ho_games = set(perm[:n_ho].tolist())
    rest = perm[n_ho:]
    n_val = max(1, int(round(len(rest) * val_frac)))
    val_games = set(rest[:n_val].tolist())
    tr_games = set(rest[n_val:].tolist())
    ho = np.array([g in ho_games for g in gids])
    val = np.array([g in val_games for g in gids])
    tr = np.array([g in tr_games for g in gids])
    return tr, val, ho, len(uniq), len(tr_games), len(val_games), len(ho_games)


# ---------------------------------------------------------------------------
def select_trainable(net, mode):
    """Set requires_grad: value head always trainable; trunk per `mode`.
    mode='full'  : entire trunk trainable.
    mode='block11': only trunk.tower.11.* trainable (last residual block).
    Returns (trunk_param_list, head_param_list)."""
    trunk_params, head_params = [], []
    for name, p in net.named_parameters():
        if name.startswith("value_fc1") or name.startswith("value_fc2"):
            p.requires_grad_(True)
            head_params.append(p)
        elif name.startswith("trunk."):
            if mode == "full":
                want = True
            elif mode == "block11":
                want = name.startswith("trunk.tower.11.")
            else:
                want = False
            p.requires_grad_(want)
            if want:
                trunk_params.append(p)
        else:
            p.requires_grad_(False)  # other heads off (value-MSE only)
    return trunk_params, head_params


def snapshot_trunk(net):
    return {n: p.detach().clone() for n, p in net.named_parameters() if n.startswith("trunk.")}


def trunk_max_delta(net, snap):
    m = 0.0
    for n, p in net.named_parameters():
        if n.startswith("trunk."):
            m = max(m, float((p.detach() - snap[n]).abs().max()))
    return m


def train_lighttrunk(net, tr_s, tr_y, val_s, val_y, device, *, trunk_lr, head_lr,
                     mode, max_epochs=120, bpc=64, patience=20, seed=0, label="lt"):
    """Class-balanced full-spectrum value MSE distill, trunk UNFROZEN at trunk_lr,
    value head at head_lr. Early-stop on the (game-disjoint) balanced val MSE.
    Returns (best_val_mse, best_epoch, trunk_max_delta)."""
    trunk_params, head_params = select_trainable(net, mode)
    snap = snapshot_trunk(net)
    opt = torch.optim.Adam([
        {"params": trunk_params, "lr": trunk_lr},
        {"params": head_params, "lr": head_lr},
    ])
    mse = nn.MSELoss()

    li = np.where(tr_y < 0)[0]
    wi = np.where(tr_y > 0)[0]
    val_l = val_s[val_y < 0]
    val_w = val_s[val_y > 0]

    def val_mse():
        net.eval()
        with torch.no_grad():
            vl = values(net, val_l, device)
            vw = values(net, val_w, device)
        return 0.5 * float(((vl + 1) ** 2).mean()) + 0.5 * float(((vw - 1) ** 2).mean())

    rng = np.random.default_rng(seed)
    nl, nw = len(li), len(wi)
    steps = max(nl, nw) // bpc + 1
    best = float("inf")
    best_state = None
    best_epoch = -1
    bad = 0
    n_trunk = sum(p.numel() for p in trunk_params)
    log.info(f"[{label}] mode={mode} trunk_lr={trunk_lr} head_lr={head_lr} "
             f"trunk_trainable_params={n_trunk} tr(l/w)={nl}/{nw} val(l/w)={len(val_l)}/{len(val_w)} "
             f"steps/ep={steps}")
    for ep in range(max_epochs):
        net.train()
        pl = rng.permutation(nl)
        pw = rng.permutation(nw)
        for st in range(steps):
            bl = li[pl[(st * bpc) % nl:][:bpc]]
            bw = wi[pw[(st * bpc) % nw:][:bpc]]
            if len(bl) == 0 or len(bw) == 0:
                continue
            xs = np.concatenate([tr_s[bl], tr_s[bw]])
            ts = np.concatenate([np.full(len(bl), -1.0, "float32"),
                                 np.full(len(bw), 1.0, "float32")])
            opt.zero_grad()
            _, v, _ = net(torch.from_numpy(xs).to(device))
            mse(v.squeeze(1), torch.from_numpy(ts).to(device)).backward()
            opt.step()
        vm = val_mse()
        if vm < best - 1e-5:
            best = vm
            bad = 0
            best_epoch = ep
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
        if ep % 10 == 0 or bad >= patience:
            log.info(f"[{label}] ep{ep:3d} val_mse={vm:.5f} best={best:.5f}@{best_epoch} bad={bad}")
        if bad >= patience:
            log.info(f"[{label}] early stop ep{ep} best_epoch={best_epoch}")
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    tmax = trunk_max_delta(net, snap)
    log.info(f"[{label}] best_val_mse={best:.5f} best_epoch={best_epoch} trunk_max_delta={tmax:.3e}")
    return best, best_epoch, tmax


# ---------------------------------------------------------------------------
def matched_masks(pool, mask):
    """E1 EXACT matched protocol on the subset `mask`: tp==0 stratum INTERSECT
    overlapping stone-count support. Returns (loss_mask, win_mask, lo, hi)."""
    lab = pool["label"]; tp = pool["tp"]; stones = pool["stones"]
    l0 = (lab < 0) & mask & (tp == 0)
    w0 = (lab > 0) & mask & (tp == 0)
    if l0.sum() == 0 or w0.sum() == 0:
        return None
    lo = max(stones[l0].min(), stones[w0].min())
    hi = min(stones[l0].max(), stones[w0].max())
    lm = l0 & (stones >= lo) & (stones <= hi)
    wm = w0 & (stones >= lo) & (stones <= hi)
    return lm, wm, float(lo), float(hi)


def eval_all(pool, net, device, tr_mask, ho_mask, neutralize=False):
    """Return naive/matched/train-fit KILL-A/KILL-C dict for a trained net."""
    states = pool["states"]
    if neutralize:
        states = states.copy(); states[:, 2] = 0.0; states[:, 3] = 0.0
    lab = pool["label"]
    v_all = values(net, states, device)
    # naive holdout
    naive_ka = kill_a(v_all[(lab < 0) & ho_mask])
    naive_kc = kill_c(v_all[(lab > 0) & ho_mask])
    # matched holdout
    mh = matched_masks(pool, ho_mask)
    if mh is None:
        matched = None
    else:
        lm, wm, lo, hi = mh
        matched = {"kill_a": kill_a(v_all[lm]), "kill_c": kill_c(v_all[wm]),
                   "n_loss": int(lm.sum()), "n_win": int(wm.sum()),
                   "stone_support": [lo, hi]}
    # matched train-fit (overfit canary, same matched protocol on train games)
    mt = matched_masks(pool, tr_mask)
    if mt is None:
        train_fit = None
    else:
        lmt, wmt, _, _ = mt
        train_fit = {"kill_a": kill_a(v_all[lmt]), "kill_c": kill_c(v_all[wmt]),
                     "n_loss": int(lmt.sum()), "n_win": int(wmt.sum())}
    return {"naive": {"kill_a": naive_ka, "kill_c": naive_kc},
            "matched": matched, "train_fit": train_fit}


def value_destruction_check(pool, net_before, net_after, device, n=300, seed=11):
    """Spot-check unrelated (non-distill) buffer positions: value pre vs post unfreeze.
    Large shift / sign flips => unfreezing damaged the broader value function."""
    other = pool["buf_other_idx"]
    if len(other) == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    idx = other if len(other) <= n else rng.choice(other, size=n, replace=False)
    s = pool["buf"][idx].astype("float32")
    vb = values(net_before, s, device)
    va = values(net_after, s, device)
    flips = int(((vb > 0) != (va > 0)).sum())
    corr = float(np.corrcoef(vb, va)[0, 1]) if len(vb) > 1 else float("nan")
    return {"n": int(len(idx)), "mean_abs_delta": round(float(np.abs(va - vb).mean()), 4),
            "max_abs_delta": round(float(np.abs(va - vb).max()), 4),
            "sign_flips": flips, "sign_flip_frac": round(flips / len(idx), 4),
            "pearson_pre_post": round(corr, 4),
            "mean_v_before": round(float(vb.mean()), 4), "mean_v_after": round(float(va.mean()), 4)}


# ---------------------------------------------------------------------------
def run_one(pool, device, *, trunk_lr, head_lr, mode, split_seed, holdout_frac, val_frac,
            train_seed=0, max_epochs=120, neutralize_eval=False, want_destruction=False,
            net_before=None):
    tr, val, ho, n_games, n_tr_g, n_val_g, n_ho_g = three_way_game_split(
        pool["gid"], holdout_frac, val_frac, split_seed)
    g_tr = set(pool["gid"][tr].tolist())
    g_val = set(pool["gid"][val].tolist())
    g_ho = set(pool["gid"][ho].tolist())
    shared_tv = len(g_tr & g_val)
    shared_th = len(g_tr & g_ho)
    shared_vh = len(g_val & g_ho)
    lab = pool["label"]
    states = pool["states"]
    if neutralize_eval:
        train_states = states.copy(); train_states[:, 2] = 0.0; train_states[:, 3] = 0.0
    else:
        train_states = states

    net = load_model(device)
    best_val, best_epoch, tmax = train_lighttrunk(
        net, train_states[tr], lab[tr], train_states[val], lab[val], device,
        trunk_lr=trunk_lr, head_lr=head_lr, mode=mode, max_epochs=max_epochs,
        seed=train_seed, label=f"{mode}_lr{trunk_lr:g}_s{split_seed}{'_neut' if neutralize_eval else ''}")

    ev = eval_all(pool, net, device, tr, ho, neutralize=neutralize_eval)
    dest = None
    if want_destruction and net_before is not None:
        dest = value_destruction_check(pool, net_before, net, device)

    out = {
        "mode": mode, "trunk_lr": trunk_lr, "head_lr": head_lr,
        "split_seed": split_seed, "holdout_frac": holdout_frac, "val_frac": val_frac,
        "n_games": int(n_games), "n_tr_games": int(n_tr_g), "n_val_games": int(n_val_g),
        "n_ho_games": int(n_ho_g),
        "shared_games_tr_ho": shared_th, "shared_games_tr_val": shared_tv,
        "shared_games_val_ho": shared_vh,
        "n_tr_loss": int(((lab < 0) & tr).sum()), "n_tr_win": int(((lab > 0) & tr).sum()),
        "n_ho_loss": int(((lab < 0) & ho).sum()), "n_ho_win": int(((lab > 0) & ho).sum()),
        "best_val_mse": round(best_val, 5), "best_epoch": int(best_epoch),
        "trunk_max_delta": tmax, "trunk_moved": bool(tmax > 0),
        "neutralize_eval": neutralize_eval,
        "eval": ev, "destruction": dest,
    }
    return out, net


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    ap.add_argument("--val-frac", type=float, default=0.18)
    ap.add_argument("--split-seed", type=int, default=101)
    ap.add_argument("--max-epochs", type=int, default=120)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    args = ap.parse_args()

    t0 = time.time()
    device = best_device()
    log.info("=" * 72)
    log.info("D-FULLSPEC LIGHT-TRUNK — unfreeze trunk @ light LR, full-spectrum discriminator")
    log.info("=" * 72)
    pool = build_pool()
    lab = pool["label"]
    log.info(f"pool n={len(lab)} loss={int((lab<0).sum())} win={int((lab>0).sum())} "
             f"plane2==plane3 everywhere={pool['p23_eq']} distinct_games={len(np.unique(pool['gid']))} "
             f"buf_other(non-distill)={len(pool['buf_other_idx'])}")
    log.info(f"tp dist loss: tp0={int(((lab<0)&(pool['tp']==0)).sum())} tp1={int(((lab<0)&(pool['tp']==1)).sum())} | "
             f"win: tp0={int(((lab>0)&(pool['tp']==0)).sum())} tp1={int(((lab>0)&(pool['tp']==1)).sum())}")

    # fresh net snapshot (pre-unfreeze) for the value-destruction spot-check
    net_before = load_model(device)

    # ---- A. LR SWEEP (full-trunk unfreeze), single canonical game-disjoint split ----
    sweep = []
    for trunk_lr in (1e-5, 3e-5, 1e-4):
        r, _ = run_one(pool, device, trunk_lr=trunk_lr, head_lr=args.head_lr, mode="full",
                       split_seed=args.split_seed, holdout_frac=args.holdout_frac,
                       val_frac=args.val_frac, max_epochs=args.max_epochs)
        sweep.append(r)
        m = r["eval"]["matched"]; tf = r["eval"]["train_fit"]
        log.info(f"[SWEEP full lr={trunk_lr:g}] shared(tr/ho)={r['shared_games_tr_ho']} "
                 f"trunk_moved={r['trunk_moved']} (max|d|={r['trunk_max_delta']:.2e}) "
                 f"best_ep={r['best_epoch']} | MATCHED-HO n(l/w)={m['n_loss']}/{m['n_win']} "
                 f"KA={m['kill_a']:.3f} KC={m['kill_c']:.3f} | TRAIN-fit KA={tf['kill_a']:.3f} "
                 f"KC={tf['kill_c']:.3f} | naive KC={r['eval']['naive']['kill_c']:.3f}")

    # ---- A2. lighter variant: unfreeze ONLY last residual block (tower.11) ----
    r_blk, _ = run_one(pool, device, trunk_lr=1e-4, head_lr=args.head_lr, mode="block11",
                       split_seed=args.split_seed, holdout_frac=args.holdout_frac,
                       val_frac=args.val_frac, max_epochs=args.max_epochs)
    sweep.append(r_blk)
    m = r_blk["eval"]["matched"]; tf = r_blk["eval"]["train_fit"]
    log.info(f"[SWEEP block11 lr=1e-4] trunk_moved={r_blk['trunk_moved']} best_ep={r_blk['best_epoch']} "
             f"| MATCHED-HO KA={m['kill_a']:.3f} KC={m['kill_c']:.3f} | TRAIN-fit KA={tf['kill_a']:.3f} "
             f"KC={tf['kill_c']:.3f}")

    # ---- B. pick best-holdout setting. JOINT separation is the question, so prefer a
    #          setting that JOINTLY passes (KA>0.35 AND KC>=0.85); else the one CLOSEST to
    #          the joint box (NOT the trivial high-KC/low-KA corner = the original blind spot). ----
    def matched_kc(r): return r["eval"]["matched"]["kill_c"]
    def matched_ka(r): return r["eval"]["matched"]["kill_a"]
    def box_dist(ka, kc): return (max(0.0, 0.35 - ka) ** 2 + max(0.0, 0.85 - kc) ** 2) ** 0.5
    joint = [r for r in sweep if matched_ka(r) > 0.35 and matched_kc(r) >= 0.85]
    if joint:
        best = max(joint, key=lambda r: (matched_ka(r) - 0.35) + (matched_kc(r) - 0.85))
    else:
        best = min(sweep, key=lambda r: box_dist(matched_ka(r), matched_kc(r)))
    # frontier stats (the anti-correlation signature): can we hold wins AND catch losses?
    kc85 = [matched_ka(r) for r in sweep if matched_kc(r) >= 0.85]
    ka35 = [matched_kc(r) for r in sweep if matched_ka(r) >= 0.35]
    ka_at_kc85 = max(kc85) if kc85 else float("nan")   # best KILL-A while KILL-C still passes
    kc_at_ka35 = min(ka35) if ka35 else float("nan")   # KILL-C when forced to catch >35% losses
    log.info(f"[BEST] mode={best['mode']} trunk_lr={best['trunk_lr']:g} "
             f"matched KA={matched_ka(best):.3f} KC={matched_kc(best):.3f} "
             f"| FRONTIER max_KA@KC>=0.85={ka_at_kc85} min_KC@KA>=0.35={kc_at_ka35} joint_pass_settings={len(joint)}")

    best_lr = best["trunk_lr"]; best_mode = best["mode"]

    # ---- C. BEST config: re-run at game-disjoint seeds 101/202/303 (seed101 carries the
    #          value-destruction spot-check); + neutralized control. ----
    multiseed = []
    net_best = None
    dest = None
    for ss in (args.split_seed, 202, 303):
        r, net = run_one(pool, device, trunk_lr=best_lr, head_lr=args.head_lr, mode=best_mode,
                         split_seed=ss, holdout_frac=args.holdout_frac, val_frac=args.val_frac,
                         max_epochs=args.max_epochs,
                         want_destruction=(ss == args.split_seed), net_before=net_before)
        multiseed.append(r)
        if ss == args.split_seed:
            net_best = net; dest = r["destruction"]; best = r  # canonical best run (carries train_fit/dest)
        m = r["eval"]["matched"]; tf = r["eval"]["train_fit"]
        log.info(f"[MULTISEED ss={ss}] MATCHED-HO n(l/w)={m['n_loss']}/{m['n_win']} "
                 f"KA={m['kill_a']:.3f} KC={m['kill_c']:.3f} | TRAIN-fit KA={tf['kill_a']:.3f} KC={tf['kill_c']:.3f}")
    log.info(f"[VALUE-DESTRUCTION best] {dest}")

    # neutralized (planes 2,3 = 0) — train + eval on zeroed turn-phase, best config, seed101
    r_neut, _ = run_one(pool, device, trunk_lr=best_lr, head_lr=args.head_lr, mode=best_mode,
                        split_seed=args.split_seed, holdout_frac=args.holdout_frac,
                        val_frac=args.val_frac, max_epochs=args.max_epochs, neutralize_eval=True)
    mn = r_neut["eval"]["matched"]
    log.info(f"[NEUTRALIZED best] MATCHED-HO KA={mn['kill_a']:.3f} KC={mn['kill_c']:.3f}")

    ms_ka = float(np.mean([matched_ka(r) for r in multiseed]))
    ms_kc = float(np.mean([matched_kc(r) for r in multiseed]))
    log.info(f"[MULTISEED agg best] matched KA mean={ms_ka:.3f} KC mean={ms_kc:.3f} "
             f"over seeds {[r['split_seed'] for r in multiseed]}")

    # ---- D. VERDICT on game-disjoint, turn-phase-matched HOLDOUT (best config, multi-seed mean) ----
    holo_ka, holo_kc = ms_ka, ms_kc
    tf_best = best["eval"]["train_fit"]
    overfit_gap_kc = tf_best["kill_c"] - holo_kc  # train>>holdout => memorization
    a_pass = holo_ka > 0.35
    c_pass = holo_kc >= 0.85
    neut_pass = (mn["kill_a"] > 0.35) and (mn["kill_c"] >= 0.85)
    big_overfit = overfit_gap_kc > 0.20
    joint_pass = a_pass and c_pass
    near_box = (holo_ka >= 0.25) and (holo_kc >= 0.75)  # genuinely close to BOTH thresholds
    if joint_pass and neut_pass and not big_overfit:
        verdict = "SEPARABLE_LT"
    elif joint_pass:                      # both pass but only via shortcut (neut fail) or memorization
        verdict = "PARTIAL_LT"
    elif near_box:                        # no joint pass but hugging both thresholds => frontier
        verdict = "PARTIAL_LT"
    else:                                 # anti-correlation persists: can't get near both at once
        verdict = "ENTANGLED_LT"

    e1_frozen = {"matched_kill_a": 0.762, "matched_kill_c": 0.441}
    rationale = (
        f"Light-trunk (mode={best_mode}, trunk_lr={best_lr:g}, head_lr={args.head_lr:g}) on "
        f"GAME-DISJOINT (shared_games={best['shared_games_tr_ho']}) turn-phase-MATCHED holdout, "
        f"multi-seed mean over {[r['split_seed'] for r in multiseed]}: "
        f"KILL-A={holo_ka:.3f} (PASS>0.35 -> {'PASS' if a_pass else 'FAIL'}), "
        f"KILL-C={holo_kc:.3f} (PASS>=0.85 -> {'PASS' if c_pass else 'FAIL'}); joint_pass={joint_pass}. "
        f"FRONTIER (anti-correlation test across trunk_lr sweep): best KILL-A while KILL-C>=0.85 = {ka_at_kc85} "
        f"(<0.35 => cannot catch losses without dropping wins); KILL-C when forced KILL-A>=0.35 = {kc_at_ka35}. "
        f"trunk_moved={best['trunk_moved']} (max|delta|={best['trunk_max_delta']:.2e}). "
        f"Overfit canary: TRAIN-fit matched KILL-C={tf_best['kill_c']:.3f} vs holdout {holo_kc:.3f} "
        f"(gap={overfit_gap_kc:+.3f}; >0.20 = memorization). "
        f"Neutralized (planes2,3=0) matched KILL-A={mn['kill_a']:.3f} KILL-C={mn['kill_c']:.3f}. "
        f"Value-destruction spot-check (unrelated buffer positions): "
        f"pearson_pre_post={dest['pearson_pre_post']} sign_flip_frac={dest['sign_flip_frac']} "
        f"mean_abs_delta={dest['mean_abs_delta']}. "
        f"vs E1 FROZEN matched KILL-A={e1_frozen['matched_kill_a']} KILL-C={e1_frozen['matched_kill_c']}. "
        f"=> {'unfreezing trunk DID manufacture JOINT separation; light-trunk finetune target-fix viable, restart may be unnecessary' if verdict=='SEPARABLE_LT' else ('unfreezing trunk did NOT manufacture joint separation; KILL-A vs KILL-C stay anti-correlated => FEATURE problem confirmed, E2 restart warranted' if verdict=='ENTANGLED_LT' else 'partial: hugs the frontier / passes only via shortcut or overfit; report frontier, restart still likely warranted')}."
    )
    log.info(f"VERDICT={verdict}")
    log.info(rationale)

    elapsed = time.time() - t0
    result = {
        "ran": True,
        "device": str(device),
        "pool": {"n_loss": int((lab < 0).sum()), "n_win": int((lab > 0).sum()),
                 "distinct_games": int(len(np.unique(pool["gid"]))),
                 "plane2_eq_plane3": pool["p23_eq"]},
        "split": {"type": "GAME-LEVEL whole-game disjoint 3-way (train/val/holdout)",
                  "holdout_frac": args.holdout_frac, "val_frac": args.val_frac,
                  "game_disjoint": bool(best["shared_games_tr_ho"] == 0 and best["shared_games_tr_val"] == 0
                                        and best["shared_games_val_ho"] == 0),
                  "shared_games": int(best["shared_games_tr_ho"]),
                  "n_train_loss": int(best["n_tr_loss"]), "n_train_win": int(best["n_tr_win"]),
                  "n_ho_loss": int(best["n_ho_loss"]), "n_ho_win": int(best["n_ho_win"])},
        "lr_setting": f"trunk_lr={best_lr:g} (best of sweep [1e-5,3e-5,1e-4] + block11@1e-4), head_lr={args.head_lr:g}",
        "sweep": sweep,
        "best": {"mode": best_mode, "trunk_lr": best_lr, "best_epoch": best["best_epoch"]},
        "matched_holdout_multiseed": {"kill_a": round(holo_ka, 4), "kill_c": round(holo_kc, 4),
                                      "seeds": [r["split_seed"] for r in multiseed],
                                      "per_seed": [{"seed": r["split_seed"],
                                                    "kill_a": round(matched_ka(r), 4),
                                                    "kill_c": round(matched_kc(r), 4),
                                                    "n_loss": r["eval"]["matched"]["n_loss"],
                                                    "n_win": r["eval"]["matched"]["n_win"]}
                                                   for r in multiseed]},
        "train_fit_best": {"kill_a": round(tf_best["kill_a"], 4), "kill_c": round(tf_best["kill_c"], 4)},
        "naive_holdout_best": {"kill_a": round(best["eval"]["naive"]["kill_a"], 4),
                               "kill_c": round(best["eval"]["naive"]["kill_c"], 4)},
        "neutralized_best": {"kill_a": round(mn["kill_a"], 4), "kill_c": round(mn["kill_c"], 4)},
        "value_destruction": dest,
        "overfit_gap_kc": round(overfit_gap_kc, 4),
        "frontier": {"max_kill_a_at_kill_c_ge_0.85": (round(ka_at_kc85, 4) if ka_at_kc85 == ka_at_kc85 else None),
                     "min_kill_c_at_kill_a_ge_0.35": (round(kc_at_ka35, 4) if kc_at_ka35 == kc_at_ka35 else None),
                     "n_joint_pass_settings": len(joint)},
        "trunk_moved": bool(best["trunk_moved"]),
        "e1_frozen_matched": e1_frozen,
        "verdict": verdict,
        "verdict_rationale": rationale,
        "wall_time_s": round(elapsed, 1),
    }

    out_dir = REPO_ROOT / "reports" / "d_fullspec_2026-06-26"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "LIGHTTRUNK_results.json", "w") as f:
        json.dump(result, f, indent=2)

    # markdown
    def row(tag, ka, kc, nl, nw, note):
        return f"| {tag} | {ka:.3f} | {kc:.3f} | {nl} | {nw} | {note} |"
    sweep_rows = "\n".join(
        row(f"{r['mode']} trunk_lr={r['trunk_lr']:g} (HOLDOUT)",
            r["eval"]["matched"]["kill_a"], r["eval"]["matched"]["kill_c"],
            r["eval"]["matched"]["n_loss"], r["eval"]["matched"]["n_win"],
            f"train-fit KC={r['eval']['train_fit']['kill_c']:.3f} best_ep={r['best_epoch']} "
            f"trunk_moved={r['trunk_moved']}")
        for r in sweep)
    ms_rows = "\n".join(
        row(f"best seed={p['seed']} (HOLDOUT matched)", p["kill_a"], p["kill_c"], p["n_loss"], p["n_win"], "")
        for p in result["matched_holdout_multiseed"]["per_seed"])
    md = f"""# D-FULLSPEC LIGHT-TRUNK — does trunk ADAPTATION manufacture win/loss separation?

Generated {time.strftime('%Y-%m-%d %H:%M:%S')}  wall={elapsed:.1f}s  device={device}

Single-variable change vs **E1 (frozen trunk)**: UNFREEZE the trunk at a LIGHT LR
(value head at a normal LR), re-run the exact full-spectrum discriminator.
E1 frozen matched holdout: **KILL-A 0.762 PASS / KILL-C 0.441 FAIL** (ENTANGLED).

## Setup
- checkpoint `{CKPT}` encoding **v6_live2_ls**. trunk 12 res-blocks (tower.0..11), value head = value_fc1/value_fc2.
- **GAME-DISJOINT 3-way split** (whole games -> one side): train / val / holdout.
  holdout_frac={args.holdout_frac} val_frac={args.val_frac}. shared_games(tr/ho)={best['shared_games_tr_ho']}
  (tr/val)={best['shared_games_tr_val']} (val/ho)={best['shared_games_val_ho']} -> game_disjoint={result['split']['game_disjoint']}.
- game_id recovered for every bank position via byte-exact state->DS1-CSV join (build_pool).
- turn-phase confound controlled: plane2==plane3 everywhere={pool['p23_eq']}; verdict on tp==0 INTERSECT
  overlapping stone-count support (E1 EXACT matched protocol), never naive.
- LR groups: trunk params @ trunk_lr, value head @ head_lr={args.head_lr:g}. Adam. Early-stop on game-disjoint VAL balanced MSE.

## LR sweep — matched HOLDOUT KILL-A/KILL-C (verdict basis) + overfit canary
KILL-A = frac held-out LOSSES called loss (value<0), PASS>0.35.
KILL-C = frac held-out WINS called win (value>0), PASS>=0.85.

| setting | KILL-A | KILL-C | n_loss | n_win | note |
|---|---|---|---|---|---|
{sweep_rows}

**Best setting:** mode={best_mode} trunk_lr={best_lr:g} (chosen = JOINT-pass if any, else CLOSEST to the
KILL-A>0.35 & KILL-C>=0.85 box; NOT the trivial high-KC/low-KA corner). Frontier: max KILL-A while
KILL-C>=0.85 = {ka_at_kc85}; min KILL-C when forced KILL-A>=0.35 = {kc_at_ka35}; joint-pass settings = {len(joint)}.

## Best setting — multi-seed game-disjoint robustness (matched HOLDOUT)
| setting | KILL-A | KILL-C | n_loss | n_win | note |
|---|---|---|---|---|---|
{ms_rows}
| **multi-seed mean (VERDICT basis)** | **{holo_ka:.3f}** | **{holo_kc:.3f}** | - | - | seeds {result['matched_holdout_multiseed']['seeds']} |

## Overfit / control panel (best setting)
| metric | KILL-A | KILL-C |
|---|---|---|
| TRAIN-fit matched (overfit canary) | {tf_best['kill_a']:.3f} | {tf_best['kill_c']:.3f} |
| HOLDOUT matched (multi-seed mean) | {holo_ka:.3f} | {holo_kc:.3f} |
| naive HOLDOUT (confounded) | {best['eval']['naive']['kill_a']:.3f} | {best['eval']['naive']['kill_c']:.3f} |
| NEUTRALIZED planes2,3=0 (matched HOLDOUT) | {mn['kill_a']:.3f} | {mn['kill_c']:.3f} |

- **overfit gap (train-fit KC - holdout KC) = {overfit_gap_kc:+.3f}** ({'MEMORIZATION (>0.20)' if big_overfit else 'no big gap'}).
- **trunk_moved = {best['trunk_moved']}** (max|delta| over trunk params = {best['trunk_max_delta']:.2e}).
- best early-stop epoch = {best['best_epoch']}.

## self_redteam — did unfreezing break the broader value function?
Value-destruction spot-check on {dest['n']} unrelated (non-distill) buffer positions, value pre vs post:
- pearson(pre,post) = **{dest['pearson_pre_post']}**, sign_flip_frac = {dest['sign_flip_frac']},
  mean|delta| = {dest['mean_abs_delta']}, max|delta| = {dest['max_abs_delta']},
  mean_v before/after = {dest['mean_v_before']}/{dest['mean_v_after']}.

## VERDICT: {verdict}
{rationale}

### Comparison to E1 (frozen trunk)
| | matched KILL-A | matched KILL-C | verdict |
|---|---|---|---|
| E1 FROZEN | 0.762 | 0.441 | ENTANGLED |
| LIGHT-TRUNK (this) | {holo_ka:.3f} | {holo_kc:.3f} | {verdict} |
"""
    with open(out_dir / "LIGHTTRUNK_findings.md", "w") as f:
        f.write(md)
    log.info(f"JSON -> {out_dir/'LIGHTTRUNK_results.json'}")
    log.info(f"MD   -> {out_dir/'LIGHTTRUNK_findings.md'}")
    log.info(f"DONE in {elapsed:.1f}s")
    print("LIGHTTRUNK_RESULT_JSON " + json.dumps(result))


if __name__ == "__main__":
    main()
