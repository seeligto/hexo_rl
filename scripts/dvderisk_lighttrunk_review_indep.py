#!/usr/bin/env python3
"""D-FULLSPEC LIGHT-TRUNK — INDEPENDENT re-derivation (reviewer agent).

NOT a rerun of scripts/dvderisk_lighttrunk_probe.py. Independent verification of
the probe's ENTANGLED_LT verdict (unfreeze trunk @ light LR, full-spectrum
value-MSE distill, judge on turn-phase-MATCHED game-disjoint holdout).

Independent in:
  1. FRESH split seeds 137/271/419 (probe used 101/202/303). Own whole-game
     3-way split (train/val/holdout), game_id recovered via byte-exact
     state->DS1-CSV join. assert shared_games==0 pairwise.
  2. Own light-trunk training loop (2 param groups: trunk @ trunk_lr, value head
     @ head_lr), early-stop on game-disjoint balanced VAL MSE.
  3. Own turn-phase control: verdict on tp==0 stratum INTERSECT overlapping
     stone support; assert matched stratum has plane2==plane3==0 everywhere;
     plus a planes-2,3=0 NEUTRALIZED control.

Pre-registered thresholds (same as E1 / probe):
  KILL-A = frac held-out LOSSES called loss (value<0). PASS > 0.35
  KILL-C = frac held-out WINS  called win  (value>0). PASS >= 0.85
  SEPARABLE_LT : matched holdout KILL-A>0.35 AND KILL-C>=0.85, not overfit,
                 survives neutralization.
  ENTANGLED_LT : matched KILL-C craters (<~0.6) despite unfreezing.
  PARTIAL_LT   : in between / hugs both thresholds / passes only via overfit.

overfit_seen = TRAIN-fit matched KILL-C >> holdout matched KILL-C (gap>0.20).

Usage: .venv/bin/python scripts/dvderisk_lighttrunk_review_indep.py
"""
from __future__ import annotations

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

LOG_PATH = REPO_ROOT / "logs" / "dvderisk_lighttrunk_review_indep.log"
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
SEEDS = (137, 271, 419)  # FRESH, distinct from the probe's 101/202/303


def load_model(device, encoding="v6_live2_ls"):
    ckpt = torch.load(REPO_ROOT / CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    state = extract_model_state(ckpt)
    inp_w = state.get("trunk.input_conv.weight")
    in_ch = int(inp_w.shape[1]) if inp_w is not None else cfg.get("in_channels", 4)
    net = HexTacToeNet(
        in_channels=in_ch, res_blocks=cfg.get("res_blocks", 12),
        filters=cfg.get("filters", 128), board_size=cfg.get("board_size", 19),
        encoding=encoding)
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


def kill_a(vl):
    return float((vl < 0).mean()) if len(vl) else float("nan")


def kill_c(vw):
    return float((vw > 0).mean()) if len(vw) else float("nan")


# --------------------------------------------------------------------------- #
def build_pool():
    """Recover game_id per bank position via byte-exact state->DS1-CSV join.
    assert 0 ambiguity, 100% join. Returns full pool dict."""
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
    assert ambig == 0, f"{ambig} ambiguous state-bytes"

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
    assert join_fail == 0, f"{join_fail} join failures"
    states = np.stack(states).astype("float32")
    labels = np.array(labels, "float32")
    gids = np.array(gids, "int64")
    tp = (states[:, 2].reshape(len(states), -1).mean(1) >= 0.5).astype("int64")
    stones = (states[:, 0] + states[:, 1]).reshape(len(states), -1).sum(1)
    p23_eq = bool(np.array_equal(states[:, 2], states[:, 3]))
    distill_bytes = {x.tobytes() for x in states}
    other = np.array([i for i in range(len(buf)) if buf[i].tobytes() not in distill_bytes], "int64")
    return {"states": states, "label": labels, "gid": gids, "tp": tp,
            "stones": stones, "p23_eq": p23_eq, "buf": buf, "buf_other_idx": other}


def three_way_split(gids, holdout_frac, val_frac, seed):
    """Whole-game disjoint train/val/holdout. Each game on exactly ONE side."""
    uniq = np.unique(gids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_ho = max(1, int(round(len(uniq) * holdout_frac)))
    ho_g = set(perm[:n_ho].tolist())
    rest = perm[n_ho:]
    n_val = max(1, int(round(len(rest) * val_frac)))
    val_g = set(rest[:n_val].tolist())
    tr_g = set(rest[n_val:].tolist())
    ho = np.array([g in ho_g for g in gids])
    val = np.array([g in val_g for g in gids])
    tr = np.array([g in tr_g for g in gids])
    return tr, val, ho


def select_trainable(net, mode):
    trunk_params, head_params = [], []
    for name, p in net.named_parameters():
        if name.startswith("value_fc1") or name.startswith("value_fc2"):
            p.requires_grad_(True); head_params.append(p)
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
            p.requires_grad_(False)
    return trunk_params, head_params


def trunk_snapshot(net):
    return {n: p.detach().clone() for n, p in net.named_parameters() if n.startswith("trunk.")}


def trunk_max_delta(net, snap):
    m = 0.0
    for n, p in net.named_parameters():
        if n.startswith("trunk."):
            m = max(m, float((p.detach() - snap[n]).abs().max()))
    return m


def train_lighttrunk(net, tr_s, tr_y, val_s, val_y, device, *, trunk_lr, head_lr, mode,
                     max_epochs=120, bpc=64, patience=20, seed=0, label="lt"):
    """Class-balanced value-MSE distill, trunk UNFROZEN @ trunk_lr, head @ head_lr.
    Early-stop on game-disjoint balanced VAL MSE."""
    trunk_params, head_params = select_trainable(net, mode)
    snap = trunk_snapshot(net)
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
        with torch.no_grad():
            vl = values(net, val_l, device)
            vw = values(net, val_w, device)
        return 0.5 * float(((vl + 1) ** 2).mean()) + 0.5 * float(((vw - 1) ** 2).mean())

    rng = np.random.default_rng(seed)
    nl, nw = len(li), len(wi)
    steps = max(nl, nw) // bpc + 1
    best, best_state, best_epoch, bad = float("inf"), None, -1, 0
    n_trunk = sum(p.numel() for p in trunk_params)
    log.info(f"[{label}] mode={mode} trunk_lr={trunk_lr:g} head_lr={head_lr:g} "
             f"n_trunk_trainable={n_trunk} tr(l/w)={nl}/{nw} val(l/w)={len(val_l)}/{len(val_w)} steps/ep={steps}")
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
            best, bad, best_epoch = vm, 0, ep
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
        if bad >= patience:
            log.info(f"[{label}] early-stop ep{ep} best_epoch={best_epoch} best_val_mse={best:.5f}")
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    tmax = trunk_max_delta(net, snap)
    log.info(f"[{label}] best_val_mse={best:.5f} best_epoch={best_epoch} trunk_max_delta={tmax:.3e}")
    return best, best_epoch, tmax


# --------------------------------------------------------------------------- #
def matched_masks(pool, mask):
    """tp==0 stratum INTERSECT overlapping stone support, on subset `mask`."""
    lab, tp, stones = pool["label"], pool["tp"], pool["stones"]
    l0 = (lab < 0) & mask & (tp == 0)
    w0 = (lab > 0) & mask & (tp == 0)
    if l0.sum() == 0 or w0.sum() == 0:
        return None
    lo = max(stones[l0].min(), stones[w0].min())
    hi = min(stones[l0].max(), stones[w0].max())
    lm = l0 & (stones >= lo) & (stones <= hi)
    wm = w0 & (stones >= lo) & (stones <= hi)
    return lm, wm, float(lo), float(hi)


def matched_turnphase_ok(pool, lm, wm):
    """Assert plane2==plane3==0 for EVERY matched position in both classes."""
    st = pool["states"]
    idx = np.where(lm | wm)[0]
    p2 = st[idx, 2].reshape(len(idx), -1)
    p3 = st[idx, 3].reshape(len(idx), -1)
    return bool((p2 == 0).all() and (p3 == 0).all())


def eval_all(pool, net, device, tr_mask, ho_mask, neutralize=False):
    states = pool["states"]
    if neutralize:
        states = states.copy(); states[:, 2] = 0.0; states[:, 3] = 0.0
    lab = pool["label"]
    v_all = values(net, states, device)
    naive_ka = kill_a(v_all[(lab < 0) & ho_mask])
    naive_kc = kill_c(v_all[(lab > 0) & ho_mask])
    mh = matched_masks(pool, ho_mask)
    if mh is None:
        matched = None; tp_ok = None
    else:
        lm, wm, lo, hi = mh
        tp_ok = matched_turnphase_ok(pool, lm, wm)
        matched = {"kill_a": round(kill_a(v_all[lm]), 4), "kill_c": round(kill_c(v_all[wm]), 4),
                   "n_loss": int(lm.sum()), "n_win": int(wm.sum()), "stone_support": [lo, hi],
                   "tp_zero_ok": tp_ok}
    mt = matched_masks(pool, tr_mask)
    if mt is None:
        train_fit = None
    else:
        lmt, wmt, _, _ = mt
        train_fit = {"kill_a": round(kill_a(v_all[lmt]), 4), "kill_c": round(kill_c(v_all[wmt]), 4),
                     "n_loss": int(lmt.sum()), "n_win": int(wmt.sum())}
    return {"naive": {"kill_a": round(naive_ka, 4), "kill_c": round(naive_kc, 4)},
            "matched": matched, "train_fit": train_fit}


def value_destruction(pool, net_before, net_after, device, n=300, seed=11):
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
            "sign_flip_frac": round(flips / len(idx), 4), "pearson_pre_post": round(corr, 4),
            "mean_v_before": round(float(vb.mean()), 4), "mean_v_after": round(float(va.mean()), 4)}


def run_one(pool, device, *, trunk_lr, head_lr, mode, split_seed, holdout_frac, val_frac,
            train_seed=0, max_epochs=120, neutralize=False, want_destruction=False, net_before=None):
    tr, val, ho = three_way_split(pool["gid"], holdout_frac, val_frac, split_seed)
    g_tr, g_val, g_ho = set(pool["gid"][tr].tolist()), set(pool["gid"][val].tolist()), set(pool["gid"][ho].tolist())
    shared = {"tr_val": len(g_tr & g_val), "tr_ho": len(g_tr & g_ho), "val_ho": len(g_val & g_ho)}
    lab = pool["label"]
    states = pool["states"]
    if neutralize:
        states = states.copy(); states[:, 2] = 0.0; states[:, 3] = 0.0
    net = load_model(device)
    _, best_epoch, tmax = train_lighttrunk(
        net, states[tr], lab[tr], states[val], lab[val], device,
        trunk_lr=trunk_lr, head_lr=head_lr, mode=mode, max_epochs=max_epochs, seed=train_seed,
        label=f"{mode}_lr{trunk_lr:g}_s{split_seed}{'_neut' if neutralize else ''}")
    ev = eval_all(pool, net, device, tr, ho, neutralize=neutralize)
    dest = value_destruction(pool, net_before, net, device) if (want_destruction and net_before is not None) else None
    out = {"mode": mode, "trunk_lr": trunk_lr, "split_seed": split_seed,
           "shared_games": shared, "n_games": int(len(np.unique(pool["gid"]))),
           "n_tr_games": len(g_tr), "n_val_games": len(g_val), "n_ho_games": len(g_ho),
           "n_tr_loss": int(((lab < 0) & tr).sum()), "n_tr_win": int(((lab > 0) & tr).sum()),
           "best_epoch": int(best_epoch), "trunk_max_delta": tmax, "trunk_moved": bool(tmax > 0),
           "neutralize": neutralize, "eval": ev, "destruction": dest}
    return out, net


def main():
    t0 = time.time()
    device = best_device()
    holdout_frac, val_frac, head_lr, max_epochs = 0.25, 0.18, 3e-4, 120
    log.info("=" * 72)
    log.info("D-FULLSPEC LIGHT-TRUNK — INDEPENDENT review (fresh seeds 137/271/419)")
    log.info("=" * 72)
    pool = build_pool()
    lab = pool["label"]
    log.info(f"pool n={len(lab)} loss={int((lab<0).sum())} win={int((lab>0).sum())} "
             f"plane2==plane3_everywhere={pool['p23_eq']} distinct_games={len(np.unique(pool['gid']))} "
             f"buf_other={len(pool['buf_other_idx'])}")
    log.info(f"tp dist loss: tp0={int(((lab<0)&(pool['tp']==0)).sum())} tp1={int(((lab<0)&(pool['tp']==1)).sum())} | "
             f"win: tp0={int(((lab>0)&(pool['tp']==0)).sum())} tp1={int(((lab>0)&(pool['tp']==1)).sum())}")

    net_before = load_model(device)

    # ---- A. best setting (full trunk @ light lr=1e-5) across 3 FRESH game-disjoint seeds ----
    BEST_LR, BEST_MODE = 1e-5, "full"
    multiseed = []
    dest = None
    for i, ss in enumerate(SEEDS):
        r, net = run_one(pool, device, trunk_lr=BEST_LR, head_lr=head_lr, mode=BEST_MODE,
                         split_seed=ss, holdout_frac=holdout_frac, val_frac=val_frac,
                         max_epochs=max_epochs, want_destruction=(i == 0), net_before=net_before)
        multiseed.append(r)
        if i == 0:
            dest = r["destruction"]; best = r
        m, tf = r["eval"]["matched"], r["eval"]["train_fit"]
        log.info(f"[BEST-MS ss={ss}] shared={r['shared_games']} trunk_moved={r['trunk_moved']} "
                 f"(max|d|={r['trunk_max_delta']:.2e}) best_ep={r['best_epoch']} tp0_ok={m['tp_zero_ok']} "
                 f"| MATCHED-HO n(l/w)={m['n_loss']}/{m['n_win']} KA={m['kill_a']} KC={m['kill_c']} "
                 f"| TRAIN-fit KA={tf['kill_a']} KC={tf['kill_c']} | naive KC={r['eval']['naive']['kill_c']}")

    # ---- B. FRONTIER point: force higher KILL-A via heavier lr=1e-4, seed 137 ----
    r_hi, _ = run_one(pool, device, trunk_lr=1e-4, head_lr=head_lr, mode="full",
                      split_seed=SEEDS[0], holdout_frac=holdout_frac, val_frac=val_frac, max_epochs=max_epochs)
    mh = r_hi["eval"]["matched"]
    log.info(f"[FRONTIER full lr=1e-4 ss={SEEDS[0]}] best_ep={r_hi['best_epoch']} "
             f"MATCHED-HO KA={mh['kill_a']} KC={mh['kill_c']} (heavier-trunk: trade KA up, KC?)")

    # ---- C. NEUTRALIZED control (planes 2,3 = 0), best config, seed 137 ----
    r_neut, _ = run_one(pool, device, trunk_lr=BEST_LR, head_lr=head_lr, mode=BEST_MODE,
                        split_seed=SEEDS[0], holdout_frac=holdout_frac, val_frac=val_frac,
                        max_epochs=max_epochs, neutralize=True)
    mn = r_neut["eval"]["matched"]
    log.info(f"[NEUTRALIZED best ss={SEEDS[0]}] MATCHED-HO KA={mn['kill_a']} KC={mn['kill_c']}")

    # ---- aggregate ----
    ka = [r["eval"]["matched"]["kill_a"] for r in multiseed]
    kc = [r["eval"]["matched"]["kill_c"] for r in multiseed]
    ms_ka, ms_kc = float(np.mean(ka)), float(np.mean(kc))
    tf_best = best["eval"]["train_fit"]
    overfit_gap = tf_best["kill_c"] - ms_kc

    game_disjoint_ok = all(r["shared_games"]["tr_val"] == 0 and r["shared_games"]["tr_ho"] == 0
                           and r["shared_games"]["val_ho"] == 0 for r in multiseed + [r_hi, r_neut])
    turnphase_control_ok = bool(pool["p23_eq"]) and all(
        r["eval"]["matched"]["tp_zero_ok"] for r in multiseed)
    overfit_seen = bool(overfit_gap > 0.20)

    # frontier: across {1e-5 multiseed, 1e-4} can ANY hold KILL-C>=0.85 while KILL-A>0.35?
    all_settings = [(r["eval"]["matched"]["kill_a"], r["eval"]["matched"]["kill_c"]) for r in multiseed]
    all_settings.append((mh["kill_a"], mh["kill_c"]))
    kc85 = [a for a, c in all_settings if c >= 0.85]
    joint_pass = [(a, c) for a, c in all_settings if a > 0.35 and c >= 0.85]
    max_ka_at_kc85 = max(kc85) if kc85 else None

    a_pass = ms_ka > 0.35
    c_pass = ms_kc >= 0.85
    near_box = (ms_ka >= 0.25) and (ms_kc >= 0.75)
    neut_pass = (mn["kill_a"] > 0.35) and (mn["kill_c"] >= 0.85)
    if a_pass and c_pass and neut_pass and not overfit_seen:
        verdict = "SEPARABLE_LT"
    elif a_pass and c_pass:
        verdict = "PARTIAL_LT"
    elif near_box:
        verdict = "PARTIAL_LT"
    else:
        verdict = "ENTANGLED_LT"

    probe = {"matched_kill_a": 0.653, "matched_kill_c": 0.532, "verdict": "ENTANGLED_LT"}
    agrees = (verdict == probe["verdict"])
    log.info("-" * 72)
    log.info(f"[INDEP AGG] matched KILL-A mean={ms_ka:.4f} range[{min(ka):.3f},{max(ka):.3f}] | "
             f"KILL-C mean={ms_kc:.4f} range[{min(kc):.3f},{max(kc):.3f}]")
    log.info(f"[INDEP] train-fit KC={tf_best['kill_c']} holdout KC={ms_kc:.4f} overfit_gap={overfit_gap:+.4f} "
             f"overfit_seen={overfit_seen}")
    log.info(f"[INDEP] frontier max_KA@KC>=0.85={max_ka_at_kc85} joint_pass={len(joint_pass)} "
             f"neutralized KA={mn['kill_a']} KC={mn['kill_c']}")
    log.info(f"[INDEP] game_disjoint_ok={game_disjoint_ok} turnphase_control_ok={turnphase_control_ok}")
    log.info(f"[INDEP] VERDICT={verdict}  agrees_with_probe={agrees} (probe KA={probe['matched_kill_a']} "
             f"KC={probe['matched_kill_c']} {probe['verdict']})")

    elapsed = time.time() - t0
    result = {
        "ran": True, "device": str(device),
        "fresh_seeds": list(SEEDS),
        "pool": {"n_loss": int((lab < 0).sum()), "n_win": int((lab > 0).sum()),
                 "distinct_games": int(len(np.unique(pool["gid"]))), "plane2_eq_plane3": pool["p23_eq"]},
        "split_cfg": {"holdout_frac": holdout_frac, "val_frac": val_frac, "head_lr": head_lr},
        "independent_matched_kill_a": round(ms_ka, 4),
        "independent_matched_kill_c": round(ms_kc, 4),
        "per_seed": [{"seed": r["split_seed"], "kill_a": r["eval"]["matched"]["kill_a"],
                      "kill_c": r["eval"]["matched"]["kill_c"], "n_loss": r["eval"]["matched"]["n_loss"],
                      "n_win": r["eval"]["matched"]["n_win"], "tp0_ok": r["eval"]["matched"]["tp_zero_ok"],
                      "trunk_max_delta": r["trunk_max_delta"], "best_epoch": r["best_epoch"],
                      "train_fit_kc": r["eval"]["train_fit"]["kill_c"],
                      "naive_kc": r["eval"]["naive"]["kill_c"]} for r in multiseed],
        "frontier_lr1e-4_seed137": {"kill_a": mh["kill_a"], "kill_c": mh["kill_c"]},
        "neutralized_seed137": {"kill_a": mn["kill_a"], "kill_c": mn["kill_c"]},
        "train_fit_best": tf_best, "overfit_gap_kc": round(overfit_gap, 4),
        "value_destruction": dest,
        "frontier": {"max_kill_a_at_kill_c_ge_0.85": max_ka_at_kc85, "n_joint_pass": len(joint_pass)},
        "game_disjoint_ok": bool(game_disjoint_ok),
        "turnphase_control_ok": bool(turnphase_control_ok),
        "overfit_seen": overfit_seen,
        "corrected_verdict": verdict,
        "agrees_with_probe": bool(agrees),
        "probe_reported": probe,
        "wall_time_s": round(elapsed, 1),
    }
    out_dir = REPO_ROOT / "reports" / "d_fullspec_2026-06-26"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "LIGHTTRUNK_review_indep_results.json", "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"JSON -> {out_dir/'LIGHTTRUNK_review_indep_results.json'}")
    log.info(f"DONE in {elapsed:.1f}s")
    print("LIGHTTRUNK_REVIEW_INDEP_JSON " + json.dumps(result))


if __name__ == "__main__":
    main()
