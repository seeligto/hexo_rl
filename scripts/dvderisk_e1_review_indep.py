#!/usr/bin/env python3
"""D-FULLSPEC E1 REVIEW — INDEPENDENT re-derivation of KILL-A / KILL-C.

NOT a rerun of dvderisk_e1_fullspec.py. Independent in three ways:
  1. GAME-LEVEL split with a FRESH seed (not the DS1-CSV train/holdout split).
     game_id recovered for every bank position via a state-bytes join against
     the DS1 buffer (livetail_bank + selfplay_bank concat) + dvderisk_ds1 CSV.
     Whole games go to ONE side => train/holdout share NO game (no leakage),
     and the loss/win classes never cross-leak a game either.
  2. Own class-balanced frozen-trunk training loop (equal loss/win per batch).
  3. Own turn-phase control: matched (tp==0 stratum, where plane2==plane3==0
     for every position) PLUS a stone-count-reweighted matched KILL-C to strip
     residual occupancy confound, PLUS a tp==1 breakdown, PLUS multi-seed
     robustness over distinct game partitions.

Confound: encoding v6_live2 planes 2,3 = moves_remaining_bcast / ply_parity_bcast.
Here they are a single BINARY board-constant scalar (0 or 1), identical per
position. LOSSES ~95% tp=0, WINS ~66% tp=1 (holdout) => a frozen trunk can
separate the classes on tp alone. Verdict judged ONLY on tp==0 (turn-phase held
fixed at 0 for both classes), never naive.

Pre-registered (same thresholds as E1):
  KILL-A = frac held-out LOSSES (-1) called LOSS (value<0). PASS > 0.35
  KILL-C = frac held-out WINS  (+1) called WIN  (value>0). PASS >= 0.85
  SEPARABLE : KILL-A>0.35 AND KILL-C>=0.85 on matched
  PARTIAL   : exactly one passes, the failing one within 0.10 of threshold
  ENTANGLED : the failing metric is a crater (> 0.10 from threshold)

Usage:
    .venv/bin/python scripts/dvderisk_e1_review_indep.py
"""
from __future__ import annotations

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

LOG_PATH = REPO_ROOT / "logs" / "dvderisk_e1_review_indep.log"
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


def freeze_trunk(net):
    for name, p in net.named_parameters():
        p.requires_grad_(not name.startswith("trunk."))


def verify_frozen(net):
    ok = True
    moved = []
    for name, p in net.named_parameters():
        if name.startswith("trunk.") and p.requires_grad:
            log.error(f"FREEZE FAIL {name}"); ok = False
        if (not name.startswith("trunk.")) and p.requires_grad:
            moved.append(name)
    return ok, moved


@torch.no_grad()
def values(net, states, device, bs=256):
    net.eval()
    out = []
    for s in range(0, len(states), bs):
        x = torch.from_numpy(states[s:s + bs]).to(device)
        _, v, _ = net(x)
        out.append(v.squeeze(1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def kill_a(vl):  # frac losses called loss
    return float((vl < 0).mean()) if len(vl) else float("nan")


def kill_c(vw):  # frac wins called win
    return float((vw > 0).mean()) if len(vw) else float("nan")


# ---------------------------------------------------------------------------
def build_pool():
    """Return dict with states/label/game_id/tp/stones for the full 1166-pos pool,
    game_id recovered via state-bytes join. Asserts 100% join, 0 ambiguity."""
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
    tp = (states[:, 2].reshape(len(states), -1).mean(1) >= 0.5).astype("int64")  # binary turn-phase
    stones = (states[:, 0] + states[:, 1]).reshape(len(states), -1).sum(1)
    # confirm plane2==plane3 everywhere
    p23_eq = np.array_equal(states[:, 2], states[:, 3])
    return {"states": states, "label": labels, "gid": gids, "tp": tp,
            "stones": stones, "p23_eq": bool(p23_eq)}


def game_split(gids, holdout_frac, seed):
    """Whole-game split: every game's positions land on ONE side. Returns boolean
    holdout mask. Guarantees train/holdout share no game."""
    uniq = np.unique(gids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_ho = max(1, int(round(len(uniq) * holdout_frac)))
    ho_games = set(perm[:n_ho].tolist())
    return np.array([g in ho_games for g in gids])


def train_head(net, tr_states, tr_label, device, *, lr=3e-4, max_epochs=120,
               bpc=64, patience=15, val_frac=0.15, seed=0, label="indep"):
    """Class-balanced frozen-trunk value-head MSE distill. Optimizer over
    value_fc1/value_fc2 ONLY. Early-stop on BALANCED val MSE."""
    li = np.where(tr_label < 0)[0]
    wi = np.where(tr_label > 0)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(li); rng.shuffle(wi)
    nlv = max(1, int(len(li) * val_frac)); nwv = max(1, int(len(wi) * val_frac))
    val_l, tr_l = tr_states[li[:nlv]], tr_states[li[nlv:]]
    val_w, tr_w = tr_states[wi[:nwv]], tr_states[wi[nwv:]]

    vh = [net.value_fc1.weight, net.value_fc1.bias, net.value_fc2.weight, net.value_fc2.bias]
    opt = torch.optim.Adam(vh, lr=lr)
    mse = nn.MSELoss()

    def val_mse():
        net.eval()
        with torch.no_grad():
            vl = values(net, val_l, device); vw = values(net, val_w, device)
        return 0.5 * float(((vl + 1) ** 2).mean()) + 0.5 * float(((vw - 1) ** 2).mean())

    best = float("inf"); best_state = None; bad = 0
    nl, nw = len(tr_l), len(tr_w)
    steps = max(nl, nw) // bpc + 1
    log.info(f"[{label}] train loss={nl} win={nw} val(l/w)={nlv}/{nwv} steps/ep={steps}")
    for ep in range(max_epochs):
        net.train()
        pl = rng.permutation(nl); pw = rng.permutation(nw)
        for st in range(steps):
            bl = pl[(st * bpc) % nl:][:bpc]; bw = pw[(st * bpc) % nw:][:bpc]
            if len(bl) == 0 or len(bw) == 0:
                continue
            xs = np.concatenate([tr_l[bl], tr_w[bw]])
            ts = np.concatenate([np.full(len(bl), -1.0, "float32"),
                                 np.full(len(bw), 1.0, "float32")])
            opt.zero_grad()
            _, v, _ = net(torch.from_numpy(xs).to(device))
            mse(v.squeeze(1), torch.from_numpy(ts).to(device)).backward()
            opt.step()
        vm = val_mse()
        if vm < best - 1e-5:
            best = vm; bad = 0
            best_state = [p.detach().clone() for p in vh]
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state is not None:
        with torch.no_grad():
            for p, b in zip(vh, best_state):
                p.copy_(b)
    return best


def matched_eval(pool, ho_mask, vl_all, vw_all):
    """Turn-phase-matched KILL-A/KILL-C on holdout: restrict BOTH classes to
    tp==0 (plane2==plane3==0) intersect overlapping stone support. Plus
    stone-reweighted KILL-C and caliper-1:1."""
    lab = pool["label"]; tp = pool["tp"]; stones = pool["stones"]
    is_loss = (lab < 0) & ho_mask
    is_win = (lab > 0) & ho_mask
    # full-holdout value arrays aligned to pool index
    # vl_all/vw_all are per-pool-index values (single array v_all)
    v_all = vl_all  # same array
    l0 = is_loss & (tp == 0)
    w0 = is_win & (tp == 0)
    if l0.sum() == 0 or w0.sum() == 0:
        return None
    lo = max(stones[l0].min(), stones[w0].min())
    hi = min(stones[l0].max(), stones[w0].max())
    lm = l0 & (stones >= lo) & (stones <= hi)
    wm = w0 & (stones >= lo) & (stones <= hi)
    m_ka = kill_a(v_all[lm]); m_kc = kill_c(v_all[wm])

    # stone-reweighted KILL-C: bin wins by stone count, weight to match the loss
    # stone distribution (removes residual occupancy confound within tp==0).
    edges = np.arange(lo - 0.5, hi + 1.5, 4.0)
    lh, _ = np.histogram(stones[lm], bins=edges)
    lh = lh / max(lh.sum(), 1)
    w_stone = stones[wm]; w_val = v_all[wm]
    wbin = np.clip(np.digitize(w_stone, edges) - 1, 0, len(lh) - 1)
    wh_counts = np.array([(wbin == b).sum() for b in range(len(lh))])
    weights = np.zeros(len(w_stone))
    for b in range(len(lh)):
        m = wbin == b
        if wh_counts[b] > 0 and lh[b] > 0:
            weights[m] = lh[b] / wh_counts[b]
    rw_kc = float((weights * (w_val > 0)).sum() / weights.sum()) if weights.sum() > 0 else float("nan")

    # caliper 1:1 stone match within tp==0 (no stone-support pre-trim)
    rng = np.random.default_rng(7)
    l_idx = np.where(l0)[0]; w_idx = np.where(w0)[0]
    avail = list(l_idx); used_l = []; used_w = []
    for wi_ in w_idx:
        cands = [li_ for li_ in avail if abs(stones[li_] - stones[wi_]) <= 2.0]
        if cands:
            pick = min(cands, key=lambda li_: abs(stones[li_] - stones[wi_]))
            avail.remove(pick); used_l.append(pick); used_w.append(wi_)
    cal_ka = kill_a(v_all[used_l]) if used_l else float("nan")
    cal_kc = kill_c(v_all[used_w]) if used_w else float("nan")

    # tp==1 breakdown (where the naive inflation lives)
    w1 = is_win & (tp == 1); l1 = is_loss & (tp == 1)
    return {
        "n_loss": int(lm.sum()), "n_win": int(wm.sum()),
        "stone_support": [float(lo), float(hi)],
        "kill_a": round(m_ka, 4), "kill_c": round(m_kc, 4),
        "kill_c_stone_reweighted": round(rw_kc, 4),
        "caliper_n": len(used_l),
        "caliper_kill_a": round(cal_ka, 4), "caliper_kill_c": round(cal_kc, 4),
        "tp1_n_win": int(w1.sum()), "tp1_kill_c": round(kill_c(v_all[w1]), 4),
        "tp1_n_loss": int(l1.sum()), "tp1_kill_a": round(kill_a(v_all[l1]), 4),
    }


def run_seed(pool, device, split_seed, train_seed, holdout_frac=0.20, neutralize=False):
    states = pool["states"]
    if neutralize:
        states = states.copy(); states[:, 2] = 0.0; states[:, 3] = 0.0
    ho = game_split(pool["gid"], holdout_frac, split_seed)
    tr = ~ho
    # leakage check
    g_tr = set(pool["gid"][tr].tolist()); g_ho = set(pool["gid"][ho].tolist())
    shared = len(g_tr & g_ho)
    lab = pool["label"]
    n_tr_loss = int(((lab < 0) & tr).sum()); n_tr_win = int(((lab > 0) & tr).sum())
    n_ho_loss = int(((lab < 0) & ho).sum()); n_ho_win = int(((lab > 0) & ho).sum())

    net = load_model(device)
    freeze_trunk(net)
    f_before, moved_before = verify_frozen(net)
    train_head(net, states[tr], lab[tr], device, seed=train_seed,
               label=f"seed{split_seed}{'_neut' if neutralize else ''}")
    f_after, moved_after = verify_frozen(net)

    v_all = values(net, states, device)  # per-pool-index value
    naive_ka = kill_a(v_all[(lab < 0) & ho]); naive_kc = kill_c(v_all[(lab > 0) & ho])
    matched = matched_eval(pool, ho, v_all, v_all)

    return {
        "split_seed": split_seed, "train_seed": train_seed, "neutralize": neutralize,
        "frozen_before": f_before, "frozen_after": f_after,
        "moved_params": moved_before,
        "shared_games": shared,
        "n_tr_loss": n_tr_loss, "n_tr_win": n_tr_win,
        "n_ho_loss": n_ho_loss, "n_ho_win": n_ho_win,
        "naive_kill_a": round(naive_ka, 4), "naive_kill_c": round(naive_kc, 4),
        "matched": matched,
    }


def main():
    t0 = time.time()
    device = best_device()
    log.info("=" * 72)
    log.info("D-FULLSPEC E1 REVIEW — INDEPENDENT game-level-split re-derivation")
    log.info("=" * 72)
    pool = build_pool()
    lab = pool["label"]
    log.info(f"pool n={len(lab)} loss={int((lab<0).sum())} win={int((lab>0).sum())} "
             f"plane2==plane3 everywhere={pool['p23_eq']} distinct_games={len(np.unique(pool['gid']))}")
    log.info(f"tp distribution loss: tp0={int(((lab<0)&(pool['tp']==0)).sum())} "
             f"tp1={int(((lab<0)&(pool['tp']==1)).sum())} | "
             f"win: tp0={int(((lab>0)&(pool['tp']==0)).sum())} "
             f"tp1={int(((lab>0)&(pool['tp']==1)).sum())}")

    # multi-seed game-level split robustness (3 distinct game partitions)
    runs = []
    for ss in (101, 202, 303):
        r = run_seed(pool, device, split_seed=ss, train_seed=0)
        runs.append(r)
        m = r["matched"]
        log.info(f"[SPLIT {ss}] shared_games={r['shared_games']} "
                 f"tr(l/w)={r['n_tr_loss']}/{r['n_tr_win']} ho(l/w)={r['n_ho_loss']}/{r['n_ho_win']} "
                 f"| naive KA={r['naive_kill_a']} KC={r['naive_kill_c']} "
                 f"| MATCHED n(l/w)={m['n_loss']}/{m['n_win']} KA={m['kill_a']} KC={m['kill_c']} "
                 f"rwKC={m['kill_c_stone_reweighted']} | tp1 win n={m['tp1_n_win']} KC={m['tp1_kill_c']}")

    # neutralized control (planes2,3 zeroed) on seed 101
    neut = run_seed(pool, device, split_seed=101, train_seed=0, neutralize=True)
    mn = neut["matched"]
    log.info(f"[NEUTRALIZED seed101] naive KA={neut['naive_kill_a']} KC={neut['naive_kill_c']} "
             f"| MATCHED KA={mn['kill_a']} KC={mn['kill_c']} rwKC={mn['kill_c_stone_reweighted']}")

    # aggregate matched across split seeds
    mka = np.array([r["matched"]["kill_a"] for r in runs])
    mkc = np.array([r["matched"]["kill_c"] for r in runs])
    rwkc = np.array([r["matched"]["kill_c_stone_reweighted"] for r in runs])
    agg_ka, agg_kc = float(mka.mean()), float(mkc.mean())
    log.info(f"[AGG matched] KILL-A mean={agg_ka:.4f} range[{mka.min():.3f},{mka.max():.3f}] "
             f"| KILL-C mean={agg_kc:.4f} range[{mkc.min():.3f},{mkc.max():.3f}] "
             f"| rwKC mean={rwkc.mean():.4f}")

    # verdict from MY aggregate matched numbers
    NEAR = 0.10
    a_pass = agg_ka > 0.35
    c_pass = agg_kc >= 0.85
    a_near = abs(agg_ka - 0.35) <= NEAR
    c_near = abs(agg_kc - 0.85) <= NEAR
    if a_pass and c_pass:
        verdict = "SEPARABLE"
    elif (a_pass and c_near) or (c_pass and a_near):
        verdict = "PARTIAL"
    else:
        verdict = "ENTANGLED"

    # class balance: balanced batches enforce per-step balance; report train ratio
    tr_loss = np.mean([r["n_tr_loss"] for r in runs])
    tr_win = np.mean([r["n_tr_win"] for r in runs])
    class_balance_ok = bool(0.5 <= tr_loss / tr_win <= 2.0)  # pool roughly balanced AND batches balanced

    frozen_ok = all(r["frozen_before"] and r["frozen_after"] for r in runs) and \
        all(set(r["moved_params"]) == {"value_fc1.weight", "value_fc1.bias",
                                       "value_fc2.weight", "value_fc2.bias"} for r in runs)

    turnphase_control_ok = bool(pool["p23_eq"]) and all(r["shared_games"] == 0 for r in runs)

    elapsed = time.time() - t0
    result = {
        "ran": True,
        "device": str(device),
        "pool": {"n_loss": int((lab < 0).sum()), "n_win": int((lab > 0).sum()),
                 "distinct_games": int(len(np.unique(pool["gid"]))),
                 "plane2_eq_plane3": pool["p23_eq"]},
        "split": "GAME-LEVEL whole-game disjoint, fresh seeds 101/202/303 (NOT DS1-CSV split)",
        "runs": runs,
        "neutralized": neut,
        "aggregate_matched": {
            "kill_a_mean": round(agg_ka, 4), "kill_a_range": [round(float(mka.min()), 4), round(float(mka.max()), 4)],
            "kill_c_mean": round(agg_kc, 4), "kill_c_range": [round(float(mkc.min()), 4), round(float(mkc.max()), 4)],
            "kill_c_stone_reweighted_mean": round(float(rwkc.mean()), 4),
        },
        "verdict": verdict,
        "class_balance_ok": class_balance_ok,
        "turnphase_control_ok": turnphase_control_ok,
        "frozen_ok": frozen_ok,
        "wall_time_s": round(elapsed, 1),
    }
    out_dir = REPO_ROOT / "reports" / "d_fullspec_2026-06-26"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "E1_review_indep_results.json", "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"VERDICT={verdict} class_balance_ok={class_balance_ok} "
             f"turnphase_control_ok={turnphase_control_ok} frozen_ok={frozen_ok}")
    log.info(f"DONE in {elapsed:.1f}s")
    print("E1_REVIEW_INDEP_JSON " + json.dumps(result))


if __name__ == "__main__":
    main()
