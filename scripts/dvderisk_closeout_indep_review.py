#!/usr/bin/env python3
"""D-FULLSPEC CLOSEOUT — INDEPENDENT re-derivation (does NOT call the probe script).

Re-derives the central claim of scripts/dvderisk_closeout_probe.py:
  Is win/loss RECOVERABLE from 272357's FROZEN trunk representation by a flexible
  PRE-POOL spatial readout, on a GAME-DISJOINT, TURN-PHASE-MATCHED holdout?

INDEPENDENT pieces (re-implemented here, NOT imported from the probe under test):
  * own game-disjoint whole-game 3-way split (FRESH seeds, assert shared_games==0)
  * own matched-stratum derivation (tp==0 INTERSECT overlapping stone support)
  * own turn-phase guard (planes 2,3 -> win/loss, must be ~0.5 on matched holdout)
  * own frozen trunk-output extraction (hook) + value-head RECONSTRUCTION sanity
    (pooled -> value_fc1 -> relu -> value_fc2 -> tanh must reproduce net value)
  * own flexible probes (pre-pool spatial conv + pooled MLP), own rank-AUC + KILL-C/A

Reuses ONLY the data layer (byte-exact bank->game_id join) and checkpoint loader
from dvderisk_lighttrunk_probe — these are shared DATA/MODEL infra, not the
decodability claim under review.

Read-only on the npz banks. Writes a JSON report; commits nothing.
"""
from __future__ import annotations

import json
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

import scripts.dvderisk_lighttrunk_probe as L  # data join + checkpoint loader ONLY

OUT_DIR = REPO_ROOT / "reports" / "d_fullspec_2026-06-26"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRESH_SEEDS = [7, 13, 29, 41, 53]   # deliberately disjoint from probe's 101/202/303
HOLDOUT_FRAC = 0.25
VAL_FRAC = 0.18


# --------------------------------------------------------------------------- #
# Independent metrics
# --------------------------------------------------------------------------- #
def rank_auc(scores, labels):
    """ROC-AUC, tie-averaged ranks. labels: 1=win(pos), 0=loss(neg)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    sv = scores[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[labels == 1].sum()
    n_p, n_n = len(pos), len(neg)
    return float((r_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def kill_c(logits, labels):   # frac WINS (1) scored win (logit>0)
    m = labels == 1
    return float((logits[m] > 0).mean()) if m.any() else float("nan")


def kill_a(logits, labels):   # frac LOSSES (0) scored loss (logit<0)
    m = labels == 0
    return float((logits[m] < 0).mean()) if m.any() else float("nan")


# --------------------------------------------------------------------------- #
# Independent split + matched-stratum + tp guard
# --------------------------------------------------------------------------- #
def game_disjoint_split(gids, holdout_frac, val_frac, seed):
    uniq = np.unique(gids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_ho = max(1, int(round(len(uniq) * holdout_frac)))
    ho_g = set(perm[:n_ho].tolist())
    rest = perm[n_ho:]
    n_val = max(1, int(round(len(rest) * val_frac)))
    val_g = set(rest[:n_val].tolist())
    tr_g = set(rest[n_val:].tolist())
    tr = np.array([g in tr_g for g in gids])
    val = np.array([g in val_g for g in gids])
    ho = np.array([g in ho_g for g in gids])
    return tr, val, ho


def matched_indices(pool, mask):
    """tp==0 INTERSECT overlapping stone-count support, on subset `mask`.
    Returns (idx, y, lo, hi) where y=1 win / 0 loss. None if a class empty."""
    lab, tp, stones = pool["label"], pool["tp"], pool["stones"]
    l0 = (lab < 0) & mask & (tp == 0)
    w0 = (lab > 0) & mask & (tp == 0)
    if l0.sum() == 0 or w0.sum() == 0:
        return None
    lo = max(stones[l0].min(), stones[w0].min())
    hi = min(stones[l0].max(), stones[w0].max())
    lm = l0 & (stones >= lo) & (stones <= hi)
    wm = w0 & (stones >= lo) & (stones <= hi)
    idx = np.where(lm | wm)[0]
    y = (lab[idx] > 0).astype("int64")
    return idx, y, float(lo), float(hi)


def tp_guard_auc(pool, idx, y):
    """Best win/loss AUC from planes 2,3 means alone on the matched holdout.
    ~0.5 => matched stratum carries NO turn-phase shortcut."""
    s = pool["states"][idx]
    f2 = s[:, 2].reshape(len(s), -1).mean(1)
    f3 = s[:, 3].reshape(len(s), -1).mean(1)
    return round(max(rank_auc(f2, y), rank_auc(-f2, y),
                     rank_auc(f3, y), rank_auc(-f3, y)), 4)


# --------------------------------------------------------------------------- #
# Independent frozen-trunk feature extraction (+ value-head reconstruction check)
# --------------------------------------------------------------------------- #
def extract_trunk(net, states, device, bs=128):
    net.eval()
    cap = {}
    h = net.trunk.register_forward_hook(lambda m, i, o: cap.__setitem__("o", o.detach()))
    pooled, prepool, net_val, recon_val = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(states), bs):
            x = torch.from_numpy(states[s:s + bs]).to(device)
            _, v, _ = net(x)
            o = cap["o"]
            p = torch.cat([o.mean(dim=(2, 3)), o.amax(dim=(2, 3))], dim=1)
            # reconstruct the value head from the captured pooled vector
            vr = torch.tanh(net.value_fc2(F.relu(net.value_fc1(p))))
            pooled.append(p.float().cpu())
            prepool.append(o.float().cpu())
            net_val.append(v.squeeze(1).float().cpu())
            recon_val.append(vr.squeeze(1).float().cpu())
    h.remove()
    pooled = torch.cat(pooled).numpy()
    prepool = torch.cat(prepool).numpy()
    recon_err = float((torch.cat(net_val) - torch.cat(recon_val)).abs().max())
    return pooled, prepool, recon_err


# --------------------------------------------------------------------------- #
# Independent flexible probes
# --------------------------------------------------------------------------- #
class PrepoolConv(nn.Module):
    """Flexible spatial readout over frozen (C,19,19): 1x1->BN->3x3->BN->gpool->fc.
    Distinct arch from the probe-under-test (BN, wider mid, heavier dropout)."""
    def __init__(self, c_in, mid=48, p=0.4):
        super().__init__()
        self.c1 = nn.Conv2d(c_in, mid, 1)
        self.b1 = nn.BatchNorm2d(mid)
        self.c2 = nn.Conv2d(mid, mid, 3, padding=1)
        self.b2 = nn.BatchNorm2d(mid)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(2 * mid, 1)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = F.relu(self.b2(self.c2(h)))
        h = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.fc(self.drop(h)).squeeze(1)


class PooledMLP(nn.Module):
    def __init__(self, d_in, hidden=128, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(True),
                                 nn.Dropout(p), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_probe(make_model, Xtr, ytr, Xval, yval, Xho, yho, device, *, spatial,
                seed, max_epochs=250, patience=35, lr=1e-3, wd=1e-3, bs=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if spatial:
        mu = Xtr.mean(axis=(0, 2, 3), keepdims=True)
        sd = Xtr.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    else:
        mu = Xtr.mean(0, keepdims=True)
        sd = Xtr.std(0, keepdims=True) + 1e-6
    norm = lambda A: ((A - mu) / sd).astype("float32")
    Xtr_t = torch.from_numpy(norm(Xtr)).to(device)
    Xval_t = torch.from_numpy(norm(Xval)).to(device)
    Xho_t = torch.from_numpy(norm(Xho)).to(device)
    ytr_t = torch.from_numpy(ytr.astype("float32")).to(device)

    model = make_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()
    li = np.where(ytr == 0)[0]
    wi = np.where(ytr == 1)[0]
    nl, nw = len(li), len(wi)
    rng = np.random.default_rng(seed)
    steps = max(nl, nw) // bs + 1
    best_val, best_state, best_ep, bad = -1.0, None, -1, 0
    for ep in range(max_epochs):
        model.train()
        pl, pw = rng.permutation(nl), rng.permutation(nw)
        for st in range(steps):
            bl = li[pl[(st * bs) % nl:][:bs]]
            bw = wi[pw[(st * bs) % nw:][:bs]]
            if len(bl) == 0 or len(bw) == 0:
                continue
            idx = np.concatenate([bl, bw])
            opt.zero_grad()
            bce(model(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va = rank_auc(model(Xval_t).cpu().numpy(), yval)
        if va > best_val + 1e-4:
            best_val, best_ep, bad = va, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        ho = model(Xho_t).cpu().numpy()
        tr = model(Xtr_t).cpu().numpy()
    return {
        "ho_auc": rank_auc(ho, yho), "ho_kill_c": kill_c(ho, yho), "ho_kill_a": kill_a(ho, yho),
        "tr_auc": rank_auc(tr, ytr), "best_val_auc": round(best_val, 4), "best_epoch": best_ep,
        "n_ho_win": int((yho == 1).sum()), "n_ho_loss": int((yho == 0).sum()),
    }


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = L.build_pool()
    lab = pool["label"]
    print(f"[indep] pool n={len(lab)} loss={int((lab<0).sum())} win={int((lab>0).sum())} "
          f"games={len(np.unique(pool['gid']))} p23_eq={pool['p23_eq']}", flush=True)

    net = L.load_model(device)
    pooled_all, prepool_all, recon_err = extract_trunk(net, pool["states"], device)
    C = prepool_all.shape[1]
    print(f"[indep] frozen extract: pooled{pooled_all.shape} prepool{prepool_all.shape} "
          f"value_head_recon_max_err={recon_err:.2e}", flush=True)
    assert recon_err < 1e-3, "pooled extraction does NOT reconstruct value head -> wrong tensor"

    per_seed = []
    for seed in FRESH_SEEDS:
        tr, val, ho = game_disjoint_split(pool["gid"], HOLDOUT_FRAC, VAL_FRAC, seed)
        g_tr = set(pool["gid"][tr].tolist())
        g_val = set(pool["gid"][val].tolist())
        g_ho = set(pool["gid"][ho].tolist())
        shared = len(g_tr & g_val) + len(g_tr & g_ho) + len(g_val & g_ho)
        mtr = matched_indices(pool, tr)
        mval = matched_indices(pool, val)
        mho = matched_indices(pool, ho)
        if mtr is None or mho is None or mval is None:
            print(f"[seed {seed}] matched stratum empty, skip", flush=True)
            continue
        itr, ytr, _, _ = mtr
        ival, yval, _, _ = mval
        iho, yho, _, _ = mho
        guard = tp_guard_auc(pool, iho, yho)
        # plane2==plane3 within matched holdout
        p23_ho = bool(np.array_equal(pool["states"][iho][:, 2], pool["states"][iho][:, 3]))

        conv = train_probe(lambda: PrepoolConv(C), prepool_all[itr], ytr,
                           prepool_all[ival], yval, prepool_all[iho], yho, device,
                           spatial=True, seed=seed)
        mlp = train_probe(lambda: PooledMLP(pooled_all.shape[1]), pooled_all[itr], ytr,
                          pooled_all[ival], yval, pooled_all[iho], yho, device,
                          spatial=False, seed=seed)
        per_seed.append({"seed": seed, "shared_games": shared, "tp_guard_auc": guard,
                         "p23_eq_matched_ho": p23_ho,
                         "n_tr_lw": [int((ytr == 0).sum()), int((ytr == 1).sum())],
                         "n_ho_lw": [int((yho == 0).sum()), int((yho == 1).sum())],
                         "prepool_conv": conv, "pooled_mlp": mlp})
        print(f"[seed {seed}] shared={shared} tp_guard={guard} p23eq_ho={p23_ho} "
              f"ho(l/w)={int((yho==0).sum())}/{int((yho==1).sum())}\n"
              f"   PREPOOL_CONV ho_auc={conv['ho_auc']:.3f} ho_kc={conv['ho_kill_c']:.3f} "
              f"ho_ka={conv['ho_kill_a']:.3f} tr_auc={conv['tr_auc']:.3f} "
              f"(gap {conv['tr_auc']-conv['ho_auc']:+.3f}) ep={conv['best_epoch']}\n"
              f"   POOLED_MLP   ho_auc={mlp['ho_auc']:.3f} ho_kc={mlp['ho_kill_c']:.3f} "
              f"ho_ka={mlp['ho_kill_a']:.3f} tr_auc={mlp['tr_auc']:.3f}", flush=True)

    def agg(key, field):
        return round(float(np.mean([s[key][field] for s in per_seed])), 4)

    prepool_auc = agg("prepool_conv", "ho_auc")
    prepool_kc = agg("prepool_conv", "ho_kill_c")
    prepool_tr = agg("prepool_conv", "tr_auc")
    pooled_auc = agg("pooled_mlp", "ho_auc")
    pooled_kc = agg("pooled_mlp", "ho_kill_c")
    guards = [s["tp_guard_auc"] for s in per_seed]
    shared_total = sum(s["shared_games"] for s in per_seed)
    game_disjoint_ok = shared_total == 0
    turnphase_ok = all(abs(g - 0.5) <= 0.06 for g in guards) and \
        all(s["p23_eq_matched_ho"] for s in per_seed) and pool["p23_eq"]
    probe_overfit = (prepool_tr - prepool_auc) > 0.20

    # corrected verdict from MY numbers
    KC_REOPEN, AUC_SEP = 0.85, 0.72
    best_auc = max(prepool_auc, pooled_auc)
    best_kc = max(prepool_kc, pooled_kc)
    prepool_beats_pooled = (prepool_auc > pooled_auc + 0.02) and (prepool_kc >= KC_REOPEN)
    if prepool_beats_pooled and prepool_auc >= AUC_SEP:
        verdict = "REOPENED_READOUT"
    elif best_kc >= KC_REOPEN and best_auc >= AUC_SEP:
        verdict = "REOPENED_READOUT"   # some readout separates
    else:
        verdict = "CLOSED_ENTANGLED"

    out = {
        "ran": True, "device": device, "fresh_seeds": FRESH_SEEDS,
        "value_head_recon_max_err": recon_err, "filters": int(C),
        "indep_prepool_auc": prepool_auc, "indep_prepool_kill_c": prepool_kc,
        "indep_prepool_train_auc": prepool_tr,
        "indep_pooled_auc": pooled_auc, "indep_pooled_kill_c": pooled_kc,
        "tp_guard_auc_per_seed": guards,
        "game_disjoint_ok": game_disjoint_ok, "shared_games_total": shared_total,
        "turnphase_control_ok": turnphase_ok, "probe_overfit": probe_overfit,
        "prepool_beats_pooled": prepool_beats_pooled,
        "corrected_verdict": verdict, "per_seed": per_seed,
        "wall_time_s": round(time.time() - t0, 1),
    }
    with open(OUT_DIR / "CLOSEOUT_INDEP_REVIEW.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n==== INDEP SUMMARY ====")
    print(f"prepool_conv  HO AUC={prepool_auc} KILL-C={prepool_kc} (train AUC={prepool_tr})")
    print(f"pooled_mlp    HO AUC={pooled_auc} KILL-C={pooled_kc}")
    print(f"prepool_beats_pooled={prepool_beats_pooled}  game_disjoint_ok={game_disjoint_ok} "
          f"turnphase_ok={turnphase_ok} probe_overfit={probe_overfit}")
    print(f"CORRECTED_VERDICT={verdict}")
    print("INDEP_REVIEW_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
