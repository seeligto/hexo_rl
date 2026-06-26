#!/usr/bin/env python3
"""D-FULLSPEC E1 RED-TEAM — adversarial audit of the ENTANGLED verdict.

Four attack axes (default skepticism):
  1. TURN-PHASE SHORTCUT — does E1's matched eval truly hold plane2
     (moves_remaining) AND plane3 (ply_parity) constant? Re-run a turn-phase
     logistic probe RESTRICTED to the matched set (should collapse to ~0.5 if
     genuinely neutralized). Re-run the planes-2,3-neutralized distill.
  2. LEAKAGE — near-dup positions across train/holdout (game_id join via the
     reconstructed win-bank -> DS1 mapping; byte-identical + Hamming near-dup on
     the stone planes). Could a SEPARABLE be leakage? (verdict is ENTANGLED, so
     leakage can only HELP separation -> strengthens ENTANGLED if present.)
  3. UNDER-POWERED — is ENTANGLED an artifact of too-few tp==0 TRAINING wins to
     LEARN the matched-stratum win manifold (vs true inseparability)? Count the
     tp distribution of TRAINING wins; the matched eval is tp==0 only.
  4. EFFECTIVE-N — are the matched holdout wins distinct GAMES or copies
     (§D-ARGMAX deterministic-regime collapse)?

Read-only on every npz bank. Writes durable artifacts under
reports/d_fullspec_2026-06-26/. No commits.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.checkpoints import extract_model_state
from hexo_rl.utils.device import best_device
from scripts.dvderisk_ds3_probe import freeze_trunk, verify_freezing

WIN_THR = 99999000.0
CKPT = "checkpoints/checkpoint_00272357.pt"
OUT = REPO / "reports" / "d_fullspec_2026-06-26"
OUT.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
def load_bank(name):
    d = np.load(REPO / f"data/{name}.npz", allow_pickle=False)
    s = d["states"].astype("float32")
    t = d["target"].astype("float32")
    tp2 = s[:, 2].reshape(len(s), -1).mean(1)
    tp3 = s[:, 3].reshape(len(s), -1).mean(1)
    stones = (s[:, 0] + s[:, 1]).reshape(len(s), -1).sum(1)
    return {"s": s, "t": t, "tp2": tp2, "tp3": tp3, "stones": stones}


def load_concat():
    s1 = np.load(REPO / "data/livetail_bank_e928c854.npz")["states"].astype("float32")
    s2 = np.load(REPO / "data/selfplay_bank12k_v6_live2_ls_50k.npz")["states"].astype("float32")
    return np.concatenate([s1, s2], axis=0)


def win_selection():
    """Replicate build_killc.py selection; return per-split list of (bidx, game_id, ply, pos_hash)."""
    concat_n = 31878
    sel = {"train": [], "holdout": []}
    for cf, split in [("data/dvderisk_ds1_train.csv", "train"),
                      ("data/dvderisk_ds1_holdout.csv", "holdout")]:
        for r in csv.DictReader(open(REPO / cf)):
            nv = float(r["net_value"]); sb = float(r["sealbot_score"]); bidx = int(r["buffer_idx"])
            if sb >= WIN_THR and nv > 0.0 and bidx < concat_n:
                sel[split].append((bidx, r["game_id"], int(r["ply"]), r["pos_hash"]))
    return sel


def load_net(device, encoding="v6_live2_ls"):
    ckpt = torch.load(REPO / CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    state = extract_model_state(ckpt)
    inp_w = state.get("trunk.input_conv.weight")
    in_ch = int(inp_w.shape[1]) if inp_w is not None else cfg.get("in_channels", 4)
    net = HexTacToeNet(in_channels=in_ch, res_blocks=cfg.get("res_blocks", 12),
                       filters=cfg.get("filters", 128), board_size=cfg.get("board_size", 19),
                       encoding=encoding)
    net.load_state_dict(state, strict=True)
    return net.to(device)


@torch.no_grad()
def values(net, states, device, bs=256):
    net.eval()
    out = []
    for i in range(0, len(states), bs):
        x = torch.from_numpy(states[i:i + bs]).to(device)
        _, v, _ = net(x)
        out.append(v.squeeze(1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def auc_mw(scores, labels):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    alls = np.concatenate([pos, neg])
    order = np.argsort(alls, kind="mergesort")
    ranks = np.empty(len(alls)); ranks[order] = np.arange(1, len(alls) + 1)
    # tie-correct
    s = alls[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2
            ranks[order[i:j + 1]] = avg
        i = j + 1
    sp = ranks[:len(pos)].sum()
    return float((sp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_logistic(Xtr, ytr, Xho, yho, device, iters=2000, lr=0.1):
    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr_n = (Xtr - mu) / sd
    w = torch.zeros(Xtr.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr); bce = nn.BCEWithLogitsLoss()
    for _ in range(iters):
        opt.zero_grad(); loss = bce(Xtr_n @ w + b, ytr_t); loss.backward(); opt.step()
    with torch.no_grad():
        Xho_t = torch.tensor(Xho, dtype=torch.float32, device=device)
        sc = ((Xho_t - mu) / sd @ w + b).cpu().numpy()
        sct = (Xtr_n @ w + b).detach().cpu().numpy()
    return auc_mw(sc, yho), auc_mw(sct, ytr)


def train_vhead(net, loss_s, win_s, device, lr=3e-4, epochs=120, bpc=64, seed=0):
    vh = [net.value_fc1.weight, net.value_fc1.bias, net.value_fc2.weight, net.value_fc2.bias]
    opt = torch.optim.Adam(vh, lr=lr); lf = nn.MSELoss()
    rng = np.random.default_rng(seed)
    TL, TW = loss_s, win_s
    for ep in range(epochs):
        net.train(); pl = rng.permutation(len(TL)); pw = rng.permutation(len(TW))
        steps = max(len(TL), len(TW)) // bpc + 1
        for st in range(steps):
            bl = pl[(st * bpc) % len(TL):][:bpc]; bw = pw[(st * bpc) % len(TW):][:bpc]
            if len(bl) == 0 or len(bw) == 0:
                continue
            xs = np.concatenate([TL[bl], TW[bw]])
            ts = np.concatenate([np.full(len(bl), -1.0, "float32"), np.full(len(bw), 1.0, "float32")])
            opt.zero_grad(); _, v, _ = net(torch.from_numpy(xs).to(device))
            lf(v.squeeze(1), torch.from_numpy(ts).to(device)).backward(); opt.step()


def board_sig(s):
    """Byte signature of the two stone planes (0,1) — turn-phase-independent board id."""
    return s[:2].tobytes()


def hamming_stones(a, b):
    """# cells where stone occupancy differs across planes 0,1."""
    return int(np.sum((a[:2] != b[:2])))


# ===========================================================================
def main():
    t0 = time.time()
    device = best_device()
    log("=" * 72)
    log("D-FULLSPEC E1 RED-TEAM — adversarial audit")
    log(f"device={device}")
    R = {}

    lt = load_bank("distill_d7_train"); lh = load_bank("distill_d7_holdout")
    wt = load_bank("killc_netwin_train"); wh = load_bank("killc_netwin_holdout")
    log(f"n loss tr/ho={len(lt['s'])}/{len(lh['s'])} win tr/ho={len(wt['s'])}/{len(wh['s'])}")

    # ---- provenance: win-bank -> DS1 game_id ----
    sel = win_selection()
    concat = load_concat()
    wt_meta = sel["train"]; wh_meta = sel["holdout"]
    assert len(wt_meta) == len(wt["s"]) and len(wh_meta) == len(wh["s"])
    # verify state match
    vmatch = all(np.array_equal(concat[wt_meta[i][0]], wt["s"][i]) for i in range(len(wt_meta)))
    vmatchh = all(np.array_equal(concat[wh_meta[i][0]], wh["s"][i]) for i in range(len(wh_meta)))
    log(f"win-bank<->DS1 reconstruct verified train={vmatch} holdout={vmatchh}")

    # ---- map LOSS bank to concat (byte) to recover provenance if present ----
    concat_sig = {}
    for i in range(len(concat)):
        concat_sig.setdefault(concat[i].tobytes(), i)
    def map_loss(states):
        idxs = []
        for s in states:
            idxs.append(concat_sig.get(s.tobytes(), -1))
        return idxs
    lt_cidx = map_loss(lt["s"]); lh_cidx = map_loss(lh["s"])
    log(f"loss bank in concat: train {sum(1 for x in lt_cidx if x>=0)}/{len(lt_cidx)} "
        f"holdout {sum(1 for x in lh_cidx if x>=0)}/{len(lh_cidx)}")
    # join loss concat idx -> DS1 row (game_id) via buffer_idx
    ds1 = {}
    for cf in ["data/dvderisk_ds1_train.csv", "data/dvderisk_ds1_holdout.csv"]:
        for r in csv.DictReader(open(REPO / cf)):
            ds1[int(r["buffer_idx"])] = (r["game_id"], int(r["ply"]), r["pos_hash"],
                                          cf.endswith("holdout.csv"))
    def loss_games(cidx):
        gids = []
        for ci in cidx:
            if ci >= 0 and ci in ds1:
                gids.append(ds1[ci])
        return gids
    lt_games = loss_games(lt_cidx); lh_games = loss_games(lh_cidx)
    log(f"loss bank joined to DS1: train {len(lt_games)}/{len(lt_cidx)} holdout {len(lh_games)}/{len(lh_cidx)}")

    # =====================================================================
    # AXIS 1 — TURN-PHASE SHORTCUT
    # =====================================================================
    log("\n" + "#" * 30 + " AXIS 1: TURN-PHASE " + "#" * 30)
    # 1a. planes 2 & 3 identical per position?
    p23_id_lh = bool(np.allclose(lh["tp2"], lh["tp3"]))
    p23_id_wh = bool(np.allclose(wh["tp2"], wh["tp3"]))
    # exact per-cell identity check
    p23_cell_lh = bool(np.array_equal(lh["s"][:, 2], lh["s"][:, 3]))
    p23_cell_wh = bool(np.array_equal(wh["s"][:, 2], wh["s"][:, 3]))
    log(f"planes2==3 per-pos scalar: loss_ho={p23_id_lh} win_ho={p23_id_wh} | "
        f"per-cell identical: loss_ho={p23_cell_lh} win_ho={p23_cell_wh}")

    # 1b. build matched set (E1's def): tp2<0.5 INTERSECT overlapping stone support
    lh_tp0 = lh["tp2"] < 0.5; wh_tp0 = wh["tp2"] < 0.5
    lo = max(lh["stones"][lh_tp0].min(), wh["stones"][wh_tp0].min())
    hi = min(lh["stones"][lh_tp0].max(), wh["stones"][wh_tp0].max())
    l_mask = lh_tp0 & (lh["stones"] >= lo) & (lh["stones"] <= hi)
    w_mask = wh_tp0 & (wh["stones"] >= lo) & (wh["stones"] <= hi)
    log(f"matched stone support [{lo:.0f},{hi:.0f}] n_loss={int(l_mask.sum())} n_win={int(w_mask.sum())}")

    # 1c. residual turn-phase within matched: means of plane2/3
    log(f"WITHIN matched: plane2 mean loss={lh['tp2'][l_mask].mean():.4f} win={wh['tp2'][w_mask].mean():.4f} | "
        f"plane3 mean loss={lh['tp3'][l_mask].mean():.4f} win={wh['tp3'][w_mask].mean():.4f}")
    log(f"WITHIN matched: plane2 uniq loss={np.unique(np.round(lh['tp2'][l_mask],4))} "
        f"win={np.unique(np.round(wh['tp2'][w_mask],4))}")

    # 1d. turn-phase logistic probe RESTRICTED to matched set (LOOCV-ish: train on matched, eval on matched)
    # if planes2,3 truly constant within matched, probe AUC -> ~0.5
    Xm = np.stack([np.concatenate([lh["tp2"][l_mask], wh["tp2"][w_mask]]),
                   np.concatenate([lh["tp3"][l_mask], wh["tp3"][w_mask]])], axis=1)
    ym = np.concatenate([np.zeros(int(l_mask.sum())), np.ones(int(w_mask.sum()))])
    matched_tp_auc, _ = fit_logistic(Xm, ym, Xm, ym, device)
    # also stone-count probe within matched (residual occupancy shortcut)
    Xs = np.concatenate([lh["stones"][l_mask], wh["stones"][w_mask]]).reshape(-1, 1)
    matched_stone_auc, _ = fit_logistic(Xs, ym, Xs, ym, device)
    log(f"WITHIN-matched turn-phase(plane2,3) probe AUC={matched_tp_auc:.3f} "
        f"(0.5=neutralized); stone-count probe AUC={matched_stone_auc:.3f}")
    R["axis1_turnphase"] = {
        "planes23_identical_percell": {"loss_ho": p23_cell_lh, "win_ho": p23_cell_wh},
        "matched_n_loss": int(l_mask.sum()), "matched_n_win": int(w_mask.sum()),
        "within_matched_turnphase_probe_auc": round(matched_tp_auc, 4),
        "within_matched_stonecount_probe_auc": round(matched_stone_auc, 4),
        "within_matched_plane2_loss_mean": round(float(lh["tp2"][l_mask].mean()), 5),
        "within_matched_plane2_win_mean": round(float(wh["tp2"][w_mask].mean()), 5),
    }

    # =====================================================================
    # AXIS 2 — LEAKAGE
    # =====================================================================
    log("\n" + "#" * 30 + " AXIS 2: LEAKAGE " + "#" * 30)
    # game_id overlap: win bank
    wt_g = set(m[1] for m in wt_meta); wh_g = set(m[1] for m in wh_meta)
    win_gid_overlap = wt_g & wh_g
    log(f"WIN bank game_id: train={len(wt_g)} holdout={len(wh_g)} overlap={len(win_gid_overlap)}")
    # loss bank game_id overlap
    lt_g = set(m[0] for m in lt_games); lh_g = set(m[0] for m in lh_games)
    loss_gid_overlap = lt_g & lh_g
    log(f"LOSS bank game_id: train={len(lt_g)} holdout={len(lh_g)} overlap={len(loss_gid_overlap)} "
        f"(joined frac tr={len(lt_games)/max(1,len(lt_cidx)):.2f} ho={len(lh_games)/max(1,len(lh_cidx)):.2f})")
    # cross-bank game overlap (a holdout WIN sharing game w/ a train LOSS = same game both classes)
    cross_gid = (wh_g | lh_g) & (wt_g | lt_g)
    log(f"cross-bank holdout-game ∩ train-game = {len(cross_gid)}")

    # byte-identical board (planes0,1) across train/holdout
    def sigs(bank):
        return [board_sig(s) for s in bank["s"]]
    wt_sig = set(sigs(wt)); wh_sig = sigs(wh)
    lt_sig = set(sigs(lt)); lh_sig = sigs(lh)
    win_byte_dup = sum(1 for s in wh_sig if s in wt_sig)
    loss_byte_dup = sum(1 for s in lh_sig if s in lt_sig)
    log(f"byte-identical board (planes0,1) holdout∈train: win={win_byte_dup}/{len(wh_sig)} "
        f"loss={loss_byte_dup}/{len(lh_sig)}")

    # near-dup: min Hamming (stone planes) holdout->train, on MATCHED sets (verdict basis)
    def min_hamming(holb, holmask, trb):
        tr = trb["s"]
        res = []
        for s in holb["s"][holmask]:
            best = min(hamming_stones(s, t) for t in tr)
            res.append(best)
        return np.array(res)
    win_mh = min_hamming(wh, w_mask, wt)
    loss_mh = min_hamming(lh, l_mask, lt)
    log(f"MATCHED win  holdout->train min-Hamming(stones): "
        f"min={win_mh.min()} med={np.median(win_mh):.0f} <=2:{int((win_mh<=2).sum())} <=4:{int((win_mh<=4).sum())} of {len(win_mh)}")
    log(f"MATCHED loss holdout->train min-Hamming(stones): "
        f"min={loss_mh.min()} med={np.median(loss_mh):.0f} <=2:{int((loss_mh<=2).sum())} <=4:{int((loss_mh<=4).sum())} of {len(loss_mh)}")
    near = (int((win_mh <= 2).sum()) + int((loss_mh <= 2).sum())) > 0
    R["axis2_leakage"] = {
        "win_gid_overlap": len(win_gid_overlap), "loss_gid_overlap": len(loss_gid_overlap),
        "cross_bank_game_overlap": len(cross_gid),
        "win_byte_dup_holdout_in_train": win_byte_dup, "loss_byte_dup_holdout_in_train": loss_byte_dup,
        "matched_win_minhamming_le2": int((win_mh <= 2).sum()),
        "matched_loss_minhamming_le2": int((loss_mh <= 2).sum()),
        "matched_win_minhamming_min": int(win_mh.min()), "matched_loss_minhamming_min": int(loss_mh.min()),
        "near_dup_found": bool(near),
    }

    # =====================================================================
    # AXIS 3 — UNDER-POWERED (tp distribution of TRAINING wins)
    # =====================================================================
    log("\n" + "#" * 30 + " AXIS 3: UNDER-POWERED " + "#" * 30)
    wt_tp0 = int((wt["tp2"] < 0.5).sum()); wt_tp1 = int((wt["tp2"] >= 0.5).sum())
    lt_tp0 = int((lt["tp2"] < 0.5).sum()); lt_tp1 = int((lt["tp2"] >= 0.5).sum())
    log(f"TRAIN wins  tp0(moves_rem==1)={wt_tp0} tp1={wt_tp1}  (matched eval is tp0 ONLY)")
    log(f"TRAIN losses tp0={lt_tp0} tp1={lt_tp1}")
    # how many train wins fall inside the matched stone support AND tp0?
    wt_tp0_mask = (wt["tp2"] < 0.5) & (wt["stones"] >= lo) & (wt["stones"] <= hi)
    lt_tp0_mask = (lt["tp2"] < 0.5) & (lt["stones"] >= lo) & (lt["stones"] <= hi)
    log(f"TRAIN wins in matched stratum (tp0 & stones[{lo:.0f},{hi:.0f}])={int(wt_tp0_mask.sum())} ; "
        f"TRAIN losses in matched stratum={int(lt_tp0_mask.sum())}")

    # DISCRIMINATOR EXPERIMENT: train the frozen value head ONLY on tp0 data
    # (matched distribution). If matched KILL-C jumps, ENTANGLED was a
    # train-distribution (under-power on tp0 wins) artifact, NOT feature conflation.
    net_tp0 = load_net(device); freeze_trunk(net_tp0); verify_freezing(net_tp0)
    train_vhead(net_tp0, lt["s"][lt["tp2"] < 0.5], wt["s"][wt["tp2"] < 0.5], device, epochs=200, seed=0)
    vlx = values(net_tp0, lh["s"], device); vwx = values(net_tp0, wh["s"], device)
    tp0_matched_ka = float((vlx[l_mask] < 0).mean()); tp0_matched_kc = float((vwx[w_mask] > 0).mean())
    log(f"[tp0-ONLY distill] matched KILL-A={tp0_matched_ka:.3f} KILL-C={tp0_matched_kc:.3f} "
        f"(train on {int((wt['tp2']<0.5).sum())} tp0 wins + {int((lt['tp2']<0.5).sum())} tp0 losses)")

    # capacity check: can the head fit the matched TRAIN wins it is given? (train KILL-C on tp0 train wins)
    tr_kc_tp0 = float((vwx[(wt["tp2"] < 0.5)][:0] > 0).mean()) if False else None
    # measure train-set fit on the tp0 training wins themselves
    vw_train = values(net_tp0, wt["s"], device)
    fit_kc_train_tp0 = float((vw_train[wt["tp2"] < 0.5] > 0).mean())
    vl_train = values(net_tp0, lt["s"], device)
    fit_ka_train_tp0 = float((vl_train[lt["tp2"] < 0.5] < 0).mean())
    log(f"[tp0-ONLY distill] TRAIN-fit on tp0 wins KILL-C={fit_kc_train_tp0:.3f} "
        f"tp0 losses KILL-A={fit_ka_train_tp0:.3f} (can the head even fit its own tp0 train wins?)")
    R["axis3_underpowered"] = {
        "train_wins_tp0": wt_tp0, "train_wins_tp1": wt_tp1,
        "train_losses_tp0": lt_tp0, "train_losses_tp1": lt_tp1,
        "train_wins_in_matched_stratum": int(wt_tp0_mask.sum()),
        "train_losses_in_matched_stratum": int(lt_tp0_mask.sum()),
        "tp0only_distill_matched_kill_a": round(tp0_matched_ka, 4),
        "tp0only_distill_matched_kill_c": round(tp0_matched_kc, 4),
        "tp0only_trainfit_kill_c_on_tp0_wins": round(fit_kc_train_tp0, 4),
        "tp0only_trainfit_kill_a_on_tp0_losses": round(fit_ka_train_tp0, 4),
    }

    # =====================================================================
    # AXIS 4 — EFFECTIVE-N (distinct games)
    # =====================================================================
    log("\n" + "#" * 30 + " AXIS 4: EFFECTIVE-N " + "#" * 30)
    # distinct games among matched holdout wins (n=34)
    matched_win_meta = [wh_meta[i] for i in range(len(wh_meta)) if w_mask[i]]
    matched_win_games = set(m[1] for m in matched_win_meta)
    all_win_games = set(m[1] for m in wh_meta)
    # distinct board sigs among matched holdout wins
    matched_win_sigs = set(board_sig(wh["s"][i]) for i in range(len(wh["s"])) if w_mask[i])
    log(f"holdout WIN total: n={len(wh_meta)} distinct_games={len(all_win_games)}")
    log(f"MATCHED holdout WIN: n={int(w_mask.sum())} distinct_games={len(matched_win_games)} "
        f"distinct_boards={len(matched_win_sigs)}")
    # per-game copy count
    from collections import Counter
    gc = Counter(m[1] for m in matched_win_meta)
    log(f"MATCHED win per-game counts: {dict(sorted(gc.items(), key=lambda x:-x[1])[:8])} ... "
        f"max_copies={max(gc.values()) if gc else 0}")
    # matched holdout losses distinct games
    matched_loss_games = set()
    lh_idx_in_mask = np.where(l_mask)[0]
    # map matched loss states -> game via concat join
    for i in lh_idx_in_mask:
        ci = lh_cidx[i]
        if ci >= 0 and ci in ds1:
            matched_loss_games.add(ds1[ci][0])
    log(f"MATCHED holdout LOSS: n={int(l_mask.sum())} distinct_games(joined)={len(matched_loss_games)}")
    R["axis4_effective_n"] = {
        "matched_win_n": int(w_mask.sum()), "matched_win_distinct_games": len(matched_win_games),
        "matched_win_distinct_boards": len(matched_win_sigs),
        "matched_win_max_copies_per_game": max(gc.values()) if gc else 0,
        "matched_loss_n": int(l_mask.sum()), "matched_loss_distinct_games": len(matched_loss_games),
        "all_holdout_win_distinct_games": len(all_win_games),
    }

    R["wall_s"] = round(time.time() - t0, 1)
    with open(OUT / "E1_redteam_results.json", "w") as f:
        json.dump(R, f, indent=2)
    log("\nJSON -> " + str(OUT / "E1_redteam_results.json"))
    log(f"DONE {R['wall_s']}s")
    print("REDTEAM_JSON " + json.dumps(R))


if __name__ == "__main__":
    main()
