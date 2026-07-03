#!/usr/bin/env python3
"""D-PFIT L1 — MULTI-WINDOW-AWARE frozen-trunk policy fit + deploy-search-flip.

THE cheap laptop architecture test of the L1 decode/policy wager
(`docs/designs/coupled_valuez_decode_design.md` §0/§1.3). L1 bets that
MULTI-WINDOW no-drop decode recovers the 20/32 single-window-STRANDED
saving-move priors (P1b deploy prior 0.139), lifting the game-disjoint
held-out deploy-search trap-flip ABOVE the P1b single-window held-out ceiling
(16%) toward the clean-transfer 42% / oracle 69%.

SPLIT (this script answers the ARCHITECTURE half, no GPU, no self-play):
  can ANY fit that targets the MULTI-WINDOW DECODED root prior reach the saving
  move on the stranded-20, where the single-window NLL fit (P1b) could NOT?

vs P1b (`scripts/dpfit_fit.py` / `scripts/dpfit_search_mechanism.py`):
  P1b fit the SINGLE-PRIMARY-window logit at `to_flat(saving)` (NLL). For a
  STRANDED trap (primary window is NOT a deploy cluster center) that logit is in
  a frame the deploy scatter-max never reads → deploy prior stranded at 0.139.
  HERE we backprop through the ACTUAL deploy decode: per-cluster head forward →
  softmax → SCATTER-MAX over the K deploy clusters → renormalise over the legal
  set → push the DECODED root prior on the saving move high. Scatter-max is
  differentiable like maxpool (gradient flows to the argmax covering cluster).
  Frozen trunk + value + aux; train ONLY policy_conv / policy_fc. fp32 CPU.

UPPER-BOUND framing: a DIRECT supervised fit on the decoded root prior is a
STRONGER training signal than self-play visit-injection can realise, so the
held-out flip here UPPER-BOUNDS the L1 GPU smoke. If even this can't lift
held-out flip above 16% → the decode-recovers-stranded wager is ARCHITECTURALLY
FALSE (deploy-backup carries those traps; don't fund the L1 GPU smoke). If it
lifts → the wager is architecturally sound, GPU smoke justified.

VALUE-VETO exclusion: the RED-TEAM (`redteam_forced_prior_rows.json`) FORCED the
prior to ~1.0 at the deploy root; traps that STILL didn't flip (frc != 'saving')
are VALUE-bound, NOT decode-stranded → NOT recoverable by L1 → excluded from the
L1-attributable lift.

Deploy decode = the TRUE multi-window no-drop legal-set path
(`run_gumbel_on_board(..., legal_set=True, gumbel_scale=0.0)`; the d1m deploy
head). P1b's deploy used `legal_set=False` (single-window dense) — reproduced
here as the comparison anchor.

Run: .venv/bin/python scripts/dpfit_l1_mwfit_probe.py
CPU/laptop, ~15-25 min. Commits nothing. Report -> reports/d_tactical_2026-06-26/.
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

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import engine  # noqa: E402

from hexo_rl.env.game_state import GameState  # noqa: E402
from scripts.eval.gumbel_greedy_bot import _build_engine  # noqa: E402
from hexo_rl.eval.gumbel_search_py import run_gumbel_on_board  # noqa: E402
from scripts.dpfit_export import (  # noqa: E402
    replay, ENC, CKPT_DIR, BUCKET_CKPT, DEFAULT_CORPUS, window_center,
    N_CELLS, S, HALF,
)
from scripts.dpfit_fit import (  # noqa: E402
    cache_trunk_out, head_logp, freeze_policy_only, head_state, restore_head,
    _HEAD_PREFIXES,
)

NPZ = REPO_ROOT / "data" / "dpfit_traps.npz"
REDTEAM = REPO_ROOT / "reports" / "d_tactical_2026-06-26" / "redteam_forced_prior_rows.json"
REPORT = REPO_ROOT / "reports" / "d_tactical_2026-06-26" / "l1_mwfit_probe_report.md"
OUT_DIR = Path("/tmp/claude-1000/-home-timmy-Work-Hexo-hexo-rl/"
               "9afbb966-16fe-4ad6-af0e-a5a86db4689a/scratchpad/dpfit_l1")
BUCKETS = ("s150k", "s175k", "s200k")

# Deploy knobs — match run_t0 / deploy_strength_eval (gumbel_scale=0.0 deterministic head).
DEPLOY = dict(n_sims=150, m=16, c_visit=50.0, c_scale=1.0, c_puct=1.5,
              dirichlet=False, gumbel_scale=0.0)


# ---------------------------------------------------------------------------
# per-trap multi-window data (cluster planes + scatter-max coverage map)
# ---------------------------------------------------------------------------

def cluster_planes(board, spec, in_channels) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """(K,in_channels,19,19) cluster frames + K centers, matching infer_batch."""
    st = GameState.from_board(board)
    tensor, centers = st.to_tensor()
    if tensor.shape[1] != in_channels:
        tensor = tensor[:, list(spec.kept_plane_indices)]
    centers = [(int(c[0]), int(c[1])) for c in centers]
    return tensor.astype(np.float32), centers


def coverage_map(board, centers: List[Tuple[int, int]], n_actions: int,
                 saving: Tuple[int, int]) -> Dict[str, Any]:
    """Build the legal-move x cluster scatter-max coverage for the deploy decode.

    Replicates infer_batch: legal moves with a valid to_flat (in the global
    window) form the renorm set; each move's scatter-max is over the clusters
    whose window covers it. Returns torch index/mask tensors + the saving row.
    """
    K = len(centers)
    legal = list(board.legal_moves())
    rows = []       # (q,r) with valid to_flat
    gather_idx = []  # per move: list length K of local idx (0 where uncovered)
    cover = []       # per move: list length K of 1.0/0.0
    saving_row = -1
    saving_flat = board.to_flat(*saving)
    for (q, r) in legal:
        idx = board.to_flat(q, r)
        if idx >= n_actions - 1:
            continue
        gi = [0] * K
        cm = [0.0] * K
        for k, (cq, cr) in enumerate(centers):
            wq, wr = q - cq + HALF, r - cr + HALF
            if 0 <= wq < S and 0 <= wr < S:
                gi[k] = wq * S + wr
                cm[k] = 1.0
        if idx == saving_flat:
            saving_row = len(rows)
        rows.append((int(q), int(r)))
        gather_idx.append(gi)
        cover.append(cm)
    assert saving_row >= 0, "saving move not in legal/in-window set"
    return {
        "gather_idx": torch.tensor(gather_idx, dtype=torch.long),   # (L,K)
        "cover_mask": torch.tensor(cover, dtype=torch.float64),     # (L,K)
        "saving_row": saving_row,
        "n_legal": len(rows),
        "saving_flat": int(saving_flat),
    }


def decoded_dist(probs_K: torch.Tensor, cov: Dict[str, Any]) -> torch.Tensor:
    """Differentiable scatter-max decode → renormalised global policy over the
    legal set. ``probs_K`` (K,362) per-cluster softmax probs (grad-carrying).

    g[l] = max_k cover[l,k] * probs_K[k, gather_idx[l,k]]  (scatter-max, maxpool-grad)
    G    = g / sum_l g                                      (renorm over legal)
    """
    gi = cov["gather_idx"]                       # (L,K)
    cm = cov["cover_mask"].to(probs_K.dtype)     # (L,K)
    # gathered[l,k] = probs_K[k, gi[l,k]]  via gather on (K,362) then transpose.
    g_kl = torch.gather(probs_K, 1, gi.t())      # (K,L): probs_K[k, gi.t()[k,l]]
    g_lk = g_kl.t() * cm                          # (L,K), uncovered -> 0
    g = g_lk.max(dim=1).values                    # (L,) scatter-max
    G = g / g.sum().clamp_min(1e-12)
    return G


def decoded_sm_np(model, feats_K: torch.Tensor, cov: Dict[str, Any]) -> float:
    """Measure the decoded root prior on the saving move (no grad)."""
    with torch.no_grad():
        logp = head_logp(model, feats_K)          # (K,362)
        probs = logp.exp().double()
        G = decoded_dist(probs, cov)
        return float(G[cov["saving_row"]].item())


# ---------------------------------------------------------------------------
# multi-window-aware fit (frozen trunk; train policy_conv/policy_fc only)
# ---------------------------------------------------------------------------

def fit_mw(model, train_entries: List[Dict[str, Any]], epochs: int, lr: float,
           target_mass: float = 0.97, verbose: bool = False) -> Dict[str, Any]:
    """Fit the shared policy head so the DECODED multi-window root prior on each
    train trap's saving move is high. Mutates ``model`` in place.

    train_entries: list of {feats (K,Ctrunk,19,19 cached), cov}.
    """
    freeze_policy_only(model)
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        # Concatenate all clusters across train traps -> one head forward.
        feats_all = torch.cat([e["feats"] for e in train_entries], dim=0)
        logp_all = head_logp(model, feats_all)        # (sumK,362) log_softmax
        probs_all = logp_all.exp().double()
        loss = torch.zeros((), dtype=torch.float64)
        off = 0
        sms = []
        for e in train_entries:
            K = e["feats"].shape[0]
            probs_K = probs_all[off:off + K]
            off += K
            G = decoded_dist(probs_K, e["cov"])
            sm = G[e["cov"]["saving_row"]].clamp_min(1e-12)
            loss = loss - torch.log(sm)
            sms.append(float(sm.item()))
        loss = loss / len(train_entries)
        loss.backward()
        opt.step()
        mean_sm = float(np.mean(sms))
        if (ep + 1) % 50 == 0 or ep == 0 or ep == epochs - 1:
            hist.append((ep + 1, float(loss.item()), mean_sm))
            if verbose:
                print(f"      ep{ep+1:4d} loss={loss:.4f} mean_decoded_sm={mean_sm:.3f}")
        if mean_sm >= target_mass:
            break
    model.eval()
    return {"history": hist}


# ---------------------------------------------------------------------------
# data load
# ---------------------------------------------------------------------------

def load_all() -> Dict[str, Any]:
    d = np.load(NPZ, allow_pickle=True)
    inw = d["in_window"]
    recs = [json.loads(l) for l in DEFAULT_CORPUS.read_text().splitlines() if l.strip()]
    corpus = {r["pos_id"]: r for r in recs if r.get("is_proven_core")}
    redteam = {r["pos_id"]: r for r in json.load(REDTEAM.open())}
    return {
        "pos_id": d["pos_id"][inw].astype(str),
        "bucket": d["bucket"][inw].astype(str),
        "band": d["depth_band"][inw].astype(str),
        "corpus": corpus,
        "redteam": redteam,
    }


def build_trap_data(meta, engines) -> Dict[str, Dict[str, Any]]:
    """Per-trap: board, cluster feats (cached frozen trunk), coverage, flags."""
    out = {}
    for i, pid in enumerate(meta["pos_id"]):
        bk = meta["bucket"][i]
        eng = engines[bk]
        model = eng.model
        spec = eng.encoding_spec
        r = meta["corpus"][pid]
        b = replay(r["parent_move_seq"])
        saving = (int(r["refuting_move"][0]), int(r["refuting_move"][1]))
        blunder = (int(r["blunder_move"][0]), int(r["blunder_move"][1]))
        planes, centers = cluster_planes(b, spec, model.in_channels)
        feats = cache_trunk_out(model, planes, torch.device("cpu"))  # (K,C,19,19) frozen
        cov = coverage_map(b, centers, model.n_actions, saving)
        wc = window_center(b)
        primc = wc in centers
        rt = meta["redteam"].get(pid, {})
        out[pid] = {
            "pid": pid, "bucket": bk, "band": str(meta["band"][i]),
            "board": b, "saving": saving, "blunder": blunder,
            "feats": feats, "cov": cov, "centers": centers, "K": len(centers),
            "primc": bool(primc),                       # primary IS a deploy cluster center
            "stranded": (not primc),                    # the multi-window lever's target set
            "frc": rt.get("frc"),                       # forced-prior oracle outcome
            "frc_recoverable": (rt.get("frc") == "saving"),
        }
    return out


# ---------------------------------------------------------------------------
# deploy search flip
# ---------------------------------------------------------------------------

def classify(played, saving, blunder) -> str:
    if played is None:
        return "none"
    pl = (int(played[0]), int(played[1]))
    if pl == saving:
        return "saving"
    if pl == blunder:
        return "blunder"
    return "other"


def deploy_flip(eng, board, saving, blunder, legal_set: bool) -> str:
    g = run_gumbel_on_board(eng, board, **DEPLOY, legal_set=legal_set,
                            rng=np.random.default_rng(0))
    return classify(g["played_move"], saving, blunder)


# ---------------------------------------------------------------------------
# collateral on normal (non-trap) positions
# ---------------------------------------------------------------------------

def normal_boards(corpus, bucket_pids, stride=6, floor=20, cap=10):
    seen = set()
    boards = []
    for pid in bucket_pids:
        seq = corpus[pid]["parent_move_seq"]
        L = len(seq)
        for cut in range(L - stride, floor - 1, -stride):
            sub = seq[:cut]
            key = tuple((int(q), int(r)) for q, r in sub)
            if key in seen:
                continue
            seen.add(key)
            b = replay(sub)
            if b.check_win() or b.legal_move_count() == 0:
                continue
            boards.append(b)
            if len(boards) >= cap:
                return boards
    return boards


def mw_decoded_full(model, board, spec) -> Tuple[np.ndarray, Optional[int]]:
    """Full multi-window decoded global policy (==infer_batch) for collateral."""
    planes, centers = cluster_planes(board, spec, model.in_channels)
    feats = cache_trunk_out(model, planes, torch.device("cpu"))
    with torch.no_grad():
        probs = head_logp(model, feats).exp().double().cpu().numpy()  # (K,362)
    n_actions = model.n_actions
    gp = np.zeros(n_actions)
    for (q, r) in board.legal_moves():
        idx = board.to_flat(q, r)
        if idx >= n_actions - 1:
            continue
        best = 0.0
        for k, (cq, cr) in enumerate(centers):
            wq, wr = q - cq + HALF, r - cr + HALF
            if 0 <= wq < S and 0 <= wr < S:
                p = probs[k, wq * S + wr]
                if p > best:
                    best = p
        gp[idx] = best
    s = gp.sum()
    if s > 1e-9:
        gp /= s
    top1 = int(gp.argmax()) if s > 1e-9 else None
    return gp, top1


def collateral(raw_model, fit_model, boards, raw_eng, fit_eng, spec) -> Dict[str, Any]:
    if not boards:
        return {"n": 0}
    top1_agree = 0
    kls = []
    deploy_disagree = 0
    for b in boards:
        gp_raw, t_raw = mw_decoded_full(raw_model, b, spec)
        gp_fit, t_fit = mw_decoded_full(fit_model, b, spec)
        if t_raw is not None and t_raw == t_fit:
            top1_agree += 1
        # KL(raw||fit) over the legal support
        mask = gp_raw > 1e-12
        kl = float(np.sum(gp_raw[mask] * (np.log(gp_raw[mask]) - np.log(np.clip(gp_fit[mask], 1e-12, None)))))
        kls.append(kl)
        pr = run_gumbel_on_board(raw_eng, b, **DEPLOY, legal_set=True, rng=np.random.default_rng(0))["played_move"]
        pf = run_gumbel_on_board(fit_eng, b, **DEPLOY, legal_set=True, rng=np.random.default_rng(0))["played_move"]
        if pr is not None and pf is not None and tuple(pr) != tuple(pf):
            deploy_disagree += 1
    n = len(boards)
    return {"n": n, "top1_agree": top1_agree / n, "mean_kl": float(np.mean(kls)),
            "deploy_disagree": deploy_disagree, "deploy_disagree_rate": deploy_disagree / n}


# ---------------------------------------------------------------------------
# main driver
# ---------------------------------------------------------------------------

def run(epochs: int, lr: float, collat_cap: int) -> Dict[str, Any]:
    device = torch.device("cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_all()
    engines = {bk: _build_engine(str(CKPT_DIR / BUCKET_CKPT[bk]), ENC, device) for bk in BUCKETS}
    spec = engines["s150k"].encoding_spec
    print("[build] caching frozen-trunk cluster features for 32 traps ...")
    traps = build_trap_data(meta, engines)
    n_strand = sum(1 for t in traps.values() if t["stranded"])
    n_recov_strand = sum(1 for t in traps.values() if t["stranded"] and t["frc_recoverable"])
    print(f"[build] 32 in-window: stranded={n_strand} (primc={32-n_strand}); "
          f"stranded&frc-recoverable={n_recov_strand}")

    per_pos: Dict[str, Dict[str, Any]] = {}

    for bk in BUCKETS:
        bk_pids = [pid for pid in traps if traps[pid]["bucket"] == bk]
        n = len(bk_pids)
        print(f"\n[bucket {bk}] n={n}")
        base = engines[bk].model
        init_head = head_state(base)

        # ---- raw decoded sm (== eng.infer) ----
        for pid in bk_pids:
            t = traps[pid]
            per_pos[pid] = {
                "bucket": bk, "band": t["band"], "stranded": t["stranded"],
                "primc": t["primc"], "frc": t["frc"],
                "frc_recoverable": t["frc_recoverable"], "K": t["K"],
                "sm_raw": decoded_sm_np(base, t["feats"], t["cov"]),
            }

        # ---- in-sample multi-window fit (all bucket traps) ----
        ins_model = _build_engine(str(CKPT_DIR / BUCKET_CKPT[bk]), ENC, device).model
        # re-cache feats under ins_model's (identical) frozen trunk == same numbers
        ins_entries = [{"feats": traps[pid]["feats"], "cov": traps[pid]["cov"]} for pid in bk_pids]
        res = fit_mw(ins_model, ins_entries, epochs, lr, verbose=True)
        for pid in bk_pids:
            per_pos[pid]["sm_ins"] = decoded_sm_np(ins_model, traps[pid]["feats"], traps[pid]["cov"])
        ins_mean = np.mean([per_pos[pid]["sm_ins"] for pid in bk_pids])
        print(f"  IN-SAMPLE decoded sm: mean={ins_mean:.3f} "
              f">0.5={sum(per_pos[pid]['sm_ins']>0.5 for pid in bk_pids)}/{n}")
        torch.save({"model_state": ins_model.state_dict()}, OUT_DIR / f"insample_{bk}.pt")

        # ---- LOO held-out (within-bucket, game-disjoint) ----
        work = _build_engine(str(CKPT_DIR / BUCKET_CKPT[bk]), ENC, device).model
        loo_heads = {}
        for j, pid in enumerate(bk_pids):
            restore_head(work, init_head)
            train = [{"feats": traps[p]["feats"], "cov": traps[p]["cov"]}
                     for p in bk_pids if p != pid]
            fit_mw(work, train, epochs, lr)
            per_pos[pid]["sm_ho"] = decoded_sm_np(work, traps[pid]["feats"], traps[pid]["cov"])
            loo_heads[pid] = head_state(work)
        torch.save(loo_heads, OUT_DIR / f"loo_heads_{bk}.pt")
        ho_mean = np.mean([per_pos[pid]["sm_ho"] for pid in bk_pids])
        print(f"  HELD-OUT decoded sm: mean={ho_mean:.3f} "
              f">0.5={sum(per_pos[pid]['sm_ho']>0.5 for pid in bk_pids)}/{n}")

    # ---- deploy-search flip (legal_set True = multi-window deploy; False = P1b anchor) ----
    print("\n[deploy] running gumbel SH flips (legal_set=True multi-window) ...")
    raw_eng = engines
    ins_eng = {bk: _build_engine(str(OUT_DIR / f"insample_{bk}.pt"), ENC, device) for bk in BUCKETS}
    work_eng = {bk: _build_engine(str(CKPT_DIR / BUCKET_CKPT[bk]), ENC, device) for bk in BUCKETS}
    loo_heads = {bk: torch.load(OUT_DIR / f"loo_heads_{bk}.pt", weights_only=False) for bk in BUCKETS}

    t0 = time.time()
    for i, pid in enumerate(meta["pos_id"]):
        t = traps[pid]
        bk = t["bucket"]
        b, sv, bl = t["board"], t["saving"], t["blunder"]
        # multi-window deploy (the L1 regime)
        per_pos[pid]["ctrl_ls"] = deploy_flip(raw_eng[bk], b, sv, bl, legal_set=True)
        per_pos[pid]["ins_ls"] = deploy_flip(ins_eng[bk], b, sv, bl, legal_set=True)
        restore_head(work_eng[bk].model, loo_heads[bk][pid])
        per_pos[pid]["ho_ls"] = deploy_flip(work_eng[bk], b, sv, bl, legal_set=True)
        # single-window deploy (P1b anchor; ctrl + insample only)
        per_pos[pid]["ctrl_sw"] = deploy_flip(raw_eng[bk], b, sv, bl, legal_set=False)
        per_pos[pid]["ins_sw"] = deploy_flip(ins_eng[bk], b, sv, bl, legal_set=False)
        if (i + 1) % 8 == 0:
            print(f"  deploy {i+1}/32  ({time.time()-t0:.0f}s)")

    # ---- collateral (multi-window decoded policy + legal_set deploy, raw vs in-sample) ----
    print("\n[collateral] normal-position multi-window decode raw vs in-sample-fit ...")
    collat = {}
    for bk in BUCKETS:
        bk_pids = [pid for pid in traps if traps[pid]["bucket"] == bk]
        nbs = normal_boards(meta["corpus"], bk_pids, cap=collat_cap)
        collat[bk] = collateral(raw_eng[bk].model, ins_eng[bk].model, nbs,
                                 raw_eng[bk], ins_eng[bk], spec)
        print(f"  [{bk}] {collat[bk]}")

    out = {"per_pos": per_pos, "collateral": collat,
           "n_stranded": n_strand, "n_recov_stranded": n_recov_strand}
    _report(out)
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _frac(rows, key, val="saving"):
    n = len(rows)
    k = sum(1 for r in rows if r.get(key) == val)
    return k, n


def _report(out: Dict[str, Any]) -> None:
    pp = out["per_pos"]
    rows = list(pp.values())
    n = len(rows)
    L = []
    L.append("# D-PFIT L1 — multi-window-aware fit + deploy-search-flip (laptop architecture probe)\n\n")
    L.append(f"Deploy knobs: {DEPLOY}; multi-window deploy = `legal_set=True` (d1m head). "
             f"32 in-window proven-core traps; per-bucket s150k/s175k/s200k nets; fp32 CPU.\n\n")
    L.append(f"Stranded (primary NOT a deploy cluster center) = {out['n_stranded']}/32; "
             f"of those, frc-recoverable (RED-TEAM forced prior flips) = {out['n_recov_stranded']}.\n\n")

    # ---- decoded root prior on the saving move ----
    L.append("## (a) Decoded multi-window root prior on the saving move (== eng.infer scatter-max)\n\n")
    L.append("| subset | n | raw mean | mw-fit IN-SAMPLE mean | mw-fit HELD-OUT mean | "
             "in-sample >0.5 | held-out >0.5 |\n|---|---|---|---|---|---|---|\n")

    def smrow(label, subset):
        if not subset:
            return
        rr = np.mean([r["sm_raw"] for r in subset])
        ri = np.mean([r["sm_ins"] for r in subset])
        rh = np.mean([r["sm_ho"] for r in subset])
        gi = sum(1 for r in subset if r["sm_ins"] > 0.5)
        gh = sum(1 for r in subset if r["sm_ho"] > 0.5)
        L.append(f"| {label} | {len(subset)} | {rr:.3f} | {ri:.3f} | {rh:.3f} | "
                 f"{gi}/{len(subset)} | {gh}/{len(subset)} |\n")

    smrow("ALL in-window", rows)
    smrow("STRANDED (primc=F)", [r for r in rows if r["stranded"]])
    smrow("STRANDED & frc-recoverable", [r for r in rows if r["stranded"] and r["frc_recoverable"]])
    smrow("primc=T (single==multi)", [r for r in rows if not r["stranded"]])
    L.append("\n*P1b single-window fit stranded the deploy prior at 0.139 (mean) on the 20 "
             "primc=F traps — the number this row tests against.*\n\n")

    # ---- deploy-search flip ----
    L.append("## (b) Deploy-search trap-flip (played move == saving)\n\n")
    L.append("### Multi-window deploy (`legal_set=True`, the L1 regime)\n\n")
    L.append("| condition | ALL 32 | STRANDED-20 | STRANDED&recoverable | primc-12 |\n|---|---|---|---|---|\n")

    def fliprow(label, key):
        def f(sub):
            k, m = _frac(sub, key)
            return f"{k}/{m} ({100*k/m:.0f}%)" if m else "-"
        allr = rows
        st = [r for r in rows if r["stranded"]]
        rec = [r for r in rows if r["stranded"] and r["frc_recoverable"]]
        pc = [r for r in rows if not r["stranded"]]
        L.append(f"| {label} | {f(allr)} | {f(st)} | {f(rec)} | {f(pc)} |\n")

    fliprow("CONTROL (raw net)", "ctrl_ls")
    fliprow("mw-fit IN-SAMPLE (upper bound)", "ins_ls")
    fliprow("mw-fit HELD-OUT (LOO, game-disjoint)", "ho_ls")
    L.append("\n### Single-window deploy (`legal_set=False`, P1b anchor)\n\n")
    L.append("| condition | ALL 32 |\n|---|---|\n")
    for label, key in [("CONTROL (raw)", "ctrl_sw"), ("mw-fit IN-SAMPLE", "ins_sw")]:
        k, m = _frac(rows, key)
        L.append(f"| {label} | {k}/{m} ({100*k/m:.0f}%) |\n")
    L.append("\n*P1b reference (single-window NLL fit, `legal_set=False`): control 4/32=12%, "
             "in-sample 10/32=31%, HELD-OUT 5/32=16%.*\n\n")

    # per-bucket held-out multi-window
    L.append("### Per-bucket multi-window held-out flip→saving (CONTROL → in-sample → held-out)\n\n")
    L.append("| bucket | n | control | in-sample | held-out |\n|---|---|---|---|---|\n")
    for bk in BUCKETS:
        br = [r for r in rows if r["bucket"] == bk]
        nb = len(br)
        c = sum(1 for r in br if r["ctrl_ls"] == "saving")
        ii = sum(1 for r in br if r["ins_ls"] == "saving")
        h = sum(1 for r in br if r["ho_ls"] == "saving")
        L.append(f"| {bk} | {nb} | {c}/{nb} | {ii}/{nb} | {h}/{nb} |\n")

    # ---- collateral ----
    L.append("\n## (c) Collateral — multi-window decoded policy raw vs in-sample mw-fit (normal positions)\n\n")
    L.append("| bucket | n | top1 agree | mean KL(raw‖fit) | deploy played disagree |\n|---|---|---|---|---|\n")
    for bk, cm in out["collateral"].items():
        if cm.get("n", 0) == 0:
            L.append(f"| {bk} | 0 | - | - | - |\n")
            continue
        L.append(f"| {bk} | {cm['n']} | {cm['top1_agree']:.3f} | {cm['mean_kl']:.3f} | "
                 f"{cm['deploy_disagree']}/{cm['n']} ({cm['deploy_disagree_rate']:.2f}) |\n")
    L.append("\n*P1b single-window collateral (KILL-C analog) FAILED: policy top1 0.24-0.34, "
             "deploy disagreement 0.60-1.00.*\n\n")

    # ---- per-position dump ----
    L.append("## Per-position (stranded subset)\n\n")
    L.append("| pos_id | bkt | band | K | frc | sm raw→ins→ho | ctrl_ls | ins_ls | ho_ls |\n"
             "|---|---|---|---|---|---|---|---|---|\n")
    for pid, r in sorted(pp.items()):
        if not r["stranded"]:
            continue
        L.append(f"| {pid} | {r['bucket']} | {r['band']} | {r['K']} | {r['frc']} | "
                 f"{r['sm_raw']:.2f}→{r['sm_ins']:.2f}→{r['sm_ho']:.2f} | "
                 f"{r['ctrl_ls']} | {r['ins_ls']} | {r['ho_ls']} |\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("".join(L))
    (OUT_DIR / "l1_results.json").write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "".join(L))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--collat-cap", type=int, default=10)
    args = ap.parse_args()
    run(args.epochs, args.lr, args.collat_cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
