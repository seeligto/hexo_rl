#!/usr/bin/env python3
"""D-PFIT P1b — Deliverable (A): frozen-trunk POLICY-only head fit + CV + collateral.

The dispatcher's literal probe: "can the frozen-trunk policy head REPRESENT the
solver-proven saving moves, and does it generalize game-disjoint without
collateral damage?". P1a already showed the RAW head ranks the saving move well
(median rank 4); this stage drives the head to a NEAR-ONE-HOT prior on the saving
move (the policy-route upper bound consumed by `dpfit_search_mechanism.py`).

WHAT IS FROZEN (policy-only fit — per the P1b spec)
---------------------------------------------------
Train ONLY `policy_conv.*` + `policy_fc.*`. Freeze `trunk.*`, `value_fc*`, and
EVERY aux head (`opp_reply_*`, `value_var`, `ownership_head`, `threat_head`,
`chain_head`, `ply_index_head`, `cluster_pool`, `global_encoder`,
`gpool_bias_branch`). The value head stays BLIND by construction — the whole
point of (B) is to ask whether a boosted policy prior survives the blind-value
deploy search.

FAST FIT (frozen trunk ⇒ cache trunk features once)
---------------------------------------------------
The trunk is frozen, so its output `out=(N,C,19,19)` is constant across epochs
and across every LOO fold. We cache it ONCE per bucket (base net) and fit the
head on the cached features. The head forward replicates
`min_max_window_head`'s policy branch byte-for-byte:
    log_policy = log_softmax( policy_fc( relu(policy_conv(out)).flatten(1) ) )
fp32 CPU (no autocast — autocast fp16 shifts mass).

GAME-DISJOINT CV
----------------
All 32 in-window traps are distinct game_id ⇒ leave-one-out over games == LOO
over traps. Fits are PER-BUCKET (each bucket has its own base net): a held-out
trap in bucket b is scored under net_b fitted on bucket-b's OTHER traps.

COLLATERAL (KILL-C analog)
--------------------------
Normal (non-trap) positions = ancestor truncations of each bucket's trap
parent move-seqs (real same-game, same-bucket positions, definitely non-trap).
Measure policy top-1 agreement + mean KL(old‖new) before vs after the
in-sample fit. LOW collateral = top-1 preserved on ≥0.85 + small KL.

Run: .venv/bin/python scripts/dpfit_fit.py            (Deliverable A; writes fits + report)
Consumed by: scripts/dpfit_search_mechanism.py (Deliverable B).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import engine  # noqa: E402

from scripts.eval.gumbel_greedy_bot import _build_engine  # noqa: E402
from scripts.dpfit_export import (  # noqa: E402
    replay, primary_planes, ENC, S, N_CELLS, BUCKET_CKPT, CKPT_DIR, DEFAULT_CORPUS,
)

NPZ = REPO_ROOT / "data" / "dpfit_traps.npz"
FIT_DIR = Path("/tmp/claude-1000/-home-timmy-Work-Hexo-hexo-rl/"
               "9afbb966-16fe-4ad6-af0e-a5a86db4689a/scratchpad/dpfit_fits")
BUCKETS = ("s150k", "s175k", "s200k")

# Head params that ARE trained. Everything else is frozen.
_HEAD_PREFIXES = ("policy_conv.", "policy_fc.")


# ---------------------------------------------------------------------------
# freeze / head-forward / fit
# ---------------------------------------------------------------------------

def freeze_policy_only(model) -> Dict[str, List[str]]:
    """requires_grad True ONLY for policy_conv.* / policy_fc.*; all else False."""
    trainable, frozen = [], []
    for name, p in model.named_parameters():
        if name.startswith(_HEAD_PREFIXES):
            p.requires_grad_(True)
            trainable.append(name)
        else:
            p.requires_grad_(False)
            frozen.append(name)
    return {"trainable": trainable, "frozen": frozen}


@torch.no_grad()
def cache_trunk_out(model, planes: np.ndarray, device: torch.device) -> torch.Tensor:
    """(N,C,19,19) frozen trunk features for the v6 (has_pass) single-window path.

    Replicates HexTacToeNet.forward up to the head: in_channels==4 here so no
    index_select; has_pass_slot ⇒ mask=None. Detached fp32.
    """
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(planes)).float().to(device)
    if model._input_channels is not None:
        x = x.index_select(1, model.input_channel_index)
    out = model.trunk(x, mask=None, mask_sum_hw=None)
    return out.detach()


def head_logp(model, out: torch.Tensor) -> torch.Tensor:
    """Policy log-softmax from cached trunk `out` — byte-identical to
    min_max_window_head's policy branch."""
    p = F.relu(model.policy_conv(out)).flatten(1)
    logits = model.policy_fc(p)
    return F.log_softmax(logits, dim=1)


def saving_mass_rank(logp: torch.Tensor, target_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row mass + rank (0=top) on the saving-move index."""
    prob = logp.exp().double().cpu().numpy()
    masses = np.array([prob[i, target_idx[i]] for i in range(len(target_idx))])
    ranks = np.array([int((prob[i] > prob[i, target_idx[i]]).sum()) for i in range(len(target_idx))])
    return masses, ranks


def fit_head(model, cached_out: torch.Tensor, target_idx: np.ndarray,
             epochs: int, lr: float, target_mass: float = 0.97,
             verbose: bool = False) -> Dict[str, Any]:
    """Fit policy_conv+policy_fc (frozen trunk) on one-hot saving targets.

    Aggressive (near-one-hot) by design — the policy-route UPPER BOUND. Returns
    history + final in-sample saving mass/rank. Mutates `model` in place.
    """
    freeze_policy_only(model)
    model.train()  # only affects dropout (none in head); GroupNorm uses no running stats
    tgt = torch.from_numpy(target_idx.astype(np.int64)).to(cached_out.device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        logp = head_logp(model, cached_out)
        loss = F.nll_loss(logp, tgt)
        loss.backward()
        opt.step()
        if (ep + 1) % 25 == 0 or ep == 0 or ep == epochs - 1:
            with torch.no_grad():
                m, r = saving_mass_rank(head_logp(model, cached_out), target_idx)
            hist.append((ep + 1, float(loss.item()), float(m.mean()), int((r == 0).sum())))
            if verbose:
                print(f"      ep{ep+1:4d} loss={loss:.4f} mean_mass={m.mean():.3f} "
                      f"rank0={int((r==0).sum())}/{len(target_idx)}")
            if m.mean() >= target_mass:
                break
    model.eval()
    with torch.no_grad():
        m, r = saving_mass_rank(head_logp(model, cached_out), target_idx)
    return {"history": hist, "insample_mass": m, "insample_rank": r}


def head_state(model) -> Dict[str, torch.Tensor]:
    """Just the trained head params (tiny) — for cheap LOO head-swap."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if k.startswith(_HEAD_PREFIXES)}


def restore_head(model, head_sd: Dict[str, torch.Tensor]) -> None:
    """Load a saved head snapshot back into model (policy_conv/policy_fc only).

    The model carries an unpicklable Rust `_spec`, so deepcopy is impossible;
    rebuilding from ckpt is the fresh-model path and snapshot/restore is the
    cheap reset between LOO folds (trunk frozen ⇒ only the head changes)."""
    model.load_state_dict(head_sd, strict=False)


# ---------------------------------------------------------------------------
# collateral normal positions (ancestor truncations of bucket trap seqs)
# ---------------------------------------------------------------------------

def collateral_planes(corpus_by_pos: Dict[str, Any], bucket: str, pos_ids: List[str],
                      stride: int = 2, floor: int = 16, cap: int = 80) -> np.ndarray:
    """Non-trap same-bucket positions: ancestor truncations of trap parent seqs.

    For each bucket trap with parent_move_seq length L, take boards at lengths
    L-stride, L-2*stride, ... down to `floor`. Dedup by stone-set. fp32 (M,4,19,19).
    """
    seen = set()
    frames = []
    for pid in pos_ids:
        seq = corpus_by_pos[pid]["parent_move_seq"]
        L = len(seq)
        for cut in range(L - stride, floor - 1, -stride):
            sub = seq[:cut]
            key = tuple((int(q), int(r)) for q, r in sub)
            if key in seen:
                continue
            seen.add(key)
            b = replay(sub)
            frames.append(primary_planes(b))
            if len(frames) >= cap:
                return np.stack(frames).astype(np.float32)
    if not frames:
        return np.zeros((0, 4, S, S), dtype=np.float32)
    return np.stack(frames).astype(np.float32)


def collateral_metrics(base_model, fitted_model, planes: np.ndarray,
                       device: torch.device) -> Dict[str, float]:
    """top-1 agreement + mean KL(old‖new) over collateral positions."""
    if len(planes) == 0:
        return {"n": 0, "top1_agree": float("nan"), "mean_kl": float("nan")}
    with torch.no_grad():
        out = cache_trunk_out(base_model, planes, device)  # frozen trunk same for both
        lp_old = head_logp(base_model, out).double()
        lp_new = head_logp(fitted_model, out).double()
        top1 = (lp_old.argmax(1) == lp_new.argmax(1)).float().mean().item()
        # KL(old||new) = sum p_old (logp_old - logp_new)
        kl = (lp_old.exp() * (lp_old - lp_new)).sum(1)
        return {"n": int(len(planes)), "top1_agree": float(top1),
                "mean_kl": float(kl.mean().item()), "max_kl": float(kl.max().item())}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_traps() -> Dict[str, Any]:
    d = np.load(NPZ, allow_pickle=True)
    inw = d["in_window"]
    return {
        "planes": d["planes"][inw].astype(np.float32),
        "target_idx": d["target_idx"][inw].astype(np.int64),
        "pos_id": d["pos_id"][inw].astype(str),
        "bucket": d["bucket"][inw].astype(str),
        "game_id": d["game_id"][inw].astype(str),
        "band": d["depth_band"][inw].astype(str),
    }


def load_corpus_by_pos() -> Dict[str, Any]:
    recs = [json.loads(l) for l in DEFAULT_CORPUS.read_text().splitlines() if l.strip()]
    return {r["pos_id"]: r for r in recs if r.get("is_proven_core")}


def base_model_for(bucket: str, device: torch.device):
    eng = _build_engine(str(CKPT_DIR / BUCKET_CKPT[bucket]), ENC, device)
    return eng.model


# ---------------------------------------------------------------------------
# Deliverable (A) driver
# ---------------------------------------------------------------------------

def run_A(epochs: int, lr: float, save_fits: bool) -> Dict[str, Any]:
    device = torch.device("cpu")  # fp32 deterministic
    traps = load_traps()
    corpus = load_corpus_by_pos()
    FIT_DIR.mkdir(parents=True, exist_ok=True)

    freeze_summary = None
    per_pos = {}        # pos_id -> dict(insample/heldout mass+rank)
    collat = {}
    fit_artifacts = {}  # bucket -> {insample_head, loo_head: {pos_id: head_state}}

    for bk in BUCKETS:
        idx = np.where(traps["bucket"] == bk)[0]
        planes = traps["planes"][idx]
        tgt = traps["target_idx"][idx]
        pids = list(traps["pos_id"][idx])
        n = len(idx)
        print(f"\n[bucket {bk}] n_in_window={n}")

        base = base_model_for(bk, device)        # untouched reference (original head)
        if freeze_summary is None:
            freeze_summary = freeze_policy_only(base)
        init_head = head_state(base)              # original head snapshot
        cached = cache_trunk_out(base, planes, device)  # frozen trunk feats (LOO reuse)

        # ---- in-sample fit on ALL bucket traps ----
        ins_model = base_model_for(bk, device)    # fresh from ckpt (deepcopy impossible: Rust _spec)
        ins_cached = cache_trunk_out(ins_model, planes, device)
        res = fit_head(ins_model, ins_cached, tgt, epochs, lr, verbose=True)
        ins_mass, ins_rank = res["insample_mass"], res["insample_rank"]
        print(f"  IN-SAMPLE: mean_mass={ins_mass.mean():.3f} "
              f"rank0={int((ins_rank==0).sum())}/{n} "
              f"medmass={np.median(ins_mass):.3f}")

        # collateral on the in-sample fit (UPPER-BOUND / near-one-hot — the (B) engine)
        col_planes = collateral_planes(corpus, bk, pids)
        cm = collateral_metrics(base, ins_model, col_planes, device)
        # secondary: MODERATE fit (early-stop mean_mass~0.5, just enough to lift
        # rank to ~0 without near-one-hot saturation) — fairer KILL-C read of what
        # a realistic soft-target training-z would cost.
        mod_model = base_model_for(bk, device)
        mod_cached = cache_trunk_out(mod_model, planes, device)
        fit_head(mod_model, mod_cached, tgt, epochs, lr, target_mass=0.50)
        with torch.no_grad():
            mm, mr = saving_mass_rank(head_logp(mod_model, mod_cached), tgt)
        cm_mod = collateral_metrics(base, mod_model, col_planes, device)
        cm["moderate"] = cm_mod
        cm["moderate_insample_rank0"] = int((mr == 0).sum())
        cm["moderate_mean_mass"] = float(mm.mean())
        collat[bk] = cm
        print(f"  COLLATERAL[upper-bound] n={cm['n']} top1_agree={cm['top1_agree']:.3f} "
              f"mean_KL={cm['mean_kl']:.4f} max_KL={cm['max_kl']:.4f}")
        print(f"  COLLATERAL[moderate mass={cm['moderate_mean_mass']:.2f} "
              f"rank0={cm['moderate_insample_rank0']}/{n}] "
              f"top1_agree={cm_mod['top1_agree']:.3f} mean_KL={cm_mod['mean_kl']:.4f}")

        # ---- LOO held-out (game-disjoint) ----
        # Reuse ONE work-model; reset its head to the original snapshot per fold
        # (frozen trunk ⇒ cached `out` from base is valid for every fold).
        work = base_model_for(bk, device)
        loo_heads = {}
        ho_mass = np.zeros(n)
        ho_rank = np.zeros(n, dtype=int)
        for j in range(n):
            restore_head(work, init_head)
            keep = np.array([k for k in range(n) if k != j])
            fit_head(work, cached[keep], tgt[keep], epochs, lr)
            # score the held-out trap (cached[j] from the frozen trunk)
            with torch.no_grad():
                m, r = saving_mass_rank(head_logp(work, cached[j:j + 1]), tgt[j:j + 1])
            ho_mass[j], ho_rank[j] = float(m[0]), int(r[0])
            loo_heads[pids[j]] = head_state(work)

        print(f"  HELD-OUT(LOO): mean_mass={ho_mass.mean():.3f} "
              f"rank0={int((ho_rank==0).sum())}/{n} rank<10={int((ho_rank<10).sum())}/{n} "
              f"medrank={int(np.median(ho_rank))}")

        for j, pid in enumerate(pids):
            per_pos[pid] = {
                "bucket": bk, "band": str(traps["band"][idx][j]),
                "insample_mass": float(ins_mass[j]), "insample_rank": int(ins_rank[j]),
                "heldout_mass": float(ho_mass[j]), "heldout_rank": int(ho_rank[j]),
                "raw_rank": None,  # filled from baseline if available
            }

        # ---- save fit artifacts for Deliverable B ----
        if save_fits:
            torch.save({"model_state": ins_model.state_dict()},
                       FIT_DIR / f"insample_{bk}.pt")
            torch.save(loo_heads, FIT_DIR / f"loo_heads_{bk}.pt")
        fit_artifacts[bk] = {"insample_ckpt": str(FIT_DIR / f"insample_{bk}.pt"),
                             "loo_heads_ckpt": str(FIT_DIR / f"loo_heads_{bk}.pt")}

    out = {"per_pos": per_pos, "collateral": collat,
           "fit_artifacts": fit_artifacts,
           "n_trainable": len(freeze_summary["trainable"]),
           "n_frozen": len(freeze_summary["frozen"]),
           "trainable_params": freeze_summary["trainable"]}
    (FIT_DIR / "fit_A_results.json").write_text(json.dumps(out, indent=2, default=str))
    _print_A_summary(out, traps)
    return out


def _print_A_summary(out: Dict[str, Any], traps: Dict[str, Any]) -> None:
    pp = out["per_pos"]
    print("\n" + "=" * 78)
    print("DELIVERABLE (A) — frozen-trunk policy-only fit")
    print(f"  trainable params: {out['n_trainable']}  ({', '.join(out['trainable_params'])})")
    print(f"  frozen params:    {out['n_frozen']}")

    def agg(rows, key_m, key_r):
        m = np.array([r[key_m] for r in rows]); r = np.array([r[key_r] for r in rows])
        return (f"n={len(rows)} rank0={int((r==0).sum())}/{len(rows)} "
                f"rank<10={int((r<10).sum())}/{len(rows)} "
                f"medrank={int(np.median(r))} meanmass={m.mean():.3f}")

    allrows = list(pp.values())
    print(f"\n  IN-SAMPLE : {agg(allrows,'insample_mass','insample_rank')}")
    print(f"  HELD-OUT  : {agg(allrows,'heldout_mass','heldout_rank')}")
    for bk in BUCKETS:
        rows = [r for r in allrows if r["bucket"] == bk]
        print(f"    [{bk}] in-sample {agg(rows,'insample_mass','insample_rank')}")
        print(f"    [{bk}] held-out  {agg(rows,'heldout_mass','heldout_rank')}")

    print("\n  COLLATERAL (in-sample fit; bar: top1>=0.85):")
    for bk, cm in out["collateral"].items():
        flag = "OK" if cm["top1_agree"] >= 0.85 else "HIGH"
        print(f"    [{bk}] n={cm['n']} top1_agree={cm['top1_agree']:.3f} [{flag}] "
              f"mean_KL={cm['mean_kl']:.4f} max_KL={cm['max_kl']:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--no-save", action="store_true", help="skip saving fit artifacts")
    args = ap.parse_args()
    run_A(args.epochs, args.lr, save_fits=not args.no_save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
