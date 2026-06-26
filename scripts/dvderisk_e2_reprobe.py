#!/usr/bin/env python3
"""D-FULLSPEC E2 — frozen-random-trunk reprobe on v6_live2_ft (8-plane encoding).

Gating test BEFORE the bootstrap-restart commit. If the new 8-plane encoding
(v6_live2 + 4 domain threat planes) enables KILL-C >= 0.85 on the matched
holdout set with a FROZEN RANDOM trunk, the feature gap is confirmed and the
bootstrap-restart is justified.

Threat planes added (planes 4-7):
  4: cur_open_line_count   -- per-cell count of unblocked 6-windows with >=1 cur stone / 6
  5: opp_open_line_count   -- same for opponent / 6
  6: cur_best_window_fill  -- max cur-stone count over unblocked windows containing cell / 6
  7: opp_best_window_fill  -- max opp-stone count over unblocked windows containing cell / 6

Algorithm mirrors engine/src/board/threats.rs:check_window() but accumulates
per-cell aggregate statistics (count / max) over all windows, not just >=3-stone
windows on empty cells.

Pre-registered pass/fail:
  Phase 1 (random trunk):
    SEPARABLE-R : KILL-A > 0.35 AND matched KILL-C >= 0.85
    PARTIAL-R   : KILL-A > 0.35 AND 0.60 <= matched KILL-C < 0.85
    ENTANGLED-R : KILL-A > 0.35 AND matched KILL-C < 0.60

Data (read-only npz banks, states already (N,4,19,19)):
  data/distill_d7_train.npz    493 d7 LOSSES  target -1
  data/distill_d7_holdout.npz  108 d7 LOSSES  target -1
  data/killc_netwin_train.npz  465 net WINS   target +1
  data/killc_netwin_holdout.npz 100 net WINS  target +1

Usage:
    .venv/bin/python scripts/dvderisk_e2_reprobe.py [--epochs 400] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.utils.device import best_device

# Reuse E1 helpers.
from scripts.dvderisk_ds3_probe import freeze_trunk, verify_freezing

LOG_PATH = REPO_ROOT / "logs" / "dvderisk_e2_reprobe.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(str(LOG_PATH), mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threat plane computation (pure numpy, mirrors threats.rs semantics)
# ---------------------------------------------------------------------------

WIN_LEN = 6
HEX_AXES = [(1, 0), (0, 1), (1, -1)]   # (dq, dr) in window coords


def compute_threat_planes_single(cur: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """Compute 4 threat planes for a single 19x19 position.

    cur, opp: (19, 19) float32 binary stone masks.
    Returns (4, 19, 19) float32:
      [0] cur_open_line_count   /6
      [1] opp_open_line_count   /6
      [2] cur_best_window_fill  /6
      [3] opp_best_window_fill  /6

    Algorithm: slide all WIN_LEN=6 windows on each of 3 hex axes.
    For each window: check p_opp_count == 0 (unblocked for current player).
    Accumulate per-cell: count of such windows (open_line), max p_own_count (best_fill).

    Mirrors engine/src/board/threats.rs:scan_line() + check_window() window-scan
    iteration pattern. Uses window coordinates directly (no axial coord conversion
    needed since the 19x19 layout IS the window representation).

    Note on axis (1,-1): in window coordinates (row=wq, col=wr), this axis moves
    wq+1, wr-1 — i.e. row increases, col decreases. Standard numpy-compatible.
    """
    H, W = cur.shape   # 19, 19
    # 4 planes: [cur_olc, opp_olc, cur_bwf, opp_bwf]
    cur_olc = np.zeros((H, W), dtype=np.float32)
    opp_olc = np.zeros((H, W), dtype=np.float32)
    cur_bwf = np.zeros((H, W), dtype=np.float32)
    opp_bwf = np.zeros((H, W), dtype=np.float32)

    for dq, dr in HEX_AXES:
        # Iterate over all length-WIN_LEN windows along this axis.
        # Window start at (sq, sr); cells at (sq + t*dq, sr + t*dr) for t=0..5.
        # Q range for sq: such that all 6 cells stay in [0,H-1].
        # For dq=+1: sq in [0, H-WIN_LEN], so sq in [0, H-WIN_LEN] = [0, 13].
        # For dq=0: sq in [0, H-1].
        # For dq=-1: not used (we use positive direction only; HEX_AXES are positive).
        # Since dq,dr in {(1,0),(0,1),(1,-1)}, dq >= 0 always. dr can be -1.
        # For dr=-1: sr must be in [WIN_LEN-1, W-1] so sr_start = WIN_LEN-1..W-1.

        if dq == 1:
            q_range = range(0, H - WIN_LEN + 1)        # 0..13
        else:
            q_range = range(0, H)

        if dr == 1:
            r_range = range(0, W - WIN_LEN + 1)        # 0..13
        elif dr == -1:
            r_range = range(WIN_LEN - 1, W)            # 5..18
        else:  # dr == 0
            r_range = range(0, W)

        for sq in q_range:
            for sr in r_range:
                # Collect the 6 cells of this window.
                qs = [sq + t * dq for t in range(WIN_LEN)]
                rs = [sr + t * dr for t in range(WIN_LEN)]
                # All in-bounds by construction.
                cur_count = sum(cur[qs[t], rs[t]] for t in range(WIN_LEN))
                opp_count = sum(opp[qs[t], rs[t]] for t in range(WIN_LEN))

                # Update per-cell statistics for each cell in this window.
                for t in range(WIN_LEN):
                    qi, ri = qs[t], rs[t]
                    # Open window for current player: no opp stones, >=1 cur stone.
                    if opp_count == 0 and cur_count >= 1:
                        cur_olc[qi, ri] += 1.0
                    # Best window fill for current player (unblocked by opp).
                    if opp_count == 0 and cur_count > cur_bwf[qi, ri]:
                        cur_bwf[qi, ri] = cur_count
                    # Open window for opponent: no cur stones, >=1 opp stone.
                    if cur_count == 0 and opp_count >= 1:
                        opp_olc[qi, ri] += 1.0
                    # Best window fill for opponent (unblocked by cur).
                    if cur_count == 0 and opp_count > opp_bwf[qi, ri]:
                        opp_bwf[qi, ri] = opp_count

    out = np.stack([cur_olc, opp_olc, cur_bwf, opp_bwf], axis=0)
    out /= float(WIN_LEN)   # normalize: WIN_LEN is both the max fill and a
                             # reasonable count normalizer for typical positions.
    return out


def compute_threat_planes_batch(states: np.ndarray) -> np.ndarray:
    """Compute 4 threat planes for a batch of (N, 4, 19, 19) states.

    States are v6_live2 encoded: plane 0 = cur stones, plane 1 = opp stones.
    Returns (N, 4, 19, 19) float32 threat plane array.
    The returned array may be concatenated with states along axis=1 to form
    (N, 8, 19, 19) v6_live2_ft encoded states.
    """
    N = states.shape[0]
    threat = np.zeros((N, 4, 19, 19), dtype=np.float32)
    log.info(f"Computing threat planes for {N} positions...")
    for i in range(N):
        cur = states[i, 0]  # current player stones
        opp = states[i, 1]  # opponent stones
        threat[i] = compute_threat_planes_single(cur, opp)
        if (i + 1) % 200 == 0:
            log.info(f"  {i+1}/{N}")
    return threat


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_DIR = REPO_ROOT / "data"

def load_bank(name: str):
    path = DATA_DIR / name
    d = np.load(str(path), allow_pickle=False)
    states = d["states"].astype("float32")   # (N, 4, 19, 19)
    target = d["target"].astype("float32")   # (N,)
    # Turn-phase from plane 2 (broadcast scalar: moves_remaining==2).
    tp = states[:, 2].reshape(len(states), -1).mean(axis=1)
    # Stone count from planes 0+1.
    stones = (states[:, 0] + states[:, 1]).reshape(len(states), -1).sum(axis=1)
    return {"states": states, "target": target, "tp": tp, "stones": stones}


def augment_states(states: np.ndarray) -> np.ndarray:
    """Add 4 threat planes to (N,4,19,19) states -> (N,8,19,19)."""
    threat = compute_threat_planes_batch(states)
    return np.concatenate([states, threat], axis=1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_fresh_model_8ch(device: torch.device) -> HexTacToeNet:
    """Build a fresh randomly-initialized 8-channel model."""
    net = HexTacToeNet(
        in_channels=8,
        res_blocks=12,
        filters=128,
        board_size=19,
        encoding="v6_live2_ft",   # new encoding — must be registered for audit
    )
    return net.to(device)


# ---------------------------------------------------------------------------
# Value eval helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def value_outputs(net: HexTacToeNet, states: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    net.eval()
    out = []
    for start in range(0, len(states), batch_size):
        x = torch.from_numpy(states[start:start + batch_size]).to(device)
        _, v, _ = net(x)
        out.append(v.squeeze(1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def kill_a(values_loss: np.ndarray) -> float:
    return float((values_loss < 0).mean()) if len(values_loss) else float("nan")


def kill_c(values_win: np.ndarray) -> float:
    return float((values_win > 0).mean()) if len(values_win) else float("nan")


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


# ---------------------------------------------------------------------------
# Turn-phase matched evaluation (mirrors E1 matched set construction)
# ---------------------------------------------------------------------------

def matched_eval(values_loss: np.ndarray, tp_loss: np.ndarray, stones_loss: np.ndarray,
                 values_win: np.ndarray, tp_win: np.ndarray, stones_win: np.ndarray,
                 label: str) -> dict:
    """Evaluate KILL-A and KILL-C on the turn-phase-matched (tp==0) set.

    tp==0 means plane 2 == 0 (moves_remaining==1, end-of-turn stone).
    Stone-count support intersection: stones in [12, 74] for both classes.
    """
    # tp==0 stratum.
    mask_loss_tp0 = (tp_loss < 0.5)
    mask_win_tp0  = (tp_win  < 0.5)

    loss_stones_tp0 = stones_loss[mask_loss_tp0]
    win_stones_tp0  = stones_win[mask_win_tp0]

    if len(loss_stones_tp0) == 0 or len(win_stones_tp0) == 0:
        log.warning(f"[{label}] No tp==0 positions in one class; skip matched eval.")
        return {"matched_kill_a": float("nan"), "matched_kill_c": float("nan"), "n_loss_matched": 0, "n_win_matched": 0}

    # Stone-count overlap support.
    lo = max(float(loss_stones_tp0.min()), float(win_stones_tp0.min()))
    hi = min(float(loss_stones_tp0.max()), float(win_stones_tp0.max()))
    # Clamp to [12, 74] as per E1 matched set definition.
    lo = max(lo, 12.0)
    hi = min(hi, 74.0)

    mask_loss_m = mask_loss_tp0 & (stones_loss >= lo) & (stones_loss <= hi)
    mask_win_m  = mask_win_tp0  & (stones_win  >= lo) & (stones_win  <= hi)

    vl = values_loss[mask_loss_m]
    vw = values_win[mask_win_m]

    ka = kill_a(vl)
    kc = kill_c(vw)
    log.info(f"[{label}] MATCHED tp0 stone-support[{lo:.0f},{hi:.0f}]: "
             f"n_loss={len(vl)} n_win={len(vw)} KILL-A={ka:.4f} KILL-C={kc:.4f}")
    kc_lo, kc_hi = wilson_ci(int((vw > 0).sum()), len(vw))
    log.info(f"[{label}] KILL-C Wilson95 [{kc_lo:.3f}, {kc_hi:.3f}]")
    return {
        "matched_kill_a": ka,
        "matched_kill_c": kc,
        "matched_kill_c_ci_lo": kc_lo,
        "matched_kill_c_ci_hi": kc_hi,
        "n_loss_matched": len(vl),
        "n_win_matched": len(vw),
    }


# ---------------------------------------------------------------------------
# Value head training
# ---------------------------------------------------------------------------

def train_value_head(net: HexTacToeNet,
                     states_loss: np.ndarray, states_win: np.ndarray,
                     device: torch.device,
                     n_epochs: int = 400,
                     batch_size: int = 64,
                     lr: float = 1e-3) -> None:
    """Train ONLY the value head on binary targets {-1, +1}.

    Trunk must already be frozen before calling. Uses class-balanced batches.
    """
    # Build optimizer over value head params only (verified frozen trunk).
    value_params = [p for name, p in net.named_parameters()
                    if "value_fc" in name or "value_var" in name]
    if not value_params:
        raise RuntimeError("No value head params found; check parameter naming.")
    opt = torch.optim.Adam(value_params, lr=lr)
    loss_fn = nn.MSELoss()

    # Class-balanced: 1:1 loss/win sampling.
    n_loss = len(states_loss)
    n_win  = len(states_win)
    n_half = min(batch_size // 2, min(n_loss, n_win))

    all_loss_t = torch.from_numpy(states_loss).to(device)
    all_win_t  = torch.from_numpy(states_win).to(device)

    net.train()
    rng = np.random.default_rng(seed=2026)

    for ep in range(n_epochs):
        li = rng.integers(0, n_loss, size=n_half)
        wi = rng.integers(0, n_win,  size=n_half)
        xb = torch.cat([all_loss_t[li], all_win_t[wi]], dim=0)
        tb = torch.cat([
            torch.full((n_half,), -1.0, device=device),
            torch.full((n_half,), +1.0, device=device),
        ])
        _, v, _ = net(xb)
        loss = loss_fn(v.squeeze(1), tb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            log.info(f"  ep {ep+1}/{n_epochs} MSE={loss.item():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="D-FULLSPEC E2 reprobe")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_phase1", action="store_true",
                        help="Skip Phase 1 (random trunk). Use for Phase 2 only.")
    args = parser.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = best_device()
    log.info(f"device={device} epochs={args.epochs} seed={args.seed}")

    # --- Load banks (read-only) ---
    log.info("Loading banks...")
    loss_tr  = load_bank("distill_d7_train.npz")
    loss_ho  = load_bank("distill_d7_holdout.npz")
    win_tr   = load_bank("killc_netwin_train.npz")
    win_ho   = load_bank("killc_netwin_holdout.npz")

    n_lt, n_lh = len(loss_tr["states"]), len(loss_ho["states"])
    n_wt, n_wh = len(win_tr["states"]),  len(win_ho["states"])
    log.info(f"Bank sizes: loss_train={n_lt} loss_holdout={n_lh} "
             f"win_train={n_wt} win_holdout={n_wh}")

    # --- Augment with threat planes ---
    log.info("=== Augmenting states with threat planes (4-plane -> 8-plane) ===")
    log.info("Computing loss_train threat planes...")
    lt_8 = augment_states(loss_tr["states"])
    log.info("Computing loss_holdout threat planes...")
    lh_8 = augment_states(loss_ho["states"])
    log.info("Computing win_train threat planes...")
    wt_8 = augment_states(win_tr["states"])
    log.info("Computing win_holdout threat planes...")
    wh_8 = augment_states(win_ho["states"])

    # Sanity check: threat plane range.
    log.info(f"Threat plane stats (win_train): "
             f"cur_olc mean={wt_8[:,4].mean():.3f} max={wt_8[:,4].max():.3f} | "
             f"opp_olc mean={wt_8[:,5].mean():.3f} max={wt_8[:,5].max():.3f} | "
             f"cur_bwf mean={wt_8[:,6].mean():.3f} max={wt_8[:,6].max():.3f} | "
             f"opp_bwf mean={wt_8[:,7].mean():.3f} max={wt_8[:,7].max():.3f}")
    log.info(f"Threat plane stats (loss_train): "
             f"cur_olc mean={lt_8[:,4].mean():.3f} max={lt_8[:,4].max():.3f} | "
             f"opp_olc mean={lt_8[:,5].mean():.3f} max={lt_8[:,5].max():.3f} | "
             f"cur_bwf mean={lt_8[:,6].mean():.3f} max={lt_8[:,6].max():.3f} | "
             f"opp_bwf mean={lt_8[:,7].mean():.3f} max={lt_8[:,7].max():.3f}")

    # Verify threat planes differ between win and loss classes (sanity).
    log.info(f"OPP_BWT holdout: loss mean={lh_8[:,7].mean():.4f} win mean={wh_8[:,7].mean():.4f}")
    log.info(f"OPP_OLC holdout: loss mean={lh_8[:,5].mean():.4f} win mean={wh_8[:,5].mean():.4f}")

    results = {}

    # ==========================================================================
    # PHASE 1: Frozen random-initialized 8-channel trunk
    # ==========================================================================
    if not args.skip_phase1:
        log.info("\n=== PHASE 1: Frozen random trunk (8 channels) ===")
        net_r = build_fresh_model_8ch(device)
        log.info(f"Fresh model built: in_channels=8 res_blocks=12 filters=128")

        # Baseline KILL-A/KILL-C BEFORE any training.
        va_base = value_outputs(net_r, lh_8, device)
        vw_base = value_outputs(net_r, wh_8, device)
        log.info(f"[PHASE1 BASELINE naive] KILL-A={kill_a(va_base):.4f} KILL-C={kill_c(vw_base):.4f}")

        freeze_trunk(net_r)
        frozen_ok = verify_freezing(net_r)
        log.info(f"frozen_ok={frozen_ok}")

        log.info("Training value head (Phase 1)...")
        train_value_head(net_r, lt_8, wt_8, device, n_epochs=args.epochs)
        log.info("Training complete.")

        va_r = value_outputs(net_r, lh_8, device)
        vw_r = value_outputs(net_r, wh_8, device)
        log.info(f"[PHASE1 NAIVE post] KILL-A={kill_a(va_r):.4f} KILL-C={kill_c(vw_r):.4f}")
        log.info(f"[PHASE1] value mean: loss={va_r.mean():.4f} win={vw_r.mean():.4f}")

        m1 = matched_eval(va_r, loss_ho["tp"], loss_ho["stones"],
                          vw_r, win_ho["tp"], win_ho["stones"],
                          "PHASE1")

        ka1 = m1["matched_kill_a"]
        kc1 = m1["matched_kill_c"]

        # Verdict.
        if ka1 > 0.35 and kc1 >= 0.85:
            verdict1 = "SEPARABLE-R"
        elif ka1 > 0.35 and kc1 >= 0.60:
            verdict1 = "PARTIAL-R"
        elif ka1 > 0.35:
            verdict1 = "ENTANGLED-R"
        else:
            verdict1 = "KILL-A-FAIL"

        log.info(f"[PHASE1 VERDICT] {verdict1} | matched KILL-A={ka1:.4f} KILL-C={kc1:.4f}")
        results["phase1"] = {
            "verdict": verdict1,
            "matched_kill_a": ka1,
            "matched_kill_c": kc1,
            "matched_kill_c_ci": [m1.get("matched_kill_c_ci_lo"), m1.get("matched_kill_c_ci_hi")],
            "n_loss_matched": m1["n_loss_matched"],
            "n_win_matched": m1["n_win_matched"],
        }

        if verdict1 == "SEPARABLE-R":
            log.info("SEPARABLE-R: random trunk + new features achieves KILL-C >= 0.85.")
            log.info("Features alone carry the separating signal. Bootstrap-restart confirmed.")
        elif verdict1 == "PARTIAL-R":
            log.info("PARTIAL-R: random trunk partial. Run Phase 2 (pretrained trunk).")
        else:
            log.info(f"{verdict1}: random trunk insufficient. Phase 2 may still recover.")

    # ==========================================================================
    # Write results
    # ==========================================================================
    elapsed = time.time() - t0
    results["elapsed_s"] = round(elapsed, 1)

    report_dir = REPO_ROOT / "reports" / "d_fullspec_2026-06-26"
    report_dir.mkdir(parents=True, exist_ok=True)

    out_json = report_dir / "E2_reprobe_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results written to {out_json}")

    # Final summary.
    log.info("\n=== E2 REPROBE SUMMARY ===")
    if "phase1" in results:
        r1 = results["phase1"]
        log.info(f"Phase 1 verdict: {r1['verdict']} | "
                 f"matched KILL-A={r1['matched_kill_a']:.4f} "
                 f"KILL-C={r1['matched_kill_c']:.4f} "
                 f"CI={r1['matched_kill_c_ci']}")
        if r1['verdict'] == "SEPARABLE-R":
            log.info("=> Bootstrap-restart with v6_live2_ft CONFIRMED by random-trunk probe.")
        elif r1['verdict'] in ("PARTIAL-R", "ENTANGLED-R"):
            log.info("=> Proceed to Phase 2 (pretrained trunk probe).")
            log.info("   Build fresh model, train 5k steps on bootstrap corpus,")
            log.info("   freeze trunk, run same discriminator protocol.")
    log.info(f"Total time: {elapsed:.1f}s")

    return results


if __name__ == "__main__":
    main()
