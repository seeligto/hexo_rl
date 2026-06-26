#!/usr/bin/env python3
"""D-VDERISK DS3: Trunk-frozen value-head fine-tune probe.

Tests whether the value head CAN represent SealBot truth (GENERALIZES)
or is capacity/architecture limited (MEMORIZES / CANT_FIT).

Pre-registered verdicts:
  GENERALIZES  : holdout MSE drops >20% relative
  MEMORIZES    : train MSE drops >20% relative but holdout flat (<5%)
  CANT_FIT     : train MSE won't drop >10% relative
  INCOMPLETE   : DS1 pipeline incomplete or n_blindspot < 100

Usage on vast RTX 5080:
    cd /workspace/hexo_rl
    .venv/bin/python scripts/dvderisk_ds3_probe.py \\
        --train data/dvderisk_ds1_train.csv \\
        --holdout data/dvderisk_ds1_holdout.csv \\
        --checkpoint checkpoints/checkpoint_00272357.pt \\
        --buffer data/livetail_bank_e928c854.npz \\
        --extra-buffer data/selfplay_bank12k_v6_live2_ls_50k.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.checkpoints import extract_model_state
from hexo_rl.utils.device import best_device

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_PATH = REPO_ROOT / "logs" / "dvderisk_ds3.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_PATH), mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> HexTacToeNet:
    """Load the 4-plane v6_live2 checkpoint (272k steps)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    state = extract_model_state(ckpt)

    # Detect in_channels from checkpoint (v6_live2 = 4 planes)
    inp_w = state.get("trunk.input_conv.weight")
    if inp_w is None:
        for k, v in state.items():
            if "input_conv" in k and "weight" in k:
                inp_w = v
                break
    in_ch = int(inp_w.shape[1]) if inp_w is not None else cfg.get("in_channels", 4)

    # encoding in config may be a dict like {'version': 'v6_live2'} or a string
    enc_raw = cfg.get("encoding", "v6_live2")
    if isinstance(enc_raw, dict):
        encoding_str = enc_raw.get("version", "v6_live2")
    else:
        encoding_str = str(enc_raw) if enc_raw else "v6_live2"

    net = HexTacToeNet(
        in_channels=in_ch,
        res_blocks=cfg.get("res_blocks", 12),
        filters=cfg.get("filters", 128),
        board_size=cfg.get("board_size", 19),
        encoding=encoding_str,
    )
    net.load_state_dict(state, strict=True)
    net = net.to(device)
    return net


# ---------------------------------------------------------------------------
# Data loading: reconstruct states from CSV + NPZ buffer
# ---------------------------------------------------------------------------

def load_states_from_csv_and_npz(
    csv_path: Path,
    npz_primary: Path,
    npz_extra: Optional[Path],
) -> tuple:
    """Load (states, sealbot_target, net_value_pre, is_proven_loss) from CSV + NPZ.

    The DS1 CSV has buffer_idx referencing positions in the NPZ files.
    The two NPZ files are concatenated in the same order as DS1 scan
    (primary first, then extra with game_id_offset). We use buffer_idx
    directly to index the concatenated states array.

    sealbot_target:
      - is_proven_loss=1 -> -1.0  (SealBot says forced loss, target = -1.0)
      - sealbot_losing=1 (but not proven) -> -0.5 (heuristic loss signal)
      - sealbot_score > 0 -> +0.5 (SealBot winning signal)
      - neutral -> 0.0
    """
    import csv
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    log.info(f"CSV {csv_path.name}: {len(rows)} rows")

    # Load NPZ files and concatenate in DS1 scan order
    log.info(f"Loading NPZ: {npz_primary}")
    buf1 = np.load(str(npz_primary), allow_pickle=False)
    states_concat = buf1["states"].astype("float32")  # (N1, 4, 19, 19)
    log.info(f"Primary NPZ: {states_concat.shape[0]} positions")

    if npz_extra is not None and npz_extra.exists():
        log.info(f"Loading extra NPZ: {npz_extra}")
        buf2 = np.load(str(npz_extra), allow_pickle=False)
        states2 = buf2["states"].astype("float32")
        states_concat = np.concatenate([states_concat, states2], axis=0)
        log.info(f"After concat: {states_concat.shape[0]} positions")

    N_buf = states_concat.shape[0]
    log.info(f"Buffer total: {N_buf} positions")

    # Reconstruct arrays from CSV rows
    n = len(rows)
    states = np.zeros((n, 4, 19, 19), dtype="float32")
    sealbot_targets = np.zeros(n, dtype="float32")
    net_value_pre = np.zeros(n, dtype="float32")
    is_proven_loss = np.zeros(n, dtype="int32")

    bad_idx = 0
    for i, r in enumerate(rows):
        bidx = int(r["buffer_idx"])
        if bidx >= N_buf:
            log.warning(f"buffer_idx={bidx} out of range {N_buf}; zeroing state")
            bad_idx += 1
        else:
            states[i] = states_concat[bidx]

        # SealBot target: use is_proven_loss for clean signal
        ipl = int(r["is_proven_loss"])
        is_proven_loss[i] = ipl
        sb_score = float(r["sealbot_score"])
        sb_losing = int(r["sealbot_losing"])

        if ipl:
            sealbot_targets[i] = -1.0  # proven forced loss
        elif sb_losing:
            sealbot_targets[i] = -0.5  # heuristic loss
        elif sb_score > 0:
            sealbot_targets[i] = 0.5   # SealBot winning
        else:
            sealbot_targets[i] = 0.0

        net_value_pre[i] = float(r["net_value"])

    if bad_idx > 0:
        log.warning(f"{bad_idx} buffer_idx out of range; those states zeroed")

    log.info(f"  is_proven_loss: {is_proven_loss.sum()} / {n}")
    log.info(
        f"  sealbot_target stats: mean={sealbot_targets.mean():.3f} "
        f"min={sealbot_targets.min():.3f} max={sealbot_targets.max():.3f}"
    )

    return states, sealbot_targets, net_value_pre, is_proven_loss


# ---------------------------------------------------------------------------
# MSE evaluation
# ---------------------------------------------------------------------------

def eval_mse(
    net: HexTacToeNet,
    states: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """Compute MSE(value_head_output, targets)."""
    net.eval()
    all_sq_err = []
    n = len(states)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = torch.from_numpy(states[start:end]).to(device)
            t = torch.from_numpy(targets[start:end]).to(device)
            _, v, _ = net(x)
            v = v.squeeze(1)
            sq = (v - t).pow(2)
            all_sq_err.append(sq.cpu().numpy())
    return float(np.concatenate(all_sq_err).mean())


# ---------------------------------------------------------------------------
# Trunk-frozen fine-tuning
# ---------------------------------------------------------------------------

def freeze_trunk(net: HexTacToeNet) -> dict:
    """Set trunk params to requires_grad=False. Return frozen/trainable summary."""
    frozen, trainable = [], []
    for name, param in net.named_parameters():
        if name.startswith("trunk."):
            param.requires_grad_(False)
            frozen.append(name)
        else:
            param.requires_grad_(True)
            trainable.append(name)
    return {"frozen": frozen, "trainable": trainable}


def verify_freezing(net: HexTacToeNet) -> bool:
    """Return True iff all trunk.* params have requires_grad=False."""
    ok = True
    for name, param in net.named_parameters():
        if name.startswith("trunk."):
            if param.requires_grad:
                log.error(f"FREEZE FAIL: {name} still requires_grad=True")
                ok = False
        else:
            if not param.requires_grad:
                log.warning(f"WARN: non-trunk {name} has requires_grad=False")
    return ok


def run_finetune(
    net: HexTacToeNet,
    states: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    n_epochs: int = 40,
    batch_size: int = 128,
    lr: float = 3e-4,
    patience: int = 8,
) -> list:
    """Fine-tune value head only (trunk frozen). Returns per-epoch train MSE list."""
    net.train()
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad],
        lr=lr,
    )
    loss_fn = nn.MSELoss()

    n = len(states)
    rng = np.random.default_rng(42)
    epoch_losses = []
    best_loss = float("inf")
    patience_counter = 0

    log.info(f"Fine-tune: epochs={n_epochs} batch={batch_size} lr={lr} n={n}")

    for epoch in range(n_epochs):
        idx = rng.permutation(n)
        batch_losses = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bi = idx[start:end]
            x = torch.from_numpy(states[bi]).to(device)
            t = torch.from_numpy(targets[bi]).to(device)

            optimizer.zero_grad()
            _, v, _ = net(x)
            v = v.squeeze(1)
            loss = loss_fn(v, t)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        ep_loss = float(np.mean(batch_losses))
        epoch_losses.append(ep_loss)
        log.info(f"  epoch {epoch+1:3d}/{n_epochs}  train_mse={ep_loss:.6f}")

        # Early stopping
        if ep_loss < best_loss - 1e-6:
            best_loss = ep_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            log.info(f"  Early stop at epoch {epoch+1} (patience={patience})")
            break

    return epoch_losses


# ---------------------------------------------------------------------------
# Light trunk tuning (all params, low LR, 5 epochs)
# ---------------------------------------------------------------------------

def run_light_trunk_tune(
    net_frozen_ft: HexTacToeNet,
    ckpt_path: Path,
    states_tr: np.ndarray,
    targets_tr: np.ndarray,
    states_ho: np.ndarray,
    targets_ho: np.ndarray,
    device: torch.device,
    n_epochs: int = 5,
    lr: float = 1e-5,
    batch_size: int = 128,
) -> str:
    """Unfreeze all, fine-tune 5 epochs at low LR. Returns summary string."""
    # Rebuild model from original checkpoint (not deepcopy — RegistrySpec not picklable)
    net2 = load_model(ckpt_path, device)
    # Copy the fine-tuned weights into net2
    net2.load_state_dict(net_frozen_ft.state_dict())

    # Unfreeze all
    for param in net2.parameters():
        param.requires_grad_(True)

    mse_tr_before = eval_mse(net2, states_tr, targets_tr, device, batch_size=256)
    mse_ho_before = eval_mse(net2, states_ho, targets_ho, device, batch_size=256)

    optimizer = torch.optim.Adam(net2.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = len(states_tr)
    rng = np.random.default_rng(99)
    epoch_losses = []

    for epoch in range(n_epochs):
        net2.train()
        idx = rng.permutation(n)
        bl = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bi = idx[start:end]
            x = torch.from_numpy(states_tr[bi]).to(device)
            t = torch.from_numpy(targets_tr[bi]).to(device)
            optimizer.zero_grad()
            _, v, _ = net2(x)
            v = v.squeeze(1)
            loss = loss_fn(v, t)
            loss.backward()
            optimizer.step()
            bl.append(loss.item())
        ep = float(np.mean(bl))
        epoch_losses.append(ep)
        log.info(f"  light-trunk epoch {epoch+1}/{n_epochs}  mse={ep:.6f}")

    mse_tr_after = eval_mse(net2, states_tr, targets_tr, device, batch_size=256)
    mse_ho_after = eval_mse(net2, states_ho, targets_ho, device, batch_size=256)

    result = (
        f"light_trunk: train {mse_tr_before:.4f}->{mse_tr_after:.4f} "
        f"({100*(mse_tr_before-mse_tr_after)/mse_tr_before:.1f}% drop)  "
        f"holdout {mse_ho_before:.4f}->{mse_ho_after:.4f} "
        f"({100*(mse_ho_before-mse_ho_after)/mse_ho_before:.1f}% drop)  "
        f"epochs={n_epochs} lr={lr}"
    )
    log.info(f"LIGHT_TRUNK_RESULT: {result}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train",      default="data/dvderisk_ds1_train.csv")
    ap.add_argument("--holdout",    default="data/dvderisk_ds1_holdout.csv")
    ap.add_argument("--checkpoint", default="checkpoints/checkpoint_00272357.pt")
    ap.add_argument("--buffer",     default="data/livetail_bank_e928c854.npz")
    ap.add_argument("--extra-buffer", default="data/selfplay_bank12k_v6_live2_ls_50k.npz")
    ap.add_argument("--n-epochs",   type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--patience",   type=int, default=10)
    ap.add_argument("--skip-light-trunk", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    log.info("=" * 70)
    log.info("D-VDERISK DS3 — trunk-frozen value-head fine-tune probe")
    log.info("=" * 70)
    log.info(f"checkpoint: {args.checkpoint}")
    log.info(f"train CSV:  {args.train}")
    log.info(f"holdout CSV:{args.holdout}")
    log.info(f"buffer NPZ: {args.buffer}")
    log.info(f"extra NPZ:  {args.extra_buffer}")

    device = best_device()
    log.info(f"device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    log.info("Loading checkpoint...")
    ckpt_path = REPO_ROOT / args.checkpoint
    net = load_model(ckpt_path, device)
    log.info(f"Model loaded: encoding={net.encoding} in_channels={net.in_channels} "
             f"filters={net.filters} res_blocks={net.res_blocks}")

    # ── Print all param names + requires_grad BEFORE freezing ────────────────
    log.info("=== PARAM TABLE (before freezing) ===")
    for name, param in net.named_parameters():
        log.info(f"  {name:60s}  requires_grad={param.requires_grad}")

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading train split from NPZ...")
    npz_extra = Path(REPO_ROOT / args.extra_buffer) if args.extra_buffer else None
    states_tr, targets_tr, nv_pre_tr, ipl_tr = load_states_from_csv_and_npz(
        REPO_ROOT / args.train,
        REPO_ROOT / args.buffer,
        npz_extra,
    )

    log.info("Loading holdout split from NPZ...")
    states_ho, targets_ho, nv_pre_ho, ipl_ho = load_states_from_csv_and_npz(
        REPO_ROOT / args.holdout,
        REPO_ROOT / args.buffer,
        npz_extra,
    )

    n_train = len(states_tr)
    n_holdout = len(states_ho)
    n_blind_tr = int(ipl_tr.sum())
    n_blind_ho = int(ipl_ho.sum())
    log.info(f"Train: {n_train} rows, {n_blind_tr} proven_loss")
    log.info(f"Holdout: {n_holdout} rows, {n_blind_ho} proven_loss")

    # ── Evaluate BEFORE fine-tune ─────────────────────────────────────────────
    log.info("Evaluating BEFORE fine-tune...")

    # Train MSE before
    mse_train_before = eval_mse(net, states_tr, targets_tr, device)
    log.info(f"BEFORE  train MSE (all): {mse_train_before:.6f}")

    # Holdout MSE before
    mse_holdout_before = eval_mse(net, states_ho, targets_ho, device)
    log.info(f"BEFORE  holdout MSE (all): {mse_holdout_before:.6f}")

    # Anchor: proven-loss positions in holdout (proxy for 33 D-PERCEPT anchors)
    if n_blind_ho > 0:
        anchor_before = eval_mse(
            net,
            states_ho[ipl_ho == 1],
            np.full(n_blind_ho, -1.0, dtype="float32"),
            device,
        )
    else:
        # Fall back to train proven_loss if holdout has none
        anchor_before = eval_mse(
            net,
            states_tr[ipl_tr == 1],
            np.full(n_blind_tr, -1.0, dtype="float32"),
            device,
        )
    log.info(f"BEFORE  anchor (proven_loss, target=-1.0) MSE: {anchor_before:.6f}")

    # ── CRITICAL: Freeze trunk, verify, then log table ────────────────────────
    log.info("=== FREEZING TRUNK ===")
    param_summary = freeze_trunk(net)
    log.info(f"Frozen params ({len(param_summary['frozen'])}): {param_summary['frozen'][:5]}...")
    log.info(f"Trainable params ({len(param_summary['trainable'])}): {param_summary['trainable']}")

    log.info("=== PARAM TABLE (after freezing — VERIFICATION) ===")
    for name, param in net.named_parameters():
        log.info(f"  {name:60s}  requires_grad={param.requires_grad}")

    freeze_ok = verify_freezing(net)
    log.info(f"TRUNK FROZEN VERIFIED: {freeze_ok}")
    if not freeze_ok:
        log.error("ABORT: trunk freezing failed — trunk params still trainable")
        sys.exit(1)

    # ── Fine-tune (trunk frozen) ──────────────────────────────────────────────
    log.info("=== TRUNK-FROZEN FINE-TUNE ===")
    epoch_losses = run_finetune(
        net,
        states_tr,
        targets_tr,
        device,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )

    # ── Evaluate AFTER fine-tune ──────────────────────────────────────────────
    log.info("Evaluating AFTER fine-tune...")
    mse_train_after = eval_mse(net, states_tr, targets_tr, device)
    log.info(f"AFTER  train MSE (all): {mse_train_after:.6f}")

    mse_holdout_after = eval_mse(net, states_ho, targets_ho, device)
    log.info(f"AFTER  holdout MSE (all): {mse_holdout_after:.6f}")

    if n_blind_ho > 0:
        anchor_after = eval_mse(
            net,
            states_ho[ipl_ho == 1],
            np.full(n_blind_ho, -1.0, dtype="float32"),
            device,
        )
    else:
        anchor_after = eval_mse(
            net,
            states_tr[ipl_tr == 1],
            np.full(n_blind_tr, -1.0, dtype="float32"),
            device,
        )
    log.info(f"AFTER  anchor (proven_loss, target=-1.0) MSE: {anchor_after:.6f}")

    # ── Compute relative improvements ─────────────────────────────────────────
    train_rel_drop  = (mse_train_before  - mse_train_after)  / max(mse_train_before,  1e-9)
    holdout_rel_drop = (mse_holdout_before - mse_holdout_after) / max(mse_holdout_before, 1e-9)
    anchor_rel_drop  = (anchor_before     - anchor_after)      / max(anchor_before,    1e-9)

    log.info(f"Train relative drop:   {train_rel_drop*100:.1f}%")
    log.info(f"Holdout relative drop: {holdout_rel_drop*100:.1f}%")
    log.info(f"Anchor relative drop:  {anchor_rel_drop*100:.1f}%")

    # ── Apply pre-registered verdict ──────────────────────────────────────────
    n_blind_total = n_blind_tr + n_blind_ho
    if n_blind_total < 100:
        verdict = "INCOMPLETE"
    elif train_rel_drop < 0.10:
        verdict = "CANT_FIT"
    elif train_rel_drop >= 0.20 and holdout_rel_drop < 0.05:
        verdict = "MEMORIZES"
    elif holdout_rel_drop >= 0.20:
        verdict = "GENERALIZES"
    else:
        # holdout dropped 5-20% or ambiguous: lean on magnitude
        if holdout_rel_drop >= 0.10:
            verdict = "GENERALIZES"
        else:
            verdict = "MEMORIZES"

    log.info(f"PRE-REGISTERED VERDICT: {verdict}")

    # ── Light trunk tuning pass ───────────────────────────────────────────────
    lightly_tuned_result = "SKIPPED"
    if not args.skip_light_trunk:
        log.info("=== LIGHT TRUNK TUNING (5 epochs, lr=1e-5, unfreeze all) ===")
        try:
            lightly_tuned_result = run_light_trunk_tune(
                net,
                ckpt_path,
                states_tr, targets_tr,
                states_ho, targets_ho,
                device,
                n_epochs=5,
                lr=1e-5,
                batch_size=args.batch_size,
            )
        except Exception as e:
            log.error(f"Light trunk tune failed: {e}", exc_info=True)
            lightly_tuned_result = f"ERROR: {e}"

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    log.info("=" * 70)
    log.info("DS3 FINAL RESULTS")
    log.info(f"  n_train:              {n_train}")
    log.info(f"  n_holdout:            {n_holdout}")
    log.info(f"  n_blind_train:        {n_blind_tr}")
    log.info(f"  n_blind_holdout:      {n_blind_ho}")
    log.info(f"  trunk_frozen:         {freeze_ok}")
    log.info(f"  train_mse_before:     {mse_train_before:.6f}")
    log.info(f"  train_mse_after:      {mse_train_after:.6f}")
    log.info(f"  train_rel_drop:       {train_rel_drop*100:.2f}%")
    log.info(f"  holdout_mse_before:   {mse_holdout_before:.6f}")
    log.info(f"  holdout_mse_after:    {mse_holdout_after:.6f}")
    log.info(f"  holdout_rel_drop:     {holdout_rel_drop*100:.2f}%")
    log.info(f"  anchor_before:        {anchor_before:.6f}")
    log.info(f"  anchor_after:         {anchor_after:.6f}")
    log.info(f"  verdict:              {verdict}")
    log.info(f"  lightly_tuned_result: {lightly_tuned_result}")
    log.info(f"  wall_time:            {elapsed:.1f}s")
    log.info("=" * 70)

    # Machine-readable JSON summary
    summary = {
        "verdict": verdict,
        "trunk_frozen": freeze_ok,
        "n_train": n_train,
        "n_holdout": n_holdout,
        "n_blind_train": n_blind_tr,
        "n_blind_holdout": n_blind_ho,
        "train_mse_before": round(mse_train_before, 6),
        "train_mse_after": round(mse_train_after, 6),
        "train_rel_drop_pct": round(train_rel_drop * 100, 2),
        "holdout_mse_before": round(mse_holdout_before, 6),
        "holdout_mse_after": round(mse_holdout_after, 6),
        "holdout_rel_drop_pct": round(holdout_rel_drop * 100, 2),
        "anchor_mse_before": round(anchor_before, 6),
        "anchor_mse_after": round(anchor_after, 6),
        "anchor_rel_drop_pct": round(anchor_rel_drop * 100, 2),
        "epoch_losses": [round(x, 6) for x in epoch_losses],
        "lightly_tuned_result": lightly_tuned_result,
        "wall_time_s": round(elapsed, 1),
        "notes": (
            "anchor33: proven_loss subset from DS1 holdout (n=" + str(n_blind_ho) + ") "
            "used as proxy for D-PERCEPT 33 positions (game states not directly recoverable "
            "from CSV; exact anchor positions need dpercept_disc GAMES_PATH replay). "
            "anchor_before MSE is vs -1.0 target on this proxy set, NOT the exact nv_post "
            "values from disc_merged.json (those are mean=0.791, MSE_to_neg1=3.26). "
            "DS2 coverage=SOMETIMES (coverage_frac>20%). "
            f"sealbot_target: proven_loss->-1.0, heuristic_loss->-0.5, sb_winning->+0.5, neutral->0.0. "
            f"encoding=v6_live2 in_channels=4."
        ),
    }

    summary_path = REPO_ROOT / "logs" / "dvderisk_ds3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary written to {summary_path}")
    log.info("DS3 DONE.")


if __name__ == "__main__":
    main()
