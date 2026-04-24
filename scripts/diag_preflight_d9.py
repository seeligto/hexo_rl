#!/usr/bin/env python3
"""Pre-flight sanity pass for D9 resume.

Loads ckpt_12190, assembles one mixed batch (augment=True), runs a forward
pass without stepping the optimizer, and reports:
  - Per-component loss (policy CE corpus, policy CE selfplay, value MSE, ownership MSE)
  - Policy entropy: corpus rows vs selfplay rows
  - Gradient norm (computed, not applied)

Usage:
    .venv/bin/python scripts/diag_preflight_d9.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hexo_rl.utils.config import load_config
from hexo_rl.training.trainer import Trainer
from hexo_rl.training.batch_assembly import load_pretrained_buffer
from hexo_rl.monitoring.events import emit_event
from hexo_rl.training.aux_decode import decode_ownership

CHECKPOINT = "checkpoints/checkpoint_00012190.pt"
REPLAY_BUFFER_BIN = "checkpoints/replay_buffer.bin"
BATCH_SIZE = 256  # matches production batch_size


def main() -> None:
    # ── Config ────────────────────────────────────────────────────────────────
    config = load_config(
        "configs/model.yaml",
        "configs/training.yaml",
        "configs/selfplay.yaml",
        "configs/game_replay.yaml",
        "configs/monitoring.yaml",
        "configs/monitors.yaml",
    )
    model_config  = config.get("model", {})
    train_config  = config.get("training", {})
    mcts_config   = config.get("mcts", {})
    self_config   = config.get("selfplay", {})
    combined_config = {**config, **model_config, **train_config, **mcts_config, **self_config}

    from hexo_rl.utils.device import best_device
    device = best_device()
    print(f"device: {device}")

    # Disable compile for this diagnostic — faster startup, no Dynamo state.
    combined_config["torch_compile"]     = False
    combined_config["torch_compile_inf"] = False

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading checkpoint: {CHECKPOINT}")
    trainer = Trainer.load_checkpoint(
        CHECKPOINT,
        checkpoint_dir="checkpoints",
        config_overrides={"torch_compile": False},
    )
    trainer.model.eval()  # eval for forward pass inspection; no BN noise
    print(f"  step={trainer.step}")

    # ── Load pretrained (corpus) buffer ───────────────────────────────────────
    mixing_cfg = train_config.get("mixing", combined_config.get("mixing", {}))
    print(f"Loading corpus from: {mixing_cfg.get('pretrained_buffer_path', '?')}")
    pretrained_buffer = load_pretrained_buffer(
        mixing_cfg, combined_config,
        emit_fn=emit_event,
        buffer_size=0, buffer_capacity=0,
    )
    assert pretrained_buffer is not None, "corpus buffer failed to load"
    print(f"  corpus loaded ok")

    # ── Load self-play buffer ─────────────────────────────────────────────────
    from engine import ReplayBuffer
    capacity  = int(combined_config.get("buffer_capacity", 500_000))
    buf_size  = int(combined_config.get("board_size", 19))
    n_planes  = int(combined_config.get("n_input_planes", 18))
    n_actions = 19 * 19 + 1  # 362
    buffer = ReplayBuffer(capacity=capacity)

    rbp = Path(REPLAY_BUFFER_BIN)
    if rbp.exists():
        print(f"Loading replay buffer: {rbp}")
        n = buffer.load_from_path(str(rbp))
        print(f"  {n} positions loaded")
    else:
        print(f"WARNING: {rbp} not found — self-play slice will be near-empty")

    if buffer.size < 10:
        print("ABORT: self-play buffer has < 10 positions — cannot assemble a meaningful batch.")
        sys.exit(1)

    # ── Assemble one batch ────────────────────────────────────────────────────
    # Mirror loop.py's compute_pretrained_weight logic at step 12190.
    decay_steps   = float(mixing_cfg.get("mixing_decay_steps", 50_000))
    initial_w     = float(mixing_cfg.get("mixing_initial_w",   0.50))
    min_w         = float(mixing_cfg.get("mixing_min_w",       0.10))
    w_pre = max(min_w, initial_w * math.exp(-trainer.step / decay_steps))
    n_pre  = max(1, int(math.ceil(BATCH_SIZE * w_pre)))
    n_self = BATCH_SIZE - n_pre
    print(f"\nBatch split: n_pre={n_pre} (w={w_pre:.3f}), n_self={n_self}")

    # augment=True — the flag under test.
    s_pre, c_pre, p_pre, o_pre, own_pre, wl_pre, ifs_pre = pretrained_buffer.sample_batch(n_pre, True)
    s_sp,  c_sp,  p_sp,  o_sp,  own_sp,  wl_sp,  ifs_sp  = buffer.sample_batch(n_self, True)

    states    = np.concatenate([s_pre,   s_sp],   axis=0)
    chain_pl  = np.concatenate([c_pre,   c_sp],   axis=0)
    policies  = np.concatenate([p_pre,   p_sp],   axis=0)
    outcomes  = np.concatenate([o_pre,   o_sp],   axis=0)
    own_raw   = np.concatenate([own_pre, own_sp],  axis=0)
    ifs       = np.concatenate([ifs_pre, ifs_sp],  axis=0)

    print(f"Batch assembled: states={states.shape}  dtype={states.dtype}")
    print(f"  augment=True confirmed — shapes would be (N*12,...) if scatter applied")

    # ── Forward pass (no optimizer step) ──────────────────────────────────────
    trainer.optimizer.zero_grad()

    fp16       = getattr(trainer, "fp16", False)
    amp_dtype  = getattr(trainer, "amp_dtype", torch.float16)
    own_weight = float(combined_config.get("ownership_weight", 0.0))
    chain_weight = float(combined_config.get("aux_chain_weight", 0.0))
    use_own    = own_weight > 0.0

    states_t   = torch.from_numpy(states).to(device)
    if not fp16:
        states_t = states_t.float()
    policies_t = torch.from_numpy(policies).to(device)
    outcomes_t = torch.from_numpy(outcomes).to(device)
    ifs_t      = torch.from_numpy(np.asarray(ifs, dtype=np.uint8)).to(device).bool()

    own_t = None
    if use_own:
        own_t = decode_ownership(own_raw, device)

    with autocast(device_type=device.type, dtype=amp_dtype, enabled=fp16):
        fwd = trainer.model(states_t, ownership=use_own, chain=chain_weight > 0.0)
        log_policy = fwd[0]  # (B, N_ACTIONS)
        v_logit    = fwd[2]  # (B,)
        own_pred   = fwd[3] if use_own else None

        # Policy CE: only valid rows (non-zero target sum).
        policy_valid = policies_t.sum(dim=1) > 1e-6

        # ── Per-slice policy CE ───────────────────────────────────────────────
        def policy_ce_slice(lp: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> float:
            valid = mask & policy_valid
            if not valid.any():
                return float("nan")
            lp_v  = lp[valid]
            tgt_v = tgt[valid]
            ce = -(tgt_v * lp_v).sum(dim=-1).mean()
            return ce.item()

        corpus_mask  = torch.zeros(BATCH_SIZE, dtype=torch.bool, device=device)
        corpus_mask[:n_pre] = True
        selfplay_mask = ~corpus_mask

        policy_ce_corpus   = policy_ce_slice(log_policy, policies_t, corpus_mask)
        policy_ce_selfplay = policy_ce_slice(log_policy, policies_t, selfplay_mask)

        # ── Value MSE ─────────────────────────────────────────────────────────
        # BCE value loss (§arch): sigmoid + BCE.
        value_bce = nn.functional.binary_cross_entropy_with_logits(
            v_logit.squeeze(-1), (outcomes_t + 1.0) / 2.0
        )

        # ── Ownership MSE ─────────────────────────────────────────────────────
        own_mse = float("nan")
        if use_own and own_pred is not None and own_t is not None and n_pre < BATCH_SIZE:
            own_pred_sp = own_pred[n_pre:].squeeze(1)
            own_t_sp    = own_t[n_pre:]
            own_mse = nn.functional.mse_loss(own_pred_sp, own_t_sp).item()

        # ── Policy entropy ────────────────────────────────────────────────────
        def entropy_slice(lp: torch.Tensor, mask: torch.Tensor) -> float:
            valid = mask & policy_valid
            if not valid.any():
                return float("nan")
            p = torch.exp(lp[valid].float())
            return torch.special.entr(p).sum(dim=-1).mean().item()

        ent_corpus   = entropy_slice(log_policy, corpus_mask)
        ent_selfplay = entropy_slice(log_policy, selfplay_mask)

        # ── Total loss (for gradient norm) ────────────────────────────────────
        policy_loss_t = -(policies_t[policy_valid] * log_policy[policy_valid]).sum(dim=-1).mean()
        total_loss = policy_loss_t + value_bce

    # Backward — compute gradients but DO NOT step.
    if fp16 and getattr(trainer, "scaler", None) is not None:
        trainer.scaler.scale(total_loss).backward()
        trainer.scaler.unscale_(trainer.optimizer)
    else:
        total_loss.backward()

    grad_norm = 0.0
    for p in trainer.model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = math.sqrt(grad_norm)

    # Zero out — we never step.
    trainer.optimizer.zero_grad()

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PRE-FLIGHT SANITY PASS — ckpt_12190")
    print("=" * 60)
    print(f"  policy CE  (corpus,   n={n_pre}):   {policy_ce_corpus:.4f}")
    print(f"  policy CE  (selfplay, n={n_self}):  {policy_ce_selfplay:.4f}")
    print(f"  value BCE  (all):                   {value_bce.item():.4f}")
    print(f"  ownership MSE (selfplay only):       {own_mse:.4f}" if not math.isnan(own_mse) else f"  ownership MSE: n/a (use_own={use_own})")
    print(f"  policy entropy (corpus):             {ent_corpus:.4f}")
    print(f"  policy entropy (selfplay):           {ent_selfplay:.4f}")
    print(f"  gradient norm:                       {grad_norm:.4f}")
    print("=" * 60)
    print("\nDIAGNOSTICS:")
    if policy_ce_selfplay > policy_ce_corpus:
        print("  OK  policy CE selfplay > corpus (expected — selfplay is harder to fit)")
    else:
        print("  WARN  policy CE selfplay <= corpus — unexpected; check scatter LUT / augmentation")
    if ent_selfplay > 0.1:
        print(f"  OK  selfplay entropy {ent_selfplay:.4f} > 0.1 (not collapsed)")
    else:
        print(f"  WARN  selfplay entropy {ent_selfplay:.4f} near zero — check scatter LUT corruption")
    if 0.3 < grad_norm < 5.0:
        print(f"  OK  grad norm {grad_norm:.4f} in expected range (~1.1)")
    else:
        print(f"  WARN  grad norm {grad_norm:.4f} outside [0.3, 5.0] — investigate before running")
    print()


if __name__ == "__main__":
    main()
