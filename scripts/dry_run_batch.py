"""Dry-run preflight for §118 recovery: load ckpt_12190, assemble one batch
(augment=True, recent_buffer=None for isolated check), run one training step,
assert gradient norm ≤2.0 and no NaN in any loss component.

Usage:
  python scripts/dry_run_batch.py

Exit 0 = PASS, exit 1 = FAIL.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from hexo_rl.utils.config import load_config
from hexo_rl.utils.device import best_device


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
from hexo_rl.training.trainer import Trainer
from hexo_rl.training.batch_assembly import (
    allocate_batch_buffers,
    load_pretrained_buffer,
    assemble_mixed_batch,
)

CHECKPOINT = "checkpoints/checkpoint_00012190.pt"
VARIANT    = "configs/variants/phase118_recovery.yaml"
GRAD_NORM_LIMIT = 2.0

_BASE_CONFIGS = [
    "configs/model.yaml",
    "configs/training.yaml",
    "configs/selfplay.yaml",
    "configs/game_replay.yaml",
    "configs/monitoring.yaml",
    "configs/monitors.yaml",
]

LOSS_KEYS = ("loss", "policy_loss", "value_loss", "opp_reply_loss",
             "ownership_loss", "threat_loss", "chain_loss")


def main() -> None:
    config = load_config(*_BASE_CONFIGS, VARIANT)

    seed_everything(int(config.get("seed", 42)))
    device = best_device()
    print(f"device: {device}")

    from hexo_rl.model.tf32 import resolve_and_apply as _tf32
    _tf32(config)

    train_cfg   = config.get("training", config)
    mixing_cfg  = train_cfg.get("mixing", config.get("mixing", {}))
    batch_size  = int(train_cfg.get("batch_size", config.get("batch_size", 256)))
    augment     = bool(train_cfg.get("augment", config.get("augment", True)))

    combined_config = {**config, **config.get("model", {}), **train_cfg,
                       **config.get("mcts", {}), **config.get("selfplay", {})}

    print(f"augment={augment}  (must be True)")
    assert augment, "FAIL: augment=False in merged config — production run must use augment=True"

    # Load checkpoint.
    config_overrides: dict = {"torch_compile": False}
    for k in ("uncertainty_weight", "recency_weight", "ownership_weight",
              "threat_weight", "eta_min"):
        if combined_config.get(k) is not None:
            config_overrides[k] = combined_config[k]

    trainer = Trainer.load_checkpoint(
        CHECKPOINT,
        checkpoint_dir="checkpoints",
        device=device,
        fallback_config=combined_config,
        config_overrides=config_overrides,
    )
    print(f"checkpoint loaded: step={trainer.step}")
    assert trainer.step == 12190, f"FAIL: expected step 12190, got {trainer.step}"

    # Load corpus buffer.
    def _noop_emit(ev: dict) -> None:
        pass

    from engine import ReplayBuffer
    pretrained_buffer = load_pretrained_buffer(
        mixing_cfg, combined_config, _noop_emit,
        buffer_size=0, buffer_capacity=0,
    )
    assert pretrained_buffer is not None and pretrained_buffer.size > 0, \
        "FAIL: corpus buffer empty — run 'make corpus.npz' first"
    print(f"corpus loaded: {pretrained_buffer.size} positions")

    import math as _math
    w_pre  = max(0.1, 0.8 * _math.exp(-12190 / 20000))
    n_pre  = max(1, int(_math.ceil(batch_size * w_pre)))
    n_self = batch_size - n_pre

    # Load real self-play buffer — synthetic all-zero states cause fp16 overflow.
    _sp_path = mixing_cfg.get("buffer_persist_path", "checkpoints/replay_buffer.bin")
    sp_buffer = ReplayBuffer(capacity=250_000)
    if Path(_sp_path).exists():
        sp_buffer.load_from_path(str(_sp_path))
        print(f"self-play buffer loaded: {sp_buffer.size} positions from {_sp_path}")
    else:
        # Fallback: sample real positions from corpus buffer (avoids fp16 overflow).
        _s, _c, _p, _o, _own2, _wl2, _ = pretrained_buffer.sample_batch(max(n_self * 2, 256), False)
        sp_buffer2 = ReplayBuffer(capacity=max(n_self * 2, 256))
        _own2f = _own2.reshape(-1, 361).astype(np.uint8)
        _wl2f  = _wl2.reshape(-1, 361).astype(np.uint8)
        sp_buffer2.push_game(_s, _c, _p, _o, _own2f, _wl2f)
        sp_buffer = sp_buffer2
        print(f"self-play buffer (corpus fallback): {sp_buffer.size} rows")

    bufs = allocate_batch_buffers(batch_size, 362)
    batch_size_cfg = batch_size

    (states, chain_planes, policies, outcomes,
     ownership, winning_line, is_full_search, n_recent_batch) = assemble_mixed_batch(
        pretrained_buffer, sp_buffer, None,
        n_pre, n_self, batch_size, batch_size_cfg,
        recency_weight=0.0, bufs=bufs, train_step=12190,
        augment=augment,
    )
    print(f"batch assembled: shape={states.shape}, n_pre={n_pre}, n_recent={n_recent_batch}")

    loss_info = trainer.train_step_from_tensors(
        states, policies, outcomes,
        chain_planes=chain_planes,
        ownership_targets=ownership,
        threat_targets=winning_line,
        is_full_search=is_full_search,
        n_pretrain=n_pre,
        n_recent=n_recent_batch,
    )

    print("\n── Loss components ─────────────────────────────────────────────")
    failed = False
    for k in LOSS_KEYS:
        v = loss_info.get(k)
        if v is None:
            continue
        nan_flag = " *** NaN ***" if (v is None or not math.isfinite(v)) else ""
        print(f"  {k:30s} = {v:.6f}{nan_flag}")
        if nan_flag:
            failed = True

    grad_norm = float(loss_info.get("grad_norm", float("nan")))
    gn_flag = " *** EXCEEDS LIMIT ***" if grad_norm > GRAD_NORM_LIMIT else ""
    print(f"\n  {'grad_norm':30s} = {grad_norm:.6f}{gn_flag}")
    if gn_flag or not math.isfinite(grad_norm):
        failed = True

    print(f"\n  {'policy_entropy_pretrain':30s} = {loss_info.get('policy_entropy_pretrain', float('nan')):.4f}")
    print(f"  {'selfplay_model_entropy_batch':30s} = {loss_info.get('selfplay_model_entropy_batch', loss_info.get('policy_entropy_selfplay', float('nan'))):.4f}")
    print(f"  {'policy_entropy_recent':30s} = {loss_info.get('policy_entropy_recent', float('nan')):.4f}")
    print(f"  {'policy_entropy_uniform_selfplay':30s} = {loss_info.get('policy_entropy_uniform_selfplay', float('nan')):.4f}")

    if failed:
        print("\nRESULT: FAIL")
        sys.exit(1)
    else:
        print("\nRESULT: PASS — grad_norm ≤2.0, no NaN, augment=True confirmed")
        sys.exit(0)


if __name__ == "__main__":
    main()
