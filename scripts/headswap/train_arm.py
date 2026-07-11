"""D-F HEADSWAP — single-arm frozen-trunk value-head training.

Binds to scripts/headswap/RECIPE.md §"Arms", §"Model surface", §"KEY TRAINING
FACT". Trains ONE value head (scalar A/D or 65-bin B/C) on the run2 replay
buffer, target = game OUTCOME z ONLY (INV-D1). The 65-bin arm trains as PLAIN
per-row two-hot CE (buffer is K=1/row; argmin-cluster is inference-only, §Scoring).

CROSS-ARM SAMPLE STREAM (RECIPE: "SAME replay sample stream, identical draw
seeds"):
    The Rust ReplayBuffer RNG is ENTROPY-seeded (engine make_rng), unaffected by
    torch/numpy seeds, and has NO seed arg / NO index-return path (verified:
    sample_batch_with_pos_impl draws both index and augmentation sym from the
    internal rng). Two independent buffer instances therefore produce DIFFERENT
    streams. To guarantee a byte-identical stream across all arms we PRE-DRAW the
    full sequence of `steps` batches ONCE into an on-disk numpy MEMMAP (states f16,
    outcomes f32, value_target_valid u8) keyed by (buffer sha, steps, batch). Every
    arm + every LR-grid cell replays that same disk-backed sequence. Head init is
    the only per-`--seed` randomness.

MEMORY (bounded, no explosion — the cache is on DISK, read one batch at a time):
    Build peak ~= buffer (2.35 GB, FREED before training) + one batch. Train peak
    ~= model (~17 MB) + one batch (~0.7 MB) + OS page cache (reclaimable). RAM is
    INDEPENDENT of `steps` — a 10k or 100k run uses the same resident memory; only
    the on-disk memmap grows. Nothing accumulates across the training loop (per-step
    tensors are locals freed each iter; loss_curve stores floats only).

Usage::
    .venv/bin/python -m scripts.headswap.train_arm \\
        --arm B --trunk <ckpt.pt> --buffer <replay.bin> \\
        --steps 20000 --seed 0 --lr 2e-3 --out reports/headswap/arm_B_seed0 \\
        [--batch-cache reports/headswap/batch_cache_s20000.pt] [--batch 256]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from engine import ReplayBuffer
from hexo_rl.training.losses import compute_value_loss
from scripts.headswap.model_heads import (
    build_head,
    freeze_for_arm,
    load_trunk,
    value_feature,
)
from scripts.headswap.targets import two_hot_ce_loss

WEIGHT_DECAY = 1e-4       # production AdamW default (trainer.py:5)
BUFFER_CAPACITY = 3_000_000  # >= 250k live positions; over-alloc is harmless
ENCODING = "v6_live2_ls"


def _buffer_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _cache_files(cache_dir: Path):
    return (
        cache_dir / "states.f16.dat",
        cache_dir / "outcomes.f32.dat",
        cache_dir / "valid.u8.dat",
        cache_dir / "manifest.json",
    )


def _open_ro(s_path, o_path, v_path, shape_s, shape_2) -> dict:
    """Open the cache memmaps READ-ONLY (mode='r'): disk-backed, lazily paged,
    zero RAM load. The training loop copies out one batch at a time."""
    return {
        "states": np.memmap(s_path, dtype=np.float16, mode="r", shape=shape_s),
        "outcomes": np.memmap(o_path, dtype=np.float32, mode="r", shape=shape_2),
        "value_valid": np.memmap(v_path, dtype=np.uint8, mode="r", shape=shape_2),
    }


def build_or_load_batch_cache(
    buffer_path: str,
    steps: int,
    batch: int,
    cache_dir: Path,
) -> dict:
    """Return read-only memmaps {states f16, outcomes f32, value_valid u8}, each
    shape (steps, batch, ...), identical across arms — the SHARED sample stream.

    Backed by an on-DISK numpy memmap so RAM never scales with `steps` (see module
    MEMORY note). If the manifest matches (buffer sha, steps, batch) the cache is
    reused; else it is built ONCE: load buffer -> fill memmaps incrementally ->
    FREE the buffer (del + gc) before returning. The manifest pins the buffer
    content sha so a stale cache can never silently feed a different stream.
    """
    s_path, o_path, v_path, m_path = _cache_files(cache_dir)
    buf_sha = _buffer_sha(buffer_path)
    shape_s = (steps, batch, 4, 19, 19)
    shape_2 = (steps, batch)

    if m_path.exists():
        man = json.loads(m_path.read_text())
        if man["buffer_sha"] == buf_sha and man["steps"] == steps and man["batch"] == batch:
            print(f"[cache] reuse {cache_dir} (sha {buf_sha}, steps {steps})")
            return _open_ro(s_path, o_path, v_path, shape_s, shape_2)
        raise RuntimeError(
            f"batch cache {cache_dir} manifest mismatch "
            f"(cache {man} vs want sha={buf_sha} steps={steps} batch={batch}); "
            f"delete it or point --batch-cache elsewhere"
        )

    print(f"[cache] building {steps} batches (batch={batch}) -> memmap {cache_dir} ...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    states_mm = np.memmap(s_path, dtype=np.float16, mode="w+", shape=shape_s)
    out_mm = np.memmap(o_path, dtype=np.float32, mode="w+", shape=shape_2)
    valid_mm = np.memmap(v_path, dtype=np.uint8, mode="w+", shape=shape_2)

    buf = ReplayBuffer(BUFFER_CAPACITY, ENCODING)
    n = buf.load_from_path(buffer_path)
    if n <= 0:
        raise RuntimeError(f"buffer load returned {n} positions from {buffer_path}")
    for s in range(steps):
        (states, _chain, _pol, outcomes, _own, _wl, _ifs, _pos,
         value_valid) = buf.sample_batch_with_pos(batch, True)
        states_mm[s] = np.asarray(states)      # -> disk (memmap), not RAM
        out_mm[s] = np.asarray(outcomes)
        valid_mm[s] = np.asarray(value_valid)
        if (s + 1) % 2000 == 0:
            states_mm.flush()                  # bound dirty-page buildup
            print(f"[cache]   {s + 1}/{steps}")

    states_mm.flush(); out_mm.flush(); valid_mm.flush()
    del states_mm, out_mm, valid_mm            # close write handles
    del buf; gc.collect()                      # FREE the 2.35 GB buffer before training
    m_path.write_text(json.dumps({
        "buffer_sha": buf_sha, "buffer_size": int(n),
        "steps": steps, "batch": batch, "encoding": ENCODING,
    }))
    print(f"[cache] wrote {cache_dir} (freed buffer)")
    return _open_ro(s_path, o_path, v_path, shape_s, shape_2)


def train_arm(
    arm: str,
    trunk_path: str,
    buffer_path: str,
    steps: int,
    seed: int,
    lr: float,
    out_dir: Path,
    batch: int,
    batch_cache: Path,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[arm {arm}] device={device} steps={steps} seed={seed} lr={lr}")

    is_bin = arm in ("B", "C")

    # Shared batch stream (identical across arms), disk-backed memmaps (read-only).
    cache = build_or_load_batch_cache(buffer_path, steps, batch, batch_cache)
    states_all = cache["states"]      # memmap (steps, batch, 4, 19, 19) f16 on disk
    out_all = cache["outcomes"]       # memmap (steps, batch) f32 on disk
    valid_all = cache["value_valid"]  # memmap (steps, batch) u8 on disk
    buffer_sha = json.loads((batch_cache / "manifest.json").read_text())["buffer_sha"]

    model = load_trunk(trunk_path, device)
    head = build_head(arm).to(device)
    param_groups = freeze_for_arm(model, head, arm, head_lr=lr)
    for g in param_groups:
        g.setdefault("weight_decay", WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=WEIGHT_DECAY)

    model.train()
    head.train()
    # tower[11] batchnorm/GN stays in train mode only for C/D; frozen trunk
    # blocks have no running stats (GroupNorm), so eval/train is irrelevant for
    # the frozen portion. Keep model.train() for the unfrozen block's GN affine.

    loss_curve = []
    t0 = time.perf_counter()
    for s in range(steps):
        # Copy ONE batch out of the disk memmap (~0.7 MB) -> device. Bounded RAM.
        states = torch.from_numpy(np.ascontiguousarray(states_all[s])).to(device).float()
        outcomes = torch.from_numpy(np.ascontiguousarray(out_all[s])).to(device)
        value_mask = torch.from_numpy(np.ascontiguousarray(valid_all[s])).to(device)

        feat = value_feature(model, states)            # (batch, 256)
        logits = head(feat)                            # (batch, 1) or (batch, 65)
        if is_bin:
            loss = two_hot_ce_loss(logits, outcomes, value_mask)
        else:
            loss = compute_value_loss(logits, outcomes, value_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if s % 500 == 0 or s == steps - 1:
            lv = float(loss.detach())
            loss_curve.append({"step": s, "loss": lv})
            if not np.isfinite(lv):
                raise RuntimeError(f"[arm {arm}] non-finite loss at step {s}: {lv}")
            print(f"[arm {arm}] step {s:6d} loss {lv:.5f}")

    wall = time.perf_counter() - t0

    # Save head state + tower[11] state (C/D) so scoring uses the trained trunk.
    out_dir.mkdir(parents=True, exist_ok=True)
    head_state = {k: v.cpu() for k, v in head.state_dict().items()}
    save_blob: dict = {
        "arm": arm,
        "seed": seed,
        "lr": lr,
        "steps": steps,
        "head_shape": "bin65" if is_bin else "scalar",
        "head_state": head_state,
        "trunk_ckpt": trunk_path,
        "buffer_sha": buffer_sha,
    }
    if arm in ("C", "D"):
        blk = model.trunk.tower[11]
        save_blob["tower11_state"] = {k: v.cpu() for k, v in blk.state_dict().items()}

    head_path = out_dir / f"head_{arm}_seed{seed}.pt"
    torch.save(save_blob, head_path)

    curve_path = out_dir / f"loss_curve_{arm}_seed{seed}.json"
    with open(curve_path, "w") as f:
        json.dump(
            {
                "arm": arm, "seed": seed, "lr": lr, "steps": steps,
                "wall_s": wall, "curve": loss_curve,
                "loss_first": loss_curve[0]["loss"] if loss_curve else None,
                "loss_last": loss_curve[-1]["loss"] if loss_curve else None,
            },
            f, indent=2,
        )
    print(f"[arm {arm}] wrote {head_path} + {curve_path} ({wall:.1f}s)")
    return {
        "head_path": str(head_path),
        "curve_path": str(curve_path),
        "loss_first": loss_curve[0]["loss"] if loss_curve else None,
        "loss_last": loss_curve[-1]["loss"] if loss_curve else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="D-F HEADSWAP single-arm trainer")
    ap.add_argument("--arm", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--trunk", required=True)
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, default=2e-3, help="head LR (production 1x=2e-3)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument(
        "--batch-cache", default=None,
        help="shared batch-stream memmap DIR (default: <out>/../batch_cache_s<steps>_b<batch>)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.batch_cache:
        batch_cache = Path(args.batch_cache)
    else:
        batch_cache = out_dir.parent / f"batch_cache_s{args.steps}_b{args.batch}"

    res = train_arm(
        arm=args.arm,
        trunk_path=args.trunk,
        buffer_path=args.buffer,
        steps=args.steps,
        seed=args.seed,
        lr=args.lr,
        out_dir=out_dir,
        batch=args.batch,
        batch_cache=batch_cache,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
