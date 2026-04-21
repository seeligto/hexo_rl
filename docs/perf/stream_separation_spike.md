# Phase C3 — Stream-separation feasibility spike

**Status**: read-only design note + scratch prototype. No integration this session.
**Gated on**: Phase C1 confirming Q18-style iso-vs-live inference gap.

---

## Hypothesis

InferenceServer currently runs H2D → forward → D2H on the **default CUDA
stream** (confirmed Bucket 1 #4 and verified at runtime via B4 `cuda_stream_audit`
event). Because the default stream is implicitly synchronised with every other
default-stream op in the process — including the training thread's forward,
backward, optimizer step, and any `.cpu()`/`.item()` — inference H2D copies
serialize against trainer compute and vice versa.

With a dedicated `torch.cuda.Stream()` assigned to the InferenceServer thread,
the scheduler can issue inference H2D while the trainer's backward kernel is
still executing, and inference forward can begin as soon as its H2D is done —
even if the trainer's D2H scalar sync is pending.

**In isolation** (no concurrent trainer), dedicated-stream forward latency
should match default-stream forward latency within noise. The win materialises
only under concurrent trainer load.

---

## Scratch prototype (do NOT commit to integration)

Save to `/tmp/stream_spike.py` — run standalone. Do **not** import from
`hexo_rl`. Measures the "does it still work?" + "same in-isolation latency?"
case. Does **not** measure the real win (which requires concurrent trainer load).

```python
"""Stream-separation feasibility spike — do NOT integrate.

Verifies:
  1. torch.cuda.Stream() works with the current model + input shape.
  2. Dedicated-stream forward latency matches default-stream in isolation.
  3. H2D on dedicated stream + forward on dedicated stream + D2H on dedicated
     stream does not deadlock, given a pinned host buffer + events.

Does NOT verify:
  - Actual win under concurrent trainer load (that requires a training-thread
    sidecar and is the integration test, not the spike).
"""
from __future__ import annotations

import argparse
import statistics as S
import time

import numpy as np
import torch

from hexo_rl.model.network import HexTacToeNet


def run(batch_size: int, iters: int, use_stream: bool, pinned: bool) -> list[float]:
    device = torch.device("cuda")
    model = HexTacToeNet(
        board_size=19, in_channels=18, filters=128, res_blocks=12, se_ratio=4,
    ).to(device).eval()

    shape = (batch_size, 18, 19, 19)
    host_np = np.random.rand(*shape).astype(np.float32)
    if pinned:
        host = torch.from_numpy(host_np).pin_memory()
    else:
        host = torch.from_numpy(host_np)

    stream = torch.cuda.Stream(device=device) if use_stream else None

    # warmup
    for _ in range(10):
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            t = host.to(device, non_blocking=pinned).reshape(*shape)
            _ = model(t)
    torch.cuda.synchronize()

    times_us = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if use_stream:
            with torch.cuda.stream(stream):
                with torch.no_grad(), torch.autocast(device_type="cuda"):
                    t = host.to(device, non_blocking=pinned).reshape(*shape)
                    log_p, v, _ = model(t)
                    p = log_p.float().exp()
                    p = p / p.sum(dim=-1, keepdim=True)
                    p_host = p.to("cpu", non_blocking=pinned)
                    v_host = v.squeeze(-1).float().to("cpu", non_blocking=pinned)
                stream.synchronize()
        else:
            with torch.no_grad(), torch.autocast(device_type="cuda"):
                t = host.to(device, non_blocking=pinned).reshape(*shape)
                log_p, v, _ = model(t)
                p = log_p.float().exp()
                p = p / p.sum(dim=-1, keepdim=True)
                _ = p.cpu().numpy()
                _ = v.squeeze(-1).float().cpu().numpy()
            torch.cuda.synchronize()
        times_us.append((time.perf_counter() - t0) * 1e6)
    return times_us


def pct(xs, p):
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    print(f"# Spike: batch_size={args.batch_size} iters={args.iters}")
    for use_stream, pinned, label in [
        (False, False, "default-stream + unpinned"),
        (False, True,  "default-stream + pinned"),
        (True,  True,  "dedicated-stream + pinned"),
    ]:
        t = run(args.batch_size, args.iters, use_stream, pinned)
        print(f"  {label:30s}  p50={pct(t,0.5):6.0f} us  p95={pct(t,0.95):6.0f} us  "
              f"mean={S.mean(t):6.0f} us  n={len(t)}")
```

### Expected results (pre-run prediction)

In isolation (no concurrent trainer):

- `default-stream + unpinned` — baseline current behaviour
- `default-stream + pinned`   — slightly faster H2D; forward unchanged
- `dedicated-stream + pinned` — within ~1–2% of default-stream + pinned

That is: isolation measurement should confirm the change is **not a regression**
but also does **not** show a win. The win is observable only under concurrent
trainer load.

### What would invalidate the hypothesis?

- If dedicated-stream forward latency in isolation is >5% slower than default-stream,
  there is some misuse (e.g. missing stream.wait_event, creating a new stream per
  forward, or host pin_memory interacting badly with the allocator).
- If the script deadlocks, the stream-event protocol is wrong — usually a
  missing `stream.synchronize()` before reading `.cpu()` host tensor.

---

## Integration path (do NOT land this session)

Post-baseline (after gumbel_full sustained run completes), integration would be
a one-commit-per-change sequence:

1. **Pin host input buffer** in InferenceServer (`inference_server.py:109`).
   - Replace `torch.from_numpy(batch_np).to(device).reshape(...)` with
     pre-allocated `self._host_in = torch.empty(..., pin_memory=True)` +
     `self._host_in[:n].copy_(torch.from_numpy(batch_np).view(n, *shape))` +
     `self._dev_in = self._host_in.to(device, non_blocking=True)`.
   - Bench A/B: iso NN inference pos/s (expect flat), live pos/hr (expect small win).
2. **Fuse D2H transfer** (`inference_server.py:121-122`) — stack policy+value on
   device, single `.cpu()`.
   - Bench A/B: same.
3. **Dedicated inference CUDA stream**.
   - `self._stream = torch.cuda.Stream(device=device)` in `__init__`.
   - Wrap forward path in `with torch.cuda.stream(self._stream):`.
   - `self._stream.synchronize()` before `.cpu()` read.
   - Bench A/B: iso NN inference pos/s (expect flat), live pos/hr under training
     load (expect **primary Q18 lever** win).
4. **Optional: channels_last** at the input + model. Gate on bench — 19×19
   spatial may not yield a win.

Each step is ~30–80 LoC, feature-flaggable, reversible.

---

## Success criteria for the integrated change

Post-integration sustained run should exhibit:

- Live `inference_batch_timing.forward_us` p50 within 10% of bench NN latency
  (currently ~7.8×).
- Worker idle % (from a future Rust probe) lower than the reference run.
- `train_step_timing.total_us` p50 unchanged (trainer path not touched).
- Overall games/hr up by a fraction corresponding to the closed iso-vs-live gap.

If live/iso ratio remains large after stream separation, the bottleneck is
elsewhere (likely Rust queue-lock contention — Bucket 5 #7).
