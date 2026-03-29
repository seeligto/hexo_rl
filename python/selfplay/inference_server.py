"""
Batched GPU inference server.

Multiple concurrent self-play games accumulate leaf-evaluation requests
into a shared queue. When the queue reaches `batch_size` (or a wall-clock
timeout expires), the server fires a single model.forward() call and
distributes (policy, value) results back to the waiting callers.

This is the critical GPU-utilisation optimisation: without batching, each
game fires a single-position forward pass and the GPU sits idle between
calls.  With batching, N games fill one batch and the GPU is kept busy.

Thread model
────────────
  InferenceServer — one daemon thread.  Games push InferenceRequest objects
  to a queue, block on their per-request threading.Event, and read the
  result after the event fires.

  This is a single-process (multi-thread) design.  The WorkerPool (pool.py)
  wraps this for multi-process use via mp.Queue.

Usage
─────
    server = InferenceServer(model, device, config)
    server.start()

    # From any thread:
    policy, value = server.infer(state_tensor)  # blocks until batched

    server.stop()
    server.join()
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch

from python.model.network import HexTacToeNet


# ── Request ───────────────────────────────────────────────────────────────────

@dataclass
class InferenceRequest:
    """A single leaf-evaluation request.

    The caller creates this, puts it on the server queue, waits on `event`,
    then reads `result`.
    """
    state: np.ndarray          # (K, 18, 19, 19) float16
    event: threading.Event = field(default_factory=threading.Event)
    result: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None)
    # result is (cluster_policies, cluster_values):
    #   cluster_policies shape (K, 362) float32
    #   cluster_values   shape (K,) float32


# ── Server ────────────────────────────────────────────────────────────────────

class InferenceServer(threading.Thread):
    """Daemon thread that batches leaf-state inference requests.

    Args:
        model:       The neural network (HexTacToeNet).
        device:      torch.device to run inference on.
        config:      Config dict.  Used keys (under ``selfplay`` sub-dict if
                     present, else top-level):
                       inference_batch_size  (default 64)
                       inference_max_wait_ms (default 10.0)
    """

    def __init__(
        self,
        model: HexTacToeNet,
        device: torch.device,
        config: Dict[str, Any],
    ) -> None:
        super().__init__(daemon=True, name="inference-server")
        self.model  = model
        self.device = device

        sp = config.get("selfplay", config)
        self._batch_size   = int(sp.get("inference_batch_size",  64))
        self._max_wait_s   = float(sp.get("inference_max_wait_ms", 10.0)) / 1000.0

        self._queue: queue.Queue[InferenceRequest] = queue.Queue()
        self._stop  = threading.Event()

        # Counters (read from tests / monitoring; not locked — approximate ok).
        self._forward_count: int = 0
        self._total_requests: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize to (K, 18, 19, 19) float16 contiguous array."""
        state_arr = np.asarray(state)
        if state_arr.ndim == 3:
            state_arr = state_arr[None, ...]
        return np.ascontiguousarray(state_arr, dtype=np.float16)

    def infer_many(self, states: list[np.ndarray]) -> list[Tuple[np.ndarray, np.ndarray]]:
        """Submit many states and wait until all are resolved."""
        if not states:
            return []

        requests: list[InferenceRequest] = []
        for state in states:
            req = InferenceRequest(state=self._normalize_state(state))
            self._queue.put(req)
            requests.append(req)

        results: list[Tuple[np.ndarray, np.ndarray]] = []
        for req in requests:
            req.event.wait()
            assert req.result is not None
            results.append(req.result)
        return results

    def infer(self, state: np.ndarray):
        """Submit `state` for inference; block until the result is ready.

        Args:
            state: Board tensor, shape (18, 19, 19), dtype float16.

        Returns:
            policy_probs: np.ndarray, shape (362,), float32, sums to 1.
            value:        float in [-1, 1] from current player's perspective.
        """
        result = self.infer_many([state])[0]
        cluster_policies, cluster_values = result
        if np.asarray(state).ndim == 3:
            return cluster_policies[0], float(cluster_values[0])
        return result

    def stop(self) -> None:
        """Signal the server to stop after draining its current batch."""
        self._stop.set()

    @property
    def forward_count(self) -> int:
        return self._forward_count

    @property
    def total_requests(self) -> int:
        return self._total_requests

    # ── Thread body ───────────────────────────────────────────────────────────

    def run(self) -> None:
        while not self._stop.is_set():
            batch = self._collect_batch()
            if batch:
                self._process_batch(batch)

        # Drain any remaining requests so callers don't hang on stop().
        while True:
            try:
                batch = self._collect_batch(drain=True)
                if not batch:
                    break
                self._process_batch(batch)
            except Exception:
                break

    # ── Internal ──────────────────────────────────────────────────────────────

    def _collect_batch(self, *, drain: bool = False) -> list[InferenceRequest]:
        """Collect up to `_batch_size` requests.

        Blocks up to `_max_wait_s` for the first item, then greedily reads
        more until `_batch_size` is reached or the deadline expires.

        With `drain=True`, uses a very short timeout (no blocking).
        """
        batch: list[InferenceRequest] = []
        timeout = 0.001 if drain else self._max_wait_s

        try:
            first = self._queue.get(timeout=timeout)
            batch.append(first)
        except queue.Empty:
            return batch

        deadline = time.monotonic() + self._max_wait_s
        while len(batch) < self._batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = self._queue.get(timeout=remaining)
                batch.append(req)
            except queue.Empty:
                break

        return batch

    def _process_batch(self, batch: list[InferenceRequest]) -> None:
        """Run one forward pass over the batch and resolve all requests."""
        self._total_requests += len(batch)

        # states are (K, 18, 19, 19). We concatenate them along dim=0.
        k_sizes = [req.state.shape[0] for req in batch]
        
        # Concatenate states -> (sum(K), 18, 19, 19)
        tensor = torch.cat([torch.from_numpy(req.state) for req in batch], dim=0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type):
                log_policy, value = self.model(tensor.float())

        # log_policy: (sum(K), 362) log-softmax  →  exp → probabilities
        policies_np = log_policy.exp().cpu().numpy().astype(np.float32)  # (sum(K), 362)
        values_np   = value.squeeze(-1).cpu().numpy()                    # (sum(K),)

        # Slice results back to each request
        offset = 0
        for i, req in enumerate(batch):
            k = k_sizes[i]
            req_policies = policies_np[offset:offset+k]
            req_values = values_np[offset:offset+k]
            req.result = (req_policies, req_values)
            req.event.set()
            offset += k

        self._forward_count += 1
