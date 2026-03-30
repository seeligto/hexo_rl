"""Rust-driven batched GPU inference server.

Rust owns request concurrency via `RustInferenceBatcher`. Python only runs a
thin `while True` loop: fetch fused batch from Rust, execute model forward,
submit policy/value outputs back to Rust, and wake blocked game threads.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import numpy as np
import torch

from native_core import RustInferenceBatcher  # type: ignore[attr-defined]
from python.model.network import HexTacToeNet


# ── Server ────────────────────────────────────────────────────────────────────

class InferenceServer(threading.Thread):
    """Thin Python inference loop backed by a Rust-owned batching queue."""

    def __init__(
        self,
        model: HexTacToeNet,
        device: torch.device,
        config: Dict[str, Any],
        batcher: Optional[RustInferenceBatcher] = None,
    ) -> None:
        super().__init__(daemon=True, name="inference-server")
        self.model = model
        self.device = device

        sp = config.get("selfplay", config)
        self._batch_size = int(sp.get("inference_batch_size", 64))
        self._max_wait_ms = int(float(sp.get("inference_max_wait_ms", 10.0)))

        board_size = int(getattr(model, "board_size", 19))
        in_channels = int(config.get("in_channels", config.get("model", {}).get("in_channels", 18)))
        self._policy_len = board_size * board_size + 1
        self._feature_len = in_channels * board_size * board_size
        self._shape = (in_channels, board_size, board_size)

        self._batcher = batcher or RustInferenceBatcher(
            feature_len=self._feature_len,
            policy_len=self._policy_len,
        )
        self._stop_event = threading.Event()
        self._forward_count = 0
        self._total_requests = 0

    @property
    def batcher(self) -> RustInferenceBatcher:
        return self._batcher

    def stop(self) -> None:
        self._stop_event.set()
        self._batcher.close()

    def submit_and_wait(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """Compatibility helper for single direct requests via Rust queue."""
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        policy, value = self._batcher.submit_request_and_wait(arr.tolist())
        return np.asarray(policy, dtype=np.float32), float(value)

    def infer(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        return self.submit_and_wait(state)

    def infer_many(self, states: list[np.ndarray]) -> list[tuple[np.ndarray, float]]:
        return [self.submit_and_wait(state) for state in states]

    @property
    def forward_count(self) -> int:
        return self._forward_count

    @property
    def total_requests(self) -> int:
        return self._total_requests

    # ── Thread body ───────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                request_ids, batch = self._batcher.next_inference_batch(
                    self._batch_size,
                    self._max_wait_ms,
                )
                if not request_ids:
                    continue

                self._total_requests += len(request_ids)
                batch_np = np.asarray(batch, dtype=np.float32)
                tensor = torch.from_numpy(batch_np).to(self.device).reshape(len(request_ids), *self._shape)

                try:
                    self.model.eval()
                    with torch.no_grad():
                        with torch.autocast(device_type=self.device.type):
                            log_policy, value = self.model(tensor)
                    policies = log_policy.exp().cpu().numpy().astype(np.float32)
                    values = value.squeeze(-1).cpu().numpy().astype(np.float32)
                except Exception:
                    # Never deadlock Rust waiters when model inference fails.
                    policies = np.ones((len(request_ids), self._policy_len), dtype=np.float32)
                    policies /= float(self._policy_len)
                    values = np.zeros((len(request_ids),), dtype=np.float32)

                self._batcher.submit_inference_results(
                    request_ids,
                    policies.tolist(),
                    values.tolist(),
                )
                self._forward_count += 1
        finally:
            # Ensure blocked Rust waiters are released even if this thread exits unexpectedly.
            self._batcher.close()
