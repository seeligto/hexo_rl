"""Rust-driven batched GPU inference server.

Rust owns request concurrency via `InferenceBatcher`. Python only runs a
thin `while True` loop: fetch fused batch from Rust, execute model forward,
submit policy/value outputs back to Rust, and wake blocked game threads.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

import structlog

from engine import InferenceBatcher  # type: ignore[attr-defined]
from hexo_rl.model.network import HexTacToeNet

# Perf probes log via structlog so they persist to JSONL independent of dashboards.
_perf_log = structlog.get_logger()


# ── Server ────────────────────────────────────────────────────────────────────

class InferenceServer(threading.Thread):
    """Thin Python inference loop backed by a Rust-owned batching queue."""

    def __init__(
        self,
        model: HexTacToeNet,
        device: torch.device,
        config: Dict[str, Any],
        batcher: Optional[InferenceBatcher] = None,
    ) -> None:
        super().__init__(daemon=True, name="inference-server")
        self.model = model
        self.model.eval()
        self.device = device

        sp = config.get("selfplay", config)
        self._batch_size = int(sp.get("inference_batch_size", 64))
        self._max_wait_ms = int(float(sp.get("inference_max_wait_ms", 10.0)))

        board_size = int(getattr(model, "board_size", 19))
        in_channels = int(config.get("in_channels", config.get("model", {}).get("in_channels", 18)))
        self._policy_len = board_size * board_size + 1
        self._feature_len = in_channels * board_size * board_size
        self._shape = (in_channels, board_size, board_size)

        self._batcher = batcher or InferenceBatcher(
            feature_len=self._feature_len,
            policy_len=self._policy_len,
        )
        self._stop_event = threading.Event()
        self._weights_lock = threading.Lock()
        self._forward_count = 0
        self._total_requests = 0

        # Pinned host staging buffer for async H2D (Bucket 1 #5, E2 row 1).
        # Size: batch_size * feature_len * 4B ≈ 1 MB for default (64 * 18*19*19 * 4).
        # Enables DMA engine copy on CUDA (non_blocking=True); no-op on CPU.
        if self.device.type == "cuda":
            self._h2d_staging = torch.empty(
                (self._batch_size, in_channels, board_size, board_size),
                dtype=torch.float32,
                pin_memory=True,
            )
        else:
            self._h2d_staging = None

        # Perf-investigation probes (docs/perf/instrumentation_notes.md).
        _diag = config.get("diagnostics") if isinstance(config.get("diagnostics"), dict) else {}
        self._perf_timing = bool(_diag.get("perf_timing", False))
        self._perf_sync_cuda = bool(_diag.get("perf_sync_cuda", False))

        # Autocast dtype — must match trainer for weight-sync consistency.
        # Config: "fp16" (default) or "bf16". bf16 on Ampere+/Ada avoids the
        # GradScaler overhead on the training side; inference-side just
        # enables the same autocast target.
        _amp_raw = str(config.get("amp_dtype", "fp16")).lower()
        if _amp_raw in ("fp16", "float16", "half"):
            self._amp_dtype = torch.float16
        elif _amp_raw in ("bf16", "bfloat16"):
            self._amp_dtype = torch.bfloat16
        else:
            raise ValueError(f"amp_dtype must be 'fp16' or 'bf16', got {_amp_raw!r}")

    @property
    def batcher(self) -> InferenceBatcher:
        return self._batcher

    def stop(self) -> None:
        self._stop_event.set()
        self._batcher.close()

    def load_state_dict_safe(self, state_dict: dict) -> None:
        """Thread-safe weight swap — blocks until any in-flight forward completes."""
        with self._weights_lock:
            self.model.load_state_dict(state_dict)
            self.model.eval()

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
        _perf = self._perf_timing
        _sync = self._perf_sync_cuda and self.device.type == "cuda"

        # B4: log which CUDA stream we're on, once at thread start.
        # Q18 smoking gun: if this matches trainer stream (both default), no overlap.
        if self.device.type == "cuda":
            try:
                current_stream = torch.cuda.current_stream(self.device)
                default_stream = torch.cuda.default_stream(self.device)
                _perf_log.info(
                    "cuda_stream_audit",
                    context="inference_server",
                    current_stream_ptr=int(current_stream.cuda_stream),
                    default_stream_ptr=int(default_stream.cuda_stream),
                    on_default_stream=current_stream.cuda_stream == default_stream.cuda_stream,
                )
            except Exception as exc:  # noqa: BLE001
                _perf_log.warning("cuda_stream_audit_failed", context="inference_server", error=str(exc))

        try:
            while not self._stop_event.is_set():
                try:
                    _t_fetch_start = time.perf_counter() if _perf else 0.0
                    request_ids, batch = self._batcher.next_inference_batch(
                        self._batch_size,
                        self._max_wait_ms,
                    )
                    if not request_ids:
                        continue
                    _t_fetched = time.perf_counter() if _perf else 0.0

                    self._total_requests += len(request_ids)

                    try:
                        # Ensure the batch is explicitly C-contiguous for safe pointer arithmetic in Rust as_slice()
                        # Coordinate Sentinels (usize::MAX): These are handled in the Rust core during tensor extraction;
                        # out-of-window indices are zeroed before reaching this fused batch.
                        batch_np = np.ascontiguousarray(batch, dtype=np.float32)
                        n = len(request_ids)
                        if self._h2d_staging is not None:
                            # Staged async H2D: CPU→pinned copy, then DMA to GPU.
                            # Previous batch's H2D is already complete by this point
                            # (prior forward + .cpu() synced default stream), so
                            # reusing the staging buffer is safe.
                            self._h2d_staging[:n].copy_(
                                torch.from_numpy(batch_np).view(n, *self._shape)
                            )
                            tensor = self._h2d_staging[:n].to(self.device, non_blocking=True)
                        else:
                            tensor = torch.from_numpy(batch_np).to(self.device).reshape(n, *self._shape)
                        if _perf:
                            if _sync:
                                torch.cuda.synchronize()
                            _t_h2d_done = time.perf_counter()
                        if self._forward_count == 0:
                            assert not self.model.training, (
                                "InferenceServer model entered hot loop in train() mode; "
                                "eval() should be set at __init__ and re-applied in load_state_dict_safe"
                            )
                        with self._weights_lock:
                            with torch.inference_mode():
                                with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype):
                                    log_policy, value, _v_logit = self.model(tensor)
                        if _perf:
                            if _sync:
                                torch.cuda.synchronize()
                            _t_forward_done = time.perf_counter()

                        # .float() ensures float32 regardless of autocast dtype
                        # (bfloat16 on CPU, float16 on CUDA — NumPy only supports float16).
                        # Re-normalize after exp() to correct reduced-precision rounding drift.
                        probs = log_policy.float().exp()
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                        policies = np.ascontiguousarray(probs.cpu().numpy())
                        values = np.ascontiguousarray(value.squeeze(-1).float().cpu().numpy())
                        if _perf:
                            _t_d2h_done = time.perf_counter()

                        self._batcher.submit_inference_results(
                            request_ids,
                            policies,
                            values,
                        )
                        if _perf:
                            _perf_log.info(
                                "inference_batch_timing",
                                batch_n=len(request_ids),
                                fetch_wait_us=(_t_fetched - _t_fetch_start) * 1e6,
                                h2d_us=(_t_h2d_done - _t_fetched) * 1e6,
                                forward_us=(_t_forward_done - _t_h2d_done) * 1e6,
                                d2h_scatter_us=(_t_d2h_done - _t_forward_done) * 1e6,
                                sync_cuda=_sync,
                                forward_count=self._forward_count + 1,
                            )
                    except Exception as exc:
                        # Explicitly signal failure to Rust waiters rather than returning dummy data
                        # or failing silently. This allows Rust to handle the error properly.
                        error_msg = f"Model inference failed: {str(exc)}"
                        self._batcher.submit_inference_failure(request_ids, error_msg)
                        # We don't raise here so the server can potentially recover for the next batch
                        continue
                    
                    self._forward_count += 1
                except Exception as exc:
                    import traceback
                    print(f"InferenceServer loop error: {exc}")
                    traceback.print_exc()
                    if self._stop_event.is_set():
                        break
        finally:
            # Ensure blocked Rust waiters are released even if this thread exits unexpectedly.
            self._batcher.close()
