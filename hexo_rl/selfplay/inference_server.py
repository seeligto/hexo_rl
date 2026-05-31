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
from hexo_rl.encoding import EncodingSpec as RegistrySpec
from hexo_rl.encoding import resolve_from_config as registry_resolve_from_config
from hexo_rl.model.network import HexTacToeNet, WIRE_CHANNELS

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
        encoding_spec: Optional[RegistrySpec] = None,
    ) -> None:
        super().__init__(daemon=True, name="inference-server")
        self.model = model
        self.model.eval()
        self.device = device

        sp = config.get("selfplay", config)
        self._batch_size = int(sp.get("inference_batch_size", 64))
        self._max_wait_ms = int(float(sp.get("inference_max_wait_ms", 10.0)))

        # §172 A4.2 — encoding spec sourced from new `hexo_rl.encoding`
        # registry. §176 P3 — legacy `hexo_rl.utils.encoding` NamedTuple
        # shim retired; only the registry dataclass is accepted now.
        # Standalone callers (no kwarg) fall back to resolving from config
        # (default v6).
        if encoding_spec is None:
            self.encoding_spec: RegistrySpec = registry_resolve_from_config(config)
        elif isinstance(encoding_spec, RegistrySpec):
            self.encoding_spec = encoding_spec
        else:
            raise TypeError(
                f"InferenceServer: unrecognised encoding_spec type "
                f"{type(encoding_spec).__name__!r}; expected hexo_rl.encoding.EncodingSpec"
            )
        # H2D staging tensors size to the trunk window (the spatial dim the
        # model actually accepts). For v6/v6w25/v7*/v8 trunk_size == board_size,
        # so this is currently a no-op semantic shift; future α multi-window
        # K-cluster encodings will diverge (trunk_size != board_size). §176 P18.
        board_size = self.encoding_spec.trunk_size
        # Policy len from the registry — `policy_logit_count` already
        # accounts for the per-encoding pass-slot bit (v8: 625, v6/v7full: 362,
        # v6w25: 626).
        self._policy_len = self.encoding_spec.policy_logit_count
        # Rust workers emit exactly `spec.kept_plane_indices` planes (v6 → 8,
        # v6tp → 10 incl. turn-phase 16/17). The wire width is the active
        # encoding's plane count — NOT the v6-hardcoded WIRE_CHANNELS=8.
        # Sub-selection (sweep `input_channels` variants) happens inside
        # model.forward() via index_select, but the wire/H2D width is what
        # Rust actually sends = spec.n_planes.
        wire_channels = self.encoding_spec.n_planes
        self._feature_len = wire_channels * board_size * board_size
        self._shape = (wire_channels, board_size, board_size)

        self._batcher = batcher or InferenceBatcher(
            feature_len=self._feature_len,
            policy_len=self._policy_len,
        )
        self._stop_event = threading.Event()
        self._weights_lock = threading.Lock()
        self._forward_count = 0
        self._total_requests = 0

        self._setup_inference_path(sp, board_size)

        # Pinned host staging buffer for async H2D (Bucket 1 #5, E2 row 1).
        # Size: batch_size * feature_len * 4B (e.g. 64 * 8 * 19 * 19 * 4 ≈ 0.5 MB
        # at WIRE_CHANNELS=8). Enables DMA engine copy on CUDA
        # (non_blocking=True); no-op on CPU.
        if self.device.type == "cuda":
            self._h2d_staging = torch.empty(
                (self._batch_size, wire_channels, board_size, board_size),
                dtype=torch.float32,
                pin_memory=True,
            )
        else:
            self._h2d_staging = None

        # Perf-investigation probes (docs/perf/instrumentation_notes.md).
        _diag = config.get("diagnostics") if isinstance(config.get("diagnostics"), dict) else {}
        self._perf_timing = bool(_diag.get("perf_timing", False))
        self._perf_sync_cuda = bool(_diag.get("perf_sync_cuda", False))
        if self._perf_sync_cuda and torch.cuda.is_available():
            _perf_log.warning(
                "perf_sync_cuda_enabled_serialising_stream",
                context="inference_server",
                impact="expect_30_50_pct_throughput_drop",
                remedy="unset_diagnostics.perf_sync_cuda_in_production_config",
            )

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

    def _setup_inference_path(self, sp: dict, board_size: int) -> None:
        """Configure trace OR compile path for the inference model (§176 P22).

        Mutually exclusive: trace_inference and compile_inference can't both
        be enabled. Sets self._trace_inference, self._traced_model,
        self._compile_inference, self._compile_mode, self._compile_dynamic.
        May replace self.model with a torch.compile wrapper.

        Called once at __init__. Run-loop hot path reads the resolved
        attributes; no per-batch overhead from this helper.
        """
        # TorchScript trace of the eval forward. The 2026-04-25 py-spy profile
        # (project_dispatch_pyspy_2026-04-25.md) attributed ~33% of dispatcher
        # wall time to L208 — pure CPython overhead iterating ~100 nn.Module
        # _call_impl invocations per forward (12 ResBlocks × 7 modules + 7
        # heads). Tracing collapses that into a single ScriptModule whose
        # parameters share storage with `model`, so load_state_dict_safe's
        # in-place tensor mutation continues to flow into the traced graph
        # without re-tracing.
        self._trace_inference = bool(sp.get("trace_inference", True))
        self._traced_model: Optional[torch.jit.ScriptModule] = None
        if self._trace_inference:
            try:
                self.model.requires_grad_(False)
                with torch.inference_mode():
                    _example = torch.zeros(
                        self._batch_size, *self._shape, device=self.device,
                    )
                    self._traced_model = torch.jit.trace(
                        self.model, _example, strict=False,
                    )
                _perf_log.info(
                    "inference_trace_compiled",
                    context="inference_server",
                    batch_size=self._batch_size,
                    board_size=board_size,
                )
            except Exception as exc:  # noqa: BLE001
                _perf_log.warning(
                    "inference_trace_failed_falling_back",
                    context="inference_server",
                    error=str(exc)[:200],
                )
                self._traced_model = None

        # torch.compile knob (compile_retry_20260426 Phase 2). Mutually
        # exclusive with trace — both attack the same bottleneck (Python
        # dispatch / kernel-launch overhead) and stacking them does not
        # compose (probe arm 6 confirmed). Mode `default` is thread-safe
        # from any caller; `reduce-overhead` requires the dispatcher
        # thread's TLS to own the cudagraph_trees context (Phase 3 work).
        self._compile_inference = bool(sp.get("compile_inference", False))
        self._compile_mode = str(sp.get("compile_inference_mode", "default"))
        self._compile_dynamic = bool(sp.get("compile_inference_dynamic", True))
        if self._compile_inference and self._trace_inference:
            raise ValueError(
                "compile_inference and trace_inference are mutually exclusive; "
                "set one to false in the selfplay config."
            )
        if self._compile_inference:
            try:
                self.model = torch.compile(
                    self.model,
                    mode=self._compile_mode,
                    dynamic=self._compile_dynamic,
                )
                _perf_log.info(
                    "inference_compile_enabled",
                    context="inference_server",
                    mode=self._compile_mode,
                    dynamic=self._compile_dynamic,
                )
            except Exception as exc:  # noqa: BLE001
                _perf_log.warning(
                    "inference_compile_failed_falling_back",
                    context="inference_server",
                    error=str(exc)[:200],
                )
                self._compile_inference = False

    @property
    def batcher(self) -> InferenceBatcher:
        return self._batcher

    def stop(self) -> None:
        self._stop_event.set()
        self._batcher.close()

    def load_state_dict_safe(self, state_dict: dict) -> None:
        """Thread-safe weight swap — blocks until any in-flight forward completes.

        Callers pass bare (non-``_orig_mod.*``-prefixed) state_dict keys.
        When ``self.model`` is a ``torch._dynamo.OptimizedModule`` (compile
        path), ``load_state_dict`` would otherwise demand the prefixed keys.
        Unwrap once here so the load targets the underlying parameters in
        place; the OptimizedModule wrapper continues to dispatch through
        them unchanged. Same propagation path the trace path relies on.

        Phase B' Class-1: bumps the InferenceBatcher's monotonic
        `model_version` after the swap so workers can attribute each move
        to a specific weight epoch (Class-1 stale-dispatch probe).
        """
        with self._weights_lock:
            target = getattr(self.model, "_orig_mod", self.model)
            target.load_state_dict(state_dict)
            target.eval()
            self.model.eval()
        # Bump after release — workers reading the atomic don't gate on
        # the lock, only on the post-swap visibility of new params.
        new_version = self._batcher.bump_model_version()
        _perf_log.info(
            "inference_model_version_bump",
            context="inference_server",
            model_version=new_version,
        )

    def submit_and_wait(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """Synchronous single-state inference for test / diagnostic use.

        Runs the model forward in-process under ``_weights_lock``, mirroring
        the dispatcher loop's hot path (trace/compile model selection, prep,
        autocast). Bypasses the Rust ``InferenceBatcher`` queue — production
        self-play goes through the dispatcher thread + Rust workers and
        does not call this method.

        Post-`00b7d2b` the Rust-side single-request PyO3 surface
        ``submit_request_and_wait`` is retired; this method is the
        coordinated Python-side replacement (see `00b7d2b` commit body
        and `audit/rust-engine/wave_3/pre_3d/H1_recon.md` Group C).

        Raises:
            ValueError: prefixed with ``"Model inference failed: "`` if the
                wrapped model forward raises. Translates the underlying
                ``RuntimeError`` so callers can wait on ``threading.Event``
                without deadlocking on a thread-bound exception (the dispatcher
                path carries the same contract through
                ``submit_inference_failure``).
        """
        # Match dispatcher's batch prep contract (explicit C-contiguous f32).
        # Patching ``np.ascontiguousarray`` in this module from the test suite
        # triggers ValueError here, mirroring the dispatcher's batch-prep
        # error path and unblocking the caller (no Rust queue waiter exists
        # under the direct path).
        arr = np.ascontiguousarray(state, dtype=np.float32).reshape(self._shape)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        # Choose the traced graph when available — shares parameter storage
        # with ``self.model`` so weight swaps via ``load_state_dict_safe``
        # propagate without re-tracing (same path the dispatcher uses).
        fwd_model = self._traced_model if self._traced_model is not None else self.model
        try:
            with self._weights_lock:
                with torch.inference_mode():
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self._amp_dtype,
                        enabled=self.device.type == "cuda",
                    ):
                        log_policy, value, _v_logit = fwd_model(tensor)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Model inference failed: {exc}") from exc

        probs = log_policy.float().exp()
        probs = probs / probs.sum(dim=-1, keepdim=True)
        policy_np = probs.squeeze(0).cpu().numpy().astype(np.float32)
        value_f = float(value.squeeze().cpu().item())

        self._total_requests += 1
        self._forward_count += 1
        return policy_np, value_f

    def infer(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        return self.submit_and_wait(state)

    @property
    def forward_count(self) -> int:
        return self._forward_count

    @property
    def total_requests(self) -> int:
        return self._total_requests

    # ── Thread body ───────────────────────────────────────────────────────────

    def _padding_active(self) -> bool:
        """Compile-inference reduce-overhead path uses CUDA-graph replay; the
        captured graph requires a fixed input shape, so each batch must be
        padded up to ``self._batch_size`` and outputs sliced back to the
        actual request count.
        """
        return (
            self._compile_inference
            and self._compile_mode == "reduce-overhead"
            and self._h2d_staging is not None
        )

    def _warmup_compile_path(self) -> None:
        """CUDA-graph TLS warmup for compile+reduce-overhead (§123, §176 P22).

        The cudagraph_trees state lives in C++ dynamic TLS — the first
        forward must run on this dispatcher thread so the captured graph
        binds here, not to the main thread that built the OptimizedModule
        wrapper. We also pad the warmup tensor to the production batch_size
        so the graph is captured for the steady-state shape, not a smaller
        one-off. Failures degrade to fall-back-on-first-batch behaviour.

        Called once at the top of run() (per-thread). No-op for non-CUDA
        or non-compile+reduce-overhead configurations.
        """
        if (
            self._compile_inference
            and self._compile_mode == "reduce-overhead"
            and self.device.type == "cuda"
        ):
            try:
                with self._weights_lock:
                    with torch.inference_mode():
                        with torch.autocast(
                            device_type=self.device.type,
                            dtype=self._amp_dtype,
                        ):
                            if self._h2d_staging is not None:
                                self._h2d_staging.zero_()
                                warmup_tensor = self._h2d_staging.to(
                                    self.device, non_blocking=True,
                                )
                            else:
                                warmup_tensor = torch.zeros(
                                    self._batch_size, *self._shape,
                                    device=self.device,
                                )
                            _ = self.model(warmup_tensor)
                torch.cuda.synchronize()
                _perf_log.info(
                    "inference_compile_warmup_dispatcher",
                    context="inference_server",
                    batch_size=self._batch_size,
                    mode=self._compile_mode,
                )
            except Exception as exc:  # noqa: BLE001
                _perf_log.warning(
                    "inference_compile_warmup_failed",
                    context="inference_server",
                    error=str(exc)[:200],
                )

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

        self._warmup_compile_path()

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
                        # §P75 (Wave 5a Batch E): Rust contract on the bound
                        # supplier guarantees `batch` is already a float32
                        # C-contiguous numpy array — `next_inference_batch`
                        # returns `PyArray1::from_vec(py, flat_features)
                        # .reshape([n, feature_len])` (see
                        # engine/src/inference_bridge.rs:~370), and
                        # `IntoPyArray` builds a contiguous buffer that
                        # `reshape` cannot strip without copying. The
                        # previous defensive `np.ascontiguousarray(batch,
                        # dtype=np.float32)` performed an unconditional
                        # ~740 KB memcpy per batch (v6 batch=64 ×
                        # feature_len=2888 × 4 B) + ~1-2 µs of dtype
                        # inspection. Replaced with a debug-only assert
                        # that disappears under `python -O`.
                        if __debug__:
                            assert batch.dtype == np.float32, (
                                f"InferenceBatcher.next_inference_batch returned dtype={batch.dtype}; "
                                "Rust contract guarantees float32"
                            )
                            assert batch.flags["C_CONTIGUOUS"], (
                                f"InferenceBatcher.next_inference_batch returned flags={batch.flags}; "
                                "Rust contract guarantees C-contiguous"
                            )
                        batch_np = batch
                        n = len(request_ids)
                        _pad = self._padding_active()
                        if self._h2d_staging is not None:
                            assert n <= self._batch_size, (
                                f"inference batch size {n} exceeds staging capacity "
                                f"{self._batch_size} — config divergence between "
                                f"InferenceBatcher and InferenceServer"
                            )
                            # Staged async H2D: CPU→pinned copy, then DMA to GPU.
                            # Previous batch's H2D is already complete by this point
                            # (prior forward + .cpu() synced default stream), so
                            # reusing the staging buffer is safe.
                            self._h2d_staging[:n].copy_(
                                torch.from_numpy(batch_np).view(n, *self._shape)
                            )
                            if _pad:
                                # Zero padding for compile+reduce-overhead's CUDA
                                # graph (fixed shape required). Padded rows are
                                # discarded post-forward via host[:n] slicing.
                                if n < self._batch_size:
                                    self._h2d_staging[n:].zero_()
                                tensor = self._h2d_staging.to(
                                    self.device, non_blocking=True,
                                )
                            else:
                                tensor = self._h2d_staging[:n].to(
                                    self.device, non_blocking=True,
                                )
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
                        # Use the traced graph when available (the trace shares
                        # parameter storage with self.model, so weight swaps
                        # via load_state_dict_safe propagate without re-tracing).
                        fwd_model = self._traced_model if self._traced_model is not None else self.model
                        with self._weights_lock:
                            with torch.inference_mode():
                                # autocast enabled on CUDA only; CPU float16 is unsupported
                                # (CPU autocast accepts bfloat16 only — disable entirely on CPU).
                                with torch.autocast(
                                    device_type=self.device.type,
                                    dtype=self._amp_dtype,
                                    enabled=self.device.type == "cuda",
                                ):
                                    log_policy, value, _v_logit = fwd_model(tensor)
                        if _perf:
                            if _sync:
                                torch.cuda.synchronize()
                            _t_forward_done = time.perf_counter()

                        # .float() ensures float32 regardless of autocast dtype
                        # (bfloat16 on CPU, float16 on CUDA — NumPy only supports float16).
                        # Re-normalize after exp() to correct reduced-precision rounding drift.
                        probs = log_policy.float().exp()
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                        # Merged D2H: one cudaMemcpyAsync instead of two.
                        # Layout: [fwd_n, policy_len + 1] — last column carries
                        # the squeezed scalar value. Splitting on host is
                        # L2-cache cheap (~70 KB at batch=192) compared to a
                        # second D2H.
                        v = value.squeeze(-1).float().unsqueeze(-1)
                        host = torch.cat([probs, v], dim=-1).cpu().numpy()
                        # host shape is (n, …) under the variable-shape path
                        # and (batch_size, …) under the padded path; slice
                        # to the actual request count either way to drop
                        # any padded-zero rows before they reach Rust.
                        policies = np.ascontiguousarray(host[:n, :self._policy_len])
                        values = np.ascontiguousarray(host[:n, self._policy_len])
                        if _perf:
                            _t_d2h_done = time.perf_counter()

                        self._batcher.submit_inference_results(
                            request_ids,
                            policies,
                            values,
                        )
                        if _perf:
                            # submit_us closes the 2nd (return) PyO3 crossing so the
                            # 5 buckets sum to the full fetch→submit cycle. The fetch
                            # crossing already lives inside fetch_wait_us.
                            _t_submit_done = time.perf_counter()
                            _perf_log.info(
                                "inference_batch_timing",
                                batch_n=len(request_ids),
                                fetch_wait_us=(_t_fetched - _t_fetch_start) * 1e6,
                                h2d_us=(_t_h2d_done - _t_fetched) * 1e6,
                                forward_us=(_t_forward_done - _t_h2d_done) * 1e6,
                                d2h_scatter_us=(_t_d2h_done - _t_forward_done) * 1e6,
                                submit_us=(_t_submit_done - _t_d2h_done) * 1e6,
                                sync_cuda=_sync,
                                forward_count=self._forward_count + 1,
                            )
                    except Exception as exc:
                        # Explicitly signal failure to Rust waiters rather than returning dummy data
                        # or failing silently. This allows Rust to handle the error properly.
                        # Format kept stable for downstream tests / log parsers.
                        error_msg = f"Model inference failed: {exc}"
                        import traceback as _tb
                        # Surface the type + traceback in the log even when
                        # str(exc) is empty (e.g. cudagraph_trees AssertionError).
                        _perf_log.error(
                            "inference_forward_failed",
                            context="inference_server",
                            error_type=type(exc).__name__,
                            error=str(exc)[:300] or repr(exc)[:300],
                            tb=_tb.format_exc()[:1500],
                        )
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
