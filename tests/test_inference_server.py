"""
Tests for InferenceServer — batched GPU leaf evaluation.

Verifies:
  1. All requests receive valid results (correct shapes, finite values).
  2. Results arrive via batching: ceil(N*leaves / batch_size) forward calls.
  3. Server handles concurrent requests from multiple threads.
  4. Correctness of output shapes and probability normalization.
"""

from __future__ import annotations

import math
import sys
import threading
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.inference_server import InferenceServer

# ── Fixtures ──────────────────────────────────────────────────────────────────

BOARD_CHANNELS = 8
BOARD_SIZE     = 19
N_ACTIONS      = BOARD_SIZE * BOARD_SIZE + 1  # 362


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def model(device: torch.device) -> HexTacToeNet:
    net = HexTacToeNet(
        board_size=BOARD_SIZE,
        in_channels=BOARD_CHANNELS,
        filters=64,       # small for test speed
        res_blocks=2,
    ).to(device)
    net.eval()
    return net


def _random_state() -> np.ndarray:
    return np.random.randn(BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float16)


def _make_server(model: HexTacToeNet, device: torch.device, batch_size: int = 8) -> InferenceServer:
    cfg = {"selfplay": {"inference_batch_size": batch_size, "inference_max_wait_ms": 20.0}}
    return InferenceServer(model, device, cfg)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestInferenceServerSingleThread:
    """Sequential requests — just verify correctness, not batching behaviour."""

    def test_policy_shape_and_sums_to_one(self, model, device):
        server = _make_server(model, device, batch_size=4)
        server.start()
        try:
            state = _random_state()
            policy, value = server.infer(state)
            assert policy.shape == (N_ACTIONS,), f"policy shape: {policy.shape}"
            assert abs(policy.sum() - 1.0) < 1e-4, f"policy sum: {policy.sum()}"
        finally:
            server.stop()
            server.join(timeout=2.0)

    def test_value_in_range(self, model, device):
        server = _make_server(model, device, batch_size=4)
        server.start()
        try:
            state = _random_state()
            policy, value = server.infer(state)
            assert -1.0 <= value <= 1.0, f"value out of range: {value}"
        finally:
            server.stop()
            server.join(timeout=2.0)

    def test_policy_is_finite(self, model, device):
        server = _make_server(model, device, batch_size=4)
        server.start()
        try:
            state = _random_state()
            policy, value = server.infer(state)
            assert np.all(np.isfinite(policy)), "policy contains inf or nan"
            assert math.isfinite(value), f"value is not finite: {value}"
        finally:
            server.stop()
            server.join(timeout=2.0)


class TestInferenceServerBatching:
    """Verify forward-call count matches expected batching behaviour."""

    def test_n_requests_batched_into_ceil_n_div_batch(self, model, device):
        # N=16 requests, batch_size=8 → expect ceil(16/8) = 2 forward calls.
        # We fire requests concurrently so the server actually batches them.
        batch_size = 8
        n_requests = 16
        server = _make_server(model, device, batch_size=batch_size)
        server.start()

        results: List[Tuple[np.ndarray, float]] = [None] * n_requests  # type: ignore
        barrier = threading.Barrier(n_requests)

        def worker(idx: int) -> None:
            state = _random_state()
            barrier.wait()  # synchronise start so requests arrive together
            results[idx] = server.infer(state)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_requests)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        server.stop()
        server.join(timeout=2.0)

        # All results arrived.
        assert all(r is not None for r in results), "some requests did not get a result"

        # Forward calls: with perfect batching, exactly ceil(16/8) = 2.
        # Allow up to 2× overhead (timing jitter can split batches).
        expected = math.ceil(n_requests / batch_size)
        assert server.forward_count <= expected * 2, (
            f"too many forward calls: {server.forward_count} (expected ~{expected})"
        )
        assert server.forward_count >= 1, "no forward calls made"

    def test_all_results_valid_under_concurrency(self, model, device):
        n_requests = 24
        server = _make_server(model, device, batch_size=8)
        server.start()

        errors: List[str] = []
        lock = threading.Lock()

        def worker() -> None:
            state = _random_state()
            policy, value = server.infer(state)
            if policy.shape != (N_ACTIONS,):
                with lock:
                    errors.append(f"bad policy shape: {policy.shape}")
            if not np.all(np.isfinite(policy)):
                with lock:
                    errors.append("policy has non-finite values")
            if not (-1.0 <= value <= 1.0):
                with lock:
                    errors.append(f"value out of range: {value}")

        threads = [threading.Thread(target=worker) for _ in range(n_requests)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        server.stop()
        server.join(timeout=2.0)

        assert errors == [], f"Errors from concurrent workers:\n" + "\n".join(errors)

    def test_total_requests_counted_correctly(self, model, device):
        n_requests = 10
        server = _make_server(model, device, batch_size=4)
        server.start()
        for _ in range(n_requests):
            server.infer(_random_state())
        server.stop()
        server.join(timeout=2.0)
        assert server.total_requests == n_requests


class TestInferenceServerTrace:
    """torch.jit.trace path: parity with the untraced module + weight-swap propagation.

    Locks in the dispatch optimisation from project_dispatch_pyspy_2026-04-25.md.
    The traced ScriptModule must:
      1. produce numerically equivalent outputs to the untraced module,
      2. follow in-place weight mutation by load_state_dict_safe (parameters
         are shared, not copied),
      3. be cleanly disabled via selfplay.trace_inference=false in config.
    """

    def _server(
        self,
        model: HexTacToeNet,
        device: torch.device,
        *,
        trace: bool,
        batch_size: int = 4,
    ) -> InferenceServer:
        cfg = {
            "selfplay": {
                "inference_batch_size": batch_size,
                "inference_max_wait_ms": 20.0,
                "trace_inference": trace,
            }
        }
        return InferenceServer(model, device, cfg)

    def test_traced_matches_untraced(self, model, device):
        """Trace must produce the same policy / value as the untraced module."""
        np.random.seed(0)
        states = [_random_state() for _ in range(6)]

        s_off = self._server(model, device, trace=False)
        s_off.start()
        try:
            ref = [s_off.infer(s) for s in states]
        finally:
            s_off.stop()
            s_off.join(timeout=2.0)

        s_on = self._server(model, device, trace=True)
        assert s_on._traced_model is not None, "trace did not compile on the test model"
        s_on.start()
        try:
            traced = [s_on.infer(s) for s in states]
        finally:
            s_on.stop()
            s_on.join(timeout=2.0)

        for i, ((p_ref, v_ref), (p_tr, v_tr)) in enumerate(zip(ref, traced)):
            assert p_tr.shape == p_ref.shape, f"state {i}: policy shape mismatch"
            # fp16 path: drift up to a few ×1e-3 on policy probs; values
            # come from the same fp16 logits → same tolerance.
            max_p = float(np.abs(p_tr - p_ref).max())
            assert max_p < 5e-3, f"state {i}: policy diverged max={max_p}"
            assert abs(v_tr - v_ref) < 5e-3, (
                f"state {i}: value diverged tr={v_tr} ref={v_ref}"
            )

    def test_traced_follows_weight_swap(self, device):
        """load_state_dict_safe must mutate parameters in place; trace must see new weights."""
        net = HexTacToeNet(
            board_size=BOARD_SIZE, in_channels=BOARD_CHANNELS, filters=64, res_blocks=2,
        ).to(device)
        net.eval()

        server = self._server(net, device, trace=True)
        assert server._traced_model is not None
        server.start()
        try:
            np.random.seed(123)
            state = _random_state()
            p_before, v_before = server.infer(state)

            new_sd = {k: torch.randn_like(v) if v.dtype.is_floating_point else v
                      for k, v in net.state_dict().items()}
            server.load_state_dict_safe(new_sd)

            p_after, v_after = server.infer(state)
        finally:
            server.stop()
            server.join(timeout=2.0)

        diff_p = float(np.abs(p_after - p_before).max())
        diff_v = abs(v_after - v_before)
        assert diff_p > 1e-3, (
            f"traced model did not pick up weight swap: policy max diff {diff_p}"
        )
        assert diff_v > 1e-3 or diff_p > 1e-2, (
            f"traced model did not pick up weight swap: value diff {diff_v}"
        )

    def test_trace_disabled_via_config(self, model, device):
        server = self._server(model, device, trace=False)
        assert server._trace_inference is False
        assert server._traced_model is None
        # Smoke: still functional without the trace.
        server.start()
        try:
            policy, _ = server.infer(_random_state())
            assert policy.shape == (N_ACTIONS,)
        finally:
            server.stop()
            server.join(timeout=2.0)


class TestInferenceServerCompile:
    """torch.compile path on the InferenceServer (compile_retry_20260426).

    Locks in the Phase 3 padding + thread-init contract:
      1. config-time mutex with trace_inference,
      2. padded-shape forward returns correct shape and does not leak
         padded-zero rows into ``output[:n]``,
      3. weight swap propagates through the OptimizedModule wrapper.
    """

    def _server(
        self,
        model: HexTacToeNet,
        device: torch.device,
        *,
        compile_on: bool,
        mode: str = "default",
        dynamic: bool = True,
        trace_on: bool = False,
        batch_size: int = 8,
    ) -> InferenceServer:
        cfg = {
            "selfplay": {
                "inference_batch_size": batch_size,
                "inference_max_wait_ms": 20.0,
                "trace_inference": trace_on,
                "compile_inference": compile_on,
                "compile_inference_mode": mode,
                "compile_inference_dynamic": dynamic,
            }
        }
        return InferenceServer(model, device, cfg)

    def test_compile_and_trace_mutex(self, model, device):
        with pytest.raises(ValueError, match="mutually exclusive"):
            self._server(
                model, device,
                compile_on=True, trace_on=True,
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="torch.compile reduce-overhead requires CUDA",
    )
    def test_compile_inference_padding_correctness(self, device):
        """Padded forward at batch_n in {1, batch_size-1, batch_size}:
        output shapes must match request count, padded zero rows must
        not leak into the returned slice, and outputs must match an
        eager reference within fp16 + cudagraph tolerance.

        Subprocess isolation: torch.compile(mode="reduce-overhead") uses
        cudagraph_trees, which keeps state in C++ TLS keyed by thread.
        Running this test in the parent pytest process pollutes (and is
        polluted by) other tests' Dynamo / jit.trace state and triggers
        a cudagraph AssertionError on the dispatcher thread. The padding
        logic itself is correctness-checked here; we run it in a clean
        subprocess so the suite stays green and the assertion still
        catches a real regression.
        """
        import subprocess
        import json

        repo_root = Path(__file__).resolve().parent.parent
        py = sys.executable
        helper = """
import json, sys, threading
sys.path.insert(0, %r)
import numpy as np
import torch
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.inference_server import InferenceServer

dev = torch.device('cuda')
net = HexTacToeNet(board_size=19, in_channels=8, filters=64, res_blocks=2).to(dev)
net.eval()
batch_size = 8
np.random.seed(2026)
states = [np.random.randn(8, 19, 19).astype(np.float16) for _ in range(batch_size)]

# Eager reference forward (raw model, no compile / trace).
x_ref = torch.from_numpy(np.stack(states).astype(np.float32)).to(dev)
with torch.no_grad(), torch.autocast(device_type='cuda'):
    lp_ref, v_ref, _ = net(x_ref)
    p_ref = lp_ref.float().exp()
    p_ref = (p_ref / p_ref.sum(dim=-1, keepdim=True)).cpu().numpy()
    v_ref = v_ref.squeeze(-1).float().cpu().numpy()

cfg = {'selfplay': {'inference_batch_size': batch_size, 'inference_max_wait_ms': 20.0,
                    'trace_inference': False, 'compile_inference': True,
                    'compile_inference_mode': 'reduce-overhead',
                    'compile_inference_dynamic': False}}
srv = InferenceServer(net, dev, cfg)
assert srv._compile_inference, 'compile fallback active'
assert srv._padding_active(), 'padding gate did not engage'
srv.start()
results = {}
try:
    for n in (1, batch_size - 1, batch_size):
        outs = [None] * n
        barrier = threading.Barrier(n)
        def worker(i, s):
            barrier.wait()
            outs[i] = srv.infer(s)
        threads = [threading.Thread(target=worker, args=(i, states[i])) for i in range(n)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=15.0)
        assert all(o is not None for o in outs), f'n={n}: missing results'
        rows = []
        for i, (pol, val) in enumerate(outs):
            assert pol.shape == (362,), f'n={n} i={i} shape {pol.shape}'
            assert np.all(np.isfinite(pol)), f'n={n} i={i} non-finite policy'
            max_p = float(np.abs(pol - p_ref[i]).max())
            val_diff = float(abs(val - float(v_ref[i])))
            rows.append({'i': i, 'max_p_diff': max_p, 'val_diff': val_diff})
        results[n] = rows
finally:
    srv.stop(); srv.join(timeout=5.0)
print('JSON_OUT' + json.dumps(results))
""" % (str(repo_root),)

        proc = subprocess.run(
            [py, "-c", helper],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=120.0,
        )
        # The subprocess may exit -11 (SIGSEGV) during interpreter teardown:
        # PT 2.11 + CUDA 13 occasionally crashes during cudagraph_trees
        # finalisation when reduce-overhead state is torn down. The test
        # passes if the JSON marker landed before teardown — that means
        # the padded forward + slice produced the expected outputs.
        marker = "JSON_OUT"
        idx = proc.stdout.find(marker)
        if idx < 0:
            raise AssertionError(
                f"no JSON marker in subprocess stdout (rc={proc.returncode}):\n"
                f"STDOUT: {proc.stdout[-2000:]}\n"
                f"STDERR: {proc.stderr[-2000:]}"
            )
        results = json.loads(proc.stdout[idx + len(marker):].strip())
        # Tolerance 1e-2 — fp16 + cudagraph drift slightly larger than
        # trace path's 5e-3 due to fused-kernel rounding.
        for n, rows in results.items():
            assert len(rows) == int(n), f"n={n}: expected {n} rows, got {len(rows)}"
            for r in rows:
                assert r["max_p_diff"] < 1e-2, (
                    f"n={n} i={r['i']} policy diverged {r['max_p_diff']:.4e}"
                )
                assert r["val_diff"] < 1e-2, (
                    f"n={n} i={r['i']} value diverged {r['val_diff']:.4e}"
                )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="torch.compile requires CUDA",
    )
    def test_compile_inference_weight_swap_propagates(self, device):
        """compile, run forward A, load_state_dict_safe new weights,
        run forward B; outputs differ ≥ 1e-3 on at least one element.
        Tests propagation through OptimizedModule.
        """
        net = HexTacToeNet(
            board_size=BOARD_SIZE, in_channels=BOARD_CHANNELS,
            filters=64, res_blocks=2,
        ).to(device)
        net.eval()

        # default mode is sufficient: it wraps in OptimizedModule and exercises
        # the same load_state_dict propagation path as reduce-overhead, without
        # the CUDA-graph TLS constraint that requires the dispatcher-thread
        # warmup. (Reduce-overhead is covered by the padding test.)
        server = self._server(
            net, device,
            compile_on=True, mode="default", dynamic=True,
            batch_size=4,
        )
        assert server._compile_inference is True, (
            "compile failed at init — test is meaningless"
        )
        server.start()
        try:
            np.random.seed(7)
            state = _random_state()
            p_before, v_before = server.infer(state)

            new_sd = {
                k: torch.randn_like(v) if v.dtype.is_floating_point else v
                for k, v in net.state_dict().items()
            }
            server.load_state_dict_safe(new_sd)

            p_after, v_after = server.infer(state)
        finally:
            server.stop()
            server.join(timeout=5.0)

        diff_p = float(np.abs(p_after - p_before).max())
        diff_v = abs(v_after - v_before)
        assert diff_p > 1e-3 or diff_v > 1e-3, (
            f"OptimizedModule did not propagate weight swap: "
            f"policy max diff {diff_p}, value diff {diff_v}"
        )


class TestInferenceServerFailureHandling:
    def test_batch_prep_error_unblocks_workers(self, device):
        """Regression C-003: batch-prep (np.ascontiguousarray) error must unblock workers.

        Before fix, prep ran outside inner try so workers blocked forever on failure.
        """
        import unittest.mock as mock

        class IdentityNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                n = x.shape[0]
                pol = torch.ones(n, N_ACTIONS, device=x.device) / N_ACTIONS
                val = torch.zeros(n, 1, device=x.device)
                return pol.log(), val, val

        model = IdentityNet().to(device)
        model.eval()
        server = InferenceServer(
            model,
            device,
            {"selfplay": {"inference_batch_size": 4, "inference_max_wait_ms": 20.0}},
        )
        server.start()
        try:
            state = _random_state()
            done = threading.Event()
            error_caught = []

            def _call() -> None:
                try:
                    server.infer(state)
                except Exception as e:
                    error_caught.append(str(e))
                finally:
                    done.set()

            import hexo_rl.selfplay.inference_server as _is_mod
            with mock.patch.object(_is_mod.np, "ascontiguousarray", side_effect=ValueError("bad array")):
                t = threading.Thread(target=_call, daemon=True)
                t.start()
                hung = not done.wait(5.0)

            t.join(timeout=2.0)
            assert not hung, "server.infer() hung — workers not unblocked on batch-prep error"
            assert len(error_caught) == 1, f"expected one error, got {error_caught}"
        finally:
            server.stop()
            server.join(timeout=2.0)

    def test_infer_returns_on_model_forward_exception(self, device):
        """Regression: inference failures must not deadlock callers waiting on req.event."""
        class FailingNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Ensure the module can be moved to device.
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x: torch.Tensor):
                raise RuntimeError("boom")

        model = FailingNet().to(device)
        model.eval()

        # Small batch size to keep the test deterministic/tight.
        server = InferenceServer(
            model,
            device,
            {"selfplay": {"inference_batch_size": 4, "inference_max_wait_ms": 20.0}},
        )
        server.start()
        try:
            state = _random_state()
            done = threading.Event()
            error_caught = []

            def _call() -> None:
                try:
                    server.infer(state)
                except ValueError as e:
                    error_caught.append(str(e))
                finally:
                    done.set()

            t = threading.Thread(target=_call, daemon=True)
            t.start()

            assert done.wait(5.0), "server.infer() hung waiting for results"
            assert len(error_caught) == 1
            assert "Model inference failed: boom" in error_caught[0]
        finally:
            server.stop()
            server.join(timeout=2.0)
