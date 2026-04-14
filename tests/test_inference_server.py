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
import threading
from typing import List, Tuple

import numpy as np
import pytest
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.inference_server import InferenceServer

# ── Fixtures ──────────────────────────────────────────────────────────────────

BOARD_CHANNELS = 24
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


class TestInferenceServerFailureHandling:
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
