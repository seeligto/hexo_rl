"""Regression: load_state_dict_safe must not race with an in-flight forward pass."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.inference_server import InferenceServer

BOARD_CHANNELS = 8
BOARD_SIZE = 19


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_model(device: torch.device, seed: int = 0) -> HexTacToeNet:
    torch.manual_seed(seed)
    net = HexTacToeNet(
        board_size=BOARD_SIZE,
        in_channels=BOARD_CHANNELS,
        filters=32,
        res_blocks=1,
    ).to(device)
    net.eval()
    return net


def _random_state() -> np.ndarray:
    return np.random.randn(BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float16)


def test_load_state_dict_during_forward_is_safe(device: torch.device) -> None:
    """load_state_dict_safe blocks until any in-flight forward completes.

    After the swap the model parameters must byte-equal the new weights —
    no torn writes.
    """
    model_a = _make_model(device, seed=1)
    model_b = _make_model(device, seed=2)

    # Verify A and B actually differ so the test is not vacuous.
    params_b = {k: v.clone() for k, v in model_b.state_dict().items()}
    assert not all(
        torch.equal(model_a.state_dict()[k], params_b[k])
        for k in params_b
    ), "model_a and model_b must differ — fix seed or model init"

    cfg = {"selfplay": {"inference_batch_size": 4, "inference_max_wait_ms": 50.0}}
    server = InferenceServer(model_a, device, cfg)
    server.start()

    forward_started = threading.Event()
    forward_can_finish = threading.Event()
    errors: list[str] = []

    # Monkey-patch _weights_lock.acquire to signal when forward is waiting on it,
    # then stall it briefly so load_state_dict_safe races into the lock.
    original_run = server.run

    # Instead of patching internals, drive the race with timing:
    # kick off many concurrent infers so the lock is held continuously,
    # while a parallel thread calls load_state_dict_safe.
    n_infers = 20
    infer_results: list[tuple[np.ndarray, float]] = []
    infer_lock = threading.Lock()

    def _infer() -> None:
        try:
            policy, value = server.infer(_random_state())
            with infer_lock:
                infer_results.append((policy, value))
        except Exception as exc:
            with infer_lock:
                errors.append(f"infer failed: {exc}")

    def _swap() -> None:
        # Small sleep to let a forward start first.
        time.sleep(0.005)
        try:
            server.load_state_dict_safe(params_b)
        except Exception as exc:
            errors.append(f"load_state_dict_safe failed: {exc}")

    threads = [threading.Thread(target=_infer, daemon=True) for _ in range(n_infers)]
    swap_thread = threading.Thread(target=_swap, daemon=True)

    for t in threads:
        t.start()
    swap_thread.start()

    for t in threads:
        t.join(timeout=10.0)
    swap_thread.join(timeout=5.0)

    server.stop()
    server.join(timeout=3.0)

    assert errors == [], f"Race errors:\n" + "\n".join(errors)

    # After swap, model weights must byte-equal model_b.
    final_sd = server.model.state_dict()
    for k, expected in params_b.items():
        actual = final_sd[k]
        assert torch.equal(actual, expected), (
            f"param '{k}' torn after concurrent load_state_dict_safe — "
            f"max diff = {(actual.float() - expected.float()).abs().max():.6f}"
        )
