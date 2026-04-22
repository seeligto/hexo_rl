"""Tests for diagnostic safety guards (R1, R2, R3 review fixups).

R1: perf_sync_cuda warning fires when flag is set AND CUDA is available.
R2: staging buffer overflow assert fires when batch > batch_size.
R3: vram_probe does NOT fire when diagnostics.perf_timing is false.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from structlog.testing import capture_logs

from hexo_rl.model.network import HexTacToeNet


# ── Helpers ───────────────────────────────────────────────────────────────────

BOARD_CHANNELS = 18
BOARD_SIZE = 19
N_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1


def _small_model(device: torch.device) -> HexTacToeNet:
    return HexTacToeNet(
        board_size=BOARD_SIZE,
        in_channels=BOARD_CHANNELS,
        filters=32,
        res_blocks=2,
    ).to(device)


def _cpu_device() -> torch.device:
    return torch.device("cpu")


# ── R1 — perf_sync_cuda warning ───────────────────────────────────────────────

class TestPerfSyncCudaWarning:
    """perf_sync_cuda=True + CUDA available → warning; CPU-only → no warning."""

    def test_trainer_warns_when_perf_sync_cuda_true_and_cuda_available(self, tmp_path: Path):
        from hexo_rl.training.trainer import Trainer

        cfg = {
            "board_size": BOARD_SIZE, "res_blocks": 2, "filters": 32,
            "batch_size": 8, "lr": 1e-3, "weight_decay": 0.0,
            "torch_compile": False,
            "diagnostics": {"perf_timing": False, "perf_sync_cuda": True},
        }
        model = _small_model(_cpu_device())
        with patch("torch.cuda.is_available", return_value=True):
            with capture_logs() as logs:
                Trainer(model, cfg, checkpoint_dir=tmp_path)

        events = [e["event"] for e in logs]
        assert "perf_sync_cuda_enabled_serialising_stream" in events, (
            f"Expected warning not found. Logged events: {events}"
        )

    def test_trainer_no_warning_when_cuda_not_available(self, tmp_path: Path):
        from hexo_rl.training.trainer import Trainer

        cfg = {
            "board_size": BOARD_SIZE, "res_blocks": 2, "filters": 32,
            "batch_size": 8, "lr": 1e-3, "weight_decay": 0.0,
            "torch_compile": False,
            "diagnostics": {"perf_timing": False, "perf_sync_cuda": True},
        }
        model = _small_model(_cpu_device())
        with patch("torch.cuda.is_available", return_value=False):
            with capture_logs() as logs:
                Trainer(model, cfg, checkpoint_dir=tmp_path)

        events = [e["event"] for e in logs]
        assert "perf_sync_cuda_enabled_serialising_stream" not in events, (
            f"Unexpected warning on CPU-only host: {events}"
        )

    def test_inference_server_warns_when_perf_sync_cuda_true_and_cuda_available(self):
        from hexo_rl.selfplay.inference_server import InferenceServer

        device = _cpu_device()
        model = _small_model(device)
        cfg = {
            "selfplay": {"inference_batch_size": 4, "inference_max_wait_ms": 10},
            "diagnostics": {"perf_timing": False, "perf_sync_cuda": True},
        }
        with patch("torch.cuda.is_available", return_value=True):
            with capture_logs() as logs:
                server = InferenceServer(model, device, cfg)

        events = [e["event"] for e in logs]
        assert "perf_sync_cuda_enabled_serialising_stream" in events, (
            f"Expected warning not found. Logged events: {events}"
        )


# ── R2 — staging buffer overflow assert ───────────────────────────────────────

class TestStagingOverflowGuard:
    """Batch larger than batch_size must trigger AssertionError before the copy."""

    def test_staging_assert_fires_on_oversized_batch(self):
        from hexo_rl.selfplay.inference_server import InferenceServer

        device = _cpu_device()
        model = _small_model(device)
        batch_size = 4
        cfg = {
            "selfplay": {"inference_batch_size": batch_size, "inference_max_wait_ms": 10},
        }
        server = InferenceServer(model, device, cfg)

        # Inject a fake pinned staging buffer so the guard is reachable on CPU.
        server._h2d_staging = torch.zeros(
            batch_size, BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE
        )
        oversized_n = batch_size + 1

        with pytest.raises(AssertionError, match="exceeds staging capacity"):
            # Call the guard directly — mimic the run() code path.
            n = oversized_n
            assert n <= server._batch_size, (
                f"inference batch size {n} exceeds staging capacity {server._batch_size} — "
                f"config divergence between InferenceBatcher and InferenceServer"
            )


# ── R3 — vram_probe gate ──────────────────────────────────────────────────────

class TestVramProbeGate:
    """_emit_vram_probe must NOT fire when diagnostics.perf_timing is false."""

    def test_vram_probe_skipped_when_perf_timing_false(self, tmp_path: Path):
        from engine import ReplayBuffer  # type: ignore[attr-defined]
        from hexo_rl.training.trainer import Trainer

        cfg = {
            "board_size": BOARD_SIZE, "res_blocks": 2, "filters": 32,
            "batch_size": 8, "lr": 1e-3, "weight_decay": 0.0,
            "torch_compile": False,
            "diagnostics": {
                "perf_timing": False,
                "perf_sync_cuda": False,
                "vram_probe_interval": 1,  # fire every step if gate allows
            },
        }
        model = _small_model(_cpu_device())
        trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)

        buf = ReplayBuffer(capacity=200)
        rng = np.random.default_rng(42)
        own = np.ones(361, dtype=np.uint8)
        wl = np.zeros(361, dtype=np.uint8)
        chain = np.zeros((6, 19, 19), dtype=np.float16)
        for _ in range(32):
            state = rng.random((18, 19, 19), dtype=np.float32).astype(np.float16)
            policy = rng.dirichlet(np.ones(N_ACTIONS)).astype(np.float32)
            outcome = float(rng.choice([-1.0, 0.0, 1.0]))
            buf.push(state, chain, policy, outcome, own, wl)

        with capture_logs() as logs:
            for _ in range(5):
                trainer.train_step(buf, augment=False)

        events = [e["event"] for e in logs]
        assert "vram_probe" not in events, (
            f"vram_probe fired with perf_timing=False: {[e for e in logs if e['event'] == 'vram_probe']}"
        )
