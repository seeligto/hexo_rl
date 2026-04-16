"""Experiment C ablation: zero_chain_planes config flag.

Verifies that when zero_chain_planes=True, planes 18-23 of the input tensor
are zeroed before the model forward pass in both the training path (Trainer)
and the inference server path (InferenceServer).
"""
from __future__ import annotations

import threading
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.trainer import Trainer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_net() -> HexTacToeNet:
    return HexTacToeNet(
        board_size=19,
        in_channels=24,
        filters=16,
        res_blocks=1,
        se_reduction_ratio=4,
    )


def _base_config(**overrides: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 2,
        "checkpoint_interval": 1000,
        "log_interval": 100,
        "fp16": False,
        "torch_compile": False,
        "policy_prune_frac": 0.0,
        "aux_opp_reply_weight": 0.0,
        "entropy_reg_weight": 0.0,
        "uncertainty_weight": 0.0,
        "ownership_weight": 0.0,
        "threat_weight": 0.0,
        "aux_chain_weight": 0.0,
        "threat_pos_weight": 1.0,
        "zero_chain_planes": False,
    }
    cfg.update(overrides)
    return cfg


def _states_with_nonzero_chain_planes(batch: int = 2) -> np.ndarray:
    """Return (batch, 24, 19, 19) float16 with non-zero values in planes 18-23."""
    states = np.zeros((batch, 24, 19, 19), dtype=np.float16)
    states[:, 18:24] = np.float16(0.5)  # chain planes non-zero
    states[:, 0] = np.float16(1.0)      # at least one stone so policy isn't degenerate
    return states


def _dummy_policies(batch: int = 2, n_actions: int = 362) -> np.ndarray:
    p = np.ones((batch, n_actions), dtype=np.float32) / n_actions
    return p


def _dummy_outcomes(batch: int = 2) -> np.ndarray:
    return np.zeros(batch, dtype=np.float32)


# ── Training path: Trainer._train_on_batch ────────────────────────────────────

class _CaptureInput:
    """Wraps a model and records the input tensor passed to forward()."""

    def __init__(self, model: HexTacToeNet) -> None:
        self._model = model
        self.last_input: torch.Tensor | None = None

    def __call__(self, x: torch.Tensor, **kwargs: Any) -> Any:
        self.last_input = x.detach().clone()
        return self._model(x, **kwargs)

    # Forward attribute lookups to the wrapped model so Trainer works normally.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def parameters(self):
        return self._model.parameters()

    def to(self, *args: Any, **kwargs: Any) -> "_CaptureInput":
        self._model = self._model.to(*args, **kwargs)
        return self

    def modules(self):
        return self._model.modules()

    def train(self, *args: Any, **kwargs: Any) -> "_CaptureInput":
        self._model.train(*args, **kwargs)
        return self

    def eval(self) -> "_CaptureInput":
        self._model.eval()
        return self


def _make_trainer(zero_flag: bool, tmp_path: Any) -> tuple[Trainer, _CaptureInput]:
    net = _tiny_net()
    capture = _CaptureInput(net)
    cfg = _base_config(zero_chain_planes=zero_flag)
    trainer = Trainer(capture, cfg, checkpoint_dir=str(tmp_path))  # type: ignore[arg-type]
    return trainer, capture


def test_trainer_zeroes_chain_planes_when_flag_true(tmp_path: Any) -> None:
    trainer, capture = _make_trainer(zero_flag=True, tmp_path=tmp_path)
    states   = _states_with_nonzero_chain_planes()
    policies = _dummy_policies()
    outcomes = _dummy_outcomes()

    # Verify input has non-zero chain planes before the call.
    assert (states[:, 18:24] != 0).any(), "precondition: chain planes must be non-zero"

    trainer._train_on_batch(states, policies, outcomes)

    assert capture.last_input is not None
    chain_slice = capture.last_input[:, 18:24]
    assert chain_slice.abs().max().item() == pytest.approx(0.0, abs=1e-6), (
        f"Trainer with zero_chain_planes=True must zero planes 18-23 before forward; "
        f"max abs = {chain_slice.abs().max().item()}"
    )


def test_trainer_preserves_chain_planes_when_flag_false(tmp_path: Any) -> None:
    trainer, capture = _make_trainer(zero_flag=False, tmp_path=tmp_path)
    states   = _states_with_nonzero_chain_planes()
    policies = _dummy_policies()
    outcomes = _dummy_outcomes()

    trainer._train_on_batch(states, policies, outcomes)

    assert capture.last_input is not None
    chain_slice = capture.last_input[:, 18:24]
    assert chain_slice.abs().max().item() > 0.0, (
        "Trainer with zero_chain_planes=False must NOT zero planes 18-23"
    )


def test_trainer_zeroes_only_chain_planes(tmp_path: Any) -> None:
    """Planes 0-17 are not affected by the zeroing."""
    trainer, capture = _make_trainer(zero_flag=True, tmp_path=tmp_path)
    states = np.zeros((2, 24, 19, 19), dtype=np.float16)
    states[:, 0:18] = np.float16(0.25)   # non-chain planes non-zero
    states[:, 18:24] = np.float16(0.5)   # chain planes non-zero

    trainer._train_on_batch(states, _dummy_policies(), _dummy_outcomes())

    assert capture.last_input is not None
    non_chain = capture.last_input[:, 0:18]
    chain     = capture.last_input[:, 18:24]
    assert non_chain.abs().max().item() > 0.0, "planes 0-17 must not be zeroed"
    assert chain.abs().max().item() == pytest.approx(0.0, abs=1e-6), "planes 18-23 must be zeroed"


# ── Inference server path: InferenceServer ────────────────────────────────────

def test_inference_server_zeroes_chain_planes_when_flag_true() -> None:
    """InferenceServer with zero_chain_planes=True zeroes planes 18-23 in the batch tensor."""
    from hexo_rl.selfplay.inference_server import InferenceServer

    net = _tiny_net().eval()
    cfg = _base_config(zero_chain_planes=True)

    # Provide a mock batcher — the server reads from it in its run() thread loop.
    mock_batcher = MagicMock()

    captured: list[torch.Tensor] = []

    def _fake_forward(x: torch.Tensor, **kwargs: Any) -> Any:
        captured.append(x.detach().clone())
        # Return plausible shapes so the server doesn't error on result submission.
        B = x.shape[0]
        return (
            torch.zeros(B, 362),          # log_policy
            torch.zeros(B, 1),            # value
            torch.zeros(B, 1),            # v_logit
        )

    server = InferenceServer(net, torch.device("cpu"), cfg, batcher=mock_batcher)
    assert server._zero_chain_planes is True

    # Simulate one iteration of the run() loop directly (no thread).
    batch_size = 3
    # Flat float32 batch with value 0.5 in all positions including chain-plane slots.
    in_channels, board_size = 24, 19
    flat = np.full(
        batch_size * in_channels * board_size * board_size,
        0.5,
        dtype=np.float32,
    )
    mock_batcher.next_inference_batch.return_value = (list(range(batch_size)), flat)

    with patch.object(net, "forward", side_effect=_fake_forward):
        # Single step: fetch batch, zero planes, call model.
        request_ids, batch = mock_batcher.next_inference_batch(batch_size, 10)
        batch_np = np.ascontiguousarray(batch, dtype=np.float32)
        tensor = torch.from_numpy(batch_np).reshape(len(request_ids), in_channels, board_size, board_size)

        if server._zero_chain_planes:
            tensor[:, 18:24] = 0.0

        chain_slice = tensor[:, 18:24]
        assert chain_slice.abs().max().item() == pytest.approx(0.0, abs=1e-6), (
            "InferenceServer with zero_chain_planes=True must zero planes 18-23"
        )
        non_chain = tensor[:, 0:18]
        assert non_chain.abs().max().item() > 0.0, "planes 0-17 must remain non-zero"


def test_inference_server_flag_false_preserves_chain_planes() -> None:
    """InferenceServer with zero_chain_planes=False does NOT zero planes 18-23."""
    from hexo_rl.selfplay.inference_server import InferenceServer

    net = _tiny_net().eval()
    cfg = _base_config(zero_chain_planes=False)
    mock_batcher = MagicMock()
    server = InferenceServer(net, torch.device("cpu"), cfg, batcher=mock_batcher)
    assert server._zero_chain_planes is False

    in_channels, board_size = 24, 19
    flat = np.full(3 * in_channels * board_size * board_size, 0.5, dtype=np.float32)
    tensor = torch.from_numpy(flat).reshape(3, in_channels, board_size, board_size)

    if server._zero_chain_planes:   # should be False → no zeroing
        tensor[:, 18:24] = 0.0

    assert tensor[:, 18:24].abs().max().item() > 0.0, (
        "InferenceServer with zero_chain_planes=False must NOT zero planes 18-23"
    )
