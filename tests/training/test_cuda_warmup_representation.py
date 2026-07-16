"""S7 F2 — cuda_warmup must be representation-aware (graph vs grid).

Closes S7 Part-1 Finding F2 (reports/probes/gnn_integration/S7_smoke_gate.md):
`hexo_rl/training/lifecycle.py::cuda_warmup`, called unconditionally from
`hexo_rl/training/loop.py::run_training_loop`, built a dense
`(1, C, board_size, board_size)` dummy tensor and called `inf_model(_dummy)`
— `GnnNet` implements only `forward_batch(wire)` (graph input), never a bare
`forward()`, so `nn.Module`'s default `forward` raised `NotImplementedError`
before self-play ever started. No prior GNN smoke exercised this codepath
(every WP-1..WP-5b test drives `Trainer.train_step()` /
`InferenceServer._run_graph_loop` directly, never `scripts/train.py`'s full
`run_training_loop`).

CUDA-gated (`cuda_warmup` is a device.type == "cuda" no-op on CPU by design
— nothing to warm up off-GPU).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from hexo_rl.model.gnn_net import GnnNet
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.lifecycle import build_inference_model, cuda_warmup
from hexo_rl.training.trainer import Trainer

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda_warmup is a no-op off CUDA")


def _make_gnn_trainer() -> Trainer:
    device = torch.device("cpu")
    model = GnnNet()  # probe-284k ctor defaults — matches GNN_CONFIG_DEFAULTS fallback
    cfg: dict = {
        "encoding": "gnn_axis_v1",
        "n_value_bins": 65,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 4,
        "checkpoint_interval": 1000,
        "log_interval": 1000,
        "fp16": False,
        "grad_clip": 1.0,
        "value_head_type": "dist65",
    }
    return Trainer(model, cfg, checkpoint_dir="/tmp/hexo_test_s7f2_gnn_ckpts", device=device)


def _make_dense_trainer() -> Trainer:
    device = torch.device("cpu")
    model = HexTacToeNet(board_size=19, res_blocks=1, filters=8, in_channels=8)
    cfg: dict = {
        "encoding": "v6",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 4,
        "checkpoint_interval": 1000,
        "log_interval": 1000,
        "fp16": False,
        "grad_clip": 1.0,
        "res_blocks": 1,
        "filters": 8,
        "in_channels": 8,
    }
    return Trainer(model, cfg, checkpoint_dir="/tmp/hexo_test_s7f2_dense_ckpts", device=device)


@_CUDA
def test_cuda_warmup_graph_does_not_raise_and_touches_forward_batch():
    """S7 F2 regression guard: GnnNet has no bare forward() — cuda_warmup's
    graph branch must call forward_batch instead, and must not raise
    NotImplementedError (the exact S7 Part-1 crash)."""
    device = torch.device("cuda")
    trainer = _make_gnn_trainer()
    inf_model, arch = build_inference_model(trainer, device)
    assert arch.spec.representation == "graph"

    with patch.object(inf_model, "forward_batch", wraps=inf_model.forward_batch) as spy:
        cuda_warmup(inf_model, device, arch.board_size, spec=arch.spec)
    spy.assert_called_once()


@_CUDA
def test_cuda_warmup_graph_without_spec_kwarg_falls_back_to_dense_and_raises():
    """Sanity check on the OLD failure mode: omitting `spec` (or passing a
    non-graph default) on a GnnNet must still hit the dense branch and raise
    NotImplementedError — proves the graph branch is gated on `spec`, not a
    silent try/except that would mask a real config-vs-model mismatch."""
    device = torch.device("cuda")
    trainer = _make_gnn_trainer()
    inf_model, arch = build_inference_model(trainer, device)
    with pytest.raises(NotImplementedError):
        cuda_warmup(inf_model, device, arch.board_size)  # spec omitted -> default "grid"


@_CUDA
def test_cuda_warmup_dense_untouched_byte_identical():
    """Dense (representation='grid') path is byte-identical to pre-S7: calls
    the model's bare forward (via __call__), never forward_batch (which a
    HexTacToeNet doesn't even implement — an AttributeError there would fail
    this test), no raise."""
    device = torch.device("cuda")
    trainer = _make_dense_trainer()
    inf_model, arch = build_inference_model(trainer, device)
    assert arch.spec.representation == "grid"
    assert not hasattr(inf_model, "forward_batch")

    with patch.object(inf_model, "forward", wraps=inf_model.forward) as spy:
        cuda_warmup(inf_model, device, arch.board_size, spec=arch.spec)
    spy.assert_called_once()


@_CUDA
def test_cuda_warmup_dense_spec_none_still_works_pre_s7_call_signature():
    """Every pre-S7 caller/test passed no `spec` kwarg at all — must remain
    valid (grid default) so this is a strictly additive signature change."""
    device = torch.device("cuda")
    trainer = _make_dense_trainer()
    inf_model, arch = build_inference_model(trainer, device)
    cuda_warmup(inf_model, device, arch.board_size)  # no raise, no spec kwarg
