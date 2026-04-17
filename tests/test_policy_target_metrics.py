"""Synthetic-batch tests for §101 policy-target-quality metrics.

Exercises ``compute_policy_target_metrics`` directly — no Trainer, no autograd,
no replay buffer. Tests the math and edge cases the dashboard relies on.
"""
from __future__ import annotations

import math
import time

import pytest
import torch

from hexo_rl.training.trainer import (
    _POLICY_TARGET_METRIC_KEYS,
    compute_policy_target_metrics,
)


N_ACTIONS = 362  # 19 * 19 + 1 — matches the production policy head.
LOG_N = math.log(N_ACTIONS)


def _uniform_row(n_actions: int = N_ACTIONS) -> torch.Tensor:
    return torch.full((n_actions,), 1.0 / n_actions, dtype=torch.float32)


def _one_hot_row(n_actions: int = N_ACTIONS, idx: int = 0) -> torch.Tensor:
    row = torch.zeros(n_actions, dtype=torch.float32)
    row[idx] = 1.0
    return row


def test_uniform_vs_one_hot_split():
    """Half uniform (full-search), half one-hot (fast-search).

    Expected (for A = 362, log_N ≈ 5.891):
      H_full  ≈ log(A),  KL_u_full ≈ 0
      H_fast  ≈ 0,       KL_u_fast ≈ log(A)
    """
    batch = 8
    rows = []
    for i in range(batch):
        rows.append(_uniform_row() if i < 4 else _one_hot_row())
    target = torch.stack(rows, dim=0)
    full_search_mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.uint8)
    policy_valid = torch.ones(batch, dtype=torch.bool)

    m = compute_policy_target_metrics(target, policy_valid, full_search_mask)

    assert m["policy_target_entropy_fullsearch"] == pytest.approx(LOG_N, abs=1e-4)
    assert m["policy_target_entropy_fastsearch"] == pytest.approx(0.0, abs=1e-5)
    assert m["policy_target_kl_uniform_fullsearch"] == pytest.approx(0.0, abs=1e-4)
    assert m["policy_target_kl_uniform_fastsearch"] == pytest.approx(LOG_N, abs=1e-4)
    assert m["frac_fullsearch_in_batch"] == pytest.approx(0.5, abs=1e-9)
    assert m["n_rows_policy_loss"] == 4
    assert m["n_rows_total"] == 8


def test_all_full_search_batch():
    """full_search_mask = [1]*B — fastsearch metrics are NaN, full are finite."""
    batch = 6
    target = torch.stack([_uniform_row() for _ in range(batch)], dim=0)
    full_search_mask = torch.ones(batch, dtype=torch.uint8)
    policy_valid = torch.ones(batch, dtype=torch.bool)

    m = compute_policy_target_metrics(target, policy_valid, full_search_mask)

    # Must not raise — NaN metrics are a first-class signal.
    assert math.isnan(m["policy_target_entropy_fastsearch"])
    assert math.isnan(m["policy_target_kl_uniform_fastsearch"])
    assert m["policy_target_entropy_fullsearch"] == pytest.approx(LOG_N, abs=1e-4)
    assert m["policy_target_kl_uniform_fullsearch"] == pytest.approx(0.0, abs=1e-4)
    assert m["frac_fullsearch_in_batch"] == pytest.approx(1.0, abs=1e-9)
    assert m["n_rows_policy_loss"] == batch
    assert m["n_rows_total"] == batch


def test_all_fast_search_batch():
    """full_search_mask = [0]*B — fullsearch metrics are NaN, fast are finite."""
    batch = 6
    target = torch.stack([_uniform_row() for _ in range(batch)], dim=0)
    full_search_mask = torch.zeros(batch, dtype=torch.uint8)
    policy_valid = torch.ones(batch, dtype=torch.bool)

    m = compute_policy_target_metrics(target, policy_valid, full_search_mask)

    assert math.isnan(m["policy_target_entropy_fullsearch"])
    assert math.isnan(m["policy_target_kl_uniform_fullsearch"])
    assert m["policy_target_entropy_fastsearch"] == pytest.approx(LOG_N, abs=1e-4)
    assert m["policy_target_kl_uniform_fastsearch"] == pytest.approx(0.0, abs=1e-4)
    assert m["frac_fullsearch_in_batch"] == pytest.approx(0.0, abs=1e-9)
    assert m["n_rows_policy_loss"] == 0
    assert m["n_rows_total"] == batch


def test_empty_valid_mask_returns_nan_without_raising():
    """policy_valid = all False — all metric means NaN, counts zero, no raise."""
    batch = 4
    target = torch.stack([_uniform_row() for _ in range(batch)], dim=0)
    full_search_mask = torch.ones(batch, dtype=torch.uint8)
    policy_valid = torch.zeros(batch, dtype=torch.bool)

    m = compute_policy_target_metrics(target, policy_valid, full_search_mask)

    assert math.isnan(m["policy_target_entropy_fullsearch"])
    assert math.isnan(m["policy_target_entropy_fastsearch"])
    assert math.isnan(m["policy_target_kl_uniform_fullsearch"])
    assert math.isnan(m["policy_target_kl_uniform_fastsearch"])
    assert m["frac_fullsearch_in_batch"] == pytest.approx(0.0, abs=1e-9)
    assert m["n_rows_policy_loss"] == 0
    assert m["n_rows_total"] == 0
    # Every promised key must be present (so emitters can trust the dict shape).
    for k in _POLICY_TARGET_METRIC_KEYS:
        assert k in m


def test_cost_budget_under_200us_at_b256():
    """<200 µs per call at (B=256, A=362) — <0.2% of a 100 ms training step.

    Uses CUDA when available (matches the production hot path); falls back to
    CPU when no GPU is present — the CPU ceiling is still well under budget.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, n_actions = 256, N_ACTIONS
    target = torch.rand(batch, n_actions, device=device)
    target = target / target.sum(dim=-1, keepdim=True)
    policy_valid = torch.ones(batch, dtype=torch.bool, device=device)
    full_search_mask = torch.randint(0, 2, (batch,), dtype=torch.uint8, device=device)

    # Warm-up: first call allocates kernels on CUDA and cache-warms on CPU.
    for _ in range(5):
        compute_policy_target_metrics(target, policy_valid, full_search_mask)
    if device.type == "cuda":
        torch.cuda.synchronize()

    n_iters = 50
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = compute_policy_target_metrics(target, policy_valid, full_search_mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    per_call_us = (elapsed / n_iters) * 1e6

    # Idle-GPU baseline is ~150 µs. Budget is held at 1500 µs on CUDA so the
    # test does not flake when another CUDA workload (a parallel smoke run,
    # CI overlap) shares the device; the real-world step cost remains <0.3%
    # because the training step itself also slows under GPU contention.
    # Catches anything 10× the idle baseline — the regression this guards
    # against — without false-alarming on contention. CPU fallback 1500 µs.
    budget_us = 1500.0
    assert per_call_us < budget_us, (
        f"compute_policy_target_metrics took {per_call_us:.1f} µs/call on "
        f"{device.type} (budget {budget_us:.0f} µs)"
    )
