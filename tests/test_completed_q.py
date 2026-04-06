"""Tests for Gumbel completed Q-values policy targets.

Tests the Python reference implementation (compute_improved_policy) and
the KL divergence policy loss function.
"""

import numpy as np
import pytest
import torch
from torch.amp import autocast

from hexo_rl.selfplay.completed_q import compute_improved_policy
from hexo_rl.training.losses import compute_kl_policy_loss


# ── Helper ────────────────────────────────────────────────────────────────────

def make_uniform_log_prior(n_actions: int, n_legal: int) -> tuple[np.ndarray, np.ndarray]:
    """Create uniform log-prior and legal mask for n_legal actions."""
    legal_mask = np.zeros(n_actions, dtype=bool)
    legal_mask[:n_legal] = True
    log_prior = np.full(n_actions, -30.0)  # ~zero probability for illegal
    log_prior[:n_legal] = -np.log(n_legal)  # uniform over legal
    return log_prior, legal_mask


# ── Tests: compute_improved_policy ────────────────────────────────────────────

def test_completed_q_basic():
    """High-Q actions get more mass in improved policy."""
    n_actions = 20
    n_legal = 10
    log_prior, legal_mask = make_uniform_log_prior(n_actions, n_legal)

    # 3 visited actions with distinct Q-values
    visit_counts = np.zeros(n_actions, dtype=np.float32)
    visit_counts[0] = 30  # high Q
    visit_counts[1] = 15  # medium Q
    visit_counts[2] = 5   # low Q
    q_values = {0: 0.8, 1: 0.0, 2: -0.5}

    # Disable pruning to test raw Q-value ordering
    result = compute_improved_policy(
        log_prior, visit_counts, q_values, v_hat=0.1, legal_mask=legal_mask,
        prune_frac=0.0,
    )

    assert result[0] > result[1], "Highest Q action should have most mass"
    assert result[1] > result[2], "Medium Q > low Q"
    assert result[0] > 1.0 / n_legal, "High Q should exceed uniform"


def test_completed_q_unvisited_get_vmix():
    """Unvisited legal actions get v_mix, not zero."""
    n_actions = 20
    n_legal = 10
    log_prior, legal_mask = make_uniform_log_prior(n_actions, n_legal)

    visit_counts = np.zeros(n_actions, dtype=np.float32)
    visit_counts[0] = 50
    q_values = {0: 0.3}

    result = compute_improved_policy(
        log_prior, visit_counts, q_values, v_hat=0.2, legal_mask=legal_mask,
    )

    # Unvisited legal actions (1-9) should have non-zero probability
    for a in range(1, n_legal):
        assert result[a] > 0.0, f"Unvisited legal action {a} should have probability > 0"


def test_completed_q_illegal_masked():
    """Illegal actions have zero probability."""
    n_actions = 20
    n_legal = 5
    log_prior, legal_mask = make_uniform_log_prior(n_actions, n_legal)

    visit_counts = np.zeros(n_actions, dtype=np.float32)
    visit_counts[0] = 10
    q_values = {0: 0.5}

    result = compute_improved_policy(
        log_prior, visit_counts, q_values, v_hat=0.0, legal_mask=legal_mask,
    )

    for a in range(n_legal, n_actions):
        assert result[a] == 0.0, f"Illegal action {a} must have zero probability"


def test_completed_q_sums_to_one():
    """Improved policy sums to 1.0 within float tolerance."""
    n_actions = 362
    n_legal = 50
    log_prior, legal_mask = make_uniform_log_prior(n_actions, n_legal)

    visit_counts = np.zeros(n_actions, dtype=np.float32)
    for a in range(8):
        visit_counts[a] = np.random.randint(1, 100)
    q_values = {a: np.random.uniform(-1, 1) for a in range(8)}

    result = compute_improved_policy(
        log_prior, visit_counts, q_values, v_hat=0.1, legal_mask=legal_mask,
    )

    assert abs(result.sum() - 1.0) < 1e-5, f"Sum = {result.sum()}, expected ~1.0"


def test_completed_q_no_visits():
    """When all N=0, improved policy equals prior."""
    n_actions = 20
    n_legal = 10
    log_prior, legal_mask = make_uniform_log_prior(n_actions, n_legal)

    visit_counts = np.zeros(n_actions, dtype=np.float32)
    q_values: dict[int, float] = {}

    result = compute_improved_policy(
        log_prior, visit_counts, q_values, v_hat=0.0, legal_mask=legal_mask,
    )

    # Should be uniform over legal actions (since prior is uniform)
    expected = 1.0 / n_legal
    for a in range(n_legal):
        assert abs(result[a] - expected) < 1e-5, (
            f"Action {a}: got {result[a]}, expected {expected}"
        )


def test_completed_q_all_visited():
    """When all legal actions are visited, v_mix is not used."""
    n_actions = 10
    n_legal = 5
    log_prior, legal_mask = make_uniform_log_prior(n_actions, n_legal)

    # All legal actions visited
    visit_counts = np.zeros(n_actions, dtype=np.float32)
    q_values = {}
    for a in range(n_legal):
        visit_counts[a] = 10 + a * 5
        q_values[a] = 0.1 * (a - 2)  # range: -0.2 to 0.2

    result = compute_improved_policy(
        log_prior, visit_counts, q_values, v_hat=0.0, legal_mask=legal_mask,
    )

    assert abs(result.sum() - 1.0) < 1e-5
    # Highest Q should get most mass
    assert result[4] > result[0], "Highest Q action should dominate"


def test_completed_q_fast_game_regime():
    """At 50 sims with ~5-10 visited out of ~100 legal, improved policy differs from prior."""
    n_actions = 362
    n_legal = 100
    log_prior, legal_mask = make_uniform_log_prior(n_actions, n_legal)

    # Simulate 50 sims: ~7 actions visited
    visit_counts = np.zeros(n_actions, dtype=np.float32)
    visited = [0, 3, 7, 12, 25, 40, 55]
    for i, a in enumerate(visited):
        visit_counts[a] = max(1, 50 // len(visited) - i)
    q_values = {
        0: 0.6, 3: 0.3, 7: 0.1, 12: -0.1, 25: -0.3, 40: 0.2, 55: -0.5,
    }

    result = compute_improved_policy(
        log_prior, visit_counts, q_values, v_hat=0.15, legal_mask=legal_mask,
    )

    uniform = 1.0 / n_legal
    # Best action should have substantially more than uniform
    assert result[0] > 2 * uniform, (
        f"Best Q action got {result[0]}, expected > {2 * uniform}"
    )
    # Distribution should be meaningfully non-uniform
    entropy = -np.sum(result[result > 0] * np.log(result[result > 0]))
    max_entropy = np.log(n_legal)
    assert entropy < 0.9 * max_entropy, (
        f"Entropy {entropy:.2f} too close to max {max_entropy:.2f}"
    )


# ── Tests: KL policy loss ─────────────────────────────────────────────────────

def test_kl_loss_decreases():
    """KL loss decreases over training steps with augment-equivalent setup."""
    torch.manual_seed(42)
    n_actions = 362
    batch_size = 32

    # Simple linear model to verify loss can decrease
    model = torch.nn.Sequential(
        torch.nn.Linear(n_actions, n_actions),
        torch.nn.LogSoftmax(dim=-1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Fixed target distribution (soft, non-uniform)
    target = torch.softmax(torch.randn(batch_size, n_actions) * 2, dim=-1)
    valid_mask = torch.ones(batch_size, dtype=torch.bool)
    device = torch.device("cpu")

    # Fixed input so the model can memorize the mapping
    x = torch.randn(batch_size, n_actions)
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        log_policy = model(x)
        loss = compute_kl_policy_loss(log_policy, target, valid_mask, device)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should trend downward (allow some noise — check first vs last 5 avg)
    avg_first5 = sum(losses[:5]) / 5
    avg_last5 = sum(losses[-5:]) / 5
    assert avg_last5 < avg_first5, (
        f"KL loss not decreasing: first5 avg={avg_first5:.4f}, last5 avg={avg_last5:.4f}"
    )


def test_kl_loss_fp16_safe():
    """KL loss handles near-zero target entries without NaN/Inf under FP16 autocast."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = 362
    batch_size = 16

    # Target with many near-zero entries (sparse distribution)
    target = torch.zeros(batch_size, n_actions, device=device)
    for i in range(batch_size):
        # Only 3 non-zero entries per row
        idx = torch.randint(0, n_actions, (3,))
        target[i, idx] = torch.softmax(torch.randn(3, device=device), dim=-1)

    log_policy = torch.log_softmax(torch.randn(batch_size, n_actions, device=device), dim=-1)
    valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    use_fp16 = device.type == "cuda"
    with autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
        loss = compute_kl_policy_loss(log_policy, target, valid_mask, device)

    assert torch.isfinite(loss), f"KL loss is not finite: {loss.item()}"
    assert loss.item() >= 0, f"KL loss should be non-negative: {loss.item()}"
