"""
Phase 1 tests for Trainer (training step + checkpoint round-trip).

Run with: .venv/bin/pytest tests/test_trainer.py -v
"""
import json
from pathlib import Path
from unittest.mock import patch, call

import numpy as np
import pytest
import torch

from engine import ReplayBuffer
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.checkpoints import normalize_model_state_dict_keys
from hexo_rl.training.losses import compute_policy_loss, compute_kl_policy_loss
from hexo_rl.training.trainer import Trainer, prune_policy_targets


FAST_CONFIG = {
    "board_size":           19,
    "res_blocks":           2,
    "filters":              32,
    "batch_size":           8,
    "lr":                   2e-3,
    "weight_decay":         1e-4,
    "checkpoint_interval":  5,
    "log_interval":         1,
    "torch_compile":        False,
}


def make_trainer(tmp_path: Path) -> Trainer:
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    return Trainer(model, FAST_CONFIG, checkpoint_dir=tmp_path)


def fill_buffer(size: int = 32) -> ReplayBuffer:
    buf = ReplayBuffer(capacity=200)
    rng   = np.random.default_rng(0)
    own   = np.ones(361, dtype=np.uint8)
    wl    = np.zeros(361, dtype=np.uint8)
    chain = np.zeros((6, 19, 19), dtype=np.float16)
    for _ in range(size):
        state   = rng.random((18, 19, 19), dtype=np.float32).astype(np.float16)
        policy  = rng.dirichlet(np.ones(362)).astype(np.float32)
        outcome = float(rng.choice([-1.0, 0.0, 1.0]))
        buf.push(state, chain, policy, outcome, own, wl)
    return buf


# ── Training step ─────────────────────────────────────────────────────────────

def test_train_step_returns_loss_keys(tmp_path: Path):
    trainer = make_trainer(tmp_path)
    buf     = fill_buffer()
    result  = trainer.train_step(buf)
    assert "loss"        in result
    assert "policy_loss" in result
    assert "value_loss"  in result


def test_train_step_loss_is_finite(tmp_path: Path):
    trainer = make_trainer(tmp_path)
    buf     = fill_buffer()
    result  = trainer.train_step(buf)
    # policy_entropy_pretrain / _selfplay are intentionally NaN on the
    # single-buffer path (no pretrain/selfplay distinction). grad_norm
    # can be inf (pre-clip) but must not be NaN.
    # §101: fast-search target metrics are NaN when the batch carries no
    # quick-search rows (default: full_search_mask is None ⇒ all full).
    _nan_allowed = {
        "policy_entropy_pretrain", "policy_entropy_selfplay",
        # no recent_buffer → n_recent=0 → recent split undefined
        "policy_entropy_recent",
        "policy_target_entropy_fastsearch",
        "policy_target_kl_uniform_fastsearch",
    }
    for k, v in result.items():
        if k in _nan_allowed:
            continue  # NaN is correct when no split is in effect
        elif k == "grad_norm":
            assert not np.isnan(v), f"{k} = {v} is NaN"
        else:
            assert np.isfinite(v), f"{k} = {v} is not finite"


def test_train_step_increments_step(tmp_path: Path):
    trainer = make_trainer(tmp_path)
    buf     = fill_buffer()
    assert trainer.step == 0
    trainer.train_step(buf)
    assert trainer.step == 1
    trainer.train_step(buf)
    assert trainer.step == 2


def test_loss_decreases_over_multiple_steps(tmp_path: Path):
    """Policy loss should decrease when the model is trained on fixed data.

    Uses augment=False so the test is deterministic regardless of the Rust
    buffer's internal RNG seed. We're testing the optimizer loop, not augmentation.
    """
    torch.manual_seed(42)
    model   = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, FAST_CONFIG, checkpoint_dir=tmp_path)
    buf     = fill_buffer(size=64)

    first_loss = trainer.train_step(buf, augment=False)["loss"]
    last_loss  = first_loss
    for _ in range(199):
        last_loss = trainer.train_step(buf, augment=False)["loss"]

    # Over 200 steps the loss should drop noticeably from the initial value.
    assert last_loss < first_loss, (
        f"loss did not decrease: first={first_loss:.4f} last={last_loss:.4f}"
    )


# ── Checkpoint save ───────────────────────────────────────────────────────────

def test_checkpoint_saved_at_interval(tmp_path: Path):
    trainer = make_trainer(tmp_path)
    buf     = fill_buffer()
    # checkpoint_interval = 5, so step 5 triggers a save
    for _ in range(5):
        trainer.train_step(buf)
    ckpt_files = list(tmp_path.glob("checkpoint_*.pt"))
    assert len(ckpt_files) == 1
    assert ckpt_files[0].name == "checkpoint_00000005.pt"


def test_inference_only_saved_with_checkpoint(tmp_path: Path):
    trainer = make_trainer(tmp_path)
    buf     = fill_buffer()
    for _ in range(5):
        trainer.train_step(buf)
    assert (tmp_path / "inference_only.pt").exists()


def test_checkpoint_log_json_written(tmp_path: Path):
    trainer = make_trainer(tmp_path)
    buf     = fill_buffer()
    for _ in range(5):
        trainer.train_step(buf)
    log_path = tmp_path / "checkpoint_log.json"
    assert log_path.exists()
    with open(log_path) as f:
        log = json.load(f)
    assert len(log) == 1
    assert log[0]["step"] == 5
    assert "loss" in log[0]


# ── Checkpoint round-trip ─────────────────────────────────────────────────────

def test_checkpoint_round_trip(tmp_path: Path):
    """Load a checkpoint and verify model outputs match before/after."""
    torch.manual_seed(0)
    model   = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, FAST_CONFIG, checkpoint_dir=tmp_path)
    buf     = fill_buffer()

    for _ in range(5):
        trainer.train_step(buf)

    ckpt_path = tmp_path / "checkpoint_00000005.pt"
    assert ckpt_path.exists()

    # Record model outputs before reload.
    x = torch.zeros(1, 18, 19, 19, device=trainer.device)
    trainer.model.eval()
    with torch.no_grad():
        log_p_before, v_before, _ = trainer.model(x)

    # Reload.
    restored = Trainer.load_checkpoint(ckpt_path, checkpoint_dir=tmp_path)
    assert restored.step == 5

    x_r = torch.zeros(1, 18, 19, 19, device=restored.device)
    restored.model.eval()
    with torch.no_grad():
        log_p_after, v_after, _ = restored.model(x_r)

    # Compare on CPU.
    assert torch.allclose(log_p_before.cpu(), log_p_after.cpu(), atol=1e-4), \
        "policy mismatch after reload"
    assert torch.allclose(v_before.cpu(), v_after.cpu(), atol=1e-4), \
        "value mismatch after reload"


def test_checkpoint_optimizer_state_preserved(tmp_path: Path):
    """AdamW momentum state should survive a round-trip."""
    model   = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, FAST_CONFIG, checkpoint_dir=tmp_path)
    buf     = fill_buffer()
    for _ in range(5):
        trainer.train_step(buf)

    ckpt_path = tmp_path / "checkpoint_00000005.pt"
    restored  = Trainer.load_checkpoint(ckpt_path, checkpoint_dir=tmp_path)

    orig_state = trainer.optimizer.state_dict()
    rest_state = restored.optimizer.state_dict()
    assert len(orig_state["state"]) == len(rest_state["state"]), \
        "optimizer state group count mismatch"


def test_scheduler_steps_each_train_step(tmp_path: Path):
    cfg = {
        **FAST_CONFIG,
        "lr_schedule": "cosine",
        "total_steps": 20,
        "min_lr": 1e-5,
        "fp16": False,  # FP16 overflow can skip scheduler.step() — disable for determinism
    }
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    buf = fill_buffer()

    assert trainer.scheduler is not None
    start_epoch = trainer.scheduler.last_epoch

    trainer.train_step(buf)
    assert trainer.scheduler.last_epoch == start_epoch + 1

    trainer.train_step(buf)
    assert trainer.scheduler.last_epoch == start_epoch + 2


def test_scheduler_state_round_trip(tmp_path: Path):
    cfg = {
        **FAST_CONFIG,
        "lr_schedule": "cosine",
        "total_steps": 20,
        "min_lr": 1e-5,
    }
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    buf = fill_buffer()

    for _ in range(5):
        trainer.train_step(buf)

    ckpt_path = tmp_path / "checkpoint_00000005.pt"
    restored = Trainer.load_checkpoint(ckpt_path, checkpoint_dir=tmp_path)

    assert trainer.scheduler is not None
    assert restored.scheduler is not None
    assert restored.scheduler.last_epoch == trainer.scheduler.last_epoch


def test_load_checkpoint_allows_config_override(tmp_path: Path):
    cfg = {
        **FAST_CONFIG,
        "lr_schedule": "cosine",
        "total_steps": 20,
        "min_lr": 1e-5,
    }
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    buf = fill_buffer()

    for _ in range(5):
        trainer.train_step(buf)

    ckpt_path = tmp_path / "checkpoint_00000005.pt"
    restored = Trainer.load_checkpoint(
        ckpt_path,
        checkpoint_dir=tmp_path,
        config_overrides={"total_steps": 100},
    )

    assert restored.config.get("total_steps") == 100
    assert restored.scheduler is not None
    assert int(restored.scheduler.T_max) == 100


def test_normalize_state_dict_adds_tower_aliases():
    state = {
        "trunk.tower.0.conv1.weight": torch.randn(8, 8, 3, 3),
        "_orig_mod.module.policy_fc.weight": torch.randn(10, 20),
    }

    from hexo_rl.training.checkpoints import normalize_model_state_dict_keys
    normalized = normalize_model_state_dict_keys(state)

    assert "trunk.tower.0.conv1.weight" in normalized
    assert "tower.0.conv1.weight" in normalized
    assert "policy_fc.weight" in normalized


def test_load_weights_only_checkpoint_infers_architecture(tmp_path: Path):
    base = HexTacToeNet(board_size=9, in_channels=18, res_blocks=2, filters=32)
    base_state = base.state_dict()

    # Simulate a bootstrap-style checkpoint that only has trunk.* keys.
    trunk_only = {
        k: v
        for k, v in base_state.items()
        if not k.startswith("tower.")
    }

    ckpt_path = tmp_path / "bootstrap_like.pt"
    torch.save(trunk_only, ckpt_path)

    # Fallback config omits architecture keys so inference can populate them
    # from the checkpoint (B-007: config must agree with ckpt or be silent).
    fallback = {
        "batch_size": 8,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "checkpoint_interval": 5,
        "log_interval": 1,
    }

    restored = Trainer.load_checkpoint(ckpt_path, checkpoint_dir=tmp_path, fallback_config=fallback)

    assert restored.model.board_size == 9
    assert restored.model.res_blocks == 2
    assert restored.model.filters == 32


# ── Policy target pruning ────────────────────────────────────────────────────

def test_prune_policy_targets_basic():
    """Verify pruning zeros entries at/below threshold and renormalizes."""
    pi = torch.tensor([[0.5, 0.01, 0.4, 0.09]])
    pruned = prune_policy_targets(pi, threshold_frac=0.02)

    # threshold = 0.02 * 0.5 = 0.01; strict > means 0.01 is zeroed
    assert pruned[0, 1].item() == 0.0, "entry at threshold should be zeroed"
    assert pruned[0, 0].item() > 0.0, "max entry should be kept"
    assert pruned[0, 2].item() > 0.0, "0.4 should be kept"
    assert pruned[0, 3].item() > 0.0, "0.09 should be kept"
    assert abs(pruned.sum().item() - 1.0) < 1e-5, "should renormalize to 1.0"


def test_prune_policy_targets_zero_frac_noop():
    """threshold_frac=0 should return the input unchanged."""
    pi = torch.tensor([[0.5, 0.3, 0.2]])
    result = prune_policy_targets(pi, threshold_frac=0.0)
    assert torch.allclose(pi, result)


def test_prune_policy_targets_batch():
    """Pruning should work independently per row."""
    pi = torch.tensor([
        [0.8, 0.1, 0.05, 0.05],
        [0.25, 0.25, 0.25, 0.25],
    ])
    pruned = prune_policy_targets(pi, threshold_frac=0.10)
    # Row 0: threshold = 0.10 * 0.8 = 0.08. Keep 0.8 and 0.1; prune 0.05, 0.05
    assert pruned[0, 2].item() == 0.0
    assert pruned[0, 3].item() == 0.0
    assert pruned[0, 0].item() > 0.0
    assert pruned[0, 1].item() > 0.0
    # Row 1: threshold = 0.10 * 0.25 = 0.025. All entries > 0.025, so all kept
    assert (pruned[1] > 0).all()
    # Both rows sum to 1.0
    assert abs(pruned[0].sum().item() - 1.0) < 1e-5
    assert abs(pruned[1].sum().item() - 1.0) < 1e-5


# ── Grad norm, value accuracy, lr in loss_info ──────────────────────────────


def test_train_step_returns_grad_norm(tmp_path: Path):
    """train_step must return a non-negative, non-NaN grad_norm.

    Note: grad_norm may be inf when FP16 GradScaler detects overflow and
    skips the optimizer step — this is correct diagnostic behaviour.
    """
    trainer = make_trainer(tmp_path)
    buf = fill_buffer()
    result = trainer.train_step(buf)
    assert "grad_norm" in result
    assert result["grad_norm"] >= 0.0
    assert not np.isnan(result["grad_norm"]), f"grad_norm = {result['grad_norm']}"


def test_train_step_returns_value_accuracy(tmp_path: Path):
    """train_step must return value_accuracy in [0, 1]."""
    trainer = make_trainer(tmp_path)
    buf = fill_buffer()
    result = trainer.train_step(buf)
    assert "value_accuracy" in result
    assert 0.0 <= result["value_accuracy"] <= 1.0


def test_train_step_returns_lr(tmp_path: Path):
    """train_step must return the current learning rate."""
    trainer = make_trainer(tmp_path)
    buf = fill_buffer()
    result = trainer.train_step(buf)
    assert "lr" in result
    assert result["lr"] > 0.0


def test_grad_norm_uses_config_grad_clip(tmp_path: Path):
    """grad_clip from config should be used as the clipping threshold."""
    cfg = {**FAST_CONFIG, "grad_clip": 0.5}
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    buf = fill_buffer()
    result = trainer.train_step(buf)
    assert "grad_norm" in result
    # Pre-clip norm can be inf if gradients are very large — that's valid.
    assert result["grad_norm"] >= 0.0
    assert not np.isnan(result["grad_norm"])


def test_lr_changes_with_scheduler(tmp_path: Path):
    """When a scheduler is active, lr in loss_info should reflect scheduler state."""
    cfg = {
        **FAST_CONFIG,
        "lr_schedule": "cosine",
        "total_steps": 10,
        "min_lr": 1e-5,
    }
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    buf = fill_buffer()

    result1 = trainer.train_step(buf)
    for _ in range(4):
        trainer.train_step(buf)
    result5 = trainer.train_step(buf)

    # Cosine schedule should decrease LR over time
    assert result5["lr"] < result1["lr"]


# ── RecentBuffer ──────────────────────────────────────────────────────────────

from hexo_rl.training.recency_buffer import RecentBuffer


def make_recent_buffer(capacity: int = 64) -> RecentBuffer:
    buf = RecentBuffer(capacity=capacity)
    rng = np.random.default_rng(1)
    for _ in range(capacity // 2):
        buf.push(
            rng.random((18, 19, 19), dtype=np.float32).astype(np.float16),
            policy=rng.dirichlet(np.ones(362)).astype(np.float32),
            outcome=float(rng.choice([-1.0, 1.0])),
        )
    return buf


def test_recent_buffer_size_tracks_pushes():
    buf = RecentBuffer(capacity=10)
    assert buf.size == 0
    rng = np.random.default_rng(0)
    for i in range(7):
        buf.push(np.zeros((18, 19, 19), dtype=np.float16),
                 policy=np.ones(362, dtype=np.float32) / 362, outcome=1.0)
        assert buf.size == i + 1


def test_recent_buffer_caps_at_capacity():
    buf = RecentBuffer(capacity=4)
    for _ in range(10):
        buf.push(np.zeros((18, 19, 19), dtype=np.float16),
                 policy=np.ones(362, dtype=np.float32) / 362, outcome=0.0)
    assert buf.size == 4


def test_recent_buffer_sample_shapes():
    buf = make_recent_buffer(capacity=32)
    states, chain_planes, policies, outcomes, ownership, winning_line, is_full_search = buf.sample(8)
    assert states.shape         == (8, 18, 19, 19)
    assert chain_planes.shape   == (8, 6, 19, 19)
    assert policies.shape       == (8, 362)
    assert outcomes.shape       == (8,)
    assert ownership.shape      == (8, 361)
    assert winning_line.shape   == (8, 361)
    assert is_full_search.shape == (8,)
    assert ownership.dtype      == np.uint8
    assert winning_line.dtype   == np.uint8
    assert is_full_search.dtype == np.uint8


def test_recent_buffer_sample_empty_raises():
    buf = RecentBuffer(capacity=16)
    with pytest.raises(ValueError, match="empty"):
        buf.sample(4)


def test_train_step_with_recent_buffer_returns_loss_keys(tmp_path: Path):
    cfg = {**FAST_CONFIG, "recency_weight": 0.75}
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    buf = fill_buffer()
    recent = make_recent_buffer()
    result = trainer.train_step(buf, augment=False, recent_buffer=recent)
    assert "loss"        in result
    assert "policy_loss" in result
    assert "value_loss"  in result


def test_train_step_recent_buffer_loss_is_finite(tmp_path: Path):
    cfg = {**FAST_CONFIG, "recency_weight": 0.75}
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    buf = fill_buffer()
    recent = make_recent_buffer()
    result = trainer.train_step(buf, augment=False, recent_buffer=recent)
    # §101: fast-search target metrics are NaN when the batch carries no
    # quick-search rows (default: full_search_mask is None ⇒ all full).
    _nan_allowed = {
        "policy_entropy_pretrain", "policy_entropy_selfplay",
        "policy_target_entropy_fastsearch",
        "policy_target_kl_uniform_fastsearch",
    }
    for k, v in result.items():
        if k in _nan_allowed:
            continue  # NaN is correct when no split is in effect
        elif k == "grad_norm":
            assert not np.isnan(v), f"{k} = {v} is NaN"
        else:
            assert np.isfinite(v), f"{k} = {v} is not finite"


def test_train_step_recent_buffer_zero_weight_falls_back(tmp_path: Path):
    """recency_weight=0 should fall back to full-buffer sampling without error."""
    cfg = {**FAST_CONFIG, "recency_weight": 0.0}
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    buf = fill_buffer()
    recent = make_recent_buffer()
    result = trainer.train_step(buf, augment=False, recent_buffer=recent)
    assert "loss" in result


def test_recent_buffer_save_load_round_trip(tmp_path: Path):
    buf = make_recent_buffer(capacity=64)
    path = str(tmp_path / "recent.npz")
    n_saved = buf.save_to_path(path)
    assert n_saved == buf.size

    restored = RecentBuffer(capacity=64)
    n_loaded = restored.load_from_path(path)
    assert n_loaded == n_saved
    assert restored.size == buf.size

    orig = buf.sample(32)
    rest = restored.sample(32)
    assert orig[0].shape == rest[0].shape  # states
    assert orig[2].shape == rest[2].shape  # policies


def test_recent_buffer_save_load_full_ring(tmp_path: Path):
    """Full ring (size == capacity) preserves all entries after round-trip."""
    buf = RecentBuffer(capacity=8)
    rng = np.random.default_rng(42)
    for i in range(12):  # overfill to exercise ring wraparound
        buf.push(
            rng.random((18, 19, 19), dtype=np.float32).astype(np.float16),
            policy=rng.dirichlet(np.ones(362)).astype(np.float32),
            outcome=float(rng.choice([-1.0, 1.0])),
        )
    assert buf.size == 8

    path = str(tmp_path / "recent_full.npz")
    buf.save_to_path(path)
    restored = RecentBuffer(capacity=8)
    restored.load_from_path(path)
    assert restored.size == 8


def test_recent_buffer_save_empty_returns_zero(tmp_path: Path):
    buf = RecentBuffer(capacity=16)
    n = buf.save_to_path(str(tmp_path / "empty.npz"))
    assert n == 0


# ── Fast-game policy masking ──────────────────────────────────────────────────


def test_fast_game_policy_loss_is_zero():
    """Fast-game positions (all-zero policy) must contribute zero policy loss.

    In Rust game_runner.rs:286-291, fast-game positions are stored with
    policy = vec![0.0; n_actions] before crossing the PyO3 boundary.
    The sum < 1e-6 mask in trainer.py:280 + losses.py:35 must fire on those
    rows so they are excluded from the cross-entropy computation.
    """
    from hexo_rl.training.losses import compute_policy_loss

    device = torch.device("cpu")
    n_actions = 362

    # Row 0: fast-game position — policy all zeros (as zeroed by Rust)
    # Row 1: normal position — uniform policy
    target_policy = torch.zeros(2, n_actions)
    target_policy[1] = 1.0 / n_actions  # uniform

    # Fake log-policy from a model (uniform log-softmax)
    log_policy = torch.full((2, n_actions), -torch.log(torch.tensor(float(n_actions))))

    valid_mask = target_policy.sum(dim=1) > 1e-6

    # Mask: row 0 (fast game) must be invalid, row 1 (normal) must be valid
    assert not valid_mask[0].item(), "fast-game row should be masked out"
    assert valid_mask[1].item(), "normal row should be included"

    loss = compute_policy_loss(log_policy, target_policy, valid_mask, device)

    # fast-game row is excluded; loss is computed on the normal row only
    assert loss.item() > 0.0, "normal-position policy loss must be non-zero"

    # Verify: if ALL rows are fast-game (all zero), total policy loss is zero
    all_fast_policy = torch.zeros(2, n_actions)
    all_fast_mask = all_fast_policy.sum(dim=1) > 1e-6
    assert not all_fast_mask.any(), "all-fast batch should have empty valid mask"
    loss_all_fast = compute_policy_loss(log_policy, all_fast_policy, all_fast_mask, device)
    assert loss_all_fast.item() == 0.0, "all-fast-game batch must produce zero policy loss"


# ── Value uncertainty head (trainer integration) ──────────────────────────────

def test_uncertainty_head_appears_in_loss_info(tmp_path: Path):
    """With uncertainty_weight > 0, loss_info must contain avg_sigma and uncertainty_loss."""
    cfg = {**FAST_CONFIG, "uncertainty_weight": 0.05}
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    buf = fill_buffer()
    result = trainer.train_step(buf, augment=False)
    assert "avg_sigma" in result, "avg_sigma missing from loss_info"
    assert "uncertainty_loss" in result, "uncertainty_loss missing from loss_info"
    assert result["avg_sigma"] > 0.0, "avg_sigma should be positive"
    assert np.isfinite(result["uncertainty_loss"]), "uncertainty_loss must be finite"
    assert np.isfinite(result["avg_sigma"]), "avg_sigma must be finite"


def test_uncertainty_head_absent_when_weight_zero(tmp_path: Path):
    """With uncertainty_weight=0 (default), loss_info must NOT contain avg_sigma."""
    cfg = {**FAST_CONFIG}  # no uncertainty_weight key → defaults to 0.0
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    buf = fill_buffer()
    result = trainer.train_step(buf, augment=False)
    assert "avg_sigma" not in result


def test_train_step_logs_grad_norm(tmp_path: Path):
    """train_step must emit a structlog 'train_step' event with grad_norm, total_loss, and step."""
    import structlog.testing

    trainer = make_trainer(tmp_path)
    buf = fill_buffer()

    with structlog.testing.capture_logs() as captured:
        trainer.train_step(buf, augment=False)

    train_step_events = [e for e in captured if e.get("event") == "train_step"]
    assert train_step_events, "No 'train_step' log event found"
    evt = train_step_events[0]
    assert "grad_norm"   in evt, "grad_norm missing from train_step log"
    assert "total_loss"  in evt, "total_loss missing from train_step log"
    assert "step"        in evt, "step missing from train_step log"
    assert evt["step"] == 1
    assert not np.isnan(evt["grad_norm"]), f"grad_norm is NaN: {evt}"


def test_uncertainty_loss_is_finite_with_aux(tmp_path: Path):
    """Uncertainty head must work alongside the opp_reply aux head."""
    cfg = {
        **FAST_CONFIG,
        "uncertainty_weight": 0.05,
        "aux_opp_reply_weight": 0.15,
    }
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    buf = fill_buffer()
    result = trainer.train_step(buf, augment=False)
    assert "avg_sigma" in result
    assert "opp_reply_loss" in result
    assert np.isfinite(result["avg_sigma"])
    assert np.isfinite(result["uncertainty_loss"])


# ── completed_q_values: KL vs CE policy loss path (architecture review C1) ───


def test_completed_q_values_true_uses_kl_loss(tmp_path: Path):
    """When completed_q_values is True, trainer must call compute_kl_policy_loss."""
    cfg = {**FAST_CONFIG, "completed_q_values": True}
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    buf = fill_buffer()

    with patch(
        "hexo_rl.training.trainer.compute_kl_policy_loss",
        wraps=trainer.__class__.__module__ and __import__(
            "hexo_rl.training.losses", fromlist=["compute_kl_policy_loss"]
        ).compute_kl_policy_loss,
    ) as mock_kl, patch(
        "hexo_rl.training.trainer.compute_policy_loss",
    ) as mock_ce:
        result = trainer.train_step(buf, augment=False)

    mock_kl.assert_called_once()
    mock_ce.assert_not_called()
    assert "policy_loss" in result
    assert np.isfinite(result["policy_loss"])


def test_completed_q_values_false_uses_ce_loss(tmp_path: Path):
    """When completed_q_values is False (default), trainer must use CE policy loss."""
    cfg = {**FAST_CONFIG, "completed_q_values": False}
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    buf = fill_buffer()

    with patch(
        "hexo_rl.training.trainer.compute_policy_loss",
        wraps=__import__(
            "hexo_rl.training.losses", fromlist=["compute_policy_loss"]
        ).compute_policy_loss,
    ) as mock_ce, patch(
        "hexo_rl.training.trainer.compute_kl_policy_loss",
    ) as mock_kl:
        result = trainer.train_step(buf, augment=False)

    mock_ce.assert_called_once()
    mock_kl.assert_not_called()
    assert "policy_loss" in result
    assert np.isfinite(result["policy_loss"])


def test_completed_q_values_absent_defaults_to_ce(tmp_path: Path):
    """When completed_q_values is absent from config, trainer defaults to CE."""
    cfg = {**FAST_CONFIG}  # no completed_q_values key
    assert "completed_q_values" not in cfg
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    buf = fill_buffer()

    with patch(
        "hexo_rl.training.trainer.compute_policy_loss",
        wraps=__import__(
            "hexo_rl.training.losses", fromlist=["compute_policy_loss"]
        ).compute_policy_loss,
    ) as mock_ce, patch(
        "hexo_rl.training.trainer.compute_kl_policy_loss",
    ) as mock_kl:
        result = trainer.train_step(buf, augment=False)

    mock_ce.assert_called_once()
    mock_kl.assert_not_called()

# ── Checkpoint retention policy ───────────────────────────────────────────────

def test_eval_checkpoints_not_pruned(tmp_path: Path):
    """Eval-step checkpoints survive rotation; rolling window keeps 10 most recent.

    Setup: eval_interval=5000, max_checkpoints_kept=10, checkpoint_interval=500.
    Simulate 30 checkpoint files at steps 500..15000.
    Eval steps: 5000, 10000, 15000.
    Non-eval steps: 27 files — rotation keeps newest 10.
    """
    from hexo_rl.training.checkpoints import prune_checkpoints

    eval_interval = 5000
    max_kept = 10
    all_steps = list(range(500, 15001, 500))  # 500, 1000, ..., 15000  (30 steps)
    assert len(all_steps) == 30

    eval_steps = {s for s in all_steps if s % eval_interval == 0}       # {5000, 10000, 15000}
    rolling_steps = sorted(s for s in all_steps if s not in eval_steps)  # 27 steps
    expected_rolling = set(rolling_steps[-max_kept:])                    # 10 newest non-eval

    # Create fake checkpoint files.
    for step in all_steps:
        (tmp_path / f"checkpoint_{step:08d}.pt").touch()

    predicate = lambda s: s > 0 and s % eval_interval == 0
    prune_checkpoints(tmp_path, max_kept, preserve_predicate=predicate)

    present = {int(p.stem.split("_")[1]) for p in tmp_path.glob("checkpoint_*.pt")}

    # All eval checkpoints must survive.
    for step in eval_steps:
        assert step in present, f"eval checkpoint {step} was pruned"

    # Newest 10 rolling checkpoints must survive.
    for step in expected_rolling:
        assert step in present, f"rolling checkpoint {step} (top-10) was pruned"

    # Older rolling checkpoints must be gone.
    expected_absent = set(rolling_steps[:-max_kept])
    for step in expected_absent:
        assert step not in present, f"old rolling checkpoint {step} should have been pruned"

    # Total present = 3 eval + 10 rolling = 13.
    assert len(present) == len(eval_steps) + max_kept


# ── F-002: BatchNorm checkpoint rejection ─────────────────────────────────────

def test_normalize_rejects_bn_keys():
    """normalize_model_state_dict_keys must raise RuntimeError on pre-§99 BN keys (F-002)."""
    state = {
        "trunk.tower.0.bn1.weight": torch.zeros(8),
        "trunk.tower.0.bn1.running_mean": torch.zeros(8),
    }
    with pytest.raises(RuntimeError, match="BatchNorm"):
        normalize_model_state_dict_keys(state)


def test_normalize_rejects_bn_keys_with_orig_mod_prefix():
    """BN keys prefixed with _orig_mod. must also be caught before normalisation."""
    state = {
        "_orig_mod.trunk.tower.0.input_bn.weight": torch.zeros(32),
        "_orig_mod.trunk.tower.0.bn2.bias": torch.zeros(32),
    }
    with pytest.raises(RuntimeError, match="BatchNorm"):
        normalize_model_state_dict_keys(state)


def test_load_checkpoint_rejects_bn_checkpoint(tmp_path: Path):
    """Trainer.load_checkpoint must propagate the RuntimeError on pre-§99 checkpoints (F-002)."""
    bn_state = {
        "trunk.tower.0.bn1.weight": torch.zeros(32),
        "trunk.tower.0.bn1.running_mean": torch.zeros(32),
        "trunk.tower.0.bn1.running_var": torch.ones(32),
        "trunk.tower.0.bn1.bias": torch.zeros(32),
    }
    ckpt = {
        "step": 0,
        "model_state": bn_state,
        "config": FAST_CONFIG,
        "optimizer_state": {},
        "scaler_state": {},
    }
    ckpt_path = tmp_path / "stale_bn.pt"
    torch.save(ckpt, ckpt_path)
    with pytest.raises(RuntimeError, match="BatchNorm"):
        Trainer.load_checkpoint(ckpt_path, fallback_config=FAST_CONFIG)


# ── F-003: strict=False key-set guard ─────────────────────────────────────────

def test_load_checkpoint_no_keys_silently_dropped(tmp_path: Path):
    """After a save→load round-trip no model key must be silently dropped by strict=False (F-003).

    Uses augment=False so the fill pattern is deterministic.
    """
    trainer = make_trainer(tmp_path)
    ckpt_path = trainer.save_checkpoint()

    loaded = Trainer.load_checkpoint(ckpt_path)
    expected_keys = set(trainer.model.state_dict().keys())
    loaded_keys = set(loaded.model.state_dict().keys())

    missing = expected_keys - loaded_keys
    extra = loaded_keys - expected_keys
    assert not missing, f"strict=False silently dropped keys: {sorted(missing)[:5]}"
    assert not extra, f"unexpected extra keys after load: {sorted(extra)[:5]}"


# ── F-005: selective policy loss mask ─────────────────────────────────────────

def test_selective_policy_loss_drops_quick_search_rows():
    """Policy loss computed over [full, full, quick, quick] rows must equal the loss
    computed over just the [full, full] rows — quick-search rows must be excluded (F-005).

    Tests compute_policy_loss and compute_kl_policy_loss (both used in §100) directly
    to pin the `valid_mask & full_search_mask` logic without going through train_step.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    B = 4

    # Rows 0,1: full-search — distinct non-uniform policies.
    # Rows 2,3: quick-search — identical uniform policies (should be ignored).
    rng = np.random.default_rng(0)
    log_policies = torch.from_numpy(
        np.log(rng.dirichlet(np.ones(362), size=B).astype(np.float32) + 1e-9)
    )
    targets = torch.from_numpy(rng.dirichlet(np.ones(362), size=B).astype(np.float32))
    valid_mask = torch.ones(B, dtype=torch.bool)
    full_search_mask = torch.tensor([1, 1, 0, 0], dtype=torch.uint8)

    # CE loss gated by full_search_mask.
    loss_masked = compute_policy_loss(log_policies, targets, valid_mask, device,
                                      full_search_mask=full_search_mask)
    # CE loss over only the full-search rows (rows 0 and 1).
    loss_full_only = compute_policy_loss(log_policies[:2], targets[:2],
                                         valid_mask[:2], device,
                                         full_search_mask=None)

    assert abs(loss_masked.item() - loss_full_only.item()) < 1e-5, (
        f"CE masked={loss_masked.item():.6f} vs full-only={loss_full_only.item():.6f}: "
        "quick-search rows must not contribute to policy loss"
    )

    # KL loss (completed-Q path) obeys the same mask.
    loss_kl_masked = compute_kl_policy_loss(log_policies, targets, valid_mask, device,
                                             full_search_mask=full_search_mask)
    loss_kl_full_only = compute_kl_policy_loss(log_policies[:2], targets[:2],
                                                valid_mask[:2], device,
                                                full_search_mask=None)
    assert abs(loss_kl_masked.item() - loss_kl_full_only.item()) < 1e-5, (
        f"KL masked={loss_kl_masked.item():.6f} vs full-only={loss_kl_full_only.item():.6f}: "
        "quick-search rows must not contribute to KL policy loss"
    )


def test_selective_policy_loss_all_quick_search_returns_nan():
    """When every row is quick-search, policy loss must be NaN (empty denominator)."""
    device = torch.device("cpu")
    rng = np.random.default_rng(1)
    log_p = torch.log(torch.from_numpy(
        rng.dirichlet(np.ones(362), size=4).astype(np.float32)
    ) + 1e-9)
    targets = torch.from_numpy(rng.dirichlet(np.ones(362), size=4).astype(np.float32))
    valid_mask = torch.ones(4, dtype=torch.bool)
    full_search_mask = torch.zeros(4, dtype=torch.uint8)  # all quick-search

    loss = compute_policy_loss(log_p, targets, valid_mask, torch.device("cpu"),
                               full_search_mask=full_search_mask)
    assert torch.isnan(loss) or loss.item() == 0.0, (
        f"expected NaN or 0 when all rows are quick-search, got {loss.item()}"
    )


# ── B-002: strict=False key-set guard (extended) ──────────────────────────────

def test_load_checkpoint_raises_on_unknown_missing_key(tmp_path: Path):
    """Checkpoint missing a model key (not in the BN allowlist) must fail load.

    Simulates a stale checkpoint where the trunk input_conv weight was dropped
    outside the BN migration. Silent strict=False would initialise input_conv
    randomly and destroy trunk integrity.
    """
    trainer = make_trainer(tmp_path)
    ckpt_path = trainer.save_checkpoint()

    # Corrupt the checkpoint: remove a weight that is neither a BN artifact
    # nor a tower/trunk.tower alias.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    removed_key = "trunk.input_conv.weight"
    assert removed_key in ckpt["model_state"], "precondition: key must be present"
    del ckpt["model_state"][removed_key]
    torch.save(ckpt, ckpt_path)

    with pytest.raises(RuntimeError, match="Checkpoint load_state_dict mismatch"):
        Trainer.load_checkpoint(ckpt_path, checkpoint_dir=tmp_path)


# ── B-003: scheduler state resume guard ───────────────────────────────────────

def test_missing_scheduler_state_raises(tmp_path: Path):
    """Resume with scheduler configured but ckpt lacking scheduler_state must raise."""
    cfg = {
        **FAST_CONFIG,
        "lr_schedule": "cosine",
        "total_steps": 100,
        "eta_min": 1e-5,
    }
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    ckpt_path = trainer.save_checkpoint()

    # Strip scheduler_state, simulating a legacy checkpoint.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt["scheduler_state"] = None
    torch.save(ckpt, ckpt_path)

    with pytest.raises(ValueError, match="scheduler_state missing"):
        Trainer.load_checkpoint(ckpt_path, checkpoint_dir=tmp_path)


def test_missing_scheduler_state_allow_fresh_scheduler(tmp_path: Path):
    """--allow-fresh-scheduler override lets load succeed with fresh scheduler."""
    cfg = {
        **FAST_CONFIG,
        "lr_schedule": "cosine",
        "total_steps": 100,
        "eta_min": 1e-5,
    }
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    ckpt_path = trainer.save_checkpoint()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt["scheduler_state"] = None
    torch.save(ckpt, ckpt_path)

    restored = Trainer.load_checkpoint(
        ckpt_path,
        checkpoint_dir=tmp_path,
        config_overrides={"allow_fresh_scheduler": True},
    )
    assert restored.scheduler is not None


# ── B-004: NaN-loss GradScaler decay ──────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP16 GradScaler only enabled on CUDA")
def test_nan_loss_decays_scale(tmp_path: Path):
    """NaN-loss guard must decay the fp16 scale without state-machine violation.

    Calling `GradScaler.update()` with no prior `step()` raises
    'No inf checks were recorded prior to update.' The guard must use
    `update(new_scale=)` so the scale halves (backoff_factor) cleanly.
    """
    cfg = {**FAST_CONFIG, "fp16": True}
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    buf = fill_buffer()

    # One successful step initialises the scaler's internal scale tensor.
    trainer.train_step(buf, augment=False)

    scale_before = trainer.scaler.get_scale()
    backoff = trainer.scaler.get_backoff_factor()

    # Force the NaN-loss guard by replacing compute_total_loss with a NaN producer.
    def _nan_total(*args, **kwargs):
        return torch.tensor(float("nan"), device=trainer.device)

    with patch("hexo_rl.training.trainer.compute_total_loss", side_effect=_nan_total):
        result = trainer.train_step(buf, augment=False)

    assert np.isnan(result["loss"])
    scale_after = trainer.scaler.get_scale()
    assert abs(scale_after - scale_before * backoff) < 1e-3, (
        f"expected scale to halve: before={scale_before} after={scale_after} backoff={backoff}"
    )


# ── B-007: force-override in _infer_model_hparams ─────────────────────────────

def test_hparam_mismatch_raises(tmp_path: Path):
    """Config explicitly setting a model hparam that disagrees with the checkpoint
    must raise ValueError rather than silently overriding.
    """
    base = HexTacToeNet(board_size=9, in_channels=18, res_blocks=2, filters=32)
    base_state = base.state_dict()
    ckpt_path = tmp_path / "weights_only.pt"
    torch.save(base_state, ckpt_path)

    # Explicit res_blocks=4 disagrees with ckpt-inferred 2.
    fallback = {
        "board_size": 9,
        "in_channels": 18,
        "res_blocks": 4,
        "filters": 32,
        "batch_size": 8,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "checkpoint_interval": 5,
        "log_interval": 1,
    }

    with pytest.raises(ValueError, match="res_blocks.*disagrees"):
        Trainer.load_checkpoint(ckpt_path, checkpoint_dir=tmp_path, fallback_config=fallback)
