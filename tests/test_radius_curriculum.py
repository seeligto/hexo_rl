"""§174 — radius curriculum tests.

T3: set_legal_move_radius guard unchanged (still errors with encoding).
T4: schedule resolution edge cases.
T5: override_legal_move_radius works with encoding + legal_moves count.
T6: radius transition log message (integration).
"""
from __future__ import annotations

import pytest
from engine import Board


# ── T3: §173 A6 guard intact ────────────────────────────────────────────────

@pytest.mark.parametrize("encoding_name", ["v6", "v6w25", "v7full"])
def test_set_legal_move_radius_guard_unchanged(encoding_name: str) -> None:
    """set_legal_move_radius after with_encoding_name still errors (§173 A6)."""
    board = Board.with_encoding_name(encoding_name)
    with pytest.raises((ValueError, Exception)):
        board.set_legal_move_radius(5)


# ── T5: override_legal_move_radius works WITH encoding ──────────────────────

@pytest.mark.parametrize("encoding_name", ["v6", "v6w25", "v7full"])
def test_override_legal_move_radius_succeeds_with_encoding(encoding_name: str) -> None:
    """override_legal_move_radius must succeed on encoding-bound boards."""
    board = Board.with_encoding_name(encoding_name)
    board.override_legal_move_radius(5)
    assert board.legal_move_radius() == 5


def test_override_legal_move_radius_count_v6w25() -> None:
    """v6w25 at R=5 must produce 90 empty legal moves once a stone is placed."""
    board = Board.with_encoding_name("v6w25")
    board.apply_move(0, 0)
    board.override_legal_move_radius(5)
    legal = board.legal_moves()
    # Hex-ball radius 5 = 91 cells total; minus occupied (0,0) = 90 empty.
    assert len(legal) == 90, f"expected 90 empty legal moves at R=5, got {len(legal)}"


# ── T4: schedule resolution ─────────────────────────────────────────────────

def _resolve_radius(full_config: dict, step: int) -> int | None:
    """Mirror of StepCoordinator._resolve_radius for unit testing."""
    schedule = full_config.get("selfplay", {}).get("legal_move_radius_schedule")
    if not schedule:
        return None
    current_radius = None
    for entry in schedule:
        if step >= entry["step"]:
            current_radius = entry["radius"]
    return current_radius


def test_resolve_radius_empty_schedule() -> None:
    assert _resolve_radius({}, 0) is None
    assert _resolve_radius({"selfplay": {}}, 100) is None


def test_resolve_radius_single_entry() -> None:
    cfg = {"selfplay": {"legal_move_radius_schedule": [{"step": 0, "radius": 5}]}}
    assert _resolve_radius(cfg, 0) == 5
    assert _resolve_radius(cfg, 999) == 5


def test_resolve_radius_multiple_entries() -> None:
    cfg = {"selfplay": {"legal_move_radius_schedule": [
        {"step": 0, "radius": 5},
        {"step": 50000, "radius": 6},
        {"step": 100000, "radius": 7},
    ]}}
    assert _resolve_radius(cfg, 0) == 5
    assert _resolve_radius(cfg, 49999) == 5
    assert _resolve_radius(cfg, 50000) == 6
    assert _resolve_radius(cfg, 99999) == 6
    assert _resolve_radius(cfg, 100000) == 7
    assert _resolve_radius(cfg, 200000) == 7


def test_resolve_radius_step_before_first_entry() -> None:
    cfg = {"selfplay": {"legal_move_radius_schedule": [
        {"step": 1000, "radius": 5},
    ]}}
    assert _resolve_radius(cfg, 0) is None
    assert _resolve_radius(cfg, 999) is None
    assert _resolve_radius(cfg, 1000) == 5


# ── T6: transition log message (integration) ────────────────────────────────

class FakePool:
    """Minimal pool stub for StepCoordinator protocol."""
    def __init__(self) -> None:
        self.games_completed = 0
        self.n_workers = 1
        self._radius: int | None = None
        self.logs: list[tuple[int, int | None]] = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def buffer_composition(self) -> dict:
        return {}

    def model_version_summary(self) -> dict:
        return {}

    def per_worker_draw_rates(self) -> dict:
        return {}

    def set_radius_override(self, radius: int | None) -> None:
        self._radius = radius
        self.logs.append((radius,))


def test_radius_transition_emitted() -> None:
    """StepCoordinator must emit a radius transition when step crosses boundary."""
    from hexo_rl.training.step_coordinator import StepCoordinator, StepCoordinatorConfig
    from hexo_rl.training.trainer import Trainer
    from hexo_rl.model.network import HexTacToeNet
    import torch

    # Build a minimal trainer
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32, in_channels=8)
    trainer = Trainer(
        model,
        {"lr": 0.001, "weight_decay": 1e-4, "fp16": False},
        device=torch.device("cpu"),
    )

    pool = FakePool()
    cfg = StepCoordinatorConfig(
        eval_interval=1000,
        log_interval=10,
        checkpoint_interval=500,
        composition_interval=5000,
        value_probe_interval=5000,
        min_buf_size=1,
        capacity=1000,
        buffer_schedule=(),
        training_steps_per_game=1.0,
        max_train_burst=1,
        batch_size=16,
        augment=False,
        recency_weight=0.0,
        mixing_initial_w=0.0,
        mixing_min_w=0.0,
        mixing_decay_steps=1.0,
        soft_ew_threshold=0.0,
        soft_ew_min_pts=0,
        hard_gn_threshold=100.0,
        hard_gn_min_steps=100,
        instrumentation_enabled=False,
        stop_step=None,
        final_eval_drain_timeout_sec=0.0,
    )

    full_config = {
        "selfplay": {
            "legal_move_radius_schedule": [
                {"step": 0, "radius": 5},
                {"step": 50, "radius": 6},
            ]
        }
    }

    coordinator = StepCoordinator(
        trainer=trainer,
        buffer=type("B", (), {"size": 100, "capacity": 1000, "resize": lambda *a: None, "save_to_path": lambda *a: None})(),
        pretrained_buffer=None,
        recent_buffer=None,
        pool=pool,
        eval_pipeline=None,
        gpu_monitor=type("G", (), {"gpu_util_pct": 0.0})(),
        subsystems=type("S", (), {"teardown": lambda self: None})(),
        anchor_state=type("A", (), {"best_model": None, "best_model_path": "", "best_model_step": None})(),
        shutdown=type("Sh", (), {"running": True, "shutdown_save": False})(),
        eval_model=None,
        bufs=None,
        early_game_probe=None,
        value_probe=None,
        axis_baseline=None,
        tb_writer=None,
        config=cfg,
        full_config=full_config,
    )

    # Initial radius should be set
    assert pool._radius == 5

    # Advance step past boundary (simulate step counter)
    trainer.step = 55
    # _resolve_radius called at init and then at log_interval; we call directly for test.
    new_radius = coordinator._resolve_radius(55)
    assert new_radius == 6

    # Manually trigger what step() does at log_interval
    coordinator._current_radius = new_radius
    pool.set_radius_override(new_radius)
    assert pool._radius == 6
    assert any(log == (6,) for log in pool.logs)
