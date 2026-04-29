"""Tests for the §115 early-game policy-entropy probe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hexo_rl.monitoring.early_game_probe import (
    DEFAULT_FIXTURE_PATH,
    EARLY_GAME_ENTROPY_WARN_THRESHOLD,
    EarlyGameProbe,
    _FIXTURE_PLIES,
    load_fixture,
    save_fixture,
)
from hexo_rl.model.network import HexTacToeNet


def _tiny_net() -> HexTacToeNet:
    """Full-board geometry, minimal depth so the test runs under 1 s on CPU."""
    return HexTacToeNet(board_size=19, filters=16, res_blocks=2)


# ── Fixture generation is deterministic ────────────────────────────────────────


def test_generate_fixture_is_deterministic(tmp_path: Path) -> None:
    p1 = tmp_path / "f1.npz"
    p2 = tmp_path / "f2.npz"
    save_fixture(p1)
    save_fixture(p2)
    b1 = np.load(p1)
    b2 = np.load(p2)
    for key in ("states", "plies", "seeds", "version"):
        np.testing.assert_array_equal(b1[key], b2[key], err_msg=f"key {key} differs")


def test_load_fixture_shape(tmp_path: Path) -> None:
    p = tmp_path / "probe.npz"
    payload = save_fixture(p)
    assert payload.states.shape == (len(_FIXTURE_PLIES), 8, 19, 19)
    assert payload.states.dtype == np.float16
    assert payload.plies.tolist() == list(_FIXTURE_PLIES)
    # Re-load through the reader.
    re = load_fixture(p)
    np.testing.assert_array_equal(re.states, payload.states)
    np.testing.assert_array_equal(re.plies, payload.plies)


def test_load_missing_fixture_regenerates(tmp_path: Path) -> None:
    p = tmp_path / "missing.npz"
    assert not p.exists()
    _ = load_fixture(p)
    assert p.exists()


# ── Probe compute ──────────────────────────────────────────────────────────────


def test_probe_emits_expected_keys(tmp_path: Path) -> None:
    p = tmp_path / "probe.npz"
    save_fixture(p)
    probe = EarlyGameProbe(device=torch.device("cpu"), fixture_path=p)
    metrics = probe.compute(_tiny_net())
    required = {
        "early_game_entropy_mean",
        "early_game_entropy_max",
        "early_game_top1_mass_mean",
        "early_game_entropy_by_ply",
        "early_game_top1_mass_by_ply",
    }
    assert required.issubset(metrics.keys())
    assert len(metrics["early_game_entropy_by_ply"]) == len(_FIXTURE_PLIES)
    assert len(metrics["early_game_top1_mass_by_ply"]) == len(_FIXTURE_PLIES)


def test_probe_entropy_is_nats_within_log_n(tmp_path: Path) -> None:
    """Every per-ply entropy must sit in [0, log(362)]."""
    p = tmp_path / "probe.npz"
    save_fixture(p)
    probe = EarlyGameProbe(device=torch.device("cpu"), fixture_path=p)
    metrics = probe.compute(_tiny_net())
    max_nats = float(np.log(19 * 19 + 1))
    for h in metrics["early_game_entropy_by_ply"]:
        assert 0.0 <= h <= max_nats + 1e-4, f"entropy {h} out of range [0, {max_nats}]"


def test_probe_deterministic_across_calls(tmp_path: Path) -> None:
    """Same model + same fixture → identical entropy (within fp tolerance)."""
    p = tmp_path / "probe.npz"
    save_fixture(p)
    probe = EarlyGameProbe(device=torch.device("cpu"), fixture_path=p)
    net = _tiny_net()
    m1 = probe.compute(net)
    m2 = probe.compute(net)
    assert m1["early_game_entropy_mean"] == pytest.approx(
        m2["early_game_entropy_mean"], abs=1e-5
    )
    for a, b in zip(m1["early_game_entropy_by_ply"], m2["early_game_entropy_by_ply"]):
        assert a == pytest.approx(b, abs=1e-5)


def test_probe_preserves_training_mode(tmp_path: Path) -> None:
    """compute() must restore train() mode if the model was training."""
    p = tmp_path / "probe.npz"
    save_fixture(p)
    probe = EarlyGameProbe(device=torch.device("cpu"), fixture_path=p)
    net = _tiny_net()
    net.train()
    assert net.training
    probe.compute(net)
    assert net.training, "probe must leave model in its original training mode"
    net.eval()
    probe.compute(net)
    assert not net.training


def test_warn_threshold_is_between_healthy_and_broken() -> None:
    """Sanity check on the threshold choice — bootstrap-v4 is ~3-4, broken is ~5.4."""
    assert 3.5 < EARLY_GAME_ENTROPY_WARN_THRESHOLD < 5.5


# ── Bootstrap-v4 smoke (skipped if the checkpoint isn't present) ───────────────


@pytest.mark.slow
def test_bootstrap_v4_entropy_range() -> None:
    """On a real bootstrap-v4 checkpoint the probe should read 2.5-4.5 nats."""
    ckpt_path = Path("checkpoints/bootstrap_model.pt")
    if not ckpt_path.exists():
        pytest.skip("bootstrap_model.pt not present")

    # Bootstrap checkpoint ships as a plain state_dict (no config blob), so
    # load it directly into a fresh HexTacToeNet with the canonical geometry.
    # Skip pre-P3 (18-plane) checkpoints — the guard rejects them.
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    if isinstance(state_dict, dict):
        _w = state_dict.get("trunk.input_conv.weight")
        if _w is not None and _w.dim() == 4 and _w.shape[1] == 18:
            pytest.skip("bootstrap_model.pt is a pre-P3 18-plane checkpoint; skipping until 8-plane bootstrap is trained")
    from hexo_rl.training.checkpoints import normalize_model_state_dict_keys
    state_dict = normalize_model_state_dict_keys(state_dict)
    net = HexTacToeNet(board_size=19, filters=128, res_blocks=12)
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    probe = EarlyGameProbe(device=torch.device("cpu"))
    metrics = probe.compute(net)
    mean_h = metrics["early_game_entropy_mean"]
    assert 2.5 < mean_h < 4.5, (
        f"bootstrap-v4 probe entropy {mean_h:.3f} outside expected 2.5-4.5 nat band; "
        f"per-ply: {metrics['early_game_entropy_by_ply']}"
    )


# ── Default fixture committed to the repo ──────────────────────────────────────


def test_default_fixture_is_committed() -> None:
    """`fixtures/early_game_probe_v1.npz` must ship with the repo."""
    assert DEFAULT_FIXTURE_PATH.exists(), (
        f"{DEFAULT_FIXTURE_PATH} missing — run "
        "`python scripts/build_early_game_probe.py` and commit the result."
    )
