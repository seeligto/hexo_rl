"""Tests for the /api/analyze policy viewer endpoint."""
from __future__ import annotations

import ast
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Find any checkpoint to test with. Only 8-channel checkpoints are runnable
# against the current 8-plane model (HEXB v6 wire format, P3 migration).
CKPT_DIR = Path("checkpoints")
_EXPECTED_IN_CHANNELS = 8


def _checkpoint_in_channels(path: Path) -> int | None:
    """Return the input-conv in_channels for a checkpoint, or None if unknown."""
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("model_state", "model_state_dict", "state_dict"):
            if isinstance(payload, dict) and key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
        if not isinstance(payload, dict):
            return None
        for k in ("trunk.input_conv.weight", "input_conv.weight"):
            if k in payload:
                return int(payload[k].shape[1])
        return None
    except Exception:
        return None


_all_ckpts = sorted(CKPT_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True) if CKPT_DIR.exists() else []
AVAILABLE_CKPTS = [p for p in _all_ckpts if _checkpoint_in_channels(p) == _EXPECTED_IN_CHANNELS]
TEST_CKPT = str(AVAILABLE_CKPTS[0]) if AVAILABLE_CKPTS else None

needs_checkpoint = pytest.mark.skipif(
    TEST_CKPT is None,
    reason=f"No {_EXPECTED_IN_CHANNELS}-plane checkpoint available",
)


@pytest.fixture(scope="module")
def app():
    """Create a Flask test app with the analyze blueprint."""
    from flask import Flask
    from hexo_rl.monitoring.analyze_api import analyze_bp

    app = Flask(__name__, static_folder=str(Path("hexo_rl/monitoring/static")))
    app.register_blueprint(analyze_bp)
    app.config["TESTING"] = True
    return app


@pytest.fixture(scope="module")
def client(app):
    return app.test_client()


class TestCheckpointList:
    def test_checkpoint_list_returns_entries(self, client):
        resp = client.get("/api/analyze/checkpoints")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        if AVAILABLE_CKPTS:
            assert len(data) >= 1
            assert "path" in data[0]
            assert "size_mb" in data[0]
            assert "mtime" in data[0]


@needs_checkpoint
class TestAnalyzeEmptyBoard:
    def test_analyze_empty_board(self, client):
        resp = client.post("/api/analyze", json={
            "checkpoint": TEST_CKPT,
            "moves": [],
        })
        assert resp.status_code == 200
        data = resp.get_json()

        # Policy sums to ~1.0
        policy = data["policy"]
        assert len(policy) > 0
        total = sum(e["prob"] for e in policy)
        assert abs(total - 1.0) < 0.01, f"Policy sum = {total}"

        # Finite value in [-1, 1]
        v = data["value"]
        assert -1.0 <= v <= 1.0

        # Positive entropy
        assert data["entropy_nats"] > 0
        assert 0 <= data["entropy_uniform_fraction"] <= 1.0

        # Turn info
        assert data["next_to_move"] == 1  # P1 moves first
        assert data["moves_remaining"] == 1  # single stone on first move
        assert data["legal_moves_count"] > 0


@needs_checkpoint
class TestAnalyzeWithMoves:
    def test_analyze_with_5_moves(self, client):
        moves = [
            {"q": 0, "r": 0, "player": 1},
            {"q": 1, "r": 0, "player": 0},
            {"q": -1, "r": 0, "player": 0},
            {"q": 0, "r": 1, "player": 1},
            {"q": 0, "r": -1, "player": 1},
        ]
        resp = client.post("/api/analyze", json={
            "checkpoint": TEST_CKPT,
            "moves": moves,
        })
        assert resp.status_code == 200
        data = resp.get_json()

        # After 5 moves: ply 0 = P1 (1 stone), ply 1-2 = P0 (2 stones),
        # ply 3-4 = P1 (2 stones). After 5 moves, P0 to move with 2 remaining.
        assert data["next_to_move"] == -1  # P0
        assert data["moves_remaining"] == 2


@needs_checkpoint
class TestAnalyzeMCTS:
    def test_mcts_puct(self, client):
        resp = client.post("/api/analyze", json={
            "checkpoint": TEST_CKPT,
            "moves": [],
            "mcts": {"enabled": True, "mode": "puct", "simulations": 20},
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "mcts" in data
        mcts = data["mcts"]
        assert mcts["total_sims"] >= 15  # at least ~75% of requested (batching)
        assert len(mcts["visits"]) > 0
        assert mcts["visits"][0]["visits"] > 0

    def test_mcts_gumbel(self, client):
        resp = client.post("/api/analyze", json={
            "checkpoint": TEST_CKPT,
            "moves": [],
            "mcts": {"enabled": True, "mode": "gumbel", "simulations": 20},
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "mcts" in data
        mcts = data["mcts"]
        assert mcts["total_sims"] >= 5  # Gumbel SH may use fewer sims
        assert len(mcts["visits"]) > 0


@needs_checkpoint
class TestCheckpointCache:
    def test_checkpoint_cache_lru(self, client):
        from hexo_rl.monitoring.analyze_api import _cache, _cache_lock, _MAX_CACHE

        # Clear cache
        with _cache_lock:
            _cache.clear()

        # Create symlinks to simulate multiple checkpoints
        with tempfile.TemporaryDirectory() as tmp:
            links = []
            for i in range(4):
                link = Path(tmp) / f"ckpt_{i}.pt"
                link.symlink_to(Path(TEST_CKPT).resolve())
                links.append(str(link))

            for link in links:
                resp = client.post("/api/analyze", json={
                    "checkpoint": link,
                    "moves": [],
                })
                assert resp.status_code == 200

            with _cache_lock:
                assert len(_cache) <= _MAX_CACHE

    def test_stale_checkpoint_reload(self, client):
        from hexo_rl.monitoring.analyze_api import _cache, _cache_lock

        with _cache_lock:
            _cache.clear()

        # Load once
        resp = client.post("/api/analyze", json={
            "checkpoint": TEST_CKPT,
            "moves": [],
        })
        assert resp.status_code == 200

        abs_path = str(Path(TEST_CKPT).resolve())
        with _cache_lock:
            assert abs_path in _cache
            old_mtime = _cache[abs_path]["mtime"]

        # Touch the file to make it "newer"
        future_time = time.time() + 100
        os.utime(TEST_CKPT, (future_time, future_time))
        try:
            resp = client.post("/api/analyze", json={
                "checkpoint": TEST_CKPT,
                "moves": [],
            })
            assert resp.status_code == 200

            with _cache_lock:
                assert _cache[abs_path]["mtime"] > old_mtime
        finally:
            # Restore mtime
            os.utime(TEST_CKPT, (old_mtime, old_mtime))


class TestErrorHandling:
    def test_invalid_checkpoint_returns_404(self, client):
        resp = client.post("/api/analyze", json={
            "checkpoint": "checkpoints/nonexistent_model.pt",
            "moves": [],
        })
        assert resp.status_code == 404
        data = resp.get_json()
        assert "error" in data

    def test_malformed_moves_returns_400(self, client):
        resp = client.post("/api/analyze", json={
            "checkpoint": TEST_CKPT or "checkpoints/x.pt",
            "moves": "not a list",
        })
        assert resp.status_code == 400

    def test_missing_checkpoint_returns_400(self, client):
        resp = client.post("/api/analyze", json={
            "moves": [],
        })
        assert resp.status_code == 400


class TestMonitoringInvariant:
    def test_no_trainer_import(self):
        """AST check that analyze_api.py does not import from hexo_rl.training or selfplay.pool."""
        src = Path("hexo_rl/monitoring/analyze_api.py").read_text()
        tree = ast.parse(src)
        forbidden = {"hexo_rl.training", "hexo_rl.selfplay.pool"}
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for prefix in forbidden:
                    if node.module == prefix or node.module.startswith(prefix + "."):
                        violations.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in forbidden:
                        if alias.name == prefix or alias.name.startswith(prefix + "."):
                            violations.append(alias.name)
        assert not violations, f"Forbidden imports found in analyze_api.py: {violations}"


class TestModelLoaderSync:
    """Verify model_loader.py stays in sync with Trainer's static methods."""

    def _make_fake_state_dict(self):
        """Build a minimal state_dict with enough keys to exercise inference."""
        import torch
        sd = {
            "trunk.input_conv.weight": torch.randn(128, 18, 3, 3),
            "policy_fc.weight": torch.randn(362, 2 * 19 * 19),
        }
        for i in range(6):
            sd[f"trunk.tower.{i}.conv1.weight"] = torch.randn(128, 128, 3, 3)
        return sd

    def test_extract_model_state_matches(self):
        from hexo_rl.training.trainer import Trainer
        from hexo_rl.viewer.model_loader import _extract_model_state

        sd = self._make_fake_state_dict()
        for payload in [
            {"model_state": sd},
            {"model_state_dict": sd},
            {"state_dict": sd},
            sd,  # weights-only
        ]:
            trainer_result = Trainer._extract_model_state(payload)
            loader_result = _extract_model_state(payload)
            assert trainer_result.keys() == loader_result.keys()

    def test_infer_model_hparams_matches(self):
        from hexo_rl.training.trainer import Trainer
        from hexo_rl.viewer.model_loader import _infer_model_hparams

        sd = self._make_fake_state_dict()
        trainer_hp = Trainer._infer_model_hparams(sd)
        loader_hp = _infer_model_hparams(sd)
        assert trainer_hp == loader_hp, f"Trainer={trainer_hp} vs loader={loader_hp}"
