"""T7 — per-checkpoint E1 value-health CLI tests.

Frozen metrics M1-M4 on the 234-probe (docs/designs/e1_metric_freeze.md).
These unit tests use TINY synthetic nets + a 3-loss / 3-safe mini fixture
(derived from a synthetic source game so zobrist reconstruction matches);
they never touch the real 234-probe or 49MB checkpoints.

Run: .venv/bin/python -m pytest scripts/e1/tests/ -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hexo_rl.eval.eval_board import make_eval_board
from hexo_rl.model.network import HexTacToeNet

from scripts.e1.validate_ckpt import (
    ROW_KEYS,
    DEFAULT_PROBE,
    DEFAULT_NEGATIVES,
    _PROBE_SHA256,
    _NEGATIVES_SHA256,
    validate_ckpt,
)

ENCODING = "v6_live2_ls"
RADIUS = 4
FAST_FILTERS = 16
FAST_BLOCKS = 1


# ── synthetic net + checkpoint helpers ───────────────────────────────────────


def _make_net(value_head_type: str = "scalar") -> HexTacToeNet:
    torch.manual_seed(0)
    return HexTacToeNet(
        filters=FAST_FILTERS,
        res_blocks=FAST_BLOCKS,
        encoding=ENCODING,
        value_head_type=value_head_type,
    ).eval()


def _save_ckpt(path: Path, net: HexTacToeNet, step: int = 5000) -> Path:
    payload = {
        "model_state": net.state_dict(),
        "metadata": {"encoding_name": ENCODING, "schema_version": 1},
        "step": step,
    }
    torch.save(payload, path)
    return path


# ── synthetic source-game + probe/negatives fixture ──────────────────────────


def _build_game_and_rows(tmp_path: Path):
    """Build ONE synthetic game; derive 3 loss + 3 safe probe rows whose
    (opening_idx, head_as_p1, t, zobrist) reconstruct exactly to real boards.

    Returns (games_path, probe_path, negatives_path).
    """
    board = make_eval_board(ENCODING, RADIUS)
    moves = []
    zob_at_t = {}
    for t in range(8):
        zob_at_t[t] = str(board.zobrist_hash())
        legal = board.legal_moves()
        q, r = legal[t % len(legal)]
        moves.append([int(q), int(r)])
        board.apply_move(int(q), int(r))

    game = {
        "opening_idx": 0,
        "head_as_p1": True,
        "radius": RADIUS,
        "moves": moves,
        "winner": "p1",
    }
    games_path = tmp_path / "games.jsonl"
    games_path.write_text(json.dumps(game) + "\n")

    def _row(t: int, s: str) -> dict:
        return {
            "opening_idx": 0,
            "head_as_p1": True,
            "t": t,
            "zobrist": zob_at_t[t],
            "set": s,
            "wp": "WP1" if s == "loss" else "NEG",
            "v_raw": 0.1,
        }

    # 3 loss positions (t=1,3,5), 3 safe positions (t=2,4,6).
    probe_rows = [_row(1, "loss"), _row(3, "loss"), _row(5, "loss")]
    safe_rows = [_row(2, "safe"), _row(4, "safe"), _row(6, "safe")]
    probe_path = tmp_path / "probe.jsonl"
    negatives_path = tmp_path / "negatives.jsonl"
    probe_path.write_text("\n".join(json.dumps(r) for r in probe_rows) + "\n")
    negatives_path.write_text("\n".join(json.dumps(r) for r in safe_rows) + "\n")
    return games_path, probe_path, negatives_path


# ── tests ─────────────────────────────────────────────────────────────────────


def test_scalar_row_schema_and_metrics(tmp_path: Path):
    games_path, probe_path, negatives_path = _build_game_and_rows(tmp_path)
    net = _make_net("scalar")
    ckpt = _save_ckpt(tmp_path / "scalar.pt", net)
    out = tmp_path / "series.jsonl"

    row = validate_ckpt(
        str(ckpt), "scalar", str(out),
        probe_path=str(probe_path), negatives_path=str(negatives_path),
        games_path=str(games_path),
    )

    # All schema keys present.
    assert set(row.keys()) == set(ROW_KEYS)
    assert row["arm"] == "scalar"
    assert row["encoding"] == ENCODING
    assert row["n_loss"] == 3
    assert row["n_safe"] == 3

    # M3: scalar arm populates decoded_auc, tail_mass_auc is null.
    assert row["decoded_auc"] is not None
    assert row["tail_mass_auc"] is None

    # ECE in [0,1].
    assert 0.0 <= row["ece"] <= 1.0

    # M1 = mean decoded-v over the 3 loss positions (scalar = min-pool tanh).
    dev = torch.device("cpu")
    from scripts.e1.validate_ckpt import _score_positions, _load_net
    loaded, spec, _ = _load_net(str(ckpt), "scalar", dev)
    loss_scores, _ = _score_positions(
        loaded, spec, dev, _load_rows(probe_path), _index_games(games_path),
    )
    expected_m1 = sum(s["v"] for s in loss_scores) / len(loss_scores)
    assert abs(row["mean_v_on_losses"] - expected_m1) < 1e-6

    # M4 = fraction of safe controls with decoded-v <= -0.5.
    safe_scores, _ = _score_positions(
        loaded, spec, dev, _load_rows(negatives_path), _index_games(games_path),
    )
    expected_fp = sum(1 for s in safe_scores if s["v"] <= -0.5) / len(safe_scores)
    assert abs(row["false_pessimism"] - expected_fp) < 1e-9

    # recognition_lag_mean_v_on_losses must be null (harness not wired).
    assert row["recognition_lag_mean_v_on_losses"] is None
    assert row["recognition_lag_note"] == "harness not wired; use mean_v_on_losses"

    # One row appended to out.
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["arm"] == "scalar"


def test_dist_row_populates_tail_mass_auc(tmp_path: Path):
    games_path, probe_path, negatives_path = _build_game_and_rows(tmp_path)
    net = _make_net("dist65")
    ckpt = _save_ckpt(tmp_path / "dist65.pt", net)
    out = tmp_path / "series.jsonl"

    row = validate_ckpt(
        str(ckpt), "dist65", str(out),
        probe_path=str(probe_path), negatives_path=str(negatives_path),
        games_path=str(games_path),
    )
    assert set(row.keys()) == set(ROW_KEYS)
    assert row["arm"] == "dist65"
    # M3: dist arm populates tail_mass_auc, decoded_auc is null.
    assert row["tail_mass_auc"] is not None
    assert row["decoded_auc"] is None
    assert 0.0 <= row["ece"] <= 1.0
    assert -1.0 <= row["mean_v_on_losses"] <= 1.0
    assert 0.0 <= row["false_pessimism"] <= 1.0


def test_determinism_same_ckpt_byte_identical_row(tmp_path: Path):
    games_path, probe_path, negatives_path = _build_game_and_rows(tmp_path)
    net = _make_net("dist65")
    ckpt = _save_ckpt(tmp_path / "dist65.pt", net)
    out1 = tmp_path / "a.jsonl"
    out2 = tmp_path / "b.jsonl"
    kw = dict(
        probe_path=str(probe_path), negatives_path=str(negatives_path),
        games_path=str(games_path),
    )
    r1 = validate_ckpt(str(ckpt), "dist65", str(out1), **kw)
    r2 = validate_ckpt(str(ckpt), "dist65", str(out2), **kw)
    assert r1 == r2


def test_arm_mismatch_dist_ckpt_scored_as_scalar_is_caught(tmp_path: Path):
    """A dist65 checkpoint declared arm='scalar' must be caught, not silently
    mis-decoded (the gated loader detects value_fc2_bins from the state dict)."""
    games_path, probe_path, negatives_path = _build_game_and_rows(tmp_path)
    net = _make_net("dist65")
    ckpt = _save_ckpt(tmp_path / "dist65.pt", net)
    out = tmp_path / "series.jsonl"
    with pytest.raises(ValueError, match="value_head_type"):
        validate_ckpt(
            str(ckpt), "scalar", str(out),
            probe_path=str(probe_path), negatives_path=str(negatives_path),
            games_path=str(games_path),
        )


def test_arm_mismatch_scalar_ckpt_scored_as_dist_is_caught(tmp_path: Path):
    games_path, probe_path, negatives_path = _build_game_and_rows(tmp_path)
    net = _make_net("scalar")
    ckpt = _save_ckpt(tmp_path / "scalar.pt", net)
    out = tmp_path / "series.jsonl"
    with pytest.raises(ValueError, match="value_head_type"):
        validate_ckpt(
            str(ckpt), "dist65", str(out),
            probe_path=str(probe_path), negatives_path=str(negatives_path),
            games_path=str(games_path),
        )


# ── probe SHA guard ───────────────────────────────────────────────────────────


def test_sha_guard_tampered_default_probe_raises(tmp_path: Path, monkeypatch):
    """When DEFAULT_PROBE is re-pointed to a wrong-SHA file, validate_ckpt halts."""
    import scripts.e1.validate_ckpt as _vc

    games_path, probe_path, negatives_path = _build_game_and_rows(tmp_path)
    net = _make_net("scalar")
    ckpt = _save_ckpt(tmp_path / "scalar.pt", net)
    out = tmp_path / "series.jsonl"

    # Tamper: write valid JSON but wrong content -> wrong SHA.
    tampered = tmp_path / "tampered_probe.jsonl"
    tampered.write_text(probe_path.read_text() + json.dumps({"extra": "row"}) + "\n")

    # Point DEFAULT_PROBE + DEFAULT_NEGATIVES at files that look like defaults
    # (same path strings) but with a wrong SHA for probe.
    monkeypatch.setattr(_vc, "DEFAULT_PROBE", str(tampered))
    monkeypatch.setattr(_vc, "DEFAULT_NEGATIVES", str(tampered))

    with pytest.raises(RuntimeError, match="SHA mismatch"):
        validate_ckpt(
            str(ckpt), "scalar", str(out),
            # Pass the monkeypatched DEFAULT values so the guard fires.
            probe_path=str(tampered),
            negatives_path=str(tampered),
            games_path=str(games_path),
        )


def test_sha_guard_custom_path_bypasses(tmp_path: Path):
    """Custom --probe/--negatives paths bypass the SHA guard (no raise)."""
    games_path, probe_path, negatives_path = _build_game_and_rows(tmp_path)
    net = _make_net("scalar")
    ckpt = _save_ckpt(tmp_path / "scalar.pt", net)
    out = tmp_path / "series.jsonl"

    # Custom paths (neither equals DEFAULT_PROBE nor DEFAULT_NEGATIVES) — guard
    # is unconditionally skipped for non-default paths regardless of SHA.
    row = validate_ckpt(
        str(ckpt), "scalar", str(out),
        probe_path=str(probe_path),
        negatives_path=str(negatives_path),
        games_path=str(games_path),
    )
    assert set(row.keys()) == set(ROW_KEYS)
    assert row["n_loss"] == 3


def test_unresolved_loss_set_raises(tmp_path: Path):
    """All loss rows unresolvable (empty games index) -> RuntimeError."""
    # Build probe rows that reference opening_idx=0, but provide an EMPTY games
    # index so resolve_game returns None for all rows -> loss_scores = [].
    games_path, probe_path, negatives_path = _build_game_and_rows(tmp_path)
    net = _make_net("scalar")
    ckpt = _save_ckpt(tmp_path / "scalar.pt", net)
    out = tmp_path / "series.jsonl"

    # Empty games file -> every row skipped.
    empty_games = tmp_path / "empty_games.jsonl"
    empty_games.write_text("")

    with pytest.raises(RuntimeError, match="no loss positions resolved"):
        validate_ckpt(
            str(ckpt), "scalar", str(out),
            probe_path=str(probe_path),
            negatives_path=str(negatives_path),
            games_path=str(empty_games),
        )


# ── slow smoke against the real probe + a real checkpoint (skip if absent) ────

_REPO = Path(__file__).resolve().parents[3]
_REAL_PROBE = _REPO / "reports/valprobe/probe_set_v1.jsonl"
_REAL_NEG = _REPO / "reports/valprobe/negatives_v1.jsonl"
_REAL_GAMES = _REPO / "reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl"
_REAL_CKPT = _REPO / "checkpoints/run2_retro/checkpoint_00248000.pt"


@pytest.mark.slow
def test_slow_smoke_real_probe_scalar(tmp_path: Path):
    for p in (_REAL_PROBE, _REAL_NEG, _REAL_GAMES, _REAL_CKPT):
        if not p.exists():
            pytest.skip(f"missing real artifact {p}")
    # 193/234 loss positions are WP2 (regen games). If the WP2 regen games are
    # absent (operator-run), full-234 is unavailable — assert against the
    # resolvable subset (WP1 + all 651 safe) instead of skipping the whole read.
    from scripts.e1.validate_ckpt import _default_wp2_games

    wp2 = _default_wp2_games()
    out = tmp_path / "series.jsonl"
    row = validate_ckpt(
        str(_REAL_CKPT), "scalar", str(out),
        probe_path=str(_REAL_PROBE), negatives_path=str(_REAL_NEG),
        games_path=str(_REAL_GAMES),
    )
    assert set(row.keys()) == set(ROW_KEYS)
    assert row["n_safe"] == 651
    if len(wp2) == 5:
        assert row["n_loss"] == 234
    else:
        # WP1-only (41) resolvable without WP2 regen games.
        assert row["n_loss"] >= 41
    assert 0.0 <= row["ece"] <= 1.0
    assert -1.0 <= row["mean_v_on_losses"] <= 1.0
    assert row["decoded_auc"] is not None and row["tail_mass_auc"] is None


# ── local helpers mirroring validate_ckpt internals (test-only) ──────────────


def _load_rows(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _index_games(path: Path):
    from scripts.e1.validate_ckpt import index_games
    return index_games(str(path))
