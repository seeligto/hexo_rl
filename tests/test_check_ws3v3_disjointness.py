"""Tests for scripts/check_ws3v3_disjointness.py — synthetic corpora covering
PASS + each violation class."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import engine  # noqa: E402


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "check_ws3v3_disjointness", REPO_ROOT / "scripts" / "check_ws3v3_disjointness.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _seq(n: int):
    b = engine.Board.with_encoding_name("v6_live2_ls")
    seqout = []
    candidates = [(0, 0), (2, -1), (-2, 1), (3, -2), (-3, 2), (1, 3), (-1, -3), (4, 0), (-4, 0), (5, -1)]
    for q, r in candidates:
        if len(seqout) >= n:
            break
        if not b.check_win() and (q, r) in b.legal_moves():
            b.apply_move(q, r)
            seqout.append([q, r])
    return seqout


def _seq_alt(n: int):
    """A move sequence that does NOT share a prefix with _seq() (different
    opening move) — for constructing genuinely position-disjoint fixtures."""
    b = engine.Board.with_encoding_name("v6_live2_ls")
    seqout = []
    candidates = [(-1, 0), (6, -2), (-6, 2), (7, -3), (-7, 3), (2, 5), (-2, -5), (8, -1)]
    for q, r in candidates:
        if len(seqout) >= n:
            break
        if not b.check_win() and (q, r) in b.legal_moves():
            b.apply_move(q, r)
            seqout.append([q, r])
    return seqout


def _mk_seed(seed_id, source_game_idx, seed_moves, source_file="per_game_seald5.jsonl"):
    return {
        "seed_id": seed_id, "bucket": "seed", "source_file": source_file,
        "source_game_idx": source_game_idx, "parent_pos_id": "expand_gX",
        "cut": 0, "seed_moves": seed_moves, "mate_distance": 3, "in_window": True,
    }


def _mk_eval(pos_id, game_idx, parent_move_seq):
    return {
        "pos_id": pos_id, "source_game_id": pos_id, "game_idx": game_idx,
        "bucket": "expand", "encoding": "v6_live2_ls", "parent_move_seq": parent_move_seq,
        "saving_move": [0, 0], "blunder_move": [1, 1],
    }


def test_disjoint_corpora_pass_clean() -> None:
    m = _load_module()
    seeds = [_mk_seed("seed_g1_p4_k0", 1, _seq_alt(4))]
    evals = [_mk_eval("evalpos_g2", 2, _seq(6))]

    game_collisions = m.check_game_level(seeds, evals, m.DEFAULT_EVAL_SOURCE_FILE)
    dup, ns_collide, prefix_viol = m.check_position_level(seeds, evals)
    assert not game_collisions
    assert not dup
    assert not ns_collide
    assert not prefix_viol


def test_game_level_violation_same_source_game_idx() -> None:
    m = _load_module()
    seeds = [_mk_seed("seed_g7_p4_k0", 7, _seq(4))]
    evals = [_mk_eval("evalpos_g7", 7, _seq(6))]  # SAME game_idx as the seed
    collisions = m.check_game_level(seeds, evals, m.DEFAULT_EVAL_SOURCE_FILE)
    assert collisions == {("per_game_seald5.jsonl", 7)}


def test_position_level_prefix_violation() -> None:
    m = _load_module()
    full = _seq(6)
    seeds = [_mk_seed("seed_g9_p3_k3", 9, full[:3])]  # a strict prefix of the eval parent
    evals = [_mk_eval("evalpos_g20", 20, full)]  # disjoint game_idx, but prefix-colliding moves
    dup, ns_collide, prefix_viol = m.check_position_level(seeds, evals)
    assert not dup
    assert not ns_collide
    assert prefix_viol == [("seed_g9_p3_k3", "evalpos_g20")]


def test_position_level_equal_seed_and_eval_parent_is_a_violation() -> None:
    m = _load_module()
    full = _seq(5)
    seeds = [_mk_seed("seed_gX", 55, full)]
    evals = [_mk_eval("evalpos_gY", 66, full)]  # equal (not just prefix)
    _dup, _ns, prefix_viol = m.check_position_level(seeds, evals)
    assert prefix_viol == [("seed_gX", "evalpos_gY")]


def test_pos_id_namespace_collision() -> None:
    m = _load_module()
    seeds = [_mk_seed("shared_id_123", 1, _seq(4))]
    evals = [_mk_eval("shared_id_123", 2, _seq(6))]  # same string, different id space
    dup, ns_collide, _prefix = m.check_position_level(seeds, evals)
    assert not dup
    assert ns_collide == {"shared_id_123"}


def test_duplicate_seed_id_within_corpus() -> None:
    m = _load_module()
    seeds = [_mk_seed("dup_id", 1, _seq(4)), _mk_seed("dup_id", 2, _seq(3))]
    dup, _ns, _prefix = m.check_position_level(seeds, [])
    assert dup == ["dup_id"]


def test_leakage_distances_symmetric_board_is_max_jaccard_one() -> None:
    m = _load_module()
    full = _seq(5)
    seeds = [_mk_seed("seed_gZ", 1, full)]
    evals = [_mk_eval("evalpos_gW", 2, full)]  # identical terminal stone set
    dists = m.leakage_distances(seeds, evals, "v6_live2_ls")
    assert len(dists) == 1
    assert abs(dists[0] - 1.0) < 1e-9


def test_distribution_stats_shape() -> None:
    m = _load_module()
    stats = m.distribution_stats([0.1, 0.2, 0.3, 0.4, 0.5])
    assert stats["n"] == 5
    assert stats["max"] == 0.5
    assert stats["min"] == 0.1
    assert abs(stats["median"] - 0.3) < 1e-9
    assert stats["mean"] is not None

    empty = m.distribution_stats([])
    assert empty["n"] == 0
    assert empty["max"] is None


def _write_jsonl(path: Path, rows) -> None:
    import json
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_cli_exit_code_zero_on_clean_pass(tmp_path: Path) -> None:
    import subprocess
    seed_path = tmp_path / "seed_corpus.jsonl"
    eval_path = tmp_path / "heldout.jsonl"
    _write_jsonl(seed_path, [_mk_seed("seed_g1_p4_k0", 1, _seq_alt(4))])
    _write_jsonl(eval_path, [_mk_eval("evalpos_g2", 2, _seq(6))])

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_ws3v3_disjointness.py"),
         "--seed-corpus", str(seed_path), "--eval-traps", str(eval_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def test_cli_exit_code_nonzero_on_game_level_collision(tmp_path: Path) -> None:
    import subprocess
    seed_path = tmp_path / "seed_corpus.jsonl"
    eval_path = tmp_path / "heldout.jsonl"
    _write_jsonl(seed_path, [_mk_seed("seed_g7_p4_k0", 7, _seq(4))])
    _write_jsonl(eval_path, [_mk_eval("evalpos_g7", 7, _seq(6))])  # same game_idx

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_ws3v3_disjointness.py"),
         "--seed-corpus", str(seed_path), "--eval-traps", str(eval_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "VIOLATION" in result.stdout
