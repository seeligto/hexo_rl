"""GNN-integration WP-5b commit B (P8) — corpus export tests.

Covers the delta doc §12 export test-plan row: replay a tiny fixture game
set -> export .hexg -> a lightweight export-round-trip parity smoke (the
NEW logic this commit adds — stone/player/move-target extraction from
`bc_data.replay_positions` into a push-argument row — argmax-cell ==
played-move + `builder_impl==1` + the full 18-assertion collate contract
passes); every record satisfies the push guards (implicit: push succeeds);
save_to_path -> load_from_path conserves count. ADV: a held-out game in the
input -> hard-fail; a games-manifest mismatch -> hard-fail; a malformed
game -> LOUD-skip-with-count; the output sha never a held-out sha.

Scoping note (OQ-4): the exhaustive builder byte-parity sweep (Rust
builder vs the `build_axis_graph_raw` Python oracle over >=1000 positions)
already lives in `tests/test_hexo_graph_parity.py` (WP-1) and
`sample_wire_matches_direct_builder_unaugmented`
(`engine/src/replay_buffer/hexg/tests.rs`, WP-5a) — both prove the BUILDER
itself. This file's parity check targets the NEW surface commit B adds (the
export's stone/target extraction), not a re-proof of the builder.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import pytest

torch = pytest.importorskip("torch")

from engine import Board, HexgBuffer  # noqa: E402
from hexo_rl.selfplay.graph_collate import collate_graph_batch  # noqa: E402
from scripts.export_gnn_hexg_corpus import (  # noqa: E402
    ExportStats,
    games_manifest_sha256,
    load_qualifying_records,
    run,
    select_candidate_paths,
    sha256_file,
)

ENC = "gnn_axis_v1"


# ── Synthetic legal-game generator (deterministic, no RNG) ───────────────────
#
# Random legal play does not reliably terminate (the board is theoretically
# infinite, six-in-a-row is rare by chance) — construct a P1 straight-6 win
# deliberately: P1's opening move + `pad_dummy_turns` wasted P1 turns (padding
# the game length past MIN_GAME_LENGTH=15 / moveCount>=20) before P1 starts
# claiming the win-axis cells; P2 always plays a legal cell off that axis.

def _build_synthetic_game(pad_dummy_turns: int = 3) -> Tuple[List[Tuple[int, int]], List[int], int]:
    b = Board()
    p1_targets = [(i, 0) for i in range(6)]
    moves: List[Tuple[int, int]] = []
    players: List[int] = []

    move0 = p1_targets[0]
    moves.append(move0)
    players.append(1)
    b.apply_move(*move0)
    placed_p1 = 1
    dummy_p1_turns_done = 0
    winner = None
    for _ in range(200):
        if b.winner() is not None:
            winner = b.winner()
            break
        legal = set(b.legal_moves())
        if not legal:
            break
        player = b.current_player
        n_this_turn = b.moves_remaining
        for _ in range(n_this_turn):
            legal = set(b.legal_moves())
            if not legal:
                break
            if player == 1 and dummy_p1_turns_done < pad_dummy_turns:
                move = next(m for m in legal if m not in p1_targets)
            elif player == 1 and placed_p1 < len(p1_targets):
                want = p1_targets[placed_p1]
                if want in legal:
                    move = want
                    placed_p1 += 1
                else:
                    move = next((m for m in legal if m not in p1_targets), next(iter(legal)))
            else:
                move = next((m for m in legal if m not in p1_targets), next(iter(legal)))
            moves.append(move)
            players.append(player)
            b.apply_move(*move)
            w = b.winner()
            if w is not None:
                winner = w
                break
        if player == 1 and dummy_p1_turns_done < pad_dummy_turns:
            dummy_p1_turns_done += 1
        if winner is not None:
            break
    assert winner is not None, "synthetic game generator failed to terminate"
    return moves, players, winner


def _write_game_json(
    tmp_path: Path, uuid: str, moves: List[Tuple[int, int]], players: List[int], winner: int,
    *, p1_id: str = "p1", p2_id: str = "p2", mtime: "float | None" = None,
) -> Path:
    move_entries = [
        {"moveNumber": i + 2, "playerId": (p1_id if pl == 1 else p2_id), "x": int(q), "y": int(r), "timestamp": i}
        for i, ((q, r), pl) in enumerate(zip(moves, players))
    ]
    data = {
        "id": uuid,
        "players": [
            {"playerId": p1_id, "displayName": "A", "elo": 1500},
            {"playerId": p2_id, "displayName": "B", "elo": 1500},
        ],
        "gameOptions": {"rated": True},
        "moveCount": len(moves),
        "gameResult": {"winningPlayerId": (p1_id if winner == 1 else p2_id), "reason": "six-in-a-row"},
        "moves": move_entries,
    }
    path = tmp_path / f"{uuid}.json"
    path.write_text(json.dumps(data))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _corrupt_game_json(tmp_path: Path, uuid: str, *, mtime: "float | None" = None) -> Path:
    """A game whose move list occupies the SAME cell twice — `apply_move`
    raises mid-replay (`ValueError: cell already occupied`), the exact
    per-game oddity class §5.4 names."""
    moves = [(0, 0), (1, 0), (0, 0), (2, 0)] + [(i, 1) for i in range(3, 20)]
    players = [1, -1, 1, -1] + [1 if i % 2 == 0 else -1 for i in range(len(moves) - 4)]
    return _write_game_json(
        tmp_path, uuid, moves, players, winner=1, mtime=mtime,
    )


def _illegal_last_ply_game() -> Tuple[List[Tuple[int, int]], List[int], int]:
    """WP5b commit-B red-team BREAK-2 fixture: a LEGAL move-list (from
    `_build_synthetic_game`) with ONE extra move appended at the very end —
    a duplicate of the first move's cell (already occupied on the board,
    illegal). `replay_positions` yields-then-applies, so this illegal
    move's position (stones from every prior ply, `move` = the duplicate
    cell) IS yielded before `apply_move` raises; because it lands at
    exactly the last in-window ply, `replay_game`'s
    `positions[-1].ply < expected_last_ply` truncation proxy does NOT fire
    (`positions[-1].ply == expected_last_ply` — off-by-one blind spot).
    Distinct from `_corrupt_game_json`, whose duplicate move sits mid-game
    (well before the last ply) and IS caught by that proxy today."""
    moves, players, winner = _build_synthetic_game()
    dup_cell = moves[0]
    moves = moves + [dup_cell]
    players = players + [players[-1]]
    assert len(moves) >= 20, "fixture must still clear the moveCount>=20 ingestion filter"
    return moves, players, winner


_TWO_UUIDS = (
    "aaaaaaaa-0000-0000-0000-000000000001",
    "aaaaaaaa-0000-0000-0000-000000000002",
)


def _make_two_games(tmp_path: Path, mtime: float) -> Path:
    raw_dir = tmp_path / "raw_human"
    raw_dir.mkdir()
    for uuid in _TWO_UUIDS:
        moves, players, winner = _build_synthetic_game()
        _write_game_json(raw_dir, uuid, moves, players, winner, mtime=mtime)
    return raw_dir


def _expected_sha_for(raw_dir: Path, cutoff_ts: float) -> str:
    stats = ExportStats()
    candidates = select_candidate_paths(raw_dir, cutoff_ts)
    records = load_qualifying_records(candidates, raw_dir, stats)
    return games_manifest_sha256(records)


# ─────────────────────────────────────────────────────────────────────────────


def test_happy_path_export_two_games(tmp_path):
    cutoff_ts = time.time() + 3600  # future cutoff — every fixture file qualifies
    raw_dir = _make_two_games(tmp_path, mtime=time.time() - 100)
    expected_sha = _expected_sha_for(raw_dir, cutoff_ts)
    out_path = tmp_path / "export.hexg"

    stats = run(
        raw_dir, out_path, expected_sha,
        heldout_dir=tmp_path / "no_such_heldout_dir", cutoff=_iso(cutoff_ts),
    )

    assert stats.n_games_exported == 2
    assert stats.n_games_skipped == 0
    assert stats.n_positions_exported > 0
    assert stats.games_manifest_sha256 == expected_sha
    assert out_path.exists()
    assert stats.output_sha256 == sha256_file(out_path)


def test_save_load_roundtrip_conserves_count(tmp_path):
    cutoff_ts = time.time() + 3600
    raw_dir = _make_two_games(tmp_path, mtime=time.time() - 100)
    expected_sha = _expected_sha_for(raw_dir, cutoff_ts)
    out_path = tmp_path / "export.hexg"
    stats = run(raw_dir, out_path, expected_sha,
                heldout_dir=tmp_path / "no_such_heldout_dir", cutoff=_iso(cutoff_ts))

    reloaded = HexgBuffer(capacity=max(1, stats.n_positions_exported), encoding=ENC)
    n_loaded = reloaded.load_from_path(str(out_path))
    assert n_loaded == stats.n_positions_exported
    assert reloaded.size == stats.n_positions_exported


def test_export_round_trip_native_builder_parity(tmp_path):
    """Every exported record: builder_impl==1 (F7, native rebuild), full
    18-assertion collate passes, and the policy-target argmax cell equals
    the actual human-played move (the export's OWN new extraction logic —
    proves the push-argument row was built correctly, not a re-proof of the
    builder itself, see module docstring)."""
    cutoff_ts = time.time() + 3600
    raw_dir = _make_two_games(tmp_path, mtime=time.time() - 100)
    expected_sha = _expected_sha_for(raw_dir, cutoff_ts)
    out_path = tmp_path / "export.hexg"
    stats = run(raw_dir, out_path, expected_sha,
                heldout_dir=tmp_path / "no_such_heldout_dir", cutoff=_iso(cutoff_ts))

    buf = HexgBuffer(capacity=max(1, stats.n_positions_exported), encoding=ENC)
    buf.load_from_path(str(out_path))
    assert buf.size == stats.n_positions_exported

    wire, tg = buf.sample_graph_batch(buf.size, augment=False)
    assert int(wire.builder_impl) == 1, "F7 — sampled wire must be native-built"
    assert int(wire.contract_version) == 1

    batch = collate_graph_batch(
        wire, device="cpu", semantic="full", target_argmax_cells=tg.target_argmax_cells,
    )
    assert batch.n_graphs == buf.size

    # Every row's target-argmax cell must be a legal node AND (since every
    # exported row is a one-hot human-move target) the argmax must be valid.
    argmax_cells = tg.target_argmax_cells
    assert len(argmax_cells) == buf.size
    assert all(c is not None for c in argmax_cells), "every one-hot human-move row must have a valid argmax"


def test_malformed_game_is_loud_skipped_with_count(tmp_path):
    cutoff_ts = time.time() + 3600
    raw_dir = tmp_path / "raw_human"
    raw_dir.mkdir()
    moves, players, winner = _build_synthetic_game()
    _write_game_json(raw_dir, _TWO_UUIDS[0], moves, players, winner, mtime=time.time() - 100)
    _corrupt_game_json(raw_dir, _TWO_UUIDS[1], mtime=time.time() - 100)

    expected_sha = _expected_sha_for(raw_dir, cutoff_ts)
    out_path = tmp_path / "export.hexg"
    stats = run(raw_dir, out_path, expected_sha,
                heldout_dir=tmp_path / "no_such_heldout_dir", cutoff=_iso(cutoff_ts))

    assert stats.n_games_exported == 1, "the corrupt game must be skipped, not exported"
    assert stats.n_games_skipped == 1
    assert any(k.startswith("replay_error:") for k in stats.skip_reasons)


def test_illegal_move_at_last_in_window_ply_is_gated_and_skip_counted(tmp_path):
    """BREAK-2 (WP5b commit-B red-team, confirmed reproduction): an illegal
    move at exactly the last in-window ply used to slip the truncation
    proxy — a poisoned row (a one-hot visit target on an OCCUPIED/illegal
    cell) got exported AND `n_games_skipped` stayed 0 (the skip-count
    integrity violation — the export summary lied). Post-fix: the whole
    game is LOUD-skipped (per-row visit-cell legality gate in
    `replay_game`), the count is honest, and the export still completes
    clean with only the good game's rows in the persisted artifact."""
    cutoff_ts = time.time() + 3600
    raw_dir = tmp_path / "raw_human"
    raw_dir.mkdir()
    good_moves, good_players, good_winner = _build_synthetic_game()
    _write_game_json(raw_dir, _TWO_UUIDS[0], good_moves, good_players, good_winner,
                      mtime=time.time() - 100)
    bad_moves, bad_players, bad_winner = _illegal_last_ply_game()
    _write_game_json(raw_dir, _TWO_UUIDS[1], bad_moves, bad_players, bad_winner,
                      mtime=time.time() - 100)

    expected_sha = _expected_sha_for(raw_dir, cutoff_ts)
    out_path = tmp_path / "export.hexg"
    stats = run(raw_dir, out_path, expected_sha,
                heldout_dir=tmp_path / "no_such_heldout_dir", cutoff=_iso(cutoff_ts))

    # The honest count: exactly 1 game exported (the good one), exactly 1
    # skipped (the illegal-last-ply one) — pre-fix this was 2 exported / 0
    # skipped (the poisoned row silently made it into the buffer, no count).
    assert stats.n_games_exported == 1, "the illegal-last-ply game must NOT be exported"
    assert stats.n_games_skipped == 1, "skip-count must be honest (BREAK-2: this stayed 0 pre-fix)"
    assert any(k.startswith("replay_error:") for k in stats.skip_reasons)

    # No poisoned row reached the persisted artifact: round-trip through the
    # same buffer/collate contract `test_export_round_trip_native_builder_
    # parity` exercises — clean (no sample-time mass-drop guard needed,
    # because the bad row never got pushed in the first place).
    buf = HexgBuffer(capacity=max(1, stats.n_positions_exported), encoding=ENC)
    buf.load_from_path(str(out_path))
    assert buf.size == stats.n_positions_exported
    wire, tg = buf.sample_graph_batch(buf.size, augment=False)
    assert int(wire.builder_impl) == 1
    batch = collate_graph_batch(
        wire, device="cpu", semantic="full", target_argmax_cells=tg.target_argmax_cells,
    )
    assert batch.n_graphs == buf.size


def test_heldout_game_hard_fails(tmp_path):
    cutoff_ts = time.time() + 3600
    raw_dir = _make_two_games(tmp_path, mtime=time.time() - 100)
    # heldout_dir contains a BYTE-IDENTICAL copy of one of the input games —
    # same move sequence -> same _game_hash -> must hard-fail.
    heldout_dir = tmp_path / "heldout_s5"
    heldout_dir.mkdir()
    src_json = (raw_dir / f"{_TWO_UUIDS[0]}.json").read_text()
    (heldout_dir / f"{_TWO_UUIDS[0]}.json").write_text(src_json)

    expected_sha = _expected_sha_for(raw_dir, cutoff_ts)
    out_path = tmp_path / "export.hexg"
    with pytest.raises(ValueError, match="HELD-OUT"):
        run(raw_dir, out_path, expected_sha, heldout_dir=heldout_dir, cutoff=_iso(cutoff_ts))


def test_games_manifest_mismatch_hard_fails(tmp_path):
    cutoff_ts = time.time() + 3600
    raw_dir = _make_two_games(tmp_path, mtime=time.time() - 100)
    out_path = tmp_path / "export.hexg"
    with pytest.raises(ValueError, match="games-manifest sha mismatch"):
        run(raw_dir, out_path, "0" * 64,
            heldout_dir=tmp_path / "no_such_heldout_dir", cutoff=_iso(cutoff_ts))


def test_output_sha_hard_fails_when_it_collides_with_a_registered_heldout_sha(tmp_path, monkeypatch):
    cutoff_ts = time.time() + 3600
    raw_dir = _make_two_games(tmp_path, mtime=time.time() - 100)
    expected_sha = _expected_sha_for(raw_dir, cutoff_ts)
    out_path = tmp_path / "export.hexg"

    # First export legitimately to learn the real output sha, then re-run
    # with `held_out_shas` monkeypatched to include it — proves the output-
    # site gate (§5.2) actually fires on a collision.
    stats = run(raw_dir, out_path, expected_sha,
                heldout_dir=tmp_path / "no_such_heldout_dir", cutoff=_iso(cutoff_ts))
    real_output_sha = stats.output_sha256

    import scripts.export_gnn_hexg_corpus as export_mod
    monkeypatch.setattr(export_mod, "held_out_shas", lambda: frozenset({real_output_sha}))
    with pytest.raises(ValueError, match="HELD-OUT sha"):
        run(raw_dir, out_path, expected_sha,
            heldout_dir=tmp_path / "no_such_heldout_dir", cutoff=_iso(cutoff_ts))


def test_game_id_minted_via_next_game_id_monotonic_across_games(tmp_path):
    """§6 binding: one `HexgBuffer.next_game_id()` per game, monotonic, never
    a script-local atomic that could reset/collide."""
    cutoff_ts = time.time() + 3600
    raw_dir = _make_two_games(tmp_path, mtime=time.time() - 100)
    expected_sha = _expected_sha_for(raw_dir, cutoff_ts)
    out_path = tmp_path / "export.hexg"
    stats = run(raw_dir, out_path, expected_sha,
                heldout_dir=tmp_path / "no_such_heldout_dir", cutoff=_iso(cutoff_ts))

    buf = HexgBuffer(capacity=max(1, stats.n_positions_exported), encoding=ENC)
    buf.load_from_path(str(out_path))
    # A fresh buffer's next_game_id must be rebased past every loaded game_id
    # (WP-5a `load_rebases_next_game_id_past_loaded_max` — Rust-side pin;
    # this is the Python-boundary regression proof for the export's own
    # minted ids specifically).
    fresh_id = buf.next_game_id()
    wire, tg = buf.sample_graph_batch(buf.size, augment=False)
    assert fresh_id >= stats.n_games_exported


def _iso(ts: float) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")
