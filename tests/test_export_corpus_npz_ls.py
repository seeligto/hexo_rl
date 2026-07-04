"""Tests for the v6_live2_ls per-cluster-row emission path (D-RUN2 Decision 7).

Covers `hexo_rl.bootstrap.dataset.replay_game_to_triples_ls` (the legal-set
scatter replayer — dmulticluster_362_legalset_design.md §4 Tier-2 variant-(b))
and the `scripts/export_corpus_npz.py --encoding v6_live2_ls` end-to-end build
from a portable JSONL.

Key semantic under test vs the v6_live2 path: the played move is scattered
across ALL containing cluster windows (one dense-362 one-hot row per window),
instead of the first containing window only — off-window mass is NOT dropped.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from hexo_rl.bootstrap.dataset import (
    replay_game_to_triples,
    replay_game_to_triples_ls,
)
from hexo_rl.encoding import lookup as lookup_encoding

ROOT = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "export_corpus_npz", ROOT / "scripts" / "export_corpus_npz.py",
)
assert _SPEC is not None and _SPEC.loader is not None
npz_exporter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(npz_exporter)

_LS = lookup_encoding("v6_live2_ls")
_KEPT = list(_LS.kept_plane_indices)


def _two_colony_game() -> list[tuple[int, int]]:
    """Two colonies ~14 apart (> cluster_threshold=5 → 2 clusters/windows);
    the move at (7, 0) lies within both 19×19 windows (dist ≤ 9 from both
    centroids) → must scatter into 2 rows under _ls."""
    return [(0, 0),                       # P1 opener
            (14, 0), (15, 0),             # P2 founds colony B (outside A's window at ply 1)
            (1, 0), (2, 0),               # P1 grows A
            (16, 0), (17, 0),             # P2 grows B
            (7, 0), (3, 0),               # P1 — (7,0) contained in BOTH windows
            (18, 0), (19, 0),             # P2
            (4, 0), (5, 0),               # P1
            (20, 0), (21, 0)]             # P2


def _replay_ls(moves, winner=1):
    return replay_game_to_triples_ls(
        moves, winner,
        kept_plane_indices=_KEPT,
        policy_size=_LS.policy_logit_count,
        k_max=_LS.k_max,
    )


def test_ls_scatters_move_across_all_containing_windows():
    moves = _two_colony_game()
    s, p, o, ply = _replay_ls(moves)
    # shapes: registry-driven, one-hot rows
    assert s.shape[1:] == (len(_KEPT), 19, 19)
    assert p.shape[1] == _LS.policy_logit_count
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    # ply 7 = (7,0), contained in both cluster windows → exactly 2 rows
    rows7 = np.where(ply == 7)[0]
    assert len(rows7) == 2
    cells = sorted(int(p[r].argmax()) for r in rows7)
    # different window frames → different LOCAL cells for the same move
    assert cells[0] != cells[1]
    # the two rows encode DIFFERENT window views of the same position
    assert not np.array_equal(s[rows7[0]], s[rows7[1]])
    # outcome identical across scatter rows of one ply
    assert o[rows7[0]] == o[rows7[1]]


def test_ls_vs_v6_live2_first_window_rows_agree_and_extra_mass_exists():
    """The _ls first-containing-window row per ply must reproduce the v6_live2
    target; the EXTRA rows are exactly the mass v6_live2 zeroed."""
    moves = _two_colony_game()
    s_ls, p_ls, o_ls, ply = _replay_ls(moves)
    s_v6, _, p_v6, o_v6 = replay_game_to_triples(moves, 1)

    # v6 path emits one row per emitted ply, in emission order; _ls rows for a
    # ply are consecutive with the first-containing window first.
    emitted_plies = sorted(set(ply.tolist()))
    assert len(emitted_plies) == len(p_v6)
    for row_v6, pi in enumerate(emitted_plies):
        first_ls_row = int(np.where(ply == pi)[0][0])
        assert int(p_ls[first_ls_row].argmax()) == int(p_v6[row_v6].argmax())
        # states agree too (v6 row sliced to the _ls kept planes)
        np.testing.assert_array_equal(
            s_ls[first_ls_row], s_v6[row_v6][_KEPT]
        )
    # scatter produced strictly more rows than the v6 drop semantics
    assert len(p_ls) > len(p_v6)
    # and every extra row's target cell carries mass v6_live2 never emitted
    # for that ply (different window frame → different local cell)
    rows7 = np.where(ply == 7)[0]
    v6_row7 = emitted_plies.index(7)
    extra_cells = {int(p_ls[r].argmax()) for r in rows7[1:]}
    assert int(p_v6[v6_row7].argmax()) not in extra_cells


def test_ls_drops_ply_outside_all_windows():
    """A move outside EVERY cluster window has no representable dense target —
    skipped on both paths (ply 1 of the two-colony game: (14,0) with only
    (0,0) on the board)."""
    moves = _two_colony_game()
    _, _, _, ply = _replay_ls(moves)
    assert 1 not in set(ply.tolist())


def test_ls_end_to_end_from_jsonl(tmp_path: Path):
    """Full exporter run: --from-jsonl --encoding v6_live2_ls builds an NPZ
    with registry shapes + sidecar metadata + scatter telemetry."""
    moves = _two_colony_game()
    # pad to MIN_GAME_LENGTH (15) — already 15 plies
    line = json.dumps({"game_hash": "g", "moves": [list(m) for m in moves],
                       "winner": 1, "source": "human", "elo": [1300, 1300]})
    jsonl = tmp_path / "c.jsonl"
    jsonl.write_text(line + "\n")
    out = tmp_path / "ls.npz"
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_corpus_npz.py"),
         "--from-jsonl", str(jsonl), "--encoding", "v6_live2_ls",
         "--human-only", "--no-compress", "--out", str(out)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, res.stdout + res.stderr
    d = np.load(out)
    assert d["states"].shape[1:] == (len(_KEPT), 19, 19)
    assert d["policies"].shape[1] == _LS.policy_logit_count
    np.testing.assert_allclose(d["policies"].sum(axis=1), 1.0)
    assert d["states"].shape[0] == d["weights"].shape[0] == d["outcomes"].shape[0]
    # sampled plies are [POSITION_START, game_len) = 2..14 → 13 plies, ply 7
    # scatters into 2 rows → 14 rows
    assert d["states"].shape[0] == 14
    meta = json.loads((out.parent / (out.name + ".metadata.json")).read_text())
    assert meta["encoding_name"] == "v6_live2_ls"
    assert meta["n_positions"] == 14
    assert meta["extra"]["ls_scatter_extra_rows"] == 1
    assert meta["extra"]["ls_plies_emitted"] == 13
    assert meta["extra"]["k_max"] == _LS.k_max
