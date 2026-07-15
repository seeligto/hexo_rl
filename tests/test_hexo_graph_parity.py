"""Byte-exact parity: Rust `hexo-graph` builder vs the Python oracle.

The Rust crate `engine/hexo-graph` (C1 / WP-1) ports
`hexo_rl/bots/strix_v1_graph.py::build_axis_graph_raw` for the LEGACY
relative+threat schema (`node_feat_dim=11`, the amended ragged-contract v1
wire format). The Python builder is the TEST ORACLE; the Rust builder is the
production path. This test drives the Rust builder over >=1000 real positions
and asserts:

  * INTEGER outputs byte-exact (node order, edge_index, masks, node_coords,
    n_stones, policy scatter indices, window_center, checksums) via hard
    equality asserts — any mismatch fails the test immediately.
  * FLOAT features (node_feat, edge_attr) equal to <= 1e-6 (both cast float32,
    mirroring the oracle's Python-float -> torch.float32 deploy path).

Bridge: a standalone `[[bin]]` harness in the crate (feature `harness`) reads a
positions JSON and writes the graph payloads as JSON arrays; this test invokes
it via subprocess. NO PyO3 / no `engine` crate touched (that is WP-3's seam).
The harness output is `{"harness_schema_version": 1, "graphs": [...]}` — test
scaffolding, not the WP-3 wire contract.

BUILD THE HARNESS FIRST (Makefile-independent):

    cargo build --release -j4 -p hexo-graph --features harness

If the binary is absent the test SKIPS with this command in the message.

Position coverage: the frozen WP-A self-play set (320 real positions,
`reports/probes/gnn_integration/wpa_positions.json`) augmented to >=1000 by (a)
degenerate boards — the empty board (game start, ply 0) and 1-stone boards
(review MUST-FIX #2); (b) a current-player / moves-remaining sweep; (c)
prefix-truncations of the real stone lists (the replay JSONLs under
logs/replays/ hold only empty draw games, so per-ply sweeping them yields
nothing — see WP1_builder.md); (d) a SECOND GEOMETRY sweep (review #12): ~100
base positions rebuilt at win_length=5 / radius=4 so the non-default
parameterization of the threat windows, walk depth, and legal ring is
oracle-tested, not just compiled. Every geometry is a real self-play board, a
real-board prefix, or a canonical degenerate board.

WP-1 empty-board fix (launch blocker): the n_stones==1 degenerate boards
STILL assert byte-exact against the REAL Python oracle (that code path is
untouched). The n_stones==0 (empty board) degenerate boards do NOT — the
Rust builder's empty-board legal-move derivation now DELIBERATELY diverges
from the oracle's untested vacuous `[]` to instead mirror the dense engine's
own empty-board rule (`Board::legal_moves_set()`, 5x5 fallback,
`engine/src/board/moves.rs:95-101`); see the in-loop comment at the n_stones
== 0 branch below for the full rationale and the dense-shape assertions that
replace oracle equality for that case.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from hexo_rl.bots.strix_v1_graph import build_axis_graph_raw
from hexo_rl.diagnostics.forced_win_detector import window_center as _window_center

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "target" / "release" / "hexo_graph_harness"
POSITIONS = ROOT / "reports" / "probes" / "gnn_integration" / "wpa_positions.json"
BUILD_CMD = "cargo build --release -j4 -p hexo-graph --features harness"

TRUNK_SZ = 19
WIN_LENGTH = 6
RADIUS = 6
# Second geometry (review #12): a non-default (win_length, radius) pair the
# oracle fully supports, swept over a ~100-position subset.
ALT_WIN_LENGTH = 5
ALT_RADIUS = 4
ALT_SUBSET = 100
FLOAT_TOL = 1e-6
HARNESS_SCHEMA_VERSION = 1


def _canonical_slot(q: int, r: int, cq: int, cr: int, trunk_sz: int = TRUNK_SZ) -> int:
    """Byte-parity with engine `core.rs::window_flat_idx_at_geom` (-1 off-window)."""
    half = (trunk_sz - 1) // 2
    wq = q - cq + half
    wr = r - cr + half
    if 0 <= wq < trunk_sz and 0 <= wr < trunk_sz:
        return wq * trunk_sz + wr
    return -1


def _build_inputs() -> list[dict]:
    """Real self-play positions augmented to >=1000 (see module docstring).

    Each input dict carries explicit win_length/radius so the oracle call and
    the harness build the SAME parameterization per position.
    """
    base = json.loads(POSITIONS.read_text())["positions"]
    inputs: list[dict] = []

    def add(stones, cp, mr, wl=WIN_LENGTH, rad=RADIUS):
        inputs.append(
            {
                "stones": stones,
                "current_player": cp,
                "moves_remaining": mr,
                "win_length": wl,
                "radius": rad,
            }
        )

    # (a) degenerate boards vs the REAL oracle (review MUST-FIX #2):
    # empty board (the actual pre-move-1 state of every game) x player x moves,
    # and 1-stone boards for both stone owners x both sides to move.
    for cp in (1, -1):
        for mr in (1, 2):
            add([], cp, mr)
    for stone_p in (1, -1):
        for cp in (1, -1):
            add([[0, 0, stone_p]], cp, 2)

    for p in base:
        cp = int(p["current_player"])
        mr = int(p["moves_remaining"])
        stones = [[int(a), int(b), int(c)] for a, b, c in p["stones"]]
        # (b) as-recorded + player/moves sweep — exercises player_feat /
        # own_is_p1 / moves_feat
        add(stones, cp, mr)
        add(stones, -cp, 3 - mr)
        # (c) prefix truncations — distinct real-board geometries
        n = len(stones)
        if n >= 4:
            for frac in (0.25, 0.5, 0.75):
                k = max(1, int(round(n * frac)))
                if 1 <= k < n:
                    add(stones[:k], cp, mr)

    # (d) second geometry (review #12): first ALT_SUBSET base positions at
    # win_length=5 / radius=4 — exercises the threat-window sizing (2*wl-1
    # cells, wl-2 axis threshold), the axis-walk depth (window=wl-1), and the
    # legal-ring radius against the oracle.
    for p in base[:ALT_SUBSET]:
        stones = [[int(a), int(b), int(c)] for a, b, c in p["stones"]]
        add(stones, int(p["current_player"]), int(p["moves_remaining"]),
            wl=ALT_WIN_LENGTH, rad=ALT_RADIUS)

    return inputs


def _run_harness(inputs: list[dict]) -> list[dict]:
    payload = {
        "win_length": WIN_LENGTH,
        "radius": RADIUS,
        "trunk_size": TRUNK_SZ,
        "positions": inputs,
    }
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "in.json"
        out = Path(td) / "out.json"
        inp.write_text(json.dumps(payload))
        subprocess.run([str(HARNESS), str(inp), str(out)], check=True)
        doc = json.loads(out.read_text())
    assert doc["harness_schema_version"] == HARNESS_SCHEMA_VERSION
    return doc["graphs"]


@pytest.mark.skipif(not POSITIONS.exists(), reason=f"missing {POSITIONS}")
def test_hexo_graph_rust_matches_python_oracle():
    if not HARNESS.exists():
        pytest.skip(f"harness binary absent; build with: {BUILD_CMD}")

    inputs = _build_inputs()
    assert len(inputs) >= 1000, f"only {len(inputs)} positions (<1000)"
    n_alt_geom = sum(1 for s in inputs if s["win_length"] != WIN_LENGTH)
    assert n_alt_geom >= ALT_SUBSET, "second-geometry sweep missing"

    rust = _run_harness(inputs)
    assert len(rust) == len(inputs)

    max_float_diff = 0.0
    n_edges_total = 0

    for spec, R in zip(inputs, rust):
        sm = {(q, r): pl for q, r, pl in spec["stones"]}
        cp = spec["current_player"]
        mr = spec["moves_remaining"]
        n_stones = len(sm)

        if n_stones == 0:
            # WP-1 launch-blocker fix: n_stones == 0 is a DELIBERATE,
            # DOCUMENTED divergence from the Python graph oracle, not a
            # regression to catch via byte-exact oracle equality. The oracle
            # (`legal_moves_from_stones`) returns a vacuous `[]` for an empty
            # `stone_map` because its only real caller, `strix_v1_bot.py`,
            # always short-circuits its own opening move before reaching the
            # graph builder — the empty board is untested, out-of-domain
            # input for that "port", not an authoritative oracle answer.
            # Live graph self-play has no such short-circuit: MCTS roots a
            # search at ply 0 straight into this builder, and a vacuous
            # legal set raised `EmptyLegalSet` before the game could start
            # (`reports/probes/gnn_integration/WP5b_commitA_impl.md`
            # "Concerns / findings"). The authority for a 0-stone board's
            # legal set is the DENSE engine's own empty-board rule
            # (`Board::legal_moves_set()`, `engine/src/board/moves.rs:95-101`):
            # a fixed 5x5 region (25 cells) around the origin, independent of
            # `radius`. Assert that shape directly — this keeps the
            # regression class caught (a reintroduced vacuous path fails
            # these asserts) without asserting a Python-oracle equality that
            # is no longer the correct answer for this input.
            assert R["n_stones"] == 0
            assert R["num_nodes"] == 26, "25 dense-5x5 legal cells + 1 dummy"
            assert R["n_nodes_checksum"] == 26
            assert R["builder_impl"] == 1
            assert R["current_player"] == (1 if cp == 1 else -1)
            assert tuple(R["window_center"]) == (0, 0)
            expected_cells = sorted((dq, dr) for dq in range(-2, 3) for dr in range(-2, 3))
            # node layout is [stones(0) | legal(25) | dummy(1)] -> the last
            # coord pair is the dummy row; strip it POSITIONALLY (dummy's
            # (0,0) coord collides with a real legal cell's value, so a
            # value-based dedup would silently under-count).
            all_cells = list(zip(R["node_coords"][0::2], R["node_coords"][1::2]))
            legal_cells = all_cells[:-1]
            assert sorted(legal_cells) == expected_cells
            assert R["legal_node_gather"] == list(range(25))
            n_edges_total += len(R["edge_src"])
            continue

        g = build_axis_graph_raw(
            sm, cp, mr,
            win_length=spec["win_length"], radius=spec["radius"],
            relative_stones=True, threat_features=True, prune_empty_edges=False,
        )
        n_edges_total += len(g["edge_src"])

        # --- INTEGER byte-exact (hard equality; mismatch fails immediately) ---
        assert R["num_nodes"] == g["num_nodes"]
        assert R["n_nodes_checksum"] == g["num_nodes"]
        assert R["n_stones"] == n_stones
        assert R["edge_src"] == g["edge_src"]
        assert R["edge_dst"] == g["edge_dst"]
        assert R["legal_mask"] == [int(x) for x in g["legal_mask"]]
        assert R["stone_mask"] == [int(x) for x in g["stone_mask"]]
        assert R["node_coords"] == g["coords"]
        assert R["current_player"] == (1 if cp == 1 else -1)
        assert R["builder_impl"] == 1

        # gather rows + policy scatter (contract-added; oracle geometry
        # references — slots are i32 with -1 = off-window sentinel, per the
        # amended contract §2.1 / WP-3 option-(b) ruling)
        assert R["legal_node_gather"] == [n_stones + j for j in range(len(g["legal_coords"]))]
        cq, cr = _window_center([(s[0], s[1]) for s in spec["stones"]])
        assert list(R["window_center"]) == [cq, cr]
        exp_slots = [_canonical_slot(q, r, cq, cr) for (q, r) in g["legal_coords"]]
        assert R["policy_dst_slot"] == exp_slots

        # --- FLOAT <= 1e-6 (both float32) ---
        f_py = np.asarray(g["features"], dtype=np.float32)
        f_r = np.asarray(R["node_feat"], dtype=np.float32)
        assert f_py.shape == f_r.shape
        if f_py.size:
            max_float_diff = max(max_float_diff, float(np.abs(f_py - f_r).max()))

        ea_py = np.asarray(g["edge_attr"], dtype=np.float32).reshape(-1)
        ea_r = np.asarray(R["edge_attr"], dtype=np.float32)
        assert ea_py.shape == ea_r.shape
        if ea_py.size:
            max_float_diff = max(max_float_diff, float(np.abs(ea_py - ea_r).max()))

    assert max_float_diff <= FLOAT_TOL, f"float diff {max_float_diff} > {FLOAT_TOL}"
    print(
        f"\nPARITY OK: n={len(inputs)} positions ({n_alt_geom} at "
        f"wl={ALT_WIN_LENGTH}/r={ALT_RADIUS}), {n_edges_total} edges, "
        f"ints byte-asserted, max_float_diff={max_float_diff:.2e}"
    )
