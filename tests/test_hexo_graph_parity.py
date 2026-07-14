"""Byte-exact parity: Rust `hexo-graph` builder vs the Python oracle.

The Rust crate `engine/hexo-graph` (C1 / WP-1) ports
`hexo_rl/bots/strix_v1_graph.py::build_axis_graph_raw` for the LEGACY
relative+threat schema (`node_feat_dim=11`, the amended ragged-contract v1
wire format). The Python builder is the TEST ORACLE; the Rust builder is the
production path. This test drives the Rust builder over >=1000 real positions
and asserts:

  * INTEGER outputs byte-exact (node order, edge_index, masks, node_coords,
    n_stones, policy scatter indices, window_center, checksums): max diff == 0.
  * FLOAT features (node_feat, edge_attr) equal to <= 1e-6 (both cast float32,
    mirroring the oracle's Python-float -> torch.float32 deploy path).

Bridge: a standalone `[[bin]]` harness in the crate (feature `harness`) reads a
positions JSON and writes the graph payloads as JSON arrays; this test invokes
it via subprocess. NO PyO3 / no `engine` crate touched (that is WP-3's seam).

BUILD THE HARNESS FIRST (Makefile-independent):

    cargo build --release -j4 -p hexo-graph --features harness

If the binary is absent the test SKIPS with this command in the message.

Position coverage: the frozen WP-A self-play set (320 real positions,
`reports/probes/gnn_integration/wpa_positions.json`) augmented to >=1000 by (a)
a current-player / moves-remaining sweep and (b) prefix-truncations of the real
stone lists (the replay JSONLs under logs/replays/ hold only empty draw games,
so per-ply sweeping them yields nothing — see WP1_builder.md). Every geometry
is a real self-play board or a real-board prefix.
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
FLOAT_TOL = 1e-6


def _canonical_slot(q: int, r: int, cq: int, cr: int, trunk_sz: int = TRUNK_SZ) -> int:
    """Byte-parity with engine `core.rs::window_flat_idx_at_geom` (-1 off-window)."""
    half = (trunk_sz - 1) // 2
    wq = q - cq + half
    wr = r - cr + half
    if 0 <= wq < trunk_sz and 0 <= wr < trunk_sz:
        return wq * trunk_sz + wr
    return -1


def _build_inputs() -> list[dict]:
    """Real self-play positions augmented to >=1000 (see module docstring)."""
    base = json.loads(POSITIONS.read_text())["positions"]
    inputs: list[dict] = []
    for p in base:
        cp = int(p["current_player"])
        mr = int(p["moves_remaining"])
        stones = [[int(a), int(b), int(c)] for a, b, c in p["stones"]]
        # (a) as-recorded
        inputs.append({"stones": stones, "current_player": cp, "moves_remaining": mr})
        # (a) player/moves sweep — exercises player_feat / own_is_p1 / moves_feat
        inputs.append({"stones": stones, "current_player": -cp, "moves_remaining": 3 - mr})
        # (b) prefix truncations — distinct real-board geometries
        n = len(stones)
        if n >= 4:
            for frac in (0.25, 0.5, 0.75):
                k = max(1, int(round(n * frac)))
                if 1 <= k < n:
                    inputs.append(
                        {"stones": stones[:k], "current_player": cp, "moves_remaining": mr}
                    )
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
        return json.loads(out.read_text())


@pytest.mark.skipif(not POSITIONS.exists(), reason=f"missing {POSITIONS}")
def test_hexo_graph_rust_matches_python_oracle():
    if not HARNESS.exists():
        pytest.skip(f"harness binary absent; build with: {BUILD_CMD}")

    inputs = _build_inputs()
    assert len(inputs) >= 1000, f"only {len(inputs)} positions (<1000)"

    rust = _run_harness(inputs)
    assert len(rust) == len(inputs)

    max_int_diff = 0
    max_float_diff = 0.0
    n_edges_total = 0

    for spec, R in zip(inputs, rust):
        sm = {(q, r): pl for q, r, pl in spec["stones"]}
        cp = spec["current_player"]
        mr = spec["moves_remaining"]
        g = build_axis_graph_raw(
            sm, cp, mr, win_length=WIN_LENGTH, radius=RADIUS, relative_stones=True,
            threat_features=True, prune_empty_edges=False,
        )
        n_stones = len(sm)
        n_edges_total += len(g["edge_src"])

        # --- INTEGER byte-exact ---
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

        # gather rows + policy scatter (contract-added; oracle geometry references)
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
        f"\nPARITY OK: n={len(inputs)} positions, {n_edges_total} edges, "
        f"max_int_diff={max_int_diff}, max_float_diff={max_float_diff:.2e}"
    )
