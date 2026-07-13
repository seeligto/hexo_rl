"""Graph-builder numerical cross-check for the GNN-BC probe (D-L WP3).

Compares the FIDELITY-GATED reference builder
(``hexo_rl.bots.strix_v1_graph.build_axis_graph_raw``, used by the GNN-BC arm
unchanged) against an INDEPENDENT re-derivation
(``hexo_rl.probes.gnn_bc.graph_check.axis_edge_set``, written from scratch) on 10
positions. Exact match on the axis-edge SET + 5-dim edge features + dummy edges =
both validated (a subtly-wrong builder would fake an ARCH-NULL by feeding the GNN
a corrupted representation).

Writes reports/probes/gnn_bc/cross_check.md. CPU, cheap. DO run before training.

Usage:
    .venv/bin/python -m hexo_rl.probes.gnn_bc.cross_check
"""
from __future__ import annotations

import sys
from pathlib import Path

from hexo_rl.bots.strix_v1_graph import build_axis_graph_raw
from hexo_rl.probes.gnn_bc.graph_check import axis_edge_set, reference_edge_set

# The 10 cross-check positions (first 10 of the strix fidelity fixtures).
_FIX = Path(__file__).resolve().parents[3] / "reports/tourney/strix_fidelity"


def _load_positions():
    # Explicit-path load (the bare name ``fixtures`` collides with tests/fixtures/).
    import importlib.util  # noqa: PLC0415
    spec = importlib.util.spec_from_file_location("_strix_fixtures", _FIX / "fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.POSITIONS[:10], mod.WIN_LENGTH, mod.RADIUS


def run() -> dict:
    positions, WL, RAD = _load_positions()
    rows = []
    all_ok = True
    for pos in positions:
        stone_map = {tuple(k): v for k, v in pos["stones"]}
        g = build_axis_graph_raw(
            stone_map, pos["current_player"], pos["moves_remaining"],
            win_length=WL, radius=RAD,
            prune_empty_edges=True, threat_features=True, relative_stones=True,
        )
        ref = reference_edge_set(g)
        ind = axis_edge_set(stone_map, win_length=WL, radius=RAD, prune_empty_edges=True)
        ref_keys = set(ref["axis_edges"].keys())
        ind_keys = set(ind["axis_edges"].keys())
        axis_match = ref_keys == ind_keys
        attr_match = axis_match and all(
            ref["axis_edges"][k] == ind["axis_edges"][k] for k in (ref_keys & ind_keys)
        )
        dummy_match = ref["dummy_pairs"] == ind["dummy_pairs"]
        ok = axis_match and attr_match and dummy_match
        all_ok = all_ok and ok
        rows.append({
            "name": pos["name"], "nodes": g["num_nodes"],
            "n_axis_ref": len(ref_keys), "n_axis_ind": len(ind_keys),
            "axis_set_match": axis_match, "edge_attr_match": attr_match,
            "dummy_match": dummy_match, "ok": ok,
        })
    return {"rows": rows, "all_match": all_ok}


def main() -> int:
    result = run()
    lines = ["# GNN-BC graph-builder cross-check (D-L WP3)", "",
             "Reference (fidelity-gated, used by the GNN arm): "
             "`hexo_rl.bots.strix_v1_graph.build_axis_graph_raw`.",
             "Independent re-derivation: `hexo_rl.probes.gnn_bc.graph_check.axis_edge_set` "
             "(from scratch, no shared code).", "",
             "| position | nodes | n_axis(ref) | n_axis(ind) | axis-set | edge-attr | dummy | OK |",
             "|---|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        lines.append(
            f"| {r['name']} | {r['nodes']} | {r['n_axis_ref']} | {r['n_axis_ind']} | "
            f"{r['axis_set_match']} | {r['edge_attr_match']} | {r['dummy_match']} | {r['ok']} |")
    verdict = "EXACT MATCH — both builders validated" if result["all_match"] else "MISMATCH — investigate"
    lines += ["", f"**Verdict: {verdict}** (10/10 positions).", ""]
    out = Path(__file__).resolve().parents[3] / "reports/probes/gnn_bc/cross_check_edges.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}")
    return 0 if result["all_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
