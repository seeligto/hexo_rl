"""S7 round-2 (F5b/F7/F8) — `.in_channels` consumer sweep completeness pin.

`.in_channels` is a `HexTacToeNet`-only (dense/grid) attribute; `GnnNet` has
none — every genuine consumer that can receive EITHER representation must be
representation-aware (`hexo_rl.model.build_net.model_representation`, an
upstream dispatch guard, or reading off a dataclass field that is ALWAYS
declared regardless of representation, e.g. `InfModelArch`/`ArchSpec`), never
a bare live-model-instance read. `AttributeError: 'GnnNet' object has no
attribute 'in_channels'` was the exact S7 round-2 crash signature at three
independent call sites (F5b `anchor.py`, F7's root cause `inference.py`, F8
`inference.py`) — this is that bug CLASS, not three unrelated bugs.

This test does not (and cannot, via grep) prove every site is CORRECT — it
pins the CENSUS: the exact set of files touching `.in_channels` as of the S7
round-2 fix wave, classified below. A file dropping off is fine (nothing to
re-triage). A NEW file appearing means a NEW `.in_channels` consumer landed
somewhere in the tree without going through this sweep's classification —
fails loud so it gets triaged instead of silently reintroducing the bug
class. See `reports/probes/gnn_integration/S7_round2_fixes.md` for the full
per-site classification this census summarizes.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCAN_DIRS = ("hexo_rl", "scripts")
_PATTERN = re.compile(r"\.in_channels\b")

# ── Census, S7 round-2 fix wave (2026-07-15) ─────────────────────────────────
#
# "fixed"           — representation-guarded THIS fix wave (F5b/F7-root/F8).
# "already-safe"    — reads a dataclass field (InfModelArch/ArchSpec) that is
#                      ALWAYS declared for both representations, never a live
#                      model instance attribute; or gated upstream by a
#                      dispatch layer that never routes a GnnNet here
#                      (defender_dispatch.needs_no_drop_bot).
# "dense-only-scope" — the consumer is dense-by-construction/by-name (a v6-*
#                      class, a CNN-specific historical probe/diagnosis
#                      script, or a definition site, not a consumer) and is
#                      not reachable from the run4 launch/eval path today.
# "definition"       — the attribute's OWN declaration site, not a read.
_CENSUS: Dict[str, str] = {
    "hexo_rl/training/anchor.py": "fixed",                      # F5b
    "hexo_rl/selfplay/inference.py": "fixed",                   # F7-root / F8
    "hexo_rl/training/orchestrator.py": "already-safe",         # explicit graph short-circuit before the read
    "hexo_rl/training/lifecycle.py": "already-safe",            # InfModelArch.in_channels dataclass field
    "hexo_rl/training/loop.py": "already-safe",                 # InfModelArch.in_channels dataclass field
    "hexo_rl/monitoring/early_game_probe.py": "already-safe",   # resolve_arch(...).ArchSpec field, not a live model
    "hexo_rl/model/build_net.py": "definition/docstring",       # model_representation's own docstring prose
    "hexo_rl/model/network.py": "definition",                   # HexTacToeNet.__init__ declares self.in_channels
    "hexo_rl/model/global_token.py": "dense-only-scope",        # internal CNN-only submodule of HexTacToeNet's Trunk
    "hexo_rl/eval/v6_argmax_bot.py": "dense-only-scope",        # v6-specific bot by name/construction
    "hexo_rl/eval/k_cluster_mcts_bot.py": "already-safe",       # only dispatched via defender_dispatch (policy_pool != legal_set_scatter_max for gnn_axis_v1) — a GnnNet never reaches this class today
    "hexo_rl/eval/eval_pipeline.py": "definition/docstring",    # F7 fix comment prose, no code read
    "hexo_rl/monitoring/analyze_api.py": "dense-only-scope",    # standalone dashboard dev-tool, not on the run4 launch/eval path; flagged not fixed
    "scripts/probe_threat_logits.py": "dense-only-scope",       # threat-plane probe — no graph analogue (per-node threat features, not planes)
    "scripts/bench_v6w25_nn.py": "dense-only-scope",            # v6w25-specific bench script
    "scripts/dpfit_l1_mwfit_probe.py": "dense-only-scope",      # D-PFIT CNN-only investigation, banked/closed
    "scripts/dvderisk_e1_fullspec.py": "dense-only-scope",      # DVDERISK CNN-only probe script
    "scripts/dvderisk_e2_featablation.py": "dense-only-scope",  # DVDERISK CNN-only probe script
    "scripts/dvderisk_e2_indep_review.py": "dense-only-scope",  # DVDERISK CNN-only probe script (comment only)
    "scripts/dvderisk_ds3_probe.py": "dense-only-scope",        # DVDERISK CNN-only probe script
    "scripts/diagnosis/overspread_d3_redteam_split.py": "dense-only-scope",  # D-OVERSPREAD CNN-only diagnosis script
    "scripts/diagnosis/overspread_d3_fork_affinity.py": "dense-only-scope",  # D-OVERSPREAD CNN-only diagnosis script
    "scripts/e1/validate_ckpt.py": "dense-only-scope",          # E1 dist-head CNN-only validation script
    "scripts/headswap/model_heads.py": "dense-only-scope",      # headswap CNN-only tool
}


def _scan() -> set[str]:
    found: set[str] = set()
    for d in _SCAN_DIRS:
        for path in (_REPO_ROOT / d).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(errors="ignore")
            if _PATTERN.search(text):
                found.add(str(path.relative_to(_REPO_ROOT)))
    return found


def test_in_channels_consumer_census_has_no_untriaged_sites():
    found = _scan()
    known = set(_CENSUS)
    untriaged = found - known
    assert not untriaged, (
        f"New `.in_channels` consumer site(s) not in the S7 round-2 census: "
        f"{sorted(untriaged)}. Classify each as fixed/already-safe/"
        "dense-only-scope/definition in this test's _CENSUS (and fix it if it "
        "is a genuine live-model consumer reachable by a GnnNet) before "
        "extending the census."
    )


def test_census_fixed_sites_still_present_and_guarded():
    """The two fix-class files must still exist and still contain
    `.in_channels` (i.e. this test isn't vacuous because the lines were
    simply deleted) — AND each remaining occurrence must be textually
    reachable only behind a representation guard (heuristic: the file
    imports `model_representation` or checks `self._is_graph` /
    `representation ==` somewhere before the LAST `.in_channels` line)."""
    for rel_path, status in _CENSUS.items():
        if status != "fixed":
            continue
        path = _REPO_ROOT / rel_path
        text = path.read_text()
        assert _PATTERN.search(text), f"{rel_path}: expected .in_channels still present (dense branch)"
        guarded = (
            "model_representation" in text
            or "_is_graph" in text
            or "representation ==" in text
            or "representation !=" in text
        )
        assert guarded, (
            f"{rel_path}: contains .in_channels but no visible representation "
            "guard (model_representation / _is_graph / representation == "
            "check) — the S7 fix must have regressed."
        )
