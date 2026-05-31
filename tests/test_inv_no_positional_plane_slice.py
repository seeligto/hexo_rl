"""§P5-CT de-hardcoding INV (positional-slice facet).

L65's lesson: a name-grep ledger (WIRE_CHANNELS / KEPT_PLANE_INDICES / {8,11})
missed bare positional plane slices like `states[:, 4]` — the v6-opponent-slot
hardcode that crashed a 4-plane run. This INV greps the live tree for bare
`[:, <int>]` plane slices and fails on any not in an explicit allowlist of
definitional SOURCE-layout sites.

Companion `test_encoding_arch_resolver.py` carries the other INV facet
(resolve_arch == registry for every encoding).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

# Bare second-axis integer slice: `x[:, 4]`, `tensor[:, 16, :, :]`. This is the
# canonical (batch, plane, H, W) plane index — the exact L65 signature.
_PLANE_SLICE_RE = re.compile(r"\[\s*:\s*,\s*(\d+)\s*[,\]]")

_SCAN_ROOTS = ("hexo_rl", "scripts")
_SKIP_PARTS = frozenset({"__pycache__", ".venv", "target", "node_modules", ".git"})

# Allowlist: files where a bare `[:, <int>]` is a DEFINITIONAL source-plane
# access (the fixed 18-/21-plane source contract), NOT a kept-slice that must
# track the encoding. Each entry is justified — do not extend without a reason.
_ALLOWLISTED_FILES: dict[str, str] = {
    # canonical encoder: writes turn-phase scalars to SOURCE planes 16/17.
    "hexo_rl/env/game_state.py": "source-plane writes (canonical 18-plane encoder)",
    # SOURCE-layout reader: plane 0 cur / HISTORY_LEN opp (fixed source contract).
    "hexo_rl/training/axis_distribution.py": "source-layout stone read (plane 0 cur)",
    # native v8 dataset builder — writes the 11-plane v8 layout directly.
    "hexo_rl/bootstrap/dataset_v8.py": "v8-native dataset builder (whole-file allowlisted)",
    # synthetic NN bench input — builds a fake source tensor for throughput timing.
    "scripts/bench_v6w25_nn.py": "synthetic NN bench input (source-layout)",
    # `policies[:, 0]` — action-index 0 of a POLICY array, not a state plane.
    "scripts/probe_tf32_channels_last.py": "policy action index, not a state plane",
}

# Inline escape hatch for one-off justified lines outside an allowlisted file.
_LINE_OK_TOKEN = "plane-literal-ok"


def _strip_comments_and_strings(line: str) -> str:
    no_comment = line.split("#", 1)[0]
    return re.sub(r"\"[^\"]*\"|'[^']*'", "", no_comment)


def _iter_py_files():
    for root in _SCAN_ROOTS:
        base = _REPO / root
        if not base.is_dir():
            continue
        for p in base.rglob("*.py"):
            if any(part in _SKIP_PARTS for part in p.parts):
                continue
            if "test" in p.name.lower():
                continue
            yield p


def _scan() -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for p in _iter_py_files():
        rel = str(p.relative_to(_REPO))
        if rel in _ALLOWLISTED_FILES:
            continue
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), start=1):
            if _LINE_OK_TOKEN in line:
                continue
            if _PLANE_SLICE_RE.search(_strip_comments_and_strings(line)):
                hits.append((rel, i, line.strip()))
    return hits


def test_no_unallowlisted_positional_plane_slice():
    hits = _scan()
    assert hits == [], (
        "bare positional plane slice(s) `[:, <int>]` on a live path — route through "
        "the resolver (cur_stone_slot / opp_stone_slot / kept_plane_indices) or "
        "allowlist with justification:\n"
        + "\n".join(f"  {f}:{ln}  {src}" for f, ln, src in hits)
    )


# ── detector teeth (so the INV can't silently rot into a no-op) ──────────────

@pytest.mark.parametrize("snippet", [
    "op = states[:, 4]",
    "tensor[:, 16, :, :] = mr_val",
    "x = pre_states[:, 0]",
])
def test_detector_flags_positional_slice(snippet):
    assert _PLANE_SLICE_RE.search(_strip_comments_and_strings(snippet))


@pytest.mark.parametrize("snippet", [
    "op = states[:, opp_slot]",          # registry-derived index — fine
    "cur = pre_states[:, _cur_slot]",    # routed — fine
    "# op = states[:, 4] in a comment",  # comment — stripped
    "label = cfg['states[:, 4]']",       # inside a string literal — stripped
])
def test_detector_ignores_non_hazards(snippet):
    assert not _PLANE_SLICE_RE.search(_strip_comments_and_strings(snippet))
