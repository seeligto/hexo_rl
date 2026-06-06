"""INV: the off-window adversary (D-EXPLOIT Phase 1) is EVAL-PATH ONLY.

The adversary + its geometry helpers deliberately model the single-window action blind
spot for an OFFLINE exploitability probe. They must never be imported by the self-play,
training, or hot-path engine code — a leak would pull probe-only attack logic into the
position-generation hot path (same hazard class as the Hammerhead NNUE bot).

Pure source grep — imports nothing, so it runs even where torch/engine are unbuilt.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

_PATTERN = re.compile(r"\boffwindow_adversary\w*\b|\bOffWindowAdversaryBot\b|\boffwindow_geom\b")

# Files allowed to reference the adversary — the eval/probe path and its tests only.
_ALLOWLIST = {
    "hexo_rl/bots/offwindow_adversary_bot.py",
    "hexo_rl/bots/offwindow_geom.py",
    "scripts/exploit_probe.py",
    # D-EXPLOIT Phase 3 — the in-pipeline exploitability eval opponent (eval-path only).
    "hexo_rl/eval/offwindow_probe.py",
    "hexo_rl/eval/evaluator.py",
    "hexo_rl/eval/opponent_runners.py",
    "hexo_rl/eval/eval_pipeline.py",
    "tests/test_offwindow_adversary.py",
    "tests/test_offwindow_geom.py",
    "tests/test_offwindow_adversary_eval_path_only.py",
}

_SCAN_DIRS = ("hexo_rl", "engine", "scripts")
_EXCLUDE_PARTS = {"__pycache__", "vendor", ".git", "target", "node_modules"}


def _iter_source_files():
    for d in _SCAN_DIRS:
        root = _REPO / d
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix not in (".py", ".rs"):
                continue
            if _EXCLUDE_PARTS & set(path.parts):
                continue
            yield path


def test_offwindow_adversary_referenced_only_on_eval_path():
    offenders: list[str] = []
    for path in _iter_source_files():
        rel = path.relative_to(_REPO).as_posix()
        if rel in _ALLOWLIST:
            continue
        if _PATTERN.search(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(rel)
    assert not offenders, (
        f"off-window adversary leaked off the eval/probe path into: {offenders}. It must "
        "be imported only from the adversary module, scripts/exploit_probe.py, and tests."
    )


def test_training_and_selfplay_have_no_offwindow_adversary():
    """Tighter guard on the self-play / training hot path."""
    offenders = []
    for sub in ("training", "selfplay"):
        root = _REPO / "hexo_rl" / sub
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if _EXCLUDE_PARTS & set(path.parts):
                continue
            if _PATTERN.search(path.read_text(encoding="utf-8", errors="ignore")):
                offenders.append(path.relative_to(_REPO).as_posix())
    assert not offenders, f"off-window adversary referenced in training/selfplay: {offenders}"
