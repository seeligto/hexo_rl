"""INV: the Hammerhead NNUE opponent is EVAL-PATH ONLY.

Hard constraint from the §P6 second-opponent wiring: the vendored Hammerhead
engine (and its ``NnueBot`` wrapper) must never be imported by the self-play,
training, or hot-path engine code. It rides exactly the same Python eval path
SealBot does — ``hexo_rl/eval`` + ``scripts/run_sealbot_eval.py`` — and nothing
else. A leak into the self-play/training tree would pull a heavyweight minimax
engine into the position-generation hot path.

This is a pure source grep — it does NOT import Hammerhead, so it runs even on
hosts where the vendored bot is not built (CI, vast pre-build).
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# Token that betrays a dependency on the vendored bot. Case-sensitive on the
# wrapper symbols; the engine module is lowercased "hammerhead".
_PATTERN = re.compile(r"\bnnue_bot\b|\bNnueBot\b|\bhammerhead\b", re.IGNORECASE)

# Files allowed to reference the bot — the eval path and its tests, plus this
# invariant itself. Paths are repo-relative.
_ALLOWLIST = {
    "hexo_rl/bots/nnue_bot.py",
    "hexo_rl/eval/evaluator.py",
    "hexo_rl/eval/opponent_runners.py",
    "hexo_rl/eval/eval_pipeline.py",
    "hexo_rl/eval/result_types.py",  # declares wr_nnue/ci_nnue keys (no import)
    "scripts/run_sealbot_eval.py",
    "tests/test_nnue_bot.py",
    "tests/test_nnue_eval_path_only.py",
}

# Trees scanned for a leak. vendor/ (the submodule itself) and caches excluded.
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


def test_hammerhead_referenced_only_on_eval_path():
    offenders: list[str] = []
    for path in _iter_source_files():
        rel = path.relative_to(_REPO).as_posix()
        if rel in _ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if _PATTERN.search(text):
            offenders.append(rel)
    assert not offenders, (
        "Hammerhead NNUE bot leaked off the eval path into: "
        f"{offenders}. It must be imported only from hexo_rl/eval + "
        "scripts/run_sealbot_eval.py (eval-path only)."
    )


def test_training_tree_has_no_hammerhead_import():
    """Tighter, explicit guard on the self-play/training hot path."""
    training = _REPO / "hexo_rl" / "training"
    offenders = []
    for path in training.rglob("*.py"):
        if _EXCLUDE_PARTS & set(path.parts):
            continue
        if _PATTERN.search(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(path.relative_to(_REPO).as_posix())
    assert not offenders, f"hammerhead referenced in training hot path: {offenders}"
