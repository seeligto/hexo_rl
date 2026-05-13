"""INV12 (§176 §E) — viewer never imported by training/selfplay/engine/eval.

Pinned per docs/09_VIEWER_SPEC.md §1 passive-observer invariant.
Renderer code must never propagate exceptions to hot paths.

NOTE: when P78 lands (bots/ extracted to top-level hexo_rl/bots/), the
grep target set below does not change — bots/ is orthogonal to viewer/.
Per master §C P78 + reviewer §7.

NOTE: hexo_rl/engine/ does not exist as a Python package (engine is Rust-only,
exposed via PyO3). That dir is skipped silently when absent.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).parent.parent.parent / "hexo_rl"
HOT_PATH_DIRS = ["training", "selfplay", "engine", "eval"]
FORBIDDEN_IMPORT = re.compile(
    r"^\s*(?:from\s+hexo_rl\.viewer|import\s+hexo_rl\.viewer)",
    re.MULTILINE,
)


def test_viewer_not_imported_by_hot_paths() -> None:
    """No file under hot-path dirs may import hexo_rl.viewer."""
    violations: list[str] = []
    for d in HOT_PATH_DIRS:
        dir_path = ROOT / d
        if not dir_path.exists():
            # python pkg may not exist for some hot-path dirs (e.g. engine is Rust-only)
            continue
        for py in dir_path.rglob("*.py"):
            text = py.read_text()
            for m in FORBIDDEN_IMPORT.finditer(text):
                line_no = text[: m.start()].count("\n") + 1
                violations.append(f"{py.relative_to(ROOT.parent)}:{line_no}")
    assert not violations, (
        "VIEWER_SPEC §1 violation — hexo_rl.viewer imported in hot paths:\n"
        + "\n".join(violations)
    )


def test_emit_event_is_fire_and_forget(monkeypatch) -> None:
    """emit_event swallows all renderer exceptions; never propagates.

    Strategy: _renderers is a module-level list with no unregister API.
    Use monkeypatch.setattr to replace it with a fresh list containing
    a bad renderer for the duration of this test. monkeypatch restores
    the original list on teardown.
    """
    import hexo_rl.monitoring.events as events_mod

    called: list[bool] = []

    class _RaisingRenderer:
        def on_event(self, payload: dict) -> None:
            called.append(True)
            raise RuntimeError("intentional renderer failure — must not propagate")

    # Replace the module-level list so we don't pollute global renderer state.
    monkeypatch.setattr(events_mod, "_renderers", [_RaisingRenderer()])

    # Must not raise.
    events_mod.emit_event({"event": "test_inv12"})

    assert called, "renderer was not called — emit_event did not dispatch"
