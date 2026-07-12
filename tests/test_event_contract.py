"""D-J DASH WP3 — THE ONE LAW: every dashboard panel cites a live producer.

Manifest-driven contract (replaces the pre-rebuild consumed⊆produced test that
enumerated terminal_dashboard.py + index.html — both retired/rebuilt). The panel
contract is now `hexo_rl/monitoring/dashboard_manifest.yaml`: every panel binding
names a PRODUCER, and this test asserts the producer is LIVE.

## Two producer channels (design §1.1)

The monitor has TWO event streams with different schemas; the two renderers read
different streams, so the produced-set is gathered PER CHANNEL:

  * ``emit_event``  — ``emit_event({"event": "<name>"})`` → logs/events_*.jsonl.
                      Grep: ``"event"\\s*:\\s*"<name>"``. Consumed by the web dashboard.
  * ``structlog``   — ``log.info("<name>", ...)`` → logs/<run_name>.jsonl.
                      Grep: ``.(info|warning|error)("<name>"``, WHOLE-FILE finditer
                      (train_step_summary is a multiline call). Consumed by the d1m TUI.

A manifest binding declares its ``channel``; this test checks the binding's event
against the produced-set for THAT channel. The old test was blind to the structlog
channel entirely — this finally guards the d1m TUI's producers.

## Mutation check

``test_mutation_fake_producer_is_flagged`` constructs a manifest with a nonsense
producer and asserts the checker flags it — proving the ONE LAW actually bites
(a docstring/message false-positive can't mask it because the name is nonsense).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_REPO = Path(__file__).parent.parent
_MANIFEST = _REPO / "hexo_rl" / "monitoring" / "dashboard_manifest.yaml"

# ── Producer enumerators (two channels) ─────────────────────────────────────
_EMIT_PATTERN = re.compile(r'"event"\s*:\s*"([^"]+)"')
# structlog: .info/.warning/.error("<lower_snake_name>", ...) — whole-file finditer.
_STRUCTLOG_PATTERN = re.compile(r'\.(?:info|warning|error)\(\s*["\']([a-z_][a-z0-9_]*)["\']')


def _gather(pattern: re.Pattern, dirs: list[str]) -> frozenset[str]:
    names: set[str] = set()
    for d in dirs:
        root = _REPO / d
        for py in root.rglob("*.py"):
            parts = py.relative_to(root).parts
            if "__pycache__" in parts or "tests" in parts or py.name.startswith("test_"):
                continue
            try:
                src = py.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for m in pattern.finditer(src):
                names.add(m.group(1))
    return frozenset(names)


def _produced_by_channel() -> dict[str, frozenset[str]]:
    return {
        "emit_event": _gather(_EMIT_PATTERN, ["hexo_rl"]),
        # startup (scripts/train.py) + eval_pipeline structlog events live in both dirs.
        "structlog": _gather(_STRUCTLOG_PATTERN, ["hexo_rl", "scripts"]),
    }


# ── Manifest walkers ────────────────────────────────────────────────────────
def _load_manifest() -> dict:
    return yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))


def _iter_event_bindings(manifest: dict):
    """Yield (label, channel, event) for every source_type: event binding.

    Covers panel bindings AND the top-level planner_regime guard source.
    """
    pr = manifest.get("planner_regime")
    if pr and pr.get("source_type") == "event":
        yield ("planner_regime", pr["channel"], pr["event"])
    for panel in manifest.get("panels", []):
        for bname, b in panel.get("bindings", {}).items():
            if b.get("source_type") == "event":
                yield (f"{panel['id']}:{bname}", b["channel"], b["event"])


def _iter_path_bindings(manifest: dict):
    """Yield (label, producer_path) for every derived/file binding."""
    for panel in manifest.get("panels", []):
        for bname, b in panel.get("bindings", {}).items():
            if b.get("source_type") in ("derived", "file"):
                yield (f"{panel['id']}:{bname}", b["producer_path"])


def _missing_producers(manifest: dict, produced: dict[str, frozenset[str]]) -> list[tuple]:
    """Bindings whose event is NOT in the produced-set for its channel."""
    out = []
    for label, channel, event in _iter_event_bindings(manifest):
        if event not in produced.get(channel, frozenset()):
            out.append((label, channel, event))
    return out


# ── Tests ────────────────────────────────────────────────────────────────────
def test_every_manifest_event_binding_has_live_producer() -> None:
    """THE ONE LAW: every panel binding's event is produced in its channel."""
    manifest = _load_manifest()
    produced = _produced_by_channel()
    missing = _missing_producers(manifest, produced)
    assert not missing, (
        "Manifest binding(s) cite an event with NO live producer in their channel:\n"
        + "\n".join(f"  - {label}: {event!r} (channel {channel})" for label, channel, event in missing)
    )


def test_every_derived_and_file_producer_path_exists() -> None:
    """derived/file bindings must cite a producer script/module that exists."""
    manifest = _load_manifest()
    missing = [
        (label, path) for label, path in _iter_path_bindings(manifest)
        if not (_REPO / path).exists()
    ]
    assert not missing, (
        "Manifest binding(s) cite a producer_path that does not exist:\n"
        + "\n".join(f"  - {label}: {path}" for label, path in missing)
    )


def test_mutation_fake_producer_is_flagged() -> None:
    """A panel citing a nonsense producer MUST be flagged (the law bites)."""
    produced = _produced_by_channel()
    fake_manifest = {
        "panels": [
            {
                "id": "fake.panel",
                "bindings": {
                    "web": {
                        "source_type": "event",
                        "channel": "emit_event",
                        "event": "__no_such_event_producer__",
                    }
                },
            }
        ]
    }
    missing = _missing_producers(fake_manifest, produced)
    assert missing == [("fake.panel:web", "emit_event", "__no_such_event_producer__")], (
        "mutation check failed to flag a nonexistent producer — the ONE LAW is not enforced"
    )


def test_produced_sets_nonempty() -> None:
    """Sanity: both channel enumerators must find producers (regex not broken)."""
    produced = _produced_by_channel()
    assert len(produced["emit_event"]) >= 15, sorted(produced["emit_event"])
    assert len(produced["structlog"]) >= 15, sorted(produced["structlog"])


def test_seed_events_present() -> None:
    """Core events the dashboard depends on must remain produced."""
    produced = _produced_by_channel()
    emit_seed = {
        "training_step", "iteration_complete", "game_complete", "eval_complete",
        "system_stats", "run_start", "value_probe_drift", "buffer_composition",
        "worker_draw_rate", "value_spread", "resolved_config",
    }
    structlog_seed = {
        "train_step_summary", "train_step", "evaluation_round_complete",
        "forced_win_trend", "startup",
    }
    assert not (emit_seed - produced["emit_event"]), sorted(emit_seed - produced["emit_event"])
    assert not (structlog_seed - produced["structlog"]), sorted(structlog_seed - produced["structlog"])
