"""WP3-C1: Event contract test — consumed ⊆ produced.

Every event NAME consumed by a renderer / dashboard / gate must also be
PRODUCED somewhere (training loop, eval, selfplay, monitoring daemons).
Orphan consumers (a consumer reading an event name no producer ever emits)
are a silent-failure class this test catches.

## Enumerator design

PRODUCED — gathered by scanning non-test Python source for the literal pattern
    "event": "<name>"
in dicts that are passed to emit_event (or assigned as the `event` key before
being handed off). The grep-style regex `"event":[\\s]*"[^"]+"` reliably covers
every call site without running the training loop.

CONSUMED — gathered from the registered-renderer consumer files:
  * hexo_rl/monitoring/terminal_dashboard.py   — `if event == "name":`
  * hexo_rl/monitoring/static/index.html       — `socket.on('name', ...)`
  * (web_dashboard.py passes all events through by event_name → not a filter)
  * (alert_rules.py: check_value_spread_canary is called WITH a payload from the
     terminal renderer after it matches "value_spread" — not an independent consumer
     of a new event name; covered by terminal_dashboard)

Both sets are derived from live source on disk; the test stays live as code changes.

## Known-reconciled events (not orphans, but historically flagged)

``value_spread`` — consumed by terminal_dashboard + run_feed_reader (JSONL channel).
    PRODUCED by hexo_rl/monitoring/value_spread_canary.py:fire_canary(), called from
    hexo_rl/training/trainer.py:save_checkpoint() on every checkpoint save.
    Sprint-log memory note "t3/v_spread instrument VOID" records that run2 charts
    were EMPTY because fire_canary was either not reached (no checkpoint saved) or
    the canary failed silently — a RUNTIME observability gap, not a code-level orphan.
    Contract verdict: PRODUCED (code is wired; runtime gap is a separate concern).

``value_spread_alt`` — NOT a separate event name. ``alt_spread`` is a FIELD inside
    the ``value_spread`` event payload. The "227/227 skipped" finding means the alt
    bank computation returns NaN under the live encoding (plane-mismatch skip in
    compute_value_spread_alt). Field-level concern; no event-level orphan exists.

## Exception list

``_KNOWN_ORPHANS`` holds event names that are consumed but whose production is
conditional / indirect in a documented way. The test asserts the live orphan set
== _KNOWN_ORPHANS (not merely ⊆), so removing a name from _KNOWN_ORPHANS while
it is still absent from produced causes a failure — keeping the exception list honest.
Currently empty because all consumed names resolve to produced at the code level.
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# PRODUCER files / patterns
# ---------------------------------------------------------------------------

# Non-test Python source directories that contain emit_event calls.
_PRODUCER_DIRS = [
    _REPO / "hexo_rl",
]

# Pattern: "event": "some_name"
_PRODUCED_PATTERN = re.compile(r'"event"\s*:\s*"([^"]+)"')


def _gather_produced() -> frozenset[str]:
    """Scan non-test Python sources for ``"event": "<name>"`` assignments.

    Covers every emit_event call site without importing or running the code.
    Returns a frozenset of distinct produced event names.
    """
    names: set[str] = set()
    for root in _PRODUCER_DIRS:
        for py_path in root.rglob("*.py"):
            # Skip __pycache__ and test files — tests are not producers.
            rel = py_path.relative_to(root)
            parts = rel.parts
            if "__pycache__" in parts:
                continue
            # Skip directories named "tests" or files matching test_*.py
            if "tests" in parts or py_path.name.startswith("test_"):
                continue
            try:
                src = py_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in _PRODUCED_PATTERN.finditer(src):
                names.add(match.group(1))
    return frozenset(names)


# ---------------------------------------------------------------------------
# CONSUMER files / patterns
# ---------------------------------------------------------------------------

# terminal_dashboard.py: Python renderer that pattern-matches on event name.
# Pattern: `if event == "name":` (single or double quotes)
_TERMINAL_CONSUMER_FILE = _REPO / "hexo_rl" / "monitoring" / "terminal_dashboard.py"
_TERMINAL_PATTERN = re.compile(r'if\s+event\s*==\s*["\']([^"\']+)["\']')

# index.html: JS web dashboard, uses socket.on('<name>', handler).
# Pattern: socket.on('<name>', ...)
_JS_CONSUMER_FILE = _REPO / "hexo_rl" / "monitoring" / "static" / "index.html"
_JS_PATTERN = re.compile(r"socket\.on\(['\"]([^'\"]+)['\"]")

# Socket.IO infrastructure events — not training-loop events.
_JS_INFRA_EVENTS = frozenset({"connect", "disconnect", "replay_history"})


def _gather_consumed() -> frozenset[str]:
    """Enumerate event names consumed by registered renderers.

    Sources:
    - terminal_dashboard.py: ``if event == "name":`` branches
    - index.html: ``socket.on('name', ...)`` subscriptions (excl. infra)

    web_dashboard.py forwards ALL events by their name without filtering —
    not a consumer that could add an orphan. alert_rules.py is called from
    terminal_dashboard after the event is matched — not an independent consumer
    of a new name. Both are therefore not enumerated here.

    Returns a frozenset of distinct consumed event names.
    """
    names: set[str] = set()

    # Python terminal renderer
    if _TERMINAL_CONSUMER_FILE.exists():
        src = _TERMINAL_CONSUMER_FILE.read_text(encoding="utf-8", errors="replace")
        for match in _TERMINAL_PATTERN.finditer(src):
            names.add(match.group(1))

    # JS web dashboard
    if _JS_CONSUMER_FILE.exists():
        src = _JS_CONSUMER_FILE.read_text(encoding="utf-8", errors="replace")
        for match in _JS_PATTERN.finditer(src):
            name = match.group(1)
            if name not in _JS_INFRA_EVENTS:
                names.add(name)

    return frozenset(names)


# ---------------------------------------------------------------------------
# Exception list (documented orphans / conditional producers)
# ---------------------------------------------------------------------------

# Names that are consumed but whose production is conditional / indirect in
# a way that is documented above. Adding a name here silences the failure;
# removing a name while it is still absent from produced causes a failure.
# Currently empty — all consumed names resolve to produced at the code level.
_KNOWN_ORPHANS: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_consumed_subset_of_produced() -> None:
    """Every consumed event name must appear in the produced set.

    Orphan consumers (consumed but not produced) indicate silent-failure risk:
    a renderer waits for an event that the training loop never emits.
    The test names each orphan explicitly on failure.
    """
    produced = _gather_produced()
    consumed = _gather_consumed()

    orphans = consumed - produced - _KNOWN_ORPHANS

    assert not orphans, (
        f"Orphan consumer(s) found — consumed but never produced:\n"
        + "\n".join(f"  - {name!r}" for name in sorted(orphans))
        + "\n\nFix: either add an emit_event call that produces the event, "
        "or add the name to _KNOWN_ORPHANS with a justification comment."
    )


def test_known_orphans_still_orphaned() -> None:
    """Keep the exception list honest: any name in _KNOWN_ORPHANS that is NOW
    produced should be removed from the list (it's no longer a known orphan).

    This prevents _KNOWN_ORPHANS from silently hiding fixed orphans.
    """
    if not _KNOWN_ORPHANS:
        return  # nothing to check

    produced = _gather_produced()
    consumed = _gather_consumed()

    stale_exceptions = _KNOWN_ORPHANS & produced
    if stale_exceptions:
        # These names are now produced — they're no longer orphans.
        raise AssertionError(
            f"Names in _KNOWN_ORPHANS are now produced — remove them from the list:\n"
            + "\n".join(f"  - {name!r}" for name in sorted(stale_exceptions))
        )


def test_produced_set_contains_seed_events() -> None:
    """Regression: core dashboard events must always remain in the produced set.

    Guards against accidental deletion of emit_event call sites for events
    the dashboard depends on.
    """
    produced = _gather_produced()
    required = {
        "training_step",
        "iteration_complete",
        "game_complete",
        "eval_complete",
        "system_stats",
        "run_start",
        "run_end",
        "value_probe_drift",
        "buffer_composition",
        "model_version_summary",
        "worker_draw_rate",
        "value_spread",
    }
    missing = required - produced
    assert not missing, (
        f"Core produced events missing from source scan: {sorted(missing)}"
    )


def test_consumed_set_size() -> None:
    """Sanity: consumed set must have at least N distinct event names.

    Guards against the enumerator returning an empty set due to a regex bug
    or a moved file — an empty consumed set would always pass the subset check.
    """
    consumed = _gather_consumed()
    assert len(consumed) >= 8, (
        f"Consumed set too small ({len(consumed)} names) — enumerator may be broken. "
        f"Got: {sorted(consumed)}"
    )


def test_produced_set_size() -> None:
    """Sanity: produced set must have at least N distinct event names."""
    produced = _gather_produced()
    assert len(produced) >= 15, (
        f"Produced set too small ({len(produced)} names) — enumerator may be broken. "
        f"Got: {sorted(produced)}"
    )


def test_value_spread_is_produced() -> None:
    """value_spread must be in the produced set.

    Reconciliation: the sprint-log memory note 'value_spread instrument VOID'
    records that during run2 the dashboard chart was EMPTY because fire_canary
    was not reached (checkpoint save gated by a different issue) — a RUNTIME
    gap, not a code-level one. At code level, fire_canary() in
    hexo_rl/monitoring/value_spread_canary.py emits event 'value_spread', and
    it is called from hexo_rl/training/trainer.py:save_checkpoint(). If this
    test fails, the emit call was removed and must be restored.
    """
    produced = _gather_produced()
    assert "value_spread" in produced, (
        "'value_spread' missing from produced set — fire_canary emit_event call removed?"
    )


def test_value_spread_alt_not_a_separate_event() -> None:
    """value_spread_alt is NOT a separate event name — it is the alt_spread FIELD
    inside the value_spread event payload.

    This test documents the reconciliation: the 'value_spread_alt: 227/227 skipped'
    finding from the sprint log refers to alt_spread=NaN due to plane-mismatch in
    compute_value_spread_alt (the alt bank requires a different encoding than v6_live2).
    There is no orphan consumer at the event-name level.

    If someone ever adds a separate 'value_spread_alt' event, this test will catch
    a regression in the reconciliation assumption.
    """
    produced = _gather_produced()
    consumed = _gather_consumed()
    assert "value_spread_alt" not in consumed, (
        "'value_spread_alt' appeared as an event-name consumer — "
        "update the reconciliation note in this file if intentional."
    )
    assert "value_spread_alt" not in produced, (
        "'value_spread_alt' appeared as a separately produced event — "
        "update the reconciliation note if intentional."
    )
