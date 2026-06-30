"""D-WS3 L1 smoke — tests for the trap-flip gate logic + the held-out corpus
reconstruction. Pure-logic tests (classify/decide) always run; the engine-backed
reconstruction tests skip when the binding or the exported corpus is absent."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(modname: str, relpath: str):
    spec = importlib.util.spec_from_file_location(modname, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Module-level imports of run_l1_trapflip_smoke pull in torch/engine; load lazily
# only inside the tests that need it. classify/decide are pure but live in that
# module, so guard their import too.
def _l1():
    try:
        return _load("l1_smoke", "scripts/eval/run_l1_trapflip_smoke.py")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"run_l1_trapflip_smoke import failed (binding/torch absent): {e}")


# ── pure verdict logic (the dispatcher pre-reg) ──────────────────────────────
def test_classify():
    l1 = _l1()
    assert l1.classify([1, 2], [1, 2], [3, 4]) == "saving"
    assert l1.classify([3, 4], [1, 2], [3, 4]) == "blunder"
    assert l1.classify([5, 5], [1, 2], [3, 4]) == "other"
    assert l1.classify(None, [1, 2], [3, 4]) == "none"


def test_decide_generalizes():
    l1 = _l1()
    # candidate >= pass AND > baseline AND kill clears
    assert l1.decide(0.30, 0.16, 0.05, 0.25, 0.16, 0.16) == "GENERALIZES"


def test_decide_memorizes():
    l1 = _l1()
    assert l1.decide(0.16, 0.16, 0.05, 0.25, 0.16, 0.16) == "MEMORIZES"
    assert l1.decide(0.12, 0.10, 0.05, 0.25, 0.16, 0.16) == "MEMORIZES"


def test_decide_kill_dominates():
    l1 = _l1()
    # KILL fires even when the flip would otherwise pass
    assert l1.decide(0.40, 0.16, 0.20, 0.25, 0.16, 0.16) == "KILL>16%"


def test_decide_indeterminate_band():
    l1 = _l1()
    # flip between floor and pass, kill clears
    assert l1.decide(0.20, 0.16, 0.05, 0.25, 0.16, 0.16) == "INDETERMINATE"


def test_decide_pass_requires_margin_over_baseline():
    l1 = _l1()
    # candidate == pass threshold but NOT > baseline -> not a generalize
    assert l1.decide(0.25, 0.25, 0.05, 0.25, 0.16, 0.16) != "GENERALIZES"


def test_decide_kill_unrun_withholds_generalize():
    l1 = _l1()
    # would-generalize flip but the KILL co-gate never ran (kill_rate None) -> the
    # pre-reg requires KILL to CLEAR, so withhold GENERALIZES (review MAJOR).
    assert l1.decide(0.30, 0.16, None, 0.25, 0.16, 0.16) == "INDETERMINATE_KILL_UNRUN"
    # a non-generalizing flip with kill None still reads as no-lift, not a false pass.
    assert l1.decide(0.15, 0.16, None, 0.25, 0.16, 0.16) == "MEMORIZES"


def test_decide_memorizes_baseline_relative():
    l1 = _l1()
    # F4: baseline already >floor under multi-window; a candidate at/below baseline =
    # no lift = MEMORIZES even though it clears the static 16% floor.
    assert l1.decide(0.21, 0.21, 0.05, 0.25, 0.16, 0.16) == "MEMORIZES"
    assert l1.decide(0.20, 0.22, 0.05, 0.25, 0.16, 0.16) == "MEMORIZES"


# ── engine-backed corpus reconstruction (skip if binding/corpus absent) ──────
HELDOUT = REPO / "reports/d_tactical_2026-06-26/heldout_traps.jsonl"


def _have_engine() -> bool:
    try:
        import engine  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not HELDOUT.exists(), reason="held-out corpus not exported")
def test_heldout_schema_and_disjointness():
    rows = [json.loads(l) for l in open(HELDOUT) if l.strip()]
    assert rows, "held-out corpus is empty"
    required = {"pos_id", "source_game_id", "parent_move_seq", "post_move_seq",
                "saving_move", "blunder_move", "in_window", "current_player_parent"}
    for r in rows:
        assert required <= set(r), f"row {r.get('pos_id')} missing {required - set(r)}"
        assert list(r["saving_move"]) != list(r["blunder_move"]), "degenerate trap leaked"
    fids_path = REPO / "reports/d_tactical_2026-06-26/finetune_game_ids.json"
    if fids_path.exists():
        fids = set(json.load(open(fids_path)))
        held = {r["source_game_id"] for r in rows}
        assert held.isdisjoint(fids), "held-out set NOT game-disjoint from finetune ids"


@pytest.mark.skipif(not (_have_engine() and HELDOUT.exists()),
                    reason="engine binding or held-out corpus absent")
def test_board_from_trap_replays_turn_phase():
    z2 = _load("z2_ladder", "scripts/eval/run_z2_standalone_ladder.py")
    rows = [json.loads(l) for l in open(HELDOUT) if l.strip()]

    def pm1(p):
        s = str(p)
        return 1 if ("One" in s or s == "1") else -1

    for r in rows[:5]:
        bp = z2._board_from_trap(r, "v6_live2_ls", which="parent")
        bpost = z2._board_from_trap(r, "v6_live2_ls", which="post")
        assert pm1(bp.current_player) == int(r["current_player_parent"]), r["pos_id"]
        assert pm1(bpost.current_player) == int(r["current_player_post"]), r["pos_id"]
        # the saving move is a legal choice at the parent board (the L1 lever)
        legal = [tuple(m) for m in bp.legal_moves()]
        assert tuple(r["saving_move"]) in legal, f"saving move not legal at parent {r['pos_id']}"
