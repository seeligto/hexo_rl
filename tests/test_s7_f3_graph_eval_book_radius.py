"""S7 F3 — graph checkpoints resolve to the standard EVALFAIR d5 book at r=5.

Closes S7 Part-2 Finding F3 (reports/probes/gnn_integration/S7_smoke_gate.md):
a graph-representation checkpoint (``representation="graph"``, e.g.
``gnn_axis_v1``) carries no ``legal_move_radius_schedule`` by design (no
windowed radius curriculum), so ``radius_from_checkpoint`` used to return
``None`` and ``resolve_book_for_radius`` immediately raised ("No book
registered for radius=None") before a single game was played. PINNED
CONTROLLER RULING (reports/probes/gnn_integration/S7_blocker_fixes.md):
graph checkpoints resolve EXPLICITLY to r=5 (the D-LADDER instrument
convention), logged (not silent), operator-overridable via
``--graph-eval-book-radius`` / ``graph_eval_book_radius_override``, with a
dense checkpoint's resolution left byte-identical.

Ckpt dicts here are synthetic minimal ``{"config": {...}, "step": N}``
payloads (mirrors ``tests/test_confres_6d_offline_radius.py``'s pattern) —
cheaper than S7's own full Trainer/GnnNet/HexgBuffer mint (S7 Part-2
diagnostic) and sufficient: the fix under test is pure config-resolution
logic in ``scripts.evalfair.core``, not the graph forward path (already
covered by ``tests/training/test_gnn_train_step.py`` et al.).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytest

from scripts.evalfair.core import (
    GRAPH_EVALFAIR_BOOK_RADIUS,
    radius_from_checkpoint,
)


def _make_fake_book(radius_stage: int, seed: int = 0) -> Dict:
    """Minimal fake book_v2 dict (mirrors test_retro_slope_fixes.py)."""
    return {
        "book_id": f"evalfair_r{radius_stage}_v2",
        "seed": seed,
        "radius_stage": radius_stage,
        "sampler_commit": "test",
        "openings": [],
    }


def _graph_ckpt(step: int = 40_000) -> Dict:
    return {"config": {"encoding": "gnn_axis_v1", "selfplay": {}}, "step": step}


def _dense_ckpt(step: int = 40_000, encoding: str = "v6_live2_ls") -> Dict:
    return {"config": {"encoding": encoding, "selfplay": {}}, "step": step}


# ── radius_from_checkpoint ────────────────────────────────────────────────────


def test_graph_ckpt_no_schedule_defaults_to_r5():
    assert GRAPH_EVALFAIR_BOOK_RADIUS == 5
    assert radius_from_checkpoint(_graph_ckpt()) == 5


def test_graph_ckpt_respects_operator_override():
    assert radius_from_checkpoint(
        _graph_ckpt(), graph_eval_book_radius_override=4,
    ) == 4


def test_dense_ckpt_no_schedule_stays_none_byte_identical():
    """The F3 mapping must NEVER fire for a dense (representation='grid') ckpt
    — a dense ckpt missing its schedule is a genuine bug the existing
    HARD-ERROR gates must still catch, not a legitimate graph no-schedule
    state."""
    assert radius_from_checkpoint(_dense_ckpt()) is None
    # An override passed for a dense job is inert — only graph ckpts consult it.
    assert radius_from_checkpoint(
        _dense_ckpt(), graph_eval_book_radius_override=5,
    ) is None


def test_dense_ckpt_with_schedule_unaffected_by_f3():
    """Pre-existing schedule-scan behavior is completely untouched (F3 only
    adds a fallback for the no-schedule + graph case)."""
    sched = [{"step": 0, "radius": 4}, {"step": 200_000, "radius": 5}]
    ck = {"config": {"encoding": "v6_live2_ls", "selfplay": {"legal_move_radius_schedule": sched}},
          "step": 100_000}
    assert radius_from_checkpoint(ck) == 4


def test_unregistered_encoding_treated_as_non_graph_no_raise():
    """A malformed/unregistered encoding name must never crash the resolver —
    conservative default (not-graph, None) preserves the dense fall-through
    to the existing HARD-ERROR gates."""
    ck = {"config": {"encoding": "not_a_real_encoding_xyz", "selfplay": {}}, "step": 1}
    assert radius_from_checkpoint(ck) is None


def test_no_declared_encoding_at_all_stays_none():
    """A bare weights-only strip with no config['encoding'] key at all."""
    assert radius_from_checkpoint({"config": {}, "step": 5}) is None


# ── resolve_book_for_radius composition ───────────────────────────────────────


def test_graph_ckpt_resolves_to_r5_book():
    from scripts.evalfair.retro_slope import resolve_book_for_radius

    book_r4 = _make_fake_book(4)
    book_r5 = _make_fake_book(5)
    books = {4: book_r4, 5: book_r5}

    radius = radius_from_checkpoint(_graph_ckpt())
    book = resolve_book_for_radius(radius, books, "gnn_ckpt.pt")
    assert book is book_r5
    assert book["radius_stage"] == 5


def test_unknown_override_radius_raises():
    """--graph-eval-book-radius naming a radius with no loaded book raises —
    validated by resolve_book_for_radius's existing book-map check, no
    duplicated validation."""
    from scripts.evalfair.retro_slope import resolve_book_for_radius

    books = {4: _make_fake_book(4), 5: _make_fake_book(5)}
    radius = radius_from_checkpoint(_graph_ckpt(), graph_eval_book_radius_override=7)
    assert radius == 7
    with pytest.raises(ValueError, match="No book registered for radius=7"):
        resolve_book_for_radius(radius, books, "gnn_ckpt.pt")


# ── run_arm threading (S7 Part-2 Finding F3: "run_arm hardcodes
#    radius_stage_override=None ... not threaded from any caller") ───────────


def _stub_graph_cfg() -> Dict:
    return {
        "encoding": "gnn_axis_v1",
        "selfplay": {"gumbel_m": 16, "c_visit": 50, "c_scale": 1.0,
                     "playout_cap": {"n_sims_full": 150}},
        "mcts": {"c_puct": 1.5},
    }


def test_run_arm_graph_ckpt_no_hard_error_with_default_mapping(monkeypatch, tmp_path):
    """Pre-F3: a per-stage r5 book against a graph ckpt (unresolvable radius,
    no override threaded) HARD-ERRORED at require_offline_radius before a
    single game was played (S7 Part-2). Post-F3: radius_from_checkpoint's
    own graph mapping resolves radius=5 with NO override needed, so the
    book/ckpt radius guard passes and run_arm proceeds (zero-opening book
    here, so it completes without needing a real model)."""
    import scripts.evalfair.core as core

    monkeypatch.setattr(core.torch, "load", lambda *a, **k: {"config": _stub_graph_cfg(), "step": 40_000})
    monkeypatch.setattr(core, "_ckpt_sha", lambda p: "deadbeef")
    monkeypatch.setattr(core, "sealbot_depth_from_config", lambda: 5)
    monkeypatch.setattr(core, "load_model_with_encoding", lambda *a, **k: None)

    book = _make_fake_book(5)  # radius_stage=5, zero openings -> no pairs played
    arm = core.ArmSpec(label="head")

    result = core.run_arm(
        "fake_gnn.pt", arm, book, out_dir=str(tmp_path / "run_arm_out"),
        book_seed=1, expect_encoding="gnn_axis_v1",
    )
    assert result["radius"] == 5
    assert result["n_pairs"] == 0


def test_run_arm_threads_explicit_override_to_its_own_radius_resolution(monkeypatch, tmp_path):
    """The override argument (S7 F3 fix) reaches run_arm's OWN internal
    radius_from_checkpoint call — the exact gap S7 Part-2 named ("not
    threaded from any caller"). Book pinned to r4; override=4 must make the
    per-stage guard agree (no mismatch raise)."""
    import scripts.evalfair.core as core

    monkeypatch.setattr(core.torch, "load", lambda *a, **k: {"config": _stub_graph_cfg(), "step": 40_000})
    monkeypatch.setattr(core, "_ckpt_sha", lambda p: "deadbeef")
    monkeypatch.setattr(core, "sealbot_depth_from_config", lambda: 5)
    monkeypatch.setattr(core, "load_model_with_encoding", lambda *a, **k: None)

    book = _make_fake_book(4)
    arm = core.ArmSpec(label="head")

    result = core.run_arm(
        "fake_gnn.pt", arm, book, out_dir=str(tmp_path / "run_arm_override_out"),
        book_seed=1, expect_encoding="gnn_axis_v1",
        graph_eval_book_radius_override=4,
    )
    assert result["radius"] == 4


def test_run_arm_dense_ckpt_still_hard_errors_unresolvable_radius(monkeypatch, tmp_path):
    """Dense-path regression guard: a dense ckpt with an unresolvable radius
    against a per-stage book must still HARD-ERROR — the F3 graph fallback
    must never mask a genuine dense-side bug (byte-identical to
    test_confres_6d_offline_radius.py's existing coverage, re-asserted here
    against the post-F3 code path)."""
    import scripts.evalfair.core as core
    from hexo_rl.config.resolve.radius import OfflineRadiusUnresolvableError

    _stub_cfg = {
        "encoding": "v6_live2_ls",
        "selfplay": {"gumbel_m": 16, "c_visit": 50, "c_scale": 1.0,
                     "playout_cap": {"n_sims_full": 150}},
        "mcts": {"c_puct": 1.5},
    }
    monkeypatch.setattr(core.torch, "load", lambda *a, **k: {"config": _stub_cfg, "step": 100_000})
    monkeypatch.setattr(core, "_ckpt_sha", lambda p: "deadbeef")
    monkeypatch.setattr(core, "sealbot_depth_from_config", lambda: 5)

    book = {"book_id": "evalfair_r4_v2", "radius_stage": 4, "openings": []}
    arm = core.ArmSpec(label="head")
    with pytest.raises(OfflineRadiusUnresolvableError):
        core.run_arm(
            "fake_dense.pt", arm, book, out_dir=str(tmp_path / "run_arm_dense_out"),
            book_seed=1, expect_encoding="v6_live2_ls",
        )


# ── stage2_d5_eval CLI/param threading (mantis_pull_eval.py) ──────────────────


def test_stage2_d5_eval_threads_override_to_run_arm(monkeypatch, tmp_path):
    """mantis_pull_eval.stage2_d5_eval's graph_eval_book_radius_override param
    (--graph-eval-book-radius) must reach BOTH its own book-selection
    radius_from_checkpoint call and run_arm — asserted via a spy on run_arm's
    kwargs (the S7 Part-2 crash site: resolve_book_for_radius used to raise
    before run_arm was ever reached)."""
    import scripts.eval.mantis_pull_eval as mantis
    import scripts.evalfair.core as core

    monkeypatch.setattr(core.torch, "load", lambda *a, **k: {"config": _stub_graph_cfg(), "step": 40_000})

    captured: Dict = {}

    def _fake_run_arm(ckpt_path, arm, book, *, out_dir, workers, n_boot, book_seed,
                       expect_encoding, n_pairs=None, graph_eval_book_radius_override=None,
                       **kw):
        captured["book_radius_stage"] = book.get("radius_stage")
        captured["override"] = graph_eval_book_radius_override
        return {"wr": 0.0, "pair_ci": [0.0, 0.0], "radius": book.get("radius_stage"), "eff_n": 0}

    monkeypatch.setattr(core, "run_arm", _fake_run_arm)

    book_r5_path = tmp_path / "book_r5.json"
    book_r5_path.write_text(
        __import__("json").dumps(_make_fake_book(5))
    )

    result = mantis.stage2_d5_eval(
        ckpt_path="fake_gnn.pt",
        book_r4=None,
        book_r5=str(book_r5_path),
        out_dir=str(tmp_path / "d5_out"),
        expect_encoding="gnn_axis_v1",
    )
    assert captured["book_radius_stage"] == 5
    assert captured["override"] is None  # no override passed -> default r5 mapping
    assert result["radius"] == 5
