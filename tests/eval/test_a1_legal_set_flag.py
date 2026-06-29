"""§D-RECONFIRM R1 — the A1 harness must thread the multi-window decode into BOTH arms.

The original A1 (+0.165) ran the SINGLE-window deploy head (legal_set=False). R1 re-measures
the paired delta on the corrected MULTI-WINDOW head the net was trained under. The load-bearing
guarantee: `--legal-set` flips legal_set on the baseline AND on the solver-backup's inner head,
so the paired delta isolates the solver — not a windowing mismatch between arms.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from hexo_rl.eval.solver_backup_bot import SolverBackupBot

_HARNESS = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "run_a1_solver_backup.py"
_KNOBS = {"gumbel_m": 16, "n_sims_full": 150, "c_visit": 50.0, "c_scale": 1.0, "c_puct": 1.25}


def _load_harness():
    spec = importlib.util.spec_from_file_location("run_a1_solver_backup", _HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_baseline_arm_threads_legal_set():
    m = _load_harness()
    sw = m._build_cand("baseline", object(), _KNOBS, 0, seed=1, window_half=9, legal_set=False)
    ls = m._build_cand("baseline", object(), _KNOBS, 0, seed=1, window_half=9, legal_set=True)
    assert sw._legal_set is False and ls._legal_set is True
    assert ls.name().endswith(",ls)")  # provenance tag distinguishes the two arms in logs


def test_backup_arm_inner_head_threads_legal_set():
    """The solver-backup arm must also run multi-window — else the paired delta confounds the
    solver with a single-window vs multi-window baseline gap (the whole point of R1)."""
    m = _load_harness()
    ls = m._build_cand("backup_d6", object(), _KNOBS, 6, seed=1, window_half=9, legal_set=True)
    assert isinstance(ls, SolverBackupBot)
    assert ls._inner._legal_set is True
    sw = m._build_cand("backup_d6", object(), _KNOBS, 6, seed=1, window_half=9, legal_set=False)
    assert sw._inner._legal_set is False


def test_default_is_single_window_preserves_original_a1():
    """Omitting legal_set must reproduce the original (handicapped) single-window instrument —
    no silent behavior change to the shipped A1."""
    m = _load_harness()
    base = m._build_cand("baseline", object(), _KNOBS, 0, seed=1, window_half=9)
    assert base._legal_set is False
