"""T-REFUSE: retro_slope.py raises on --override-n-sims; run_eval accepts + stamps deploy_matched=false."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

PYTHON = ".venv/bin/python"


def test_retro_slope_refuses_override_n_sims(tmp_path):
    """retro_slope.py must raise (SystemExit or ValueError) when --override-n-sims is passed."""
    result = subprocess.run(
        [PYTHON, "-m", "scripts.evalfair.retro_slope", "--override-n-sims", "75"],
        capture_output=True,
        text=True,
        cwd="/home/timmy/Work/Hexo/hexo_rl",
    )
    assert result.returncode != 0, (
        "retro_slope.py should exit non-zero when --override-n-sims is passed"
    )


def test_retro_slope_refuses_solver_override(tmp_path):
    """retro_slope.py must refuse --solver-backup (deploy-matched only)."""
    result = subprocess.run(
        [PYTHON, "-m", "scripts.evalfair.retro_slope", "--solver-backup"],
        capture_output=True,
        text=True,
        cwd="/home/timmy/Work/Hexo/hexo_rl",
    )
    assert result.returncode != 0, (
        "retro_slope.py should exit non-zero when --solver-backup is passed"
    )


def test_run_eval_accepts_override_n_sims_flag():
    """run_eval.py's argument parser must accept --override-n-sims without crash at parse time."""
    import importlib.util, types, sys as _sys

    # Test that the CLI argument is registered (parse-only test without running games)
    result = subprocess.run(
        [PYTHON, "-m", "scripts.evalfair.run_eval", "--help"],
        capture_output=True,
        text=True,
        cwd="/home/timmy/Work/Hexo/hexo_rl",
    )
    assert result.returncode == 0, f"run_eval --help failed:\n{result.stderr}"
    assert "--override-n-sims" in result.stdout, "--override-n-sims not in help output"


def test_retro_slope_override_is_programmatic():
    """Importing retro_slope and calling its refuse guard with n_sims_override set raises."""
    from scripts.evalfair.retro_slope import check_deploy_matched_only

    with pytest.raises((ValueError, SystemExit)):
        check_deploy_matched_only(n_sims_override=75, solver_backup=False)


def test_retro_slope_solver_backup_is_programmatic():
    """Importing retro_slope and calling its refuse guard with solver_backup=True raises."""
    from scripts.evalfair.retro_slope import check_deploy_matched_only

    with pytest.raises((ValueError, SystemExit)):
        check_deploy_matched_only(n_sims_override=None, solver_backup=True)
