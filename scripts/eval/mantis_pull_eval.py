"""WP3 — on-demand external eval reader (laptop-side).

Operator-triggered script that:
  Stage 0   rsync pull NEW banked checkpoints from vast (read-only, idempotent).
            Degrades gracefully when vast is unreachable — processes whatever is local.
  Stage 1   value-health: calls T7 validate_ckpt -> M1-M4, appends to series.jsonl.
  Stage 2   d5 fair-book (64 pairs, 150 sims, g=0, pair-bootstrap CI): reuses
            scripts/evalfair/run_retro_ckpt.py's run_arm path.
  Stage 3   kraken-MCTS (200 sims, temp0, 32 pairs): reuses
            scripts/evalfair/head_vs_krakenbot.py; gated behind --skip-kraken /
            absent-asset skip-with-reason.
  Stage 3b  strix-g128 bar (128 sims, Gumbel-AZ, 32 pairs): reuses
            scripts/evalfair/head_vs_strix.py; gated behind --with-strix (default OFF) /
            absent-asset skip-with-reason. Runs strix in ITS OWN venv (hexo_rs +
            torch_geometric); the delegation child is strix_g128_child.py.
            D-K tournament field ceiling (+313 Elo). run3 bar from day one.
  Stage 4   append a per-bar row to the series, print slope table (Theil-Sen) + verdict.

Done-marker pattern: each processed checkpoint gets a <stem>.mantis_done sentinel.
Idempotent: re-running skips done-marked checkpoints.

CLI::

    .venv/bin/python scripts/eval/mantis_pull_eval.py \\
        --ckpt-dir checkpoints/run2_retro \\
        --series-out reports/e1/value_health_series.jsonl \\
        --retro-out reports/evalfair/retro_slope \\
        --book-r4 tests/fixtures/opening_books/evalfair_r4_v2.json \\
        --book-r5 tests/fixtures/opening_books/evalfair_r5_v2.json \\
        --skip-kraken

    # Target a specific checkpoint:
    .venv/bin/python scripts/eval/mantis_pull_eval.py \\
        --ckpt checkpoints/run2_retro/checkpoint_00248000.pt \\
        --skip-kraken

    # With strix-g128 bar (D-K ceiling, run3):
    .venv/bin/python scripts/eval/mantis_pull_eval.py \\
        --ckpt checkpoints/run3/checkpoint_00050000.pt \\
        --with-strix

Constraints:
  - Read-only wrt vast. Serial, niced. workers=1 (load-34 crash on record at higher).
  - Gated loader (encoding v6_live2_ls). No new eval math — composes existing tools.
  - Deterministic series rows.
  - strix-g128 requires strix's dedicated venv (hexo_rs + torch_geometric); gated
    behind --with-strix so default runs are unaffected (byte-identical when absent).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch


# ── Stage 0 helpers: rsync pull ──────────────────────────────────────────────


def _build_rsync_cmd(
    host: str,
    remote_path: str,
    local_path: str,
    extra_flags: Sequence[str] = (),
) -> List[str]:
    """Build an rsync pull command (local_path is ALWAYS the destination).

    Read-only pull: rsync from remote to local, never the reverse.
    """
    return [
        "rsync",
        "-avz",
        "--progress",
        *extra_flags,
        f"{host}:{remote_path}",
        local_path,
    ]


def rsync_pull(
    host: str,
    remote_path: str,
    local_path: str,
    timeout: int = 300,
) -> bool:
    """Pull new checkpoints from vast via rsync.

    Returns True on success, False when vast is unreachable or rsync fails.
    Never raises — degrades gracefully so the caller can process local files.
    """
    cmd = _build_rsync_cmd(host, remote_path, local_path)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        if result.returncode == 0:
            return True
        print(
            f"[mantis] rsync failed (rc={result.returncode}): {result.stderr[:200]}",
            file=sys.stderr,
        )
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        print(f"[mantis] rsync unreachable: {e}", file=sys.stderr)
        return False


# ── Done-marker + checkpoint collection ──────────────────────────────────────


def _done_marker(ckpt_path: Path) -> Path:
    return ckpt_path.parent / (ckpt_path.stem + ".mantis_done")


def collect_new_ckpts(ckpt_dir: Path) -> List[Path]:
    """Return all .pt files in ckpt_dir that lack a .mantis_done sentinel.

    Sorted ascending by filename (step order).
    """
    ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"))
    return [p for p in ckpts if not _done_marker(p).exists()]


def _mark_done(ckpt_path: Path, summary: Dict[str, Any]) -> None:
    _done_marker(ckpt_path).write_text(json.dumps(summary, indent=2))


# ── Arm inference ─────────────────────────────────────────────────────────────


def infer_arm_from_ckpt(ckpt_path: str) -> str:
    """Return 'dist65' if the checkpoint contains a dist value head, else 'scalar'.

    Reads only the state_dict keys — never loads the full model weights.
    Mirrors the post-load guard in checkpoint_loader.py.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        return "scalar"
    state = raw.get("model_state", raw.get("state_dict", {}))
    if any("value_fc2_bins" in k for k in (state or {})):
        return "dist65"
    return "scalar"


# ── Stage 1: value-health ─────────────────────────────────────────────────────


def stage1_value_health(
    ckpt_path: str,
    arm: str,
    series_out: str,
    probe_path: Optional[str] = None,
    negatives_path: Optional[str] = None,
    games_path: Optional[str] = None,
    wp2_games: Optional[Dict[str, str]] = None,
    no_sha_check: bool = False,
) -> Dict[str, Any]:
    """Run T7 validate_ckpt and return the emitted M1-M4 row.

    Reuses scripts.e1.validate_ckpt.validate_ckpt verbatim — no new metric math.
    Writes one row to series_out (appended).
    """
    from scripts.e1.validate_ckpt import validate_ckpt as _validate_ckpt
    from scripts.e1.validate_ckpt import DEFAULT_PROBE, DEFAULT_NEGATIVES, DEFAULT_GAMES

    kw: Dict[str, Any] = {
        "no_sha_check": no_sha_check,
    }
    if probe_path is not None:
        kw["probe_path"] = probe_path
    else:
        kw["probe_path"] = DEFAULT_PROBE
    if negatives_path is not None:
        kw["negatives_path"] = negatives_path
    else:
        kw["negatives_path"] = DEFAULT_NEGATIVES
    if games_path is not None:
        kw["games_path"] = games_path
    else:
        kw["games_path"] = DEFAULT_GAMES
    if wp2_games is not None:
        kw["wp2_games"] = wp2_games

    return _validate_ckpt(ckpt_path, arm, series_out, **kw)


# ── Stage 2: d5 fair-book eval ────────────────────────────────────────────────


def stage2_d5_eval(
    ckpt_path: str,
    book_r4: Optional[str],
    book_r5: Optional[str],
    out_dir: str,
    workers: int = 1,
    n_boot: int = 2000,
    expect_encoding: str = "v6_live2_ls",
    n_pairs: Optional[int] = None,
) -> Dict[str, Any]:
    """Run deploy-matched d5 eval (64 pairs, 150 sims, g=0, pair-bootstrap CI).

    Reuses scripts.evalfair.core.run_arm verbatim via the run_retro_ckpt path.
    Resume-safe: if result.json exists in out_dir, returns the cached result.

    Workers forced to 1 (load-34 crash record at higher).
    """
    from scripts.evalfair.core import ArmSpec, extract_deploy_knobs, radius_from_checkpoint, run_arm
    from scripts.evalfair.retro_slope import resolve_book_for_radius
    from scripts.evalfair.book import load_book

    out = Path(out_dir)
    result_path = out / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    books_by_radius: Dict[int, Dict[str, Any]] = {}
    if book_r4:
        b = load_book(Path(book_r4))
        books_by_radius[int(b["radius_stage"])] = b
    if book_r5:
        b = load_book(Path(book_r5))
        books_by_radius[int(b["radius_stage"])] = b
    if not books_by_radius:
        raise ValueError("stage2_d5_eval: supply at least one of book_r4 / book_r5")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    radius = radius_from_checkpoint(ck)
    book = resolve_book_for_radius(radius, books_by_radius, ckpt_path)

    arm = ArmSpec(label="sims150")
    result = run_arm(
        ckpt_path,
        arm,
        book,
        out_dir=out_dir,
        workers=max(1, workers),
        n_boot=n_boot,
        book_seed=book.get("seed", 20260709),
        expect_encoding=expect_encoding,
        n_pairs=n_pairs,
    )
    return result


# ── Stage 3: kraken eval ─────────────────────────────────────────────────────


def _run_kraken_eval(
    ckpt_path: str,
    book_r5: str,
    out_dir: str,
    kraken_asset: str,
    n_pairs: int = 32,
    kraken_sims: int = 200,
    kraken_temp: float = 0.0,
    n_boot: int = 2000,
    expect_encoding: str = "v6_live2_ls",
) -> Dict[str, Any]:
    """Run kraken-MCTS (200 sims, temp0, 32 pairs). Returns result dict."""
    from scripts.evalfair.head_vs_krakenbot import run_head_vs_krakenbot
    from scripts.evalfair.book import load_book

    book = load_book(Path(book_r5))
    return run_head_vs_krakenbot(
        ckpt=ckpt_path,
        book=book,
        out_dir=out_dir,
        kraken_path=kraken_asset,
        kraken_mcts=True,
        kraken_sims=kraken_sims,
        kraken_temp=kraken_temp,
        n_boot=n_boot,
        book_seed=book.get("seed", 20260710),
        expect_encoding=expect_encoding,
        n_pairs=n_pairs,
    )


def _stage3_kraken(
    ckpt_path: str,
    out_dir: str,
    book_r5: Optional[str],
    skip_kraken: bool,
    kraken_asset: str,
    n_pairs: int = 32,
    kraken_sims: int = 200,
    kraken_temp: float = 0.0,
    n_boot: int = 2000,
    expect_encoding: str = "v6_live2_ls",
) -> Dict[str, Any]:
    """Gate and run kraken eval. Returns dict with 'skipped' + optional result fields."""
    if skip_kraken:
        return {"skipped": True, "reason": "skip_kraken=True (operator flag)"}

    if not Path(kraken_asset).exists():
        return {
            "skipped": True,
            "reason": f"kraken asset absent: {kraken_asset} (F2 — tie to absent weights)",
        }

    if book_r5 is None:
        return {"skipped": True, "reason": "no --book-r5 supplied; kraken needs r5 book"}

    result_path = Path(out_dir) / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    return _run_kraken_eval(
        ckpt_path=ckpt_path,
        book_r5=book_r5,
        out_dir=out_dir,
        kraken_asset=kraken_asset,
        n_pairs=n_pairs,
        kraken_sims=kraken_sims,
        kraken_temp=kraken_temp,
        n_boot=n_boot,
        expect_encoding=expect_encoding,
    )


# ── Stage 3b: strix-g128 eval ────────────────────────────────────────────────


def _run_strix_eval(
    ckpt_path: str,
    book_r5: str,
    out_dir: str,
    strix_ckpt: str,
    n_pairs: int = 32,
    strix_n_sims: int = 128,
    n_boot: int = 2000,
    expect_encoding: str = "v6_live2_ls",
) -> Dict[str, Any]:
    """Run strix-g128 bar (128 sims, Gumbel-AZ, 32 pairs). Returns result dict."""
    from scripts.evalfair.head_vs_strix import run_head_vs_strix
    from scripts.evalfair.book import load_book

    book = load_book(Path(book_r5))
    return run_head_vs_strix(
        ckpt=ckpt_path,
        book=book,
        out_dir=out_dir,
        strix_ckpt=strix_ckpt,
        strix_n_sims=strix_n_sims,
        n_pairs=n_pairs,
        n_boot=n_boot,
        book_seed=book.get("seed", 20260710),
        expect_encoding=expect_encoding,
    )


def _stage3b_strix(
    ckpt_path: str,
    out_dir: str,
    book_r5: Optional[str],
    with_strix: bool,
    strix_ckpt: str,
    n_pairs: int = 32,
    strix_n_sims: int = 128,
    n_boot: int = 2000,
    expect_encoding: str = "v6_live2_ls",
) -> Dict[str, Any]:
    """Gate and run strix-g128 eval. Returns dict with 'skipped' + optional result fields.

    Mirrors _stage3_kraken exactly: skip-with-reason on flag / absent-asset / no-book.
    When enabled, delegates to strix_g128_child.py in strix's own venv (subprocess).
    """
    if not with_strix:
        return {"skipped": True, "reason": "with_strix=False (default OFF; pass --with-strix to enable)"}

    if not Path(strix_ckpt).exists():
        return {
            "skipped": True,
            "reason": f"strix asset absent: {strix_ckpt} (tie to absent weights — run3 from day one)",
        }

    if book_r5 is None:
        return {"skipped": True, "reason": "no --book-r5 supplied; strix needs r5 book"}

    result_path = Path(out_dir) / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    return _run_strix_eval(
        ckpt_path=ckpt_path,
        book_r5=book_r5,
        out_dir=out_dir,
        strix_ckpt=strix_ckpt,
        n_pairs=n_pairs,
        strix_n_sims=strix_n_sims,
        n_boot=n_boot,
        expect_encoding=expect_encoding,
    )


# ── Stage 4: series append + slope table ─────────────────────────────────────


def append_series_row(series_path: Path, row: Dict[str, Any]) -> None:
    """Append one JSONL row to the WP3 series file. Thread-safe append (single writer)."""
    series_path = Path(series_path)
    series_path.parent.mkdir(parents=True, exist_ok=True)
    with open(series_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _theil_sen(steps: List[int], vals: List[float]) -> float:
    """Theil-Sen slope estimator: median of pairwise slopes."""
    slopes = []
    for i in range(len(steps)):
        for j in range(i + 1, len(steps)):
            dx = steps[j] - steps[i]
            if dx != 0:
                slopes.append((vals[j] - vals[i]) / dx)
    return float(np.median(slopes)) if slopes else float("nan")


def _pair_bootstrap_ci(
    steps: List[int],
    vals: List[float],
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple:
    """Bootstrap CI on Theil-Sen slope for scalar metrics (jointly resampled step+val pairs).

    Resamples (step, val) JOINTLY so the paired relationship is preserved.
    For wr_d5 which has per-pair game scores, use _wr_d5_bootstrap_ci instead.
    """
    if len(steps) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    steps_a = np.asarray(steps, dtype=float)
    vals_a = np.asarray(vals, dtype=float)
    n = len(steps_a)
    boot_slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_slopes.append(_theil_sen(steps_a[idx].tolist(), vals_a[idx].tolist()))
    valid = [s for s in boot_slopes if not np.isnan(s)]
    if not valid:
        return float("nan"), float("nan")
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


def build_slope_table(
    rows: List[Dict[str, Any]],
    metrics: List[str],
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute Theil-Sen slope + bootstrap CI for each metric across the row series.

    Returns {metric: {theil_sen_slope, ci: [lo, hi], n_pts}} for each metric.
    Rows with None values for a metric are skipped for that metric.

    For 'wr_d5': uses pair_bootstrap_slope_ci over per-pair game scores (correct
    game-level resampling).  Requires rows to carry 'per_pair_scores_d5'.
    For all other scalar metrics: resamples (step, val) JOINTLY (correct).
    """
    from scripts.evalfair.compute_slope_report import pair_bootstrap_slope_ci  # tested, reused verbatim

    table: Dict[str, Any] = {}
    for metric in metrics:
        pts = [(r["step"], r.get(metric), r.get("per_pair_scores_d5")) for r in rows if r.get(metric) is not None]
        if len(pts) < 2:
            table[metric] = {"theil_sen_slope": float("nan"), "ci": [float("nan"), float("nan")], "n_pts": len(pts)}
            continue
        steps_m = [p[0] for p in pts]
        vals_m = [p[1] for p in pts]
        slope = _theil_sen(steps_m, vals_m)
        if metric == "wr_d5":
            per_pair = [p[2] for p in pts]
            if all(p is not None and len(p) > 0 for p in per_pair):
                ci_lo, ci_hi = pair_bootstrap_slope_ci(steps_m, per_pair, n_boot=n_boot, seed=seed)
            else:
                # Fallback: jointly-resampled scalar CI (no game-level data available)
                ci_lo, ci_hi = _pair_bootstrap_ci(steps_m, vals_m, n_boot=n_boot, seed=seed)
        else:
            ci_lo, ci_hi = _pair_bootstrap_ci(steps_m, vals_m, n_boot=n_boot, seed=seed)
        table[metric] = {"theil_sen_slope": slope, "ci": [ci_lo, ci_hi], "n_pts": len(pts)}
    return table


def _print_slope_table(table: Dict[str, Any]) -> None:
    print("\n[mantis] Slope table (Theil-Sen per 100k steps):")
    for metric, info in table.items():
        slope = info["theil_sen_slope"]
        ci = info["ci"]
        n = info["n_pts"]
        s100 = slope * 100_000 if not np.isnan(slope) else float("nan")
        ci_str = (
            f"[{ci[0]*100_000:.4f}, {ci[1]*100_000:.4f}]"
            if not (np.isnan(ci[0]) or np.isnan(ci[1]))
            else "[nan, nan]"
        )
        print(f"  {metric:35s}: {s100:+.4f}/100k-steps  CI={ci_str}  n={n}")


# ── Main orchestration ────────────────────────────────────────────────────────


def _build_wp3_row(
    ckpt_path: str,
    step: Optional[int],
    arm: str,
    vh_row: Dict[str, Any],
    d5_result: Dict[str, Any],
    kraken_result: Dict[str, Any],
    strix_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the per-bar WP3 series row."""
    # D5 fields
    wr_d5 = d5_result.get("wr")
    ci_d5 = d5_result.get("pair_ci")
    eff_n_d5 = d5_result.get("eff_n")
    radius = d5_result.get("radius")

    # Kraken fields
    if kraken_result.get("skipped"):
        wr_kraken = None
        ci_kraken = None
    else:
        wr_kraken = kraken_result.get("wr_head")
        ci_kraken = kraken_result.get("pair_ci")

    # Strix-g128 fields (optional bar — None when --with-strix absent or skipped)
    _strix = strix_result or {}
    if _strix.get("skipped") or not _strix:
        wr_strix = None
        ci_strix = None
    else:
        wr_strix = _strix.get("wr_head")
        ci_strix = _strix.get("pair_ci")

    return {
        "step": step or vh_row.get("step"),
        "arm": arm,
        "ckpt_sha": vh_row.get("ckpt_sha"),
        "encoding": vh_row.get("encoding", "v6_live2_ls"),
        "radius": radius,
        # Value health M1-M4
        "mean_v_on_losses": vh_row.get("mean_v_on_losses"),
        "ece": vh_row.get("ece"),
        "tail_mass_auc": vh_row.get("tail_mass_auc"),
        "decoded_auc": vh_row.get("decoded_auc"),
        "false_pessimism": vh_row.get("false_pessimism"),
        "n_loss": vh_row.get("n_loss"),
        "n_safe": vh_row.get("n_safe"),
        # D5 strength
        "wr_d5": wr_d5,
        "pair_ci_d5": ci_d5,
        "eff_n_d5": eff_n_d5,
        # Per-pair game scores (for correct pair_bootstrap_slope_ci on wr_d5)
        "per_pair_scores_d5": d5_result.get("per_pair_scores"),
        # Kraken strength
        "wr_kraken": wr_kraken,
        "pair_ci_kraken": ci_kraken,
        # Strix-g128 strength (D-K ceiling bar; None when --with-strix absent)
        "wr_strix": wr_strix,
        "pair_ci_strix": ci_strix,
    }


def _print_budget_table(
    n_ckpts: int,
    skip_kraken: bool,
    with_strix: bool,
    kraken_n_pairs: int,
    strix_n_pairs: int,
    kraken_sims: int,
    strix_n_sims: int,
) -> None:
    """Print per-bar budget (pairs x games, est wall) for every active bar.

    Wall estimates (laptop CPU, single-threaded, based on D-K timing report):
      d5 (SealBot depth-5): ~0.05 s/ply, 60 plies, 2 games/pair -> ~6 s/pair
      kraken-MCTS-200: ~0.25 s/move, 60 plies, 2 games/pair -> ~30 s/pair
      strix-g128: ~0.57 s/stone-search, 2 stones/turn, 30 turns, 2 games/pair -> ~68 s/pair
    These are ROUGH estimates; actual wall depends on position/hardware.
    """
    print("\n[mantis] Per-bar budget:")
    d5_pairs = 64
    d5_games = d5_pairs * 2
    d5_wall_est = d5_games * 60 * 0.05  # ~360 s
    print(f"  {'d5 (SealBot-d5)':30s}: {d5_pairs} pairs x {d5_games} games  est ~{d5_wall_est/60:.0f} min/ckpt")

    if not skip_kraken:
        k_games = kraken_n_pairs * 2
        k_wall_est = k_games * 60 * 0.25  # ~30 s/pair
        print(f"  {'kraken-MCTS-'+str(kraken_sims):30s}: {kraken_n_pairs} pairs x {k_games} games  est ~{k_wall_est/60:.0f} min/ckpt")
    else:
        print(f"  {'kraken':30s}: SKIPPED (--skip-kraken)")

    if with_strix:
        s_games = strix_n_pairs * 2
        s_wall_est = strix_n_pairs * 2 * 30 * 2 * 0.57  # ~68 s/pair
        print(f"  {'strix-g128 (D-K ceiling +313 Elo)':30s}: {strix_n_pairs} pairs x {s_games} games  est ~{s_wall_est/60:.0f} min/ckpt  [venv delegation]")
    else:
        print(f"  {'strix-g128':30s}: SKIPPED (default OFF; pass --with-strix to enable)")

    total_min_per_ckpt = d5_wall_est / 60
    if not skip_kraken:
        total_min_per_ckpt += (kraken_n_pairs * 2 * 60 * 0.25) / 60
    if with_strix:
        total_min_per_ckpt += (strix_n_pairs * 2 * 30 * 2 * 0.57) / 60
    print(f"  {'TOTAL est':30s}: ~{total_min_per_ckpt:.0f} min/ckpt x {n_ckpts} ckpts = ~{total_min_per_ckpt*n_ckpts:.0f} min")
    print()


def run_pull_eval(
    ckpt_paths: List[str],
    series_out: str,
    retro_out: str,
    book_r4: Optional[str],
    book_r5: Optional[str],
    skip_kraken: bool = False,
    with_strix: bool = False,
    workers: int = 1,
    n_boot: int = 2000,
    expect_encoding: str = "v6_live2_ls",
    probe_path: Optional[str] = None,
    negatives_path: Optional[str] = None,
    games_path: Optional[str] = None,
    wp2_games: Optional[Dict[str, str]] = None,
    no_sha_check: bool = False,
    kraken_asset: Optional[str] = None,
    kraken_n_pairs: int = 32,
    kraken_sims: int = 200,
    kraken_temp: float = 0.0,
    strix_ckpt: Optional[str] = None,
    strix_n_pairs: int = 32,
    strix_n_sims: int = 128,
    t7_series_out: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Process each checkpoint in order: Stage 1 -> Stage 2 -> Stage 3 -> Stage 3b -> Stage 4.

    Serial, niced. workers=1 enforced (load-34 crash record at higher values).
    Returns list of emitted WP3 series rows.

    Schema separation:
      - Stage 1 T7 rows -> t7_series_out  (default: reports/e1/t7_value_health_series.jsonl)
      - Stage 4 WP3 rows -> series_out    (default: reports/e1/wp3_series.jsonl)
    These are SEPARATE files to avoid incompatible-schema dual-write.

    strix-g128 bar (Stage 3b): gated behind with_strix=True (default OFF).
    When enabled, delegates to strix_g128_child.py in strix's dedicated venv.
    Byte-identical to prior behavior when with_strix=False.
    """
    workers = 1  # mandatory — load-34 crash on record at higher

    default_kraken = str(_REPO / "checkpoints/external/kraken_v1.pt")
    kraken_path = kraken_asset or default_kraken

    default_strix = str(_REPO / "strix_checkpoint_00237000.pt")
    strix_path = strix_ckpt or default_strix

    # Stage 1 T7 rows go to a dedicated file (separate schema from WP3)
    t7_path = str(
        Path(t7_series_out)
        if t7_series_out is not None
        else _REPO / "reports/e1/t7_value_health_series.jsonl"
    )

    # Budget table: print once before processing
    _print_budget_table(
        n_ckpts=len(ckpt_paths),
        skip_kraken=skip_kraken,
        with_strix=with_strix,
        kraken_n_pairs=kraken_n_pairs,
        strix_n_pairs=strix_n_pairs,
        kraken_sims=kraken_sims,
        strix_n_sims=strix_n_sims,
    )

    emitted: List[Dict[str, Any]] = []

    for ckpt_path in ckpt_paths:
        ckpt = Path(ckpt_path)
        stem = ckpt.stem
        print(f"\n[mantis] processing {stem} ...", flush=True)

        # Infer arm from checkpoint payload
        arm = infer_arm_from_ckpt(ckpt_path)
        step_raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        step = int(step_raw["step"]) if "step" in step_raw else None

        # Stage 1: value-health — T7 schema rows go to t7_path (NOT series_out)
        print(f"[mantis] Stage 1: value-health ({arm}) ...", flush=True)
        vh_row = stage1_value_health(
            ckpt_path=ckpt_path,
            arm=arm,
            series_out=t7_path,
            probe_path=probe_path,
            negatives_path=negatives_path,
            games_path=games_path,
            wp2_games=wp2_games,
            no_sha_check=no_sha_check,
        )
        print(
            f"[mantis] Stage 1 done: M1(mean_v_on_losses)={vh_row.get('mean_v_on_losses'):.4f}  "
            f"M2(ece)={vh_row.get('ece'):.4f}  n_loss={vh_row.get('n_loss')}  n_safe={vh_row.get('n_safe')}",
            flush=True,
        )

        # Stage 2: d5 fair-book eval
        d5_out_dir = str(Path(retro_out) / stem)
        print(f"[mantis] Stage 2: d5 eval (64 pairs, deploy-matched) ...", flush=True)
        d5_result = stage2_d5_eval(
            ckpt_path=ckpt_path,
            book_r4=book_r4,
            book_r5=book_r5,
            out_dir=d5_out_dir,
            workers=workers,
            n_boot=n_boot,
            expect_encoding=expect_encoding,
        )
        wr_d5 = d5_result.get("wr", float("nan"))
        ci_d5 = d5_result.get("pair_ci", [float("nan"), float("nan")])
        print(
            f"[mantis] Stage 2 done: WR={wr_d5:.3f}  CI=[{ci_d5[0]:.3f},{ci_d5[1]:.3f}]  "
            f"radius={d5_result.get('radius')}  eff_n={d5_result.get('eff_n')}",
            flush=True,
        )

        # Stage 3: kraken eval
        kraken_out_dir = str(Path(retro_out) / stem / "kraken")
        print(
            f"[mantis] Stage 3: kraken eval (skip={skip_kraken}, asset_exists={Path(kraken_path).exists()}) ...",
            flush=True,
        )
        kraken_result = _stage3_kraken(
            ckpt_path=ckpt_path,
            out_dir=kraken_out_dir,
            book_r5=book_r5,
            skip_kraken=skip_kraken,
            kraken_asset=kraken_path,
            n_pairs=kraken_n_pairs,
            kraken_sims=kraken_sims,
            kraken_temp=kraken_temp,
            n_boot=n_boot,
            expect_encoding=expect_encoding,
        )
        if kraken_result.get("skipped"):
            print(f"[mantis] Stage 3 skipped: {kraken_result.get('reason')}", flush=True)
        else:
            wr_k = kraken_result.get("wr_head", float("nan"))
            ci_k = kraken_result.get("pair_ci", [float("nan"), float("nan")])
            print(
                f"[mantis] Stage 3 done: WR_head={wr_k:.3f}  CI=[{ci_k[0]:.3f},{ci_k[1]:.3f}]",
                flush=True,
            )

        # Stage 3b: strix-g128 eval (optional D-K ceiling bar)
        strix_out_dir = str(Path(retro_out) / stem / "strix_g128")
        print(
            f"[mantis] Stage 3b: strix-g128 eval (with_strix={with_strix}, asset_exists={Path(strix_path).exists()}) ...",
            flush=True,
        )
        strix_result = _stage3b_strix(
            ckpt_path=ckpt_path,
            out_dir=strix_out_dir,
            book_r5=book_r5,
            with_strix=with_strix,
            strix_ckpt=strix_path,
            n_pairs=strix_n_pairs,
            strix_n_sims=strix_n_sims,
            n_boot=n_boot,
            expect_encoding=expect_encoding,
        )
        if strix_result.get("skipped"):
            print(f"[mantis] Stage 3b skipped: {strix_result.get('reason')}", flush=True)
        else:
            wr_s = strix_result.get("wr_head", float("nan"))
            ci_s = strix_result.get("pair_ci", [float("nan"), float("nan")])
            print(
                f"[mantis] Stage 3b done: WR_head={wr_s:.3f}  CI=[{ci_s[0]:.3f},{ci_s[1]:.3f}]  "
                f"(vs strix-g128 +313 Elo D-K ceiling)",
                flush=True,
            )

        # Stage 4: assemble + emit per-bar WP3 row to series_out (WP3 schema only)
        wp3_row = _build_wp3_row(
            ckpt_path=ckpt_path,
            step=step,
            arm=arm,
            vh_row=vh_row,
            d5_result=d5_result,
            kraken_result=kraken_result,
            strix_result=strix_result,
        )
        append_series_row(Path(series_out), wp3_row)
        emitted.append(wp3_row)

        # Idempotency: mark done AFTER successful series append
        _mark_done(ckpt, {"step": step, "arm": arm})

        print(
            f"[mantis] Step {step}: verdict bar "
            f"M1={wp3_row.get('mean_v_on_losses'):.4f}  "
            f"WR_d5={wp3_row.get('wr_d5'):.3f}  "
            f"WR_kraken={wp3_row.get('wr_kraken')}  "
            f"WR_strix={wp3_row.get('wr_strix')}",
            flush=True,
        )

    # Slope table over emitted series
    if len(emitted) >= 2:
        slope_metrics = ["mean_v_on_losses", "wr_d5", "ece", "false_pessimism"]
        table = build_slope_table(emitted, metrics=slope_metrics, n_boot=n_boot)
        _print_slope_table(table)

    return emitted


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="WP3 mantis on-demand external eval reader (value-health + d5 + kraken + optional strix-g128)"
    )
    # Checkpoint selection
    ap.add_argument("--ckpt", default=None, help="Target a specific checkpoint .pt")
    ap.add_argument("--ckpt-dir", default=None, help="Directory of banked checkpoints (default scan)")

    # Books
    ap.add_argument(
        "--book-r4", default=str(_REPO / "tests/fixtures/opening_books/evalfair_r4_v2.json"),
        dest="book_r4",
    )
    ap.add_argument(
        "--book-r5", default=str(_REPO / "tests/fixtures/opening_books/evalfair_r5_v2.json"),
        dest="book_r5",
    )

    # Output paths
    ap.add_argument(
        "--series-out", default=str(_REPO / "reports/e1/wp3_series.jsonl"),
        dest="series_out",
        help="WP3-schema JSONL output (one row per ckpt). DO NOT share with --t7-series-out.",
    )
    ap.add_argument(
        "--t7-series-out", default=None,
        dest="t7_series_out",
        help="T7-schema JSONL output for stage1 value-health rows "
             "(default: reports/e1/t7_value_health_series.jsonl). Separate file from --series-out.",
    )
    ap.add_argument(
        "--retro-out", default=str(_REPO / "reports/evalfair/retro_slope"),
        dest="retro_out",
    )

    # Kraken
    ap.add_argument("--skip-kraken", action="store_true", dest="skip_kraken",
                    help="Skip Stage 3 kraken eval (quick runs)")
    ap.add_argument(
        "--kraken-asset",
        default=str(_REPO / "checkpoints/external/kraken_v1.pt"),
        dest="kraken_asset",
    )

    # Strix-g128 (optional D-K ceiling bar)
    ap.add_argument("--with-strix", action="store_true", dest="with_strix",
                    help="Enable Stage 3b strix-g128 bar (D-K ceiling +313 Elo). "
                         "Requires strix venv (hexo_rs + torch_geometric) at "
                         "STRIX_REPO/.venv (default /home/timmy/Work/Hexo/hexo-strix). "
                         "Default OFF — run3 bar from day one.")
    ap.add_argument(
        "--strix-ckpt",
        default=str(_REPO / "strix_checkpoint_00237000.pt"),
        dest="strix_ckpt",
        help="strix-g128 checkpoint path (default: strix_checkpoint_00237000.pt in repo root)",
    )
    ap.add_argument("--strix-n-pairs", type=int, default=32, dest="strix_n_pairs",
                    help="Number of pairs for strix-g128 bar (default: 32)")
    ap.add_argument("--strix-n-sims", type=int, default=128, dest="strix_n_sims",
                    help="strix Gumbel sim budget (default: 128 — D-K canonical eval tier)")

    # Vast pull
    ap.add_argument("--vast-host", default=None, dest="vast_host",
                    help="vast.ai SSH host (e.g. user@12.34.56.78). Omit to skip pull.")
    ap.add_argument(
        "--vast-path", default="/workspace/hexo_rl/checkpoints/run2_retro/",
        dest="vast_path",
    )
    ap.add_argument("--pull-timeout", type=int, default=300, dest="pull_timeout")

    # Probe paths
    ap.add_argument("--probe", default=None, dest="probe_path")
    ap.add_argument("--negatives", default=None, dest="negatives_path")
    ap.add_argument("--games", default=None, dest="games_path")
    ap.add_argument("--no-sha-check", action="store_true", dest="no_sha_check",
                    help="Skip frozen-probe SHA guard (dev only)")

    # Boot + misc
    ap.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    ap.add_argument("--expect-encoding", default="v6_live2_ls", dest="expect_encoding")

    args = ap.parse_args()

    # ── Stage 0: rsync pull ─────────────────────────────────────────────────
    ckpt_dir_path: Optional[Path] = None
    if args.ckpt_dir:
        ckpt_dir_path = Path(args.ckpt_dir)
    elif args.ckpt is None:
        ckpt_dir_path = _REPO / "checkpoints/run2_retro"

    if args.vast_host and ckpt_dir_path is not None:
        print(f"[mantis] Stage 0: rsync pull from {args.vast_host}:{args.vast_path} ...", flush=True)
        ok = rsync_pull(
            host=args.vast_host,
            remote_path=args.vast_path,
            local_path=str(ckpt_dir_path),
            timeout=args.pull_timeout,
        )
        if not ok:
            print("[mantis] rsync pull failed — continuing with local files.", file=sys.stderr)

    # Collect checkpoints to process
    if args.ckpt:
        ckpt_paths = [args.ckpt]
    elif ckpt_dir_path is not None and ckpt_dir_path.exists():
        ckpt_paths = [str(p) for p in collect_new_ckpts(ckpt_dir_path)]
        if not ckpt_paths:
            print("[mantis] no new checkpoints to process (all done-marked).")
            return
    else:
        ap.error("Supply --ckpt <path> or --ckpt-dir <dir>")
        return

    # ── Stages 1-4 ──────────────────────────────────────────────────────────
    # nice the process (best-effort)
    try:
        os.nice(10)
    except OSError:
        pass

    run_pull_eval(
        ckpt_paths=ckpt_paths,
        series_out=args.series_out,
        retro_out=args.retro_out,
        book_r4=args.book_r4,
        book_r5=args.book_r5,
        skip_kraken=args.skip_kraken,
        with_strix=args.with_strix,
        workers=1,
        n_boot=args.n_boot,
        expect_encoding=args.expect_encoding,
        probe_path=args.probe_path,
        negatives_path=args.negatives_path,
        games_path=args.games_path,
        no_sha_check=args.no_sha_check,
        kraken_asset=args.kraken_asset,
        strix_ckpt=args.strix_ckpt,
        strix_n_pairs=args.strix_n_pairs,
        strix_n_sims=args.strix_n_sims,
        t7_series_out=args.t7_series_out,
    )


if __name__ == "__main__":
    main()
