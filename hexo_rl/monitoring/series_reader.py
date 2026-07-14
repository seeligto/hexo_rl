"""Series reader — importable, no-server JSONL tail + valprobe/evalfair series parsers.

All functions are pure I/O; no Flask, no threads. Unit-testable without a running server.

ONE-RESOLVER LAW:
  sealbot_slope + parse_feed  → hexo_rl.monitoring.run_feed_reader (NOT reimplemented here)
  theil_sen_slope / pair_bootstrap_slope_ci → scripts.evalfair.retro_slope (NOT reimplemented)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_FIRST_LOAD_MAX_BYTES = 2 * 1024 * 1024  # 2 MB cap on first-load backfill


def tail_jsonl(
    path: str | Path,
    since_offset: int = 0,
    max_bytes: int = _FIRST_LOAD_MAX_BYTES,
) -> dict:
    """Read newline-terminated JSONL lines from *path* starting at byte offset.

    First-load (since_offset=0): caps at max_bytes from the END of the file,
    logs a warning when truncation occurs (no silent caps principle).

    Returns:
        {"lines": [str, ...], "next_offset": int, "truncated": bool}

    Each element of "lines" is a raw JSON string (caller parses).
    """
    p = Path(path)
    if not p.exists():
        return {"lines": [], "next_offset": 0, "truncated": False}

    try:
        file_size = p.stat().st_size
    except OSError:
        return {"lines": [], "next_offset": since_offset, "truncated": False}

    truncated = False
    start_offset = since_offset

    if since_offset == 0 and file_size > max_bytes:
        # Seek to (file_size - max_bytes), then skip to first newline to get
        # a clean line boundary.
        start_offset = file_size - max_bytes
        truncated = True
        log.warning(
            "tail_jsonl first-load truncated: file=%s size=%d cap=%d dropped_bytes=%d",
            str(p), file_size, max_bytes, start_offset,
        )

    lines: list[str] = []
    try:
        with open(p, "rb") as f:
            f.seek(start_offset)
            if truncated:
                # Skip partial first line
                f.readline()
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    # Partial trailing line — don't include, retry next poll
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    lines.append(text)
            next_offset = f.tell()
    except OSError as exc:
        log.warning("tail_jsonl read error: %s %s", str(p), exc)
        return {"lines": [], "next_offset": since_offset, "truncated": False}

    return {"lines": lines, "next_offset": next_offset, "truncated": truncated}


def read_value_health_series(valprobe_jsonl_path: str | Path) -> list[dict]:
    """Parse valprobe JSONL output into a list of dicts.

    Each dict has keys from the probe output (step, recognition_lag, ece, …).
    Missing/malformed lines silently skipped.
    """
    p = Path(valprobe_jsonl_path)
    if not p.exists():
        return []
    records: list[dict] = []
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        records.append(rec)
                except json.JSONDecodeError:
                    pass
    except OSError as exc:
        log.warning("read_value_health_series: %s %s", str(p), exc)
    return records


def read_external_bars(evalfair_dir: str | Path) -> list[dict]:
    """Read evalfair JSONL output files and return a merged list of records.

    Looks for *.jsonl files in evalfair_dir. Each record is expected to have
    at minimum: step, wr (win rate), opponent (e.g. "d5" or "kraken").
    Stage boundary metadata is preserved via "stage_boundary" key if present.
    """
    d = Path(evalfair_dir)
    if not d.exists():
        return []
    records: list[dict] = []
    try:
        jsonl_files = sorted(d.glob("*.jsonl"))
    except OSError:
        return []
    for jf in jsonl_files:
        try:
            with open(jf, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict):
                            records.append(rec)
                    except json.JSONDecodeError:
                        pass
        except OSError as exc:
            log.warning("read_external_bars: %s %s", str(jf), exc)
    return records


def compute_external_slope(records: list[dict]) -> dict[str, Any]:
    """Compute Theil-Sen slope + CI over external bar records.

    ONE-RESOLVER: delegates to scripts.evalfair.retro_slope.

    Returns dict with keys: slope, ci_low, ci_high (or error if import fails).
    """
    try:
        _repo = Path(__file__).resolve().parents[2]
        if str(_repo) not in sys.path:
            sys.path.insert(0, str(_repo))
        from scripts.evalfair.retro_slope import theil_sen_slope, pair_bootstrap_slope_ci
    except ImportError as exc:
        return {"error": f"retro_slope import failed: {exc}"}

    steps = [r.get("step", 0) for r in records if "step" in r and "wr" in r]
    wrs = [r["wr"] for r in records if "step" in r and "wr" in r]
    if len(steps) < 2:
        return {"slope": None, "ci_low": None, "ci_high": None, "n": len(steps)}

    slope = theil_sen_slope(steps, wrs)
    # pair_bootstrap_slope_ci resamples per-checkpoint PAIR SCORES (a list per
    # ckpt); bar records carry only the aggregate wr, so unless the JSONL rows
    # include pair_scores a bootstrap CI is uncomputable — report None rather
    # than fabricate one from scalars.
    pair_scores = [r.get("pair_scores") for r in records if "step" in r and "wr" in r]
    if all(isinstance(ps, (list, tuple)) and len(ps) > 0 for ps in pair_scores):
        ci_low, ci_high = pair_bootstrap_slope_ci(steps, pair_scores)
    else:
        ci_low = ci_high = None
    return {"slope": slope, "ci_low": ci_low, "ci_high": ci_high, "n": len(steps)}
