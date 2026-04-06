"""
Tests for structlog JSONL file sink in configure_logging.

Verifies:
  - configure_logging returns (logger, file_handle) tuple
  - JSONL file is created under logs/
  - Each emitted line is valid JSON with an 'event' key
  - train.py naming convention: run_name='train_{run_id}' → logs/train_{run_id}.jsonl
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
import structlog


def test_structlog_file_sink_creates_jsonl(tmp_path: Path):
    """configure_logging creates a JSONL file and each line is valid JSON with 'event'."""
    structlog.reset_defaults()

    from hexo_rl.monitoring.configure import configure_logging

    log, log_fh = configure_logging(
        log_dir=str(tmp_path),
        run_name="test_run",
        console=False,
    )
    try:
        log.info("test_event", key="value")
        log.info("second_event", step=1)
        log_fh.flush()
    finally:
        log_fh.close()
        structlog.reset_defaults()

    log_file = tmp_path / "test_run.jsonl"
    assert log_file.exists(), f"Expected JSONL at {log_file}"

    lines = [l for l in log_file.read_text().splitlines() if l.strip()]
    assert len(lines) >= 2, f"Expected at least 2 log lines, got {len(lines)}"
    for line in lines:
        data = json.loads(line)
        assert "event" in data, f"Each line must have an 'event' key; got: {data}"


def test_structlog_file_sink_returns_file_handle(tmp_path: Path):
    """configure_logging must return a (logger, file_handle) tuple."""
    structlog.reset_defaults()

    from hexo_rl.monitoring.configure import configure_logging

    result = configure_logging(
        log_dir=str(tmp_path), run_name="handle_test", console=False
    )
    assert isinstance(result, tuple) and len(result) == 2, (
        "configure_logging must return (log, file_handle)"
    )
    _, fh = result
    assert hasattr(fh, "close"), "Second element must be a file-like object"
    try:
        fh.close()
    finally:
        structlog.reset_defaults()


def test_structlog_file_sink_train_prefix(tmp_path: Path):
    """train.py passes run_name='train_{run_id}' → file is logs/train_{run_id}.jsonl."""
    structlog.reset_defaults()

    from hexo_rl.monitoring.configure import configure_logging

    run_id = uuid.uuid4().hex
    run_name = f"train_{run_id}"
    log, fh = configure_logging(log_dir=str(tmp_path), run_name=run_name, console=False)
    try:
        log.info("startup", step=0)
        fh.flush()
    finally:
        fh.close()
        structlog.reset_defaults()

    log_file = tmp_path / f"train_{run_id}.jsonl"
    assert log_file.exists(), f"Expected log file at {log_file}"
    events = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
    event_names = [e["event"] for e in events]
    assert "startup" in event_names, f"Expected 'startup' event in {event_names}"
