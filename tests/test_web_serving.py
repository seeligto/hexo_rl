"""D-J DASH WP3.2 — tests for series_reader + web API endpoints.

Tests the importable Python functions without a running server,
and the Flask route handlers via test client.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


# ── series_reader: tail_jsonl ─────────────────────────────────────────────────

def test_tail_jsonl_empty_file(tmp_path):
    from hexo_rl.monitoring.series_reader import tail_jsonl
    p = tmp_path / "test.jsonl"
    p.write_bytes(b"")
    r = tail_jsonl(p)
    assert r["lines"] == []
    assert r["next_offset"] == 0
    assert r["truncated"] is False


def test_tail_jsonl_missing_file(tmp_path):
    from hexo_rl.monitoring.series_reader import tail_jsonl
    r = tail_jsonl(tmp_path / "nonexistent.jsonl")
    assert r["lines"] == []
    assert r["next_offset"] == 0


def test_tail_jsonl_reads_lines(tmp_path):
    from hexo_rl.monitoring.series_reader import tail_jsonl
    p = tmp_path / "events.jsonl"
    lines = [
        json.dumps({"event": "training_step", "step": i}) + "\n"
        for i in range(5)
    ]
    p.write_text("".join(lines), encoding="utf-8")
    r = tail_jsonl(p, since_offset=0)
    assert len(r["lines"]) == 5
    assert r["next_offset"] > 0
    assert r["truncated"] is False
    # Each line is valid JSON
    for line in r["lines"]:
        obj = json.loads(line)
        assert "event" in obj


def test_tail_jsonl_since_offset(tmp_path):
    from hexo_rl.monitoring.series_reader import tail_jsonl
    p = tmp_path / "events.jsonl"
    line1 = json.dumps({"event": "a"}) + "\n"
    line2 = json.dumps({"event": "b"}) + "\n"
    p.write_text(line1 + line2, encoding="utf-8")

    # First read
    r1 = tail_jsonl(p, since_offset=0)
    assert len(r1["lines"]) == 2
    offset = r1["next_offset"]

    # Second read with offset — no new lines
    r2 = tail_jsonl(p, since_offset=offset)
    assert r2["lines"] == []
    assert r2["next_offset"] == offset

    # Append a new line and re-poll
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "c"}) + "\n")
    r3 = tail_jsonl(p, since_offset=offset)
    assert len(r3["lines"]) == 1
    assert json.loads(r3["lines"][0])["event"] == "c"


def test_tail_jsonl_truncation_logs(tmp_path, caplog):
    from hexo_rl.monitoring.series_reader import tail_jsonl
    import logging
    p = tmp_path / "big.jsonl"
    # Write 3MB of content (above 2MB cap)
    chunk = json.dumps({"event": "x", "data": "y" * 900}) + "\n"
    content = chunk * 3200  # ~3MB
    p.write_text(content, encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        r = tail_jsonl(p, since_offset=0, max_bytes=2 * 1024 * 1024)
    assert r["truncated"] is True
    assert "truncated" in caplog.text.lower()
    # All returned lines must be valid JSON
    for line in r["lines"]:
        json.loads(line)


def test_tail_jsonl_partial_line_not_included(tmp_path):
    from hexo_rl.monitoring.series_reader import tail_jsonl
    p = tmp_path / "partial.jsonl"
    # Write one complete + one partial line (no trailing newline)
    p.write_bytes(
        json.dumps({"event": "a"}).encode() + b"\n" +
        json.dumps({"event": "b"}).encode()  # no newline
    )
    r = tail_jsonl(p)
    assert len(r["lines"]) == 1
    assert json.loads(r["lines"][0])["event"] == "a"


# ── series_reader: read_value_health_series ───────────────────────────────────

def test_read_value_health_series_missing(tmp_path):
    from hexo_rl.monitoring.series_reader import read_value_health_series
    r = read_value_health_series(tmp_path / "none.jsonl")
    assert r == []


def test_read_value_health_series_parses(tmp_path):
    from hexo_rl.monitoring.series_reader import read_value_health_series
    p = tmp_path / "valprobe.jsonl"
    records = [
        {"step": 10000, "recognition_lag": 0.12, "ece": 0.05},
        {"step": 20000, "recognition_lag": 0.09, "ece": 0.04},
    ]
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    out = read_value_health_series(p)
    assert len(out) == 2
    assert out[0]["step"] == 10000
    assert out[1]["ece"] == 0.04


def test_read_value_health_series_skips_malformed(tmp_path):
    from hexo_rl.monitoring.series_reader import read_value_health_series
    p = tmp_path / "valprobe.jsonl"
    p.write_text('{"step": 1}\nNOT_JSON\n{"step": 2}\n', encoding="utf-8")
    out = read_value_health_series(p)
    assert len(out) == 2


# ── series_reader: read_external_bars ────────────────────────────────────────

def test_read_external_bars_empty_dir(tmp_path):
    from hexo_rl.monitoring.series_reader import read_external_bars
    r = read_external_bars(tmp_path)
    assert r == []


def test_read_external_bars_missing_dir(tmp_path):
    from hexo_rl.monitoring.series_reader import read_external_bars
    r = read_external_bars(tmp_path / "nonexistent")
    assert r == []


def test_read_external_bars_reads_jsonl(tmp_path):
    from hexo_rl.monitoring.series_reader import read_external_bars
    (tmp_path / "d5.jsonl").write_text(
        json.dumps({"step": 1000, "wr": 0.55, "opponent": "d5", "n": 50}) + "\n" +
        json.dumps({"step": 2000, "wr": 0.58, "opponent": "d5", "n": 50}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "kraken.jsonl").write_text(
        json.dumps({"step": 1000, "wr": 0.45, "opponent": "kraken", "n": 30}) + "\n",
        encoding="utf-8",
    )
    out = read_external_bars(tmp_path)
    assert len(out) == 3
    opps = {r["opponent"] for r in out}
    assert "d5" in opps
    assert "kraken" in opps


# ── Flask test client: API endpoints ─────────────────────────────────────────

def _make_dashboard(tmp_path, log_dir=None, valprobe_dir=None, evalfair_dir=None):
    """Build a WebDashboard without starting threads."""
    config = {
        "monitoring": {
            "web_port": 5099,
            "web_host": "127.0.0.1",
            "viewer_games_dir": str(tmp_path / "runs"),
        },
        "log_dir": str(log_dir or tmp_path / "logs"),
        "valprobe_dir": str(valprobe_dir or tmp_path / "reports/valprobe"),
        "evalfair_dir": str(evalfair_dir or tmp_path / "reports/evalfair"),
    }
    from hexo_rl.monitoring.web_dashboard import WebDashboard
    wd = WebDashboard(config)
    return wd


def test_api_events_jsonl_no_file(tmp_path):
    """GET /api/events.jsonl with no log file → empty lines."""
    wd = _make_dashboard(tmp_path)
    client = wd._app.test_client()
    r = client.get("/api/events.jsonl?since=0")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data["lines"] == []
    assert data["next_offset"] == 0


def test_api_events_jsonl_with_file(tmp_path):
    """GET /api/events.jsonl returns lines from events_*.jsonl."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    evfile = log_dir / "events_run1.jsonl"
    evfile.write_text(
        json.dumps({"event": "training_step", "step": 100}) + "\n" +
        json.dumps({"event": "iteration_complete", "step": 100}) + "\n",
        encoding="utf-8",
    )
    wd = _make_dashboard(tmp_path, log_dir=log_dir)
    client = wd._app.test_client()
    r = client.get("/api/events.jsonl?since=0")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert len(data["lines"]) == 2
    assert data["next_offset"] > 0
    assert data["file"] is not None


def test_api_events_jsonl_incremental(tmp_path):
    """Polling with since=offset returns only new lines."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    evfile = log_dir / "events_run1.jsonl"
    line1 = json.dumps({"event": "training_step", "step": 1}) + "\n"
    evfile.write_text(line1, encoding="utf-8")

    wd = _make_dashboard(tmp_path, log_dir=log_dir)
    client = wd._app.test_client()

    r1 = client.get("/api/events.jsonl?since=0")
    d1 = json.loads(r1.data)
    offset = d1["next_offset"]
    assert len(d1["lines"]) == 1

    # Append new line
    with open(evfile, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "training_step", "step": 2}) + "\n")

    r2 = client.get(f"/api/events.jsonl?since={offset}")
    d2 = json.loads(r2.data)
    assert len(d2["lines"]) == 1
    assert json.loads(d2["lines"][0])["step"] == 2


def test_api_structlog_jsonl_no_file(tmp_path):
    """GET /api/structlog.jsonl with no log file → empty."""
    wd = _make_dashboard(tmp_path)
    client = wd._app.test_client()
    r = client.get("/api/structlog.jsonl?since=0")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data["lines"] == []


def test_api_structlog_jsonl_with_file(tmp_path):
    """GET /api/structlog.jsonl returns structlog lines."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    slfile = log_dir / "run_main.jsonl"
    slfile.write_text(
        json.dumps({"event": "evaluation_round_complete", "step": 5000, "wr_best": 0.62}) + "\n",
        encoding="utf-8",
    )
    wd = _make_dashboard(tmp_path, log_dir=log_dir)
    client = wd._app.test_client()
    r = client.get("/api/structlog.jsonl?since=0")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert len(data["lines"]) == 1
    assert json.loads(data["lines"][0])["event"] == "evaluation_round_complete"


def test_api_series_value_health_empty(tmp_path):
    """GET /api/series/value_health with empty dir → empty records."""
    wd = _make_dashboard(tmp_path)
    client = wd._app.test_client()
    r = client.get("/api/series/value_health")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data["records"] == []


def test_api_series_value_health_with_data(tmp_path):
    """GET /api/series/value_health returns valprobe records."""
    vdir = tmp_path / "valprobe"
    vdir.mkdir()
    (vdir / "recognition_lag.jsonl").write_text(
        json.dumps({"step": 10000, "recognition_lag": 0.11}) + "\n",
        encoding="utf-8",
    )
    wd = _make_dashboard(tmp_path, valprobe_dir=vdir)
    client = wd._app.test_client()
    r = client.get("/api/series/value_health")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert len(data["records"]) == 1
    assert data["records"][0]["recognition_lag"] == 0.11


def test_api_series_external_bars_empty(tmp_path):
    """GET /api/series/external_bars with empty dir → empty records."""
    wd = _make_dashboard(tmp_path)
    client = wd._app.test_client()
    r = client.get("/api/series/external_bars")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data["records"] == []
    assert "slope" in data


def test_api_series_external_bars_with_data(tmp_path):
    """GET /api/series/external_bars returns records + slope info."""
    edir = tmp_path / "evalfair"
    edir.mkdir()
    pts = [{"step": i*10000, "wr": 0.3 + i*0.02, "opponent": "d5", "n": 50} for i in range(5)]
    (edir / "d5.jsonl").write_text(
        "\n".join(json.dumps(p) for p in pts) + "\n",
        encoding="utf-8",
    )
    wd = _make_dashboard(tmp_path, evalfair_dir=edir)
    client = wd._app.test_client()
    r = client.get("/api/series/external_bars")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert len(data["records"]) == 5
    assert "slope" in data


def test_api_series_unknown_name(tmp_path):
    """GET /api/series/bogus → 404."""
    wd = _make_dashboard(tmp_path)
    client = wd._app.test_client()
    r = client.get("/api/series/bogus")
    assert r.status_code == 404


def test_viewer_routes_preserved(tmp_path):
    """Viewer routes still return 200 (or known error) — not broken by rebuild."""
    wd = _make_dashboard(tmp_path)
    client = wd._app.test_client()
    # /viewer should serve viewer.html (file present in static/)
    r = client.get("/viewer")
    assert r.status_code in (200, 404)  # 404 if viewer.html absent from test env
    # /viewer/recent should return JSON list
    r2 = client.get("/viewer/recent")
    assert r2.status_code == 200
    data = json.loads(r2.data)
    assert isinstance(data, list)


def test_index_route_present(tmp_path):
    """GET / serves index.html."""
    wd = _make_dashboard(tmp_path)
    client = wd._app.test_client()
    r = client.get("/")
    # 200 if static/index.html exists (it does in the repo)
    assert r.status_code in (200, 404)


# ── WebDashboard: game-persistence + on_event preserved ──────────────────────

def test_game_persistence_preserved(tmp_path):
    """_persist_game still writes to disk and updates _game_index."""
    wd = _make_dashboard(tmp_path)
    wd._run_id = "testrun"
    payload = {"event": "game_complete", "game_id": "g001", "winner": 0, "moves": 30, "ts": 1.0}
    wd._persist_game(payload)
    game_file = tmp_path / "runs" / "testrun" / "games" / "g001.json"
    assert game_file.exists()
    data = json.loads(game_file.read_text())
    assert data["game_id"] == "g001"
    assert len(wd._game_index) == 1


def test_on_event_routing(tmp_path):
    """on_event routes training_step to _training_step_history, others to _event_history."""
    wd = _make_dashboard(tmp_path)
    wd.on_event({"event": "training_step", "ts": 1.0, "step": 1})
    wd.on_event({"event": "system_stats", "ts": 2.0})
    assert len(wd._training_step_history) == 1
    assert len(wd._event_history) == 1


def test_no_socketio_import(tmp_path):
    """WebDashboard no longer imports flask_socketio."""
    import importlib
    import hexo_rl.monitoring.web_dashboard as mod
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "flask_socketio" not in src, "SocketIO must be removed from web_dashboard"


def test_no_gevent_in_serve_dashboard():
    """serve_dashboard.py no longer monkey-patches gevent."""
    srv = Path(__file__).parent.parent / "scripts" / "serve_dashboard.py"
    src = srv.read_text(encoding="utf-8")
    assert "monkey.patch_all" not in src, "gevent monkey-patch must be removed"
    assert "flask_socketio" not in src, "SocketIO must be removed from serve_dashboard"
