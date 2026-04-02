"""
HEX BOT -- Clean Training Dashboard

Based on saiki77's hexbot training dashboard.
Zero game-specific imports. Accepts data via REST API, WebSocket, or Python API.

Black-and-white minimalist dashboard with live game visualization,
ELO progression, loss curves, and resource monitoring.

Usage:
    Standalone:   python dashboard.py
    As import:    from dashboard import Dashboard
                  dash = Dashboard(port=5001)
                  dash.start()
                  dash.add_game(moves=[[0,0],[1,1]], result=1.0)
                  dash.add_metric(iteration=1, loss=0.5, elo=1050)
"""

from __future__ import annotations

import json
import multiprocessing
import statistics
import threading
import time
from collections import deque
from datetime import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
from flask import Flask, Response, jsonify, request
from flask_socketio import SocketIO

from hexo_rl.monitoring.replay_poller import GameReplayPoller


# ---------------------------------------------------------------------------
# Resource monitor
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """Track CPU and RAM usage over time."""

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self._lock = threading.Lock()
        self._history: List[dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            self.snapshot()
            time.sleep(self.interval)

    def snapshot(self) -> dict:
        cpu_pct = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        snap = {
            "cpu_pct": round(cpu_pct, 1),
            "ram_pct": round(mem.percent, 1),
            "ram_used_gb": round(mem.used / (1024**3), 1),
            "ram_total_gb": round(mem.total / (1024**3), 1),
            "cpu_count": multiprocessing.cpu_count(),
            "ts": time.time(),
        }
        with self._lock:
            self._history.append(snap)
            if len(self._history) > 600:
                self._history = self._history[-600:]
        return snap

    def get_history(self, n: int = 120) -> List[dict]:
        with self._lock:
            return list(self._history[-n:])

    @property
    def latest(self) -> dict:
        with self._lock:
            return self._history[-1] if self._history else {}


# ---------------------------------------------------------------------------
# Data stores (thread-safe, in-memory)
# ---------------------------------------------------------------------------

class DataStore:
    """Holds all dashboard state: games, metrics, resources."""

    def __init__(self):
        self._lock = threading.Lock()
        # Games
        self.games: List[dict] = []  # last 50
        self.total_games: int = 0
        # Metrics per iteration
        self.metrics: List[dict] = []
        # ELO history
        self.elo_history: List[dict] = [{"iteration": 0, "elo": 1000.0}]
        # Loss history
        self.loss_history: List[dict] = []
        # Win rate history
        self.winrate_history: List[dict] = []
        # Game length history
        self.gamelength_history: List[dict] = []
        # Speed history
        self.speed_history: List[dict] = []
        # Rolling game-length deque (compound moves, maxlen=200) for median
        self._game_length_deque: deque = deque(maxlen=200)
        self._game_length_median: Optional[int] = None
        # Aggregate stats
        self.current_iteration: int = 0
        self.current_elo: float = 1000.0
        self.total_samples: int = 0
        self.workers: int = multiprocessing.cpu_count()
        self.pretrained_weight: Optional[float] = None
        # Per-iteration accumulators
        self._iter_wins = [0, 0, 0]  # p0, p1, draw
        self._iter_lengths: List[int] = []

    # -- Games --

    def add_game(self, moves: list, result: float,
                 metadata: Optional[dict] = None) -> dict:
        with self._lock:
            self.total_games += 1
            idx = self.total_games
            entry = {
                "game_idx": idx,
                "moves": moves,
                "result": result,
                "num_moves": len(moves),
                "metadata": metadata or {},
                "ts": time.time(),
            }
            self.games.append(entry)
            if len(self.games) > 50:
                self.games = self.games[-50:]
            # Accumulate win stats
            if result > 0:
                self._iter_wins[0] += 1
            elif result < 0:
                self._iter_wins[1] += 1
            else:
                self._iter_wins[2] += 1
            self._iter_lengths.append(len(moves))
            return entry

    def recent_games(self, n: int = 10) -> List[dict]:
        with self._lock:
            return list(self.games[-n:])

    # -- Metrics --

    def add_metric(self, **kw) -> dict:
        with self._lock:
            m = dict(kw)
            m.setdefault("ts", time.time())
            self.metrics.append(m)
            if len(self.metrics) > 2000:
                self.metrics = self.metrics[-2000:]
            # Update convenience fields
            if "iteration" in m:
                self.current_iteration = m["iteration"]
            if "elo" in m and m["elo"] is not None:
                self.current_elo = m["elo"]
                self.elo_history.append({
                    "iteration": m.get("iteration", self.current_iteration),
                    "elo": round(m["elo"], 1),
                })
            if "loss" in m and m["loss"] is not None:
                loss_entry = {"iteration": m.get("iteration", self.current_iteration)}
                if isinstance(m["loss"], dict):
                    loss_entry.update(m["loss"])
                else:
                    loss_entry["total"] = m["loss"]
                if m.get("policy_loss") is not None:
                    loss_entry["policy"] = m["policy_loss"]
                if m.get("value_loss") is not None:
                    loss_entry["value"] = m["value_loss"]
                self.loss_history.append(loss_entry)
            if "wins" in m:
                w = m["wins"]
                total = sum(w) or 1
                self.winrate_history.append({
                    "iteration": m.get("iteration", self.current_iteration),
                    "p0": round(w[0] / total * 100, 1),
                    "p1": round(w[1] / total * 100, 1),
                    "draw": round(w[2] / total * 100, 1) if len(w) > 2 else 0,
                })
            if "avg_game_length" in m:
                self.gamelength_history.append({
                    "iteration": m.get("iteration", self.current_iteration),
                    "avg_game_length": m["avg_game_length"],
                })
            if "games" in m and "self_play_time" in m and m["self_play_time"] > 0:
                self.speed_history.append({
                    "iteration": m.get("iteration", self.current_iteration),
                    "games_per_sec": round(m["games"] / m["self_play_time"], 2),
                })
            if "workers" in m:
                self.workers = m["workers"]
            if "pretrained_weight" in m and m["pretrained_weight"] is not None:
                self.pretrained_weight = float(m["pretrained_weight"])
            # Reset per-iteration accumulators on new iteration
            self._iter_wins = [0, 0, 0]
            self._iter_lengths = []
            return m

    def get_stats(self) -> dict:
        with self._lock:
            latest = self.metrics[-1] if self.metrics else {}
            total_w = sum(self._iter_wins) or 1
            avg_len = (
                round(sum(self._iter_lengths) / len(self._iter_lengths))
                if self._iter_lengths else 0
            )
            return {
                "iteration": self.current_iteration,
                "total_games": self.total_games,
                "current_elo": round(self.current_elo, 1),
                "workers": self.workers,
                "avg_game_length": avg_len,
                "win_p0": round(self._iter_wins[0] / total_w * 100, 1),
                "win_p1": round(self._iter_wins[1] / total_w * 100, 1),
                "latest_loss": latest.get("loss"),
                "latest_policy_loss": latest.get("policy_loss"),
                "latest_value_loss": latest.get("value_loss"),
                "fill_pct": latest.get("fill_pct"),
                "positions_hr": latest.get("positions_hr"),
                "buffer_size": latest.get("buffer_size", 0),
                "self_play_time": latest.get("self_play_time", 0),
                "game_length_median": self._game_length_median,
                "pretrained_weight": self.pretrained_weight,
                "sims_per_sec": latest.get("sims_per_sec"),
            }

    def record_game_length(self, compound_moves: int) -> None:
        """Thread-safe: append a compound move count and refresh cached median."""
        with self._lock:
            self._game_length_deque.append(compound_moves)
            data = sorted(self._game_length_deque)
            self._game_length_median = int(statistics.median(data)) if data else None

    def get_elo_history(self) -> List[dict]:
        with self._lock:
            return list(self.elo_history)

    def get_loss_history(self) -> List[dict]:
        with self._lock:
            return list(self.loss_history)

    def get_winrate_history(self) -> List[dict]:
        with self._lock:
            return list(self.winrate_history)

    def get_gamelength_history(self) -> List[dict]:
        with self._lock:
            return list(self.gamelength_history)

    def get_speed_history(self) -> List[dict]:
        with self._lock:
            return list(self.speed_history)



# ---------------------------------------------------------------------------
# Dashboard class (Python API)
# ---------------------------------------------------------------------------

class Dashboard:
    """Self-contained training dashboard server.

    Usage::

        dash = Dashboard(port=5001)
        dash.start()                          # background thread
        dash.add_game([[0,0],[1,1]], 1.0)     # push game
        dash.add_metric(iteration=1, loss=0.5, elo=1050)
        dash.update_progress(step=10, total=100, loss=0.32)
    """

    def __init__(self, port: int = 5001, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.store = DataStore()
        self.resource_monitor = ResourceMonitor()
        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = "hexbot-dashboard"
        self.socketio = SocketIO(
            self.app, cors_allowed_origins="*", async_mode="threading"
        )
        self._replay_poller = GameReplayPoller(
            replay_dir="logs/replays", poll_interval_s=30.0, cache_cap=50
        )
        self._corpus_cache: List[dict] = []
        self._corpus_lock = threading.Lock()
        self._server_thread: Optional[threading.Thread] = None
        self._setup_routes()
        self._setup_socket_events()

    # -- Public API --

    def start(self) -> None:
        """Start dashboard server in a background thread."""
        self.resource_monitor.start()
        self._replay_poller.start()
        self._server_thread = threading.Thread(target=self._run, daemon=True)
        self._server_thread.start()
        ts = _dt.now().strftime("%H:%M:%S")
        print(f"[{ts}] Dashboard running at http://localhost:{self.port}")

    def _run(self) -> None:
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        self.socketio.run(
            self.app, host=self.host, port=self.port,
            debug=False, allow_unsafe_werkzeug=True, log_output=False,
        )

    def add_game(self, moves: list, result: float,
                 metadata: Optional[dict] = None) -> None:
        """Push a completed game for display and replay."""
        entry = self.store.add_game(moves, result, metadata)
        self.socketio.emit("game_complete", {
            "game_idx": entry["game_idx"],
            "total_games": self.store.total_games,
            "result": result,
            "num_moves": len(moves),
            "moves": moves,
            "metadata": metadata or {},
        })
        self.socketio.emit("stats_update", self.store.get_stats())

    def add_metric(self, **kwargs: Any) -> None:
        """Push training metrics (iteration, loss, elo, wins, etc)."""
        self.store.add_metric(**kwargs)
        self.socketio.emit("stats_update", self.store.get_stats())

    def update_progress(self, step: int, total: int,
                        loss: Optional[float] = None,
                        phase: str = "training") -> None:
        """Update progress bar on the dashboard."""
        pct = round(step / max(total, 1) * 100, 1)
        self.socketio.emit("train_progress", {
            "step": step,
            "total": total,
            "loss": round(loss, 4) if loss is not None else None,
            "pct": pct,
            "phase": phase,
        })

    # -- Routes --

    def _setup_routes(self) -> None:
        app = self.app
        store = self.store
        rmon = self.resource_monitor

        @app.route("/")
        def index():
            return Response(DASHBOARD_HTML, content_type="text/html")

        # --- GET endpoints ---

        @app.route("/api/stats")
        def api_stats():
            s = store.get_stats()
            r = rmon.latest
            s["cpu_pct"] = r.get("cpu_pct", 0)
            s["ram_pct"] = r.get("ram_pct", 0)
            return jsonify(s)

        @app.route("/api/elo")
        def api_elo():
            return jsonify(store.get_elo_history())

        @app.route("/api/losses")
        def api_losses():
            return jsonify(store.get_loss_history())

        @app.route("/api/games")
        def api_games():
            n = request.args.get("n", 10, type=int)
            return jsonify(store.recent_games(n))

        @app.route("/api/resources")
        def api_resources():
            return jsonify({
                "current": rmon.latest,
                "history": rmon.get_history(),
            })

        @app.route("/api/winrates")
        def api_winrates():
            return jsonify(store.get_winrate_history())

        @app.route("/api/speed")
        def api_speed():
            return jsonify(store.get_speed_history())

        @app.route("/api/gamelength")
        def api_gamelength():
            return jsonify(store.get_gamelength_history())

        @app.route("/api/replays")
        def api_replays():
            n = request.args.get("n", 10, type=int)
            games = self._replay_poller.get_recent(n=n)
            return jsonify([
                {
                    "key": f"{g.filename}:{i}",
                    "filename": g.filename,
                    "game_length": g.game_length,
                    "winner": g.winner,
                    "timestamp": g.timestamp,
                    "checkpoint_step": g.checkpoint_step,
                    "num_moves": len(g.move_sequence),
                }
                for i, g in enumerate(games)
            ])

        @app.route("/api/replays/<path:key>")
        def api_replay_detail(key: str):
            game = self._replay_poller.get_game(key)
            if game is None:
                return jsonify({"error": "not found"}), 404
            return jsonify({
                "filename": game.filename,
                "game_length": game.game_length,
                "winner": game.winner,
                "timestamp": game.timestamp,
                "checkpoint_step": game.checkpoint_step,
                "moves": game.move_sequence,
            })

        # --- POST endpoints ---

        @app.route("/api/game", methods=["POST"])
        def api_post_game():
            d = request.get_json(force=True)
            moves = d.get("moves", [])
            result = d.get("result", 0.0)
            meta = {k: v for k, v in d.items() if k not in ("moves", "result")}
            self.add_game(moves, result, meta)
            return jsonify({"ok": True, "game_idx": store.total_games})

        @app.route("/api/metric", methods=["POST"])
        def api_post_metric():
            d = request.get_json(force=True)
            self.add_metric(**d)
            return jsonify({"ok": True})

        # --- Corpus preview endpoints ---

        @app.route("/api/reload-corpus", methods=["POST"])
        def api_reload_corpus():
            corpus_path = Path("/tmp/hexo_corpus_preview.jsonl")
            if not corpus_path.exists():
                return jsonify({"loaded": 0})
            entries = []
            for line in corpus_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(entries) >= 100:
                    break
            with self._corpus_lock:
                self._corpus_cache = entries
            return jsonify({"loaded": len(entries)})

        @app.route("/api/corpus-replays")
        def api_corpus_replays():
            with self._corpus_lock:
                games = list(self._corpus_cache)
            return jsonify([
                {
                    "key": f"corpus:{g.get('game_id', i)}",
                    "game_id": g.get("game_id", ""),
                    "game_length": g.get("game_length", 0),
                    "winner": g.get("outcome", "unknown"),
                    "timestamp": g.get("timestamp", ""),
                    "source": g.get("source", ""),
                    "num_moves": len(g.get("moves", [])),
                }
                for i, g in enumerate(games)
            ])

        @app.route("/api/corpus-replays/<path:key>")
        def api_corpus_replay_detail(key: str):
            game_id = key.replace("corpus:", "", 1)
            with self._corpus_lock:
                match = next(
                    (g for g in self._corpus_cache
                     if g.get("game_id") == game_id),
                    None,
                )
            if match is None:
                return jsonify({"error": "not found"}), 404
            return jsonify({
                "game_id": match.get("game_id", ""),
                "game_length": match.get("game_length", 0),
                "winner": match.get("outcome", "unknown"),
                "timestamp": match.get("timestamp", ""),
                "source": match.get("source", ""),
                "moves": match.get("moves", []),
            })

    # -- Socket events --

    def _setup_socket_events(self) -> None:
        sio = self.socketio
        store = self.store

        @sio.on("connect")
        def on_connect():
            sio.emit("stats_update", store.get_stats())

        @sio.on("game_result")
        def on_game_result(data):
            moves = data.get("moves", [])
            result = data.get("result", 0.0)
            meta = {k: v for k, v in data.items() if k not in ("moves", "result")}
            self.add_game(moves, result, meta)

        @sio.on("metric")
        def on_metric(data):
            self.add_metric(**data)


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HEX BOT</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#fff;color:#000;font-family:'SF Mono','Courier New',monospace;font-size:13px;line-height:1.5;
  display:flex;flex-direction:column;height:100vh;overflow:hidden}
header{border-bottom:1px solid #000;padding:14px 24px;display:flex;align-items:center;gap:20px}
.title{font-size:15px;font-weight:700;letter-spacing:4px;text-transform:uppercase}
.status{margin-left:auto;letter-spacing:3px;font-size:11px;font-weight:700}
main{display:flex;flex:1;overflow:hidden}
.left{width:50%;border-right:1px solid #000;padding:20px;display:flex;flex-direction:column;align-items:center;overflow:hidden}
.right{width:50%;display:flex;flex-direction:column;overflow-y:auto}
.chart-box{padding:16px 20px;border-bottom:1px solid #000;display:flex;flex-direction:column}
.chart-box:last-child{border-bottom:none}
.chart-header{font-weight:700;letter-spacing:2px;font-size:10px;text-transform:uppercase;
  cursor:pointer;user-select:none;display:flex;align-items:center;gap:6px}
.chart-header .toggle{font-size:8px;transition:transform .15s}
.chart-header .toggle.collapsed{transform:rotate(-90deg)}
.chart-body{overflow:hidden;transition:max-height .25s ease,opacity .2s ease}
.chart-body.collapsed{max-height:0!important;opacity:0;padding:0}
.canvas-wrap{position:relative;min-height:180px;height:180px;margin-top:8px}
.canvas-wrap canvas{position:absolute;top:0;left:0;width:100%;height:100%}
#hex-canvas-wrap{flex:1;position:relative;width:100%;min-height:0}
#hex-canvas{position:absolute;top:0;left:0;width:100%;height:100%;border:1px solid #eee}
.game-info{font-size:11px;letter-spacing:1px;min-height:18px;margin-top:8px;flex-shrink:0}
footer{border-top:2px solid #000;font:11px 'SF Mono','Courier New',monospace;flex-shrink:0}
.frow{display:flex;align-items:center;gap:0;padding:5px 20px;border-bottom:1px solid #eee}
.frow:last-child{border-bottom:none}
.fgroup-label{font-weight:700;letter-spacing:2px;font-size:9px;text-transform:uppercase;
  width:76px;flex-shrink:0;color:#888}
.frow b{font-weight:700}
.frow span{white-space:nowrap;margin-right:16px}
.sep{color:#ccc;margin-right:16px}
.res-bar{display:flex;gap:4px;align-items:center;margin-right:16px}
.res-meter{width:36px;height:7px;border:1px solid #000;display:inline-block;position:relative;vertical-align:middle}
.res-meter-fill{height:100%;background:#000;transition:width .3s}
#progress-wrap{width:100%;margin-bottom:8px;display:none}
.progress-bar{display:flex;align-items:center;gap:8px}
.progress-track{flex:1;height:6px;background:#eee;border-radius:3px;overflow:hidden}
.progress-fill{height:100%;background:#000;width:0%;transition:width .2s}
.progress-label{font:700 10px 'Courier New';min-width:120px;text-align:right}
.settings-btn{cursor:pointer;font-size:18px;margin-left:12px;user-select:none}
.settings-panel{position:absolute;top:42px;right:20px;background:#fff;border:1px solid #000;
  padding:16px;z-index:100;font:11px 'SF Mono','Courier New',monospace;min-width:260px;box-shadow:2px 2px 0 #0001}
.settings-row{display:flex;align-items:center;gap:8px;margin-bottom:10px}
.settings-row label{width:110px;font-weight:700;flex-shrink:0}
.settings-row input[type=range]{flex:1;accent-color:#000}
.settings-row span.val{width:50px;text-align:right;font-size:10px}
.gh-item{cursor:pointer;padding:1px 3px;border-radius:2px;display:inline-block;margin:0 1px;font-size:8px}
.gh-item:hover{background:#eee}
.gh-item.active{background:#000;color:#fff}
.conn-dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-left:8px;vertical-align:middle}
</style>
</head>
<body>
<header>
  <span class="title">Hex Bot Training</span>
  <span class="status" id="status">IDLE</span>
  <span class="conn-dot" id="conn-dot" style="background:#ccc" title="Disconnected"></span>
  <span class="settings-btn" onclick="toggleSettings()">&#9881;</span>
</header>
<div id="settings-panel" class="settings-panel" style="display:none">
  <div style="font-weight:700;margin-bottom:12px;letter-spacing:2px;font-size:10px;text-transform:uppercase">Settings</div>
  <div class="settings-row">
    <label>Replay speed</label>
    <input type="range" id="set-speed" min="50" max="500" value="120" step="10"
      oninput="saveSetting('replaySpeed',+this.value);el('set-speed-val').textContent=this.value+'ms'">
    <span class="val" id="set-speed-val">120ms</span>
  </div>
  <div class="settings-row">
    <label>Dot size</label>
    <input type="range" id="set-dotsize" min="1" max="5" value="2" step="0.5"
      oninput="saveSetting('dotSize',+this.value);el('set-dotsize-val').textContent=this.value;drawHex()">
    <span class="val" id="set-dotsize-val">2</span>
  </div>
  <div class="settings-row">
    <label>Grid radius</label>
    <input type="range" id="set-radius" min="1" max="4" value="2" step="1"
      oninput="saveSetting('emptyHexRadius',+this.value);el('set-radius-val').textContent=this.value;drawHex()">
    <span class="val" id="set-radius-val">2</span>
  </div>
  <div class="settings-row">
    <label>Move numbers</label>
    <input type="checkbox" id="set-movenums" checked
      onchange="saveSetting('showMoveNums',this.checked);drawHex()">
  </div>
  <div class="settings-row">
    <label>Auto-refresh</label>
    <input type="checkbox" id="set-autorefresh" checked
      onchange="saveSetting('autoRefresh',this.checked)">
  </div>
</div>
<main>
  <div class="left">
    <div id="progress-wrap">
      <div class="progress-bar">
        <div class="progress-track">
          <div class="progress-fill" id="progress-fill"></div>
        </div>
        <span class="progress-label" id="progress-label"></span>
      </div>
    </div>
    <div style="font-weight:700;letter-spacing:2px;font-size:10px;text-transform:uppercase;margin-bottom:4px">
      Training Game #<span id="game-num">&mdash;</span>
    </div>
    <div id="hex-canvas-wrap"><canvas id="hex-canvas"></canvas></div>
    <div class="game-info" id="game-info">Waiting for games...</div>
    <div id="game-history" style="width:100%;margin-top:6px;max-height:48px;overflow-y:auto;overflow-x:hidden;
      font:9px 'SF Mono',monospace;border-top:1px solid #eee;padding-top:4px;line-height:1.6">
    </div>
    <div style="font:8px 'SF Mono',monospace;color:#bbb;margin-top:2px">
      Space: pause &middot; R: restart
    </div>
  </div>
  <div class="right">
    <div class="chart-box" id="box-elo">
      <div class="chart-header" onclick="toggleChart('elo')">
        <span class="toggle" id="tog-elo">&#9660;</span> Elo Progression
      </div>
      <div class="chart-body" id="body-elo">
        <div class="canvas-wrap"><canvas id="elo-chart"></canvas></div>
      </div>
    </div>
    <div class="chart-box" id="box-loss">
      <div class="chart-header" onclick="toggleChart('loss')">
        <span class="toggle" id="tog-loss">&#9660;</span> Loss Curves
      </div>
      <div class="chart-body" id="body-loss">
        <div class="canvas-wrap"><canvas id="loss-chart"></canvas></div>
      </div>
    </div>
    <div class="chart-box" id="box-winrate">
      <div class="chart-header" onclick="toggleChart('winrate')">
        <span class="toggle" id="tog-winrate">&#9660;</span> Win Rates
      </div>
      <div class="chart-body" id="body-winrate">
        <div class="canvas-wrap"><canvas id="winrate-chart"></canvas></div>
      </div>
    </div>
    <div class="chart-box" id="box-gamelength">
      <div class="chart-header" onclick="toggleChart('gamelength')">
        <span class="toggle" id="tog-gamelength">&#9660;</span> Game Length
      </div>
      <div class="chart-body" id="body-gamelength">
        <div class="canvas-wrap"><canvas id="gamelength-chart"></canvas></div>
      </div>
    </div>
    <div class="chart-box" id="box-speed">
      <div class="chart-header" onclick="toggleChart('speed')">
        <span class="toggle" id="tog-speed">&#9660;</span> Training Speed
      </div>
      <div class="chart-body" id="body-speed">
        <div class="canvas-wrap"><canvas id="speed-chart"></canvas></div>
      </div>
    </div>
    <div class="chart-box" id="box-replays">
      <div class="chart-header" onclick="toggleChart('replays')">
        <span class="toggle" id="tog-replays">&#9660;</span> Recent Self-Play Games
      </div>
      <div class="chart-body" id="body-replays">
        <div id="replay-list" style="font-family:monospace;font-size:13px;padding:8px;max-height:200px;overflow-y:auto;">Loading...</div>
      </div>
    </div>
    <div class="chart-box" id="box-corpus" style="display:none">
      <div class="chart-header" onclick="toggleChart('corpus')">
        <span class="toggle" id="tog-corpus">&#9660;</span> Corpus Games
      </div>
      <div class="chart-body" id="body-corpus">
        <div id="corpus-list" style="font-family:monospace;font-size:13px;padding:8px;max-height:200px;overflow-y:auto;">No corpus games loaded</div>
      </div>
    </div>
  </div>
</main>
<footer>
  <div class="frow">
    <span class="fgroup-label">Training</span>
    <span>Iter <b id="s-iter">—</b></span>
    <span class="sep">·</span>
    <span>Loss <b id="s-loss">—</b></span>
    <span class="sep">·</span>
    <span>Policy <b id="s-ploss">—</b></span>
    <span class="sep">·</span>
    <span>Value <b id="s-vloss">—</b></span>
    <span class="sep">·</span>
    <span>Buffer <b id="s-fill">—</b></span>
    <span class="sep">·</span>
    <span>Pos/hr <b id="s-poshr">—</b></span>
    <span class="sep">·</span>
    <span>Corpus mix <b id="s-pmix">—</b></span>
  </div>
  <div class="frow">
    <span class="fgroup-label">Games</span>
    <span><b id="s-games">0</b> total</span>
    <span class="sep">·</span>
    <span>Avg <b id="s-len">—</b> moves</span>
    <span class="sep">·</span>
    <span>Win P0:<b id="s-w0">—</b>% P1:<b id="s-w1">—</b>%</span>
    <span class="sep">·</span>
    <span>Workers <b id="s-workers">—</b></span>
  </div>
  <div class="frow">
    <span class="fgroup-label">System</span>
    <span>Elo <b id="s-elo">1000</b></span>
    <span class="sep">·</span>
    <span class="res-bar">CPU <div class="res-meter"><div class="res-meter-fill" id="cpu-fill" style="width:0%"></div></div> <b id="s-cpu">0</b>%</span>
    <span class="sep">·</span>
    <span class="res-bar">RAM <div class="res-meter"><div class="res-meter-fill" id="ram-fill" style="width:0%"></div></div> <b id="s-ram">0</b>%</span>
  </div>
</footer>

<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js"></script>
<script>
// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
const DPR = window.devicePixelRatio || 1;
const socket = io();
socket.on('connect', () => {
  el('conn-dot').style.background = '#0a0'; el('conn-dot').title = 'Connected';
});
socket.on('disconnect', () => {
  el('conn-dot').style.background = '#c00'; el('conn-dot').title = 'Disconnected';
});

let stones0 = [], stones1 = [], moveOrder = [];  // moveOrder[i] = {q, r, num, player}

function el(id) { return document.getElementById(id); }
function setInfo(t) { el('game-info').textContent = t; }

// ---------------------------------------------------------------------------
// HiDPI canvas helper
// ---------------------------------------------------------------------------
function sizeCanvas(cv) {
  const r = cv.parentElement.getBoundingClientRect();
  const w = Math.round(r.width), h = Math.round(r.height);
  if (cv.width !== w * DPR || cv.height !== h * DPR) {
    cv.width = w * DPR; cv.height = h * DPR;
    cv.style.width = w + 'px'; cv.style.height = h + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }
  return { w, h };
}

// ---------------------------------------------------------------------------
// Collapsible charts
// ---------------------------------------------------------------------------
const chartState = {};
function toggleChart(name) {
  const body = el('body-' + name);
  const tog = el('tog-' + name);
  const collapsed = !body.classList.contains('collapsed');
  if (collapsed) {
    body.classList.add('collapsed');
    tog.classList.add('collapsed');
  } else {
    body.classList.remove('collapsed');
    tog.classList.remove('collapsed');
    // Redraw after expand
    setTimeout(fetchCharts, 50);
  }
  chartState[name] = collapsed;
}

// ---------------------------------------------------------------------------
// Hex canvas rendering
// ---------------------------------------------------------------------------
const HEX_SIZE = 18;
const S3 = Math.sqrt(3);

// Settings (persisted to localStorage)
const defaultSettings = { replaySpeed: 120, dotSize: 2, emptyHexRadius: 2, autoRefresh: true, showMoveNums: false };
let settings = Object.assign({}, defaultSettings);
try { const s = JSON.parse(localStorage.getItem('hexdash_settings')); if (s) Object.assign(settings, s); } catch(e) {}
function saveSetting(key, val) { settings[key] = val; localStorage.setItem('hexdash_settings', JSON.stringify(settings)); }
function toggleSettings() {
  const p = el('settings-panel');
  p.style.display = p.style.display === 'none' ? '' : 'none';
}

function axToPixel(q, r) {
  return [HEX_SIZE * (S3 * q + S3 / 2 * r), HEX_SIZE * (1.5 * r)];
}

function drawHex() {
  const cv = el('hex-canvas');
  const { w: W, h: H } = sizeCanvas(cv);
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const all = [...stones0, ...stones1];
  if (!all.length) {
    ctx.fillStyle = '#aaa';
    ctx.font = '12px Courier New';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for training game...', W / 2, H / 2);
    return;
  }

  // Compute bounding box
  let mnX = 1e9, mxX = -1e9, mnY = 1e9, mxY = -1e9;
  for (const [q, r] of all) {
    const [px, py] = axToPixel(q, r);
    if (px < mnX) mnX = px; if (px > mxX) mxX = px;
    if (py < mnY) mnY = py; if (py > mxY) mxY = py;
  }
  const mg = HEX_SIZE * 3;
  const spanX = mxX - mnX + mg * 2, spanY = mxY - mnY + mg * 2;
  const sc = Math.min(W / spanX, H / spanY, 2.5);
  const ox = W / 2 - (mnX + mxX) / 2 * sc, oy = H / 2 - (mnY + mxY) / 2 * sc;

  function toS(q, r) {
    const [px, py] = axToPixel(q, r);
    return [px * sc + ox, py * sc + oy];
  }
  function hexPath(cx, cy, sz) {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = Math.PI / 3 * i - Math.PI / 6;
      const hx = cx + sz * Math.cos(a), hy = cy + sz * Math.sin(a);
      i === 0 ? ctx.moveTo(hx, hy) : ctx.lineTo(hx, hy);
    }
    ctx.closePath();
  }
  const hr = HEX_SIZE * sc * 0.88;

  // Empty hex dots around placed stones
  const stoneSet = new Set(all.map(s => s[0] + ',' + s[1]));
  const emptySet = new Set();
  const dotR = settings.emptyHexRadius;
  for (const [q, r] of all) {
    for (let dq = -dotR; dq <= dotR; dq++) {
      for (let dr = -dotR; dr <= dotR; dr++) {
        if (Math.abs(dq) + Math.abs(dr) > dotR + 1) continue;
        const key = (q + dq) + ',' + (r + dr);
        if (!stoneSet.has(key) && !emptySet.has(key)) {
          emptySet.add(key);
          const [sx, sy] = toS(q + dq, r + dr);
          ctx.fillStyle = '#ccc';
          ctx.beginPath();
          ctx.arc(sx, sy, Math.max(1.5, settings.dotSize * sc), 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }

  // P0: solid black hexagons
  for (const [q, r] of stones0) {
    const [sx, sy] = toS(q, r);
    hexPath(sx, sy, hr);
    ctx.fillStyle = '#000'; ctx.fill();
    ctx.strokeStyle = '#000'; ctx.lineWidth = 1; ctx.stroke();
  }

  // P1: white hexagons with hatching
  for (const [q, r] of stones1) {
    const [sx, sy] = toS(q, r);
    hexPath(sx, sy, hr);
    ctx.fillStyle = '#fff'; ctx.fill();
    ctx.strokeStyle = '#000'; ctx.lineWidth = 1.2; ctx.stroke();
    ctx.save();
    hexPath(sx, sy, hr); ctx.clip();
    ctx.strokeStyle = '#000'; ctx.lineWidth = 0.6;
    const step = Math.max(3, 4 * sc);
    for (let d = -hr * 2; d <= hr * 2; d += step) {
      ctx.beginPath();
      ctx.moveTo(sx + d - hr, sy - hr);
      ctx.lineTo(sx + d + hr, sy + hr);
      ctx.stroke();
    }
    ctx.restore();
  }

  // Move numbers on stones
  if (settings.showMoveNums && moveOrder.length) {
    const fontSize = Math.max(7, Math.min(12, hr * 0.7));
    ctx.font = 'bold ' + fontSize + 'px Courier New';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (const mv of moveOrder) {
      const [sx, sy] = toS(mv.q, mv.r);
      ctx.fillStyle = mv.player === 0 ? '#fff' : '#000';
      ctx.fillText(mv.num, sx, sy);
    }
  }
}

// ---------------------------------------------------------------------------
// Game replay
// ---------------------------------------------------------------------------
let replayTimer = null, replayBusy = false, pendingGame = null;
let gameStats = { w0: 0, w1: 0, totalLen: 0, count: 0 };

// Shared replay state for keyboard stepping
let replayPaused = false, currentGameData = null, replayMoveIdx = 0;

// Rebuild board state from moves up to index n
function rebuildBoard(moves, n) {
  stones0 = []; stones1 = []; moveOrder = [];
  let p = 0, stt = 0;
  for (let i = 0; i < n; i++) {
    const m = moves[i];
    if (p === 0) stones0.push(m); else stones1.push(m);
    moveOrder.push({ q: m[0], r: m[1], num: i + 1, player: p });
    stt++;
    const need = (i === 0) ? 1 : 2;
    if (stt >= need) { p = 1 - p; stt = 0; }
  }
}

function replayAdvance() {
  if (!currentGameData) return;
  const d = currentGameData, moves = d.moves;
  if (replayMoveIdx >= moves.length) {
    // Game finished
    if (replayTimer) { clearInterval(replayTimer); replayTimer = null; }
    const winner = d.result > 0 ? 'P0 (black)' : 'P1 (white)';
    setInfo('Game #' + d.game_idx + ': ' + winner + ' wins in ' + moves.length + ' moves');
    setTimeout(() => {
      replayBusy = false;
      if (!replayPaused && pendingGame) { const g = pendingGame; pendingGame = null; replayGame(g); }
    }, 800);
    return;
  }
  replayMoveIdx++;
  rebuildBoard(moves, replayMoveIdx);
  drawHex();
  setInfo('Game #' + d.game_idx + ' \u2014 Move ' + replayMoveIdx + '/' + moves.length +
    (replayPaused ? ' [PAUSED]' : ''));
}

function replayGame(d) {
  replayBusy = true;
  replayPaused = false;
  currentGameData = d;
  replayMoveIdx = 0;
  stones0 = []; stones1 = []; moveOrder = [];
  drawHex();
  el('game-num').textContent = d.game_idx;
  setInfo('Game #' + d.game_idx + ' playing... (' + d.moves.length + ' moves)');
  if (replayTimer) clearInterval(replayTimer);
  replayTimer = setInterval(replayAdvance, settings.replaySpeed);
}

// ---------------------------------------------------------------------------
// Keyboard controls
// ---------------------------------------------------------------------------
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') {
    e.preventDefault();
    if (replayPaused) {
      // Resume auto-advance
      replayPaused = false;
      if (currentGameData && replayMoveIdx < currentGameData.moves.length) {
        if (replayTimer) clearInterval(replayTimer);
        replayTimer = setInterval(replayAdvance, settings.replaySpeed);
      }
      setInfo(el('game-info').textContent.replace(' [PAUSED]', ''));
    } else {
      // Pause
      replayPaused = true;
      if (replayTimer) { clearInterval(replayTimer); replayTimer = null; }
      setInfo(el('game-info').textContent + ' [PAUSED]');
    }
  }
  if (e.code === 'ArrowRight' && currentGameData) {
    e.preventDefault();
    // Pause auto-advance and step forward
    replayPaused = true;
    if (replayTimer) { clearInterval(replayTimer); replayTimer = null; }
    if (replayMoveIdx < currentGameData.moves.length) {
      replayMoveIdx++;
      rebuildBoard(currentGameData.moves, replayMoveIdx);
      drawHex();
      setInfo('Game #' + currentGameData.game_idx + ' \u2014 Move ' + replayMoveIdx + '/' + currentGameData.moves.length + ' [PAUSED]');
    }
  }
  if (e.code === 'ArrowLeft' && currentGameData) {
    e.preventDefault();
    // Pause auto-advance and step backward
    replayPaused = true;
    if (replayTimer) { clearInterval(replayTimer); replayTimer = null; }
    if (replayMoveIdx > 0) {
      replayMoveIdx--;
      rebuildBoard(currentGameData.moves, replayMoveIdx);
      drawHex();
      setInfo('Game #' + currentGameData.game_idx + ' \u2014 Move ' + replayMoveIdx + '/' + currentGameData.moves.length + ' [PAUSED]');
    }
  }
  if (e.code === 'KeyR' && currentGameData) {
    replayGame(currentGameData);
  }
});

// ---------------------------------------------------------------------------
// Game history
// ---------------------------------------------------------------------------
const gameHistoryList = [];  // store last 20 games

function addToHistory(d) {
  gameHistoryList.push(d);
  if (gameHistoryList.length > 10) gameHistoryList.shift();
  const histEl = el('game-history');
  if (!histEl) return;
  histEl.innerHTML = '';
  gameHistoryList.forEach((g, i) => {
    const span = document.createElement('span');
    span.className = 'gh-item' + (g === currentGameData ? ' active' : '');
    const w = g.result > 0 ? 'B' : 'W';
    span.textContent = '#' + g.game_idx + ' ' + w + ' ' + g.num_moves + 'mv';
    span.onclick = () => { replayPaused = false; replayGame(g); };
    histEl.appendChild(span);
  });
  histEl.scrollTop = histEl.scrollHeight;
}

// ---------------------------------------------------------------------------
// Socket events
// ---------------------------------------------------------------------------
socket.on('game_complete', d => {
  try {
    el('status').textContent = 'GAME ' + d.game_idx;
    // Progress bar
    if (d.total_games) {
      const pw = el('progress-wrap');
      if (pw) pw.style.display = '';
      const pf = el('progress-fill');
      if (pf) pf.style.width = Math.round(d.game_idx / d.total_games * 100) + '%';
      const plab = el('progress-label');
      if (plab) plab.textContent = 'Self-play ' + d.game_idx + '/' + d.total_games;
    }
    // Accumulate local game stats (footer updated by stats_update from server)
    gameStats.count++;
    gameStats.totalLen += (d.num_moves || 0);
    if (d.result > 0) gameStats.w0++; else if (d.result < 0) gameStats.w1++;
    // History + replay (NEVER interrupt a playing game)
    if (d.moves && d.moves.length) {
      addToHistory(d);
      if (replayBusy) {
        pendingGame = d;  // just queue the latest, current game plays to end
      } else {
        replayGame(d);
      }
    }
  } catch (e) { console.error('game_complete error:', e); }
});

socket.on('stats_update', d => {
  try {
    if (d.iteration != null)      el('s-iter').textContent    = d.iteration;
    if (d.total_games != null)    el('s-games').textContent   = d.total_games;
    if (d.current_elo != null)    el('s-elo').textContent     = Math.round(d.current_elo);
    if (d.workers != null)        el('s-workers').textContent = d.workers;
    if (d.win_p0 != null)         el('s-w0').textContent      = Math.round(d.win_p0);
    if (d.win_p1 != null)         el('s-w1').textContent      = Math.round(d.win_p1);
    if (d.avg_game_length)        el('s-len').textContent     = d.avg_game_length;
    if (d.latest_loss != null) {
      const lv = typeof d.latest_loss === 'object' ? d.latest_loss.total : d.latest_loss;
      if (lv != null) el('s-loss').textContent = parseFloat(lv).toFixed(4);
    }
    if (d.latest_policy_loss != null) el('s-ploss').textContent = parseFloat(d.latest_policy_loss).toFixed(4);
    if (d.latest_value_loss  != null) el('s-vloss').textContent = parseFloat(d.latest_value_loss).toFixed(4);
    if (d.fill_pct != null)    el('s-fill').textContent  = parseFloat(d.fill_pct).toFixed(1) + '%';
    if (d.pretrained_weight != null) {
      const pw = parseFloat(d.pretrained_weight);
      const sp = Math.max(0, 1.0 - pw);
      const filled = Math.round(pw * 20);
      const bar = '█'.repeat(filled) + '░'.repeat(20 - filled);
      el('s-pmix').textContent = '[' + bar + '] pre=' + pw.toFixed(2) + ' sp=' + sp.toFixed(2);
    }
    if (d.positions_hr != null) {
      const p = d.positions_hr;
      el('s-poshr').textContent = p >= 1e6 ? (p/1e6).toFixed(1)+'M' : Math.round(p).toLocaleString();
    }
    fetchCharts();
  } catch (e) { console.error('stats_update error:', e); }
});

socket.on('train_progress', d => {
  try {
    const pw = el('progress-wrap');
    const pf = el('progress-fill');
    const plab = el('progress-label');
    if (pw) pw.style.display = '';
    if (pf) pf.style.width = d.pct + '%';
    const phase = d.phase || 'Training';
    const lossStr = d.loss != null ? ' (loss ' + d.loss + ')' : '';
    if (plab) plab.textContent = phase + ' ' + d.step + '/' + d.total + lossStr;
    if (d.loss != null) el('s-loss').textContent = d.loss;
    el('status').textContent = phase.toUpperCase() + ' ' + d.step + '/' + d.total;
  } catch (e) { console.error('train_progress error:', e); }
});

// ---------------------------------------------------------------------------
// Line chart (HiDPI, auto-scaling)
// ---------------------------------------------------------------------------
function drawLineChart(cv, datasets, opts) {
  if (!cv) return;
  // Skip collapsed charts
  const body = cv.closest('.chart-body');
  if (body && body.classList.contains('collapsed')) return;

  const { w: W, h: H } = sizeCanvas(cv);
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  const pad = { t: 14, r: 16, b: 24, l: 52 };
  const pW = W - pad.l - pad.r, pH = H - pad.t - pad.b;
  if (pW < 10 || pH < 10) return;

  const pts = datasets.flatMap(d => d.data);
  if (pts.length < 2) {
    ctx.fillStyle = '#ccc'; ctx.font = '11px Courier New'; ctx.textAlign = 'center';
    ctx.fillText('No data yet', W / 2, H / 2);
    return;
  }

  let xMn = Math.min(...pts.map(p => p.x)), xMx = Math.max(...pts.map(p => p.x));
  let yMn = opts.yMin != null ? opts.yMin : Math.min(...pts.map(p => p.y));
  let yMx = opts.yMax != null ? opts.yMax : Math.max(...pts.map(p => p.y));
  if (xMn === xMx) xMx = xMn + 1;
  if (opts.yMin == null) { const yP = (yMx - yMn) * 0.1 || 1; yMn -= yP; yMx += yP; }

  function toP(x, y) {
    return [pad.l + (x - xMn) / (xMx - xMn) * pW, pad.t + pH - (y - yMn) / (yMx - yMn) * pH];
  }

  // Axes
  ctx.strokeStyle = '#000'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + pH); ctx.lineTo(pad.l + pW, pad.t + pH);
  ctx.stroke();

  // Y ticks
  ctx.fillStyle = '#000'; ctx.font = '10px Courier New'; ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const yV = yMn + (yMx - yMn) * i / 4;
    const [, sy] = toP(xMn, yV);
    ctx.fillText(yV.toFixed(1), pad.l - 4, sy + 3);
    ctx.strokeStyle = '#eee'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(pad.l, sy); ctx.lineTo(pad.l + pW, sy); ctx.stroke();
  }

  // X ticks
  ctx.textAlign = 'center';
  const xStep = Math.max(1, Math.ceil((xMx - xMn) / 6));
  for (let x = Math.ceil(xMn); x <= xMx; x += xStep) {
    const [sx] = toP(x, yMn);
    ctx.fillStyle = '#000'; ctx.fillText(x, sx, pad.t + pH + 14);
  }

  // Lines
  const defaultDash = [[], [6, 3], [2, 2], [8, 2, 2, 2]];
  datasets.forEach((ds, di) => {
    if (ds.data.length < 2) return;
    ctx.setLineDash(ds.dash || defaultDash[di % defaultDash.length] || []);
    ctx.strokeStyle = '#000'; ctx.lineWidth = 1.3;
    ctx.beginPath();
    ds.data.forEach((p, i) => {
      const [sx, sy] = toP(p.x, p.y);
      i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
    });
    ctx.stroke();
  });
  ctx.setLineDash([]);

  // Legend
  ctx.font = '10px Courier New'; ctx.textAlign = 'left';
  datasets.forEach((ds, di) => {
    const lx = pad.l + 8 + di * 90, ly = pad.t + 10;
    ctx.setLineDash(ds.dash || defaultDash[di % defaultDash.length] || []);
    ctx.strokeStyle = '#000'; ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 18, ly); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#000'; ctx.fillText(ds.label, lx + 22, ly + 3);
  });
}

// ---------------------------------------------------------------------------
// Fetch chart data from API
// ---------------------------------------------------------------------------
function fetchCharts() {
  fetch('/api/elo').then(r => r.json()).then(data => {
    if (!data.length) return;
    drawLineChart(el('elo-chart'),
      [{ label: 'ELO', data: data.map(d => ({ x: d.iteration, y: d.elo })), dash: [] }], {});
  }).catch(() => {});

  fetch('/api/losses').then(r => r.json()).then(data => {
    if (!data.length) return;
    const ds = [{ label: 'total', data: data.filter(d => d.total != null).map(d => ({ x: d.iteration, y: d.total })), dash: [] }];
    if (data.some(d => d.value != null))
      ds.push({ label: 'value', data: data.filter(d => d.value != null).map(d => ({ x: d.iteration, y: d.value })), dash: [6, 3] });
    if (data.some(d => d.policy != null))
      ds.push({ label: 'policy', data: data.filter(d => d.policy != null).map(d => ({ x: d.iteration, y: d.policy })), dash: [2, 2] });
    drawLineChart(el('loss-chart'), ds, {});
  }).catch(() => {});

  fetch('/api/winrates').then(r => r.json()).then(data => {
    if (!data.length) return;
    drawLineChart(el('winrate-chart'), [
      { label: 'P0%', data: data.map(d => ({ x: d.iteration, y: d.p0 })), dash: [] },
      { label: 'P1%', data: data.map(d => ({ x: d.iteration, y: d.p1 })), dash: [6, 3] },
      { label: 'Draw%', data: data.map(d => ({ x: d.iteration, y: d.draw })), dash: [2, 2] },
    ], { yMin: 0, yMax: 100 });
  }).catch(() => {});

  fetch('/api/gamelength').then(r => r.json()).then(data => {
    if (!data.length) return;
    drawLineChart(el('gamelength-chart'),
      [{ label: 'Avg Moves', data: data.map(d => ({ x: d.iteration, y: d.avg_game_length })), dash: [] }], {});
  }).catch(() => {});

  fetch('/api/speed').then(r => r.json()).then(data => {
    if (!data.length) return;
    drawLineChart(el('speed-chart'),
      [{ label: 'Games/s', data: data.map(d => ({ x: d.iteration, y: d.games_per_sec })), dash: [] }], {});
  }).catch(() => {});

}

// ---------------------------------------------------------------------------
// Resource polling
// ---------------------------------------------------------------------------
setInterval(() => {
  fetch('/api/stats').then(r => r.json()).then(d => {
    if (d.cpu_pct != null) {
      el('s-cpu').textContent = Math.round(d.cpu_pct);
      el('cpu-fill').style.width = Math.round(d.cpu_pct) + '%';
    }
    if (d.ram_pct != null) {
      el('s-ram').textContent = Math.round(d.ram_pct);
      el('ram-fill').style.width = Math.round(d.ram_pct) + '%';
    }
  }).catch(() => {});
}, 5000);

// ---------------------------------------------------------------------------
// Resize handler
// ---------------------------------------------------------------------------
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { drawHex(); fetchCharts(); }, 150);
});

// ---------------------------------------------------------------------------
// Init - restore settings from localStorage
// ---------------------------------------------------------------------------
(function initSettings() {
  el('set-speed').value = settings.replaySpeed;
  el('set-speed-val').textContent = settings.replaySpeed + 'ms';
  el('set-dotsize').value = settings.dotSize;
  el('set-dotsize-val').textContent = settings.dotSize;
  el('set-radius').value = settings.emptyHexRadius;
  el('set-radius-val').textContent = settings.emptyHexRadius;
  el('set-movenums').checked = settings.showMoveNums;
  el('set-autorefresh').checked = settings.autoRefresh;
})();
setTimeout(() => { drawHex(); fetchCharts(); }, 100);
fetch('/api/stats').then(r => r.json()).then(s => {
  el('s-games').textContent = s.total_games;
  el('s-elo').textContent = Math.round(s.current_elo);
}).catch(() => {});

// ---------------------------------------------------------------------------
// Recent self-play replays polling
// ---------------------------------------------------------------------------
function timeAgo(ts) {
  if (!ts) return '';
  const diff = (Date.now() - new Date(ts).getTime()) / 1000;
  if (diff < 60) return Math.round(diff) + 's ago';
  if (diff < 3600) return Math.round(diff / 60) + 'm ago';
  return Math.round(diff / 3600) + 'h ago';
}
function fetchReplays() {
  fetch('/api/replays?n=5').then(r => r.json()).then(games => {
    const box = el('replay-list');
    if (!games.length) { box.textContent = 'No replays yet'; return; }
    box.innerHTML = games.map(g =>
      '<div style="padding:2px 0;cursor:pointer" onclick="viewReplay(\'' + g.key + '\')">' +
      (g.winner === 'x_win' ? 'X won' : g.winner === 'o_win' ? 'O won' : 'Draw') +
      '  ' + g.game_length + ' moves  ' + timeAgo(g.timestamp) +
      '  (ckpt ' + g.checkpoint_step + ')</div>'
    ).join('');
  }).catch(() => {});
}
function viewReplay(key) {
  fetch('/api/replays/' + encodeURIComponent(key)).then(r => r.json()).then(g => {
    if (g.error) return;
    const result = g.winner === 'x_win' ? 1.0 : g.winner === 'o_win' ? -1.0 : 0.0;
    const label = 'ckpt-' + g.checkpoint_step;
    replayPaused = false;
    replayGame({ game_idx: label, moves: g.moves || [], result: result });
  }).catch(() => {});
}
fetchReplays();
setInterval(fetchReplays, 30000);

// ---------------------------------------------------------------------------
// Corpus games polling
// ---------------------------------------------------------------------------
function fetchCorpusReplays() {
  fetch('/api/corpus-replays').then(r => r.json()).then(games => {
    const box = el('corpus-list');
    const container = el('box-corpus');
    if (!games.length) { container.style.display = 'none'; return; }
    container.style.display = '';
    box.innerHTML = games.map(g =>
      '<div style="padding:2px 0;cursor:pointer" onclick="viewCorpusReplay(\'' + g.key + '\')">' +
      (g.winner === 'x_win' ? 'X won' : g.winner === 'o_win' ? 'O won' : g.winner === 'draw' ? 'Draw' : '?') +
      '  ' + g.game_length + ' moves' +
      (g.source ? '  [' + g.source + ']' : '') +
      '  ' + timeAgo(g.timestamp) +
      '</div>'
    ).join('');
  }).catch(() => {});
}
function viewCorpusReplay(key) {
  fetch('/api/corpus-replays/' + encodeURIComponent(key)).then(r => r.json()).then(g => {
    if (g.error) return;
    const result = g.winner === 'x_win' ? 1.0 : g.winner === 'o_win' ? -1.0 : 0.0;
    const label = (g.source ? '[' + g.source + '] ' : '') + g.game_id;
    replayPaused = false;
    replayGame({ game_idx: label, moves: g.moves || [], result: result });
  }).catch(() => {});
}
fetchCorpusReplays();
setInterval(fetchCorpusReplays, 30000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Module-level singleton for standalone mode
# ---------------------------------------------------------------------------

_default_dashboard: Optional[Dashboard] = None


def get_dashboard(port: int = 5001) -> Dashboard:
    """Get or create the module-level Dashboard singleton."""
    global _default_dashboard
    if _default_dashboard is None:
        _default_dashboard = Dashboard(port=port)
    return _default_dashboard


# ---------------------------------------------------------------------------
# Log file poller — feeds DataStore from structlog JSONL without push
# ---------------------------------------------------------------------------

class LogPoller:
    """Tail-reads structlog JSONL log files and feeds DataStore.

    On startup, backfills the full latest log silently (no per-record WebSocket
    events), then emits one refresh. Afterwards polls every `interval` seconds
    and emits live stats updates.

    Only `train_step` events are consumed — game_complete events lack move data
    needed for game replay, and win stats are already embedded in train_step.
    """

    def __init__(
        self,
        log_dir: str,
        dashboard: "Dashboard",
        interval: float = 2.0,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._dash = dashboard
        self._interval = interval
        self._log_path: Optional[Path] = None
        self._log_fh = None
        self._log_pos: int = 0
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="log-poller"
        )
        self._thread.start()

    def _loop(self) -> None:
        # Backfill silently
        self._poll(emit=False)
        step = self._dash.store.current_iteration
        ts = _dt.now().strftime("%H:%M:%S")
        print(f"[{ts}] Log backfill complete — latest step {step}")
        self._dash.socketio.emit("stats_update", self._dash.store.get_stats())
        # Live polling
        while True:
            time.sleep(self._interval)
            self._poll(emit=True)

    def _find_latest(self) -> Optional[Path]:
        try:
            logs = list(self._log_dir.glob("*.jsonl"))
        except OSError:
            return None
        return max(logs, key=lambda p: p.stat().st_mtime) if logs else None

    def _poll(self, emit: bool) -> None:
        latest = self._find_latest()
        if latest is None:
            return
        if latest != self._log_path:
            if self._log_fh is not None:
                self._log_fh.close()
            try:
                self._log_fh = latest.open("r", errors="replace")
                self._log_path = latest
                self._log_pos = 0
            except OSError:
                return
        self._log_fh.seek(self._log_pos)
        new_data = False
        for raw in self._log_fh:
            self._log_pos += len(raw.encode("utf-8", errors="replace"))
            line = raw.strip()
            if not line:
                continue
            try:
                ingested = self._ingest(json.loads(line))
                if ingested:
                    new_data = True
            except (json.JSONDecodeError, KeyError):
                pass
        if emit and new_data:
            self._dash.socketio.emit("stats_update", self._dash.store.get_stats())

    def _ingest(self, entry: dict) -> bool:
        event = entry.get("event")
        if event == "game_complete":
            gl = entry.get("game_length")
            if gl is None:
                plies = entry.get("plies", 0)
                if plies > 0:
                    gl = (plies + 1) // 2
            if gl and gl > 0:
                self._dash.store.record_game_length(int(gl))
            return True
        if event != "train_step":
            return False
        step = entry.get("step", 0)
        wins = [
            entry.get("x_wins", 0),
            entry.get("o_wins", 0),
            entry.get("draws", 0),
        ]
        # Pass games_per_hour as games/self_play_time so speed chart shows
        # games/sec correctly (games_per_sec = gph / 3600).
        gph = entry.get("games_per_hour", 0.0)
        self._dash.store.add_metric(
            iteration=step,
            loss=entry.get("total_loss"),
            policy_loss=entry.get("policy_loss"),
            value_loss=entry.get("value_loss"),
            wins=wins,
            buffer_size=entry.get("buffer_size", 0),
            games=gph,
            self_play_time=3600.0 if gph else 0.0,
            pretrained_weight=entry.get("pretrained_weight"),
        )
        return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="HEX BOT Training Dashboard")
    parser.add_argument("port", nargs="?", type=int, default=5001,
                        help="Port to listen on (default: 5001)")
    parser.add_argument("--log-dir", default="logs",
                        help="Directory containing structlog JSONL files (default: logs)")
    args = parser.parse_args()

    port = args.port
    log_dir = args.log_dir

    dash = get_dashboard(port)
    dash.resource_monitor.start()

    # Start log poller so dashboard works even without --web-dashboard on train.py
    log_path = Path(log_dir)
    if log_path.is_dir():
        poller = LogPoller(log_dir=log_dir, dashboard=dash, interval=2.0)
        poller.start()
    else:
        print(f"[WARN] --log-dir '{log_dir}' not found — log polling disabled")

    ts = _dt.now().strftime("%H:%M:%S")
    print(f"[{ts}] HEX BOT -- Clean Training Dashboard")
    print(f"[{ts}] CPU cores: {multiprocessing.cpu_count()}")
    print(f"[{ts}] Log dir: {log_dir}")
    print(f"[{ts}] Open http://localhost:{port}")

    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    dash.socketio.run(
        dash.app, host="0.0.0.0", port=port,
        debug=False, allow_unsafe_werkzeug=True, log_output=False,
    )
