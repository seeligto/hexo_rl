"""Rich-based terminal dashboard for HeXO training monitoring.

Passive observer — consumes events from emit_event(), never blocks the
training loop, never raises exceptions that propagate to train.py.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Any

from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


_EM_DASH = "\u2014"


def _fmt_loss(v: Any) -> str:
    if v is None:
        return _EM_DASH
    return f"{v:.4f}"


def _fmt_rate(v: Any, suffix: str = "") -> str:
    if v is None:
        return _EM_DASH
    if v >= 1_000:
        return f"{v / 1_000:.0f}K{suffix}"
    return f"{v:,.1f}{suffix}"


def _fmt_pct(v: Any) -> str:
    if v is None:
        return _EM_DASH
    return f"{v * 100:.1f}%"


def _fmt_int(v: Any) -> str:
    if v is None:
        return _EM_DASH
    return f"{int(v):,}"


def _fmt_plain(v: Any, dp: int = 1) -> str:
    if v is None:
        return _EM_DASH
    return f"{v:.{dp}f}"


class TerminalDashboard:
    """Rich Live terminal renderer for training events."""

    def __init__(self, config: dict) -> None:
        mon = config.get("monitoring", config)
        self._alert_entropy_min = float(mon.get("alert_entropy_min", 1.0))
        self._alert_grad_max = float(mon.get("alert_grad_norm_max", 10.0))
        self._alert_loss_window = int(mon.get("alert_loss_increase_window", 3))

        self._lock = threading.Lock()
        self._live: Live | None = None
        self._last_render_ts: float = 0.0
        self._min_render_interval = 0.25  # 4 Hz max

        # Alert tracking
        self._alerts: list[tuple[float, str]] = []  # (expiry_ts, message)
        self._recent_losses: collections.deque = collections.deque(
            maxlen=self._alert_loss_window + 1
        )

        self._state: dict[str, Any] = {
            "run_id": None,
            "step": 0,
            "loss_total": None,
            "loss_policy": None,
            "loss_value": None,
            "loss_aux": None,
            "policy_entropy": None,
            "lr": None,
            "grad_norm": None,
            "games_total": None,
            "games_per_hour": None,
            "positions_per_hour": None,
            "avg_game_length": None,
            "win_rate_p0": None,
            "win_rate_p1": None,
            "draw_rate": None,
            "sims_per_sec": None,
            "buffer_size": None,
            "buffer_capacity": None,
            "corpus_selfplay_frac": None,
            "elo_estimate": None,
            "gate_passed": None,
            "gpu_util_pct": None,
            "vram_used_gb": None,
            "vram_total_gb": None,
        }

    def start(self) -> None:
        """Start the rich Live context."""
        self._live = Live(self._build_panel(), refresh_per_second=4)
        self._live.start()

    def stop(self) -> None:
        """Stop the rich Live context cleanly."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def on_event(self, payload: dict) -> None:
        """Receive an event from emit_event() and update display."""
        event = payload.get("event")
        if event is None:
            return

        with self._lock:
            self._merge_event(payload)
            self._check_alerts(payload)
            self._render()

    # ── Internal ──────────────────────────────────────────────────────────

    def _merge_event(self, payload: dict) -> None:
        """Merge relevant fields from the payload into _state."""
        event = payload.get("event")

        if event == "run_start":
            self._state["run_id"] = payload.get("run_id")
            self._state["step"] = payload.get("step", 0)
            return

        if event == "training_step":
            for key in (
                "step", "loss_total", "loss_policy", "loss_value",
                "loss_aux", "policy_entropy", "lr", "grad_norm",
            ):
                if key in payload:
                    self._state[key] = payload[key]
            return

        if event == "iteration_complete":
            for key in (
                "step", "games_total", "games_per_hour",
                "positions_per_hour", "avg_game_length",
                "win_rate_p0", "win_rate_p1", "draw_rate",
                "sims_per_sec", "buffer_size", "buffer_capacity",
                "corpus_selfplay_frac",
            ):
                if key in payload:
                    self._state[key] = payload[key]
            return

        if event == "eval_complete":
            if "elo_estimate" in payload:
                self._state["elo_estimate"] = payload["elo_estimate"]
            if "gate_passed" in payload:
                self._state["gate_passed"] = payload["gate_passed"]
            return

        if event == "system_stats":
            for key in ("gpu_util_pct", "vram_used_gb", "vram_total_gb"):
                if key in payload:
                    self._state[key] = payload[key]
            return

        # Unknown events are silently ignored.

    def _check_alerts(self, payload: dict) -> None:
        """Evaluate alert conditions and manage active alerts."""
        now = time.time()
        expiry = now + 60.0
        event = payload.get("event")

        if event == "training_step":
            # Entropy collapse
            ent = payload.get("policy_entropy")
            if ent is not None and ent < self._alert_entropy_min:
                self._add_alert(
                    expiry, f"policy entropy {ent:.2f} — possible mode collapse"
                )

            # Grad norm spike
            gn = payload.get("grad_norm")
            if gn is not None and gn == gn and gn > self._alert_grad_max:
                self._add_alert(expiry, f"grad norm {gn:.1f} — instability")

            # Loss increasing trend
            lt = payload.get("loss_total")
            if lt is not None:
                self._recent_losses.append(lt)
                if len(self._recent_losses) > self._alert_loss_window:
                    window = list(self._recent_losses)[-self._alert_loss_window - 1:]
                    if all(window[i] < window[i + 1] for i in range(len(window) - 1)):
                        self._add_alert(
                            expiry,
                            f"loss increased {self._alert_loss_window} consecutive steps",
                        )

        if event == "eval_complete" and payload.get("gate_passed") is False:
            wr = payload.get("win_rate_vs_sealbot", 0.0)
            self._add_alert(
                expiry, f"SealBot eval FAILED — {wr:.1%} win rate"
            )

        # Expire old alerts
        self._alerts = [(t, m) for t, m in self._alerts if t > now]

    def _add_alert(self, expiry: float, message: str) -> None:
        """Add alert, replacing any with the same prefix."""
        prefix = message.split("—")[0].strip() if "—" in message else message[:20]
        self._alerts = [
            (t, m) for t, m in self._alerts
            if not (m.split("—")[0].strip() if "—" in m else m[:20]).startswith(prefix)
        ]
        self._alerts.append((expiry, message))

    def _render(self) -> None:
        """Build and push the panel to rich Live (max 4 Hz)."""
        if self._live is None:
            return
        now = time.time()
        if now - self._last_render_ts < self._min_render_interval:
            return
        self._last_render_ts = now

        try:
            panel = self._build_panel()
            self._live.update(panel)
        except Exception:
            pass  # Never let rendering crash the training loop

    def _build_panel(self) -> Panel:
        """Construct the rich Panel from current state."""
        s = self._state

        # Header
        run_id = s["run_id"][:8] if s["run_id"] else _EM_DASH
        header = f"HeXO training · phase 4.0 · run {run_id} · step {_fmt_int(s['step'])}"

        # Loss row
        loss_tbl = Table(show_header=True, box=None, padding=(0, 2), expand=True)
        for label in ("loss", "policy", "value", "aux", "entropy", "lr"):
            loss_tbl.add_column(label, justify="center")
        lr_str = _EM_DASH if s["lr"] is None else f"{s['lr']:.0e}"
        loss_tbl.add_row(
            _fmt_loss(s["loss_total"]),
            _fmt_loss(s["loss_policy"]),
            _fmt_loss(s["loss_value"]),
            _fmt_loss(s["loss_aux"]),
            _fmt_plain(s["policy_entropy"], 2),
            lr_str,
        )

        # Throughput row
        tp_tbl = Table(show_header=False, box=None, padding=(0, 2), expand=True)
        tp_tbl.add_column(justify="left")
        gph = _fmt_rate(s["games_per_hour"])
        pph = _fmt_rate(s["positions_per_hour"])
        sps = _fmt_rate(s["sims_per_sec"])
        avg = _fmt_plain(s["avg_game_length"], 0) if s["avg_game_length"] is not None else _EM_DASH
        p0 = _fmt_pct(s["win_rate_p0"])
        p1 = _fmt_pct(s["win_rate_p1"])
        dr = _fmt_pct(s["draw_rate"])
        tp_tbl.add_row(
            f"games/hr  {gph}  │  pos/hr  {pph}  │  sims/sec  {sps}"
        )
        tp_tbl.add_row(
            f"avg len   {avg}  │  P0 {p0}  P1 {p1}  draw {dr}"
        )

        # Buffer row
        buf_tbl = Table(show_header=False, box=None, padding=(0, 2), expand=True)
        buf_tbl.add_column(justify="left")
        if s["buffer_size"] is not None and s["buffer_capacity"] is not None:
            buf_pct = s["buffer_size"] / max(s["buffer_capacity"], 1) * 100
            buf_str = (
                f"buffer  {_fmt_int(s['buffer_size'])} / {_fmt_int(s['buffer_capacity'])}"
                f"  ({buf_pct:.0f}%)"
            )
        else:
            buf_str = f"buffer  {_EM_DASH}"

        sp_frac = s["corpus_selfplay_frac"]
        if sp_frac is not None:
            sp_pct = sp_frac * 100
            pre_pct = (1 - sp_frac) * 100
            mix_str = f"sp {sp_pct:.0f}%  pre {pre_pct:.0f}%"
        else:
            mix_str = f"sp {_EM_DASH}  pre {_EM_DASH}"
        buf_tbl.add_row(f"{buf_str}  │  {mix_str}")

        # System row
        elo_str = _EM_DASH if s["elo_estimate"] is None else _fmt_int(s["elo_estimate"])
        gpu_str = _EM_DASH if s["gpu_util_pct"] is None else f"{s['gpu_util_pct']:.0f}%"
        if s["vram_used_gb"] is not None and s["vram_total_gb"] is not None:
            vram_str = f"{s['vram_used_gb']:.1f}/{s['vram_total_gb']:.1f} GB"
        else:
            vram_str = _EM_DASH
        buf_tbl.add_row(f"ELO  {elo_str}  │  gpu  {gpu_str}  │  vram  {vram_str}")

        # Assemble
        outer = Table(show_header=False, box=None, expand=True, padding=0)
        outer.add_column()
        outer.add_row(loss_tbl)
        outer.add_row(tp_tbl)
        outer.add_row(buf_tbl)

        panel = Panel(outer, title=header, border_style="blue")

        # Alert line
        if self._alerts:
            alert_msgs = [m for _, m in self._alerts]
            alert_text = Text(" │ ".join(alert_msgs), style="bold red")
            # Wrap panel + alert in a group
            from rich.console import Group
            return Panel(
                Group(outer, Text(""), alert_text),
                title=header,
                border_style="blue",
            )

        return panel
