"""Rich-based terminal dashboard for HeXO training monitoring.

Passive observer — consumes events from emit_event(), never blocks the
training loop, never raises exceptions that propagate to train.py.
"""

from __future__ import annotations

import collections
import math
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
        self._alert_entropy_warn = float(mon.get("alert_entropy_warn", 2.0))
        self._alert_grad_max = float(mon.get("alert_grad_norm_max", 10.0))
        self._alert_loss_window = int(mon.get("alert_loss_increase_window", 3))
        num_actions = int(mon.get("num_actions_for_entropy_norm", 362))
        self._max_entropy = math.log(num_actions) if num_actions > 1 else 1.0
        # Mirror the web dashboard's knob so terminal + web agree on the
        # selfplay-collapse threshold (§70). Default 1.5 nats.
        self._collapse_threshold = float(mon.get("collapse_threshold_nats", 1.5))

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
            "loss_chain": None,
            "loss_ownership": None,
            "loss_threat": None,
            "policy_entropy": None,
            "policy_entropy_pretrain": None,
            "policy_entropy_selfplay": None,
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
            "anchor_promoted": None,
            "sealbot_gate_passed": None,
            "gpu_util_pct": None,
            "vram_used_gb": None,
            "vram_total_gb": None,
            "ram_used_gb": None,
            "ram_total_gb": None,
            "rss_gb": None,
            "cpu_util_pct": None,
            "batch_fill_pct": None,
            "worker_count": None,
            "phase": None,
            "policy_target_entropy": None,
            "policy_target_entropy_fullsearch":    None,
            "policy_target_entropy_fastsearch":    None,
            "policy_target_kl_uniform_fullsearch": None,
            "policy_target_kl_uniform_fastsearch": None,
            "frac_fullsearch_in_batch":            None,
            "n_rows_policy_loss":                  None,
            "n_rows_total":                        None,
            "mcts_mean_depth": None,
            "mcts_root_concentration": None,
        }

    def start(self) -> None:
        """Start the rich Live context.

        Auto-refresh is set to a near-zero rate to prevent blank
        intermediate frames when auto-refresh races with explicit updates.
        Frames are pushed explicitly via update(refresh=True) in on_event().
        """
        self._live = Live(self._build_panel(), refresh_per_second=0.1)
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
            if "worker_count" in payload:
                self._state["worker_count"] = payload["worker_count"]
            return

        if event == "training_step":
            for key in (
                "step", "loss_total", "loss_policy", "loss_value",
                "loss_aux", "loss_chain", "loss_ownership", "loss_threat",
                "policy_entropy", "policy_entropy_pretrain",
                "policy_entropy_selfplay", "policy_target_entropy",
                "policy_target_entropy_fullsearch",
                "policy_target_entropy_fastsearch",
                "policy_target_kl_uniform_fullsearch",
                "policy_target_kl_uniform_fastsearch",
                "frac_fullsearch_in_batch",
                "n_rows_policy_loss", "n_rows_total",
                "lr", "grad_norm",
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
                "corpus_selfplay_frac", "batch_fill_pct",
                "mcts_mean_depth", "mcts_root_concentration",
            ):
                if key in payload:
                    self._state[key] = payload[key]
            return

        if event == "eval_complete":
            if "elo_estimate" in payload:
                self._state["elo_estimate"] = payload["elo_estimate"]
            if "anchor_promoted" in payload:
                self._state["anchor_promoted"] = payload["anchor_promoted"]
            if "sealbot_gate_passed" in payload:
                self._state["sealbot_gate_passed"] = payload["sealbot_gate_passed"]
            return

        if event == "system_stats":
            for key in ("gpu_util_pct", "vram_used_gb", "vram_total_gb",
                        "buffer_size", "buffer_capacity",
                        "ram_used_gb", "ram_total_gb", "rss_gb", "cpu_util_pct"):
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
            # Entropy collapse (combined)
            ent = payload.get("policy_entropy")
            if ent is not None and ent < self._alert_entropy_min:
                self._add_alert(
                    expiry, f"policy entropy {ent:.2f} — possible mode collapse"
                )
            # Selfplay-stream collapse (threshold per §70; configurable).
            ent_sp = payload.get("policy_entropy_selfplay")
            if ent_sp is not None and math.isfinite(ent_sp) and ent_sp < self._collapse_threshold:
                self._add_alert(
                    expiry, f"selfplay entropy {ent_sp:.2f} — selfplay mode collapse"
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

        if event == "eval_complete" and payload.get("sealbot_gate_passed") is False:
            wr = payload.get("win_rate_vs_sealbot")
            wr_str = f"{wr:.1%}" if wr is not None else "?"
            self._add_alert(
                expiry, f"SealBot eval FAILED — {wr_str} win rate"
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
            self._live.update(panel, refresh=True)
        except Exception:
            pass  # Never let rendering crash the training loop

    def _build_panel(self) -> Panel:
        """Construct the rich Panel from current state."""
        s = self._state

        # Header
        run_id = s["run_id"][:8] if s["run_id"] else _EM_DASH
        phase_str = " [PRETRAIN]" if s["phase"] == "pretrain" else ""
        header = f"HeXO training · phase 4.0{phase_str} · run {run_id} · step {_fmt_int(s['step'])}"

        # Loss row
        loss_tbl = Table(show_header=True, box=None, padding=(0, 2), expand=True)
        for label in ("loss", "policy", "value", "aux", "chain", "own", "thr", "entropy", "lr"):
            loss_tbl.add_column(label, justify="center")
        lr_str = _EM_DASH if s["lr"] is None else f"{s['lr']:.0e}"

        # Value loss with ratio relative to random baseline (H(Bernoulli(0.5)) = ln2 ≈ 0.6931)
        val_loss = s["loss_value"]
        if val_loss is not None:
            val_ratio = val_loss / 0.6931
            val_str = f"{_fmt_loss(val_loss)} (×{val_ratio:.2f})"
        else:
            val_str = _EM_DASH

        # Entropy with % of max annotation
        ent = s["policy_entropy"]
        if ent is None:
            ent_str = _EM_DASH
        else:
            ent_pct = ent / self._max_entropy * 100
            if ent < self._alert_entropy_min:
                marker = " !!"
            elif ent < self._alert_entropy_warn:
                marker = " \u25b2"
            else:
                marker = ""
            ent_str = f"{ent:.2f} ({ent_pct:.0f}% max){marker}"

        loss_tbl.add_row(
            _fmt_loss(s["loss_total"]),
            _fmt_loss(s["loss_policy"]),
            val_str,
            _fmt_loss(s["loss_aux"]),
            _fmt_loss(s["loss_chain"]),
            _fmt_loss(s["loss_ownership"]),
            _fmt_loss(s["loss_threat"]),
            ent_str,
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
        bf = s["batch_fill_pct"]
        bf_str = f"{bf:.0f}%" if bf is not None else _EM_DASH
        gn = s["grad_norm"]
        grad_str = f"{gn:.2f}" if gn is not None else _EM_DASH
        mcts_d = s["mcts_mean_depth"]
        mcts_d_str = f"{mcts_d:.1f}" if mcts_d is not None else _EM_DASH
        mcts_c = s["mcts_root_concentration"]
        mcts_c_str = f"{mcts_c:.2f}" if mcts_c is not None else _EM_DASH
        tp_tbl.add_row(
            f"games/hr  {gph}  │  pos/hr  {pph}  │  sims/sec  {sps}  │  batch fill  {bf_str}"
        )
        tp_tbl.add_row(
            f"avg len   {avg}  │  P0 {p0}  P1 {p1}  draw {dr}"
            f"  │  MCTS depth  {mcts_d_str}  │  root concen  {mcts_c_str}  │  grad  {grad_str}"
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
        if s["ram_used_gb"] is not None and s["ram_total_gb"] is not None:
            ram_str = f"{s['ram_used_gb']:.1f}/{s['ram_total_gb']:.1f} GB"
        else:
            ram_str = _EM_DASH
        rss_str = f"{s['rss_gb']:.1f} GB" if s["rss_gb"] is not None else _EM_DASH
        cpu_str = f"{s['cpu_util_pct']:.0f}%" if s["cpu_util_pct"] is not None else _EM_DASH
        buf_tbl.add_row(
            f"ELO  {elo_str}  │  gpu  {gpu_str}  │  vram  {vram_str}"
            f"  │  ram  {ram_str}  │  rss  {rss_str}  │  cpu  {cpu_str}"
        )

        # Policy entropy split row (combined / pretrain / selfplay)
        _COLLAPSE_THRESHOLD = self._collapse_threshold
        ent_pre = s["policy_entropy_pretrain"]
        ent_sp  = s["policy_entropy_selfplay"]

        def _fmt_ent_pre(v: Any) -> str:
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                return _EM_DASH
            return f"{v:.2f}"

        def _fmt_ent_sp(v: Any) -> str:
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                return _EM_DASH
            if v < _COLLAPSE_THRESHOLD:
                return f"[bold red]{v:.2f} !![/bold red]"
            elif v < self._alert_entropy_warn:
                return f"[yellow]{v:.2f}[/yellow]"
            else:
                return f"[green]{v:.2f}[/green]"

        ent_tbl = Table(show_header=False, box=None, padding=(0, 2), expand=True)
        ent_tbl.add_column(justify="left")
        ent_tbl.add_row(
            f"entropy  combined [bold]{_fmt_plain(s['policy_entropy'], 2) if s['policy_entropy'] is not None else _EM_DASH}[/bold]"
            f"  │  pretrain {_fmt_ent_pre(ent_pre)}"
            f"  │  selfplay {_fmt_ent_sp(ent_sp)}"
            f"  [dim](collapse < {_COLLAPSE_THRESHOLD:.1f} nats)[/dim]"
        )

        # §101 policy-target quality row (D-Gumbel / D-Zeroloss split).
        def _fmt_metric(v: Any) -> str:
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                return _EM_DASH
            return f"{v:.2f}"

        h_full = _fmt_metric(s.get("policy_target_entropy_fullsearch"))
        h_fast = _fmt_metric(s.get("policy_target_entropy_fastsearch"))
        kl_full = _fmt_metric(s.get("policy_target_kl_uniform_fullsearch"))
        kl_fast = _fmt_metric(s.get("policy_target_kl_uniform_fastsearch"))
        n_full = s.get("n_rows_policy_loss")
        n_total = s.get("n_rows_total")
        n_str = (
            f"{int(n_full)}/{int(n_total)}"
            if (n_full is not None and n_total is not None)
            else _EM_DASH
        )
        tgt_tbl = Table(show_header=False, box=None, padding=(0, 2), expand=True)
        tgt_tbl.add_column(justify="left")
        tgt_tbl.add_row(
            f"policy target  H_full={h_full}  H_fast={h_fast}"
            f"  │  KL_u_full={kl_full}  KL_u_fast={kl_fast}"
            f"  │  n_full={n_str}"
        )

        # Assemble
        outer = Table(show_header=False, box=None, expand=True, padding=0)
        outer.add_column()
        outer.add_row(loss_tbl)
        outer.add_row(ent_tbl)
        outer.add_row(tgt_tbl)
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
