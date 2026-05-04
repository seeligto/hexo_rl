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
            "selfplay_model_entropy_batch": None,  # alias; drop 2026-05-28
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
            "colony_extension_fraction": None,
            "cluster_value_std_mean": None,
            "cluster_policy_disagreement_mean": None,
            "_colony_ext_counts": [],
            "_colony_ext_totals": [],
            # Phase B' Class-4 (§152) — stride-5 spam metric over last 50 games.
            "stride5_run_p90":  None,
            "row_max_p90":      None,
            "_stride5_run_history": collections.deque(maxlen=50),
            "_row_max_history":     collections.deque(maxlen=50),
            # ── Phase B' instrumentation snapshot ──────────────────────────
            "value_probe_decisive_mean": None,
            "value_probe_draw_mean":     None,
            "value_probe_decisive_std":  None,
            "value_probe_draw_std":      None,
            "value_probe_step":          None,
            # Sparkline buffers (last 30 readings).
            "_value_probe_decisive_hist": collections.deque(maxlen=30),
            "_value_probe_draw_hist":     collections.deque(maxlen=30),
            "buffer_corpus_fraction":     None,
            "buffer_draw_target_fraction": None,
            "buffer_six_terminal_fraction": None,
            "buffer_colony_terminal_fraction": None,
            "buffer_cap_terminal_fraction": None,
            "mv_median_range":            None,
            "mv_p90_range":               None,
            "mv_max_range":               None,
            "mv_median_distinct":         None,
            "mv_spearman_rho":            None,
            "mv_current_version":         None,
            "worker_draw_rates":          {},  # {worker_id: rate}
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
                "policy_entropy_selfplay", "selfplay_model_entropy_batch",  # alias; drop 2026-05-28
                "policy_target_entropy",
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
            # backward compat: old JSONL only has old key; populate new from old
            if self._state["selfplay_model_entropy_batch"] is None and self._state["policy_entropy_selfplay"] is not None:
                self._state["selfplay_model_entropy_batch"] = self._state["policy_entropy_selfplay"]
            return

        if event == "iteration_complete":
            for key in (
                "step", "games_total", "games_per_hour",
                "positions_per_hour", "avg_game_length",
                "win_rate_p0", "win_rate_p1", "draw_rate",
                "sims_per_sec", "buffer_size", "buffer_capacity",
                "corpus_selfplay_frac", "batch_fill_pct",
                "mcts_mean_depth", "mcts_root_concentration",
                "cluster_value_std_mean", "cluster_policy_disagreement_mean",
            ):
                if key in payload:
                    self._state[key] = payload[key]
            return

        if event == "game_complete":
            # §107 I1 — rolling colony-extension fraction over last 50 games.
            ct = payload.get("colony_extension_stone_count")
            tot = payload.get("colony_extension_stone_total")
            if isinstance(ct, int) and isinstance(tot, int):
                cnt_list = self._state.setdefault("_colony_ext_counts", [])
                tot_list = self._state.setdefault("_colony_ext_totals", [])
                cnt_list.append(ct)
                tot_list.append(tot)
                if len(cnt_list) > 50:
                    del cnt_list[0]
                    del tot_list[0]
                tt = sum(tot_list)
                self._state["colony_extension_fraction"] = (
                    sum(cnt_list) / tt if tt > 0 else 0.0
                )
            # §152 Class-4 — rolling stride5 / row-max P90 over last 50 games.
            sr = payload.get("stride5_run_max")
            rm = payload.get("row_max_density")
            if isinstance(sr, int) and isinstance(rm, int):
                self._state["_stride5_run_history"].append(sr)
                self._state["_row_max_history"].append(rm)
                sr_hist = sorted(self._state["_stride5_run_history"])
                rm_hist = sorted(self._state["_row_max_history"])
                n = len(sr_hist)
                self._state["stride5_run_p90"] = sr_hist[max(0, int(n * 0.9) - 1)]
                self._state["row_max_p90"]     = rm_hist[max(0, int(n * 0.9) - 1)]
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

        # ── Phase B' instrumentation events ──────────────────────────────
        if event == "value_probe_drift":
            self._state["value_probe_decisive_mean"] = payload.get("decisive_mean")
            self._state["value_probe_decisive_std"]  = payload.get("decisive_std")
            self._state["value_probe_draw_mean"]     = payload.get("draw_mean")
            self._state["value_probe_draw_std"]      = payload.get("draw_std")
            self._state["value_probe_step"]          = payload.get("step")
            d = payload.get("decisive_mean")
            if d is not None and isinstance(d, (int, float)) and math.isfinite(d):
                self._state["_value_probe_decisive_hist"].append(float(d))
            w = payload.get("draw_mean")
            if w is not None and isinstance(w, (int, float)) and math.isfinite(w):
                self._state["_value_probe_draw_hist"].append(float(w))
            return

        if event == "buffer_composition":
            for key, dst in (
                ("corpus_fraction",          "buffer_corpus_fraction"),
                ("draw_target_fraction",     "buffer_draw_target_fraction"),
                ("six_terminal_fraction",    "buffer_six_terminal_fraction"),
                ("colony_terminal_fraction", "buffer_colony_terminal_fraction"),
                ("cap_terminal_fraction",    "buffer_cap_terminal_fraction"),
            ):
                if key in payload:
                    self._state[dst] = payload[key]
            return

        if event == "model_version_summary":
            for key, dst in (
                ("median_range",   "mv_median_range"),
                ("p90_range",      "mv_p90_range"),
                ("max_range",      "mv_max_range"),
                ("median_distinct", "mv_median_distinct"),
                ("spearman_rho_range_vs_draw", "mv_spearman_rho"),
                ("current_version", "mv_current_version"),
            ):
                if key in payload:
                    self._state[dst] = payload[key]
            return

        if event == "worker_draw_rate":
            self._state["worker_draw_rates"] = dict(payload.get("per_worker", {}))
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
            ent_sp = payload.get("selfplay_model_entropy_batch", payload.get("policy_entropy_selfplay"))
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

        # §152 Class-4 stride-5 alarm — fires once row_max P90 (last 50 games)
        # crosses 30. Re-arms each game; the dedup-by-prefix logic in
        # _add_alert keeps this from spamming the panel.
        if event == "game_complete":
            rm_p90 = self._state.get("row_max_p90")
            if isinstance(rm_p90, int) and rm_p90 > 30:
                self._add_alert(
                    expiry, f"row_max P90 {rm_p90} — stride-5 spam (Class 4)"
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

        # §107 — live investigation row (I1 colony-extension, I2 cluster variance)
        _cex = s["colony_extension_fraction"]
        _cex_str = f"{_cex * 100:.1f}%" if _cex is not None else _EM_DASH
        _cvs = s["cluster_value_std_mean"]
        _cvs_str = f"{_cvs:.3f}" if _cvs is not None else _EM_DASH
        _cpd = s["cluster_policy_disagreement_mean"]
        _cpd_str = f"{_cpd:.3f}" if _cpd is not None else _EM_DASH
        tp_tbl.add_row(
            f"colony_ext={_cex_str}  │  cluster_v_std={_cvs_str}  │  cluster_pol_dis={_cpd_str}"
        )

        # §152 Class-4 — stride-5 spam P90 over last 50 games. Alarm at
        # row_max P90 > 30 (the diagnosis brief threshold).
        _sr_p90 = s.get("stride5_run_p90")
        _rm_p90 = s.get("row_max_p90")
        _sr_str = f"{_sr_p90}" if isinstance(_sr_p90, int) else _EM_DASH
        if isinstance(_rm_p90, int):
            _rm_str = (
                f"[bold red]{_rm_p90} !![/bold red]" if _rm_p90 > 30
                else f"[yellow]{_rm_p90}[/yellow]" if _rm_p90 > 20
                else f"{_rm_p90}"
            )
        else:
            _rm_str = _EM_DASH
        tp_tbl.add_row(
            f"stride5_run_p90={_sr_str}  │  row_max_p90={_rm_str}  "
            f"[dim](alarm row_max > 30)[/dim]"
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
        ent_sp  = s["selfplay_model_entropy_batch"]

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

        # ── Phase B' instrumentation row (only shown when populated) ─────
        instr_tbl: Table | None = None
        if (
            s["value_probe_step"] is not None
            or s["buffer_corpus_fraction"] is not None
            or s["mv_median_range"] is not None
            or s["worker_draw_rates"]
        ):
            instr_tbl = Table(show_header=False, box=None, padding=(0, 2), expand=True)
            instr_tbl.add_column(justify="left")

            # Value-probe row with sparkline.
            def _spark(values: collections.deque) -> str:
                if not values:
                    return "—"
                bars = "▁▂▃▄▅▆▇█"
                lo, hi = min(values), max(values)
                if hi - lo < 1e-6:
                    return bars[3] * len(values)
                return "".join(
                    bars[int((v - lo) / (hi - lo) * (len(bars) - 1))]
                    for v in values
                )

            d_mean = s["value_probe_decisive_mean"]
            w_mean = s["value_probe_draw_mean"]
            d_str = f"{d_mean:+.3f}" if isinstance(d_mean, (int, float)) and d_mean == d_mean else _EM_DASH
            w_str = f"{w_mean:+.3f}" if isinstance(w_mean, (int, float)) and w_mean == w_mean else _EM_DASH
            # Class-2 dominant signal: decisive drifting toward draw_value=-0.5.
            d_alarm = isinstance(d_mean, (int, float)) and d_mean < -0.30
            d_styled = f"[bold red]{d_str}[/bold red]" if d_alarm else d_str
            d_spark = _spark(s["_value_probe_decisive_hist"])
            w_spark = _spark(s["_value_probe_draw_hist"])
            vp_step = _fmt_int(s["value_probe_step"]) if s["value_probe_step"] is not None else _EM_DASH
            instr_tbl.add_row(
                f"[bold]Phase B' value-probe[/bold]  "
                f"decisive={d_styled} {d_spark}  │  "
                f"draw={w_str} {w_spark}  │  "
                f"step={vp_step}  [dim](class-2: decisive→-0.5 = collapse)[/dim]"
            )

            # Buffer composition row.
            corp = s["buffer_corpus_fraction"]
            dtf  = s["buffer_draw_target_fraction"]
            colf = s["buffer_colony_terminal_fraction"]
            sixf = s["buffer_six_terminal_fraction"]
            capf = s["buffer_cap_terminal_fraction"]

            def _f(v: Any, dp: int = 3) -> str:
                if v is None or (isinstance(v, float) and not math.isfinite(v)):
                    return _EM_DASH
                return f"{v:.{dp}f}"

            # Class-3 alarms: draw_target_fraction crossing 0.50 or
            # colony >> six are diagnostic of replay-buffer overrun by
            # draw-coded outcomes / colony-styled training rows.
            dtf_styled = (
                f"[bold red]{_f(dtf)}[/bold red]" if isinstance(dtf, (int, float)) and dtf > 0.50
                else _f(dtf)
            )
            col_alarm = (
                isinstance(colf, (int, float)) and isinstance(sixf, (int, float))
                and colf > sixf and (colf - sixf) > 0.10
            )
            col_styled = (
                f"[bold red]{_f(colf)}[/bold red]" if col_alarm else _f(colf)
            )
            instr_tbl.add_row(
                f"[bold]buffer comp[/bold]  "
                f"corpus={_f(corp)}  │  draw_target={dtf_styled}  │  "
                f"six={_f(sixf)}  colony={col_styled}  cap={_f(capf)}"
            )

            # Model-version row.
            mvr = s["mv_median_range"]
            mvp = s["mv_p90_range"]
            mvm = s["mv_max_range"]
            mvd = s["mv_median_distinct"]
            mvc = s["mv_current_version"]
            rho = s["mv_spearman_rho"]
            rho_str = (
                f"{rho:+.2f}" if isinstance(rho, (int, float)) and rho == rho else _EM_DASH
            )
            # Class-1 alarm: significant Spearman ρ between version range and draw outcome.
            rho_alarm = isinstance(rho, (int, float)) and rho == rho and abs(rho) > 0.20
            rho_styled = f"[bold red]{rho_str}[/bold red]" if rho_alarm else rho_str
            instr_tbl.add_row(
                f"[bold]model-ver[/bold]  cur={_fmt_int(mvc) if mvc is not None else _EM_DASH}  │  "
                f"per-game range med={_fmt_int(mvr) if mvr is not None else _EM_DASH}  "
                f"P90={_fmt_int(mvp) if mvp is not None else _EM_DASH}  "
                f"max={_fmt_int(mvm) if mvm is not None else _EM_DASH}  "
                f"distinct_med={_fmt_int(mvd) if mvd is not None else _EM_DASH}  │  "
                f"ρ(range,draw)={rho_styled}  [dim](class-1: ρ>0 = stale-dispatch)[/dim]"
            )

            # Per-worker draw rate row (compact, top variance).
            wdr = s["worker_draw_rates"] or {}
            if wdr:
                rates_sorted = sorted(wdr.items(), key=lambda kv: kv[0])
                rate_strs = []
                for wid, r in rates_sorted[:8]:
                    style = "[bold red]" if r > 0.80 else ("[yellow]" if r > 0.60 else "")
                    suffix = "[/bold red]" if style == "[bold red]" else ("[/yellow]" if style == "[yellow]" else "")
                    rate_strs.append(f"w{wid}={style}{r:.2f}{suffix}")
                more = "" if len(wdr) <= 8 else f" (+{len(wdr) - 8} more)"
                instr_tbl.add_row(
                    f"[bold]per-worker draw[/bold] (last 50)  " + "  ".join(rate_strs) + more
                )

        # Assemble
        outer = Table(show_header=False, box=None, expand=True, padding=0)
        outer.add_column()
        outer.add_row(loss_tbl)
        outer.add_row(ent_tbl)
        outer.add_row(tgt_tbl)
        outer.add_row(tp_tbl)
        outer.add_row(buf_tbl)
        if instr_tbl is not None:
            outer.add_row(instr_tbl)

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
