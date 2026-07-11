#!/usr/bin/env python3
"""d1m_monitor.py — rich TUI for the D-1M Gumbel 1M run on vast.

Live, color-coded, graphed monitor. THIN RENDERER over
hexo_rl.monitoring.run_feed_reader (B4 §D-DECIDE Track-B): all Feed-A parsing +
verdict math (eval table, 30k trajectory bins, step_at bisect, recent-colony,
gap-skipping effective rate, the sealbot_slope CI, depth_health) lives there, in
ONE pure place that is unit-testable without ssh. This file owns ONLY rich/plotext
rendering + arg parsing.

Pulls SPARSE decision-relevant events from the remote run-log over ssh
(read_remote_ssh = remote grep, never transfers the 150MB file), then reduces
LOCAL via parse_feed.

Usage:
  .venv/bin/python scripts/d1m_monitor.py            # live TUI, ~15s refresh
  .venv/bin/python scripts/d1m_monitor.py --once     # one plain frame, exit
  .venv/bin/python scripts/d1m_monitor.py --interval 30
  .venv/bin/python scripts/d1m_monitor.py --no-charts # skip plotext, sparklines

(--once supersedes scripts/d1m_status.sh — one plain frame, exit.)

In live mode press '?' to toggle the help/legend overlay, 'q' to quit.

Run identity + Feed-A parse knobs come from configs/monitoring.yaml `run_feed:`
(host, repo, default_log_path, encoding, sealbot_* CI knobs, bin/sample sizes) —
load_run_feed_config() HARD-ERRORS on a missing block/key. Env overrides still
win at the call site:
  D1M_HOST  D1M_REPO  D1M_LOG  D1M_ENCODING
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime

# repo root on sys.path so the hexo_rl package imports when run as a script
# (sys.path[0] is scripts/ when invoked as `python scripts/d1m_monitor.py`).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hexo_rl.monitoring.run_feed_reader import (  # noqa: E402
    depth_health,
    load_run_feed_config,
    parse_feed,
    read_live_tip,
    read_remote_ssh,
    resolve_run_target,
    sealbot_point_sigmas,
    sealbot_slope,
)


# ----------------------------------------------------------------------------
# Renderer-owned thresholds — entropy collapse floors, fp16 drop frac, depth
# fractions. Env-overridable. (The sealbot CI knobs + run identity + bin/sample
# sizes live in configs/monitoring.yaml run_feed:, loaded via load_run_feed_config.)
# ----------------------------------------------------------------------------
def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# policy entropy: LOW-FLOOR-ONLY collapse detector. High entropy is HEALTHY
# (upper reference = ln(policy_logit_count), derived at runtime — never typed).
POLICY_ENTROPY_COLLAPSE_FLOOR = _env_float("D1M_POLICY_ENTROPY_COLLAPSE_FLOOR", 1.0)
POLICY_ENTROPY_WARN_FLOOR = _env_float("D1M_POLICY_ENTROPY_WARN_FLOOR", 1.5)

# fp16 AMP-scale canary: alert on a SHARP self-baseline DROP (loss-scaler
# backoff burst), not an absolute level. Fraction of trailing-median.
FP16_SCALE_DROP_FRAC = _env_float("D1M_FP16_SCALE_DROP_FRAC", 0.5)

# depth-health: judge depth vs the RUN'S OWN settled baseline (fractions of it).
DEPTH_STABLE_FRAC = _env_float("D1M_DEPTH_STABLE_FRAC", 0.05)    # +/-5% of own baseline = stable
DEPTH_REGRESS_FRAC = _env_float("D1M_DEPTH_REGRESS_FRAC", 0.12)  # >12% below baseline = regression
CONC_MOVE = _env_float("D1M_CONC_MOVE", 0.03)                    # root_conc co-move threshold (abs)


def _policy_logit_count(encoding_name):
    """Runtime lookup of the run's policy-head size. DERIVED — never hardcode
    362 or ln(362)=5.89. Falls back to None if the registry is unreachable."""
    try:
        from hexo_rl.encoding import lookup  # noqa: PLC0415
        return lookup(encoding_name).policy_logit_count
    except Exception:
        return None


def _entropy_upper_ref(encoding_name):
    """Upper reference for policy entropy = ln(policy_logit_count), the uniform
    distribution's entropy over the head. DERIVED at runtime, never a literal."""
    n = _policy_logit_count(encoding_name)
    return math.log(n) if n else None


# ----------------------------------------------------------------------------
# HELP CONTENT (verified interpretations — keep accurate; do not invent)
# ----------------------------------------------------------------------------
HELP = {
    "wr_best": "candidate vs frozen best/anchor (n=200, +/-CI). Promotes when >=0.55 "
               "AND CI lower-bound >0.5; at n=200 the de-facto bar ~=0.57 (was n=400/0.55, "
               "cut for eval cost). Climbing = improving.",
    "wr_bootstrap_anchor": "DIAGNOSTIC WR vs the FROZEN 8300 bootstrap (n=50, every 120k). "
                           "Absolute-reference strength (complements the rotating wr_best). "
                           "Rising = stronger than the start; never gates promotion (floor off).",
    "wr_sealbot": "external strength ANCHOR — model plays MCTS at temp-0.5 with 0 random "
                  "openings, n=100 (Wilson95 +/-~10pp). ~0.55 is a SOFT convention (carried "
                  "from the §101 graduation gate vs KrakenBot's 0.76), NOT a hard gate: docs "
                  "say never abort on a single SealBot step. Read the SLOPE; '200k' is a "
                  "roadmap checkpoint where the climb should show, not a threshold. (Separate "
                  "argmax_n n=20 temp-0 eval = the §170 DRIFT detector, not this metric.)",
    "elo_estimate": "Bradley-Terry internal rating; smooth strength but NOISY with a "
                    "small checkpoint pool early (can dip across a real promotion). "
                    "Don't over-read single moves.",
    "forced_win_conversion": "fraction of MCTS-detected forced wins the model converts. "
                             "Coherence proxy. Read SLOPE + CO-OCCURRENCE, never an absolute "
                             "threshold: golong was 0.89->0.66 DECLINE coinciding with "
                             "fragmentation (colony up, components up). A lone decline can be "
                             "noise or a population shift (watch off_window). Warn only when a "
                             "sustained ~>20%-from-baseline decline coincides with rising colony.",
    "off_window_forced_win_rate": "forced wins landing off the NN window. Near 0 likely "
                                  "in-window-dominated/sparse. Historically contested (a "
                                  "19%->54% reinterpretation).",
    "colony_extension_fraction": "fraction of stones extending a colony (colony-attractor "
                                 "signature). Healthy <0.15; rising is the fragmentation "
                                 "co-signal that, with falling conversion, = golong.",
    "value_bce_gap": "selfplay BCE should TRACK corpus BCE. Canary FIRES if selfplay BCE "
                     "FLOORS (~0.69 = random) while corpus keeps dropping -> overfits corpus, "
                     "not learning self-play -> triggers a corpus-to-zero ablation. Watch the "
                     "GAP trend (currently widening).",
    "value_accuracy": "value_accuracy_selfplay/masked should rise; decline >10pp over 10k "
                      "steps = flag.",
    "avg_sigma": "value-spread V-spread canary (also alt_spread/t3_spread). A sustained "
                 "excursion not self-correcting by 200-300k = pre-registered reactive-lever "
                 "trigger.",
    "t3_spread": "colony-capture canary V_spread = mean V(colony bank) - mean V(extension "
                 "bank) over a frozen 40-pos T3 bank. HIGH = healthy (anchor +0.617); "
                 "COLLAPSE toward 0 = value head can't tell a dead blob from a winning open "
                 "run. WARN <0.30, SOFT-ABORT <0.20. Should hold high, NOT trend to zero.",
    "alt_spread": "same canary on the bot-corpus alt bank (T3 amplifies ~3x). Anchor +0.212; "
                  "gates T3/3: WARN <0.10, SOFT-ABORT <0.07. NaN when the run's plane count "
                  "differs from the v6 alt fixture (verdict rests on t3 then).",
    "draw_rate": "HARD-ABORT at >=0.55 for 3 consecutive evals. Currently ~0.",
    "grad_norm": "HARD-ABORT at >=10. Currently ~1.5.",
    "lr": "cosine annealing over the 1M horizon (monotone ~2e-3 -> 5e-4). If it ever ticks "
          "UP the horizon pin failed (LR cycling).",
    "policy_entropy_selfplay": "LOW-FLOOR-ONLY collapse detector. Upper reference = "
                               "ln(policy_logit_count) (the uniform-head entropy, DERIVED at "
                               "runtime from the encoding registry — shown live in the health "
                               "panel). High entropy is HEALTHY, never flagged. Only a drop "
                               "toward the collapse floor (overconfident policy) warns/aborts.",
    "xo_winrate": "first-player balance; healthy ~0.51-0.54.",
    "mcts_mean_depth": "Gumbel search depth — shallow BY DESIGN (Gumbel-SH buys breadth, "
                       "not depth; the m16/n150 root fan-out trades tree depth for sampled "
                       "actions). A raw number is ambiguous and there is NO meaningful absolute "
                       "target — read the 'depth read' verdict, which judges vs the RUN'S OWN "
                       "rolling baseline (median of history bins): within +/-5% = stable, "
                       ">12% below = REGRESSION. Alert only on a SHARP deviation from that "
                       "self-baseline; the floor line is baseline-relative, not a constant.",
    "mcts_root_concentration": "max root child visits / total root visits in [0,1]. Co-read with "
                               "depth. Under GUMBEL-SH (this run) rising concentration is the "
                               "EXPECTED Sequential-Halving behavior — DESCRIPTIVE, not an alarm. "
                               "(Under PUCT, concentration RISING while depth FALLS would read as "
                               "overconfident shallow cutoffs.) Concentration FLAT while depth "
                               "falls = game-length/structural shift, not a search-quality loss.",
    "longest_line_fraction": "n/a — NOT instrumented in this run (not emitted to the log).",
    "n_components": "n/a — NOT instrumented in this run (not emitted to the log).",
}

# ----------------------------------------------------------------------------
# threshold classifier -> rich style ("green"/"yellow"/"red"/"") matching HELP
# ----------------------------------------------------------------------------

def cls(metric, v):
    if v is None or not isinstance(v, (int, float)):
        return ""
    try:
        if metric == "draw_rate":
            return "red" if v >= 0.55 else ("yellow" if v >= 0.4 else "green")
        if metric == "grad_norm":
            return "red" if v >= 10 else ("yellow" if v >= 5 else "green")
        if metric == "policy_entropy_selfplay":
            # LOW-FLOOR ONLY. High entropy (up to ln(head)) is healthy, never
            # flagged. Only a drop toward the named collapse floor warns/aborts.
            if v < POLICY_ENTROPY_COLLAPSE_FLOOR:
                return "red"
            if v < POLICY_ENTROPY_WARN_FLOOR:
                return "yellow"
            return "green"
        if metric == "mcts_mean_depth":
            # NO absolute depth floor. Depth is shallow-by-design under Gumbel;
            # the verdict comes from depth_health()'s run-relative baseline.
            return ""
        if metric == "t3_spread":   # HIGH healthy: WARN<0.30, SOFT-ABORT<0.20
            return "red" if v < 0.20 else ("yellow" if v < 0.30 else "green")
        if metric == "alt_spread":  # alt gates = T3/3: WARN<0.10, SOFT-ABORT<0.07
            return "red" if v < 0.07 else ("yellow" if v < 0.10 else "green")
        if metric == "xo_winrate":
            return "green" if 0.47 <= v <= 0.56 else ("yellow" if 0.42 <= v <= 0.60 else "red")
        if metric == "colony":
            return "green" if v < 0.15 else ("yellow" if v < 0.25 else "red")
        if metric == "wr_sealbot":
            # absolute level is DESCRIPTIVE only — never color-gated. The
            # green/yellow read is SLOPE-based (sealbot_slope), computed in build().
            return ""
        if metric == "wr_best":
            return "green" if v >= 0.55 else ("yellow" if v >= 0.5 else "red")
        if metric == "value_bce_gap":  # selfplay - corpus
            return "red" if v >= 0.1 else ("yellow" if v >= 0.05 else "green")
        if metric == "value_accuracy":
            return "green" if v >= 0.65 else ("yellow" if v >= 0.55 else "red")
    except Exception:
        pass
    return ""


# ----------------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------------

def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return ("%%.%df" % nd) % v
    return str(v)


def wrci(wr, ci):
    s = "-" if wr is None else (fmt(wr) if isinstance(wr, float) else str(wr))
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        return "%s +/-%.2f" % (s, (ci[1] - ci[0]) / 2)
    if isinstance(ci, (int, float)):
        return "%s +/-%.2f" % (s, ci)
    return s


def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else None


SPARK = "▁▂▃▄▅▆▇█"


def sparkline(vals):
    pts = [v for v in vals if isinstance(v, (int, float))]
    if not pts:
        return "(no data)"
    lo, hi = min(pts), max(pts)
    rng = hi - lo
    if rng < 1e-12:
        return SPARK[3] * len(pts)
    return "".join(SPARK[min(7, int((v - lo) / rng * 7))] for v in pts)


# ----------------------------------------------------------------------------
# plotext chart -> string (fallback handled by caller)
# ----------------------------------------------------------------------------

def _plt():
    try:
        import plotext as plt  # noqa: PLC0415
        return plt
    except Exception:
        return None


def chart(series, title, width=58, height=10, hlines=None):
    """series: list of (label, xs, ys, color). hlines: list of (y, color)
    horizontal reference lines (e.g. a run-relative baseline / floor).
    Return string or None."""
    plt = _plt()
    series = [(l, xs, ys, c) for (l, xs, ys, c) in series if xs and ys]
    if plt is None or not series:
        return None
    try:
        plt.clf()
        plt.theme("clear")
        plt.plotsize(width, height)
        for label, xs, ys, color in series:
            plt.plot(xs, ys, label=label, marker="braille", color=color)
        for y, color in (hlines or []):
            if isinstance(y, (int, float)):
                try:
                    plt.hline(y, color)
                except Exception:
                    pass
        plt.title(title)
        s = plt.build()
        plt.clf()
        return s
    except Exception:
        return None


# ----------------------------------------------------------------------------
# fetch — single ssh round-trip via the IO helper, then PURE local parse.
# ----------------------------------------------------------------------------

def fetch(cfg, host, repo, log):
    try:
        # remote subsamples high-freq events (train_step/game_complete fire every
        # step) -> KB transfer; parse_feed must then NOT re-subsample (=1).
        records = read_remote_ssh(host, repo, log,
                                  ts_sample=cfg.ts_sample, gc_sample=cfg.gc_sample)
    except RuntimeError as e:
        return {"error": str(e)}
    snap = parse_feed(records, bin_width=cfg.bin_width, ts_sample=1, gc_sample=1)
    out = snap.as_dict()
    out["log"] = log  # run identity for run-specific panels (e.g. run2 curriculum)
    # LIVE-TIP override (resume- + clock-skew-robust): the glob read subsamples and
    # its END-block captures the last FILE's tail (an archived restart segment), so
    # the step/last-event lag the true tip after a resume. read_live_tip resolves the
    # newest-mtime log, tails its true-latest summary/train_step un-subsampled, and
    # reads the REMOTE clock for a skew-free age. Override the header/step fields with
    # it; the trajectory/eval panels keep the full-history glob parse.
    tip = read_live_tip(host, repo, log)
    if tip:
        out["live_tip"] = tip
        # Compare by TIMESTAMP not step: a restart can reset step numbers while
        # wall-clock always advances. Step comparison failed when an archived
        # segment had a higher step watermark than the freshly-restarted live log.
        _ts = tip.get("last_summary")
        if _ts:
            _cur = out.get("last_summary")
            if not _cur or (_ts.get("timestamp") or "") >= (_cur.get("timestamp") or ""):
                out["last_summary"] = _ts
        _tt = tip.get("last_train")
        if _tt:
            _cur = out.get("last_train")
            if not _cur or (_tt.get("timestamp") or "") >= (_cur.get("timestamp") or ""):
                out["last_train"] = _tt
    return out


def _run2_variant_cfg():
    """run2_mw_fresh curriculum knobs, read from the variant yaml (local repo copy).

    Returns (radius_schedule, mixing) or (None, None) when the yaml is absent —
    the panel degrades to live-log fields only.
    """
    try:
        import yaml
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "configs" / "variants" / "run2_mw_fresh.yaml"
        cfg = yaml.safe_load(p.read_text())
        sched = (cfg.get("selfplay") or {}).get("legal_move_radius_schedule")
        mixing = cfg.get("mixing") or {}
        return sched, mixing
    except Exception:
        return None, None


def _run2_panel(step, ls, fwt, eta_rate):
    """run2_mw_fresh curriculum / mixing panel — the run-defining knobs the
    generic D-1M panels don't surface: radius stage (+ OQ8 widen-early signature
    gate), corpus mixing weight decay, buffer fill, entropy vs the MEASURED
    2.1-2.9 band (sprint-log corrected 2026-07-04; the old ~3-6 citation is
    falsified)."""
    from rich.table import Table
    from rich.text import Text
    from rich.panel import Panel
    from rich.console import Group

    t = Table.grid(padding=(0, 1))
    t.add_column(style="bold", justify="right")
    t.add_column()

    sched, mixing = _run2_variant_cfg()
    if sched:
        cur = max((e for e in sched if e["step"] <= step), key=lambda e: e["step"], default=sched[0])
        nxt = min((e for e in sched if e["step"] > step), key=lambda e: e["step"], default=None)
        stage_idx = sched.index(cur) + 1
        if nxt is not None:
            rem = nxt["step"] - step
            eta = "  ETA %.1fh" % (rem / eta_rate) if eta_rate > 0 else ""
            nxt_txt = "next r=%d @ %s (%s away%s)" % (
                nxt["radius"], f"{nxt['step']:,}", f"{rem:,}", eta)
        else:
            nxt_txt = "final stage (hard rule ceiling)"
        t.add_row("radius stage", "S%d  r=%d   %s" % (stage_idx, cur["radius"], nxt_txt))
        conv_last = None
        if fwt:
            conv_last = fwt[-1].get("forced_win_conversion")
        sig = "conv %s vs >=0.85 x2 25k-reads (slope <=+0.02, n>=30)" % (
            ("%.3f" % conv_last) if conv_last is not None else "?")
        t.add_row("OQ8 widen-early", Text(sig, style=(
            "bold yellow" if (conv_last is not None and conv_last >= 0.85) else "dim")))

    w_pre = ls.get("pretrained_weight")
    if w_pre is not None:
        floor = (mixing or {}).get("min_pretrained_weight", 0.1)
        t.add_row("mixing w_pre", "%.3f  (0.8 -> %.1f floor over 200k; corpus share of batch)"
                  % (w_pre, floor))
    bs, bc = ls.get("buffer_size"), ls.get("buffer_capacity")
    if bs is not None:
        sp_pct = ls.get("buffer_self_play_pct")
        t.add_row("buffer", "%s / %s%s" % (
            f"{bs:,}", f"{bc:,}" if bc else "?",
            ("   self-play %.0f%%" % (sp_pct * 100)) if sp_pct is not None else ""))
    dr = ls.get("draw_rate")
    if dr is not None:
        t.add_row("draw_rate", Text("%.3f  (hard-abort 0.55 x3 evals)" % dr,
                                    style="bold red" if dr >= 0.45 else ""))
    pe = ls.get("policy_entropy_selfplay")
    if pe is not None:
        in_band = 2.1 <= pe <= 2.9
        t.add_row("entropy_sp", Text("%.3f  vs measured band 2.1-2.9 (<1.0 collapse)" % pe,
                                     style="" if in_band else "yellow"))
    note = Text("seeding OFF at launch (D-WS3V3 THIN-STILL); mid-run intervention if "
                "trap-flip plateaus — see run2 spec Decision 3", style="italic dim")
    return Panel(Group(t, note), title="run2 curriculum / mixing (run-defining knobs)",
                 border_style="green")


# ----------------------------------------------------------------------------
# build the rich renderables
# ----------------------------------------------------------------------------

def build(data, show_help, use_charts, cfg, default_encoding):
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.columns import Columns
    from rich.console import Group

    if "error" in data:
        return Panel(Text("FETCH ERROR: %s" % data["error"], style="bold red"),
                     title="D-1M monitor", border_style="red")

    ls = data.get("last_summary") or {}
    lt = data.get("last_train") or {}
    total = data.get("total_steps") or 1000000
    step = ls.get("step", 0)
    rounds = data.get("eval_rounds") or []
    eval_phases = data.get("eval_phases") or []
    fwt = data.get("fwt") or []
    traj = data.get("traj") or []
    # policy-entropy upper reference = ln(policy_logit_count), DERIVED at runtime
    # from the run's encoding (registry lookup) — never a literal.
    enc_name = data.get("encoding") or default_encoding
    entropy_upper_ref = _entropy_upper_ref(enc_name)
    depth_series = [(r["step"], r["depth"]) for r in traj if "depth" in r]
    conc_series = [(r["step"], r["conc"]) for r in traj if "conc" in r]
    sigma_series = [(r["step"], r["sigma"]) for r in traj if "sigma" in r]
    t3_series = [(r["step"], r["t3"]) for r in traj if "t3" in r]
    alt_series = [(r["step"], r["alt"]) for r in traj if "alt" in r]
    is_gumbel = data.get("is_gumbel")
    dh = depth_health(depth_series, conc_series, is_gumbel,
                      DEPTH_STABLE_FRAC, DEPTH_REGRESS_FRAC, CONC_MOVE)

    panels = []

    # ---- progress / ETA ----
    eta_rate = data.get("rate_effective") or data.get("rate_overall") or 0.0
    prog = Table.grid(padding=(0, 1))
    prog.add_column(justify="right", style="bold")
    prog.add_column()
    pct = (step / total * 100) if total else 0
    prog.add_row("step", "{:,} / {:,}  ({:.1f}%)".format(step, total, pct))
    prog.add_row("rate", "{:,.0f}/hr effective (eval-incl) | {:,.0f}/hr recent-1h".format(
        data.get("rate_effective", 0), data.get("rate_recent", 0)))
    prog.add_row("throughput", "{:,.0f} games/hr | {:,.0f} sims/s".format(
        ls.get("games_per_hour", 0), ls.get("sims_per_sec", 0)))
    _is_run2 = "run2" in (data.get("log") or "")
    if _is_run2:
        miles = [(5000, "5k probe gate"), (25000, "25k signature"),
                 (200000, "200k radius S2"), (1000000, "1M target")]
    else:
        miles = [(30000, "30k first-eval"), (200000, "200k gate"), (1000000, "1M target")]
    miles = [m for m in miles if m[0] <= total]
    if total not in [m[0] for m in miles]:
        miles.append((total, "target"))
    etas = []
    for tgt, label in miles:
        rem = tgt - step
        if rem <= 0:
            etas.append("%s: reached" % label)
        elif eta_rate > 0:
            h = rem / eta_rate
            etas.append("%s: %.1fh (%.1fd)" % (label, h, h / 24))
        else:
            etas.append("%s: ?" % label)
    prog.add_row("ETA", "  |  ".join(etas))
    panels.append(Panel(prog, title="progress / ETA  (eval-inclusive effective pace)", border_style="cyan"))

    # ---- strength ----
    # wr_sealbot read = SLOPE over eval history (level is descriptive only).
    # Pass each point's LOGGED ci_sealbot half-width as its measurement sigma so
    # the slope CI propagates per-point sampling noise, not just fitted residuals.
    sb_xs = [r.get("step") for r in rounds]
    sb_ys = [r.get("wr_sealbot") for r in rounds]
    sb_sigmas = sealbot_point_sigmas(rounds, cfg.sealbot_fallback_sigma)
    sb_slope = sealbot_slope(
        sb_xs, sb_ys, sb_sigmas,
        min_pts=cfg.sealbot_slope_min_points,
        ci_level=cfg.sealbot_slope_ci_level,
        min_rise=cfg.sealbot_min_rise,
        fallback_sigma=cfg.sealbot_fallback_sigma,
    )
    st = Table(title=None, expand=True, show_edge=False)
    # value-head fields per eval round -> vfc2 (numeric) + g4 (pass/fail bool)
    for c in ("step", "prom", "wr_best", "wr_sealbot", "vs_boot", "wr_random", "elo",
              "vfc2_wmax", "g4_band"):
        st.add_column(c, justify="right" if c != "prom" else "center")
    if rounds:
        for d in rounds:
            wb = d.get("wr_best")
            wsb = d.get("wr_sealbot")
            wbo = d.get("wr_bootstrap_anchor")
            g4 = d.get("g4_value_head_band_pass")
            g4txt = ("[green]pass[/]" if g4 else "[red]FAIL[/]") if g4 is not None else "[dim]-[/]"
            st.add_row(
                str(d.get("step", "?")),
                "[green]yes[/]" if d.get("promoted") else "no",
                Text(wrci(wb, d.get("ci_best")), style=cls("wr_best", wb)),
                # wsb cell DESCRIPTIVE (level shown), color from the SLOPE read
                Text(wrci(wsb, d.get("ci_sealbot")), style=sb_slope["style"]),
                (fmt(wbo, 3) if wbo is not None else "[dim]-[/]"),
                fmt(d.get("wr_random"), 2),
                fmt(d.get("elo_estimate"), 0),
                (fmt(d.get("value_fc2_weight_abs_max"), 4)
                 if d.get("value_fc2_weight_abs_max") is not None else "[dim]-[/]"),
                g4txt,
            )
        sb_note = Text("wr_sealbot read = SLOPE (level descriptive): %s   "
                       "[temp-0.5, n=100; ~0.55 a soft anchor, NOT a gate]" % sb_slope["label"],
                       style="italic " + (sb_slope["style"] or "dim"))
        body = Group(st, sb_note)
    else:
        body = Text("no COMPLETED eval rounds yet (evaluation_round_complete fires at "
                    "round END; first at step %s; now at %d)"
                    % ("25,000" if _is_run2 else "30,000", step),
                    style="dim")
    # Per-phase live view — surfaces sealbot/best_arena/random WR as each phase
    # LANDS (evaluation_games_complete), so an in-progress or interrupted round is
    # visible before the round-summary event. Grouped by round (a `random` phase
    # starts a new round); step ~ the checkpoint step (random completes ~fast).
    if eval_phases:
        groups = []
        cur = None
        for p in eval_phases:
            if cur is None or p.get("phase") == "random":
                cur = {"step": p.get("step"), "phases": {}}
                groups.append(cur)
            cur["phases"][p.get("phase")] = p.get("wr")
        pt = Table(title=None, expand=True, show_edge=False)
        for c in ("~step", "random", "sealbot", "best_arena"):
            pt.add_column(c, justify="right")
        for g in groups[-4:]:
            ph = g["phases"]
            pt.add_row(
                "~%s" % f'{g["step"]:,}' if g.get("step") is not None else "?",
                fmt(ph.get("random"), 2),
                Text(fmt(ph.get("sealbot"), 3), style=cls("wr_sealbot", ph.get("sealbot"))),
                fmt(ph.get("best_arena"), 3),
            )
        pnote = Text("per-phase (evaluation_games_complete) — live as each phase lands; "
                     "step ~checkpoint. SealBot WR is the true-north signal.",
                     style="italic dim")
        body = Group(body, pt, pnote) if rounds else Group(pt, pnote)
    panels.append(Panel(body, title="strength (eval rounds + live per-phase)", border_style="magenta"))

    # ---- coherence cluster (HEADLINE) ----
    coh = Table.grid(padding=(0, 1))
    coh.add_column(style="bold", justify="right")
    coh.add_column()
    if fwt:
        conv = [d.get("forced_win_conversion") for d in fwt]
        offw = [d.get("off_window_forced_win_rate") for d in fwt]
        coh.add_row("forced_win_conversion",
                    Text("%s   %s" % (sparkline(conv),
                         "  ".join(fmt(c, 3) for c in conv)),
                         style="yellow" if (len(conv) >= 2 and conv[-1] < conv[0] * 0.8) else ""))
        coh.add_row("off_window_forced_win",
                    "%s   %s" % (sparkline(offw), "  ".join(fmt(o, 4) for o in offw)))
    else:
        coh.add_row("forced_win_conversion", Text("(no forced_win_trend yet)", style="dim"))
    # colony from binned trajectory + recent mean
    col_series = [(r["step"], r.get("colony")) for r in traj if r.get("colony") is not None]
    col_vals = [c for _, c in col_series]
    crm = data.get("colony_recent_mean")
    coh.add_row("colony_extension",
                Text("%s   recent(%d games) %s" % (
                    sparkline(col_vals) if col_vals else "(no data)",
                    data.get("colony_recent_n", 0), fmt(crm, 3)),
                    style=cls("colony", crm)))
    coh.add_row("[dim]longest_line_fraction[/]", Text("n/a — not instrumented in this run", style="dim"))
    coh.add_row("[dim]n_components[/]", Text("n/a — not instrumented in this run", style="dim"))
    coh_note = Text("read SLOPE + CO-OCCURRENCE — falling conversion + rising colony = golong",
                    style="italic dim")
    panels.append(Panel(Group(coh, coh_note), title="COHERENCE cluster (golong kill-gate)",
                        border_style="yellow"))

    # ---- run2 curriculum / mixing (only on a run2 log) ----
    if _is_run2:
        panels.append(_run2_panel(step, ls, fwt, eta_rate))

    # ---- value-divergence canary ----
    vd = Table.grid(padding=(0, 1))
    vd.add_column(style="bold", justify="right")
    vd.add_column()
    sp, co = lt.get("value_bce_selfplay"), lt.get("value_bce_corpus")
    gap = (sp - co) if (isinstance(sp, float) and isinstance(co, float)) else None
    vd.add_row("value_bce", "selfplay %s | corpus %s" % (fmt(sp), fmt(co)))
    vd.add_row("gap (sp-co)", Text(fmt(gap), style=cls("value_bce_gap", gap)))
    gap_series = [(r["step"], r["bce_sp"] - r["bce_co"]) for r in traj
                  if "bce_sp" in r and "bce_co" in r]
    vd.add_row("gap trend", "%s   %s" % (
        sparkline([g for _, g in gap_series]) if gap_series else "(no data)",
        "n=%d bins" % len(gap_series)))
    va = lt.get("value_accuracy_selfplay")
    vd.add_row("value_acc", Text("selfplay %s | masked %s | corpus %s" % (
        fmt(va), fmt(lt.get("value_accuracy_masked")), fmt(lt.get("value_accuracy_corpus"))),
        style=cls("value_accuracy", va)))
    panels.append(Panel(vd, title="value-divergence canary (selfplay should track corpus)",
                        border_style="blue"))

    # ---- Gumbel-target health + opening diversity + AMP-stability ----
    gt = data.get("gumbel_targets") or {}
    gtr = data.get("gumbel_targets_recent") or {}
    gp = Table.grid(padding=(0, 1))
    gp.add_column(style="bold", justify="right")
    gp.add_column()

    def _gt_row(label, key, nd=4, suffix=""):
        v = gt.get(key)
        hist = gtr.get(key) or []
        spark = sparkline(hist) if len(hist) >= 2 else "(<2 pts)"
        # self-baseline read: compare latest vs trailing-window mean (descriptive)
        base = (sum(hist[:-1]) / (len(hist) - 1)) if len(hist) >= 2 else None
        delta = ("  (base %s, %+.1f%%)" % (
            fmt(base, nd), (v - base) / base * 100) if base else "")
        gp.add_row(label, "%s   %s%s%s" % (spark, fmt(v, nd), suffix, delta))

    _gt_row("target_entropy_fs", "policy_target_entropy_fullsearch")
    _gt_row("target_kl_uniform_fs", "policy_target_kl_uniform_fullsearch")
    gp.add_row("[dim]gumbel-target[/]",
               Text("full-search head: entropy + KL-to-uniform travel TOGETHER "
                    "(read self-baseline trend, no absolute target)", style="italic dim"))
    _gt_row("early_game_entropy", "early_game_entropy_mean")
    _gt_row("early_game_top1_mass", "early_game_top1_mass_mean")
    gp.add_row("[dim]opening-div[/]",
               Text("diversity = effective-n (§D-ARGMAX): low early entropy / high "
                    "top1 mass = collapsed openings, fewer DISTINCT games", style="italic dim"))
    fp16 = data.get("fp16_scale_last")
    fp16r = data.get("fp16_scale_recent") or []
    fp16_base = (_median([x for x in fp16r[:-1]]) if len(fp16r) >= 3 else None)
    fp16_style = ""
    fp16_note = ""
    if fp16 is not None and fp16_base:
        if fp16 < fp16_base * FP16_SCALE_DROP_FRAC:
            fp16_style, fp16_note = "yellow", "  SHARP DROP vs baseline %s" % fmt(fp16_base, 0)
        else:
            fp16_note = "  (baseline %s, stable)" % fmt(fp16_base, 0)
    gp.add_row("fp16_scale",
               Text("%s   %s%s" % (
                   sparkline(fp16r) if len(fp16r) >= 2 else "(<2 pts)",
                   fmt(fp16, 0), fp16_note), style=fp16_style))
    gp.add_row("[dim]amp-stability[/]",
               Text("loss-scaler level is irrelevant; a sharp drop = AMP backoff burst "
                    "(numeric instability) — alert on self-baseline only", style="italic dim"))
    panels.append(Panel(gp, title="Gumbel-target / opening-diversity / AMP-stability",
                        border_style="cyan"))

    # ---- log-derived overhead + owed signals ----
    od = Table.grid(padding=(0, 1))
    od.add_column(style="bold", justify="right")
    od.add_column()
    eof = data.get("eval_overhead_frac")
    od.add_row("eval overhead",
               Text("%s of wall  [derived from timestamp gaps (approx)]" % (
                   ("%.1f%%" % (eof * 100)) if isinstance(eof, (int, float)) else "-"),
                   style="yellow" if (isinstance(eof, (int, float)) and eof >= 0.30) else ""))
    od.add_row("[dim]distinct-game frac[/]",
               Text("owed to engine-add (no game-id / hash in game_complete); "
                    "interim salvage in F8 replay-analyzer via move-seq hashing", style="dim"))
    od.add_row("[dim]value-calibration[/]",
               Text("owed to engine-add (no per-sample predicted_value+outcome in "
                    "train_step log)", style="dim"))
    panels.append(Panel(od, title="log-derived signals (overhead + owed)", border_style="blue"))

    # ---- health ----
    he = Table.grid(padding=(0, 1))
    he.add_column(style="bold", justify="right")
    he.add_column()

    def hrow(label, metric, val, nd=3):
        he.add_row(label, Text(fmt(val, nd), style=cls(metric, val)))

    hrow("draw_rate", "draw_rate", ls.get("draw_rate"))
    hrow("grad_norm", "grad_norm", lt.get("grad_norm"))
    he.add_row("lr", Text(fmt(lt.get("lr"), 6), style=cls("lr", lt.get("lr"))))
    # policy entropy — LOW-FLOOR ONLY. Show the derived upper reference
    # ln(policy_logit_count) so the user reads the value against the healthy ceiling.
    pe = ls.get("policy_entropy_selfplay")
    _ref = ("ref ln(%s head)=%.2f" % (enc_name, entropy_upper_ref)
            if entropy_upper_ref is not None else "ref ln(head)=? (registry unavail)")
    he.add_row("policy_entropy_sp",
               Text("%s   [floor %.1f warn<%.1f | %s | high=healthy]" % (
                   fmt(pe, 3), POLICY_ENTROPY_COLLAPSE_FLOOR,
                   POLICY_ENTROPY_WARN_FLOOR, _ref),
                   style=cls("policy_entropy_selfplay", pe)))
    xw, ow = ls.get("x_winrate"), ls.get("o_winrate")
    he.add_row("x/o winrate", Text("%s / %s" % (fmt(xw, 3), fmt(ow, 3)), style=cls("xo_winrate", xw)))
    dvals = [d for _, d in depth_series]
    cvals = [c for _, c in conc_series]
    he.add_row("mcts depth / root_conc",
               Text("%s  d=%s  conc=%s" % (
                   sparkline(dvals) if dvals else "(no data)",
                   fmt(ls.get("mcts_mean_depth"), 2), fmt(ls.get("mcts_root_concentration"), 2)),
                   style=(dh["style"] if dh else cls("mcts_mean_depth", ls.get("mcts_mean_depth")))))
    if dh:
        he.add_row("  depth read", Text("%s  %s" % (dh["label"], dh["detail"]), style=dh["style"]))
    elif dvals:
        he.add_row("  depth read", Text("warming up (need >=3 bins for a baseline)", style="dim"))
    he.add_row("gpu / vram", "%s%% | %s GB" % (fmt(ls.get("gpu_util"), 0), fmt(ls.get("vram_gb"), 1)))
    svals = [g for _, g in sigma_series]
    he.add_row("avg_sigma (v_spread)",
               "%s   %s" % (sparkline(svals) if svals else "(no data)",
                            fmt(ls.get("avg_sigma"), 3)))
    # value-spread canary t3/alt — HIGH = healthy (t3 anchor +0.617; abort <0.20)
    vc = data.get("vspread_canary_last") or {}
    t3vals = [t for _, t in t3_series]
    if vc or t3vals:
        cur_t3 = vc.get("t3_spread")
        he.add_row("t3/alt v_spread",
                   Text("%s   t3=%s alt=%s" % (
                       sparkline(t3vals) if t3vals else "(no data)",
                       fmt(cur_t3, 3), fmt(vc.get("alt_spread"), 3)),
                       style=cls("t3_spread", cur_t3)))
    vs = data.get("vspread_last")
    if vs:
        he.add_row("v_spread ALERT", Text("t3=%s alt=%s" % (
            fmt(vs.get("t3_spread"), 3), fmt(vs.get("alt_spread"), 3)),
            style="yellow" if not vs.get("both_pass") else "green"))
    panels.append(Panel(he, title="health", border_style="green"))

    # ---- charts ----
    if use_charts:
        ch_strs = []
        if fwt:
            xs = [d["step"] for d in fwt]
            s = chart([("conv", xs, [d["forced_win_conversion"] for d in fwt], "yellow+"),
                       ("off_win", xs, [d["off_window_forced_win_rate"] for d in fwt], "cyan")],
                      "forced-win conversion + off_window vs step")
            if s:
                ch_strs.append(s)
        if gap_series:
            s = chart([("gap", [x for x, _ in gap_series], [g for _, g in gap_series], "blue+")],
                      "value-bce gap (selfplay - corpus) vs step")
            if s:
                ch_strs.append(s)
        if col_series:
            s = chart([("colony", [x for x, _ in col_series], [c for _, c in col_series], "red")],
                      "colony_extension_fraction vs step")
            if s:
                ch_strs.append(s)
        if rounds:
            xs = [r["step"] for r in rounds]
            s = chart([("wr_best", xs, [r.get("wr_best") for r in rounds], "magenta"),
                       ("wr_sealbot", xs, [r.get("wr_sealbot") for r in rounds], "green+")],
                      "wr_best + wr_sealbot vs step")
            if s:
                ch_strs.append(s)
        # mcts mean depth with the run's OWN baseline (green) + the -12% floor (red)
        if depth_series:
            hl = []
            if dh:
                hl = [(dh["base"], "green"), (dh["floor"], "red")]
            s = chart([("depth", [x for x, _ in depth_series],
                        [d for _, d in depth_series], "cyan+")],
                      "mcts_mean_depth vs step (line=run baseline; red=-12%% floor)",
                      hlines=hl)
            if s:
                ch_strs.append(s)
        if conc_series:
            s = chart([("root_conc", [x for x, _ in conc_series],
                        [c for _, c in conc_series], "magenta")],
                      "mcts_root_concentration vs step (co-read with depth)")
            if s:
                ch_strs.append(s)
        if sigma_series:
            s = chart([("sigma", [x for x, _ in sigma_series],
                        [g for _, g in sigma_series], "yellow+")],
                      "v_spread (avg_sigma) vs step")
            if s:
                ch_strs.append(s)
        if t3_series:
            vsser = [("t3", [x for x, _ in t3_series], [g for _, g in t3_series], "green+")]
            if alt_series:
                vsser.append(("alt", [x for x, _ in alt_series],
                              [g for _, g in alt_series], "magenta"))
            s = chart(vsser, "t3/alt v_spread vs step (HIGH=healthy; t3 abort <0.20)")
            if s:
                ch_strs.append(s)
        if ch_strs:
            cols = Columns([Text.from_ansi(s) for s in ch_strs], equal=False, expand=False)
            danger = Table.grid(padding=(0, 1))
            danger.add_column(style="bold", justify="right", no_wrap=True)
            danger.add_column()
            _rc_legend = ("[dim]rising = expected under Gumbel-SH (descriptive, not an alarm)[/]"
                          if is_gumbel else
                          "[yellow]up while depth down = overconfident shallow cutoffs (PUCT)[/]")
            for k, v in (
                ("conversion", "[yellow]down >20% from baseline + colony up = golong coherence break[/]"),
                ("value-bce gap", "[yellow]WIDENING / selfplay floors ~0.69 = corpus-overfit[/]"),
                ("colony", "[red]up above 0.15 = colony-attractor[/]"),
                ("depth", "[yellow]sharp drop (>12%) vs run self-baseline = search regression; "
                          "shallow-by-design under Gumbel[/]"),
                ("root_conc", _rc_legend),
                ("t3 / alt v_spread", "[red]down toward 0 (t3<0.20, alt<0.07) = value head can't separate colony[/]"),
                ("avg_sigma", "[yellow]sustained excursion not self-correcting by 200-300k[/]"),
                ("wr_sealbot", "[green]read the SLOPE (robust Theil-Sen, measurement-error CI lower "
                               "bound >0, above effect-size floor); level descriptive only — "
                               "~0.55 a soft anchor (temp-0.5, n=100), NEVER a single-point gate[/]"),
            ):
                danger.add_row(k, Text.from_markup(v))
            cap = Text("danger -> look (read SLOPE, not single points):", style="italic dim")
            panels.append(Panel(Group(cols, Text(""), cap, danger),
                                title="trajectories (plotext)", border_style="white"))
        else:
            panels.append(Panel(Text("plotext unavailable — sparklines shown inline above",
                                     style="dim"), title="trajectories", border_style="white"))

    # ---- help overlay ----
    if show_help:
        ht = Table(show_edge=False, expand=True)
        ht.add_column("metric", style="bold cyan", no_wrap=True)
        ht.add_column("what it means / how to read (green healthy, yellow watch, red abort)")
        for k, v in HELP.items():
            ht.add_row(k, v)
        panels.append(Panel(ht, title="HELP / legend  (press ? to hide)", border_style="bright_white"))

    return Group(*panels)


# ----------------------------------------------------------------------------
# live loop (rich.Live) with key handling, and --once
# ----------------------------------------------------------------------------

def header(host, log, data):
    from rich.text import Text
    age = ""
    tip = (data or {}).get("live_tip") or {}
    # PREFER the live-tip age (computed vast-clock-relative -> immune to the remote
    # clock skew that inflates a laptop-now minus vast-ts age). Fall back to the
    # skew-prone laptop-vs-log-ts only when the tip probe failed.
    if tip.get("age_sec") is not None:
        age = "  (last event %.0fs ago)" % tip["age_sec"]
    else:
        ls = (data or {}).get("last_summary") or {}
        t = ls.get("timestamp")
        if t:
            try:
                dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
                secs = (datetime.now(dt.tzinfo) - dt).total_seconds()
                age = "  (last event %.0fs ago, laptop-clock)" % secs
            except Exception:
                pass
    # Show which restart segment is live (basename) so a resume that rotates the
    # log is visible at a glance.
    live = ""
    lf = tip.get("live_file")
    if lf:
        import os as _os
        live = "  live=%s" % _os.path.basename(lf)
    return Text("D-1M monitor  %s:%s%s%s   %s" % (
        host, log, live, age, datetime.now().strftime("%H:%M:%S")), style="bold")


def run_live(args, cfg, host, repo, log, default_encoding):
    from rich.console import Console, Group
    from rich.live import Live

    console = Console()
    show_help = [False]
    stop = [False]

    # non-blocking key reader (toggle help '?', quit 'q')
    import select
    import termios
    import tty
    fd = sys.stdin.fileno()
    isatty = sys.stdin.isatty()
    old = termios.tcgetattr(fd) if isatty else None
    if isatty:
        tty.setcbreak(fd)

    def poll_keys():
        dirty = False
        if not isatty:
            return dirty
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == "?":
                show_help[0] = not show_help[0]
                dirty = True
            elif ch in ("q", "Q", "\x03"):
                stop[0] = True
        return dirty

    def render(live, data):
        body = build(data, show_help[0], not args.no_charts, cfg, default_encoding)
        foot = "[dim]?: help   q: quit   refresh %ds[/]" % args.interval
        live.update(Group(header(host, log, data), body, foot))
        live.refresh()

    try:
        with Live(console=console, screen=True, auto_refresh=False) as live:
            data = fetch(cfg, host, repo, log)
            render(live, data)
            while not stop[0]:
                t0 = time.time()
                while time.time() - t0 < args.interval and not stop[0]:
                    if poll_keys():          # key toggled help/quit → reflect it NOW
                        render(live, data)
                    if stop[0]:
                        break
                    time.sleep(0.08)
                if stop[0]:
                    break
                data = fetch(cfg, host, repo, log)
                render(live, data)
    finally:
        if isatty and old is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def run_once(args, cfg, host, repo, log, default_encoding):
    from rich.console import Console, Group
    data = fetch(cfg, host, repo, log)
    # plain, no-TTY-safe output; force_terminal off when not a tty
    console = Console(force_terminal=None, width=None if sys.stdout.isatty() else 120)
    body = build(data, args.help_overlay, not args.no_charts, cfg, default_encoding)
    console.print(Group(header(host, log, data), body))
    return 0 if "error" not in data else 1


def main():
    ap = argparse.ArgumentParser(description="rich TUI monitor for the D-1M run on vast")
    ap.add_argument("--once", action="store_true", help="render one plain frame and exit")
    ap.add_argument("--interval", type=int, default=15, help="live refresh seconds (default 15)")
    ap.add_argument("--no-charts", action="store_true", help="skip plotext charts (sparklines only)")
    ap.add_argument("--help-overlay", action="store_true", help="(--once) include help/legend panel")
    args = ap.parse_args()

    # run identity + parse knobs from configs/monitoring.yaml run_feed:
    # (HARD-ERROR on missing); env D1M_* overrides win at the call site.
    cfg = load_run_feed_config()
    host, repo, log, default_encoding = resolve_run_target(cfg)

    if args.once:
        sys.exit(run_once(args, cfg, host, repo, log, default_encoding))
    try:
        run_live(args, cfg, host, repo, log, default_encoding)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
