#!/usr/bin/env python3
"""d1m_monitor.py — rich TUI for the D-1M Gumbel 1M run on vast.

Live, color-coded, graphed monitor. Pulls SPARSE decision-relevant events from
the remote JSONL over ssh (remote grep+sample — never transfers the 150MB file).

Companion to scripts/d1m_status.sh (quick CLI). This is the visual/interactive
upgrade: grouped panels + in-terminal line charts + toggleable help overlay.

Usage:
  .venv/bin/python scripts/d1m_monitor.py            # live TUI, ~15s refresh
  .venv/bin/python scripts/d1m_monitor.py --once     # one plain frame, exit
  .venv/bin/python scripts/d1m_monitor.py --interval 30
  .venv/bin/python scripts/d1m_monitor.py --no-charts # skip plotext, sparklines

In live mode press '?' to toggle the help/legend overlay, 'q' to quit.

Env overrides (match d1m_status.sh):
  D1M_HOST (default vast)  D1M_REPO (default /workspace/hexo_rl)
  D1M_LOG  (default logs/d1m/d1m_gumbel_m16_n150.jsonl, relative to repo)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime

# ----------------------------------------------------------------------------
# Tunable config — NAMED, documented, env-overridable. No magic numbers buried
# inline. These gate the wr_sealbot slope read (F2) and the entropy collapse
# floor (F3); the depth-health fractions live with depth_health() further down.
# ----------------------------------------------------------------------------
def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# wr_sealbot is read as a SLOPE over its eval history, never an absolute bar.
# Need >= K points before any slope verdict; with < K render "insufficient" /
# neutral, NEVER green (false-green guard).
SEALBOT_SLOPE_MIN_POINTS = _env_int("D1M_SEALBOT_SLOPE_MIN_POINTS", 5)   # K
# F2 rework: the slope verdict must NEVER false-green a noisy plateau. GREEN
# requires ALL of: robust (Theil-Sen) slope > 0, a measurement-error-aware CI
# whose LOWER bound > 0 (small-sample t critical value, df=n-2, propagating each
# point's LOGGED ci_sealbot half-width as that point's sigma), AND an effect size
# (slope x span) over the named floor. The old z=1.96 normal-CI was the defect
# (too narrow at n=5-8 + blind to per-point sampling noise) -> deleted.
SEALBOT_SLOPE_CI_LEVEL = _env_float("D1M_SEALBOT_SLOPE_CI_LEVEL", 0.95)  # two-sided CI level
SEALBOT_SLOPE_BOOT_ITERS = _env_int("D1M_SEALBOT_SLOPE_BOOT_ITERS", 2000)  # bootstrap reps
# minimum implied rise over the observed window (slope x span). A statistically
# significant-but-trivial drift is NOT a real climb -> not green. wr_sealbot is a
# win-rate in [0,1]; require at least this much net rise across the eval history.
SEALBOT_MIN_RISE = _env_float("D1M_SEALBOT_MIN_RISE", 0.03)              # +3pp over window
# Conservative per-point sigma fallback when a point lacks a logged ci_sealbot
# AND no game count to derive a Wilson SE: widen to this half-width (sealbot eval
# is ~100 games, temp-0.5, Wilson95 ~+/-0.10 -> a deliberately CONSERVATIVE 0.12).
SEALBOT_FALLBACK_SIGMA = _env_float("D1M_SEALBOT_FALLBACK_SIGMA", 0.12)
# CI half-width -> 1-sigma divisor: a logged ci_sealbot is a 95% interval, so its
# half-width ~= 1.96 sigma. Mathematical constant (normal 97.5th pct), not a knob.
_CI95_HALFWIDTH_TO_SIGMA = 1.959963985

# Student-t two-sided 0.975 critical values by df (1..30); df>30 -> normal z.
# Mathematical constants (not tunable knobs) — used for the small-sample slope CI
# so n=5 (df=3, t=3.182) is NOT treated as n=inf (z=1.960) the way the old code did.
_T_CRIT_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
    22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}


def _t_crit(df, level=None):
    """Two-sided t critical value at the given CI level for the given df. Table
    is for level=0.95 (the default); other levels scale off the normal tail as a
    conservative approximation. df<1 -> the df=1 (widest) value; df>30 -> normal."""
    if level is None:
        level = SEALBOT_SLOPE_CI_LEVEL
    if df < 1:
        df = 1
    if level == 0.95 or abs(level - 0.95) < 1e-9:
        return _T_CRIT_975.get(int(df), 1.959963985)
    # non-default level: approximate via normal z scaled by the table's t/z ratio
    base_t = _T_CRIT_975.get(int(df), 1.959963985)
    z_level = _norm_ppf(0.5 + level / 2.0)
    return base_t / 1.959963985 * z_level


def _norm_ppf(p):
    """Inverse normal CDF (Acklam approximation) — for non-default CI levels and
    Wilson SE. Pure-stdlib; avoids a scipy/numpy dependency in the monitor."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

# policy entropy: LOW-FLOOR-ONLY collapse detector. High entropy is HEALTHY
# (upper reference = ln(policy_logit_count), derived at runtime — never typed).
# F3: the 1.0/1.5 floors below ARE the RETAINED same-regime selfplay-collapse
# anchors — the legitimately-transferred LOW floor from the old 1.0/2.0 band.
# Only the CROSS-regime 2.0 UPPER bar was dropped (high entropy is healthy here,
# bounded by the derived ln(head) ceiling, so no fixed upper bar applies).
POLICY_ENTROPY_COLLAPSE_FLOOR = _env_float("D1M_POLICY_ENTROPY_COLLAPSE_FLOOR", 1.0)
POLICY_ENTROPY_WARN_FLOOR = _env_float("D1M_POLICY_ENTROPY_WARN_FLOOR", 1.5)

# fp16 AMP-scale canary: alert on a SHARP self-baseline DROP (loss-scaler
# backoff burst), not an absolute level. Fraction of trailing-median.
FP16_SCALE_DROP_FRAC = _env_float("D1M_FP16_SCALE_DROP_FRAC", 0.5)

# Run encoding name (drives the derived entropy upper reference). Read from the
# startup config event when present; this is the fallback default for this run.
DEFAULT_ENCODING = os.environ.get("D1M_ENCODING", "v6_live2_ls")


def _policy_logit_count(encoding_name):
    """Runtime lookup of the run's policy-head size. DERIVED — never hardcode
    362 or ln(362)=5.89. Falls back to None if the registry is unreachable.

    When invoked as `python scripts/d1m_monitor.py`, sys.path[0] is scripts/ and
    the repo root (where the hexo_rl package lives) is NOT importable — so add it
    before importing. hexo_rl is not pip-installed in this venv."""
    try:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _root not in sys.path:
            sys.path.insert(0, _root)
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
# Remote collector. Runs ON vast via ssh; emits ONE compact JSON blob to stdout.
# All grep / sampling / binning happens remotely so we transfer KB not MB.
# Mirrors the parsing in scripts/d1m_status.sh (ts->step map, 30k bins, sampled
# train_step value-bce, sampled game_complete colony).
# ----------------------------------------------------------------------------
REMOTE = r'''
import json, sys, bisect
from datetime import datetime

log = sys.argv[1]
BIN = 30000          # step-bucket width for binned trajectories
TS_SAMPLE = 200      # sample every Nth train_step (value-bce trajectory)
GC_SAMPLE = 50       # sample every Nth game_complete (colony trajectory)

def ts(d):
    s = d["timestamp"] if isinstance(d, dict) else d
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

try:
    raw = open(log).read().splitlines()
except FileNotFoundError:
    print(json.dumps({"error": "log not found: " + log})); sys.exit(0)

summ, rounds, fwt, vspread, vsc = [], [], [], [], []
total_steps = None
encoding_name = None
variant_name = None
is_gumbel = None
last_train = None
tsi = gci = 0
bce = {}   # bucket -> [sum_sp, sum_co, n]
col = {}   # bucket -> [sum_colony, n]

for ln in raw:
    s = ln.strip()
    if not s.startswith("{"):
        continue
    try:
        if '"train_step_summary"' in s:
            summ.append(json.loads(s))
        elif '"evaluation_round_complete"' in s:
            rounds.append(json.loads(s))
        elif '"forced_win_trend"' in s:
            fwt.append(json.loads(s))
        elif '"value_spread_alert"' in s:
            vspread.append(json.loads(s))
        elif '"event": "value_spread"' in s:   # always-emitted canary (has step + t3/alt)
            vsc.append(json.loads(s))
        elif '"startup"' in s:
            _su = json.loads(s)
            _cfg = (_su.get("config") or {})
            total_steps = _cfg.get("total_steps", total_steps)
            encoding_name = _cfg.get("encoding", encoding_name)
            variant_name = _su.get("variant", variant_name)
            _spc = (_cfg.get("selfplay") or {})
            if _spc.get("gumbel_mcts") is not None:
                is_gumbel = bool(_spc.get("gumbel_mcts"))
        elif '"event": "train_step"' in s:
            tsi += 1
            if tsi % TS_SAMPLE == 0:
                d = json.loads(s)
                b = d.get("step", 0) // BIN
                sp, co = d.get("value_bce_selfplay"), d.get("value_bce_corpus")
                if sp is not None and co is not None:
                    e = bce.setdefault(b, [0.0, 0.0, 0]); e[0] += sp; e[1] += co; e[2] += 1
    except Exception:
        pass

# latest train_step (high-freq fields: grad_norm, lr, value_bce, value_acc)
for ln in reversed(raw):
    s = ln.strip()
    if s.startswith("{") and '"event": "train_step"' in s:
        try:
            last_train = json.loads(s); break
        except Exception:
            pass

# ts -> step map (summaries) for events lacking a step field
stimes = [ts(s) for s in summ]
ssteps = [s["step"] for s in summ]

def step_at(t):
    if not ssteps:
        return 0
    i = bisect.bisect_left(stimes, t)
    return ssteps[min(max(i, 0), len(ssteps) - 1)]

# colony by bucket (game_complete: has ts, no step)
for ln in raw:
    s = ln.strip()
    if '"event": "game_complete"' not in s:
        continue
    gci += 1
    if gci % GC_SAMPLE:
        continue
    try:
        d = json.loads(s)
        c = d.get("colony_extension_fraction")
        if c is None:
            continue
        b = step_at(ts(d)) // BIN
        e = col.setdefault(b, [0.0, 0]); e[0] += c; e[1] += 1
    except Exception:
        pass

# recent colony (last ~500 sampled game_complete, reverse scan)
colony_recent = []
for ln in reversed(raw):
    s = ln.strip()
    if '"event": "game_complete"' in s:
        try:
            c = json.loads(s).get("colony_extension_fraction")
            if c is not None:
                colony_recent.append(c)
        except Exception:
            pass
        if len(colony_recent) >= 500:
            break
colony_recent_mean = (sum(colony_recent) / len(colony_recent)) if colony_recent else None

# map each forced_win_trend ts -> step (no step field in event)
fwt_out = []
for d in fwt:
    fwt_out.append({
        "step": step_at(ts(d)),
        "forced_win_conversion": d.get("forced_win_conversion"),
        "off_window_forced_win_rate": d.get("off_window_forced_win_rate"),
        "n": d.get("n"),
    })

# binned mcts mean-depth + root-concentration + value-spread (avg_sigma) from
# train_step_summary (summaries carry a step field directly — bin like bce).
# depth+concentration travel TOGETHER: depth-down WITH concentration-up =
# overconfident shallow cutoffs; depth-down with concentration-flat = breadth
# unchanged (game-length/structural). Bin both so the monitor can co-read them.
dsg = {}   # bucket -> [sum_depth, n_depth, sum_sigma, n_sigma, sum_conc, n_conc]
for sd in summ:
    b = sd.get("step", 0) // BIN
    e = dsg.setdefault(b, [0.0, 0, 0.0, 0, 0.0, 0])
    md = sd.get("mcts_mean_depth")
    if isinstance(md, (int, float)):
        e[0] += md; e[1] += 1
    sg = sd.get("avg_sigma")
    if isinstance(sg, (int, float)):
        e[2] += sg; e[3] += 1
    rc = sd.get("mcts_root_concentration")
    if isinstance(rc, (int, float)):
        e[4] += rc; e[5] += 1

# binned t3 + alt value-spread canary (value_spread event carries step directly;
# alt_spread can be NaN on a plane-mismatch run — guarded out)
vbk = {}   # bucket -> [sum_t3, n_t3, sum_alt, n_alt]
for vd in vsc:
    b = vd.get("step", 0) // BIN
    e = vbk.setdefault(b, [0.0, 0, 0.0, 0])
    t3v = vd.get("t3_spread")
    if isinstance(t3v, (int, float)) and t3v == t3v:
        e[0] += t3v; e[1] += 1
    alv = vd.get("alt_spread")
    if isinstance(alv, (int, float)) and alv == alv:
        e[2] += alv; e[3] += 1

# binned value-bce + colony + depth/spread + t3/alt trajectory (sorted buckets)
traj = []
for b in sorted(set(list(bce) + list(col) + list(dsg) + list(vbk))):
    row = {"step": b * BIN}
    if b in bce and bce[b][2]:
        row["bce_sp"] = bce[b][0] / bce[b][2]
        row["bce_co"] = bce[b][1] / bce[b][2]
    if b in col and col[b][1]:
        row["colony"] = col[b][0] / col[b][1]
    if b in dsg and dsg[b][1]:
        row["depth"] = dsg[b][0] / dsg[b][1]
    if b in dsg and dsg[b][3]:
        row["sigma"] = dsg[b][2] / dsg[b][3]
    if b in dsg and dsg[b][5]:
        row["conc"] = dsg[b][4] / dsg[b][5]
    if b in vbk and vbk[b][1]:
        row["t3"] = vbk[b][0] / vbk[b][1]
    if b in vbk and vbk[b][3]:
        row["alt"] = vbk[b][2] / vbk[b][3]
    traj.append(row)

# rate calc helper (steps/hr between two summaries)
def rate(a, b):
    dt = (ts(b) - ts(a)).total_seconds()
    return (b["step"] - a["step"]) / dt * 3600 if dt > 0 else 0.0

# Effective rate = sum(steps)/sum(time) over CLEAN intervals: skip restart
# stop-gaps (dt>30min) and resume step-resets (ds<0). Eval-INCLUSIVE (slow
# eval-overlap intervals are summed in at their real rate) and resume-robust
# -> this is the honest ETA rate, not the swingy recent-1h pace.
eff_steps = eff_time = rec_steps = rec_time = all_steps = all_time = 0.0
overall = recent = effective = 0.0
if len(summ) >= 2:
    _cut = ts(summ[-1]).timestamp() - 3600              # 1h -> "recent" (swingy)
    _cut_eff = ts(summ[-1]).timestamp() - 12 * 3600     # 12h (>=~1 eval cycle) -> ETA rate:
    for _i in range(len(summ) - 1):                     # eval-INCLUSIVE yet reflects the CURRENT
        _a, _b = summ[_i], summ[_i + 1]                 # regime (converges after a config change).
        _dt = (ts(_b) - ts(_a)).total_seconds()
        _ds = _b["step"] - _a["step"]
        if _dt <= 0 or _dt > 1800 or _ds < 0:           # skip restart gaps + resume step-resets
            continue
        all_steps += _ds; all_time += _dt
        if ts(_a).timestamp() >= _cut_eff:
            eff_steps += _ds; eff_time += _dt
        if ts(_a).timestamp() >= _cut:
            rec_steps += _ds; rec_time += _dt
    overall = all_steps / all_time * 3600 if all_time else 0.0
    effective = eff_steps / eff_time * 3600 if eff_time else overall
    recent = rec_steps / rec_time * 3600 if rec_time else effective

# F6: Gumbel-target / opening-diversity panel — pull from the LATEST summary.
# These are the dirichlet-off full-search head's target health (entropy + KL-to-
# uniform) plus opening diversity (early-game entropy + top1 mass = effective-n).
# Also build a short trailing trajectory so the local side can show a self-baseline.
GT_KEYS = ("policy_target_entropy_fullsearch", "policy_target_kl_uniform_fullsearch",
           "early_game_entropy_mean", "early_game_top1_mass_mean")
gumbel_targets = {}
ls_summ = summ[-1] if summ else {}
for k in GT_KEYS:
    v = ls_summ.get(k)
    if isinstance(v, (int, float)):
        gumbel_targets[k] = v
# trailing window (last ~30 summaries) for self-baseline sparklines
gt_recent = {k: [] for k in GT_KEYS}
for sd in summ[-30:]:
    for k in GT_KEYS:
        v = sd.get(k)
        if isinstance(v, (int, float)):
            gt_recent[k].append(v)

# F6: fp16 AMP-scale canary — latest train_step value + a trailing window for a
# SHARP-drop self-baseline (no absolute level matters).
fp16_recent = []
_fc = 0
for ln in reversed(raw):
    s = ln.strip()
    if s.startswith("{") and '"event": "train_step"' in s:
        try:
            v = json.loads(s).get("fp16_scale")
            if isinstance(v, (int, float)):
                fp16_recent.append(v)
        except Exception:
            pass
        _fc += 1
        if _fc >= 4000:
            break
fp16_recent.reverse()
fp16_last = fp16_recent[-1] if fp16_recent else None

# F6: value-head fields per eval round — value_fc2_weight_abs_max (numeric) +
# g4_value_head_band_pass (bool). Attached to each eval round below for render.
for d in rounds:
    d["value_fc2_weight_abs_max"] = d.get("value_fc2_weight_abs_max")
    d["g4_value_head_band_pass"] = d.get("g4_value_head_band_pass")

# F7: eval-overhead from TIMESTAMP gaps (approx). For each eval, bound the eval
# window by [last train_step_summary ts with step < eval step] -> [eval ts].
# Sum of those gaps / total run wall = overhead fraction. Both event types carry
# an ISO timestamp. Labelled approx because training continues async during eval.
eval_overhead_frac = None
if len(summ) >= 2 and rounds:
    _run_wall = (ts(summ[-1]) - ts(summ[0])).total_seconds()
    _ssorted = [(ts(s), s.get("step")) for s in summ]
    _ovh = 0.0
    for r in rounds:
        es = r.get("step")
        et = ts(r) if r.get("timestamp") else None
        if es is None or et is None:
            continue
        _cands = [t for (t, st) in _ssorted if st is not None and st < es and t <= et]
        if not _cands:
            continue
        _gap = (et - max(_cands)).total_seconds()
        if _gap > 0:
            _ovh += _gap
    if _run_wall > 0:
        eval_overhead_frac = _ovh / _run_wall

out = {
    "total_steps": total_steps,
    "encoding": encoding_name,
    "variant": variant_name,
    "is_gumbel": is_gumbel,
    "last_summary": summ[-1] if summ else None,
    "last_train": last_train,
    "rate_overall": overall,
    "rate_effective": effective,
    "rate_recent": recent,
    "eval_rounds": rounds,
    "fwt": fwt_out,
    "vspread_last": vspread[-1] if vspread else None,
    "vspread_canary_last": vsc[-1] if vsc else None,
    "colony_recent_mean": colony_recent_mean,
    "colony_recent_n": len(colony_recent),
    "traj": traj,
    "n_summaries": len(summ),
    "gumbel_targets": gumbel_targets,
    "gumbel_targets_recent": gt_recent,
    "fp16_scale_last": fp16_last,
    "fp16_scale_recent": fp16_recent[-200:],
    "eval_overhead_frac": eval_overhead_frac,
}
print(json.dumps(out))
'''


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
            # F3: LOW-FLOOR ONLY. High entropy (up to ln(head)) is healthy, never
            # flagged. Only a drop toward the named collapse floor warns/aborts.
            if v < POLICY_ENTROPY_COLLAPSE_FLOOR:
                return "red"
            if v < POLICY_ENTROPY_WARN_FLOOR:
                return "yellow"
            return "green"
        if metric == "mcts_mean_depth":
            # F5: NO absolute depth floor. Depth is shallow-by-design under
            # Gumbel; the verdict comes from depth_health()'s run-relative
            # baseline. Before a baseline exists (warming up) stay neutral.
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
            # F2: absolute level is DESCRIPTIVE only — never color-gated. The
            # green/yellow read is SLOPE-based (sealbot_slope_style), computed
            # over the eval history in build(). Here return neutral.
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
# F2: wr_sealbot SLOPE read. The absolute level (~0.27 now, ~0.18 at fair
# temp-0.5) is DESCRIPTIVE only. GREEN must mean a REAL sustained climb, NEVER a
# noisy plateau. The old read (OLS slope, normal z=1.96 CI, no per-point sampling
# noise, no outlier robustness, no effect-size floor) false-greened trendless
# plateaus at the small-n regime this run lives in (5-8 points) and flipped a
# flat plateau to GREEN when a single lucky spike landed on the last eval.
#
# Rework — GREEN requires ALL of:
#   1. n >= K (false-green guard; < K -> "insufficient data", never green).
#   2. Robust slope > 0: Theil-Sen (median of pairwise slopes) — one spike can't
#      swing the median the way it swings an OLS fit.
#   3. Lower CI bound > 0, where the CI (a) uses a small-sample t critical value
#      (df=n-2) AND (b) PROPAGATES each point's LOGGED ci_sealbot measurement
#      error. Computed two ways and the MORE CONSERVATIVE (higher) lower bound is
#      taken: a measurement-error bootstrap (resample pairs for Theil-Sen +
#      jitter each y within its logged sigma) percentile CI, and an analytic
#      t-interval on the Theil-Sen slope whose SE folds in the per-point sigma.
#   4. Effect size: implied rise over the observed window (slope x span) >=
#      SEALBOT_MIN_RISE — a significant-but-trivial drift is NOT a climb.
# Anything failing -> neutral / "flat or inconclusive" (non-green).
# ----------------------------------------------------------------------------
def _sealbot_point_sigmas(rounds):
    """Per-point 1-sigma measurement error for each wr_sealbot eval point, in the
    SAME order/filtering as sealbot_slope's (step, wr) pairs. Reads the LOGGED
    ci_sealbot half-width (a ~95% interval -> /1.96 = sigma). Fallback chain when
    a point lacks ci_sealbot: Wilson SE from a logged sealbot game count if
    present, else the conservative SEALBOT_FALLBACK_SIGMA. Returns a list aligned
    to [(step, wr) for rounds with numeric step+wr_sealbot]."""
    sigmas = []
    for r in rounds:
        x, y = r.get("step"), r.get("wr_sealbot")
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            continue
        ci = r.get("ci_sealbot")
        half = None
        if isinstance(ci, (list, tuple)) and len(ci) == 2 \
                and all(isinstance(c, (int, float)) for c in ci):
            half = abs(ci[1] - ci[0]) / 2.0
        elif isinstance(ci, (int, float)):
            half = abs(ci)
        if half is not None and half > 0:
            sigmas.append(half / _CI95_HALFWIDTH_TO_SIGMA)
            continue
        # fallback: Wilson SE from a logged game count, if any field carries it
        ng = None
        for k in ("sealbot_n", "n_sealbot", "sealbot_games", "n_games_sealbot"):
            v = r.get(k)
            if isinstance(v, (int, float)) and v > 0:
                ng = float(v)
                break
        if ng:
            p = min(max(float(y), 0.0), 1.0)
            sigmas.append(max(math.sqrt(p * (1 - p) / ng), 1.0 / (2.0 * ng)))
        else:
            sigmas.append(SEALBOT_FALLBACK_SIGMA)  # conservative widen
    return sigmas


def _theil_sen(pts):
    """Median of pairwise slopes (x in steps). None if no positive-dx pair."""
    slopes = [(y2 - y1) / (x2 - x1)
              for i, (x1, y1) in enumerate(pts)
              for (x2, y2) in pts[i + 1:] if x2 != x1]
    if not slopes:
        return None
    s = sorted(slopes)
    m = len(s)
    return s[m // 2] if m % 2 else (s[m // 2 - 1] + s[m // 2]) / 2.0


def sealbot_slope(xs, ys, sigmas=None, min_pts=None,
                  ci_level=None, min_rise=None, boot_iters=None, rng_seed=None):
    """Return {label, style, slope, n} from the wr_sealbot eval history.
    style is "" (neutral) or "green"/"yellow" — green ONLY on a robust, CI-clear,
    above-floor climb (see block comment). `sigmas` = per-point 1-sigma logged
    measurement error (from _sealbot_point_sigmas); falls back to a conservative
    constant per point when absent so a missing CI can never narrow the interval."""
    if min_pts is None:
        min_pts = SEALBOT_SLOPE_MIN_POINTS
    if ci_level is None:
        ci_level = SEALBOT_SLOPE_CI_LEVEL
    if min_rise is None:
        min_rise = SEALBOT_MIN_RISE
    if boot_iters is None:
        boot_iters = SEALBOT_SLOPE_BOOT_ITERS
    triples = [(x, y, s) for x, y, s in
               zip(xs, ys, (sigmas if sigmas is not None else [None] * len(xs)))
               if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    n = len(triples)
    if n < min_pts:
        return {"label": "insufficient data (n=%d<%d)" % (n, min_pts),
                "style": "", "slope": None, "n": n}
    pts = [(x, y) for x, y, _ in triples]
    sig = [(s if isinstance(s, (int, float)) and s > 0 else SEALBOT_FALLBACK_SIGMA)
           for _, _, s in triples]
    xspan = max(x for x, _ in pts) - min(x for x, _ in pts)
    if xspan <= 0:
        return {"label": "flat x (no step spread)", "style": "", "slope": None, "n": n}

    # 2. robust point estimate
    slope = _theil_sen(pts)
    if slope is None:
        return {"label": "flat x (no step spread)", "style": "", "slope": None, "n": n}
    per100k = slope * 100000.0
    rise = slope * xspan  # implied net change over the observed window

    # 3a. measurement-error bootstrap: resample pairs (with replacement) for the
    # Theil-Sen estimate AND jitter each y within its logged sigma each rep.
    rnd = random.Random(rng_seed if rng_seed is not None else 0xD1F2)
    boots = []
    idx = list(range(n))
    for _ in range(boot_iters):
        sample = []
        for _j in range(n):
            k = rnd.choice(idx)
            x_k, y_k = pts[k]
            sample.append((x_k, y_k + rnd.gauss(0.0, sig[k])))
        bs = _theil_sen(sample)
        if bs is not None:
            boots.append(bs)
    if boots:
        boots.sort()
        a = (1.0 - ci_level) / 2.0
        lo_b = boots[min(len(boots) - 1, max(0, int(a * len(boots))))]
    else:
        lo_b = -math.inf

    # 3b. analytic t-interval on the Theil-Sen slope. SE folds in BOTH the
    # residual scatter about the robust fit AND the per-point logged sigma
    # (measurement error), divided through the x leverage (Sxx). t df = n-2.
    mx = sum(x for x, _ in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    intercept = (sum(y for _, y in pts) / n) - slope * mx
    resid = [y - (intercept + slope * x) for x, y in pts]
    sse = sum(r * r for r in resid)
    resid_var = sse / (n - 2) if n > 2 else float("inf")
    meas_var = sum(s * s for s in sig) / n  # mean per-point measurement variance
    se = math.sqrt((resid_var + meas_var) / sxx) if sxx > 0 else float("inf")
    tcrit = _t_crit(n - 2, ci_level)
    lo_t = slope - tcrit * se

    # take the MORE CONSERVATIVE (higher) lower bound of the two methods
    lo = max(lo_b, lo_t)

    # 4. effect-size gate + all conditions for green
    rise_ok = rise >= min_rise
    if slope > 0 and lo > 0 and rise_ok:
        return {"label": "RISING (+%.3f/100k, +%.1fpp over window, robust CI>0)"
                % (per100k, rise * 100),
                "style": "green", "slope": slope, "n": n}
    # descriptive non-green reasons (never green)
    if slope <= 0:
        return {"label": "flat/falling (%+.3f/100k, Theil-Sen)" % per100k,
                "style": "yellow" if slope < 0 else "", "slope": slope, "n": n}
    if lo <= 0:
        return {"label": "inconclusive (+%.3f/100k, CI straddles 0 @n=%d)"
                % (per100k, n), "style": "", "slope": slope, "n": n}
    return {"label": "trivial rise (+%.3f/100k, +%.1fpp < %.0fpp floor)"
            % (per100k, rise * 100, min_rise * 100),
            "style": "", "slope": slope, "n": n}


# ----------------------------------------------------------------------------
# fetch — single ssh round-trip, remote collector emits compact JSON
# ----------------------------------------------------------------------------

def fetch(host, repo, log):
    cmd = ["ssh", host, "cd '%s' && .venv/bin/python - '%s'" % (repo, log)]
    try:
        p = subprocess.run(cmd, input=REMOTE, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        return {"error": "ssh timeout (>90s)"}
    except Exception as e:  # noqa: BLE001
        return {"error": "ssh failed: %s" % e}
    if p.returncode != 0:
        msg = (p.stderr or "").strip().splitlines()
        msg = [m for m in msg if "vast.ai" not in m and "Have fun" not in m]
        return {"error": "ssh rc=%d: %s" % (p.returncode, " ".join(msg[-3:]) or "?")}
    # remote may print ssh banner lines before JSON — grab last JSON line
    for ln in reversed(p.stdout.splitlines()):
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                return json.loads(ln)
            except Exception:
                continue
    return {"error": "no JSON from remote (stdout=%r)" % p.stdout[:200]}


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


# ----------------------------------------------------------------------------
# depth-health discriminator. A raw depth number is ambiguous (Gumbel search is
# shallow BY DESIGN; any fixed floor is arbitrary). Instead judge depth vs the
# RUN'S OWN settled baseline, and use root_concentration co-movement to name the
# mechanism. All thresholds are fractions of the run baseline (tunable). The
# concentration read is DESCRIPTIVE under Gumbel-SH (where rising root
# concentration is the expected Sequential-Halving behavior, not a defect).
# ----------------------------------------------------------------------------
DEPTH_STABLE_FRAC = _env_float("D1M_DEPTH_STABLE_FRAC", 0.05)    # +/-5% of own baseline = stable
DEPTH_REGRESS_FRAC = _env_float("D1M_DEPTH_REGRESS_FRAC", 0.12)  # >12% below baseline = regression
CONC_MOVE = _env_float("D1M_CONC_MOVE", 0.03)                    # root_conc co-move threshold (abs)


def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else None


def depth_health(depth_series, conc_series, is_gumbel=None):
    """Run-relative depth verdict. Returns dict or None (insufficient data).

    baseline = median of history buckets (all but last 2); cur = mean of last 2.
    floor = baseline*(1-DEPTH_REGRESS_FRAC) — moves WITH the run, not a constant.
    Mechanism comes from root_concentration co-movement over the same window.
    F1: under Gumbel-SH the concentration read is DESCRIPTIVE (rising root_conc
    is expected halving behavior), not a PUCT 'overconfident shallow cutoffs'
    alarm. The depth-drop verdict itself is self-baseline-relative regardless."""
    ds = [d for _, d in depth_series]
    if len(ds) < 3:
        return None
    hist = ds[:-2] if len(ds) >= 4 else ds[:-1]
    base = _median(hist)
    cur = sum(ds[-2:]) / 2.0
    if not base:
        return None
    pct = (cur - base) / base
    floor = base * (1 - DEPTH_REGRESS_FRAC)

    conc_delta = None
    cs = [c for _, c in conc_series]
    if len(cs) >= 3:
        ch = cs[:-2] if len(cs) >= 4 else cs[:-1]
        conc_delta = (sum(cs[-2:]) / 2.0) - (_median(ch) or 0.0)

    def mech():
        if conc_delta is None:
            return ""
        if conc_delta > CONC_MOVE:
            if is_gumbel:
                # Gumbel-SH: rising root_conc is expected halving, not a defect.
                return " + root_conc rising (descriptive; expected under Gumbel-SH)"
            return " + root_conc rising -> overconfident shallow cutoffs (PUCT)"
        if conc_delta < -CONC_MOVE:
            return " + root_conc falling -> search broadening"
        return " + root_conc flat -> game-length/structural, not search-quality"

    if pct >= DEPTH_STABLE_FRAC:
        label, style, detail = "DEEPENING", "green", "+%.0f%% vs baseline %.2f" % (pct * 100, base)
    elif pct >= -DEPTH_STABLE_FRAC:
        label, style, detail = "stable", "green", "within +/-%.0f%% of baseline %.2f" % (
            DEPTH_STABLE_FRAC * 100, base)
    elif pct >= -DEPTH_REGRESS_FRAC:
        label, style, detail = "WATCH", "yellow", "down %.0f%% vs baseline %.2f%s" % (
            -pct * 100, base, mech())
    else:
        label, style, detail = "REGRESSION", "red", "down %.0f%% below floor %.2f (base %.2f)%s" % (
            -pct * 100, floor, base, mech())
    return {"base": base, "cur": cur, "pct": pct, "floor": floor,
            "conc_delta": conc_delta, "label": label, "style": style, "detail": detail}


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
# build the rich renderables
# ----------------------------------------------------------------------------

def build(data, show_help, use_charts):
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
    fwt = data.get("fwt") or []
    traj = data.get("traj") or []
    # F3: policy-entropy upper reference = ln(policy_logit_count), DERIVED at
    # runtime from the run's encoding (registry lookup) — never a literal.
    enc_name = data.get("encoding") or DEFAULT_ENCODING
    entropy_upper_ref = _entropy_upper_ref(enc_name)
    depth_series = [(r["step"], r["depth"]) for r in traj if "depth" in r]
    conc_series = [(r["step"], r["conc"]) for r in traj if "conc" in r]
    sigma_series = [(r["step"], r["sigma"]) for r in traj if "sigma" in r]
    t3_series = [(r["step"], r["t3"]) for r in traj if "t3" in r]
    alt_series = [(r["step"], r["alt"]) for r in traj if "alt" in r]
    is_gumbel = data.get("is_gumbel")
    dh = depth_health(depth_series, conc_series, is_gumbel)

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
    # F2: wr_sealbot read = SLOPE over eval history (level is descriptive only).
    # Pass each point's LOGGED ci_sealbot half-width as its measurement sigma so
    # the slope CI propagates per-point sampling noise, not just fitted residuals.
    sb_xs = [r.get("step") for r in rounds]
    sb_ys = [r.get("wr_sealbot") for r in rounds]
    sb_sigmas = _sealbot_point_sigmas(rounds)
    sb_slope = sealbot_slope(sb_xs, sb_ys, sigmas=sb_sigmas)
    st = Table(title=None, expand=True, show_edge=False)
    # F6: value-head fields per eval round -> vfc2 (numeric) + g4 (pass/fail bool)
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
                # F2: wsb cell DESCRIPTIVE (level shown), color from the SLOPE read
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
        body = Text("no eval rounds yet (first at step 30,000; now at %d)" % step,
                    style="dim")
    panels.append(Panel(body, title="strength (eval rounds)", border_style="magenta"))

    # ---- coherence cluster (HEADLINE) ----
    coh = Table.grid(padding=(0, 1))
    coh.add_column(style="bold", justify="right")
    coh.add_column()
    if fwt:
        conv = [d.get("forced_win_conversion") for d in fwt]
        offw = [d.get("off_window_forced_win_rate") for d in fwt]
        last_conv = conv[-1]
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

    # ---- F6: Gumbel-target health + opening diversity + AMP-stability ----
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

    # THE Gumbel-target health signal (dirichlet-off full-search head) — shown together
    _gt_row("target_entropy_fs", "policy_target_entropy_fullsearch")
    _gt_row("target_kl_uniform_fs", "policy_target_kl_uniform_fullsearch")
    gp.add_row("[dim]gumbel-target[/]",
               Text("full-search head: entropy + KL-to-uniform travel TOGETHER "
                    "(read self-baseline trend, no absolute target)", style="italic dim"))
    # opening diversity = §D-ARGMAX effective-n
    _gt_row("early_game_entropy", "early_game_entropy_mean")
    _gt_row("early_game_top1_mass", "early_game_top1_mass_mean")
    gp.add_row("[dim]opening-div[/]",
               Text("diversity = effective-n (§D-ARGMAX): low early entropy / high "
                    "top1 mass = collapsed openings, fewer DISTINCT games", style="italic dim"))
    # fp16 AMP-scale canary — alert on a SHARP self-baseline DROP, not absolute
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

    # ---- F7: log-derived overhead + owed signals ----
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
    # F3: policy entropy — LOW-FLOOR ONLY. Show the derived upper reference
    # ln(policy_logit_count) so the user reads the value against the healthy
    # ceiling (uniform head). High entropy is healthy; only collapse warns.
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
        # conversion + off_window over step
        if fwt:
            xs = [d["step"] for d in fwt]
            s = chart([("conv", xs, [d["forced_win_conversion"] for d in fwt], "yellow+"),
                       ("off_win", xs, [d["off_window_forced_win_rate"] for d in fwt], "cyan")],
                      "forced-win conversion + off_window vs step")
            if s:
                ch_strs.append(s)
        # value-bce gap over step
        if gap_series:
            s = chart([("gap", [x for x, _ in gap_series], [g for _, g in gap_series], "blue+")],
                      "value-bce gap (selfplay - corpus) vs step")
            if s:
                ch_strs.append(s)
        # colony over step
        if col_series:
            s = chart([("colony", [x for x, _ in col_series], [c for _, c in col_series], "red")],
                      "colony_extension_fraction vs step")
            if s:
                ch_strs.append(s)
        # wr_sealbot over step
        if rounds:
            xs = [r["step"] for r in rounds]
            s = chart([("wr_best", xs, [r.get("wr_best") for r in rounds], "magenta"),
                       ("wr_sealbot", xs, [r.get("wr_sealbot") for r in rounds], "green+")],
                      "wr_best + wr_sealbot vs step")
            if s:
                ch_strs.append(s)
        # mcts mean depth over step, with the run's OWN baseline (green) + the
        # -12% regression floor (red) as reference lines — not a fixed constant
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
        # root_concentration over step — co-read with depth (up+depth-down = cutoffs)
        if conc_series:
            s = chart([("root_conc", [x for x, _ in conc_series],
                        [c for _, c in conc_series], "magenta")],
                      "mcts_root_concentration vs step (co-read with depth)")
            if s:
                ch_strs.append(s)
        # value-spread (avg_sigma) over step
        if sigma_series:
            s = chart([("sigma", [x for x, _ in sigma_series],
                        [g for _, g in sigma_series], "yellow+")],
                      "v_spread (avg_sigma) vs step")
            if s:
                ch_strs.append(s)
        # t3 (+alt) value-spread canary over step — HIGH healthy, t3 abort <0.20
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
            # compact, centralized "what's dangerous" legend for the trajectories
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
    ls = (data or {}).get("last_summary") or {}
    t = ls.get("timestamp")
    if t:
        try:
            dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
            secs = (datetime.now(dt.tzinfo) - dt).total_seconds()
            age = "  (last event %.0fs ago)" % secs
        except Exception:
            pass
    return Text("D-1M monitor  %s:%s%s   %s" % (
        host, log, age, datetime.now().strftime("%H:%M:%S")), style="bold")


def run_live(args, host, repo, log):
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
        # returns True if a key changed UI state (so we re-render immediately)
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
        body = build(data, show_help[0], not args.no_charts)
        foot = "[dim]?: help   q: quit   refresh %ds[/]" % args.interval
        live.update(Group(header(host, log, data), body, foot))
        live.refresh()

    try:
        with Live(console=console, screen=True, auto_refresh=False) as live:
            data = fetch(host, repo, log)
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
                data = fetch(host, repo, log)
                render(live, data)
    finally:
        if isatty and old is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def run_once(args, host, repo, log):
    from rich.console import Console, Group
    data = fetch(host, repo, log)
    # plain, no-TTY-safe output; force_terminal off when not a tty
    console = Console(force_terminal=None, width=None if sys.stdout.isatty() else 120)
    body = build(data, args.help_overlay, not args.no_charts)
    console.print(Group(header(host, log, data), body))
    return 0 if "error" not in data else 1


def main():
    ap = argparse.ArgumentParser(description="rich TUI monitor for the D-1M run on vast")
    ap.add_argument("--once", action="store_true", help="render one plain frame and exit")
    ap.add_argument("--interval", type=int, default=15, help="live refresh seconds (default 15)")
    ap.add_argument("--no-charts", action="store_true", help="skip plotext charts (sparklines only)")
    ap.add_argument("--help-overlay", action="store_true", help="(--once) include help/legend panel")
    args = ap.parse_args()

    host = os.environ.get("D1M_HOST", "vast")
    repo = os.environ.get("D1M_REPO", "/workspace/hexo_rl")
    log = os.environ.get("D1M_LOG", "logs/d1m/d1m_gumbel_m16_n150.jsonl")

    if args.once:
        sys.exit(run_once(args, host, repo, log))
    try:
        run_live(args, host, repo, log)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
