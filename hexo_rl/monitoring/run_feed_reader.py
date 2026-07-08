"""run_feed_reader — the single READ-ONLY consumer of Feed A.

Feed A = the structlog run-log (`<run_name>.jsonl`) emitted by a live training
run. This module is the ONE place that parses it. Two layers:

  (1) parse_feed(records) -> RunFeedSnapshot
      PURE, transport-agnostic. No ssh, no file IO. Takes any iterable of dicts
      (already-decoded JSONL records) and reproduces the exact decision-relevant
      reduction that used to live inline in scripts/d1m_monitor.py's REMOTE
      collector string + its local verdict functions: ts->step bisect map, 30k
      step bins (value-bce / colony / depth+sigma+conc / t3+alt), recent-colony
      mean, gap-skipping effective rate, the sealbot_slope measurement-error CI,
      and the run-relative depth_health verdict. The CI math is moved VERBATIM —
      it is load-bearing (false-green guard at the n=5-8 regime this run lives in).

  (2) read_remote_ssh / read_local_jsonl
      thin IO helpers that yield decoded records for parse_feed. The remote helper
      streams the run-log over ssh (NEVER transfers the whole file — remote-side
      grep keeps only decision-relevant event types); the local helper reads a
      JSONL file off disk. Both feed the SAME pure parse_feed.

scripts/d1m_monitor.py is a thin renderer over this module — it imports
parse_feed + the verdict helpers and only owns rich/plotext rendering + argparse.

Config: load_run_feed_config() reads the `run_feed:` block of configs/monitoring.yaml
and HARD-ERRORS on a missing block/key (no buried 'logs/d1m/...jsonl' default).
"""
from __future__ import annotations

import bisect
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

# ----------------------------------------------------------------------------
# config — run identity + parse knobs. HARD-ERROR on missing (zero-literal rule:
# no silent default log path / host / encoding buried in code).
# ----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MONITORING_YAML = _REPO_ROOT / "configs" / "monitoring.yaml"

# required keys in the run_feed: block — every one HARD-ERRORS if absent.
_RUN_FEED_REQUIRED_KEYS = (
    "host",
    "repo",
    "default_log_path",
    "encoding",
    "sealbot_slope_min_points",
    "sealbot_slope_ci_level",
    "sealbot_min_rise",
    "sealbot_fallback_sigma",
    "bin_width",
    "ts_sample",
    "gc_sample",
)


@dataclass(frozen=True)
class RunFeedConfig:
    """Run identity + Feed-A parse knobs (from configs/monitoring.yaml run_feed:)."""

    host: str
    repo: str
    default_log_path: str
    encoding: str
    sealbot_slope_min_points: int
    sealbot_slope_ci_level: float
    sealbot_min_rise: float
    sealbot_fallback_sigma: float
    bin_width: int
    ts_sample: int
    gc_sample: int


def load_run_feed_config(path: Optional[str] = None) -> RunFeedConfig:
    """Load the run_feed: block of configs/monitoring.yaml. HARD-ERROR (ValueError)
    when the file, the run_feed block, or any required key is missing — there is
    NO silent default log path. Env (D1M_*) overrides are applied at the call
    site (resolve_run_target), not here."""
    p = Path(path) if path else _DEFAULT_MONITORING_YAML
    if not p.exists():
        raise ValueError("monitoring config not found: %s" % p)
    with open(p) as f:
        cfg = yaml.safe_load(f) or {}
    rf = cfg.get("run_feed")
    if not isinstance(rf, dict):
        raise ValueError(
            "monitoring config %s is missing the required 'run_feed:' block "
            "(B4: run identity + Feed-A parse knobs)" % p
        )
    missing = [k for k in _RUN_FEED_REQUIRED_KEYS if rf.get(k) is None]
    if missing:
        raise ValueError(
            "monitoring config %s run_feed: missing required keys: %s"
            % (p, ", ".join(missing))
        )
    return RunFeedConfig(
        host=str(rf["host"]),
        repo=str(rf["repo"]),
        default_log_path=str(rf["default_log_path"]),
        encoding=str(rf["encoding"]),
        sealbot_slope_min_points=int(rf["sealbot_slope_min_points"]),
        sealbot_slope_ci_level=float(rf["sealbot_slope_ci_level"]),
        sealbot_min_rise=float(rf["sealbot_min_rise"]),
        sealbot_fallback_sigma=float(rf["sealbot_fallback_sigma"]),
        bin_width=int(rf["bin_width"]),
        ts_sample=int(rf["ts_sample"]),
        gc_sample=int(rf["gc_sample"]),
    )


def resolve_run_target(cfg: RunFeedConfig) -> Tuple[str, str, str, str]:
    """(host, repo, log, encoding) with D1M_* env overrides over the config floor."""
    host = os.environ.get("D1M_HOST", cfg.host)
    repo = os.environ.get("D1M_REPO", cfg.repo)
    log = os.environ.get("D1M_LOG", cfg.default_log_path)
    encoding = os.environ.get("D1M_ENCODING", cfg.encoding)
    return host, repo, log, encoding


# ----------------------------------------------------------------------------
# mathematical constants for the sealbot_slope CI (NOT tunable knobs).
# Moved VERBATIM from scripts/d1m_monitor.py — load-bearing.
# ----------------------------------------------------------------------------
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


def _t_crit(df, level):
    """Two-sided t critical value at the given CI level for the given df. Table
    is for level=0.95 (the default); other levels scale off the normal tail as a
    conservative approximation. df<1 -> the df=1 (widest) value; df>30 -> normal."""
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


# ----------------------------------------------------------------------------
# sealbot_slope verdict — load-bearing CI math, moved VERBATIM. GREEN must mean a
# REAL sustained climb, never a noisy plateau:
#   1. n >= K (false-green guard; < K -> "insufficient data", never green).
#   2. Robust slope > 0: Theil-Sen (median of pairwise slopes).
#   3. Lower CI bound > 0: small-sample t critical value (df=n-2) AND propagates
#      each point's LOGGED ci_sealbot measurement error; the MORE CONSERVATIVE
#      (higher) of a measurement-error bootstrap CI and an analytic t-interval.
#   4. Effect size: implied rise (slope x span) >= min_rise.
# ----------------------------------------------------------------------------
def sealbot_point_sigmas(rounds, fallback_sigma):
    """Per-point 1-sigma measurement error for each wr_sealbot eval point, in the
    SAME order/filtering as sealbot_slope's (step, wr) pairs. Reads the LOGGED
    ci_sealbot half-width (a ~95% interval -> /1.96 = sigma). Fallback chain when
    a point lacks ci_sealbot: Wilson SE from a logged sealbot game count if
    present, else the conservative fallback_sigma. Returns a list aligned to
    [(step, wr) for rounds with numeric step+wr_sealbot]."""
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
            sigmas.append(fallback_sigma)  # conservative widen
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


def sealbot_slope(xs, ys, sigmas, min_pts, ci_level, min_rise,
                  fallback_sigma, boot_iters=2000, rng_seed=None):
    """Return {label, style, slope, n} from the wr_sealbot eval history.
    style is "" (neutral) or "green"/"yellow" — green ONLY on a robust, CI-clear,
    above-floor climb (see block comment). `sigmas` = per-point 1-sigma logged
    measurement error (from sealbot_point_sigmas); falls back to a conservative
    constant per point when absent so a missing CI can never narrow the interval."""
    triples = [(x, y, s) for x, y, s in
               zip(xs, ys, (sigmas if sigmas is not None else [None] * len(xs)))
               if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    n = len(triples)
    if n < min_pts:
        return {"label": "insufficient data (n=%d<%d)" % (n, min_pts),
                "style": "", "slope": None, "n": n}
    pts = [(x, y) for x, y, _ in triples]
    sig = [(s if isinstance(s, (int, float)) and s > 0 else fallback_sigma)
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
# depth-health discriminator. Judge depth vs the RUN'S OWN settled baseline; use
# root_concentration co-movement to name the mechanism. Moved VERBATIM. The
# fraction thresholds are passed in by the renderer (env-overridable knobs).
# ----------------------------------------------------------------------------
def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else None


def depth_health(depth_series, conc_series, is_gumbel,
                 stable_frac, regress_frac, conc_move):
    """Run-relative depth verdict. Returns dict or None (insufficient data).

    baseline = median of history buckets (all but last 2); cur = mean of last 2.
    floor = baseline*(1-regress_frac) — moves WITH the run, not a constant.
    Mechanism comes from root_concentration co-movement over the same window.
    Under Gumbel-SH the concentration read is DESCRIPTIVE (rising root_conc is
    expected halving behavior), not a PUCT 'overconfident shallow cutoffs' alarm."""
    ds = [d for _, d in depth_series]
    if len(ds) < 3:
        return None
    hist = ds[:-2] if len(ds) >= 4 else ds[:-1]
    base = _median(hist)
    cur = sum(ds[-2:]) / 2.0
    if not base:
        return None
    pct = (cur - base) / base
    floor = base * (1 - regress_frac)

    conc_delta = None
    cs = [c for _, c in conc_series]
    if len(cs) >= 3:
        ch = cs[:-2] if len(cs) >= 4 else cs[:-1]
        conc_delta = (sum(cs[-2:]) / 2.0) - (_median(ch) or 0.0)

    def mech():
        if conc_delta is None:
            return ""
        if conc_delta > conc_move:
            if is_gumbel:
                # Gumbel-SH: rising root_conc is expected halving, not a defect.
                return " + root_conc rising (descriptive; expected under Gumbel-SH)"
            return " + root_conc rising -> overconfident shallow cutoffs (PUCT)"
        if conc_delta < -conc_move:
            return " + root_conc falling -> search broadening"
        return " + root_conc flat -> game-length/structural, not search-quality"

    if pct >= stable_frac:
        label, style, detail = "DEEPENING", "green", "+%.0f%% vs baseline %.2f" % (pct * 100, base)
    elif pct >= -stable_frac:
        label, style, detail = "stable", "green", "within +/-%.0f%% of baseline %.2f" % (
            stable_frac * 100, base)
    elif pct >= -regress_frac:
        label, style, detail = "WATCH", "yellow", "down %.0f%% vs baseline %.2f%s" % (
            -pct * 100, base, mech())
    else:
        label, style, detail = "REGRESSION", "red", "down %.0f%% below floor %.2f (base %.2f)%s" % (
            -pct * 100, floor, base, mech())
    return {"base": base, "cur": cur, "pct": pct, "floor": floor,
            "conc_delta": conc_delta, "label": label, "style": style, "detail": detail}


# ----------------------------------------------------------------------------
# layer 1 — PURE parse. Reduces an iterable of Feed-A records to a snapshot.
# Reproduces the REMOTE collector's reduction VERBATIM (ts->step bisect, 30k
# bins, gap-skipping effective rate, recent-colony, value-spread/depth/conc
# binning, gumbel-target + fp16 + eval-overhead). No ssh, no file IO.
# ----------------------------------------------------------------------------

# Gumbel-target keys (dirichlet-off full-search head target health + opening div)
_GT_KEYS = ("policy_target_entropy_fullsearch", "policy_target_kl_uniform_fullsearch",
            "early_game_entropy_mean", "early_game_top1_mass_mean")


@dataclass
class RunFeedSnapshot:
    """Decision-relevant reduction of Feed A (the run-log). Field set + semantics
    match the JSON the old REMOTE collector emitted, so the renderer is unchanged.
    Use as_dict() for the renderer / golden-parity assertions."""

    total_steps: Optional[int] = None
    encoding: Optional[str] = None
    variant: Optional[str] = None
    is_gumbel: Optional[bool] = None
    last_summary: Optional[Dict[str, Any]] = None
    last_train: Optional[Dict[str, Any]] = None
    rate_overall: float = 0.0
    rate_effective: float = 0.0
    rate_recent: float = 0.0
    eval_rounds: List[Dict[str, Any]] = field(default_factory=list)
    eval_phases: List[Dict[str, Any]] = field(default_factory=list)
    fwt: List[Dict[str, Any]] = field(default_factory=list)
    vspread_last: Optional[Dict[str, Any]] = None
    vspread_canary_last: Optional[Dict[str, Any]] = None
    colony_recent_mean: Optional[float] = None
    colony_recent_n: int = 0
    traj: List[Dict[str, Any]] = field(default_factory=list)
    n_summaries: int = 0
    gumbel_targets: Dict[str, Any] = field(default_factory=dict)
    gumbel_targets_recent: Dict[str, List[float]] = field(default_factory=dict)
    fp16_scale_last: Optional[float] = None
    fp16_scale_recent: List[float] = field(default_factory=list)
    eval_overhead_frac: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "encoding": self.encoding,
            "variant": self.variant,
            "is_gumbel": self.is_gumbel,
            "last_summary": self.last_summary,
            "last_train": self.last_train,
            "rate_overall": self.rate_overall,
            "rate_effective": self.rate_effective,
            "rate_recent": self.rate_recent,
            "eval_rounds": self.eval_rounds,
            "eval_phases": self.eval_phases,
            "fwt": self.fwt,
            "vspread_last": self.vspread_last,
            "vspread_canary_last": self.vspread_canary_last,
            "colony_recent_mean": self.colony_recent_mean,
            "colony_recent_n": self.colony_recent_n,
            "traj": self.traj,
            "n_summaries": self.n_summaries,
            "gumbel_targets": self.gumbel_targets,
            "gumbel_targets_recent": self.gumbel_targets_recent,
            "fp16_scale_last": self.fp16_scale_last,
            "fp16_scale_recent": self.fp16_scale_recent,
            "eval_overhead_frac": self.eval_overhead_frac,
        }


def _record_ts(d):
    """ISO timestamp of a record/string -> datetime (REMOTE's ts() helper)."""
    s = d["timestamp"] if isinstance(d, dict) else d
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def parse_feed(records: Iterable[dict], bin_width: int = 30000,
               ts_sample: int = 200, gc_sample: int = 50) -> RunFeedSnapshot:
    """PURE reduction of Feed-A records -> RunFeedSnapshot. Transport-agnostic.

    `records` = iterable of already-decoded JSONL dicts (the IO helpers below
    decode + pre-filter to decision-relevant event types). bin_width / ts_sample
    / gc_sample mirror the REMOTE collector constants (BIN / TS_SAMPLE / GC_SAMPLE).

    Reduction is VERBATIM with the old REMOTE collector: counter-based sampling
    over the record stream (tsi % ts_sample, gci % gc_sample), the ts->step
    bisect map over train_step_summary events, 30k step bins, recent-colony
    reverse-scan, gap-skipping effective rate, gumbel-target/fp16/eval-overhead.
    To keep the sampling counters and reverse scans identical, the iterable is
    fully materialized once."""
    BIN = bin_width
    TS_SAMPLE = ts_sample
    GC_SAMPLE = gc_sample

    recs = list(records)

    summ: List[dict] = []
    rounds: List[dict] = []
    evphases: List[dict] = []
    fwt: List[dict] = []
    vspread: List[dict] = []
    vsc: List[dict] = []
    total_steps = None
    encoding_name = None
    variant_name = None
    is_gumbel = None
    last_train = None
    tsi = gci = 0
    bce: Dict[int, list] = {}   # bucket -> [sum_sp, sum_co, n]
    col: Dict[int, list] = {}   # bucket -> [sum_colony, n]

    for d in recs:
        if not isinstance(d, dict):
            continue
        ev = d.get("event")
        try:
            if ev == "train_step_summary":
                summ.append(d)
            elif ev == "evaluation_round_complete":
                rounds.append(d)
            elif ev == "evaluation_games_complete":
                evphases.append(d)
            elif ev == "forced_win_trend":
                fwt.append(d)
            elif ev == "value_spread_alert":
                vspread.append(d)
            elif ev == "value_spread":   # always-emitted canary (has step + t3/alt)
                vsc.append(d)
            elif ev == "startup":
                _cfg = (d.get("config") or {})
                total_steps = _cfg.get("total_steps", total_steps)
                encoding_name = _cfg.get("encoding", encoding_name)
                variant_name = d.get("variant", variant_name)
                _spc = (_cfg.get("selfplay") or {})
                if _spc.get("gumbel_mcts") is not None:
                    is_gumbel = bool(_spc.get("gumbel_mcts"))
            elif ev == "train_step":
                tsi += 1
                if tsi % TS_SAMPLE == 0:
                    b = d.get("step", 0) // BIN
                    sp, co = d.get("value_bce_selfplay"), d.get("value_bce_corpus")
                    if sp is not None and co is not None:
                        e = bce.setdefault(b, [0.0, 0.0, 0]); e[0] += sp; e[1] += co; e[2] += 1
        except Exception:
            pass

    # latest train_step (high-freq fields: grad_norm, lr, value_bce, value_acc).
    # max BY TIMESTAMP, not read-order — with multi-segment (glob) reads the last
    # file in the stream is an ARCHIVED segment, so a reversed read-order scan would
    # return a stale record; the globally-latest ts is the live run's newest step.
    _lt = [(i, d) for i, d in enumerate(recs)
           if isinstance(d, dict) and d.get("event") == "train_step"]
    if _lt:
        # max by (ISO timestamp string, read-index): live's newest step wins across
        # segments; ties (or missing ts) fall back to read-order-last (old behavior).
        last_train = max(_lt, key=lambda t: (t[1].get("timestamp") or "", t[0]))[1]

    # Multi-segment support (restart logs read together via a glob): summaries can
    # arrive out of order and with OVERLAPPING steps (a resumed run re-emits steps
    # >= its resume point). Dedup by step keeping the LATEST-timestamp record (the
    # most recent run's value at each step), then sort by step -> the ts->step
    # bisect map, the summary binning, and last_summary are all chronological and
    # overlap-clean regardless of file/segment order. No-op on a single clean log.
    if summ:
        # ISO-8601 timestamps sort lexicographically = chronologically; compare the
        # raw string (safe on records lacking a timestamp — real logs always have
        # one; a missing field just sorts first / keeps read-order on ties).
        _by_step: Dict[int, dict] = {}
        for s in summ:
            _st = s.get("step")
            if _st is None:
                continue
            _prev = _by_step.get(_st)
            if _prev is None or (s.get("timestamp") or "") >= (_prev.get("timestamp") or ""):
                _by_step[_st] = s
        # sort by TIMESTAMP (not step): the step_at bisect map needs ts-monotonic
        # order. In a single append-only log read-order already = ts-order (so this
        # is a no-op, even across a resume-rewind where step dips but ts keeps
        # rising); across glob segments it re-interleaves them chronologically.
        summ = sorted(_by_step.values(), key=lambda s: s.get("timestamp") or "")

    # ts -> step map (summaries) for events lacking a step field
    stimes = [_record_ts(s) for s in summ]
    ssteps = [s["step"] for s in summ]

    def step_at(t):
        if not ssteps:
            return 0
        i = bisect.bisect_left(stimes, t)
        return ssteps[min(max(i, 0), len(ssteps) - 1)]

    # colony by bucket (game_complete: has ts, no step)
    for d in recs:
        if not (isinstance(d, dict) and d.get("event") == "game_complete"):
            continue
        gci += 1
        if gci % GC_SAMPLE:
            continue
        try:
            c = d.get("colony_extension_fraction")
            if c is None:
                continue
            b = step_at(_record_ts(d)) // BIN
            e = col.setdefault(b, [0.0, 0]); e[0] += c; e[1] += 1
        except Exception:
            pass

    # recent colony (last ~500 sampled game_complete, reverse scan)
    colony_recent: List[float] = []
    for d in reversed(recs):
        if isinstance(d, dict) and d.get("event") == "game_complete":
            try:
                c = d.get("colony_extension_fraction")
                if c is not None:
                    colony_recent.append(c)
            except Exception:
                pass
            if len(colony_recent) >= 500:
                break
    colony_recent_mean = (sum(colony_recent) / len(colony_recent)) if colony_recent else None

    # per-phase eval WRs (evaluation_games_complete has no step field — map by ts).
    # Surfaces sealbot/best_arena/random WR as each phase LANDS, even mid-round or
    # on an interrupted round (evaluation_round_complete only fires at round END).
    # Step is the WALL-CLOCK training step at phase completion (eval runs
    # concurrently), so it reads ~a fraction past the evaluated checkpoint's step.
    eval_phases_out = []
    for d in evphases:
        try:
            eval_phases_out.append({
                "step": step_at(_record_ts(d)),
                "phase": d.get("phase"),
                "wr": d.get("winrate"),
                "n": d.get("n_games"),
            })
        except Exception:
            pass
    eval_phases_out = eval_phases_out[-12:]  # recent tail only

    # map each forced_win_trend ts -> step (no step field in event)
    fwt_out = []
    for d in fwt:
        fwt_out.append({
            "step": step_at(_record_ts(d)),
            "forced_win_conversion": d.get("forced_win_conversion"),
            "off_window_forced_win_rate": d.get("off_window_forced_win_rate"),
            "n": d.get("n"),
        })

    # binned mcts mean-depth + root-concentration + value-spread (avg_sigma) from
    # train_step_summary (summaries carry a step field directly — bin like bce).
    dsg: Dict[int, list] = {}   # bucket -> [sum_depth, n_depth, sum_sigma, n_sigma, sum_conc, n_conc]
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
    vbk: Dict[int, list] = {}   # bucket -> [sum_t3, n_t3, sum_alt, n_alt]
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

    # Effective rate = sum(steps)/sum(time) over CLEAN intervals: skip restart
    # stop-gaps (dt>30min) and resume step-resets (ds<0). Eval-INCLUSIVE and
    # resume-robust -> honest ETA rate, not the swingy recent-1h pace.
    eff_steps = eff_time = rec_steps = rec_time = all_steps = all_time = 0.0
    overall = recent = effective = 0.0
    if len(summ) >= 2:
        _cut = _record_ts(summ[-1]).timestamp() - 3600              # 1h -> "recent" (swingy)
        _cut_eff = _record_ts(summ[-1]).timestamp() - 12 * 3600     # 12h (>=~1 eval cycle) -> ETA rate
        for _i in range(len(summ) - 1):
            _a, _b = summ[_i], summ[_i + 1]
            _dt = (_record_ts(_b) - _record_ts(_a)).total_seconds()
            _ds = _b["step"] - _a["step"]
            if _dt <= 0 or _dt > 1800 or _ds < 0:           # skip restart gaps + resume step-resets
                continue
            all_steps += _ds; all_time += _dt
            if _record_ts(_a).timestamp() >= _cut_eff:
                eff_steps += _ds; eff_time += _dt
            if _record_ts(_a).timestamp() >= _cut:
                rec_steps += _ds; rec_time += _dt
        overall = all_steps / all_time * 3600 if all_time else 0.0
        effective = eff_steps / eff_time * 3600 if eff_time else overall
        recent = rec_steps / rec_time * 3600 if rec_time else effective

    # Gumbel-target / opening-diversity — from the LATEST summary + trailing window
    gumbel_targets = {}
    ls_summ = summ[-1] if summ else {}
    for k in _GT_KEYS:
        v = ls_summ.get(k)
        if isinstance(v, (int, float)):
            gumbel_targets[k] = v
    gt_recent = {k: [] for k in _GT_KEYS}
    for sd in summ[-30:]:
        for k in _GT_KEYS:
            v = sd.get(k)
            if isinstance(v, (int, float)):
                gt_recent[k].append(v)

    # fp16 AMP-scale canary — latest train_step value + a trailing window
    fp16_recent: List[float] = []
    _fc = 0
    for d in reversed(recs):
        if isinstance(d, dict) and d.get("event") == "train_step":
            try:
                v = d.get("fp16_scale")
                if isinstance(v, (int, float)):
                    fp16_recent.append(v)
            except Exception:
                pass
            _fc += 1
            if _fc >= 4000:
                break
    fp16_recent.reverse()
    fp16_last = fp16_recent[-1] if fp16_recent else None

    # value-head fields per eval round — value_fc2_weight_abs_max + g4 band-pass
    for d in rounds:
        d["value_fc2_weight_abs_max"] = d.get("value_fc2_weight_abs_max")
        d["g4_value_head_band_pass"] = d.get("g4_value_head_band_pass")

    # eval-overhead from TIMESTAMP gaps (approx)
    eval_overhead_frac = None
    if len(summ) >= 2 and rounds:
        _run_wall = (_record_ts(summ[-1]) - _record_ts(summ[0])).total_seconds()
        _ssorted = [(_record_ts(s), s.get("step")) for s in summ]
        _ovh = 0.0
        for r in rounds:
            es = r.get("step")
            et = _record_ts(r) if r.get("timestamp") else None
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

    return RunFeedSnapshot(
        total_steps=total_steps,
        encoding=encoding_name,
        variant=variant_name,
        is_gumbel=is_gumbel,
        last_summary=summ[-1] if summ else None,
        last_train=last_train,
        rate_overall=overall,
        rate_effective=effective,
        rate_recent=recent,
        eval_rounds=rounds,
        eval_phases=eval_phases_out,
        fwt=fwt_out,
        vspread_last=vspread[-1] if vspread else None,
        vspread_canary_last=vsc[-1] if vsc else None,
        colony_recent_mean=colony_recent_mean,
        colony_recent_n=len(colony_recent),
        traj=traj,
        n_summaries=len(summ),
        gumbel_targets=gumbel_targets,
        gumbel_targets_recent=gt_recent,
        fp16_scale_last=fp16_last,
        fp16_scale_recent=fp16_recent[-200:],
        eval_overhead_frac=eval_overhead_frac,
    )


# ----------------------------------------------------------------------------
# layer 2 — thin IO helpers. Decode + pre-filter to decision-relevant event
# types, yield dicts for parse_feed. NO parsing/verdict logic lives here.
# ----------------------------------------------------------------------------

# event types parse_feed consumes — used to pre-filter the remote grep so we
# transfer KB not the 150MB file. (forced_win_trend matches the 'forced_win'
# prefix grep; value_spread_alert matches 'value_spread'.)
_DECISION_EVENT_SUBSTRINGS = (
    "train_step_summary",
    "evaluation_round_complete",
    "evaluation_games_complete",  # per-phase WRs — visible mid-round / on interrupted rounds
    "forced_win_trend",
    "value_spread",
    "startup",
    '"event": "train_step"',
    '"event": "game_complete"',
)


def read_local_jsonl(path: str) -> List[dict]:
    """Read a local Feed-A JSONL file -> list of decoded records. Thin IO; all
    reduction is in parse_feed. Non-JSON / blank lines are skipped."""
    out: List[dict] = []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if not s.startswith("{"):
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                continue
    return out


# Remote subsampling awk program (piped to the remote awk over ssh stdin, so NO
# shell-quoting of the JSON patterns). train_step_summary / raw train_step /
# game_complete fire EVERY step -> the full decision-event stream is ~the whole
# 340MB file. We keep every TS-th (GC-th) of each frequent type PLUS its final
# record, and ALL rarer decision events in full -> KB-to-low-MB transfer instead
# of 340MB. Classification MUST stay in sync with _DECISION_EVENT_SUBSTRINGS.
_REMOTE_SUBSAMPLE_AWK = (
    '/train_step_summary/            { a++; if (a % TS == 1) print; la=$0; next }\n'
    '/"event": "train_step"/         { b++; if (b % TS == 1) print; lb=$0; next }\n'
    '/"event": "game_complete"/      { c++; if (c % GC == 1) print; lc=$0; next }\n'
    '/evaluation_round_complete/ || /evaluation_games_complete/ '
    '|| /forced_win_trend/ || /value_spread/ '
    '|| /startup/                    { print; next }\n'
    'END { if (la != "") print la; if (lb != "") print lb; if (lc != "") print lc }\n'
)


def read_remote_ssh(host: str, repo: str, log: str, ts_sample: int = 200,
                    gc_sample: int = 50, timeout: int = 120) -> List[dict]:
    """Stream a remote Feed-A run-log over ssh -> list of decoded records.

    The remote side does BOTH the decision-event filter AND the high-frequency
    subsample (via ``_REMOTE_SUBSAMPLE_AWK`` piped over ssh stdin): every
    ``ts_sample``-th train_step(+summary), every ``gc_sample``-th game_complete,
    each type's final record, and ALL rarer decision events in full. This is the
    KB-not-MB transfer the renderer relies on — a plain remote grep matches ~the
    whole 340MB file (train_step fires every step). The LOCAL parse_feed is then
    called with sampling DISABLED (=1) by the caller, since the stream is already
    subsampled. ssh hardened with BatchMode (no password hang) + ConnectTimeout.
    Raises RuntimeError on ssh failure / timeout so the renderer surfaces a
    FETCH ERROR panel."""
    # `log` may be a GLOB (e.g. logs/run2_mw_fresh*.jsonl) to read every restart
    # segment as one continuous history — pass it UNQUOTED so the remote shell
    # expands it into awk's file args (awk reads the program from stdin via -f -,
    # then processes each matched file; parse_feed dedups+sorts by step so segment
    # order/overlap doesn't matter). A plain path (no glob metachar) is quoted as
    # before. Guard: only treat as a glob when it actually contains * / ? / [.
    _is_glob = any(ch in log for ch in "*?[")
    _log_arg = log if _is_glob else "'%s'" % log
    remote_cmd = "cd '%s' && awk -v TS=%d -v GC=%d -f - %s" % (
        repo, max(1, int(ts_sample)), max(1, int(gc_sample)), _log_arg)
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, remote_cmd]
    try:
        p = subprocess.run(cmd, input=_REMOTE_SUBSAMPLE_AWK, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError("ssh timeout (>%ds)" % timeout)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("ssh failed: %s" % e)
    if p.returncode != 0:
        msg = (p.stderr or "").strip().splitlines()
        msg = [m for m in msg if "vast.ai" not in m and "Have fun" not in m]
        raise RuntimeError("ssh rc=%d: %s" % (p.returncode, " ".join(msg[-3:]) or "?"))
    out: List[dict] = []
    for ln in p.stdout.splitlines():
        s = ln.strip()
        if not s.startswith("{"):
            continue
        try:
            out.append(json.loads(s))
        except Exception:
            continue
    return out


# Sentinel lines delimiting the live-tip block in read_live_tip's remote output.
_TIP_NOW = "###TIP_NOW###"
_TIP_FILE = "###TIP_FILE###"


def read_live_tip(host: str, repo: str, log: str, timeout: int = 30) -> Optional[dict]:
    """Cheap probe of the ACTUALLY-LIVE run tip — robust to resumes + clock skew.

    The heavy ``read_remote_ssh`` subsamples and, over a multi-file glob, its
    END-block captures the LAST FILE's tail (an archived restart segment), not the
    live log's — so the reported step/last-event lag the true tip by up to
    ts_sample×log_interval steps and look "disconnected" right after a resume.
    This helper instead: (1) resolves the LIVE log = newest-mtime glob member (so a
    resume that rotates/re-creates the log is followed automatically), (2) tails its
    true-last train_step_summary + train_step (un-subsampled), and (3) reads the
    REMOTE clock so "last event ago" is computed vast-clock-relative (the vast box
    runs ~2h behind real UTC — a laptop-now minus vast-ts age is inflated by the
    skew and falsely reads as stale). Returns None on any failure (caller falls
    back to the parsed snapshot); never raises."""
    g = log if any(ch in log for ch in "*?[") else "'%s'" % log
    # date +%s.%N (remote epoch) ; newest-mtime member ; its last summary + train_step
    # NB: sentinels MUST be single-quoted in the echo — a bare `###...` is a shell
    # COMMENT (# starts a comment at word start) and would print nothing.
    remote = (
        "cd '%s' && echo '%s' && date +%%s.%%N && "
        "F=$(ls -t %s 2>/dev/null | head -1) && echo '%s' && echo \"$F\" && "
        "tail -c 400000 \"$F\" 2>/dev/null | grep '\"train_step_summary\"' | tail -1 && "
        "tail -c 400000 \"$F\" 2>/dev/null | grep '\"event\": \"train_step\"' | tail -1"
        % (repo, _TIP_NOW, g, _TIP_FILE)
    )
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, remote]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:  # noqa: BLE001
        return None
    if p.returncode != 0:
        return None
    now_epoch = None
    live_file = None
    summary = None
    train = None
    lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln == _TIP_NOW and i + 1 < len(lines):
            try:
                now_epoch = float(lines[i + 1])
            except Exception:
                pass
            i += 2
            continue
        if ln == _TIP_FILE and i + 1 < len(lines):
            live_file = lines[i + 1]
            i += 2
            continue
        if ln.startswith("{"):
            try:
                d = json.loads(ln)
                ev = d.get("event")
                if ev == "train_step_summary":
                    summary = d
                elif ev == "train_step":
                    train = d
            except Exception:
                pass
        i += 1
    if now_epoch is None and summary is None and train is None:
        return None
    # last-event age in the REMOTE clock frame (skew-immune)
    age = None
    tip = summary or train
    if now_epoch is not None and tip is not None and tip.get("timestamp"):
        try:
            ts = _record_ts(tip).timestamp()
            age = max(0.0, now_epoch - ts)
        except Exception:
            pass
    return {
        "live_file": live_file,
        "live_step": (tip.get("step") if tip else None),
        "last_summary": summary,
        "last_train": train,
        "age_sec": age,
    }
