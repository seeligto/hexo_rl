"""B4 §D-DECIDE Track-B — run_feed_reader (READ-ONLY consumer of Feed A).

Covers the PURE layer (parse_feed) over a SYNTHETIC Feed-A record list (NO ssh,
local fixtures only):
  - eval_rounds passthrough + value-head field attach
  - 30k trajectory bins (value-bce / colony / depth+sigma+conc / t3+alt)
  - step_at ts->step bisect interpolation (forced_win_trend has no step field)
  - recent-colony reverse-scan mean
  - effective-rate gap-skip (dt>1800 restart gaps + ds<0 resume resets excluded)
  - sealbot_slope false-green guard (<min_points -> neutral, NEVER green)
  - depth_health run-relative verdict
  - load_run_feed_config HARD-ERROR on a missing block/key
GOLDEN PARITY: parse_feed over the committed local fixture reproduces the output
the OLD inline d1m_monitor REMOTE collector emitted (captured pre-rewrite).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from hexo_rl.monitoring.run_feed_reader import (
    RunFeedConfig,
    depth_health,
    load_run_feed_config,
    parse_feed,
    read_local_jsonl,
    sealbot_point_sigmas,
    sealbot_slope,
)

FIX = Path(__file__).parent / "fixtures"
_BIN = 30000


# ── helpers to build small synthetic Feed-A record lists ──────────────────────
_BASE = datetime(2026, 6, 24, 0, 0, 0)


def _ts(sec: float) -> str:
    return (_BASE + timedelta(seconds=sec)).isoformat() + "Z"


def _summary(step, t, **kw):
    d = {"event": "train_step_summary", "timestamp": _ts(t), "step": step}
    d.update(kw)
    return d


# ── parse_feed structural assertions over a small synthetic list ──────────────

def _small_feed():
    recs = [{
        "event": "startup",
        "variant": "vtest",
        "config": {"total_steps": 200000, "encoding": "v6_live2_ls",
                   "selfplay": {"gumbel_mcts": True}},
    }]
    # summaries 5k apart, 600s apart (clean intervals) — drive ts->step + bins
    step = 0
    t = 0
    for i in range(8):
        step += 5000
        t += 600
        recs.append(_summary(step, t, mcts_mean_depth=4.0 - i * 0.1,
                             avg_sigma=0.3, mcts_root_concentration=0.5,
                             draw_rate=0.01,
                             policy_target_entropy_fullsearch=1.8))
    return recs, step, t


def test_parse_feed_basic_fields():
    recs, last_step, _ = _small_feed()
    snap = parse_feed(recs)
    assert snap.total_steps == 200000
    assert snap.encoding == "v6_live2_ls"
    assert snap.variant == "vtest"
    assert snap.is_gumbel is True
    assert snap.n_summaries == 8
    assert snap.last_summary["step"] == last_step


def test_parse_feed_eval_rounds_and_value_head_attach():
    recs, _, _ = _small_feed()
    recs.append({"event": "evaluation_round_complete", "timestamp": _ts(700),
                 "step": 30000, "wr_best": 0.55, "wr_sealbot": 0.21,
                 "ci_sealbot": [0.11, 0.31], "promoted": True})
    recs.append({"event": "evaluation_round_complete", "timestamp": _ts(720),
                 "step": 60000, "wr_best": 0.57, "wr_sealbot": 0.23,
                 "value_fc2_weight_abs_max": 0.12, "g4_value_head_band_pass": True})
    snap = parse_feed(recs)
    assert len(snap.eval_rounds) == 2
    # value-head fields attached (None default for the round lacking them)
    assert snap.eval_rounds[0]["value_fc2_weight_abs_max"] is None
    assert snap.eval_rounds[0]["g4_value_head_band_pass"] is None
    assert snap.eval_rounds[1]["value_fc2_weight_abs_max"] == 0.12
    assert snap.eval_rounds[1]["g4_value_head_band_pass"] is True


def test_parse_feed_trajectory_bins():
    recs, _, _ = _small_feed()
    # value_spread canary carries step -> t3/alt bins
    for i in range(3):
        recs.append({"event": "value_spread", "step": _BIN * (i + 1),
                     "t3_spread": 0.6, "alt_spread": 0.2})
    snap = parse_feed(recs)
    assert snap.traj, "expected binned trajectory rows"
    # every bin step is a multiple of the 30k bin width
    for row in snap.traj:
        assert row["step"] % _BIN == 0
    # depth bins exist (from summaries) — and depth is the per-bin mean
    depth_rows = [r for r in snap.traj if "depth" in r]
    assert depth_rows
    # t3/alt bins present
    t3_rows = [r for r in snap.traj if "t3" in r]
    assert len(t3_rows) == 3
    assert all(abs(r["t3"] - 0.6) < 1e-9 for r in t3_rows)


def test_parse_feed_step_at_bisect_interpolation():
    # forced_win_trend has NO step field — its step is interpolated from the
    # ts->step summary map via bisect. Place a fwt event AT a known summary ts.
    recs, _, t = _small_feed()
    # summary @ t=600 -> step 5000, @ t=1200 -> step 10000, ...
    # fwt at t=1200 should map to step 10000 (bisect_left lands on that summary)
    recs.append({"event": "forced_win_trend", "timestamp": _ts(1200),
                 "forced_win_conversion": 0.8, "off_window_forced_win_rate": 0.3,
                 "n": 100})
    snap = parse_feed(recs)
    assert len(snap.fwt) == 1
    assert snap.fwt[0]["step"] == 10000
    assert snap.fwt[0]["forced_win_conversion"] == 0.8


def test_parse_feed_recent_colony_mean():
    recs, _, t = _small_feed()
    # 4 game_complete events (carry ts, no step) — recent mean over reverse scan
    vals = [0.10, 0.12, 0.14, 0.16]
    for i, c in enumerate(vals):
        recs.append({"event": "game_complete", "timestamp": _ts(t + 10 + i),
                     "colony_extension_fraction": c})
    snap = parse_feed(recs)
    assert snap.colony_recent_n == 4
    assert abs(snap.colony_recent_mean - sum(vals) / len(vals)) < 1e-9


def test_parse_feed_effective_rate_skips_restart_and_resume_gaps():
    # Clean intervals: 5000 steps / 600s = 30000 steps/hr. Inject a restart gap
    # (dt>1800) and a resume reset (ds<0) — both must be EXCLUDED from the rate.
    recs = [{"event": "startup", "config": {"total_steps": 1000000,
             "encoding": "v6_live2_ls", "selfplay": {"gumbel_mcts": True}}}]
    step = 0
    t = 0
    for _i in range(5):
        step += 5000
        t += 600
        recs.append(_summary(step, t))
    # restart gap: 5000s elapsed (>1800) AND step jumps back (resume reset, ds<0)
    t += 5000
    step_reset = step - 3000
    recs.append(_summary(step_reset, t))
    # then resume cleanly at the same 30000/hr pace
    for _i in range(3):
        step_reset += 5000
        t += 600
        recs.append(_summary(step_reset, t))
    snap = parse_feed(recs)
    # the gap interval (dt>1800 + ds<0) is skipped -> clean rate preserved
    assert abs(snap.rate_effective - 30000.0) < 1.0
    assert abs(snap.rate_overall - 30000.0) < 1.0


def test_parse_feed_train_step_sampling_and_last_train():
    # train_step is sampled every ts_sample; last_train is the LAST train_step.
    recs = [{"event": "startup", "config": {"total_steps": 100000,
             "encoding": "v6_live2_ls"}}]
    recs.append(_summary(5000, 600))
    for i in range(20):
        recs.append({"event": "train_step", "step": (i + 1) * 100,
                     "value_bce_selfplay": 0.5, "value_bce_corpus": 0.45,
                     "grad_norm": 1.5, "lr": 0.001, "fp16_scale": 65536.0})
    snap = parse_feed(recs, ts_sample=5)  # sample every 5th -> 4 sampled
    assert snap.last_train["step"] == 2000  # the LAST train_step regardless of sampling
    assert snap.fp16_scale_last == 65536.0


# ── sealbot_slope false-green guard ───────────────────────────────────────────

def _cfg_defaults():
    return load_run_feed_config()


def test_sealbot_slope_false_green_guard_insufficient_points():
    cfg = _cfg_defaults()
    # n < min_points -> neutral ("insufficient"), NEVER green
    rounds = [{"step": 30000 * i, "wr_sealbot": 0.2 + 0.05 * i,
               "ci_sealbot": [0.1 + 0.05 * i, 0.3 + 0.05 * i]}
              for i in range(cfg.sealbot_slope_min_points - 1)]
    xs = [r["step"] for r in rounds]
    ys = [r["wr_sealbot"] for r in rounds]
    sig = sealbot_point_sigmas(rounds, cfg.sealbot_fallback_sigma)
    res = sealbot_slope(xs, ys, sig, min_pts=cfg.sealbot_slope_min_points,
                        ci_level=cfg.sealbot_slope_ci_level,
                        min_rise=cfg.sealbot_min_rise,
                        fallback_sigma=cfg.sealbot_fallback_sigma)
    assert res["style"] == ""           # neutral, not green
    assert "insufficient" in res["label"]
    assert res["n"] == cfg.sealbot_slope_min_points - 1


def test_sealbot_slope_flat_plateau_not_green():
    cfg = _cfg_defaults()
    # >= min_points but a NOISY FLAT plateau with one lucky last spike -> the
    # measurement-error CI must straddle 0 -> never green (the original defect).
    rng_vals = [0.20, 0.21, 0.19, 0.20, 0.205, 0.31]  # spike at the end
    rounds = [{"step": 30000 * i, "wr_sealbot": v, "ci_sealbot": [v - 0.10, v + 0.10]}
              for i, v in enumerate(rng_vals)]
    xs = [r["step"] for r in rounds]
    ys = [r["wr_sealbot"] for r in rounds]
    sig = sealbot_point_sigmas(rounds, cfg.sealbot_fallback_sigma)
    res = sealbot_slope(xs, ys, sig, min_pts=cfg.sealbot_slope_min_points,
                        ci_level=cfg.sealbot_slope_ci_level,
                        min_rise=cfg.sealbot_min_rise,
                        fallback_sigma=cfg.sealbot_fallback_sigma)
    assert res["style"] != "green", "noisy plateau must NOT false-green"


def test_sealbot_point_sigmas_uses_ci_then_fallback():
    cfg = _cfg_defaults()
    rounds = [
        {"step": 0, "wr_sealbot": 0.2, "ci_sealbot": [0.1, 0.3]},   # half=0.10 -> sigma 0.10/1.96
        {"step": 30000, "wr_sealbot": 0.25},                         # no ci -> fallback
    ]
    sig = sealbot_point_sigmas(rounds, cfg.sealbot_fallback_sigma)
    assert len(sig) == 2
    assert abs(sig[0] - 0.10 / 1.959963985) < 1e-9
    assert sig[1] == cfg.sealbot_fallback_sigma


# ── depth_health run-relative verdict ─────────────────────────────────────────

def test_depth_health_insufficient_data_returns_none():
    assert depth_health([(0, 4.0), (30000, 4.0)], [], True, 0.05, 0.12, 0.03) is None


def test_depth_health_stable_within_baseline_band():
    depth_series = [(i * _BIN, 4.0) for i in range(6)]
    conc_series = [(i * _BIN, 0.5) for i in range(6)]
    dh = depth_health(depth_series, conc_series, True, 0.05, 0.12, 0.03)
    assert dh is not None
    assert dh["label"] == "stable"
    assert dh["style"] == "green"


def test_depth_health_regression_below_floor():
    # baseline ~4.0 (history), last-2 mean drops well below the -12% floor
    depth_series = [(0, 4.0), (_BIN, 4.0), (2 * _BIN, 4.0), (3 * _BIN, 4.0),
                    (4 * _BIN, 3.2), (5 * _BIN, 3.0)]
    conc_series = [(i * _BIN, 0.5) for i in range(6)]
    dh = depth_health(depth_series, conc_series, True, 0.05, 0.12, 0.03)
    assert dh["label"] == "REGRESSION"
    assert dh["style"] == "red"


# ── config HARD-ERROR ─────────────────────────────────────────────────────────

def test_load_run_feed_config_reads_default():
    cfg = load_run_feed_config()
    assert isinstance(cfg, RunFeedConfig)
    assert cfg.host
    assert cfg.repo
    assert cfg.default_log_path
    assert cfg.encoding
    assert cfg.sealbot_slope_min_points >= 1
    assert cfg.bin_width == _BIN


def test_load_run_feed_config_missing_block_raises(tmp_path):
    p = tmp_path / "no_run_feed.yaml"
    p.write_text("monitoring:\n  enabled: true\n")
    with pytest.raises(ValueError, match="run_feed"):
        load_run_feed_config(str(p))


def test_load_run_feed_config_missing_key_raises(tmp_path):
    p = tmp_path / "partial.yaml"
    p.write_text(
        "run_feed:\n"
        "  host: vast\n"
        "  repo: /workspace/hexo_rl\n"
        "  # default_log_path intentionally omitted -> HARD ERROR (no silent default)\n"
        "  encoding: v6_live2_ls\n"
        "  sealbot_slope_min_points: 5\n"
        "  sealbot_slope_ci_level: 0.95\n"
        "  sealbot_min_rise: 0.03\n"
        "  sealbot_fallback_sigma: 0.12\n"
        "  bin_width: 30000\n"
        "  ts_sample: 200\n"
        "  gc_sample: 50\n"
    )
    with pytest.raises(ValueError, match="default_log_path"):
        load_run_feed_config(str(p))


def test_load_run_feed_config_missing_file_raises(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_run_feed_config(str(tmp_path / "does_not_exist.yaml"))


# ── read_local_jsonl thin IO ──────────────────────────────────────────────────

def test_read_local_jsonl_decodes_and_skips_noise(tmp_path):
    p = tmp_path / "feed.jsonl"
    p.write_text(
        '{"event": "startup", "config": {}}\n'
        "not json line\n"
        "\n"
        '{"event": "train_step", "step": 1}\n'
        "{bad json\n"
    )
    recs = read_local_jsonl(str(p))
    assert len(recs) == 2
    assert recs[0]["event"] == "startup"
    assert recs[1]["event"] == "train_step"


# ── GOLDEN PARITY: parse_feed reproduces the old REMOTE collector output ───────

def test_golden_parity_local_fixture():
    """parse_feed over the committed local fixture must reproduce, field-for-field,
    the JSON the OLD inline d1m_monitor REMOTE collector emitted (captured before
    the B4 rewrite). The parse reduction is load-bearing — this pins it byte-equal."""
    recs = read_local_jsonl(str(FIX / "feed_a_fixture.jsonl"))
    snap = parse_feed(recs).as_dict()
    golden = json.loads((FIX / "feed_a_golden.json").read_text())
    # round-trip through JSON so float repr / key order match the golden dump
    got = json.loads(json.dumps(snap, sort_keys=True))
    assert got == golden
