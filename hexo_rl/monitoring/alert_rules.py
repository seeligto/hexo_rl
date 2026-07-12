"""Alert rule evaluation for monitoring renderers.

Pure functions extracted from ``terminal_dashboard.TerminalDashboard``
(§176 P49). Each rule consumes the relevant payload fields and
``MonitoringConfig`` thresholds and returns either an alert message
string or ``None``.

Stateless: the rolling-loss-window rule receives the window list from the
caller; it does not own the deque. Renderers wire these into their own
alert lifecycle (de-duplication, TTL).

No alert message text or threshold semantics changed during extraction.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from hexo_rl.monitoring.config import MonitoringConfig


# ── Individual rules ─────────────────────────────────────────────────────


def check_entropy_collapse(
    payload: dict, cfg: MonitoringConfig
) -> Optional[str]:
    """Combined-stream entropy below ``alert_entropy_min``."""
    ent = payload.get("policy_entropy")
    if ent is not None and ent < float(cfg.alert_entropy_min):
        return f"policy entropy {ent:.2f} — possible mode collapse"
    return None


def check_selfplay_entropy_collapse(
    payload: dict, cfg: MonitoringConfig
) -> Optional[str]:
    """Selfplay-stream entropy below ``collapse_threshold_nats`` (§70).

    Prefers the canonical ``selfplay_model_entropy_batch`` field; falls
    back to ``policy_entropy_selfplay`` for legacy JSONL streams.
    """
    ent_sp = payload.get(
        "selfplay_model_entropy_batch", payload.get("policy_entropy_selfplay")
    )
    if (
        ent_sp is not None
        and isinstance(ent_sp, (int, float))
        and math.isfinite(ent_sp)
        and ent_sp < float(cfg.collapse_threshold_nats)
    ):
        return f"selfplay entropy {ent_sp:.2f} — selfplay mode collapse"
    return None


def check_grad_norm_spike(
    payload: dict, cfg: MonitoringConfig
) -> Optional[str]:
    """Grad norm above ``alert_grad_norm_max`` (NaN ignored)."""
    gn = payload.get("grad_norm")
    # ``gn == gn`` filters NaN; preserved verbatim from pre-extraction site.
    if gn is not None and gn == gn and gn > float(cfg.alert_grad_norm_max):
        return f"grad norm {gn:.1f} — instability"
    return None


def check_loss_increase_window(
    window: list, cfg: MonitoringConfig
) -> Optional[str]:
    """``alert_loss_increase_window`` consecutive strictly increasing losses.

    ``window`` is the tail of recent loss_total values (caller-owned
    deque). Fires when the last ``window`` size is greater than
    ``alert_loss_increase_window`` and every step in the last
    ``alert_loss_increase_window + 1`` samples is strictly increasing.
    """
    n = int(cfg.alert_loss_increase_window)
    if len(window) <= n:
        return None
    tail = list(window)[-n - 1:]
    if all(tail[i] < tail[i + 1] for i in range(len(tail) - 1)):
        return f"loss increased {n} consecutive steps"
    return None


def check_sealbot_gate_failed(payload: dict) -> Optional[str]:
    """``eval_complete`` payload with ``sealbot_gate_passed`` == False."""
    if payload.get("sealbot_gate_passed") is False:
        wr = payload.get("win_rate_vs_sealbot")
        wr_str = f"{wr:.1%}" if wr is not None else "?"
        return f"SealBot eval FAILED — {wr_str} win rate"
    return None


def check_sealbot_wr_hard_abort(
    wr_history: list[tuple[int, float]],
    current_step: int,
    cfg: MonitoringConfig,
) -> Optional[str]:
    """§S181-AUDIT Wave 3 Stage 2B — sliding-window SealBot WR HARD-ABORT (L50).

    Wave 2 evidence (audit/structural/wave2_real_run_analysis.md L50):
    alt V_spread held +0.18-+0.30 throughout 46k steps while wr_sealbot
    collapsed 33% → 5%. The held-out V_spread canary failed to track
    actual eval performance — sustained-run gates MUST include a
    sliding-window SealBot WR trajectory tracker as the PRIMARY trigger.

    Three triggers (any fires → HARD-ABORT recommended; caller decides
    enforcement via cfg.wr_hard_abort_enabled):
      A. Rolling-mean WR below `wr_rolling_threshold` for
         `wr_rolling_consecutive_evals` consecutive evals AFTER
         `wr_rolling_min_step` — catches gradual decline.
      B. WR < peak × `wr_collapse_from_peak_ratio` for
         `wr_collapse_consecutive_evals` consecutive evals past
         `wr_collapse_min_step` — catches Wave-2-style 33%→16% collapse
         (consecutive: avoids aborting on a self-correcting transient dip).
      C. WR < `wr_early_death_threshold` for `wr_collapse_consecutive_evals`
         consecutive evals past `wr_early_death_min_step` — §S180b early death.

    Args:
        wr_history: list of (step, wr) tuples for last N eval rounds
                    (caller-owned ring; typically last 3-5 rounds).
        current_step: current training step (typically equals the step
                      of the most recent eval round).
        cfg: MonitoringConfig with wr_* thresholds.

    Returns:
        Human-readable HARD-ABORT message if any trigger fires, else None.
        Caller decides whether to act on the message (set
        `shutdown.running = False` when `cfg.wr_hard_abort_enabled` is true).

    Stateless: caller owns the `wr_history` list.
    """
    if not cfg.wr_hard_abort_enabled or not wr_history:
        return None

    current_wr = wr_history[-1][1]
    peak_wr = max(wr for _, wr in wr_history)

    # Triggers C and B now require N CONSECUTIVE low evals (not single-point):
    # the colony attractor causes transient SealBot-WR dips that self-correct
    # (§175/L34); a single 5% dip at 75k aborted a RECOVERING golong run at 87.5k
    # (2026-06-07; 87.5k re-eval recovered to ~0.23).
    n_consec_collapse = int(cfg.wr_collapse_consecutive_evals)

    # Trigger C (early death — N consecutive below floor, past min_step 15k).
    if (
        current_step > cfg.wr_early_death_min_step
        and len(wr_history) >= n_consec_collapse
        and all(
            wr < cfg.wr_early_death_threshold
            for _, wr in wr_history[-n_consec_collapse:]
        )
    ):
        return (
            f"HARD-ABORT (L50/Wave3-C): SealBot WR {current_wr:.1%} "
            f"< {cfg.wr_early_death_threshold:.0%} for {n_consec_collapse} "
            f"consecutive evals past step {cfg.wr_early_death_min_step:,} "
            f"— §S180b-style early death"
        )

    # Trigger B (collapse from peak — N consecutive below peak×ratio, past 25k).
    if (
        current_step > cfg.wr_collapse_min_step
        and peak_wr > 0.0
        and len(wr_history) >= n_consec_collapse
        and all(
            wr < peak_wr * cfg.wr_collapse_from_peak_ratio
            for _, wr in wr_history[-n_consec_collapse:]
        )
    ):
        return (
            f"HARD-ABORT (L50/Wave3-B): SealBot WR {current_wr:.1%} "
            f"< peak {peak_wr:.1%} × {cfg.wr_collapse_from_peak_ratio:.0%} "
            f"for {n_consec_collapse} consecutive evals past step "
            f"{cfg.wr_collapse_min_step:,} — Wave-2-style collapse"
        )

    # Trigger A (rolling-mean below threshold for consecutive evals past min_step 20k).
    n_consec = int(cfg.wr_rolling_consecutive_evals)
    if (
        current_step > cfg.wr_rolling_min_step
        and len(wr_history) >= n_consec
    ):
        tail = wr_history[-n_consec:]
        if all(wr < cfg.wr_rolling_threshold for _, wr in tail):
            mean_wr = sum(wr for _, wr in tail) / len(tail)
            return (
                f"HARD-ABORT (L50/Wave3-A): rolling-mean SealBot WR "
                f"{mean_wr:.1%} < {cfg.wr_rolling_threshold:.0%} for "
                f"{n_consec} consecutive evals past step "
                f"{cfg.wr_rolling_min_step:,}"
            )

    return None


def check_strength_regression_abort(
    strength_history: list[tuple[int, float]],
    cycle_density: float,
    current_step: int,
    cfg: MonitoringConfig,
) -> Optional[str]:
    """D-EVALFOUND — checkpoint-relative STRENGTH-regression HARD-ABORT (replaces the
    SealBot-WR abort).

    Fires when the strength aggregate (current ckpt vs a FIXED frozen reference set —
    Tier-B, cycle-robust) stays below ``strength_abort_floor`` for
    ``strength_abort_consecutive_evals`` consecutive evals past ``strength_abort_min_step``.

    CYCLE-AWARE: when ``cycle_density`` (directed-3-cycle density of the reference ladder)
    is at or above ``strength_cycle_density_max`` the ladder is a non-transitive
    (rock-paper-scissors) equilibrium, not a regression, and the abort is SUPPRESSED.
    The cheaper error here is a MISSED abort, not a false one that kills a recovering run
    (§175/L34); cycle-suppression encodes that asymmetry — and applies to the STRENGTH
    axis ONLY (robustness abort is never cycle-suppressed).

    Stateless: caller owns ``strength_history``.
    """
    if not cfg.strength_abort_enabled or not strength_history:
        return None
    if cycle_density >= cfg.strength_cycle_density_max:
        return None  # non-transitive equilibrium — suppressed
    n = int(cfg.strength_abort_consecutive_evals)
    if current_step <= cfg.strength_abort_min_step or len(strength_history) < n:
        return None
    tail = strength_history[-n:]
    if all(agg < cfg.strength_abort_floor for _, agg in tail):
        mean_agg = sum(a for _, a in tail) / len(tail)
        return (
            f"HARD-ABORT (D-EVALFOUND/strength): mean reference-set aggregate "
            f"{mean_agg:.3f} < floor {cfg.strength_abort_floor:.3f} for {n} consecutive "
            f"evals past step {cfg.strength_abort_min_step:,} (cycle_density "
            f"{cycle_density:.3f} < {cfg.strength_cycle_density_max:.3f} — a real "
            f"regression, not a non-transitive cloud)"
        )
    return None


def check_strength_warn(
    strength_aggregate: Optional[float], cfg: MonitoringConfig
) -> Optional[str]:
    """D-EVALFOUND — single-eval STRENGTH WARN (spec §1a matrix row 1). Fires on ONE eval
    below the floor (no consecutive requirement — that is the ABORT). Informational; does
    not gate. None when the aggregate is absent (ref-set producer not configured)."""
    if strength_aggregate is None:
        return None
    if strength_aggregate < cfg.strength_abort_floor:
        return (
            f"WARNING (D-EVALFOUND/strength): reference-set aggregate "
            f"{strength_aggregate:.3f} < floor {cfg.strength_abort_floor:.3f} (single eval)"
        )
    return None


def check_objective_a_coverage(
    *, strength_abort_enabled: bool, robustness_abort_enabled: bool,
    offwindow_monitor_enabled: bool,
) -> Optional[str]:
    """D-EVALFOUND pre-flight (REVIEW lost-signal guard) — WARN at run start when NO
    Objective-A / off-distribution signal is active: SealBot-WR is demoted, and if the
    strength abort, the robustness abort, AND the off-window robustness monitor are all
    off, a naive run can PROMOTE an exploitable checkpoint with no off-window guard. The
    §7 recipe says enable the off-window robustness monitor; this surfaces the gap LOUDLY
    (non-blocking). ``offwindow_monitor_enabled`` is supplied by the eval pipeline (the
    only module that owns the opponents config) so this stays adversary-free."""
    if not (strength_abort_enabled or robustness_abort_enabled or offwindow_monitor_enabled):
        return (
            "WARNING (D-EVALFOUND/coverage): no Objective-A signal active — SealBot-WR is "
            "demoted and the strength abort, robustness abort, and off-window robustness "
            "monitor are all OFF. A run can promote an off-window-exploitable checkpoint "
            "unguarded. Enable the off-window robustness monitor before a live run — see "
            "docs/designs/D_EVALFOUND_design.md §7."
        )
    return None


def check_robustness_warn(exploit_rate: float, cfg: MonitoringConfig) -> Optional[str]:
    """D-EVALFOUND — off-window exploitability WATCH. Single-point WARN (operator-routed)
    when the off-window forced-win rate exceeds the fix-acceptance bar. The robustness
    gate is the ONLY instrument that sees the off-window defect (vs-SealBot false-clears);
    by default it gates PROMOTE + emits this WARN, never auto-aborts."""
    if exploit_rate is None:
        return None
    if exploit_rate > cfg.robustness_warn_threshold:
        return (
            f"WARNING (D-EVALFOUND/robustness): off-window forced-win rate "
            f"{exploit_rate:.3f} > {cfg.robustness_warn_threshold:.3f} — exploitable "
            f"(blocks promotion; demoted SealBot-WR can't see this)"
        )
    return None


def check_robustness_abort(
    robustness_history: list[tuple[int, float]],
    current_step: int,
    cfg: MonitoringConfig,
) -> Optional[str]:
    """D-EVALFOUND — off-window exploitability HARD-ABORT (operator opt-in;
    ``robustness_abort_enabled`` default False = PROMOTE+WARN only).

    Fires when the off-window forced-win rate stays above ``robustness_warn_threshold``
    for ``robustness_abort_consecutive_evals`` consecutive evals past
    ``robustness_abort_min_step``. Takes NO cycle-density argument — robustness rejection
    is unambiguous and is NEVER cycle-suppressed (unlike the strength abort)."""
    if not cfg.robustness_abort_enabled or not robustness_history:
        return None
    n = int(cfg.robustness_abort_consecutive_evals)
    if current_step <= cfg.robustness_abort_min_step or len(robustness_history) < n:
        return None
    tail = robustness_history[-n:]
    if all(rate > cfg.robustness_warn_threshold for _, rate in tail):
        mean_rate = sum(r for _, r in tail) / len(tail)
        return (
            f"HARD-ABORT (D-EVALFOUND/robustness): off-window forced-win rate mean "
            f"{mean_rate:.3f} > {cfg.robustness_warn_threshold:.3f} for {n} consecutive "
            f"evals past step {cfg.robustness_abort_min_step:,}"
        )
    return None


def check_value_spread_canary(payload: dict) -> Optional[str]:
    """``value_spread`` payload — colony-capture canary (§S181 PR-A + PR-C / FU-1 / L48).

    Dual-bank discriminator. T3 anchor V_spread = +0.617; alt anchor = +0.212.
    L48 revised the magnitude: T3 amplifies ~3× vs the corpus-drawn alt bank.
    Gates (PR-C):
      * SOFT-ABORT: T3 < +0.20 OR alt < +0.07 (dual-bank colony capture).
      * WARNING:   T3 < +0.30 OR alt < +0.10 (discriminator degrading).
    Back-compat: single-bank payloads (only `spread` set) use the T3 gates.
    Canary only — the message routes to the operator, never auto-aborts.

    §S181-AUDIT Wave 3 L50: this canary is now INFORMATIONAL ONLY for
    real runs. The PRIMARY abort trigger is the sliding-window SealBot
    WR check (`check_sealbot_wr_hard_abort`).
    """
    t3 = payload.get("t3_spread", payload.get("spread"))
    alt = payload.get("alt_spread")

    def _bad(x: Any) -> bool:
        return (x is None or not isinstance(x, (int, float))
                or not math.isfinite(x))

    def _fmt(x: Any) -> str:
        return "nan" if _bad(x) else f"{x:+.3f}"

    if _bad(t3) and _bad(alt):
        return None

    t3_soft = (not _bad(t3)) and t3 < 0.20
    alt_soft = (not _bad(alt)) and alt < 0.07
    if t3_soft or alt_soft:
        return (f"SOFT-ABORT: V_spread T3={_fmt(t3)} alt={_fmt(alt)} — "
                "dual-bank colony capture (FU-2/L48 gate)")
    t3_warn = (not _bad(t3)) and t3 < 0.30
    alt_warn = (not _bad(alt)) and alt < 0.10
    if t3_warn or alt_warn:
        return (f"WARNING: V_spread T3={_fmt(t3)} alt={_fmt(alt)} — "
                "discriminator degrading")
    return None


# ── Aggregators ──────────────────────────────────────────────────────────


def evaluate_training_step_alerts(
    payload: dict, cfg: MonitoringConfig, loss_window: list
) -> list[str]:
    """Run every ``training_step`` rule, returning fired messages in order.

    The order matches the pre-extraction evaluation site so that
    duplicate-prefix de-duplication in ``TerminalDashboard._add_alert``
    sees alerts arrive in the same sequence.
    """
    out: list[str] = []
    for msg in (
        check_entropy_collapse(payload, cfg),
        check_selfplay_entropy_collapse(payload, cfg),
        check_grad_norm_spike(payload, cfg),
        check_loss_increase_window(loss_window, cfg),
    ):
        if msg is not None:
            out.append(msg)
    return out


def emit_training_step_alerts_headless(
    payload: dict, cfg: MonitoringConfig, loss_window, logger
) -> list[str]:
    """Fire the 4 training-step alerts HEADLESS via structlog (D-J DASH WP3).

    Replaces the display-only ``evaluate_training_step_alerts`` path that the
    retired terminal_dashboard (A2) owned. Same thresholds, same rules, same
    order — but the caller (step coordinator) invokes this every log_interval
    so the alerts keep firing with NO dashboard active. ``loss_window`` is the
    caller-owned deque (loss_total tail); this appends the current step's
    ``loss_total`` before running the window rule. Each fired alert is logged
    ``logger.warning("training_alert", rule=..., message=..., step=...)``.
    Returns the fired messages (for tests / any consumer).
    """
    lt = payload.get("loss_total")
    if lt is not None:
        loss_window.append(float(lt))
    rules = (
        ("entropy_collapse", check_entropy_collapse(payload, cfg)),
        ("selfplay_entropy_collapse", check_selfplay_entropy_collapse(payload, cfg)),
        ("grad_norm_spike", check_grad_norm_spike(payload, cfg)),
        ("loss_increase_window", check_loss_increase_window(loss_window, cfg)),
    )
    out: list[str] = []
    for rule, msg in rules:
        if msg is not None:
            out.append(msg)
            logger.warning(
                "training_alert", rule=rule, message=msg, step=payload.get("step")
            )
    return out


def evaluate_eval_complete_alerts(payload: dict) -> list[str]:
    """Run every ``eval_complete`` rule, returning fired messages."""
    out: list[str] = []
    msg = check_sealbot_gate_failed(payload)
    if msg is not None:
        out.append(msg)
    return out


def evaluate_value_spread_alerts(payload: dict) -> list[str]:
    """Run every ``value_spread`` rule, returning fired messages."""
    out: list[str] = []
    msg = check_value_spread_canary(payload)
    if msg is not None:
        out.append(msg)
    return out
