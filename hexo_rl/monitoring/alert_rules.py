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


def check_value_spread_canary(payload: dict) -> Optional[str]:
    """``value_spread`` payload — colony-capture canary (§S181 PR-A + PR-C / FU-1 / L48).

    Dual-bank discriminator. T3 anchor V_spread = +0.617; alt anchor = +0.212.
    L48 revised the magnitude: T3 amplifies ~3× vs the corpus-drawn alt bank.
    Gates (PR-C):
      * SOFT-ABORT: T3 < +0.20 OR alt < +0.07 (dual-bank colony capture).
      * WARNING:   T3 < +0.30 OR alt < +0.10 (discriminator degrading).
    Back-compat: single-bank payloads (only `spread` set) use the T3 gates.
    Canary only — the message routes to the operator, never auto-aborts.
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
