"""D-EVALFOUND C2 — robustness gate primitive (Objective A: off-window exploitability).

Promotes the deterministic off-window adversary to a tracked, config-keyed gate. It is
the ONLY instrument that sees the off-window defect — vs-SealBot WR false-clears by
construction (§PRELONG-BRIDGE). A WATCH axis (gates PROMOTE + emits WARN), NOT a strength
meter and NOT cycle-suppressed (see the abort matrix in alert_rules).

Pre-registered thresholds (exploit_probe.py:124, locked 2026-06-06):
  DEFENDED ≤ 0.05 ; FORCEABLE ≥ 0.15 AND (exploit − control) ≥ 0.10 ; else INCONCLUSIVE.
Gate pass bar = exploit off-window forced-win rate ≤ ``threshold`` (0.06 fix-acceptance;
deployed single-window v6_live2 ≈ 0.235 = the defect). Thresholds are tied to
n=200/arm, sims=128, opening_plies=6 — the gate FIXES these.

Single source: the measurement reuses ``offwindow_probe.run_adversary_games`` (no drift).
Design: docs/designs/D_EVALFOUND_design.md §1b.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional

from hexo_rl.eval.gate_logic import _binomial_ci


@dataclass(frozen=True)
class RobustnessGateConfig:
    """Config-keyed gate knobs (no literals). ``encoding`` pins the expected encoding;
    a loaded-checkpoint mismatch is a hard error (off-window boundary is
    policy_logit_count-specific) unless ``force_spec_mismatch``."""
    enabled: bool = False
    arm: str = "exploit"
    control_arm: str = "control"
    threshold: float = 0.06        # fix-acceptance; gate passes iff exploit_rate <= threshold
    defended_max: float = 0.05     # DEFENDED label bound (pre-reg lock)
    forceable_min: float = 0.15    # FORCEABLE label bound (pre-reg lock)
    margin_min: float = 0.10       # FORCEABLE requires (exploit - control) >= this
    n_per_arm: int = 200
    sims: int = 128
    opening_plies: int = 6
    seed_base: int = 7000
    encoding: Optional[str] = None
    force_spec_mismatch: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RobustnessGateConfig":
        """Build from a config dict, rejecting unknown keys (config-first discipline:
        a typo must fail loud, not silently no-op)."""
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"RobustnessGateConfig: unknown config key(s) {sorted(unknown)}; "
                f"known keys: {sorted(known)}"
            )
        return cls(**d)


def classify_verdict(
    exploit_rate: float,
    control_rate: float,
    *,
    defended_max: float = 0.05,
    forceable_min: float = 0.15,
    margin_min: float = 0.10,
) -> str:
    """Pre-registered diagnostic label (exploit_probe.py:124 semantics)."""
    if exploit_rate <= defended_max:
        return "DEFENDED"
    if exploit_rate >= forceable_min and (exploit_rate - control_rate) >= margin_min:
        return "FORCEABLE"
    return "INCONCLUSIVE"


def gate_passes(exploit_rate: float, threshold: float) -> bool:
    """The PROMOTE-gate decision: pass iff the off-window forced-win rate is at or
    below the acceptance bar."""
    return exploit_rate <= threshold


def evaluate_robustness_gate(
    model_player: Any,
    encoding: str,
    spec: Any,
    cfg: RobustnessGateConfig,
    legal_move_radius: int | None = None,
) -> Dict[str, Any]:
    """Run the off-window adversary (exploit + control arms) vs ``model_player`` and
    return the gate verdict. ``encoding`` is the checkpoint's loaded label; a mismatch
    with ``cfg.encoding`` is a hard error unless ``cfg.force_spec_mismatch``.

    Measurement reuses ``offwindow_probe.run_adversary_games`` (single source).
    ``legal_move_radius`` (D-SHRIMP S4b) keeps this offline/standalone gate on the same
    curriculum-radius schema as the in-loop paths — pass the run's current radius when a
    caller wires it in (no in-loop caller today; defaults to the registry radius).
    """
    if cfg.encoding is not None and cfg.encoding != encoding and not cfg.force_spec_mismatch:
        raise ValueError(
            f"robustness gate configured for encoding '{cfg.encoding}' but checkpoint "
            f"loaded as '{encoding}'; off-window boundary is policy_logit_count-specific "
            f"— re-run exploit_probe on this checkpoint, or set force_spec_mismatch=True"
        )
    from hexo_rl.eval.offwindow_probe import run_adversary_games

    summaries: Dict[str, dict] = {}
    for arm in (cfg.arm, cfg.control_arm):
        summary, _recs = run_adversary_games(
            model_player, encoding, spec, arm, cfg.n_per_arm, cfg.sims,
            opening_plies=cfg.opening_plies, seed_base=cfg.seed_base,
            legal_move_radius=legal_move_radius,
        )
        summaries[arm] = summary

    exploit_rate = float(summaries[cfg.arm]["off_window_forced_win_rate"])
    control_rate = float(summaries[cfg.control_arm]["off_window_forced_win_rate"])
    exploit_wins = round(exploit_rate * cfg.n_per_arm)
    ci_lo, ci_hi = _binomial_ci(exploit_wins, cfg.n_per_arm)

    verdict = classify_verdict(
        exploit_rate, control_rate,
        defended_max=cfg.defended_max, forceable_min=cfg.forceable_min, margin_min=cfg.margin_min,
    )
    passed = gate_passes(exploit_rate, cfg.threshold)
    return {
        "encoding": encoding,
        "arm": cfg.arm,
        "exploit_off_window_forced_win_rate": round(exploit_rate, 4),
        "control_off_window_forced_win_rate": round(control_rate, 4),
        "margin": round(exploit_rate - control_rate, 4),
        "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4),
        "threshold": cfg.threshold,
        "verdict": verdict,
        "gate_passed": passed,
        "n_per_arm": cfg.n_per_arm,
        "sims": cfg.sims,
        "opening_plies": cfg.opening_plies,
    }
