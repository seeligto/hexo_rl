#!/usr/bin/env python3
"""§S181 Track 4 — MCTS-in-loop probe implementations (standalone).

STANDALONE helper for the §S181 structural-diagnosis wave. Implements the
four MCTS-in-loop probes designed in
`audit/structural/04_probe_dashboard_redesign.md`:

    P1  W3S0 forced-win        (open-ended 5-in-row)
    P2  W3S1 forced-win        (rhombus / ladder)
    P3  threat-following       (single immediate threat)
    P4  anti-colony            (mature colony + periphery extension)

These probes measure net+MCTS *search behavior* — the root visit
distribution — NOT static logit reads. They are the leading-indicator
redesign for the C1-C4 threat-logit probes that PASSED through all four
colony collapses (§175 / §S179 / §S180a / §S180b).

DESIGN CONSTRAINT (per §S181 Track 4 brief): this file imports NOTHING
from the main `hexo_rl` / `engine` code paths. It is a self-contained
design skeleton — fixture loading and the net+MCTS call are stubbed
behind `--dry-run` and behind clearly marked integration points so the
file is runnable for threshold inspection without a built engine.

To wire against the real stack, fill the two integration points marked
`# INTEGRATION POINT` (model load + net+MCTS search). They are isolated
so the rest of the file — fixture schema, aggregation, thresholds,
verdict logic — needs no change.

Usage (dry-run, no engine required):
    python scripts/diagnosis/new_probes.py --probe p4 --dry-run

Usage (wired, after filling integration points):
    python scripts/diagnosis/new_probes.py \\
        --probe p4 --checkpoint archive/s180b_3knob_fail/ckpts/ckpt_step10k.pt

Retrospective validation target (§5 of the audit doc):
    for s in 10 20 30 40 50; do
      python scripts/diagnosis/new_probes.py --probe p4 \\
        --checkpoint archive/s180b_3knob_fail/ckpts/ckpt_step${s}k.pt
    done
    # expect colony_pull to cross 0.20 at step 10k → validates P4 as a
    # 40K-steps-early detector for the §S180b crash.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# ── Thresholds (from audit/structural/04_probe_dashboard_redesign.md §2) ──────

#: P1 — root visit fraction on winning 6th cells of an open-ended 5-in-row.
P1_W3S0_VISIT_FRAC_MIN: float = 0.90
#: P2 — root visit fraction on the W3S1→W3S0 conversion cell(s).
P2_W3S1_VISIT_FRAC_MIN: float = 0.70
#: P3 — root visit fraction on the threat-extending cell.
P3_THREAT_VISIT_FRAC_MIN: float = 0.55
#: P3 — argmax must land on the threat cell in >= this fraction of positions.
P3_THREAT_ARGMAX_RATE_MIN: float = 0.80
#: P4 — colony_pull = visit_frac(colony) - visit_frac(periphery).
#: <= 0.0 healthy; > P4 → FAIL, between 0 and P4 → WARNING.
P4_COLONY_PULL_MAX: float = 0.20

#: Eval-matched sim count (observed `model_sims: 128` in §S180b archive).
DEFAULT_SIMS: int = 128

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
FIXTURE_DIR: Path = REPO_ROOT / "fixtures" / "structural_probes"


# ── Fixture schema ────────────────────────────────────────────────────────────


@dataclass
class ProbePosition:
    """One curated probe position.

    `board_stones`: list of (q, r, player) — the position to evaluate.
    `side_to_move`: 1 or -1.
    `target_cells`: flat-action indices that a HEALTHY net+MCTS should
        concentrate visits on (winning cell / threat cell / W3S1→W3S0 cell).
    `colony_cells`: P4 only — flat indices adjacent to the mature colony
        (the cells a colony-captured net would over-visit).
    `phase`: free-text tag for reporting.
    """

    board_stones: list[tuple[int, int, int]]
    side_to_move: int
    target_cells: list[int]
    colony_cells: list[int] = field(default_factory=list)
    phase: str = "unknown"


@dataclass
class ProbeFixture:
    """A named set of probe positions for one probe (P1-P4)."""

    probe: str
    positions: list[ProbePosition]

    @classmethod
    def load(cls, probe: str) -> "ProbeFixture":
        """Load `fixtures/structural_probes/<probe>.json`.

        The fixture is intentionally JSON (not NPZ) so it is hand-editable —
        these are curated tactical positions, not generated tensors. Schema:
            {"probe": "p4",
             "positions": [
               {"board_stones": [[q,r,p],...], "side_to_move": 1,
                "target_cells": [..], "colony_cells": [..], "phase": ".."},
               ...]}
        """
        path = FIXTURE_DIR / f"{probe}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"probe fixture missing: {path}\n"
                f"Hand-build it per audit/structural/04_probe_dashboard_redesign.md §2."
            )
        raw = json.loads(path.read_text())
        positions = [
            ProbePosition(
                board_stones=[tuple(s) for s in p["board_stones"]],
                side_to_move=int(p["side_to_move"]),
                target_cells=list(p["target_cells"]),
                colony_cells=list(p.get("colony_cells", [])),
                phase=str(p.get("phase", "unknown")),
            )
            for p in raw["positions"]
        ]
        return cls(probe=raw["probe"], positions=positions)


# ── Integration points ────────────────────────────────────────────────────────


def load_net(checkpoint: Path):  # INTEGRATION POINT 1
    """Load the policy/value net from a checkpoint.

    To wire: delegate to `hexo_rl.training.checkpoints.load_inference_model`
    (the same loader `scripts/probe_threat_logits.py` uses). Kept stubbed
    here to honor the standalone constraint.
    """
    raise NotImplementedError(
        "INTEGRATION POINT 1 — wire to hexo_rl.training.checkpoints."
        "load_inference_model. Use --dry-run for threshold inspection."
    )


def run_mcts_visits(
    net,
    position: ProbePosition,
    sims: int,
) -> dict[int, int]:  # INTEGRATION POINT 2
    """Run net+MCTS from `position`; return {flat_action_idx: visit_count}.

    To wire: build a Board from `position.board_stones` /
    `position.side_to_move`, run the production MCTS (`engine` PyO3 binding)
    for `sims` simulations at the eval sim count, and read the ROOT child
    visit counts. This is the dynamic-equivariance measurement L2 requires —
    it must use the SAME net+MCTS stack as self-play / eval.
    """
    raise NotImplementedError(
        "INTEGRATION POINT 2 — wire to the engine MCTS PyO3 binding; "
        "return root child visit counts. Use --dry-run for inspection."
    )


# ── Aggregation helpers ───────────────────────────────────────────────────────


def _visit_frac(visits: dict[int, int], cells: list[int]) -> float:
    """Fraction of total root visits landing on `cells`."""
    total = sum(visits.values())
    if total == 0:
        return 0.0
    return sum(visits.get(c, 0) for c in cells) / total


def _argmax_cell(visits: dict[int, int]) -> Optional[int]:
    """Most-visited root child (the move MCTS would play)."""
    if not visits:
        return None
    return max(visits.items(), key=lambda kv: kv[1])[0]


# ── Probe runners (P1-P4) ─────────────────────────────────────────────────────


@dataclass
class ProbeResult:
    probe: str
    metric_name: str
    metric_value: float
    threshold: float
    verdict: str  # PASS / WARNING / FAIL
    per_position: list[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def _run_visit_concentration_probe(
    probe: str,
    metric_name: str,
    threshold: float,
    fixture: ProbeFixture,
    search: Callable[[ProbePosition], dict[int, int]],
    warn_band: bool = False,
) -> ProbeResult:
    """Shared runner for P1 / P2 / P3 — mean root visit fraction on target cells."""
    fracs: list[float] = []
    per_pos: list[dict] = []
    for pos in fixture.positions:
        visits = search(pos)
        f = _visit_frac(visits, pos.target_cells)
        argmax = _argmax_cell(visits)
        fracs.append(f)
        per_pos.append(
            {
                "phase": pos.phase,
                "visit_frac": round(f, 4),
                "argmax_on_target": argmax in pos.target_cells,
            }
        )
    mean_frac = sum(fracs) / max(len(fracs), 1)
    if mean_frac >= threshold:
        verdict = "PASS"
    elif warn_band and mean_frac >= 0.5 * threshold:
        verdict = "WARNING"
    else:
        verdict = "FAIL"
    return ProbeResult(
        probe=probe,
        metric_name=metric_name,
        metric_value=mean_frac,
        threshold=threshold,
        verdict=verdict,
        per_position=per_pos,
    )


def probe_p1_w3s0(fixture, search) -> ProbeResult:
    """P1 — open-ended 5-in-row; net+MCTS must visit-concentrate the 6th cell."""
    return _run_visit_concentration_probe(
        "p1", "w3s0_visit_frac", P1_W3S0_VISIT_FRAC_MIN, fixture, search
    )


def probe_p2_w3s1(fixture, search) -> ProbeResult:
    """P2 — rhombus/ladder; net+MCTS must value the W3S1→W3S0 conversion.

    Per Q12 §4.3: W3S1 is a forced win in the 2-stones-per-turn game but the
    quiescence override does NOT recognize it — so this is a pure search+net
    property, exactly the dynamic-equivariance L2 says static probes miss.
    """
    return _run_visit_concentration_probe(
        "p2", "w3s1_visit_frac", P2_W3S1_VISIT_FRAC_MIN, fixture, search
    )


def probe_p3_threat(fixture, search) -> ProbeResult:
    """P3 — single immediate threat; visit-frac + argmax-rate, WARNING band."""
    res = _run_visit_concentration_probe(
        "p3", "threat_follow_visit_frac", P3_THREAT_VISIT_FRAC_MIN,
        fixture, search, warn_band=True,
    )
    argmax_rate = (
        sum(1 for p in res.per_position if p["argmax_on_target"])
        / max(len(res.per_position), 1)
    )
    res.extra["threat_follow_argmax_rate"] = round(argmax_rate, 4)
    # P3 verdict downgrades to WARNING if argmax-rate also fails.
    if res.verdict == "PASS" and argmax_rate < P3_THREAT_ARGMAX_RATE_MIN:
        res.verdict = "WARNING"
    return res


def probe_p4_anticolony(fixture, search) -> ProbeResult:
    """P4 — mature colony + periphery seed; measure colony_pull.

    colony_pull = mean[ visit_frac(colony_cells) - visit_frac(periphery) ].
    <= 0.0 healthy; > P4_COLONY_PULL_MAX → FAIL; between → WARNING.
    Strongest leading indicator — the MCTS-visit analogue of `colony_a`.
    """
    pulls: list[float] = []
    per_pos: list[dict] = []
    for pos in fixture.positions:
        visits = search(pos)
        colony_f = _visit_frac(visits, pos.colony_cells)
        periphery_f = _visit_frac(visits, pos.target_cells)
        pull = colony_f - periphery_f
        pulls.append(pull)
        per_pos.append(
            {
                "phase": pos.phase,
                "colony_visit_frac": round(colony_f, 4),
                "periphery_visit_frac": round(periphery_f, 4),
                "colony_pull": round(pull, 4),
            }
        )
    mean_pull = sum(pulls) / max(len(pulls), 1)
    if mean_pull <= 0.0:
        verdict = "PASS"
    elif mean_pull <= P4_COLONY_PULL_MAX:
        verdict = "WARNING"
    else:
        verdict = "FAIL"
    return ProbeResult(
        probe="p4",
        metric_name="colony_pull",
        metric_value=mean_pull,
        threshold=P4_COLONY_PULL_MAX,
        verdict=verdict,
        per_position=per_pos,
    )


_PROBES: dict[str, Callable] = {
    "p1": probe_p1_w3s0,
    "p2": probe_p2_w3s1,
    "p3": probe_p3_threat,
    "p4": probe_p4_anticolony,
}


# ── Dry-run synthetic search (no engine required) ─────────────────────────────


def _dry_run_search(position: ProbePosition) -> dict[int, int]:
    """Synthetic visit counts for threshold-inspection without a built engine.

    Emits a deliberately HEALTHY distribution (visits concentrated on
    target_cells) so a dry-run prints PASS — confirming the verdict logic
    and aggregation. Real validation requires INTEGRATION POINT 2.
    """
    visits: dict[int, int] = {}
    for c in position.target_cells:
        visits[c] = 100
    for c in position.colony_cells:
        visits[c] = 5
    visits[-1] = 10  # arbitrary off-target spread
    return visits


# ── CLI ───────────────────────────────────────────────────────────────────────


def _emit(result: ProbeResult, as_json: bool) -> None:
    if as_json:
        print(json.dumps({
            "event": "probe_complete",
            "probe_name": result.probe,
            "probe_metric": result.metric_name,
            "metric_value": round(result.metric_value, 4),
            "threshold": result.threshold,
            "probe_verdict": result.verdict,
            "extra": result.extra,
        }))
        return
    print(f"# Probe {result.probe.upper()} — {result.metric_name}")
    print(f"verdict     : {result.verdict}")
    print(f"metric value: {result.metric_value:+.4f}")
    print(f"threshold   : {result.threshold:+.4f}")
    for k, v in result.extra.items():
        print(f"{k:12s}: {v}")
    print(f"positions   : {len(result.per_position)}")
    for i, p in enumerate(result.per_position):
        print(f"  [{i:2d}] {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description="§S181 MCTS-in-loop probes (P1-P4).")
    ap.add_argument("--probe", choices=sorted(_PROBES), required=True)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Checkpoint to probe (omit with --dry-run).")
    ap.add_argument("--sims", type=int, default=DEFAULT_SIMS,
                    help=f"MCTS sims per position (default {DEFAULT_SIMS}, eval-matched).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Use synthetic healthy search; no engine/checkpoint needed.")
    ap.add_argument("--json", action="store_true", help="Emit a probe_complete JSON event.")
    args = ap.parse_args()

    try:
        fixture = ProbeFixture.load(args.probe)
    except FileNotFoundError as e:
        if args.dry_run:
            # Synthesize a 1-position fixture so the verdict path is exercised.
            fixture = ProbeFixture(
                probe=args.probe,
                positions=[ProbePosition(
                    board_stones=[], side_to_move=1,
                    target_cells=[10, 11], colony_cells=[200, 201],
                    phase="synthetic")],
            )
            print(f"[dry-run] fixture missing — using 1 synthetic position", file=sys.stderr)
        else:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)

    if args.dry_run:
        search = _dry_run_search
    else:
        if args.checkpoint is None:
            print("ERROR: --checkpoint required without --dry-run", file=sys.stderr)
            sys.exit(2)
        net = load_net(args.checkpoint)  # INTEGRATION POINT 1
        search = lambda pos: run_mcts_visits(net, pos, args.sims)  # INTEGRATION POINT 2

    result = _PROBES[args.probe](fixture, search)
    _emit(result, as_json=args.json)
    # exit 0 PASS, 1 WARNING/FAIL — same convention as probe_threat_logits.py
    sys.exit(0 if result.verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
