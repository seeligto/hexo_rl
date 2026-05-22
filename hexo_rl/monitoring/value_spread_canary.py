"""§S181 PR-A — value-spread canary.

First-class colony-capture canary. Replaces `colony_a` as the *trigger*
signal (FU-1 §5: `colony_a` is a wins-only, late, small-denominator ratio;
the value-spread probe is a static 40-position forward with a stable
denominator that crosses the abort gate ~30-40k steps earlier).

Mechanism (§S181-T3 / FU-1). A healthy value head discriminates colony
positions from open-extension positions:

    V_spread = mean V(colony bank) - mean V(extension bank)

The `bootstrap_model_v6.pt` anchor scores V_spread = +0.617. Across the
§S180b colony-captured trajectory it collapses to ~0 by step 20k — the
value head can no longer tell a dead blob from a winning open run. FU-1
pinned the +0.20 line as the abort gate.

This module forwards the value head on the frozen 40-position T3 bank
(`tests/fixtures/value_spread_bank.json`, SHA-pinned) on every checkpoint
save and emits a `value_spread` dashboard event. It is a CANARY, not a
kill switch — it surfaces WARNING / SOFT-ABORT to the operator and never
aborts training itself.

INSPECTION-ONLY. Imports the compiled `engine` bindings + the
`LocalInferenceEngine` thin wrapper — no hot-path edits, no `scripts/`
imports. The forward runs under `torch.no_grad()`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import structlog
import torch

log = structlog.get_logger(__name__)

# Repo-root-relative default fixture path.
_REPO = Path(__file__).resolve().parents[2]
_FIXTURE_PATH = _REPO / "tests" / "fixtures" / "value_spread_bank.json"

# FU-1 anchor bank SHA — the fixture MUST hash to this. Drift = STOP.
BANK_SHA256 = "934204713620d171743820aea6907cf4e117ca97c69e50052b991a3fdcc23991"

# FU-1 / skeleton FU-2 gate. Anchor V_spread = +0.617; healthy stays high.
WARN_THRESHOLD = 0.30        # WARNING below this
SOFT_ABORT_THRESHOLD = 0.20  # SOFT-ABORT signal below this (FU-2 abort gate)


# ── Bank ─────────────────────────────────────────────────────────────────
@dataclass
class CanaryResult:
    """One value-spread measurement over the 40-position bank."""

    mean_colony: float
    mean_ext: float
    spread: float
    sd_col: float
    sd_ext: float
    n: int
    n_colony: int
    n_extension: int

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


class _Bank:
    """Realized 40-position bank: colony/extension Boards + SHA identity."""

    def __init__(self, positions: list[dict], sha: str) -> None:
        # Lazy engine import — keeps the dashboard process (which never
        # touches the canary) free of the compiled extension.
        from engine import Board

        self.sha = sha
        self.boards: list = []
        self.classes: list[str] = []
        for spec in positions:
            b = Board()
            ok = True
            for q, r in spec["moves"]:
                try:
                    b.apply_move(int(q), int(r))
                except Exception:
                    ok = False
                    break
            if not ok:
                # A position that no longer realizes legally → fixture/engine
                # drift. Surface loudly; the SHA pin should already catch it.
                raise ValueError(
                    f"value-spread bank position {spec['name']!r} did not "
                    "realize to a legal board"
                )
            self.boards.append(b)
            self.classes.append(spec["pos_class"])
        self.n_colony = sum(1 for c in self.classes if c == "colony")
        self.n_extension = sum(1 for c in self.classes if "extension" in c)


def _bank_sha(positions: list[dict]) -> str:
    """Recompute the bank SHA — identical scope to
    `fu1_value_spread_ladder.bank_fixture_sha` (name + class + moves)."""
    h = hashlib.sha256()
    for spec in positions:
        h.update(spec["name"].encode())
        h.update(spec["pos_class"].encode())
        for q, r in spec["moves"]:
            h.update(f"{int(q)},{int(r)};".encode())
    return h.hexdigest()


def load_bank(fixture_path: Optional[Path | str] = None) -> _Bank:
    """Load the JSON fixture, verify the SHA pin, realize all 40 Boards.

    Raises if the fixture is missing or its SHA does not match the FU-1
    anchor `BANK_SHA256` — a corrupt or drifted bank must not silently
    feed a meaningless canary.
    """
    path = Path(fixture_path) if fixture_path is not None else _FIXTURE_PATH
    if not path.exists():
        raise FileNotFoundError(f"value-spread bank fixture missing at {path}")
    data = json.loads(path.read_text())
    positions = data["positions"]
    sha = _bank_sha(positions)
    if sha != BANK_SHA256:
        raise ValueError(
            f"value-spread bank SHA {sha} != pinned {BANK_SHA256} — fixture "
            "drifted from the FU-1 anchor. STOP."
        )
    return _Bank(positions, sha)


# ── Measurement ──────────────────────────────────────────────────────────
def compute_value_spread(
    net: torch.nn.Module,
    bank: _Bank,
    device: Optional[torch.device] = None,
) -> CanaryResult:
    """Forward the value head once over the bank; return the spread.

    Uses the same `LocalInferenceEngine` path FU-1 used, so the anchor
    reproduces V_spread = +0.617. Runs under `torch.no_grad()`; restores
    the model's prior train/eval mode (LocalInferenceEngine forces eval()).
    """
    from hexo_rl.selfplay.inference import LocalInferenceEngine

    dev = device if device is not None else torch.device("cpu")
    was_training = net.training
    try:
        eng = LocalInferenceEngine(net, dev)
        with torch.no_grad():
            _policies, values = eng.infer_batch(bank.boards)
    finally:
        if was_training:
            net.train()

    vals = np.asarray(values, dtype=np.float64)
    classes = np.asarray(bank.classes)
    col = vals[classes == "colony"]
    ext = vals[np.char.find(classes.astype(str), "extension") >= 0]

    mc = float(col.mean()) if col.size else float("nan")
    me = float(ext.mean()) if ext.size else float("nan")
    return CanaryResult(
        mean_colony=round(mc, 4),
        mean_ext=round(me, 4),
        spread=round(mc - me, 4),
        sd_col=round(float(col.std(ddof=0)), 4) if col.size else 0.0,
        sd_ext=round(float(ext.std(ddof=0)), 4) if ext.size else 0.0,
        n=int(vals.size),
        n_colony=int(col.size),
        n_extension=int(ext.size),
    )


# ── Checkpoint-save hook ─────────────────────────────────────────────────
_BANK_CACHE: Optional[_Bank] = None
_BANK_LOAD_FAILED = False


def fire_canary(
    model: torch.nn.Module,
    step: int,
    device: Optional[torch.device] = None,
) -> Optional[CanaryResult]:
    """Run the canary on a checkpoint save: measure, emit, alert.

    Fire-and-forget — never raises. A canary or fixture failure disables
    the canary for the run and logs once; training is unaffected.
    Returns the `CanaryResult` (handy for tests) or None on failure.
    """
    global _BANK_CACHE, _BANK_LOAD_FAILED
    if _BANK_LOAD_FAILED:
        return None
    try:
        if _BANK_CACHE is None:
            _BANK_CACHE = load_bank()
        result = compute_value_spread(model, _BANK_CACHE, device)
    except Exception as exc:  # noqa: BLE001 — canary must never break training
        _BANK_LOAD_FAILED = True
        log.warning("value_spread_canary_failed", step=step, error=str(exc))
        return None

    # Emit the dashboard event — emit_event itself never raises.
    try:
        from hexo_rl.monitoring.events import emit_event

        emit_event({
            "event": "value_spread",
            "step": step,
            **result.to_payload(),
            "warn_threshold": WARN_THRESHOLD,
            "soft_abort_threshold": SOFT_ABORT_THRESHOLD,
        })
    except Exception as exc:  # noqa: BLE001
        log.warning("value_spread_emit_failed", step=step, error=str(exc))

    # Surface the alert to the operator via structlog (canary, not killer).
    from hexo_rl.monitoring.alert_rules import check_value_spread_canary

    msg = check_value_spread_canary({"spread": result.spread})
    if msg is not None:
        log.warning("value_spread_alert", step=step, spread=result.spread,
                     alert=msg)
    else:
        log.info("value_spread_canary", step=step, spread=result.spread,
                 mean_colony=result.mean_colony, mean_ext=result.mean_ext)
    return result
