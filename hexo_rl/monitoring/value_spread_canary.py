"""§S181 PR-A — value-spread canary. PR-C: dual-bank (T3 + alt per L48).

First-class colony-capture canary. Replaces `colony_a` as the *trigger*
signal (FU-1 §5: `colony_a` is a wins-only, late, small-denominator ratio;
the value-spread probe is a static 40-position forward with a stable
denominator that crosses the abort gate ~30-40k steps earlier).

Mechanism (§S181-T3 / FU-1). A healthy value head discriminates colony
positions from open-extension positions:

    V_spread = mean V(colony bank) - mean V(extension bank)

The `bootstrap_model_v6.pt` anchor scores T3 V_spread = +0.617 (synthetic
T3 builder positions). Across the §S180b colony-captured trajectory it
collapses to ~0 by step 20k — the value head can no longer tell a dead
blob from a winning open run. FU-1 pinned the +0.20 line as the T3 abort
gate.

PR-C / L48 (`docs/07_PHASE4_SPRINT_LOG.md` §S181-AUDIT). Track A A3
confirmed T3 amplifies ~3× vs an alt bank drawn from real bot-corpus
positions (Pearson r=0.27). Original A3 alt bank was 8-plane v6.

WP3-C2 (2026-07-10): alt bank rebuilt under v6_live2_ls (4-plane) from
`bootstrap_corpus_v6_live2_ls.npz` — original 8-plane bank caused
in_channels=4 != alt_planes=8 → 227/227 skip (all NaN) across run2.
New alt anchor V_spread = +0.292 (run2_bootstrap_v6_live2_ls.pt, step 0).
Alt gates (percentile-basis: step-0 anchor only; calibrate on run2 series
once it emits): WARN < +0.10, SOFT-ABORT < +0.07. Gates retained from
A3 calibration — new anchor (+0.292) is comfortably above both gates.
The dual-bank canary computes BOTH and SOFT-ABORTs if either bank crosses.

This module forwards the value head on:
  * the frozen 40-position T3 bank (`tests/fixtures/value_spread_bank.json`)
  * the frozen 40-position alt bank (`tests/fixtures/value_spread_bank_alt.json`)
on every checkpoint save and emits a `value_spread` dashboard event. It
is a CANARY, not a kill switch — it surfaces WARNING / SOFT-ABORT to the
operator and never aborts training itself.

INSPECTION-ONLY. Imports the compiled `engine` bindings + the
`LocalInferenceEngine` thin wrapper for the T3 bank; the alt bank is a
direct `net(state)` forward (matches A3's reproduction path). No hot-path
edits, no `scripts/` imports. The forward runs under `torch.no_grad()`.
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

# Repo-root-relative default fixture paths.
_REPO = Path(__file__).resolve().parents[2]
_FIXTURE_PATH = _REPO / "tests" / "fixtures" / "value_spread_bank.json"
_ALT_FIXTURE_PATH = _REPO / "tests" / "fixtures" / "value_spread_bank_alt.json"

# FU-1 anchor bank SHA — the fixture MUST hash to this. Drift = STOP.
BANK_SHA256 = "934204713620d171743820aea6907cf4e117ca97c69e50052b991a3fdcc23991"
# PR-C / L48 — A3 alt bank fixture SHA (`meta.sha256`). Drift = STOP.
ALT_BANK_SHA256 = "e01ff810805c26aca0deccd4994a2537df7bbbd259f3c7cfe31dc6529f908147"

# FU-1 / skeleton FU-2 gate. T3 anchor V_spread = +0.617; healthy stays high.
WARN_THRESHOLD = 0.30        # WARNING below this (T3)
SOFT_ABORT_THRESHOLD = 0.20  # SOFT-ABORT signal below this (T3 FU-2 abort gate)
# PR-C / L48 — alt bank gates (A3 calibration: T3/3). WP3-C2: bank rebuilt
# under v6_live2_ls (4-plane); new anchor V_spread = +0.292 at step 0. Gates
# retained — anchor is 4.2× above abort gate (0.292/0.07); conservative hold.
# Re-calibrate once run2 emits a spread series (percentile-based, not imported).
ALT_WARN_THRESHOLD = 0.10
ALT_SOFT_ABORT_THRESHOLD = 0.07


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
    encoding_spec: Optional[Any] = None,
) -> CanaryResult:
    """Forward the value head once over the bank; return the spread.

    Uses the same `LocalInferenceEngine` path FU-1 used, so the anchor
    reproduces V_spread = +0.617. Runs under `torch.no_grad()`; restores
    the model's prior train/eval mode (LocalInferenceEngine forces eval()).

    `encoding_spec` selects the wire-plane slice the engine applies (v6 → 8
    planes, v6tp → 10 incl. turn-phase 16/17). Defaults to v6 inside the
    engine when None — correct for the 8-plane families this bank anchors.
    """
    from hexo_rl.selfplay.inference import LocalInferenceEngine

    dev = device if device is not None else torch.device("cpu")
    was_training = net.training
    try:
        eng = LocalInferenceEngine(net, dev, encoding_spec=encoding_spec)
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


# ── Alt bank (PR-C / L48) ────────────────────────────────────────────────
@dataclass
class _AltBank:
    """Realized 40-position alt bank: pre-built (C, 19, 19) state tensors
    drawn from real bot-corpus positions (A3 builder). WP3-C2: rebuilt under
    v6_live2_ls — C=4 (was C=8 for v6, which caused 227/227 skip in run2)."""

    states: np.ndarray   # (40, C, 19, 19) float32 — C matches live encoding
    classes: np.ndarray  # (40,) object/str — "colony" | "extension"
    sha: str


def _alt_bank_sha(positions: list[dict]) -> str:
    """A3 SHA scope (name + class + state.tobytes()) — different from T3
    (which hashes the move sequence). Matches `a3_h_bank.bank_sha`."""
    h = hashlib.sha256()
    for spec in positions:
        h.update(spec["name"].encode())
        h.update(spec["pos_class"].encode())
        arr = np.asarray(spec["state"], dtype=np.float32)
        h.update(arr.tobytes())
    return h.hexdigest()


def load_alt_bank(fixture_path: Optional[Path | str] = None) -> _AltBank:
    """Load + SHA-verify the alt bank fixture. Drift = STOP."""
    path = Path(fixture_path) if fixture_path is not None else _ALT_FIXTURE_PATH
    if not path.exists():
        raise FileNotFoundError(f"value-spread alt bank fixture missing at {path}")
    data = json.loads(path.read_text())
    positions = data["positions"]
    sha = _alt_bank_sha(positions)
    if sha != ALT_BANK_SHA256:
        raise ValueError(
            f"value-spread alt bank SHA {sha} != pinned {ALT_BANK_SHA256} — "
            "fixture drifted from the A3 anchor. STOP."
        )
    states = np.stack([np.asarray(p["state"], dtype=np.float32)
                       for p in positions])
    classes = np.array([p["pos_class"] for p in positions])
    return _AltBank(states=states, classes=classes, sha=sha)


def compute_value_spread_alt(
    net: torch.nn.Module,
    bank: _AltBank,
    device: Optional[torch.device] = None,
) -> CanaryResult:
    """Direct `net(state)` forward over the alt bank — matches A3's
    reproduction path (`a3_h_bank.forward_value`). Returns the spread."""
    dev = device if device is not None else torch.device("cpu")
    was_training = net.training
    net.eval()
    try:
        x = torch.from_numpy(bank.states).to(dev)
        with torch.no_grad():
            _logp, value, _vlogit = net(x)
        vals = value.squeeze(-1).cpu().float().numpy().astype(np.float64)
    finally:
        if was_training:
            net.train()

    col = vals[bank.classes == "colony"]
    ext = vals[bank.classes == "extension"]
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


# ── Dual canary (PR-C) ───────────────────────────────────────────────────
@dataclass
class DualCanaryResult:
    """T3 + alt V_spread measured together (PR-C / L48).

    PASS = T3 ≥ SOFT_ABORT_THRESHOLD AND alt ≥ ALT_SOFT_ABORT_THRESHOLD.
    Either failing flips `both_pass=False` → SOFT-ABORT signal.
    """

    t3_spread: float
    t3_components: dict[str, float]
    alt_spread: float
    alt_components: dict[str, float]
    both_pass: bool

    def to_payload(self) -> dict[str, Any]:
        return dict(
            t3_spread=self.t3_spread,
            t3_components=self.t3_components,
            alt_spread=self.alt_spread,
            alt_components=self.alt_components,
            both_pass=self.both_pass,
            # Back-compat with pre-PR-C single-bank renderers/tests.
            spread=self.t3_spread,
            mean_colony=self.t3_components["mean_colony"],
            mean_ext=self.t3_components["mean_ext"],
        )


def _components(r: CanaryResult) -> dict[str, float]:
    return dict(
        mean_colony=r.mean_colony, mean_ext=r.mean_ext,
        sd_col=r.sd_col, sd_ext=r.sd_ext,
        n=r.n, n_colony=r.n_colony, n_extension=r.n_extension,
    )


def _net_in_channels(net: torch.nn.Module) -> Optional[int]:
    base = getattr(net, "_orig_mod", net)
    return getattr(base, "in_channels", None)


def compute_value_spread_dual(
    net: torch.nn.Module,
    device: Optional[torch.device] = None,
    t3_bank: Optional[_Bank] = None,
    alt_bank: Optional[_AltBank] = None,
    encoding_spec: Optional[Any] = None,
) -> DualCanaryResult:
    """Forward once over each bank; return dual result + PASS verdict.

    T3 bank routes board → to_tensor → engine slice, so it works for any
    encoding given `encoding_spec`. The alt bank stores PRE-BAKED state
    tensors at a fixed plane count (the v6/8-plane FU-1/A3 anchor); it is
    fed directly into `net()` and so is inapplicable when the model's
    in_channels differ (e.g. v6tp's 10). In that case the alt bank is
    skipped (NaN, not a failure) and the verdict rests on T3 — the
    canonical +0.617 anchor metric — rather than killing the whole canary.
    """
    t3 = t3_bank if t3_bank is not None else load_bank()
    alt = alt_bank if alt_bank is not None else load_alt_bank()
    t3_r = compute_value_spread(net, t3, device, encoding_spec=encoding_spec)

    _in_ch = _net_in_channels(net)
    _alt_planes = int(alt.states.shape[1]) if alt.states.ndim >= 2 else None
    alt_applicable = _in_ch is None or _alt_planes is None or _in_ch == _alt_planes
    if alt_applicable:
        alt_r = compute_value_spread_alt(net, alt, device)
        alt_spread = alt_r.spread
        alt_components = _components(alt_r)
        both_pass = (t3_r.spread >= SOFT_ABORT_THRESHOLD
                     and alt_r.spread >= ALT_SOFT_ABORT_THRESHOLD)
    else:
        # Alt fixture plane count (e.g. 8) != model in_channels (e.g. v6tp 10).
        # Skip alt; verdict rests on T3 only.
        log.info("value_spread_alt_skipped_plane_mismatch",
                 alt_planes=_alt_planes, model_in_channels=_in_ch)
        alt_spread = float("nan")
        alt_components = {
            "mean_colony": float("nan"), "mean_ext": float("nan"),
            "sd_col": 0.0, "sd_ext": 0.0, "n": 0, "n_colony": 0, "n_extension": 0,
        }
        both_pass = t3_r.spread >= SOFT_ABORT_THRESHOLD
    return DualCanaryResult(
        t3_spread=t3_r.spread, t3_components=_components(t3_r),
        alt_spread=alt_spread, alt_components=alt_components,
        both_pass=both_pass,
    )


# ── Checkpoint-save hook ─────────────────────────────────────────────────
_BANK_CACHE: Optional[_Bank] = None
_ALT_BANK_CACHE: Optional[_AltBank] = None
_BANK_LOAD_FAILED = False


def fire_canary(
    model: torch.nn.Module,
    step: int,
    device: Optional[torch.device] = None,
    encoding: Any = None,
) -> Optional[DualCanaryResult]:
    """Run the dual-bank canary on a checkpoint save: measure, emit, alert.

    Fire-and-forget — never raises. A canary or fixture failure disables
    the canary for the run and logs once; training is unaffected.
    Returns the `DualCanaryResult` (handy for tests) or None on failure.

    `encoding` (str or {"version": ...} dict) selects the T3 bank's wire-plane
    slice so non-v6 encodings (e.g. v6tp's 10 planes) forward correctly;
    None defaults to v6 inside the engine.
    """
    global _BANK_CACHE, _ALT_BANK_CACHE, _BANK_LOAD_FAILED
    if _BANK_LOAD_FAILED:
        return None
    try:
        _spec = None
        if encoding is not None:
            from hexo_rl.encoding import (
                lookup as _lookup_encoding,
                normalize_encoding_name as _normalize_encoding_name,
            )
            _spec = _lookup_encoding(_normalize_encoding_name(encoding))
        if _BANK_CACHE is None:
            _BANK_CACHE = load_bank()
        if _ALT_BANK_CACHE is None:
            _ALT_BANK_CACHE = load_alt_bank()
        result = compute_value_spread_dual(
            model, device, t3_bank=_BANK_CACHE, alt_bank=_ALT_BANK_CACHE,
            encoding_spec=_spec,
        )
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
            "alt_warn_threshold": ALT_WARN_THRESHOLD,
            "alt_soft_abort_threshold": ALT_SOFT_ABORT_THRESHOLD,
        })
    except Exception as exc:  # noqa: BLE001
        log.warning("value_spread_emit_failed", step=step, error=str(exc))

    # Surface the alert to the operator via structlog (canary, not killer).
    from hexo_rl.monitoring.alert_rules import check_value_spread_canary

    msg = check_value_spread_canary(result.to_payload())
    if msg is not None:
        log.warning("value_spread_alert", step=step,
                    t3_spread=result.t3_spread, alt_spread=result.alt_spread,
                    both_pass=result.both_pass, alert=msg)
    else:
        log.info("value_spread_canary", step=step,
                 t3_spread=result.t3_spread, alt_spread=result.alt_spread,
                 both_pass=result.both_pass)
    return result
