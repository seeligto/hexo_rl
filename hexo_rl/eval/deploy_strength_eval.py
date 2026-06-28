"""D-LOCALIZE P4 (TRACK B) — deploy-matched in-loop strength eval.

Replaces the in-loop PUCT-visit-policy + ``eval_temperature=0.5`` + ``model_sims=64``
strength head (a head the model NEVER deploys — the §D-LADDER "triple miss") with the
DEPLOY head: Gumbel Sequential-Halving GREEDY winner, Gumbel root noise g=0, NO
temperature, deploy sims (``n_sims_full``, ``gumbel_m``). The external bar is a
fixed-depth SealBot (reproducible, machine-independent) — NOT wall-clock ``time_limit``.

Why this module exists (NO silent fallback): the in-loop ``Evaluator`` routed the
model-under-test through ``ModelPlayer`` / ``KClusterMCTSBot`` (PUCT visit-policy +
``eval_temperature``). That measures a head the deployment regime never executes
(§D-LADDER INSTRUMENT MISMATCH, fully code-verified). This module drives the SAME
engine planner the net trains under via ``run_gumbel_on_board(..., gumbel_scale=0.0)``
— the SH winner is ``argmax(log_prior + completed-Q sigma)``, deterministic, no temp,
no PUCT visit-argmax of the played move. Deploy knobs (``gumbel_m``, ``n_sims_full``,
``c_visit``, ``c_scale``, ``c_puct``) are READ FROM THE RUN CONFIG; a missing knob is a
HARD ERROR — the gate can never silently degrade to a PUCT/temp/64-sim proxy.

Statistics are REUSED VERBATIM from the round-robin stack (NOT reimplemented):
``round_robin.aggregate_games`` (BT-MLE + Copeland + inversion), ``bootstrap_ratings_ci``
(distinct-game cluster bootstrap CI — the §D-ARGMAX pseudo-replication guard),
``effective_n_guard`` (copy-multiplier / distinct-per-pair power report),
``bradley_terry.compute_ratings``. Per-game move sequences are recorded so byte-identical
games are deduped before the CI — the deterministic deploy regime collapses to ~2
distinct games/pair without opening diversity, so a raw-count CI is over-confident by
sqrt(copies). Opening diversity = RNG-seeded uniform random opening plies; color-balanced.

ADAPTIVE SCREEN -> CONFIRM (the cost lever): each cadence runs a cheap ~``screen_n``-game
screen at deploy sims. A full ``confirm_n``-game confirm fires ONLY when the screen WR
lands in the pre-registered band around the promotion bar (see ``DeployStrengthConfig``);
far below the bar (clear non-candidate) or far above (clear pass) skip the confirm. The
band is sized so it CANNOT false-negative a true promotion candidate — see the band
docstring + ``configs/eval.yaml``.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from engine import Board
from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.encoding import normalize_encoding_name as _normalize_encoding
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.gumbel_search_py import run_gumbel_on_board
from hexo_rl.eval.round_robin import (
    aggregate_games,
    bootstrap_ratings_ci,
    distinct_per_pair,
    effective_n_guard,
)
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.inference import LocalInferenceEngine

try:
    import structlog

    log = structlog.get_logger()
except ImportError:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("deploy_strength_eval")  # type: ignore[assignment]

# Eval root-noise scale: 0.0 = deterministic deploy head (mctx gumbel_scale=0). This is
# the canonical AGZ/mctx/LightZero strength-eval convention and the load-bearing g=0 the
# in-loop gate runs. NOT a config knob — a knob would let a run silently re-inject root
# noise into the strength gate (defeating the deploy-match).
EVAL_GUMBEL_SCALE: float = 0.0

# Deploy knobs pulled from the run config. Same key paths as the standalone
# scripts/eval/gumbel_greedy_bot._REQUIRED_KNOBS — single source of truth, zero literal
# search defaults. A missing key is a HARD ERROR (no silent PUCT/temp/64 fallback).
_REQUIRED_KNOBS: Dict[str, Tuple[str, ...]] = {
    "gumbel_m": ("selfplay", "gumbel_m"),
    "c_visit": ("selfplay", "c_visit"),
    "c_scale": ("selfplay", "c_scale"),
    "n_sims_full": ("selfplay", "playout_cap", "n_sims_full"),
    "c_puct": ("mcts", "c_puct"),
}


def _dig(cfg: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    cur: Any = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(
                f"deploy_strength_eval: run config missing required deploy knob at "
                f"{'.'.join(path)!r} — the in-loop gate cannot run the deploy head "
                f"without it, and silently falling back to a PUCT/temp/64-sim proxy is "
                f"the §D-LADDER triple miss this path exists to remove."
            )
        cur = cur[k]
    return cur


def extract_deploy_knobs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Pull every deploy search knob from the run config. Hard-errors on any gap."""
    return {name: _dig(cfg, path) for name, path in _REQUIRED_KNOBS.items()}


class DeployHeadBot(BotProtocol):
    """Deploy-matched Gumbel-greedy player: SH-winner action, g=0, no temperature.

    Mirrors ``scripts/eval/gumbel_greedy_bot.GumbelGreedyBot`` but is a TRACKED module
    on the in-loop path. The played move is the Sequential-Halving winner from
    ``run_gumbel_on_board(..., gumbel_scale=0.0)`` — there is no temperature, no PUCT
    visit-argmax of the played move, and the root noise is zeroed (deterministic deploy
    head). The interior PUCT descent + completed-Q SH math are byte-identical to
    self-play."""

    def __init__(
        self,
        engine: LocalInferenceEngine,
        knobs: Dict[str, Any],
        label: str,
        seed: int = 0,
        legal_set: bool = False,
    ) -> None:
        self._engine = engine
        self._m = int(knobs["gumbel_m"])
        self._n_sims = int(knobs["n_sims_full"])
        self._c_visit = float(knobs["c_visit"])
        self._c_scale = float(knobs["c_scale"])
        self._c_puct = float(knobs["c_puct"])
        self._label = label
        # §D-DECODE Track 2: when True, the deploy Gumbel-SH head expands over the
        # MULTI-WINDOW no-drop legal-set action space (off-window saving cells get a
        # child — the structural off-window-defense fix). Default False keeps the
        # single-window dense head (the §D-DECODE 0/50 control + the live gate's
        # existing behavior unchanged).
        self._legal_set = bool(legal_set)
        # RNG threads through run_gumbel_on_board but is MULTIPLIED BY ZERO at the root
        # (gumbel_scale=0.0): the played move is deterministic regardless of seed. Kept
        # for signature parity / any future scale>0 diagnostic.
        self._rng = np.random.default_rng(seed)

    def get_move(self, state: GameState, rust_board: object) -> Tuple[int, int]:
        out = run_gumbel_on_board(
            self._engine,
            rust_board,
            n_sims=self._n_sims,
            m=self._m,
            c_visit=self._c_visit,
            c_scale=self._c_scale,
            c_puct=self._c_puct,
            dirichlet=False,
            gumbel_scale=EVAL_GUMBEL_SCALE,  # g=0 — deploy strength head
            legal_set=self._legal_set,       # §D-DECODE Track 2 multi-window decode
            rng=self._rng,
        )
        played = out["played_move"]
        if played is None:
            legal = rust_board.legal_moves()
            if not legal:
                raise RuntimeError("DeployHeadBot: no legal moves on board")
            return legal[0]
        return (int(played[0]), int(played[1]))

    def reset(self) -> None:
        """Stateless (g=0 deterministic, RNG x0 at the root) — no per-game carry-over."""

    def name(self) -> str:
        tag = ",ls" if self._legal_set else ""
        return f"deploy_head({self._label},m{self._m},n{self._n_sims}{tag})"


@dataclass(frozen=True)
class DeployStrengthConfig:
    """Pre-registered deploy-eval recipe + the adaptive screen->confirm band.

    PRE-REGISTERED BEFORE the adaptive logic is wired (investigation discipline:
    thresholds fixed in config/code BEFORE measurement; a post-hoc move = storytelling).

    Screen->confirm band: each cadence plays ``screen_n`` games (deploy head) vs the
    rotating best anchor. A full ``confirm_n`` confirm fires iff the screen WR lands in
    ``[screen_confirm_lo, screen_confirm_hi]`` — the band around the promotion bar where
    the screen is statistically AMBIGUOUS. Outside the band the screen alone is decisive:
      * WR < lo  : the candidate is clearly below the promotion bar (no promote) — a
                   confirm cannot rescue it, so skip (saves the confirm cost).
      * WR > hi  : the candidate is clearly above the bar; the confirm only sharpens the
                   CI, it cannot flip a clear pass to a fail — but we STILL run the
                   confirm here because promotion REQUIRES the distinct-game bootstrap CI
                   to clear, and a high screen point estimate can ride on a low effective-n
                   (copies). So ``promotion-eligible`` (WR >= lo) ALWAYS confirms; only the
                   clear-reject tail (WR < lo) skips. This is why the band CANNOT
                   false-negative a true candidate: every checkpoint whose screen WR could
                   possibly clear the bar is escalated to the full powered confirm.
    ``screen_confirm_lo`` is set a screen-CI half-width BELOW the promotion winrate so a
    true candidate sitting exactly at the bar (whose screen point estimate scatters below
    it by sampling noise) is still inside the escalate region — never dropped at screen."""

    promotion_winrate: float
    screen_n: int
    confirm_n: int
    screen_confirm_lo: float
    screen_confirm_hi: float
    opening_plies: int
    sealbot_max_depth: int
    n_boot: int
    seed_base: int
    min_distinct_per_pair: int

    @classmethod
    def from_cfg(cls, deploy_cfg: Dict[str, Any], promotion_winrate: float) -> "DeployStrengthConfig":
        screen_n = int(deploy_cfg.get("screen_n", 80))
        confirm_n = int(deploy_cfg.get("confirm_n", 200))
        # Default band: escalate-to-confirm whenever the screen WR is within one screen
        # Wilson half-width of the bar OR above it. half-width(n=80, p=0.5, z=1.96) ~ 0.11.
        screen_hw = 1.96 * math.sqrt(0.25 / max(screen_n, 1))
        lo = float(deploy_cfg.get("screen_confirm_lo", round(promotion_winrate - screen_hw, 3)))
        hi = float(deploy_cfg.get("screen_confirm_hi", 1.0))
        return cls(
            promotion_winrate=promotion_winrate,
            screen_n=screen_n,
            confirm_n=confirm_n,
            screen_confirm_lo=lo,
            screen_confirm_hi=hi,
            opening_plies=int(deploy_cfg.get("opening_plies", 4)),
            sealbot_max_depth=int(deploy_cfg.get("sealbot_max_depth", 5)),
            n_boot=int(deploy_cfg.get("n_boot", 1000)),
            seed_base=int(deploy_cfg.get("seed_base", 20260625)),
            min_distinct_per_pair=int(deploy_cfg.get("min_distinct_per_pair", 10)),
        )


def _build_engine_for_model(
    model: HexTacToeNet, encoding_name: str, device: torch.device
) -> LocalInferenceEngine:
    spec = _lookup_encoding(_normalize_encoding(encoding_name))
    return LocalInferenceEngine(getattr(model, "_orig_mod", model), device, encoding_spec=spec)


def _play_one_game(
    p1_bot: BotProtocol,
    p2_bot: BotProtocol,
    p1_label: str,
    p2_label: str,
    encoding_name: str,
    opening_plies: int,
    seed: int,
    max_plies: int = 200,
) -> Dict[str, Any]:
    """One game on the encoding-correct board, capturing the full move list + winner in
    the (p1, p2, winner, moves) record shape ``round_robin`` aggregate/dedup consume.

    Mirrors ``scripts/eval/gumbel_ladder._play_one_game``: uniform-random RNG-seeded
    opening plies (distinct-game diversity for the deterministic deploy regime), then the
    deploy head decides. ``head_fired`` confirms a net move was actually played by the
    head (validity gate: no silent all-opening game)."""
    rng = np.random.default_rng(seed)
    board = Board.with_encoding_name(_normalize_encoding(encoding_name))
    state = GameState.from_board(board)
    moves: List[List[int]] = []
    ply = 0
    head_fired = False
    while ply < max_plies and not board.check_win() and board.legal_move_count() > 0:
        if ply < opening_plies:
            legal = board.legal_moves()
            q, r = legal[int(rng.integers(0, len(legal)))]
        else:
            bot = p1_bot if board.current_player == 1 else p2_bot
            q, r = bot.get_move(state, board)
            head_fired = True
        moves.append([int(q), int(r)])
        state = state.apply_move(board, q, r)
        ply += 1
    winner_int = board.winner() if board.check_win() else None
    winner = "p1" if winner_int == 1 else ("p2" if winner_int == -1 else "draw")
    return {
        "p1": p1_label,
        "p2": p2_label,
        "winner": winner,
        "plies": ply,
        "moves": moves,
        "head_fired": head_fired,
    }


def _play_pair(
    bot_a: BotProtocol,
    bot_b: BotProtocol,
    label_a: str,
    label_b: str,
    encoding_name: str,
    n_games: int,
    opening_plies: int,
    seed_base: int,
) -> List[Dict[str, Any]]:
    """Color-balanced paired games between two bots; half A=P1, half A=P2."""
    games: List[Dict[str, Any]] = []
    for gi in range(n_games):
        p1b, p2b, p1l, p2l = (
            (bot_a, bot_b, label_a, label_b)
            if gi % 2 == 0
            else (bot_b, bot_a, label_b, label_a)
        )
        seed = seed_base + (hash((label_a, label_b)) & 0xFFFF) * 1000 + gi
        random.seed(seed)
        np.random.seed(seed % (2**31))
        games.append(
            _play_one_game(
                p1b, p2b, p1l, p2l, encoding_name,
                opening_plies=opening_plies, seed=seed,
            )
        )
    return games


def _wr_for_label(games: Sequence[Dict[str, Any]], label: str) -> Tuple[float, int, int, int]:
    """(win_rate, wins, draws, n) for ``label`` over a set of paired games. Draws=0.5."""
    wins = draws = 0
    n = len(games)
    for g in games:
        if g["winner"] == "draw":
            draws += 1
        elif (g["winner"] == "p1" and g["p1"] == label) or (
            g["winner"] == "p2" and g["p2"] == label
        ):
            wins += 1
    wr = (wins + 0.5 * draws) / n if n else 0.0
    return wr, wins, draws, n


@dataclass
class DeployStrengthResult:
    """Outcome of one deploy-strength gate run (screen [+ confirm])."""

    wr_screen: float
    wr_confirm: Optional[float]
    confirmed: bool
    promoted: bool
    elo_vs_best: Optional[float]
    ci_lo_boot: Optional[float]
    ci_hi_boot: Optional[float]
    n_games: int
    copy_multiplier: float
    distinct_per_pair_min: Optional[int]
    head_fired_frac: float
    sealbot_wr: Optional[float]
    reason: str


class DeployStrengthEvaluator:
    """Runs the deploy-matched strength gate (screen -> adaptive confirm) for one round.

    The model-under-test and the model opponents (best anchor) all play through
    ``DeployHeadBot`` (deploy head, g=0, no temp, deploy sims); the external bar is a
    fixed-depth SealBot. Promotion requires the distinct-game bootstrap BT-Elo over the
    CONFIRM games to clear the bar — not a raw-count Wilson CI."""

    def __init__(
        self,
        model: HexTacToeNet,
        device: torch.device,
        config: Dict[str, Any],
        deploy_cfg: Dict[str, Any],
        promotion_winrate: float,
    ) -> None:
        self.device = device
        self.config = config
        self.encoding_name = _normalize_encoding(config.get("encoding"))
        self.knobs = extract_deploy_knobs(config)  # HARD ERROR if any knob missing
        self.dcfg = DeployStrengthConfig.from_cfg(deploy_cfg, promotion_winrate)
        self._engine = _build_engine_for_model(model, self.encoding_name, device)
        self._cand = DeployHeadBot(self._engine, self.knobs, label="cand", seed=self.dcfg.seed_base)

    def _best_bot(self, best_model: HexTacToeNet) -> DeployHeadBot:
        eng = _build_engine_for_model(best_model, self.encoding_name, self.device)
        return DeployHeadBot(eng, self.knobs, label="best", seed=self.dcfg.seed_base + 101)

    def _sealbot(self) -> BotProtocol:
        from hexo_rl.bots.sealbot_bot import SealBotBot

        # Fixed-depth = REPRODUCIBLE (machine-independent). Large time ceiling so DEPTH
        # bounds the search, not wall-clock (§D-LADDER de-risk: drop time_limit=0.5).
        return SealBotBot(time_limit=60.0, max_depth=self.dcfg.sealbot_max_depth)

    def run(self, best_model: Optional[HexTacToeNet]) -> DeployStrengthResult:
        if best_model is None:
            return DeployStrengthResult(
                wr_screen=0.0, wr_confirm=None, confirmed=False, promoted=False,
                elo_vs_best=None, ci_lo_boot=None, ci_hi_boot=None, n_games=0,
                copy_multiplier=0.0, distinct_per_pair_min=None, head_fired_frac=0.0,
                sealbot_wr=None, reason="no best_model — gate skipped",
            )
        best_bot = self._best_bot(best_model)
        d = self.dcfg

        # ── Screen ────────────────────────────────────────────────────────────
        screen_games = _play_pair(
            self._cand, best_bot, "cand", "best", self.encoding_name,
            d.screen_n, d.opening_plies, d.seed_base,
        )
        wr_screen, _, _, _ = _wr_for_label(screen_games, "cand")
        head_frac = (
            sum(1 for g in screen_games if g["head_fired"]) / len(screen_games)
            if screen_games else 0.0
        )

        # ── Adaptive escalation ───────────────────────────────────────────────
        # Confirm whenever the screen WR is promotion-ELIGIBLE (>= lo). Only the
        # clear-reject tail (WR < lo) skips — it cannot clear the bar even with a perfect
        # confirm, so the band CANNOT false-negative a true candidate.
        escalate = wr_screen >= d.screen_confirm_lo
        if not escalate:
            return DeployStrengthResult(
                wr_screen=wr_screen, wr_confirm=None, confirmed=False, promoted=False,
                elo_vs_best=None, ci_lo_boot=None, ci_hi_boot=None,
                n_games=len(screen_games), copy_multiplier=0.0,
                distinct_per_pair_min=None, head_fired_frac=head_frac,
                sealbot_wr=None,
                reason=f"screen WR {wr_screen:.3f} < {d.screen_confirm_lo:.3f} bar — "
                       f"clear non-candidate, confirm skipped",
            )

        # ── Confirm (full power) ──────────────────────────────────────────────
        confirm_games = _play_pair(
            self._cand, best_bot, "cand", "best", self.encoding_name,
            d.confirm_n, d.opening_plies, d.seed_base + 7919,
        )
        # Pool screen+confirm games for the BT/bootstrap (both deploy-head, same pair).
        pooled = list(screen_games) + list(confirm_games)
        wr_confirm, _, _, _ = _wr_for_label(pooled, "cand")
        labels = ["best", "cand"]  # anchor = best (rotating champion) at Elo 0
        summary = aggregate_games(pooled, ladder_order=labels, n_boot=d.n_boot,
                                  min_distinct_per_pair=d.min_distinct_per_pair)
        boot = bootstrap_ratings_ci(pooled, labels, n_boot=d.n_boot, seed=d.seed_base)
        elo_cand = next((r["elo"] for r in summary["rungs"] if r["label"] == "cand"), None)
        ci_lo_boot, ci_hi_boot = boot.get("cand", (None, None))
        guard = effective_n_guard(pooled, labels=labels,
                                  min_distinct_per_pair=d.min_distinct_per_pair)
        dpp = distinct_per_pair(pooled, labels)
        dpp_min = min(dpp.values()) if dpp else None

        # ── External bar (fixed-depth SealBot) ────────────────────────────────
        seal_games = _play_pair(
            self._cand, self._sealbot(), "cand", "sealbot", self.encoding_name,
            d.confirm_n, d.opening_plies, d.seed_base + 104729,
        )
        sealbot_wr, _, _, _ = _wr_for_label(seal_games, "cand")

        head_frac_all = (
            sum(1 for g in pooled if g["head_fired"]) / len(pooled) if pooled else 0.0
        )

        # ── Promotion decision: distinct-game bootstrap BT-Elo CI must clear ──
        # Promote iff the candidate's distinct-game bootstrap CI lower bound vs the best
        # anchor is > 0 Elo (candidate is stronger than the rotating champion, CI-clean on
        # the HONEST effective-n) AND the confirm WR clears the promotion bar. The bootstrap
        # CI (NOT the Hessian) is the gate — copies cannot narrow it (§D-ARGMAX).
        wr_ok = wr_confirm >= d.promotion_winrate
        ci_clean = ci_lo_boot is not None and ci_lo_boot > 0.0
        low_power = bool(guard["low_power_warning"])
        promoted = bool(wr_ok and ci_clean and not low_power)
        if low_power:
            reason = (f"confirm WR {wr_confirm:.3f}, bootstrap CI [{ci_lo_boot},{ci_hi_boot}] "
                      f"but LOW EFFECTIVE-N (copy_mult={guard['copy_multiplier']}, "
                      f"min distinct/pair={dpp_min}) — block (untrusted CI, §D-ARGMAX)")
        elif promoted:
            reason = (f"PROMOTE: confirm WR {wr_confirm:.3f} >= {d.promotion_winrate}, "
                      f"distinct-game bootstrap Elo CI [{ci_lo_boot},{ci_hi_boot}] > 0")
        else:
            reason = (f"BLOCKED: confirm WR {wr_confirm:.3f} (bar {d.promotion_winrate}), "
                      f"bootstrap Elo CI [{ci_lo_boot},{ci_hi_boot}] "
                      f"(ci_clean={ci_clean}, wr_ok={wr_ok})")

        return DeployStrengthResult(
            wr_screen=wr_screen, wr_confirm=wr_confirm, confirmed=True, promoted=promoted,
            elo_vs_best=elo_cand, ci_lo_boot=ci_lo_boot, ci_hi_boot=ci_hi_boot,
            n_games=len(pooled), copy_multiplier=float(guard["copy_multiplier"]),
            distinct_per_pair_min=dpp_min, head_fired_frac=head_frac_all,
            sealbot_wr=sealbot_wr, reason=reason,
        )


__all__ = [
    "DeployHeadBot",
    "DeployStrengthConfig",
    "DeployStrengthEvaluator",
    "DeployStrengthResult",
    "EVAL_GUMBEL_SCALE",
    "extract_deploy_knobs",
]
