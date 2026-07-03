#!/usr/bin/env python3
"""BUCKET A2-HARNESS — deploy-matched honest-strength eval bot (gumbel-greedy).

Replaces the PUCT+temp-0.5 proxy (`ModelPlayer`) that biases strength LOW and
injects temperature noise the deployment regime never sees. The DEPLOY-MATCHED
action under the live d1m run (`longrun_v6_live2_ls_gumbel_m16`) is the Gumbel
Sequential-Halving WINNER:

    a* = argmax_{c in survivors}  [ gumbel(c) + log_prior(c) + sigma(completedQ(c)) ]
       = masked_argmax(gumbel_noise + policy_logits + completedQ)     (NO temperature)

This is EXACTLY `engine::game_runner::gumbel_search::best_action_pool_idx`
(gumbel_search.rs:154) — the same play head the net trains under. There is no
temperature anywhere (temperature is a PUCT-visit-policy knob; the deploy path
never samples a visit distribution). There is no PUCT visit-argmax selection of
the played move (PUCT only steers INTERIOR descent under `forced_root_child`,
which is the paper's primary algo: Gumbel-root + PUCT-interior).

RED-TEAM (self-audit, see DESIGN block at bottom):
  * NO PUCT visit-argmax for the played move    — action = SH winner, not get_policy argmax.
  * NO temperature leak                          — temperature param does not exist on this bot.
  * matched perception window both sides         — both players run the SAME net + SAME planner.
  * effective-n = DISTINCT games                 — opening diversity injected via random opening
                                                   plies (RNG-seeded); dedup byte-identical
                                                   sequences + bootstrap CI over distinct games
                                                   (the §D-ARGMAX pseudo-replication lesson).

KNOWN FIDELITY GATE (off-window): the Rust `MCTSTree` (PyO3) carries a flat
`policy_logit_count`-vector per leaf; the production v6_live2_ls planner keeps
off-GLOBAL-window COVERED candidates in a ragged `overflow` channel
(`records::aggregate_policy_ls`) that the flat vector cannot represent. For
positions whose SH winner is IN-window (the empirical case: the d1m model plays
off-window ~0%, colony_extension_fraction mean 0.0011), the chosen action
matches deploy. For off-window-heavy positions this planner CANNOT score
off-window candidates and will under-select them — an OPEN kill gate, NOT a
clear (a fixed-bot WR here would false-clear an off-window defect by
construction; CLAUDE.md off-window warning). See `ready=False` rationale.

Every numeric search knob is read from the checkpoint's embedded `config`
(the ACTUAL run config) or an explicit override dict — zero hardcoded literals;
a missing required knob is a hard error.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine import Board  # noqa: E402
from hexo_rl.bootstrap.bot_protocol import BotProtocol  # noqa: E402
from hexo_rl.encoding import lookup as _lookup_encoding  # noqa: E402
from hexo_rl.encoding import normalize_encoding_name as _normalize_encoding  # noqa: E402
from hexo_rl.env.game_state import GameState  # noqa: E402
from hexo_rl.eval.checkpoint_loader import _build_model_from_spec  # noqa: E402
from hexo_rl.eval.gumbel_search_py import run_gumbel_on_board, run_puct_on_board  # noqa: E402
from hexo_rl.selfplay.inference import LocalInferenceEngine  # noqa: E402
from hexo_rl.training.checkpoints import normalize_model_state_dict_keys  # noqa: E402

Action = Tuple[int, int]

# Config key paths that carry each deploy search knob. Single source of truth;
# no scalar default — a missing key is a hard error (zero-literal discipline).
_REQUIRED_KNOBS = {
    "gumbel_m": ("selfplay", "gumbel_m"),
    "c_visit": ("selfplay", "c_visit"),
    "c_scale": ("selfplay", "c_scale"),
    "n_sims_full": ("selfplay", "playout_cap", "n_sims_full"),
    "dirichlet_enabled": ("mcts", "dirichlet_enabled"),
    "c_puct": ("mcts", "c_puct"),
}


def _dig(cfg: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    cur: Any = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"config missing required knob at {'.'.join(path)!r}")
        cur = cur[k]
    return cur


def extract_deploy_knobs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Pull every deploy search knob from a run config. Hard-errors on any gap."""
    knobs: Dict[str, Any] = {}
    for name, path in _REQUIRED_KNOBS.items():
        knobs[name] = _dig(cfg, path)
    return knobs


def _resolve_encoding_name(
    raw: Dict[str, Any], ckpt_path: "str | Path", explicit: Optional[str],
) -> str:
    """Encoding name for the planner, via the SHARED gated stamp resolution.

    D-EVALGATE fix wave (ESCAPE 5): this used to be a hand-rolled
    ``if explicit: return explicit`` — a silent override that never compared
    against the checkpoint's own stamp. It now reuses
    ``checkpoint_loader._resolve_ckpt_stamped_encoding`` (no duplicated
    logic): ``explicit`` (``--encoding``) is a DECODE-TIME cross-decode — it
    ALWAYS wins, but a disagreeing stamp is logged loudly
    (``encoding_decode_override``), never silently. With no ``--encoding``,
    the checkpoint's OWN stamp (metadata/config/raw['encoding']) is
    authoritative over the embedded run config's bare ``encoding`` field
    (which is itself just a copy of the same stamp — kept as a last-resort
    fallback for checkpoints saved before the stamp existed).
    """
    from hexo_rl.eval.checkpoint_loader import (
        _log as _ckpt_log,
        _normalize_or_raise,
        _resolve_ckpt_stamped_encoding,
    )

    ckpt_stamp_name, ckpt_stamp_source = _resolve_ckpt_stamped_encoding(raw, ckpt_path)

    if explicit:
        override_name = _normalize_or_raise(
            explicit, side="decode_override", source="--encoding",
        )
        log_fields = {
            "checkpoint": str(ckpt_path),
            "ckpt_stamp": ckpt_stamp_name,
            "ckpt_stamp_source": ckpt_stamp_source,
            "decode_as": override_name,
        }
        if ckpt_stamp_name is not None and override_name != ckpt_stamp_name:
            _ckpt_log.warning("encoding_decode_override", **log_fields)
        else:
            _ckpt_log.info("encoding_decode_override", **log_fields)
        return override_name

    if ckpt_stamp_name is not None:
        return ckpt_stamp_name

    cfg = raw.get("config", {}) if isinstance(raw, dict) else {}
    enc = cfg.get("encoding") if isinstance(cfg, dict) else None
    if isinstance(enc, str):
        return enc
    raise ValueError(
        "encoding not resolvable from checkpoint stamp or config; pass "
        "--encoding (live d1m run = 'v6_live2_ls')"
    )


def load_state_and_config(
    ckpt_path: str | Path,
) -> Tuple[dict, Dict[str, Any], Dict[str, Any]]:
    """Load a checkpoint, return (normalized_state_dict, embedded_config, raw).

    Accepts both a bare state-dict (inference_only.pt) and a full training
    checkpoint (carries 'model_state' + 'config'). A bare state-dict has no
    embedded config — caller must then supply knobs/encoding explicitly.

    ``raw`` is the untouched top-level checkpoint dict (or the bare state
    dict itself when there is no wrapper) — callers that need the checkpoint's
    OWN encoding stamp (metadata/config/raw['encoding']) use it with
    ``checkpoint_loader._resolve_ckpt_stamped_encoding`` instead of
    re-parsing the file.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: Dict[str, Any] = {}
    state: dict = raw
    if isinstance(raw, dict) and any(
        k in raw for k in ("model_state", "model_state_dict", "state_dict")
    ):
        cfg = raw.get("config", {}) or {}
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in raw and isinstance(raw[key], dict):
                state = raw[key]
                break
    state = normalize_model_state_dict_keys(state)
    return state, cfg, raw


class GumbelGreedyBot(BotProtocol):
    """Deploy-matched gumbel-greedy player: SH-winner action, no temperature.

    Drives the SAME engine planner the net trains under (Rust `MCTSTree`
    primitives via `run_gumbel_on_board`: Gumbel-root candidate selection +
    PUCT interior descent under `forced_root_child` + completed-Q SH winner).
    The played move is the Sequential-Halving winner — there is no temperature
    and no PUCT visit-argmax of the played move.
    """

    def __init__(
        self,
        engine: LocalInferenceEngine,
        knobs: Dict[str, Any],
        label: str,
        seed: int = 0,
    ) -> None:
        self._engine = engine
        self._m = int(knobs["gumbel_m"])
        self._n_sims = int(knobs["n_sims_full"])
        self._c_visit = float(knobs["c_visit"])
        self._c_scale = float(knobs["c_scale"])
        self._c_puct = float(knobs["c_puct"])
        self._dirichlet = bool(knobs["dirichlet_enabled"])
        self._label = label
        self._rng = np.random.default_rng(seed)

    def get_move(self, state: GameState, rust_board: object) -> Action:
        out = run_gumbel_on_board(
            self._engine,
            rust_board,
            n_sims=self._n_sims,
            m=self._m,
            c_visit=self._c_visit,
            c_scale=self._c_scale,
            c_puct=self._c_puct,
            dirichlet=self._dirichlet,
            rng=self._rng,
        )
        played = out["played_move"]
        if played is None:
            legal = rust_board.legal_moves()
            if not legal:
                raise RuntimeError("GumbelGreedyBot: no legal moves on board")
            return legal[0]
        return (int(played[0]), int(played[1]))

    def name(self) -> str:
        return f"gumbel_greedy({self._label},m{self._m},n{self._n_sims})"


def planner_rank_divergence(
    engine: LocalInferenceEngine, board: Board, knobs: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """How often the deploy gumbel-greedy pick != the PUCT+temp proxy pick.

    Runs BOTH planners on the SAME board (matched perception). Returns the two
    picks + a `divergent` flag. Aggregated over positions this is the
    planner-rank divergence metric: the fraction of decisions where the biased
    PUCT+temp proxy would have chosen a different move than the deploy head —
    the magnitude of the proxy's strength mis-measurement at the action level.

    PUCT proxy pick = visit-argmax (temperature=0 read of get_policy), matching
    `ModelPlayer`'s deploy proxy. Gumbel pick = SH winner.
    """
    g = run_gumbel_on_board(
        engine, board, n_sims=int(knobs["n_sims_full"]), m=int(knobs["gumbel_m"]),
        c_visit=float(knobs["c_visit"]), c_scale=float(knobs["c_scale"]),
        c_puct=float(knobs["c_puct"]), dirichlet=bool(knobs["dirichlet_enabled"]),
        rng=rng,
    )
    p = run_puct_on_board(
        engine, board, n_sims=int(knobs["n_sims_full"]),
        c_puct=float(knobs["c_puct"]), dirichlet=False,
        c_visit=float(knobs["c_visit"]), c_scale=float(knobs["c_scale"]),
        rng=rng,
    )
    gp = g["played_move"]
    pp = p["played_move"]
    return {
        "gumbel_pick": gp,
        "puct_pick": pp,
        "divergent": (gp is not None and pp is not None and tuple(gp) != tuple(pp)),
    }


def _parse_move(s: str) -> Action:
    q, r = s.strip("()").split(",")
    return int(q), int(r)


def play_smoke_game(
    bot_a: GumbelGreedyBot, bot_b: BotProtocol, encoding_name: str,
    opening_plies: int, seed: int, a_is_p1: bool,
) -> Dict[str, Any]:
    """One game; `opening_plies` RNG-seeded random opening plies for diversity.

    Opening diversity is load-bearing for an argmax/greedy regime: without it,
    deterministic gumbel-greedy from a fixed start collapses to ~2 distinct
    games (one per color) and a raw-count CI is over-confident by sqrt(copies)
    (§D-ARGMAX). The opening RNG is seeded per-game so the corpus is
    reproducible and distinct-game count is auditable.
    """
    rng = np.random.default_rng(seed)
    board = Board.with_encoding_name(encoding_name)
    state = GameState.from_board(board)
    moves: List[Action] = []
    ply = 0
    bot_a.reset()
    bot_b.reset()
    while not board.check_win() and board.legal_move_count() > 0:
        if ply < opening_plies:
            legal = board.legal_moves()
            q, r = legal[int(rng.integers(0, len(legal)))]
        else:
            is_a_turn = (board.current_player == 1) == a_is_p1
            mover = bot_a if is_a_turn else bot_b
            q, r = mover.get_move(state, board)
        moves.append((q, r))
        state = state.apply_move(board, q, r)
        ply += 1
    raw_winner = board.winner()
    if raw_winner is None:
        a_won = False
        winner = "draw"
    elif (raw_winner == 1) == a_is_p1:
        a_won = True
        winner = "A"
    else:
        a_won = False
        winner = "B"
    return {
        "winner": winner,
        "a_won": a_won,
        "n_moves": len(moves),
        "moves_list": [f"({q},{r})" for q, r in moves],
        "a_is_p1": a_is_p1,
        "seed": seed,
    }


def distinct_game_count(games: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Byte-identical move-sequence dedup (§D-ARGMAX effective-n guard)."""
    keys = [tuple(g["moves_list"]) for g in games]
    n_total = len(keys)
    n_distinct = len(set(keys))
    return {
        "n_total": n_total,
        "n_distinct": n_distinct,
        "copy_multiplier": (n_total / n_distinct) if n_distinct else 0.0,
    }


def _build_engine(
    ckpt_path: str, encoding_name: str, device: torch.device,
    expect_encoding: Optional[str] = None,
) -> LocalInferenceEngine:
    """Build a LocalInferenceEngine decoding ``ckpt_path`` under ``encoding_name``.

    ``encoding_name`` is DECODE-only (the checkpoint's own stamp is never
    consulted) — historical behaviour, kept for cross-decode callers.
    ``expect_encoding`` (D-WS3V3 A1 amendment) is the ASSERTION form: the
    checkpoint's OWN stamp must exist AND match, else this raises
    (``DeclaredEncodingMismatchError`` on disagreement, ``ValueError`` when
    the checkpoint is unstamped — an unstamped ckpt cannot satisfy an
    assertion). Pass both the same value to pin a gated read.
    """
    state, _cfg, raw = load_state_and_config(ckpt_path)
    if expect_encoding is not None:
        from hexo_rl.eval.checkpoint_loader import (
            _check_declared_vs_stamped_encoding,
            _resolve_ckpt_stamped_encoding,
        )
        stamp_name, stamp_source = _resolve_ckpt_stamped_encoding(raw, ckpt_path)
        if stamp_name is None:
            raise ValueError(
                f"--expect-encoding {expect_encoding!r}: checkpoint {ckpt_path} "
                "carries NO encoding stamp (metadata['encoding_name'] / "
                "config['encoding'] / raw['encoding']) — an unstamped checkpoint "
                "cannot satisfy an encoding assertion. Re-stamp it (e.g. "
                "scripts/make_ws3v3_warmstart.py) or drop --expect-encoding."
            )
        _check_declared_vs_stamped_encoding(expect_encoding, stamp_name, stamp_source)
    spec = _lookup_encoding(_normalize_encoding(encoding_name))
    model = _build_model_from_spec(state, spec)
    model.to(device).eval()
    return LocalInferenceEngine(model, device, encoding_spec=spec)


def main() -> None:
    ap = argparse.ArgumentParser(description="gumbel-greedy deploy-matched eval bot (SMOKE)")
    ap.add_argument("--checkpoint", required=True, help="weights (.pt)")
    ap.add_argument("--config-checkpoint", default=None,
                    help="checkpoint carrying the run config; defaults to --checkpoint. "
                         "Use a full training .pt when --checkpoint is a bare inference_only.pt.")
    ap.add_argument("--encoding", default=None, help="runtime encoding label (e.g. v6_live2_ls)")
    ap.add_argument("--opponent", choices=["self", "sealbot"], default="self")
    ap.add_argument("--n-games", type=int, default=4)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=20260624)
    ap.add_argument("--divergence-probe", action="store_true",
                    help="also report planner-rank divergence on each opening position")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_ckpt = args.config_checkpoint or args.checkpoint
    _state, cfg, raw = load_state_and_config(cfg_ckpt)
    if not cfg:
        raise ValueError(
            f"{cfg_ckpt} carries no embedded config; pass --config-checkpoint pointing "
            "at a full training checkpoint so deploy knobs are read, not guessed."
        )
    encoding_name = _resolve_encoding_name(raw, cfg_ckpt, args.encoding)
    knobs = extract_deploy_knobs(cfg)
    print(f"[knobs] {json.dumps(knobs)}  encoding={encoding_name}  device={device}")

    engine = _build_engine(args.checkpoint, encoding_name, device)
    bot_a = GumbelGreedyBot(engine, knobs, label=Path(args.checkpoint).stem, seed=args.seed_base)

    if args.opponent == "sealbot":
        from hexo_rl.bots.sealbot_bot import SealBotBot
        bot_b: BotProtocol = SealBotBot(time_limit=0.05)
    else:
        bot_b = GumbelGreedyBot(engine, knobs, label="self_b", seed=args.seed_base + 7919)

    games: List[Dict[str, Any]] = []
    t0 = time.time()
    for i in range(args.n_games):
        rec = play_smoke_game(
            bot_a, bot_b, encoding_name, args.opening_plies,
            seed=args.seed_base + i, a_is_p1=(i % 2 == 0),
        )
        games.append(rec)
        print(f"  game {i}: winner={rec['winner']:5s} moves={rec['n_moves']:3d} "
              f"a_is_p1={rec['a_is_p1']}")
    dur = time.time() - t0
    dg = distinct_game_count(games)
    a_wins = sum(1 for g in games if g["a_won"])
    print(f"[smoke] {len(games)} games in {dur:.1f}s  A_wins={a_wins}  "
          f"distinct={dg['n_distinct']}/{dg['n_total']} (copy_mult={dg['copy_multiplier']:.2f})")

    if args.divergence_probe:
        rng = np.random.default_rng(args.seed_base)
        div = 0
        n = 0
        for i in range(args.n_games):
            board = Board.with_encoding_name(encoding_name)
            grng = np.random.default_rng(args.seed_base + i)
            for _ in range(args.opening_plies):
                legal = board.legal_moves()
                if not legal:
                    break
                q, r = legal[int(grng.integers(0, len(legal)))]
                board.apply_move(q, r)
            if board.check_win() or board.legal_move_count() == 0:
                continue
            d = planner_rank_divergence(engine, board, knobs, rng)
            div += int(d["divergent"])
            n += 1
            print(f"  div[{i}] gumbel={d['gumbel_pick']} puct={d['puct_pick']} "
                  f"divergent={d['divergent']}")
        if n:
            print(f"[divergence] {div}/{n} positions diverge ({div / n:.1%})")


if __name__ == "__main__":
    main()
