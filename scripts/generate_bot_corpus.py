#!/usr/bin/env python3
"""SealBot-vs-anchor bot-game NPZ generator — §178 T1.

Generates a fixed-pool corpus of SealBot vs the v6 anchor (weak side =
anchor with MCTS n_sims=200, T=0.5, dirichlet=true) and writes it as a
v6-encoded NPZ alongside a `.metadata.json` sidecar that records anchor
SHA, SealBot version, game-generation provenance, and decisive-game
counts. Drawn games + ply-cap-truncated games are DISCARDED (only
decisive outcomes give the anchor-mistake signal that motivates the
pool — see design §4.1 + contract T1 row 6).

CLI (defaults track design §4.2):
    python scripts/generate_bot_corpus.py \\
        --anchor checkpoints/bootstrap_model_v6.pt \\
        --n-games 700 \\
        --out data/bot_corpus_s178_sealbot_vs_v6.npz \\
        --max-plies 150 \\
        --random-opening-plies 4 \\
        --think-seconds 0.5 \\
        --anchor-n-sims 200 \\
        --anchor-temperature 0.5

The script REFUSES the Makefile fallback `checkpoints/bootstrap_model.pt`
(F1 hazard — random-init v6w25 per design §10 H1). Pass an explicit v6
anchor.

Per-row NPZ schema (v6 wire format):
  states    (T, 8, 19, 19)  float16
  policies  (T, 362)        float32  one-hot at the chosen-move target_idx
  outcomes  (T,)             float32  ∈ {-1.0, +1.0}  cur-player POV
  weights   (T,)             float32  uniform 1.0

§178 T1 deliverable — Batch B (isolated). Operator runs the actual NPZ
generation on vast (~10hr single-thread per design §4.2); smoke test
exercises the path with n_games=2 + max_plies=30.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import structlog
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine import Board  # noqa: E402
from hexo_rl.bootstrap.bot_protocol import BotProtocol  # noqa: E402
from hexo_rl.bootstrap.corpus_io import save_corpus  # noqa: E402
from hexo_rl.bots.sealbot_bot import SealBotBot  # noqa: E402
from hexo_rl.encoding import lookup as _lookup_encoding  # noqa: E402
from hexo_rl.env.game_state import GameState  # noqa: E402
from hexo_rl.selfplay.worker import SelfPlayWorker  # noqa: E402
from hexo_rl.training.checkpoints import load_inference_model  # noqa: E402
from hexo_rl.utils.device import best_device  # noqa: E402

log = structlog.get_logger()

_V6 = _lookup_encoding("v6")
_BOARD_SIZE: int = _V6.board_size                 # 19 — generator is 19x19 only
_N_ACTIONS: int = _V6.policy_logit_count          # 362

# F1 hazard — Makefile default that is fresh-init random v6w25; refuse.
_FORBIDDEN_ANCHOR_NAME = "bootstrap_model.pt"


def _resolve_generator_encoding(name: str):
    """Resolve + validate the encoding for the bot-corpus generator.

    §P5-CT P0-2 fix: the generator was hardcoded to v6 end-to-end. It now takes
    an --encoding and slices the resolved `spec.kept_plane_indices` so a
    v6tp/v6_live2 bot-mix recipe gets a corpus with the right plane count
    (instead of an 8-plane corpus that crashes the batch_assembly plane-count
    guard). The windowing math (board_size 19, +9 offset, single window) only
    supports the 19x19 single-window family — refuse v8/v6w25 loudly rather
    than silently miscompute window projections.
    """
    spec = _lookup_encoding(name)
    if spec.board_size != 19 or spec.is_multi_window:
        raise ValueError(
            f"generate_bot_corpus supports only single-window 19x19 encodings "
            f"(v6/v6tp/v6_live2/v7*); got {name!r} (board_size={spec.board_size}, "
            f"multi_window={spec.is_multi_window})."
        )
    return spec


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sealbot_version() -> str:
    try:
        import hexo_rl.bots.sealbot_bot as _sb
        return getattr(_sb, "__version__", "unknown")
    except Exception:
        return "unknown"


class _AnchorMCTSBot(BotProtocol):
    """Anchor side: SelfPlayWorker MCTS with n_sims, T, dirichlet wired.

    `OurModelBot` hardcodes `use_dirichlet=False`, so we wrap the worker
    directly to honour the design §4.2 weak-side knobs (dirichlet=true,
    T=0.5, n_sims=200).
    """

    def __init__(
        self,
        checkpoint_path: Path,
        n_sims: int,
        temperature: float,
        c_puct: float,
        device: torch.device,
        encoding: str = "v6",
    ) -> None:
        # Encoding lookup baked into SelfPlayWorker via resolve_from_config —
        # supply the run's encoding explicitly + mcts knobs through config dict.
        config = {
            "encoding": encoding,
            "n_simulations": int(n_sims),
            "c_puct": float(c_puct),
            "temperature_threshold_ply": 0,  # apply `temperature` everywhere
            "dirichlet_alpha": 0.3,
            "epsilon": 0.25,
        }
        net, _spec, _label = load_inference_model(
            str(checkpoint_path), config=config, device=device,
        )
        self._worker = SelfPlayWorker(model=net, config=config, device=device)
        self._temperature = float(temperature)
        self._ckpt_name = checkpoint_path.name

    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        policy = self._worker._run_mcts(
            rust_board,
            use_dirichlet=True,
            temperature=self._temperature,
        )
        legal = rust_board.legal_moves()
        if not legal:
            raise RuntimeError("anchor.get_move called with no legal moves")
        q, r = self._worker._sample_action(policy, legal, rust_board)
        return int(q), int(r)

    def reset(self) -> None:
        return

    def name(self) -> str:
        return f"anchor({self._ckpt_name})"


def _encode_v6_row(
    state: GameState, q: int, r: int, kept_plane_indices: list[int]
) -> Optional[Tuple[np.ndarray, int]]:
    """Encode a (state, chosen-move) row, slicing the source wire to the
    encoding's kept planes.

    Returns (state_<n_planes>x19x19_f16, target_idx) or None if the move does
    not project into any cluster window (skip the row — same semantics as
    `dataset.py:62` which advances `t` only when `target_k >= 0`).

    §P5-CT P0-2: `kept_plane_indices` is the resolved encoding's slice (v6→8,
    v6tp→10, v6_live2→4), no longer the module-level v6 constant.
    """
    tensor, centers = state.to_tensor()  # (K, 18, 19, 19) f16
    target_k = -1
    target_idx = -1
    for k, (cq, cr) in enumerate(centers):
        wq = q - cq + 9
        wr = r - cr + 9
        if 0 <= wq < _BOARD_SIZE and 0 <= wr < _BOARD_SIZE:
            target_k = k
            target_idx = wq * _BOARD_SIZE + wr
            break
    if target_k < 0:
        return None
    state_row = tensor[target_k][kept_plane_indices, :, :].astype(np.float16)
    return state_row, target_idx


def _play_one_game(
    sealbot_factory: Callable[[], BotProtocol],
    anchor_factory: Callable[[], BotProtocol],
    sealbot_side: int,
    max_plies: int,
    random_opening_plies: int,
    game_idx: int,
    encoding: str,
    kept_plane_indices: list[int],
) -> Tuple[List[np.ndarray], List[int], List[int], Optional[int], int, str]:
    """Play one game; record per-row state + target_idx + cur_player.

    Returns:
      states_8       — list of (n_planes,19,19) f16 arrays (length T)
      target_idxs    — list of int target indices (length T)
      cur_players    — list of int player labels {1,-1} at each row
      winner         — int player label or None
      ply            — total plies played
      terminal       — one of {"win", "draw", "ply_cap"}
    """
    sealbot = sealbot_factory()
    anchor = anchor_factory()

    board = Board.with_encoding_name(encoding)
    state = GameState.from_board(board)

    states_8: List[np.ndarray] = []
    target_idxs: List[int] = []
    cur_players: List[int] = []

    log.info("bot_corpus_game_start", game_idx=game_idx, sealbot_side=sealbot_side)

    ply = 0
    while ply < max_plies:
        if board.check_win() or board.legal_move_count() == 0:
            break
        if ply < random_opening_plies:
            q, r = random.choice(board.legal_moves())
            # Random-opening rows: low signal-to-noise. Skip recording — keep
            # the corpus focused on bot-vs-anchor decisions.
            state = state.apply_move(board, q, r)
            ply += 1
            continue
        if board.current_player == sealbot_side:
            q, r = sealbot.get_move(state, board)
        else:
            q, r = anchor.get_move(state, board)

        # Record row pre-apply (state + chosen move).
        row = _encode_v6_row(state, int(q), int(r), kept_plane_indices)
        if row is not None:
            state8, target_idx = row
            states_8.append(state8)
            target_idxs.append(int(target_idx))
            cur_players.append(int(state.current_player))

        state = state.apply_move(board, int(q), int(r))
        ply += 1

    winner = board.winner()
    if winner is None:
        if ply >= max_plies and not board.check_win():
            terminal = "ply_cap"
        else:
            terminal = "draw"
    else:
        terminal = "win"

    return states_8, target_idxs, cur_players, winner, ply, terminal


def _build_factories(
    anchor_path: Path,
    think_seconds: float,
    anchor_n_sims: int,
    anchor_temperature: float,
    anchor_c_puct: float,
    device: torch.device,
    encoding: str = "v6",
) -> Tuple[Callable[[], BotProtocol], Callable[[], BotProtocol]]:
    """Return (sealbot_factory, anchor_factory). Both spawn FRESH per game."""
    def sealbot_factory() -> BotProtocol:
        return SealBotBot(time_limit=think_seconds)

    def anchor_factory() -> BotProtocol:
        return _AnchorMCTSBot(
            checkpoint_path=anchor_path,
            n_sims=anchor_n_sims,
            temperature=anchor_temperature,
            c_puct=anchor_c_puct,
            device=device,
            encoding=encoding,
        )
    return sealbot_factory, anchor_factory


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SealBot-vs-anchor bot-game NPZ generator (§178 T1).",
    )
    parser.add_argument("--anchor", required=True,
                        help="Path to anchor checkpoint (.pt). "
                             "REQUIRED; F1-refuses bootstrap_model.pt.")
    parser.add_argument("--encoding", type=str, default="v6",
                        help="Registry encoding name (v6/v6tp/v6_live2/v7*); "
                             "slices the corpus to the encoding's plane count. "
                             "19x19 single-window only.")
    parser.add_argument("--n-games", type=int, default=700,
                        help="Number of games to generate (design §4.2 default 700).")
    parser.add_argument(
        "--out", type=str,
        default="data/bot_corpus_s178_sealbot_vs_v6.npz",
        help="Output NPZ path (sidecar .metadata.json written next to it).",
    )
    parser.add_argument("--max-plies", type=int, default=150,
                        help="Per-game ply cap (matches selfplay max_game_moves).")
    parser.add_argument("--random-opening-plies", type=int, default=4,
                        help="Plies of random openings per game (eval.yaml convention).")
    parser.add_argument("--think-seconds", type=float, default=0.5,
                        help="SealBot think budget per move (design §4.2 default 0.5s).")
    parser.add_argument("--anchor-n-sims", type=int, default=200,
                        help="Anchor MCTS simulation count (design §4.2 default 200).")
    parser.add_argument("--anchor-temperature", type=float, default=0.5,
                        help="Anchor sampling temperature (design §4.2 default 0.5).")
    parser.add_argument("--anchor-c-puct", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42,
                        help="Side-rotation + random-opening seed for determinism.")
    parser.add_argument(
        "--n-workers", type=int, default=1,
        help="Reserved for future multi-process generation. Currently serial only "
             "(per design §4.2). Values > 1 are accepted but ignored with a warning.",
    )
    args = parser.parse_args()

    anchor_path = Path(args.anchor).resolve()
    if not anchor_path.exists():
        print(f"FATAL: anchor not found: {anchor_path}", file=sys.stderr)
        return 1
    if anchor_path.name == _FORBIDDEN_ANCHOR_NAME:
        print(
            f"FATAL: refusing --anchor {anchor_path} — F1 hazard. "
            f"`{_FORBIDDEN_ANCHOR_NAME}` is the Makefile fallback which "
            f"resolves to a random-init v6w25 checkpoint (design §10 H1). "
            f"Pass an explicit v6 anchor, e.g. checkpoints/bootstrap_model_v6.pt.",
            file=sys.stderr,
        )
        return 1

    if args.n_workers != 1:
        log.warning(
            "bot_corpus_n_workers_ignored",
            requested=args.n_workers,
            note="serial-only generation in §178 T1; multi-proc deferred.",
        )

    try:
        _spec = _resolve_generator_encoding(args.encoding)
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    _kept_plane_indices = list(_spec.kept_plane_indices)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = best_device()
    print(
        f"[bot_corpus] device={device}  anchor={anchor_path.name}  "
        f"encoding={args.encoding} ({_spec.n_planes} planes)",
        flush=True,
    )

    sealbot_factory, anchor_factory = _build_factories(
        anchor_path=anchor_path,
        think_seconds=args.think_seconds,
        anchor_n_sims=args.anchor_n_sims,
        anchor_temperature=args.anchor_temperature,
        anchor_c_puct=args.anchor_c_puct,
        device=device,
        encoding=args.encoding,
    )

    all_states: List[np.ndarray] = []
    all_target_idxs: List[int] = []
    all_outcomes: List[float] = []

    n_kept = 0
    n_drawn = 0
    n_ply_cap = 0
    side_split = {"sealbot_x": 0, "sealbot_o": 0}

    t0 = time.time()
    for game_idx in range(args.n_games):
        # Alternate SealBot side across games (50/50 split).
        sealbot_side = 1 if (game_idx % 2 == 0) else -1
        try:
            states_8, target_idxs, cur_players, winner, ply, terminal = _play_one_game(
                sealbot_factory=sealbot_factory,
                anchor_factory=anchor_factory,
                sealbot_side=sealbot_side,
                max_plies=args.max_plies,
                random_opening_plies=args.random_opening_plies,
                game_idx=game_idx,
                encoding=args.encoding,
                kept_plane_indices=_kept_plane_indices,
            )
        except Exception as exc:  # noqa: BLE001 — log + skip game; continue corpus generation.
            log.error("bot_corpus_game_failed", game_idx=game_idx, error=str(exc))
            continue

        kept = False
        if terminal == "draw":
            log.warning("bot_corpus_drawn_game_skipped", game_idx=game_idx, ply=ply)
            n_drawn += 1
        elif terminal == "ply_cap":
            log.warning("bot_corpus_ply_cap_skipped", game_idx=game_idx, ply=ply)
            n_ply_cap += 1
        else:
            # Decisive — assign +1 / -1 from cur-player POV per row.
            assert winner is not None
            for state8, target_idx, cp in zip(states_8, target_idxs, cur_players):
                all_states.append(state8)
                all_target_idxs.append(target_idx)
                all_outcomes.append(1.0 if cp == winner else -1.0)
            kept = True
            n_kept += 1
            if sealbot_side == 1:
                side_split["sealbot_x"] += 1
            else:
                side_split["sealbot_o"] += 1

        log.info(
            "bot_corpus_game_complete",
            game_idx=game_idx,
            terminal_reason=terminal,
            winner=winner,
            ply=ply,
            kept=kept,
            sealbot_side=sealbot_side,
        )

    elapsed = time.time() - t0
    n_positions = len(all_outcomes)

    if n_positions == 0:
        print("FATAL: no decisive games — nothing to write.", file=sys.stderr)
        return 2

    states_arr = np.stack(all_states, axis=0).astype(np.float16)
    policies_arr = np.zeros((n_positions, _N_ACTIONS), dtype=np.float32)
    for i, ti in enumerate(all_target_idxs):
        policies_arr[i, ti] = 1.0
    outcomes_arr = np.asarray(all_outcomes, dtype=np.float32)
    weights_arr = np.ones(n_positions, dtype=np.float32)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    anchor_sha = _sha256_of_file(anchor_path)
    extra = {
        "purpose": "§178 SealBot-vs-anchor bot corpus (T1)",
        "anchor_path": str(anchor_path),
        "anchor_sha256": anchor_sha,
        "sealbot_version": _sealbot_version(),
        "n_games_kept": n_kept,
        "n_games_drawn": n_drawn,
        "n_games_ply_cap": n_ply_cap,
        "n_games_total": n_kept + n_drawn + n_ply_cap,
        "side_split_count": side_split,
        "max_plies": int(args.max_plies),
        "random_opening_plies": int(args.random_opening_plies),
        "think_seconds": float(args.think_seconds),
        "anchor_n_sims": int(args.anchor_n_sims),
        "anchor_temperature": float(args.anchor_temperature),
        "anchor_c_puct": float(args.anchor_c_puct),
        "seed": int(args.seed),
        "generation_date_utc": datetime.now(UTC).isoformat(),
        "elapsed_sec": round(elapsed, 1),
    }

    save_corpus(
        out_path,
        arrays={
            "states": states_arr,
            "policies": policies_arr,
            "outcomes": outcomes_arr,
            "weights": weights_arr,
        },
        encoding_name=args.encoding,
        source_manifest="scripts/generate_bot_corpus.py (§178 T1)",
        extra=extra,
        compress=True,
    )
    # save_corpus auto-computes sha256 + writes sidecar metadata.json.
    sidecar_path = out_path.with_name(out_path.name + ".metadata.json")
    final_sha = _sha256_of_file(out_path) if out_path.exists() else "missing"

    log.info(
        "bot_corpus_final_npz_written",
        path=str(out_path),
        sidecar=str(sidecar_path),
        n_positions=n_positions,
        n_games_kept=n_kept,
        sha256=final_sha,
    )
    print(
        f"[bot_corpus] DONE — {n_positions:,} positions across {n_kept} decisive games "
        f"(drawn={n_drawn}, ply_cap={n_ply_cap}). Elapsed {elapsed:.1f}s.",
        flush=True,
    )
    print(f"[bot_corpus] NPZ: {out_path}", flush=True)
    print(f"[bot_corpus] sidecar: {sidecar_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
