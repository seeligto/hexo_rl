"""KrakenV1MCTSBot — MCTS wrapper for kraken_v1.pt external anchor.

Runs KrakenBot's native MCTS (200 sims, temperature=0 deterministic argmax)
so we can do a fair search-vs-search comparison. CPU-only; GPU free for our head.

Board bridge mirrors KrakenV1Bot: rust_board.get_stones()/get()/legal_moves(),
state.current_player 1/-1 -> KPlayer A/B, state.moves_remaining, pair-caching.
MCTS runs once per compound turn; second stone returned from cache.

Diagnostics: JSON lines -> reports/anchorx/*/kraken_mcts_diag.jsonl (never stderr).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

import structlog
import torch

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.env import GameState

log = structlog.get_logger()

# vendor/external/krakenbot — untracked, gitignored.
_KRAKENBOT_EXT = Path(__file__).parents[2] / "vendor" / "external" / "krakenbot"
assert _KRAKENBOT_EXT.exists(), (
    f"vendor/external/krakenbot not cloned at {_KRAKENBOT_EXT}. "
    "Run: git clone https://github.com/Ramora0/KrakenBot vendor/external/krakenbot"
)

# Use importlib to load krakenbot's game.py + mcts_bot.py under unique module
# names so they don't collide with sealbot's game.py in sys.modules.
# Both vendors have a top-level game.py; whoever loads first wins 'game' in
# sys.modules.  We alias krakenbot's modules to avoid that race.
def _load_krakenbot_module(name: str, rel_path: str):
    """Load a krakenbot module by file path under a unique sys.modules key."""
    alias = f"_krakenbot_ext_{name.replace('.', '_')}"
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(
        alias, str(_KRAKENBOT_EXT / rel_path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    # Krakenbot must be first on path when its module is executed so that its
    # own relative imports (from game import ..., from model.resnet import ...)
    # resolve to krakenbot's files.
    _insert_krakenbot_path()
    spec.loader.exec_module(mod)
    return mod


def _insert_krakenbot_path():
    """Ensure krakenbot root is first on sys.path (beats sealbot's game.py)."""
    ext = str(_KRAKENBOT_EXT)
    if sys.path and sys.path[0] == ext:
        return
    if ext in sys.path:
        sys.path.remove(ext)
    sys.path.insert(0, ext)


# Register krakenbot game module first under its canonical name so that
# krakenbot's internal imports (from model.resnet import ..., etc.) find
# krakenbot's game — not sealbot's.  We only do this if 'game' hasn't already
# been loaded from krakenbot (identified by the presence of 'ToroidalHexGame').
def _ensure_krakenbot_game_registered():
    if "game" in sys.modules and hasattr(sys.modules["game"], "ToroidalHexGame"):
        return  # already krakenbot's game
    # Load krakenbot game under alias first.
    kraken_game = _load_krakenbot_module("game", "game.py")
    # If 'game' is not yet in sys.modules or is sealbot's version, we can't
    # forcibly replace it (would break sealbot).  Instead we use the aliased
    # kraken_game directly and pass it to the mcts_bot load below.
    return kraken_game


_kraken_game_mod = _ensure_krakenbot_game_registered()

# Load model.resnet under alias so krakenbot's internal imports work.
# We must ensure the mcts package (mcts/tree.py etc.) also resolves correctly.
# Simplest: insert krakenbot path first, then do a guarded import of mcts_bot.
_insert_krakenbot_path()

# Now load mcts_bot: its top-level imports (from game import ...) will find
# krakenbot's game IF it's first on sys.path and game isn't cached from sealbot.
# If game IS already cached from sealbot, we temporarily swap it out.
_game_backup = sys.modules.get("game")
_need_restore = False
if _game_backup is not None and not hasattr(_game_backup, "ToroidalHexGame"):
    # sealbot's game is cached — swap in krakenbot's for the import.
    sys.modules["game"] = _kraken_game_mod  # type: ignore[assignment]
    _need_restore = True

try:
    # Also ensure mcts package resolves to krakenbot's mcts/ not any other.
    if "mcts" in sys.modules and not hasattr(sys.modules.get("mcts.tree"), "select_move_pair"):
        # Stale mcts in cache — clear it so krakenbot's gets loaded.
        for _k in [k for k in sys.modules if k == "mcts" or k.startswith("mcts.")]:
            del sys.modules[_k]

    import mcts_bot as _mcts_bot_mod  # type: ignore[import]
    _MCTSBot = _mcts_bot_mod.MCTSBot
finally:
    if _need_restore and _game_backup is not None:
        sys.modules["game"] = _game_backup

# Use krakenbot's HexGame and Player from the aliased module.
_KPlayer = _kraken_game_mod.Player   # type: ignore[union-attr]
HexGame = _kraken_game_mod.HexGame   # type: ignore[union-attr]

_DEFAULT_CKPT = Path(__file__).parents[2] / "checkpoints" / "external" / "kraken_v1.pt"
_DEFAULT_DIAG = Path("reports") / "anchorx" / "kraken_mcts" / "kraken_mcts_diag.jsonl"


class KrakenV1MCTSBot(BotProtocol):
    """BotProtocol wrapper for kraken_v1.pt using KrakenBot's native MCTS.

    Args:
        model_path:  Path to kraken_v1.pt. None -> checkpoints/external/kraken_v1.pt.
        n_sims:      MCTS simulations per move. Default 200 (deploy/eval default).
        temperature: Move-selection temperature. 0.0 = deterministic argmax (select_move_pair
                     branches on temperature < 0.05 -> argmax, no div-by-zero). Default 0.0.
        device:      Torch device string. None -> auto (cuda if available, else cpu).
        label:       Bot name returned by name(). Default "krakenbot_mcts".
        diag_path:   Append diagnostics JSON lines here. None -> default path.
                     Pass False to disable diagnostics entirely.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_sims: int = 200,
        temperature: float = 0.0,
        device: Optional[str] = None,
        label: str = "krakenbot_mcts",
        diag_path: Optional[object] = None,
    ) -> None:
        path = str(Path(model_path) if model_path is not None else _DEFAULT_CKPT)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._mcts = _MCTSBot(
            model_path=path,
            n_sims=n_sims,
            device=device,
            temperature=temperature,
        )
        self._n_sims = n_sims
        self._temperature = temperature
        self._device = device
        self._label = label

        # Per-turn pair cache: second stone from compound turn.
        self._pending_move: Optional[tuple[int, int]] = None

        # Diagnostics sidecar — never write to stderr.
        if diag_path is False:
            self._diag: Optional[Path] = None
        elif diag_path is None:
            self._diag = Path(_DEFAULT_DIAG)
        else:
            self._diag = Path(diag_path)  # type: ignore[arg-type]

        if self._diag is not None:
            self._diag.parent.mkdir(parents=True, exist_ok=True)

    # ── BotProtocol ──────────────────────────────────────────────────────────

    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        """Return one legal move. Compound turns: first call runs MCTS + caches
        stone2; second call returns cached stone without a second search."""

        # Return cached second stone from this compound turn.
        if self._pending_move is not None:
            move = self._pending_move
            self._pending_move = None
            q_p, r_p = move
            if rust_board.get(q_p, r_p) == 0:  # type: ignore[attr-defined]
                return move
            # Cached stone is now occupied — shouldn't happen, but guard it.
            log.warning("krakenbot_v1_mcts_cached_stone_illegal",
                        q=q_p, r=r_p, ply=state.ply)
            legal = list(rust_board.legal_moves())  # type: ignore[attr-defined]
            if not legal:
                raise RuntimeError("No legal moves on board")
            return legal[0]

        # Build HexGame from our board state.
        game = self._build_hex_game(state, rust_board)

        # Run MCTS — returns list of (q,r) pairs (1 or 2 elements).
        move_pairs: list[tuple[int, int]] = self._mcts.get_move(game)

        legal_set = set(rust_board.legal_moves())  # type: ignore[attr-defined]
        chosen1: tuple[int, int]
        chosen2: Optional[tuple[int, int]] = None

        if len(move_pairs) >= 1:
            m1 = move_pairs[0]
            if m1 in legal_set:
                chosen1 = m1
            else:
                log.warning("krakenbot_v1_mcts_stone1_illegal",
                            move=m1, ply=state.ply)
                avail = list(legal_set)
                chosen1 = avail[0] if avail else m1

            if len(move_pairs) >= 2:
                m2 = move_pairs[1]
                legal_set2 = legal_set - {chosen1}
                if m2 in legal_set2 and m2 != chosen1:
                    chosen2 = m2
                else:
                    log.warning("krakenbot_v1_mcts_stone2_illegal",
                                move=m2, ply=state.ply)
                    avail2 = list(legal_set2)
                    chosen2 = avail2[0] if avail2 else None
        else:
            # Fallback: MCTS returned empty list (shouldn't happen).
            log.warning("krakenbot_v1_mcts_empty_result", ply=state.ply)
            avail = list(legal_set)
            chosen1 = avail[0] if avail else (0, 0)

        # Cache second stone for next call.
        if chosen2 is not None and state.moves_remaining >= 2:
            self._pending_move = chosen2

        self._write_diag(
            ply=state.ply,
            pair=[chosen1] + ([chosen2] if chosen2 else []),
            n_sims=self._n_sims,
            temperature=self._temperature,
            root_value=getattr(self._mcts, "last_root_value", None),
        )
        return chosen1

    def reset(self) -> None:
        """Clear cached second stone. Call before starting a new game."""
        self._pending_move = None

    def name(self) -> str:
        return self._label

    # ── internals ─────────────────────────────────────────────────────────────

    def _build_hex_game(self, state: GameState, rust_board: object) -> HexGame:
        """Build a KrakenBot HexGame from our Rust board state.

        Maps:
          state.current_player  1/-1  ->  KPlayer.A/KPlayer.B
          state.moves_remaining       ->  HexGame.moves_left_in_turn
          rust_board.get_stones()     ->  HexGame.board {(q,r): Player}
        """
        game = HexGame(win_length=6)
        # Populate board from Rust (replaces reset()'s empty board).
        board_dict: dict = {}
        stone_count = 0
        for q, r, p in rust_board.get_stones():  # type: ignore[attr-defined]
            board_dict[(q, r)] = _KPlayer.A if p == 1 else _KPlayer.B
            stone_count += 1
        game.board = board_dict

        # current_player
        game.current_player = _KPlayer.A if state.current_player == 1 else _KPlayer.B

        # moves_left_in_turn: our moves_remaining maps directly.
        game.moves_left_in_turn = state.moves_remaining

        # move_count: used by MCTSBot to detect empty board (shortcut to center).
        # Use stone count to accurately reflect board occupancy.
        game.move_count = stone_count

        # game_over / winner: always False/NONE for positions we're asked to move from.
        game.game_over = False
        game.winner = _KPlayer.NONE

        return game

    def _write_diag(
        self,
        ply: int,
        pair: list,
        n_sims: int,
        temperature: float,
        root_value: Optional[float],
    ) -> None:
        if self._diag is None:
            return
        rec = {
            "ply": ply,
            "pair": pair,
            "n_sims": n_sims,
            "temperature": temperature,
            "root_value": root_value,
        }
        try:
            with self._diag.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError as exc:
            log.warning("krakenbot_v1_mcts_diag_write_error", exc=str(exc))
