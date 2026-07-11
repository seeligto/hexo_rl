"""KrakenV1Bot — pure-policy wrapper for kraken_v1.pt external anchor checkpoint.

Raw-policy forward (no MCTS, no Cython dependency). Pair-move caching: one
forward per compound turn, returns stone-1 then stone-2 on successive calls.
CPU-only: keeps GPU free for our head. Device param accepted but no CUDA code.

Diagnostics: one JSON line per DECISION appended to diag_path (never stderr —
64KB pipe-deadlock risk). Default path: reports/anchorx/kraken_v1_eval.jsonl.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Optional

import structlog
import torch

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.env import GameState

log = structlog.get_logger()

# vendor/external/krakenbot — untracked, gitignored. Contains model/resnet.py
# (HexResNet + board_to_planes) and game.py (Player enum).
_KRAKENBOT_EXT = Path(__file__).parents[2] / "vendor" / "external" / "krakenbot"
assert _KRAKENBOT_EXT.exists(), (
    f"vendor/external/krakenbot not cloned at {_KRAKENBOT_EXT}. "
    "Run: git clone https://github.com/Ramora0/KrakenBot vendor/external/krakenbot"
)
if str(_KRAKENBOT_EXT) not in sys.path:
    sys.path.insert(0, str(_KRAKENBOT_EXT))

from model.resnet import HexResNet, board_to_planes  # type: ignore[import]
from game import Player as _KPlayer                  # type: ignore[import]

_DEFAULT_CKPT = Path(__file__).parents[2] / "checkpoints" / "external" / "kraken_v1.pt"
_DEFAULT_DIAG = Path("reports") / "anchorx" / "kraken_v1_eval.jsonl"


def _smart_legal_fallback(
    rust_board: object,
    reason: str,
    ply: int,
) -> tuple[int, int]:
    """Uniform-random-legal fallback when argsort finds no legal pair.

    Mirrors SealBotBot's fallback pattern. Logs to structlog (not stderr).
    """
    legal = rust_board.legal_moves()  # type: ignore[attr-defined]
    if not legal:
        raise RuntimeError("No legal moves available on board")
    log.warning("krakenbot_v1_fallback", reason=reason, ply=ply, n_legal=len(legal))
    return random.choice(legal)


class KrakenV1Bot(BotProtocol):
    """BotProtocol wrapper for kraken_v1.pt — raw-policy argmax, CPU-only.

    Args:
        model_path: Path to kraken_v1.pt. None → checkpoints/external/kraken_v1.pt.
        device:     Torch device string. Default "cpu". No CUDA code used.
        label:      Bot name returned by name(). Default "krakenbot".
        seed:       Unused (argmax is deterministic). Reserved for tie-break.
        diag_path:  Append diagnostics JSON lines here. None → default path.
                    Pass False to disable diagnostics entirely.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        label: str = "krakenbot",
        seed: int = 0,
        diag_path: Optional[object] = None,
    ) -> None:
        path = Path(model_path) if model_path is not None else _DEFAULT_CKPT
        ckpt = torch.load(str(path), map_location=device, weights_only=True)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        self._model = HexResNet()
        self._model.load_state_dict(sd, strict=True)
        self._model.to(device)
        self._model.eval()
        self._model.set_padding_mode("zeros")  # infinite-board eval mode

        self._device = device
        self._label = label
        self._seed = seed
        self._pending_move: Optional[tuple[int, int]] = None

        # Diagnostics sidecar — append JSON lines; never write to stderr.
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
        """Return one legal move. On compound turns: first call forwards + caches,
        second call returns cached stone without a second forward."""

        # Return cached second stone from this compound turn.
        if self._pending_move is not None:
            move = self._pending_move
            self._pending_move = None
            q_p, r_p = move
            if rust_board.get(q_p, r_p) == 0:  # type: ignore[attr-defined]
                return move
            # Cached stone is now occupied (shouldn't happen, but guard it).
            log.warning("krakenbot_v1_cached_stone_illegal", q=q_p, r=r_p, ply=state.ply)
            return _smart_legal_fallback(rust_board, "cached_stone_illegal", state.ply)

        # Build board_dict: {(q,r) -> _KPlayer} from Rust board.
        board_dict: dict = {}
        for q, r, p in rust_board.get_stones():  # type: ignore[attr-defined]
            board_dict[(q, r)] = _KPlayer.A if p == 1 else _KPlayer.B

        # current_player: Rust uses 1=P1(A), -1=P2(B). KrakenBot Player: A=1, B=2.
        cur_player = _KPlayer.A if state.current_player == 1 else _KPlayer.B

        # Encode: dynamic bounding box + margin=6, zero padding (infinite-board mode).
        planes, off_q, off_r, h, w = board_to_planes(board_dict, cur_player, margin=6)

        x = planes.unsqueeze(0).to(self._device)
        with torch.no_grad():
            value, pair_logits, moves_left, _chain = self._model(x)

        # Legality set: moves the Rust board considers legal.
        legal_set = set(rust_board.legal_moves())  # type: ignore[attr-defined]

        N = h * w
        single_move_turn = (state.moves_remaining == 1)

        if single_move_turn:
            # Marginalize pair logits to per-cell marginals: logsumexp over partner dim.
            single_logits = pair_logits[0].logsumexp(dim=-1)  # [N]
            sorted_idxs = single_logits.argsort(descending=True)
            chosen1: Optional[tuple[int, int]] = None
            fallback_used = False
            legal_rank = -1

            for rank, idx in enumerate(sorted_idxs.tolist()):
                gq1 = idx // w
                gr1 = idx % w
                q1 = gq1 - off_q
                r1 = gr1 - off_r
                if (q1, r1) in legal_set:
                    chosen1 = (q1, r1)
                    legal_rank = rank
                    break

            if chosen1 is None:
                fallback_used = True
                chosen1 = _smart_legal_fallback(rust_board, "no_legal_single", state.ply)
                legal_rank = -1

            top_logit = float(single_logits.max().item())
            self._write_diag(
                ply=state.ply,
                pair=[(chosen1[0], chosen1[1])],
                top_logit=top_logit,
                value=float(value.item()),
                moves_left=float(moves_left.item()),
                legal_rank=legal_rank,
                fallback_used=fallback_used,
            )
            return chosen1

        # Compound turn (moves_remaining >= 2): decode best LEGAL pair.
        flat = pair_logits[0].reshape(N * N)
        sorted_pair_idxs = flat.argsort(descending=True)

        chosen1 = None
        chosen2: Optional[tuple[int, int]] = None
        fallback_used = False
        legal_rank = -1
        top_logit = float(flat.max().item())

        # Walk sorted pairs; stop at first legal distinct pair.
        # Cap at min(N*N, 500) to bound loop time — N^2 can be large on big boards.
        cap = min(N * N, 500)
        for rank, idx in enumerate(sorted_pair_idxs[:cap].tolist()):
            s1 = idx // N
            s2 = idx % N
            if s1 == s2:
                continue  # diagonal — model already masks, but guard anyway
            gq1, gr1 = s1 // w, s1 % w
            gq2, gr2 = s2 // w, s2 % w
            q1, r1 = gq1 - off_q, gr1 - off_r
            q2, r2 = gq2 - off_q, gr2 - off_r
            if (
                (q1, r1) in legal_set
                and (q2, r2) in legal_set
                and (q1, r1) != (q2, r2)
            ):
                chosen1 = (q1, r1)
                chosen2 = (q2, r2)
                legal_rank = rank
                break

        if chosen1 is None:
            # No legal pair found in top-500 — fall back to two random legal moves.
            fallback_used = True
            chosen1 = _smart_legal_fallback(rust_board, "no_legal_pair", state.ply)
            legal_set2 = legal_set - {chosen1}
            chosen2 = random.choice(list(legal_set2)) if legal_set2 else chosen1
            legal_rank = -1

        # Cache second stone for next call.
        if chosen2 is not None:
            self._pending_move = chosen2

        self._write_diag(
            ply=state.ply,
            pair=[(chosen1[0], chosen1[1]), (chosen2[0] if chosen2 else 0, chosen2[1] if chosen2 else 0)],
            top_logit=top_logit,
            value=float(value.item()),
            moves_left=float(moves_left.item()),
            legal_rank=legal_rank,
            fallback_used=fallback_used,
        )
        return chosen1

    def reset(self) -> None:
        """Clear cached second stone. Call before starting a new game."""
        self._pending_move = None

    def name(self) -> str:
        return self._label

    # ── internals ─────────────────────────────────────────────────────────────

    def _write_diag(
        self,
        ply: int,
        pair: list,
        top_logit: float,
        value: float,
        moves_left: float,
        legal_rank: int,
        fallback_used: bool,
    ) -> None:
        if self._diag is None:
            return
        rec = {
            "ply": ply,
            "pair": pair,
            "top_logit": top_logit,
            "value": value,
            "moves_left": moves_left,
            "legal_rank": legal_rank,
            "fallback_used": fallback_used,
        }
        try:
            with self._diag.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError as exc:
            # Never crash the eval loop over diagnostics.
            log.warning("krakenbot_v1_diag_write_error", exc=str(exc))
