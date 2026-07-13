"""ShrimpBot — BotProtocol wrapper for Cmiller132/hexo-bot ("shrimp").

shrimp is an 8.1M-param support-set graph net with a Gumbel-MCTS move oracle.
BOTH its featurizer and its search are RUST extensions (`shrimp._rust`) that do
not exist in our interpreter and cannot be reimplemented in-adapter without a
multi-week port (forbidden by the "no home-grown minimax" rule anyway). So this
adapter DELEGATES search to shrimp's own code, run in a network-off SUBPROCESS
in shrimp's built venv (hexo_rl/bots/shrimp_worker.py), and never imports shrimp
into our working env. This mirrors scripts/arena/bots/shrimp_child.py prior art;
the difference is this adapter speaks our BotProtocol, not the arena stdio wire.

WHAT IS FIDELITY-GUARANTEED (see reports/tourney/adapters_shrimp.md):
  - the net forward (policy + 65-bin distributional value) and the value decode
    are shrimp's OWN code, byte-for-byte — exact by construction.
  - the Gumbel-MCTS search is shrimp's OWN Rust session with the as-trained
    profile — exact by construction.
WHAT THE ADAPTER OWNS (the seam that can drift):
  - board -> engine-state reconstruction (from unordered get_stones()).
  - compound-turn assembly (search per stone, cache the second).
  - the 65-bin -> scalar reduction FOR DIAGNOSTICS ONLY (softmax over
    linspace(-1,1,65) bin centers, clamp [-1,1] — shrimp's decode_binned_value).
  - a residual fidelity ceiling: the featurizer's most-recent-stone recency flag
    (feature col 8) is order-dependent, and get_stones() is unordered, so the
    LAST opponent turn's intra-turn order is reconstructed deterministically but
    not always identically to how the game was actually played. Measured impact:
    |dvalue| ~ 5e-3, policy-logit shifts that rarely change the argmax.

CHECKPOINT: model_path is a PARAMETER (no weights baked, no per-checkpoint
constants). The tournament checkpoint swap is a one-arg change. shrimp
auto-detects arch from the state dict (infer_net_kwargs_from_state_dict); the
load-bearing SHRIMP_SUPPORT_RADIUS=4 (not in the checkpoint) is set by the worker.

Diagnostics: one JSON line per decision appended to diag_path (never stderr —
64KB pipe-deadlock risk). Default: reports/tourney/shrimp_eval.jsonl.

CROSS-VENV SEAM: the worker MUST run under the hexo-bot venv python
(built shrimp._rust + hexo_engine). This adapter runs under OUR venv. The
worker python path is a parameter (default: the standard sibling-repo venv).
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional

import structlog

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.env import GameState

log = structlog.get_logger()

# Standard sibling-repo locations. All overridable via constructor params so no
# path is baked into the tournament run.
_HEXO_BOT_ROOT = Path("/home/timmy/Work/Hexo/hexo-bot")
_DEFAULT_WORKER_PY = _HEXO_BOT_ROOT / ".venv" / "bin" / "python"
_DEFAULT_CKPT = _HEXO_BOT_ROOT / "models" / "shrimp_main7_infer.pt"
_DEFAULT_PROFILE = _HEXO_BOT_ROOT / "apps" / "showcase" / "profiles" / "shrimp_main_7.toml"
_WORKER_SCRIPT = Path(__file__).with_name("shrimp_worker.py")
_DEFAULT_DIAG = Path("reports") / "tourney" / "shrimp_eval.jsonl"


class ShrimpWorkerError(RuntimeError):
    """A fault surfaced by the shrimp subprocess worker."""


class ShrimpBot(BotProtocol):
    """BotProtocol wrapper delegating search to shrimp's Rust Gumbel-MCTS.

    Args:
        model_path:   shrimp checkpoint (.pt). None -> the placeholder at
                      hexo-bot/models/shrimp_main7_infer.pt. TOURNAMENT SWAP =
                      pass the real checkpoint path here (one-arg change).
        visits:       per-move MCTS visit budget. Default 256 — shrimp's declared
                      deploy budget (arena config + profile eval budget).
        seed:         base search seed. Default 0.
        worker_python: interpreter that runs the worker (the hexo-bot venv, which
                      has shrimp._rust). None -> the standard sibling-repo venv.
        profile:      as-trained search profile TOML. None -> shrimp_main_7.toml.
        label:        name() return. Default "shrimp".
        diag_path:    append JSON-line diagnostics here. None -> default path.
                      Pass False to disable diagnostics entirely.
        startup_timeout / move_timeout: subprocess request timeouts (seconds).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        visits: int = 256,
        seed: int = 0,
        worker_python: Optional[str] = None,
        profile: Optional[str] = None,
        label: str = "shrimp",
        diag_path: Optional[object] = None,
        startup_timeout: float = 120.0,
        move_timeout: float = 120.0,
    ) -> None:
        self._ckpt = str(Path(model_path) if model_path is not None else _DEFAULT_CKPT)
        self._worker_py = str(Path(worker_python) if worker_python is not None else _DEFAULT_WORKER_PY)
        self._profile = str(Path(profile) if profile is not None else _DEFAULT_PROFILE)
        self._visits = int(visits)
        self._seed = int(seed)
        self._label = label
        self._startup_timeout = float(startup_timeout)
        self._move_timeout = float(move_timeout)

        if not Path(self._worker_py).exists():
            raise FileNotFoundError(
                f"shrimp worker python not found at {self._worker_py}. It must be "
                "the hexo-bot venv interpreter (built shrimp._rust + hexo_engine)."
            )
        if not Path(self._ckpt).exists():
            raise FileNotFoundError(f"shrimp checkpoint not found at {self._ckpt}")

        self._pending_move: Optional[tuple[int, int]] = None
        # Opener's OPENING stone in OUR coords — the translation origin passed to
        # the worker (shrimp's engine locks the opening to (0,0); our engine does
        # not). Captured the first time the board shows exactly one opener stone
        # (unambiguous at ply 1/2 of a game started from empty).
        self._origin: Optional[tuple[int, int]] = None
        self._proc: Optional[subprocess.Popen] = None

        if diag_path is False:
            self._diag: Optional[Path] = None
        elif diag_path is None:
            self._diag = Path(_DEFAULT_DIAG)
        else:
            self._diag = Path(diag_path)  # type: ignore[arg-type]
        if self._diag is not None:
            self._diag.parent.mkdir(parents=True, exist_ok=True)

        self._start_worker()

    # ── subprocess lifecycle ──────────────────────────────────────────────────

    def _start_worker(self) -> None:
        # No network: the child only touches local files + CPU. We do not pass a
        # restricted env (torch needs PATH/HOME); the worker self-sets SHRIMP_*.
        self._proc = subprocess.Popen(
            [
                self._worker_py,
                str(_WORKER_SCRIPT),
                "--checkpoint", self._ckpt,
                "--visits", str(self._visits),
                "--seed", str(self._seed),
                "--profile", self._profile,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,  # worker logs/tracebacks -> our stderr (debuggable)
            text=True,
            bufsize=1,
        )
        reply = self._request({"op": "ready"}, timeout=self._startup_timeout)
        if not reply.get("ok"):
            raise ShrimpWorkerError(f"worker failed to become ready: {reply}")
        log.info("shrimp_worker_ready", ckpt=self._ckpt, visits=self._visits)

    def _request(self, req: dict, timeout: float) -> dict:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            raise ShrimpWorkerError("shrimp worker process is not running")
        try:
            proc.stdin.write(json.dumps(req) + "\n")  # type: ignore[union-attr]
            proc.stdin.flush()  # type: ignore[union-attr]
            line = proc.stdout.readline()  # type: ignore[union-attr]
        except (BrokenPipeError, ValueError) as exc:
            raise ShrimpWorkerError(f"worker pipe error: {exc}") from exc
        if not line:
            raise ShrimpWorkerError("worker closed stdout (crashed); see stderr")
        reply = json.loads(line)
        if "error" in reply:
            raise ShrimpWorkerError(f"worker op {req.get('op')!r} failed: {reply['error']}")
        return reply

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps({"op": "quit"}) + "\n")  # type: ignore[union-attr]
                self._proc.stdin.flush()  # type: ignore[union-attr]
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._proc = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ── board bridge ──────────────────────────────────────────────────────────

    @staticmethod
    def _stones_wire(rust_board: object) -> list:
        """(q, r, player) list in OUR player convention (1 opener / -1 responder).

        Board-scan-ordered (get_stones() returns coord-sorted); the worker
        replays these in canonical HTTT turn order. Order within a turn only
        affects the recency flag (documented fidelity ceiling)."""
        return [[int(q), int(r), int(p)] for (q, r, p) in rust_board.get_stones()]  # type: ignore[attr-defined]

    def _resolve_origin(self, stones: list) -> tuple[int, int]:
        """The opener's OPENING stone (translation origin for shrimp).

        Opener stones = player == 1. When exactly one is present (game start),
        it is unambiguously the opening — captured and cached. Once cached, it
        is reused for the rest of the game (the opening stone never moves). If
        the adapter first sees a mid-game board (opener already has >1 stone) and
        no origin was captured, fall back to the coord-min opener stone: this
        may misidentify the opening, but the featurizer is translation-equivariant
        so the evaluation is unaffected AS LONG AS the reconstruction is legal;
        only the ply-1 identity (a minor recency detail) can differ. A warning is
        logged so a caller can supply the origin explicitly if it matters."""
        opener = [(q, r) for q, r, p in stones if p == 1]
        if not opener:
            # Adapter is the opener and the board is empty: shrimp opens at (0,0);
            # our engine will accept (0,0) too. Origin is (0,0) and gets refined
            # once the opening stone is actually on the board.
            return (0, 0)
        if len(opener) == 1:
            self._origin = opener[0]
            return self._origin
        if self._origin is not None and self._origin in opener:
            return self._origin
        # Mid-game first-observation with an ambiguous opening. Deterministic
        # fallback; translation-invariance keeps the eval correct.
        fallback = min(opener)
        log.warning(
            "shrimp_origin_fallback",
            reason="ambiguous_opening_stone",
            n_opener=len(opener),
            chosen=list(fallback),
        )
        self._origin = fallback
        return fallback

    def _legal_fallback(self, rust_board: object, reason: str, ply: int) -> tuple[int, int]:
        legal = rust_board.legal_moves()  # type: ignore[attr-defined]
        if not legal:
            raise ShrimpWorkerError("no legal moves on board")
        log.warning("shrimp_fallback", reason=reason, ply=ply, n_legal=len(legal))
        return random.choice(legal)

    # ── BotProtocol ──────────────────────────────────────────────────────────

    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        # Second stone of a compound turn: replay the position (now including our
        # first stone) so the recency flag + search reflect the true mid-turn
        # state, then return the searched second stone.
        if self._pending_move is not None:
            self._pending_move = None
            move = self._search_stone(state, rust_board, ply=state.ply)
            if rust_board.get(move[0], move[1]) != 0:  # type: ignore[attr-defined]
                log.warning("shrimp_stone2_illegal", move=move, ply=state.ply)
                move = self._legal_fallback(rust_board, "stone2_illegal", state.ply)
            return move

        move = self._search_stone(state, rust_board, ply=state.ply)
        if rust_board.get(move[0], move[1]) != 0:  # type: ignore[attr-defined]
            log.warning("shrimp_stone1_illegal", move=move, ply=state.ply)
            move = self._legal_fallback(rust_board, "stone1_illegal", state.ply)

        # Compound turn: cache marker so the NEXT get_move re-searches the
        # mid-turn position (with our first stone now on the board).
        if state.moves_remaining >= 2:
            self._pending_move = move  # marker only; value re-derived next call
        return move

    def _search_stone(self, state: GameState, rust_board: object, ply: int) -> tuple[int, int]:
        stones = self._stones_wire(rust_board)
        req = {
            "op": "move",
            "stones": stones,
            "current_player": int(state.current_player),
            "moves_remaining": int(state.moves_remaining),
            "ply": int(ply),
            "origin": list(self._resolve_origin(stones)),
        }
        reply = self._request(req, timeout=self._move_timeout)
        move = (int(reply["q"]), int(reply["r"]))
        self._write_diag(
            ply=ply,
            move=move,
            root_value=reply.get("root_value"),
            value=reply.get("value"),
            n_policy=reply.get("n_policy"),
        )
        return move

    def reset(self) -> None:
        """Clear the compound-turn marker + opening origin + tell the worker a
        new game started."""
        self._pending_move = None
        self._origin = None
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._request({"op": "reset"}, timeout=self._move_timeout)
            except ShrimpWorkerError:
                pass

    def name(self) -> str:
        return self._label

    # ── fidelity harness helper (raw forward, no search) ──────────────────────

    def raw_eval(
        self,
        stones: list,
        current_player: int,
        moves_remaining: int = 2,
        origin: Optional[tuple[int, int]] = None,
    ) -> dict:
        """Raw forward_policy_value (no search) for the given position.

        stones: list of (q, r, player) with player in OUR convention (1/-1).
        origin: opener opening stone (translation origin); None -> auto-resolve.
        Returns {"value": float, "policy": [(q, r, logit), ...]} in OUR coord
        frame. Used by the fidelity harness to compare against the sandboxed
        reference."""
        wire = [[int(q), int(r), int(p)] for (q, r, p) in stones]
        org = tuple(origin) if origin is not None else self._resolve_origin(wire)
        req = {
            "op": "eval",
            "stones": wire,
            "current_player": int(current_player),
            "moves_remaining": int(moves_remaining),
            "origin": list(org),
        }
        return self._request(req, timeout=self._move_timeout)

    # ── internals ─────────────────────────────────────────────────────────────

    def _write_diag(self, ply, move, root_value, value, n_policy) -> None:
        if self._diag is None:
            return
        rec = {
            "ply": ply,
            "move": list(move),
            "root_value": root_value,
            "value": value,
            "n_policy": n_policy,
        }
        try:
            with self._diag.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError as exc:
            log.warning("shrimp_diag_write_error", exc=str(exc))
