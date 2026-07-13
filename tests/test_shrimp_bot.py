"""ShrimpBot adapter tests + 20-position fidelity check.

The fidelity check compares the adapter's raw_eval() (which drives
hexo_rl/bots/shrimp_worker.py through the BotProtocol bridge) against a
sandboxed reference (scripts/tourney/shrimp_fidelity_ref.py, run directly in the
hexo-bot venv with NO adapter in the loop). Both invoke identical shrimp code,
so a value/policy mismatch localizes to the adapter's state-reconstruction +
wire seam — the ONLY thing the adapter owns.

Skips cleanly when the hexo-bot venv or the placeholder checkpoint is absent, so
CI without the sibling repo does not fail. Marked `integration` (spawns the
shrimp subprocess); run with `pytest -m integration tests/test_shrimp_bot.py`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from engine import Board
from hexo_rl.env import GameState

_HEXO_BOT = Path("/home/timmy/Work/Hexo/hexo-bot")
_WORKER_PY = _HEXO_BOT / ".venv" / "bin" / "python"
_CKPT = _HEXO_BOT / "models" / "shrimp_main7_infer.pt"
_REF_SCRIPT = Path(__file__).parents[1] / "scripts" / "tourney" / "shrimp_fidelity_ref.py"

_have_sandbox = _WORKER_PY.exists() and _CKPT.exists() and _REF_SCRIPT.exists()
pytestmark = pytest.mark.skipif(
    not _have_sandbox,
    reason="hexo-bot venv / placeholder checkpoint / ref script not present",
)

# Fidelity tolerances. The forward is shrimp's OWN code on both sides, so equal
# featurizer inputs give bit-identical outputs; the only drift source is the
# recency flag (feature col 8) when the last opponent turn's intra-turn order
# differs. Both adapter and reference use the SAME canonical reconstruction, so
# they should agree EXACTLY; the tolerance is a floating-point cushion, not a
# model-difference budget.
VALUE_TOL = 1e-5
POLICY_TOL = 1e-4


def _fixed_positions() -> list[dict]:
    """20 fixed, legal, reproducible positions via our engine.

    Deterministic pseudo-play from a fixed seed: apply legal moves in a fixed
    order to reach a spread of ply depths, snapshotting the board at 20 depths.
    Returns dicts of {id, stones(our convention), current_player, moves_remaining}.
    """
    positions: list[dict] = []
    board = Board()
    # A fixed opening + reproducible move walk. rng seeded so re-runs match.
    import random

    rng = random.Random(20240517)
    snap_at = list(range(1, 41, 2))  # 20 odd depths: 1,3,...,39
    ply = 0
    origin: tuple[int, int] | None = None  # opener opening stone (translation origin)
    while len(positions) < 20:
        legal = board.legal_moves()
        if not legal or board.winner() is not None:
            break
        # Prefer near-origin moves so boards stay compact + comparable.
        legal.sort(key=lambda qr: (abs(qr[0]) + abs(qr[1]) + abs(qr[0] + qr[1]), qr))
        move = legal[rng.randrange(min(6, len(legal)))]
        if ply == 0:
            origin = (int(move[0]), int(move[1]))  # opener's opening stone
        board.apply_move(*move)
        ply += 1
        if ply in snap_at:
            positions.append(
                {
                    "id": len(positions),
                    "stones": [[int(q), int(r), int(p)] for q, r, p in board.get_stones()],
                    "current_player": int(board.current_player),
                    "moves_remaining": int(board.moves_remaining),
                    "origin": list(origin),
                }
            )
    return positions


def _run_reference(positions: list[dict]) -> dict[int, dict]:
    """Batch the positions through the sandboxed reference; return {id: reply}."""
    proc = subprocess.Popen(
        [str(_WORKER_PY), str(_REF_SCRIPT), "--checkpoint", str(_CKPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
    )
    out: dict[int, dict] = {}
    try:
        for pos in positions:
            proc.stdin.write(json.dumps(pos) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            assert line, "reference closed stdout"
            rep = json.loads(line)
            out[rep["id"]] = rep
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)
    return out


@pytest.fixture(scope="module")
def bot():
    from hexo_rl.bots.shrimp_bot import ShrimpBot

    b = ShrimpBot(visits=256, seed=0, diag_path=False)
    yield b
    b.close()


@pytest.mark.integration
def test_name(bot):
    assert bot.name() == "shrimp"


@pytest.mark.integration
def test_returns_legal_move_fresh_board(bot):
    board = Board()
    board.apply_move(0, 0)  # opener; now responder to move (compound turn)
    state = GameState.from_board(board)
    bot.reset()
    move = bot.get_move(state, board)
    assert move in set(board.legal_moves()), f"{move} not legal"


@pytest.mark.integration
def test_fidelity_20_positions(bot):
    """Adapter raw_eval vs sandboxed reference on 20 fixed positions.

    Reports max |dvalue| and max |dpolicy_logit| across all positions; asserts
    both within tolerance. A failure = an adapter bridge bug (fix before ship)."""
    positions = _fixed_positions()
    assert len(positions) == 20, f"got {len(positions)} positions"

    ref = _run_reference(positions)

    max_dv = 0.0
    max_dp = 0.0
    rows = []
    for pos in positions:
        pid = pos["id"]
        adapter_out = bot.raw_eval(
            [tuple(s) for s in pos["stones"]],
            pos["current_player"],
            pos["moves_remaining"],
            origin=tuple(pos["origin"]),
        )
        r = ref[pid]
        dv = abs(adapter_out["value"] - r["value"])
        # align policy by (q, r)
        a_pol = {(q, rr): lg for q, rr, lg in adapter_out["policy"]}
        r_pol = {(q, rr): lg for q, rr, lg in r["policy"]}
        assert set(a_pol) == set(r_pol), f"pos {pid}: legal-cell set mismatch"
        dp = max((abs(a_pol[k] - r_pol[k]) for k in a_pol), default=0.0)
        max_dv = max(max_dv, dv)
        max_dp = max(max_dp, dp)
        rows.append((pid, len(a_pol), dv, dp))

    # Emit a table to stderr for the report (visible under -s).
    print("\n=== shrimp fidelity: 20 positions ===", file=sys.stderr)
    print(f"{'id':>3} {'n_legal':>7} {'|dvalue|':>12} {'|dpolicy|':>12}", file=sys.stderr)
    for pid, n, dv, dp in rows:
        print(f"{pid:>3} {n:>7} {dv:>12.3e} {dp:>12.3e}", file=sys.stderr)
    print(f"MAX  |dvalue|={max_dv:.3e}  |dpolicy|={max_dp:.3e}", file=sys.stderr)

    assert max_dv <= VALUE_TOL, f"value fidelity: max |dvalue|={max_dv:.3e} > {VALUE_TOL}"
    assert max_dp <= POLICY_TOL, f"policy fidelity: max |dpolicy|={max_dp:.3e} > {POLICY_TOL}"
