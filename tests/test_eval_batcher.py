"""D-EVALFOUND C3 — cross-game batched evaluator: correctness gates.

G1 (PRIMARY, the only scatter-correctness proof): with a DETERMINISTIC inference stub
(batch-invariant by construction), N games run all-concurrent must produce BYTE-IDENTICAL
per-game transcripts to the same games run one-at-a-time. This isolates the scatter/
gather/seed/tree-step orchestration from NN-FP variance (which M-VAR showed is nonzero
across batch sizes even on float32, so it can never be the scatter proof).

A mis-scatter (sending game A's policy/value to game B's tree) changes transcripts iff
the stub is board-DEPENDENT — so the stub hashes board state to a distinct policy/value.
"""
from __future__ import annotations

import hashlib

import numpy as np

from hexo_rl.eval.eval_batcher import run_batched_games

ENC = "v6_live2"
N_ACTIONS = 362  # v6_live2 policy_logit_count


def _deterministic_stub(boards):
    """Board-dependent, batch-invariant policy+value. Distinct per board so a
    mis-scatter is observable; identical regardless of which other boards share the
    batch (the property real NN forwards lack — that is why this is the scatter gate)."""
    policies, values = [], []
    for b in boards:
        stones = tuple(sorted((int(q), int(r), int(p)) for q, r, p in b.get_stones()))
        h = hashlib.sha256(repr(stones).encode()).digest()
        seed = int.from_bytes(h[:8], "little")
        rng = np.random.default_rng(seed)
        pol = rng.random(N_ACTIONS)
        pol = (pol / pol.sum()).tolist()
        val = float(rng.random() * 2 - 1)
        policies.append(pol)
        values.append(val)
    return policies, values


class _LowestCoordBot:
    """Deterministic opponent — picks the lexicographically smallest legal move."""

    def get_move(self, state, board):
        return min(board.legal_moves())


def _opp():
    return _LowestCoordBot()


def _transcripts(setups):
    return run_batched_games(
        setups, _deterministic_stub,
        opponent_factory=_opp, encoding=ENC, model_sims=16, temperature=0.0,
        c_puct=1.5, max_plies=60, opening_plies=4,
    )


def test_g1_batched_equals_serial_byte_identical():
    setups = [(i, 1 if i % 2 == 0 else -1) for i in range(8)]  # (seed, model_side)
    batched = _transcripts(setups)                              # all 8 concurrent
    serial = [_transcripts([s])[0] for s in setups]            # one at a time
    assert len(batched) == 8
    for i, (b, s) in enumerate(zip(batched, serial)):
        assert b["moves"] == s["moves"], f"game {i} transcript diverged batched vs serial"
        assert b["winner"] == s["winner"]
        assert b["model_side"] == s["model_side"]


def test_g1_distinct_games_not_all_identical():
    # sanity: the stub + per-game openings actually produce DISTINCT games, so G1 is
    # not trivially passing on identical transcripts.
    setups = [(i, 1 if i % 2 == 0 else -1) for i in range(8)]
    batched = _transcripts(setups)
    move_seqs = {tuple(map(tuple, g["moves"])) for g in batched}
    assert len(move_seqs) > 1, "all games identical — G1 would pass trivially"


def test_g1_repeat_determinism_stub():
    # G3-analog on CPU: same batched run twice → identical.
    setups = [(i, 1 if i % 2 == 0 else -1) for i in range(6)]
    a = _transcripts(setups)
    b = _transcripts(setups)
    assert [g["moves"] for g in a] == [g["moves"] for g in b]


import pytest  # noqa: E402


@pytest.mark.parametrize("n_concurrent", [8, 16, 32])
def test_g1_scatter_scales_across_concurrency(n_concurrent):
    # spec §1d envisions N_CONCURRENT ∈ {8,16,32}: byte-identity must hold at each scale
    # (a scatter bug could be concurrency-dependent). Compare N-concurrent vs one-at-a-time.
    setups = [(i, 1 if i % 2 == 0 else -1) for i in range(n_concurrent)]
    batched = _transcripts(setups)
    serial = [_transcripts([s])[0] for s in setups]
    for i, (b, s) in enumerate(zip(batched, serial)):
        assert b["moves"] == s["moves"], f"N={n_concurrent} game {i} diverged"


# G5 (behavior-neutral reseed: per-game RNG vs old global-RNG serial WR within tolerance)
# is DEFERRED to Phase-3 banked-data validation — it requires the old globally-seeded
# legacy transcripts, which are not in the repo. G1 (scatter), G3 (repeat-determinism here),
# and the C3 measurement (M-TP: batched vs serial |ΔWR|=0.000, 2.24× faster) cover the
# CPU-deterministic + GPU-equivalence axes that ARE checkable in-repo.
