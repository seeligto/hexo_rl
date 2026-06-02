"""PIPE-1 regression — `push_game` with no `position_indices` must store 0,
not the row counter `0..T-1`.

Audit finding PIPE-1 (`codebase_consistency_audit_2026-06-02.md`):
`push_game_impl` defaulted a missing `position_indices` to `i as u16` (the
corpus row index), while the sibling `push_many_impl` correctly defaults to
`0u16`. The corpus loader (`batch_assembly.py:257,363`) calls `push_game` with
NO `position_indices`, so for any corpus with T>100 rows the ply-index loss head
(`losses.py:236`, target = clamp(position_indices/100, 0, 1)) trained rows >=100
toward a SATURATED ~1.0 target — the opposite of the config's documented
"pretrain rows have position_index = 0 -> target ~ 0".

The corpus has no ply-index concept; a missing slice must mean 0.
"""
import numpy as np
from engine import ReplayBuffer

T = 150  # > 100 so the buggy `i as u16` default saturates the ply-index target


def _v6_corpus_rows(t):
    """Minimal valid v6-shaped per-row arrays for push_game (content irrelevant)."""
    policies = np.zeros((t, 362), dtype=np.float32)
    policies[:, 0] = 1.0
    return dict(
        states=np.zeros((t, 8, 19, 19), dtype=np.float16),
        chain_planes=np.zeros((t, 6, 19, 19), dtype=np.float16),
        policies=policies,
        outcomes=np.zeros(t, dtype=np.float32),
        ownership=np.ones((t, 361), dtype=np.uint8),
        winning_line=np.zeros((t, 361), dtype=np.uint8),
    )


def test_corpus_push_game_without_position_indices_defaults_to_zero():
    """A >100-row corpus pushed via push_game with no position_indices must
    store position_index == 0 for every row (not 0..T-1)."""
    buf = ReplayBuffer(capacity=T + 50)
    rows = _v6_corpus_rows(T)
    buf.push_game(
        rows["states"], rows["chain_planes"], rows["policies"],
        rows["outcomes"], rows["ownership"], rows["winning_line"],
        game_id=1, game_length=T,
        # position_indices intentionally omitted — the corpus-loader call shape.
    )
    assert buf.size == T

    # Sample heavily (with replacement) so every stored row is virtually certain
    # to surface — under the bug the high-index rows store 1..T-1 (nonzero).
    _, _, _, _, _, _, _ifs, pos, _vv = buf.sample_batch_with_pos(4000, augment=False)
    assert pos.max() == 0, (
        f"corpus rows must default position_index=0; saw max={int(pos.max())} "
        f"(bug: push_game defaulted to the row counter i, saturating the "
        f"ply-index loss target to ~1.0 for rows >=100)"
    )


def test_push_game_explicit_position_indices_still_roundtrip():
    """Guard: the fix changes only the missing-slice default. When
    position_indices IS provided it must still round-trip unchanged."""
    buf = ReplayBuffer(capacity=T + 50)
    rows = _v6_corpus_rows(T)
    pos_in = np.full(T, 42, dtype=np.uint16)  # distinctive constant, != default 0
    buf.push_game(
        rows["states"], rows["chain_planes"], rows["policies"],
        rows["outcomes"], rows["ownership"], rows["winning_line"],
        game_id=1, game_length=T, position_indices=pos_in,
    )
    _, _, _, _, _, _, _ifs, pos, _vv = buf.sample_batch_with_pos(4000, augment=False)
    assert (pos == 42).all(), "explicit position_indices must store verbatim"
