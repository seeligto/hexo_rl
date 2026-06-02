"""B3 / PIPE-5 — `_augment_recent_rows` opp_slot must be a required arg.

The function recomputes chain planes from `states_f32[i, opp_slot]`. The old
`opp_slot: int = 4` default is correct ONLY for the v6 family (kept opp-slot 4);
under v6_live2 the opp stone is kept-slot 1, so any caller that omitted the arg
silently corrupted the recompute under the *current adopted encoding*. Both
production callers already pass the registry-derived `opp_stone_slot(encoding)`,
so dropping the default turns an omission into a loud TypeError instead of
silent corruption.
"""
import numpy as np
import pytest

from hexo_rl.training.batch_assembly import _augment_recent_rows


def _v6_batch(n=2):
    return (
        np.zeros((n, 8, 19, 19), dtype=np.float32),
        np.zeros((n, 6, 19, 19), dtype=np.float32),
        np.zeros((n, 362), dtype=np.float32),
        np.zeros((n, 361), dtype=np.uint8),
        np.zeros((n, 361), dtype=np.uint8),
    )


def test_opp_slot_is_required_no_silent_v6_default():
    s, c, p, own, wl = _v6_batch()
    with pytest.raises(TypeError):
        _augment_recent_rows(s, c, p, own, wl, augment=False)  # opp_slot omitted
