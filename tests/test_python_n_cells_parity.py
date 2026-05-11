"""§173 A3 — Regression test: n_cells == trunk_size² (not board_size²).

Pins the §173 A3 semantic fix. The pre-existing bug returned board_size²
which was wrong for multi-window v6w25 (361 instead of 625). The fix
changes Python EncodingSpec.n_cells to trunk_size² matching Rust.

v6w25 is the load-bearing case: board_size=25, trunk_size=25, cluster_window_size=25.
All existing current encodings have trunk_size==board_size so the fix is
transparent at present; but is essential for future canvas-larger-than-trunk
encodings.
"""
from __future__ import annotations

import pytest

from hexo_rl.encoding import lookup


@pytest.mark.parametrize("name,expected_n_cells", [
    ("v6", 361),           # 19*19
    ("v7full", 361),       # 19*19
    ("v6w25", 625),        # 25*25 (trunk_size=25)
    ("v8", 625),           # 25*25
    ("v8_canvas_realness", 625),  # 25*25
])
def test_n_cells_equals_trunk_size_squared(name: str, expected_n_cells: int) -> None:
    """n_cells must equal trunk_size² for all registered encodings."""
    spec = lookup(name)
    assert spec.n_cells == spec.trunk_size ** 2, (
        f"{name}: n_cells={spec.n_cells} != trunk_size²={spec.trunk_size ** 2}"
    )
    assert spec.n_cells == expected_n_cells, (
        f"{name}: n_cells={spec.n_cells} != expected {expected_n_cells}"
    )


def test_v6w25_n_cells_is_625_not_361() -> None:
    """Regression pin: v6w25 was silently returning 361 (board_size²=19²=361
    in old code) but trunk_size=25 gives 625. This test would have failed
    pre-fix, catching the semantic error."""
    spec = lookup("v6w25")
    assert spec.n_cells == 625, (
        f"v6w25 n_cells={spec.n_cells}; expected 625 (trunk_size=25). "
        "If 361, the n_cells bug was re-introduced."
    )
    assert spec.n_cells != 361, (
        "v6w25 n_cells == 361 = 19² — board_size² regression re-introduced!"
    )
