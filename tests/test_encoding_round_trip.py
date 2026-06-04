"""§172 A6 — cross-encoding round-trip, parameterized over the registry.

Single fixture exercises every registered encoding through the full
Board → planes → model forward → MCTS chain. Adding a new encoding to
`engine/src/encoding/registry.toml` causes this test to run on it
automatically; missing plumbing surfaces structurally.

Coverage per encoding:
  1. Registry stable-instance round-trip.
  2. Board construction via `Board.with_encoding_name`.
  3. Plane export. Single-window encodings exercise `to_tensor()`;
     multi-window encodings (v6w25) verify that `to_tensor()` raises
     the α dispatch guard (engine/src/board/state.rs:761), captured
     via subprocess because PyO3 abort-on-panic terminates the host
     process. The working multi-window path (`get_cluster_views`) is
     still exercised inline.
  4. HexTacToeNet construction + zero-tensor forward; asserts the
     policy head emits `spec.policy_logit_count` logits.
  5. Best-effort state-dict load against `checkpoints/bootstrap_model_<name>.pt`
     when the file exists; runs `validate_against_state_dict` for the
     spec/state-dict shape contract.
  6. MCTS one-sim smoke (Rust `MCTSTree.expand_and_backup`) — works
     for every encoding because the tree path doesn't call `to_tensor`;
     the α deferral lives at the planes-export boundary, not inside MCTS.

α design pointer for v6w25 multi-window selfplay:
`docs/designs/encoding_alpha_multiwindow_selfplay.md` (§172 Phase A7).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from engine import Board, MCTSTree

from hexo_rl.encoding import (
    EncodingSpec,
    all_specs,
    lookup,
    validate_against_state_dict,
)
from hexo_rl.model.network import HexTacToeNet


# Registered encodings — sorted for deterministic test-id ordering.
_REGISTERED: list[str] = sorted(s.name for s in all_specs())


# Repo root (relative to this file → tests/ → repo/).
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _rust_spec(name: str) -> Any:
    """Return the Rust PyRegistrySpec for `name`.

    Used by §173 A3 parity tests — calls the Rust-side registry lookup
    via the PyO3 `RegistrySpec.from_registry` classmethod (§P3.1 — was
    `EncodingSpec.from_registry`; FF.1 cross-class smell fix).
    """
    from engine import RegistrySpec as RustRegistrySpec
    return RustRegistrySpec.from_registry(name)


def _model_kwargs(spec: EncodingSpec) -> dict[str, Any]:
    """Map an EncodingSpec onto HexTacToeNet ctor kwargs.

    HexTacToeNet's `encoding` argument accepts only {"v6", "v6w25", "v8"};
    v7full / v7 / v7e30 / v7mw share the v6 wire format and load with
    encoding="v6". v8_canvas_realness is encoding="v8" + canvas_realness=True
    per §169 A4. Filters / blocks are kept tiny — this test exercises
    shape contracts, not training dynamics.
    """
    common = dict(filters=8, res_blocks=1)
    # v6tp shares the v6 wire geometry (19×19, 362 logits, single-window);
    # it only keeps two extra source planes (16/17), so in_channels=10 while
    # the model's encoding geometry is still "v6".
    if spec.name in ("v6", "v6tp", "v6_live2", "v6_live2_anchored", "v7full", "v7", "v7e30", "v7mw"):
        return {**common, "encoding": "v6", "board_size": spec.trunk_size,
                "in_channels": spec.n_planes}
    if spec.name == "v6w25":
        return {**common, "encoding": "v6w25", "board_size": spec.trunk_size,
                "in_channels": spec.n_planes}
    if spec.name == "v8":
        return {**common, "encoding": "v8", "board_size": spec.trunk_size,
                "in_channels": spec.n_planes}
    if spec.name == "v8_canvas_realness":
        return {**common, "encoding": "v8", "board_size": spec.trunk_size,
                "in_channels": spec.n_planes, "canvas_realness": True}
    raise AssertionError(
        f"unmapped encoding {spec.name!r} — extend _model_kwargs when "
        f"adding a new registry entry."
    )


def _bootstrap_ckpt_for(name: str) -> Path | None:
    """Return the bootstrap-model checkpoint for `name` if present.

    Layout: checkpoints/bootstrap_model_<name>.pt at repo root.
    """
    p = _REPO_ROOT / "checkpoints" / f"bootstrap_model_{name}.pt"
    return p if p.is_file() else None


def _uniform_policy(n: int) -> list[float]:
    return [1.0 / n] * n


@pytest.mark.parametrize("encoding_name", _REGISTERED)
def test_round_trip(encoding_name: str) -> None:
    spec = lookup(encoding_name)

    # 1. Registry stable-instance round-trip.
    assert lookup(encoding_name) is spec, (
        f"{encoding_name}: lookup() returned non-stable instance"
    )

    # 2. Board construction.
    board = Board.with_encoding_name(encoding_name)
    assert board.size == spec.board_size, (
        f"{encoding_name}: board.size {board.size} != spec.board_size "
        f"{spec.board_size}"
    )

    # 3. Plane export.
    if spec.is_multi_window:
        # The α dispatch guard surfaces as a Rust panic which aborts the
        # host process; exercise it in a subprocess and assert the α
        # deferral pointer is present in stderr.
        snippet = (
            f"import engine; "
            f"engine.Board.with_encoding_name({encoding_name!r}).to_tensor()"
        )
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode != 0, (
            f"{encoding_name}: to_tensor() unexpectedly succeeded for "
            f"multi-window encoding (α dispatch guard regressed)"
        )
        assert "α" in result.stderr, (
            f"{encoding_name}: α deferral pointer missing from panic stderr; "
            f"see docs/designs/encoding_alpha_multiwindow_selfplay.md.\n"
            f"stderr=\n{result.stderr}"
        )
        # Working multi-window plane path: cluster views.
        views, centers = board.get_cluster_views()
        assert len(views) == len(centers)
        cw = spec.cluster_window_size
        assert cw is not None, (
            f"{encoding_name}: is_multi_window=True but "
            f"cluster_window_size is None (registry validator should have "
            f"caught this)"
        )
        for v in views:
            assert v.shape == (2, cw, cw), (
                f"{encoding_name}: cluster view shape {v.shape} != "
                f"(2, {cw}, {cw})"
            )
    else:
        flat = board.to_tensor()
        # The engine emits the legacy 18-plane wire format regardless of
        # spec.n_planes (the model side slices via input_channels). The
        # spec invariant we assert here is that the flat length divides
        # cleanly by board_size² and yields an integral plane count.
        n_cells = spec.board_size * spec.board_size
        assert len(flat) % n_cells == 0, (
            f"{encoding_name}: to_tensor len {len(flat)} not divisible by "
            f"{n_cells} (board_size {spec.board_size})"
        )
        emitted = len(flat) // n_cells
        assert emitted >= spec.n_planes, (
            f"{encoding_name}: emitted {emitted} planes but spec declares "
            f"n_planes={spec.n_planes} (engine truncates wire-format below "
            f"the spec channel count)"
        )

    # 4. Model construction + forward.
    kwargs = _model_kwargs(spec)
    model = HexTacToeNet(**kwargs).eval()
    x = torch.zeros(1, spec.n_planes, spec.trunk_size, spec.trunk_size)
    with torch.no_grad():
        out = model(x)
    # forward() returns (log_policy, value, value_logit) per the inference
    # contract (hexo_rl/model/network.py:629-636).
    assert isinstance(out, tuple) and len(out) >= 3, (
        f"{encoding_name}: forward returned {type(out).__name__} of length "
        f"{len(out) if isinstance(out, tuple) else '?'}, expected ≥3-tuple"
    )
    log_policy, value, value_logit = out[0], out[1], out[2]
    assert log_policy.shape == (1, spec.policy_logit_count), (
        f"{encoding_name}: log_policy shape {tuple(log_policy.shape)} != "
        f"(1, {spec.policy_logit_count})"
    )
    assert value.shape == (1, 1)
    assert value_logit.shape == (1, 1)

    # 5. State-dict load (best-effort — only if a fixture ckpt exists).
    ckpt_path = _bootstrap_ckpt_for(encoding_name)
    if ckpt_path is not None:
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # bootstrap_model_*.pt files are stored as raw state-dicts (no
        # outer wrapper); newer saves use a {'model_state': ..., 'metadata': ...}
        # envelope per §172 A5. Accept both.
        if isinstance(loaded, dict) and "model_state" in loaded:
            state = loaded["model_state"]
        else:
            state = loaded
        assert isinstance(state, dict)
        # Spec-vs-state-dict shape contract; raises ShapeMismatchError on
        # disagreement.
        validate_against_state_dict(spec, state)

    # 6. MCTS one-sim smoke. The Rust MCTSTree path doesn't itself call
    # `to_tensor`; the α deferral lives at the planes-export boundary
    # (step 3), not inside MCTS. Policy length is the tree's internal
    # action-space convention: board_size² + 1.
    tree = MCTSTree(c_puct=1.5)
    tree.new_game(board)
    leaves = tree.select_leaves(1)
    if leaves:
        n = board.size * board.size + 1
        tree.expand_and_backup(
            [_uniform_policy(n) for _ in leaves],
            [0.0] * len(leaves),
        )
    assert tree.root_visits() <= 1


# ---------------------------------------------------------------------------
# §173 A3 — Rust ↔ Python parity tests for new helper fields + properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoding_name", _REGISTERED)
def test_helper_parity_rust_python(encoding_name: str) -> None:
    """§173 A3: Python EncodingSpec properties match Rust RegistrySpec methods.

    Covers all 5 helpers (n_cells, state_stride, chain_stride, aux_stride,
    policy_stride) and both new fields (kept_plane_indices, n_source_planes).
    """
    py = lookup(encoding_name)
    rs = _rust_spec(encoding_name)

    # n_cells: trunk_size² (A3 semantic fix — was board_size²).
    # Wave 8 Batch A FF.2 — py and rs are the same object (engine.RegistrySpec
    # type alias); parity checks now defensively assert attribute identity.
    assert py.n_cells == rs.n_cells, (
        f"{encoding_name}: py.n_cells={py.n_cells} != rs.n_cells={rs.n_cells}"
    )

    # state_stride = n_planes * n_cells.
    assert py.state_stride == rs.state_stride, (
        f"{encoding_name}: py.state_stride={py.state_stride} != "
        f"rs.state_stride={rs.state_stride}"
    )

    # chain_stride = N_CHAIN_PLANES(6) * n_cells.
    assert py.chain_stride == rs.chain_stride, (
        f"{encoding_name}: py.chain_stride={py.chain_stride} != "
        f"rs.chain_stride={rs.chain_stride}"
    )

    # aux_stride = n_cells.
    assert py.aux_stride == rs.aux_stride, (
        f"{encoding_name}: py.aux_stride={py.aux_stride} != "
        f"rs.aux_stride={rs.aux_stride}"
    )

    # policy_stride = policy_logit_count.
    assert py.policy_stride == rs.policy_stride, (
        f"{encoding_name}: py.policy_stride={py.policy_stride} != "
        f"rs.policy_stride={rs.policy_stride}"
    )

    # kept_plane_indices (new §173 A3 TOML field).
    assert tuple(py.kept_plane_indices) == tuple(rs.kept_plane_indices), (
        f"{encoding_name}: py.kept_plane_indices={py.kept_plane_indices} != "
        f"rs.kept_plane_indices={tuple(rs.kept_plane_indices)}"
    )

    # n_source_planes (new §173 A3 TOML field).
    assert py.n_source_planes == rs.n_source_planes, (
        f"{encoding_name}: py.n_source_planes={py.n_source_planes} != "
        f"rs.n_source_planes={rs.n_source_planes}"
    )
