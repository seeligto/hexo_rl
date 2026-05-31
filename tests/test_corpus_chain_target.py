"""§102.a regression — corpus chain targets at NPZ load.

Post-§97 the bootstrap corpus NPZ is 18-plane (chain planes removed from input).
`batch_assembly.load_pretrained_buffer` used to pad
``pre_chain = np.zeros((T, 6, 19, 19))`` — so corpus rows carried an all-zero
chain target, pulling the chain head toward zero on the pretrain fraction of
every mixed batch. The §102.a fix computes chain planes from the stored
stone planes at NPZ load, byte-exact with the canonical Python kernel
``_compute_chain_planes`` (same function the production training pipeline
uses in ``batch_assembly.py`` / ``bootstrap/*``).

This module pins:
    1. Loader correctness — corpus-loaded chain planes match the canonical
       Python kernel applied to the same stones (byte-exact at float16).
    2. Training step smoke — a mixed corpus+self-play batch produces a finite,
       non-zero chain loss and the corpus-row targets are not pinned to zero.

Post-`00b7d2b` the Rust PyO3 surface `engine.compute_chain_planes` is gone;
production chain-plane computation lives in Python (`_compute_chain_planes`)
on the corpus loader side and in Rust (`encode_chain_planes`, in-crate) on
the self-play worker side. The Rust kernel has its own in-crate parity
tests at `engine/src/board/state/encode.rs`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pytest
import torch

from hexo_rl.env.game_state import _CHAIN_CAP, _compute_chain_planes
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.training.batch_assembly import load_pretrained_buffer
from hexo_rl.training.losses import compute_chain_loss
from hexo_rl.training.trainer import Trainer

_V6 = _lookup_encoding("v6")
BOARD_SIZE: int = _V6.board_size
KEPT_PLANE_INDICES = list(_V6.kept_plane_indices)

HALF = (BOARD_SIZE - 1) // 2  # 9


def _zero_state() -> np.ndarray:
    """One 18-plane state — caller populates planes 0 (cur) and 8 (opp)."""
    return np.zeros((18, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)


def _set_cur(state: np.ndarray, q: int, r: int) -> None:
    state[0, q + HALF, r + HALF] = np.float16(1.0)


def _set_opp(state: np.ndarray, q: int, r: int) -> None:
    state[8, q + HALF, r + HALF] = np.float16(1.0)


def _position_open_three() -> np.ndarray:
    s = _zero_state()
    for q in (-1, 0, 1):
        _set_cur(s, q, 0)
    return s


def _position_mixed() -> np.ndarray:
    s = _zero_state()
    for q in (0, 1, 2):
        _set_cur(s, q, 0)
    _set_opp(s, 3, 0)
    _set_opp(s, -1, 0)
    for r in (0, 1, 2):
        _set_cur(s, -2, r)
    return s


def _position_axis2_run() -> np.ndarray:
    s = _zero_state()
    for k in (-2, -1, 0, 1):
        _set_cur(s, k, -k)
    _set_opp(s, 3, -3)
    return s


POSITIONS: list[Tuple[str, Callable[[], np.ndarray]]] = [
    ("open_three", _position_open_three),
    ("mixed", _position_mixed),
    ("axis2_run", _position_axis2_run),
]


def _write_corpus(path: Path, states18: np.ndarray) -> None:
    """Write an 8-plane NPZ (HEXB v6) with the keys expected by load_pretrained_buffer.

    `states18` is the full 18-plane tensor from to_tensor(); we slice to 8
    planes here so the NPZ matches the new export_corpus_npz.py output format.
    """
    T = states18.shape[0]
    # Slice 18-plane → 8-plane before saving (native HEXB v6 corpus format).
    states8 = np.ascontiguousarray(states18[:, KEPT_PLANE_INDICES])
    policies = np.zeros((T, 362), dtype=np.float32)
    policies[:, 0] = 1.0  # one-hot placeholder; parity test does not use this
    outcomes = np.zeros(T, dtype=np.float32)
    np.savez(path, states=states8, policies=policies, outcomes=outcomes)


def _load(path: Path):
    mixing_cfg = {"pretrained_buffer_path": str(path)}
    config = {"seed": 0}
    emitted: list[dict] = []
    return load_pretrained_buffer(
        mixing_cfg, config, emitted.append, buffer_size=0, buffer_capacity=0
    )


# ── Test 1 — Loader matches canonical Python kernel ───────────────────────────

def _reference_chain(cur: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """Canonical Python chain-plane reference.

    Matches the normalisation used by the corpus loader in
    `batch_assembly.load_pretrained_buffer` / `_compute_chain_planes`-callers:
    int8 planes from `_compute_chain_planes` cast to float32 and divided by
    `_CHAIN_CAP` (6.0), then truncated to float16 for buffer storage.
    """
    planes = _compute_chain_planes(cur, opp)  # int8 (6, 19, 19)
    return (planes.astype(np.float32) / float(_CHAIN_CAP)).astype(np.float16)


def test_corpus_chain_planes_match_python_kernel(tmp_path: Path) -> None:
    """Corpus-loaded chain planes match the canonical Python kernel at f16."""
    states = np.stack([fn() for _, fn in POSITIONS], axis=0)  # (T, 18, 19, 19)
    corpus_path = tmp_path / "corpus.npz"
    _write_corpus(corpus_path, states)

    buf = _load(corpus_path)
    assert buf is not None, "load_pretrained_buffer returned None for a live NPZ"

    # The sampler is weighted uniform with replacement (game_ids are all −1
    # in this synthetic NPZ, so the correlation guard is a no-op and duplicate
    # rows are possible). Match each sampled row against the full expected
    # set — state planes (0 = cur, 8 = opp) anchor the row to its expected
    # chain. augment=False selects sym_idx=0 (identity) so scatter is a
    # pure copy.
    expected_chains = np.empty((len(POSITIONS), 6, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    state_keys: list[bytes] = []
    for i in range(len(POSITIONS)):
        cur = states[i, 0].astype(np.float32)
        opp = states[i, 8].astype(np.float32)
        expected_chains[i] = _reference_chain(cur, opp)
        state_keys.append(cur.tobytes() + opp.tobytes())

    s_s, c_s, *_ = buf.sample_batch(len(POSITIONS), False)
    assert c_s.shape == (len(POSITIONS), 6, BOARD_SIZE, BOARD_SIZE)
    assert c_s.dtype == np.float16
    any_row_nonzero = False
    for i in range(len(POSITIONS)):
        cur_i = np.asarray(s_s[i, 0], dtype=np.float32)
        opp_i = np.asarray(s_s[i, 4], dtype=np.float32)  # HEXB v6: opp at buf idx 4
        key = cur_i.tobytes() + opp_i.tobytes()
        try:
            src_idx = state_keys.index(key)
        except ValueError:
            pytest.fail(f"sampled row {i} state does not match any input position")
        actual = np.asarray(c_s[i])
        if not np.all(actual == 0):
            any_row_nonzero = True
        assert np.array_equal(actual, expected_chains[src_idx]), (
            f"row {i} (source position #{src_idx}): sampled chain differs from "
            f"canonical Python kernel output"
        )
    assert any_row_nonzero, (
        "every sampled chain plane was zero — §102.a fix regressed or the "
        "corpus stone planes were empty"
    )


# ── Test 2 — Training step on mixed batch — chain loss is real ────────────────

_FAST_CONFIG = {
    "board_size":           BOARD_SIZE,
    "res_blocks":           1,
    "filters":              16,
    "batch_size":           8,
    "lr":                   1e-3,
    "weight_decay":         1e-4,
    "checkpoint_interval":  999,
    "log_interval":         999,
    "torch_compile":        False,
    "aux_chain_weight":     1.0,  # enable chain head + loss path
    "fp16":                 False,
}


def _make_tiny_trainer(tmp_path: Path) -> Trainer:
    model = HexTacToeNet(
        board_size=BOARD_SIZE,
        in_channels=18,
        filters=16,
        res_blocks=1,
        se_reduction_ratio=4,
    )
    return Trainer(
        model, _FAST_CONFIG,
        checkpoint_dir=tmp_path,
        device=torch.device("cpu"),
    )


def test_mixed_batch_chain_loss_uses_nonzero_corpus_targets(tmp_path: Path) -> None:
    """Mixed corpus+self-play batch: chain loss is finite, and the corpus-row
    targets carry real chain values — not the pre-§102.a zero padding."""
    # 4 corpus rows + 4 self-play rows, both with real stones.
    corpus_states = np.stack([_position_open_three(), _position_mixed(),
                              _position_axis2_run(),  _position_open_three()], axis=0)
    self_states   = np.stack([_position_mixed(),      _position_axis2_run(),
                              _position_open_three(), _position_mixed()],      axis=0)

    corpus_path = tmp_path / "corpus.npz"
    _write_corpus(corpus_path, corpus_states)

    buf = _load(corpus_path)
    assert buf is not None

    # Pull the corpus chain planes straight from the buffer (augment=False).
    _, corpus_chain, _, _, _, _, _, _vv = buf.sample_batch(4, False)
    assert corpus_chain.dtype == np.float16
    assert corpus_chain.shape == (4, 6, BOARD_SIZE, BOARD_SIZE)
    # The §102.a bug would leave corpus_chain all zeros. Guard against that.
    assert not np.all(corpus_chain == 0), (
        "corpus chain planes are all zero — §102.a fix regressed: chain targets "
        "are not being computed at NPZ load"
    )

    # Build self-play chain targets via the canonical Python kernel — same
    # function the corpus loader uses internally. The Rust self-play worker
    # writes chain planes via the in-crate `encode_chain_planes` kernel
    # (covered by `engine/src/board/state/encode.rs` tests); we mirror the
    # Python normalisation here so the loss path has comparable targets.
    self_chain = np.empty((4, 6, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    for i in range(4):
        cur = self_states[i, 0].astype(np.float32)
        opp = self_states[i, 8].astype(np.float32)
        self_chain[i] = _reference_chain(cur, opp)

    # Concatenate into a single 8-row batch.
    states       = np.concatenate([corpus_states, self_states], axis=0)
    chain_planes = np.concatenate([corpus_chain, self_chain],   axis=0)
    policies     = np.zeros((8, 362), dtype=np.float32)
    policies[:, 0] = 1.0
    outcomes     = np.zeros(8, dtype=np.float32)
    ownership    = np.ones((8, 19, 19), dtype=np.uint8)
    winning_line = np.zeros((8, 19, 19), dtype=np.uint8)
    is_full_search = np.ones(8, dtype=np.uint8)

    trainer = _make_tiny_trainer(tmp_path)

    # Predict chain manually to verify the loss-producing path against a
    # direct compute_chain_loss call — confirms the loss is non-zero on both
    # halves with an untrained network.
    with torch.no_grad():
        x = torch.from_numpy(states).float()
        out = trainer.model(x, chain=True)
        chain_pred = out[-1]  # (8, 6, 19, 19) float32
        chain_target = torch.from_numpy(chain_planes).float()
        full_loss   = compute_chain_loss(chain_pred, chain_target)
        corpus_loss = compute_chain_loss(chain_pred[:4], chain_target[:4])
        self_loss   = compute_chain_loss(chain_pred[4:], chain_target[4:])

    assert torch.isfinite(full_loss), "chain loss on mixed batch is non-finite"
    assert corpus_loss.item() > 0.0, (
        f"corpus-half chain loss is exactly zero ({corpus_loss.item()}): the "
        "corpus target path is still degenerate. With real chain targets and "
        "untrained network predictions this must be strictly positive."
    )
    assert self_loss.item() > 0.0, "self-play-half chain loss is unexpectedly zero"

    # Now run a full train step — verifies the mask/weight plumbing accepts
    # the corpus chain targets end-to-end without raising.
    result = trainer._train_on_batch(
        states, policies, outcomes,
        chain_planes=chain_planes,
        ownership_targets=ownership,
        threat_targets=winning_line,
        is_full_search=is_full_search,
        n_pretrain=4,
    )
    assert "loss" in result
    assert np.isfinite(result["loss"]), f"train_step total loss not finite: {result['loss']}"
