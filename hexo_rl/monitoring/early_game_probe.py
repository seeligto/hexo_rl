"""Early-game policy-entropy probe.

Fixed fixture of N reproducible early-game positions. Forward-pass only,
no gradient. Emits a handful of scalars that expose off-canonical
early-game policy collapse — the pathology documented in §116:

    curr_10k H(policy) on ply 2–7 off-canonical probe positions ≈ 5.64–5.70
    (near-uniform over 362 actions), vs bootstrap H ≈ 2.17–3.99.

Cadence: invoked from the training loop alongside ``_emit_training_events``
so overhead rides on the existing log_interval (default 10). Probe itself
is the shape of one ``model.forward`` on a batch of 10; dominant cost is
the H→D copy of the pre-pinned tensor (~0.1 ms on RTX 3070).

Entropy is computed on the **legal-action renormalized distribution**, not
the full 362-way softmax. This matches `scripts/diag_early_game.py` (§116)
so the probe numbers sit on the same scale as the diag ladder: bootstrap-v4
reads ≈ 2.2–4.0 nat per ply, the collapsed curr_10k reads ≈ 5.4–5.7.
Full-softmax entropy would put both models near `log(362) ≈ 5.89` on the
pathological checkpoints, masking the very signal we care about.

Fixture layout (``fixtures/early_game_probe_v1.npz``):

    states    : (N, 8, 19, 19)   float16  — HEXB v6 8-plane slice of GameState.to_tensor()
    plies     : (N,)             int32    — ply index for each state (0, 1, 2, …)
    seeds     : (N,)             int32    — RNG seed used to generate each state
    legal_mask: (N, 362)         uint8    — 1 at flat-action indices that are legal
    version   : () int32                   — fixture schema version (3)

Determinism guarantee: regenerating the fixture from a clean tree must
produce a byte-identical file. Sequence seeding is chosen so that a single
missing entry can be re-derived without shifting the others.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_FIXTURE_VERSION: int = 3
_FIXTURE_SEED: int = 42
_FIXTURE_PLIES: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 7, 10, 15, 20)


REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _fixture_path_for_encoding(encoding_name: str) -> Path:
    if encoding_name == "v6":
        return REPO_ROOT / "fixtures" / "early_game_probe_v1.npz"
    return REPO_ROOT / "fixtures" / f"early_game_probe_{encoding_name}_v1.npz"


DEFAULT_FIXTURE_PATH: Path = _fixture_path_for_encoding("v6")


@dataclass
class _FixturePayload:
    states:     np.ndarray  # (N, 8, B, B) float16 — encoding-aware 8-plane
    plies:      np.ndarray  # (N,) int32
    seeds:      np.ndarray  # (N,) int32
    legal_mask: np.ndarray  # (N, A) uint8  where A = policy_logit_count for encoding


def _generate_fixture_payload(
    target_plies: Tuple[int, ...] = _FIXTURE_PLIES,
    seed: int = _FIXTURE_SEED,
    encoding_name: str = "v6",
) -> _FixturePayload:
    """Build the fixture from scratch.

    For each target ply, seed ``(seed + ply)`` into ``random`` + ``np.random``
    and play ``ply`` uniformly-random legal moves from a fresh board. Records
    the resulting 8-plane tensor (encoding-aware) plus the legal-action mask.
    """
    # Deferred imports: keep the import graph cold when probe isn't used.
    from engine import Board
    from hexo_rl.env.game_state import GameState
    from hexo_rl.encoding import lookup as _lookup_encoding

    spec = _lookup_encoding(encoding_name)
    # §P5-CT P1-1: slice the RESOLVED encoding's kept planes (v6→8, v6tp→10,
    # v6_live2→4), not the module-level v6 set — else a non-8-plane fixture
    # mismatches the model forward and the probe goes dark.
    KEPT_PLANE_INDICES = list(spec.kept_plane_indices)
    board_size = spec.board_size
    n_actions = spec.policy_logit_count

    states_out: List[np.ndarray] = []
    masks_out:  List[np.ndarray] = []
    for target_ply in target_plies:
        per_ply_seed = int(seed + target_ply)
        random.seed(per_ply_seed)
        np.random.seed(per_ply_seed)

        board = Board.with_encoding_name(encoding_name)
        state = GameState.from_board(board)
        for _ in range(target_ply):
            legal = board.legal_moves()
            if not legal:
                break
            q, r = random.choice(legal)
            state = state.apply_move(board, q, r)
        tensor, _centers = state.to_tensor()
        # For multi-window encodings shallow random rollouts keep every stone
        # inside the origin cluster window — cluster 0 is unambiguous.
        aug_cluster = tensor[0]
        states_out.append(aug_cluster[list(KEPT_PLANE_INDICES)].copy())

        mask = np.zeros(n_actions, dtype=np.uint8)
        for q, r in board.legal_moves():
            flat = board.to_flat(q, r)
            if 0 <= flat < n_actions:
                mask[flat] = 1
        masks_out.append(mask)

    states = np.stack(states_out, axis=0).astype(np.float16)
    plies  = np.asarray(target_plies, dtype=np.int32)
    seeds  = np.asarray([seed + p for p in target_plies], dtype=np.int32)
    legal_mask = np.stack(masks_out, axis=0).astype(np.uint8)
    return _FixturePayload(states=states, plies=plies, seeds=seeds, legal_mask=legal_mask)


def save_fixture(path: Path | None = None, encoding_name: str = "v6") -> _FixturePayload:
    """Regenerate + write the fixture. Idempotent: two consecutive calls
    produce byte-identical files.
    """
    if path is None:
        path = _fixture_path_for_encoding(encoding_name)
    payload = _generate_fixture_payload(encoding_name=encoding_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        states=payload.states,
        plies=payload.plies,
        seeds=payload.seeds,
        legal_mask=payload.legal_mask,
        version=np.int32(_FIXTURE_VERSION),
    )
    return payload


def load_fixture(path: Path | None = None, encoding_name: str = "v6") -> _FixturePayload:
    """Load the fixture; generate it if absent."""
    if path is None:
        path = _fixture_path_for_encoding(encoding_name)
    if not path.exists():
        return save_fixture(path, encoding_name=encoding_name)
    z = np.load(path)
    version = int(z["version"])
    if version != _FIXTURE_VERSION:
        raise ValueError(
            f"early_game_probe fixture schema v{version} at {path!s} "
            f"does not match expected v{_FIXTURE_VERSION}. "
            "Regenerate with scripts/build_early_game_probe.py."
        )
    return _FixturePayload(
        states=np.asarray(z["states"], dtype=np.float16),
        plies=np.asarray(z["plies"],  dtype=np.int32),
        seeds=np.asarray(z["seeds"],  dtype=np.int32),
        legal_mask=np.asarray(z["legal_mask"], dtype=np.uint8),
    )


class EarlyGameProbe:
    """Stateless-ish probe: owns the fixture tensor on the target device."""

    def __init__(
        self,
        device: torch.device,
        fixture_path: Path | None = None,
        encoding_name: str = "v6",
    ) -> None:
        payload = load_fixture(fixture_path, encoding_name=encoding_name)
        self._plies: List[int] = payload.plies.tolist()
        # Hold the probe tensor on device as float32 — the model's forward
        # path accepts either fp16 or fp32 inputs (amp autocast downcasts
        # under CUDA); fp32 keeps the probe numerically stable for tiny
        # entropy deltas across checkpoints.
        tensor = torch.from_numpy(payload.states.astype(np.float32))
        self._states = tensor.to(device)
        # Legal-action mask as a float tensor for multiplicative masking.
        legal = torch.from_numpy(payload.legal_mask.astype(np.float32))
        self._legal_mask = legal.to(device)
        self._n_legal_per_pos: List[int] = [int(x) for x in payload.legal_mask.sum(axis=1).tolist()]
        self._device = device
        self._n_positions = self._states.shape[0]
        self.encoding_name = encoding_name

    @property
    def n_positions(self) -> int:
        return self._n_positions

    @property
    def plies(self) -> List[int]:
        return list(self._plies)

    @torch.no_grad()
    def compute(self, model: Any) -> Dict[str, float]:
        """One forward pass on the probe batch.

        Entropy is computed on the **legal-action renormalized** distribution
        (see module docstring). Illegal cells + pass action get zeroed before
        renormalisation, matching the §116 diag_early_game probe so readings
        are directly comparable across ladders.

        Returns a dict suitable for merging straight into a training_step
        event payload. Keys:

            early_game_entropy_mean       — mean H_legal (nats) across fixture
            early_game_entropy_max        — max  H_legal (nats) across fixture
            early_game_top1_mass_mean     — mean top-1 legal-renorm prob
            early_game_entropy_by_ply     — list[float] per-position H_legal
            early_game_top1_mass_by_ply   — list[float] per-position top-1 mass
        """
        if self._n_positions == 0:
            return {
                "early_game_entropy_mean": 0.0,
                "early_game_entropy_max": 0.0,
                "early_game_top1_mass_mean": 0.0,
                "early_game_entropy_by_ply": [],
                "early_game_top1_mass_by_ply": [],
            }

        # Guard against model / fixture shape mismatch (e.g. v6w25 model vs v6 fixture).
        # If the input spatial dims differ, skip the probe gracefully.
        expected_spatial = (self._states.shape[2], self._states.shape[3])
        model_in_channels = getattr(model, "in_channels", None)
        # Try to infer expected spatial size from model.board_size
        model_board_size = getattr(model, "board_size", None)
        if model_board_size is not None and expected_spatial != (model_board_size, model_board_size):
            return {
                "early_game_entropy_mean": 0.0,
                "early_game_entropy_max": 0.0,
                "early_game_top1_mass_mean": 0.0,
                "early_game_entropy_by_ply": [],
                "early_game_top1_mass_by_ply": [],
            }

        was_training = bool(getattr(model, "training", False))
        base = getattr(model, "_orig_mod", model)  # unwrap torch.compile
        base.eval()
        try:
            log_p, _value, _v_logit = base(self._states)
        finally:
            if was_training:
                base.train()

        p_full = log_p.float().exp()                              # (N, A)
        p_legal = p_full * self._legal_mask                        # zero illegal
        denom = p_legal.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        p_legal = p_legal / denom                                  # renorm to sum-1 over legal
        # log(0) appears for illegal slots after masking; clamp before log so
        # the entropy sum over illegal slots contributes 0 (p=0 · log=0).
        log_p_legal = p_legal.clamp_min(1e-12).log()
        entropy = -(p_legal * log_p_legal * self._legal_mask).sum(dim=-1)   # (N,)
        top1    = (p_legal * self._legal_mask).max(dim=-1).values           # (N,)

        entropy_cpu = entropy.detach().cpu().numpy()
        top1_cpu    = top1.detach().cpu().numpy()

        return {
            "early_game_entropy_mean":      float(entropy_cpu.mean()),
            "early_game_entropy_max":       float(entropy_cpu.max()),
            "early_game_top1_mass_mean":    float(top1_cpu.mean()),
            "early_game_entropy_by_ply":    [float(x) for x in entropy_cpu.tolist()],
            "early_game_top1_mass_by_ply":  [float(x) for x in top1_cpu.tolist()],
        }


# Threshold used for the WARNING log line emitted by the training loop.
# Bootstrap-v4 reads ~3.0–4.0 nat on this fixture; curr_10k §116 read ~5.4.
EARLY_GAME_ENTROPY_WARN_THRESHOLD: float = 4.5
