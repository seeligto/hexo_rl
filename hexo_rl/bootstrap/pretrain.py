"""
Bootstrap pretraining pipeline — Phase 4.0 architecture.

Loads human + bot corpus, applies quality/source weighting, and trains the
HexTacToeNet with Phase 4.0 losses:

    L = L_policy + L_value_BCE + aux_weight × L_opp_reply

Saves a checkpoint in the same format as Trainer.save_checkpoint() so that
scripts/train.py can resume from it seamlessly.

Usage:
    python -m python.bootstrap.pretrain [--epochs N] [--steps N] [--batch-size N]
    make pretrain          # 15 epochs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import structlog
from rich.console import Console

import engine
from engine import Board
from hexo_rl.bootstrap.dataset import replay_game_to_triples
from hexo_rl.bootstrap.generate_corpus import BOT_GAMES_DIR, INJECTED_DIR, RAW_HUMAN_DIR
from hexo_rl.env.game_state import GameState, _compute_chain_planes
from hexo_rl.model.network import HexTacToeNet, compile_model
from hexo_rl.training.losses import (
    compute_policy_loss, compute_value_loss, compute_aux_loss,
    compute_chain_loss, compute_total_loss, fp16_backward_step,
)
from hexo_rl.training.checkpoints import get_base_model, save_full_checkpoint, save_inference_weights
from hexo_rl.encoding import (
    all_specs as _all_specs,
    lookup as _lookup_encoding,
    resolve_corpus_path as _resolve_corpus_path,
    resolve_from_config as _registry_resolve_cfg,
)
from hexo_rl.utils.constants import BOARD_SIZE, BUFFER_CHANNELS
from hexo_rl.monitoring.events import emit_event
from hexo_rl.augment.luts import get_policy_scatters

log = structlog.get_logger()
console = Console()

# ── Hex augmentation ──────────────────────────────────────────────────────────
#
# Q13 chain planes live at state[18..24] and require an axis-plane remap under
# augmentation in addition to the coordinate scatter — see
# `engine/src/replay_buffer/sample.rs::apply_symmetry_24plane` + the
# `axis_perm[sym_idx]` table in `sym_tables.rs`. The pre-Q13 pretrain path
# replicated the coord scatter in Python but had two bugs:
#   (1) no axis_perm remap, so chain planes contradicted the stones in 11/12
#       augmented samples (§92/§93 F1);
#   (2) (col, row) vs (row, col) coordinate convention mismatch with the
#       Rust SymTables / _compute_chain_planes.
# Both bugs are eliminated by routing through the Rust `apply_symmetries_batch`
# binding — the same kernel ReplayBuffer.sample_batch uses, guaranteed to
# match byte-exact via `tests/test_pretrain_aug.py` (F1 guard).
#
# Policies are scattered in numpy via a once-computed per-sym index map — they
# have no chain-plane structure and the 12 tables fit in ~17 KB.
# LUTs are in hexo_rl/augment/luts.py (shared with batch_assembly).

# ── Dataset ───────────────────────────────────────────────────────────────────

class AugmentedBootstrapDataset(torch.utils.data.Dataset):
    """Pretrain dataset that yields raw (state, policy, outcome) triples.

    The 12-fold hex augmentation is applied in `augmented_collate` via the
    Rust `engine.apply_symmetries_batch` binding — this is the same scatter
    kernel the ReplayBuffer uses at sample time, guaranteed byte-exact via
    the F1 parity test (`tests/test_pretrain_aug.py`). Moving augmentation
    to collate amortises the PyO3 boundary over an entire batch and
    eliminates the Python `_apply_hex_sym` path that was corrupting Q13
    chain planes in pretrain v3.

    Chain planes (6 planes, aux head target) are computed in collate from
    the augmented stone planes (0=cur, 8=opp) — recomputing post-augment
    avoids the axis-perm remap and is self-consistent.

    §169 A3 — when ``global_crops`` is provided, ``__getitem__`` returns
    a 4-tuple including the per-position global summary crop. Augmentation
    of the crop is intentionally skipped (the GlobalTokenEncoder is
    near-rotation-invariant via KataGo gpool, so feeding the canonical-
    orientation crop alongside augmented cluster windows is a tractable
    approximation; documented as a known A3 simplification).

    Args:
        states:        (N, C, H, W) float16 array (mmap-compatible).
        policies:      (N, n_actions) float32 array.
        outcomes:      (N,) float32 array, values in {-1, 0, +1}.
        global_crops:  Optional (N, 3, 32, 32) float16 array. When present,
                       __getitem__ returns a 4-tuple.
    """

    def __init__(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        outcomes: np.ndarray,
        global_crops: Optional[np.ndarray] = None,
    ) -> None:
        self.states = states
        self.policies = policies
        self.outcomes = outcomes
        self.global_crops = global_crops

    def __len__(self) -> int:
        return len(self.outcomes)

    def __getitem__(self, idx: int):
        # Copy out of the (possibly mmapped) backing store so downstream
        # collate can batch-concat without aliasing.
        if self.global_crops is not None:
            return (
                self.states[idx].copy(),
                self.policies[idx].copy(),
                float(self.outcomes[idx]),
                self.global_crops[idx].copy(),
            )
        return (
            self.states[idx].copy(),
            self.policies[idx].copy(),
            float(self.outcomes[idx]),
        )


def make_augmented_collate(
    augment: bool,
    board_size: int = BOARD_SIZE,
    encoding: str = "v6",
    with_global_crop: bool = False,
):
    """Return a collate_fn that batches triples and applies hex augmentation.

    Two paths:

    v6 (default, `encoding="v6"`, `board_size=19`):
      - Stack states (N, 8, 19, 19) f16, upcast to f32.
      - Rust `engine.apply_symmetries_batch` for state scatter (one PyO3 hop).
      - Policy scatter via precomputed numpy index tables (12 × 362).
      - Chain planes recomputed from augmented stone planes 0 (cur) / 4 (opp).

    v8 (`encoding="v8"`, `board_size=25`, `has_pass=False`):
      - Stack states (N, 11, 25, 25) f16.
      - Pure-numpy state scatter (same hex-symmetry math as Rust kernel) —
        the Rust `apply_symmetries_batch` PyO3 binding hardcodes BOARD_SIZE=19
        and is v6-only. Per S2 §2.4 v8 splice spec, Phase B keeps state aug
        in Python; the v8 PyO3 path is a Phase D concern when self-play
        replay buffers must scatter v8 buffer rows at runtime.
      - Policy scatter (N, 625) using `get_policy_scatters(25, has_pass=False)`.
      - Chain planes recomputed at 25×25 (planes 8/9/10 are
        symmetry-invariant: off_window mask is hex-symmetric and the broadcast
        scalars are constants).

    With `augment=False`:
      - Stack as-is, no scatter.
      - Compute chain_planes from raw stone planes.
    """
    _enc_spec = _lookup_encoding(encoding)
    has_pass = _enc_spec.has_pass_slot
    scatters_np = get_policy_scatters(board_size, has_pass=has_pass) if augment else None

    def _collate(batch):
        n = len(batch)
        states = np.stack([b[0] for b in batch], axis=0)
        policies = np.stack([b[1] for b in batch], axis=0)
        outcomes = np.asarray([b[2] for b in batch], dtype=np.float32)
        # §169 A3 — global crop stays canonical (no per-batch sym applied).
        # GlobalTokenEncoder ends in KataGo gpool which is near-spatially-
        # invariant, so the orientation mismatch with augmented cluster
        # windows is bounded; documented as an A3 simplification.
        global_crops = (
            np.stack([b[3] for b in batch], axis=0)
            if with_global_crop
            else None
        )

        if augment and scatters_np is not None:
            sym_indices = np.random.randint(0, 12, size=n).astype(np.int64)

            if encoding == "v6":
                states_f32 = states.astype(np.float32, copy=False)
                states_f32 = engine.apply_symmetries_batch(
                    states_f32, sym_indices.astype(np.uint64).tolist()
                )
                scattered = np.empty_like(policies)
                for i in range(n):
                    scattered[i] = policies[i][scatters_np[int(sym_indices[i])]]
                policies = scattered
                states = states_f32.astype(np.float16, copy=False)
            else:
                # v8 / v6w25 path — pure-numpy scatter, batched per-sym to avoid
                # the per-sample take_along_axis cost. The Rust
                # apply_symmetries_batch PyO3 binding hardcodes BOARD_SIZE=19;
                # both v6w25 (25×25 K-cluster) and v8 (25×25 bbox) need numpy.
                C = states.shape[1]
                spatial = board_size * board_size
                states_flat = states.reshape(n, C, spatial)
                augmented = np.empty_like(states_flat)
                policy_aug = np.empty_like(policies)
                for sym in range(12):
                    mask_idx = np.where(sym_indices == sym)[0]
                    if mask_idx.size == 0:
                        continue
                    # For v6w25 the policy scatter is `spatial+1` long (cells
                    # + pass). State scatter must slice to spatial-only.
                    sc = scatters_np[sym]
                    state_scatter = sc[:spatial] if has_pass else sc
                    augmented[mask_idx] = states_flat[mask_idx][:, :, state_scatter]
                    policy_aug[mask_idx] = policies[mask_idx][:, sc]
                states = augmented.reshape(n, C, board_size, board_size)
                policies = policy_aug

        # Chain planes — recomputed post-augmentation from stone planes.
        # Stone plane indices: 0 = cur ply-0, 4 = opp ply-0 (v6 KEPT layout
        # carries through to v8 wire format planes 0/4).
        chain_np = np.zeros((n, 6, board_size, board_size), dtype=np.float16)
        for i in range(n):
            chain_np[i] = _compute_chain_planes(
                states[i, 0].astype(np.float32),
                states[i, 4].astype(np.float32),
            ).astype(np.float16) / 6.0

        if global_crops is not None:
            return (
                torch.from_numpy(states),
                torch.from_numpy(chain_np),
                torch.from_numpy(policies),
                torch.from_numpy(outcomes),
                torch.from_numpy(global_crops),
            )
        return (
            torch.from_numpy(states),
            torch.from_numpy(chain_np),
            torch.from_numpy(policies),
            torch.from_numpy(outcomes),
        )

    return _collate


# ── Corpus loading ────────────────────────────────────────────────────────────

def _game_winner_from_replay(moves: List[Tuple[int, int]]) -> Optional[int]:
    """Replay a move sequence and return winner (+1 or -1), or None."""
    board = Board()
    for q, r in moves:
        try:
            board.apply_move(q, r)
        except Exception:
            break
    w = board.winner()
    return int(w) if w is not None else None


def load_corpus(
    quality_scores: Dict[str, Dict],
    source_weights: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all corpus games and return flat per-position arrays.

    Returns:
        states:   (M, 18, 19, 19) float16
        policies: (M, 362) float32
        outcomes: (M,) float32
        weights:  (M,) float32 — per-position sampling weight
                  = quality_score × source_weight
    """
    all_s: List[np.ndarray] = []
    all_p: List[np.ndarray] = []
    all_o: List[np.ndarray] = []
    all_w: List[float] = []

    def _add_game(
        moves: List[Tuple[int, int]],
        winner: int,
        game_id: str,
        source: str,
    ) -> None:
        s, _chain, p, o = replay_game_to_triples(moves, winner)
        if len(o) == 0:
            return
        q_score = quality_scores.get(game_id, {}).get("quality_score", 0.5)
        src_w = source_weights.get(source, 1.0)
        weight = float(q_score * src_w)
        all_s.append(s)
        all_p.append(p)
        all_o.append(o)
        all_w.extend([weight] * len(o))

    # Human games
    human_ok = 0
    for path in sorted(RAW_HUMAN_DIR.glob("*.json")):
        try:
            with open(path) as f:
                d = json.load(f)
            if "moves" not in d:
                continue
            moves = [(int(m["x"]), int(m["y"])) for m in d["moves"]]
            winner = _game_winner_from_replay(moves)
            if winner is None:
                continue
            _add_game(moves, winner, path.stem, "human")
            human_ok += 1
        except Exception:
            continue
    log.info("loaded_human_games", count=human_ok)

    # Bot fast (0.1s think time)
    fast_ok = 0
    fast_dir = BOT_GAMES_DIR / "sealbot_fast"
    if fast_dir.exists():
        for path in sorted(fast_dir.glob("*.json")):
            try:
                with open(path) as f:
                    d = json.load(f)
                moves = [(int(m["x"]), int(m["y"])) for m in d["moves"]]
                winner = int(d["winner"]) if "winner" in d else _game_winner_from_replay(moves)
                if winner is None or winner == 0:
                    continue
                _add_game(moves, winner, path.stem, "bot_fast")
                fast_ok += 1
            except Exception:
                continue
    log.info("loaded_bot_fast_games", count=fast_ok)

    # Bot strong (0.5s think time)
    strong_ok = 0
    strong_dir = BOT_GAMES_DIR / "sealbot_strong"
    if strong_dir.exists():
        for path in sorted(strong_dir.glob("*.json")):
            try:
                with open(path) as f:
                    d = json.load(f)
                moves = [(int(m["x"]), int(m["y"])) for m in d["moves"]]
                winner = int(d["winner"]) if "winner" in d else _game_winner_from_replay(moves)
                if winner is None or winner == 0:
                    continue
                _add_game(moves, winner, path.stem, "bot_strong")
                strong_ok += 1
            except Exception:
                continue
    log.info("loaded_bot_strong_games", count=strong_ok)

    # Injected games (human-seed bot-continuation)
    injected_ok = 0
    if INJECTED_DIR.exists():
        for path in sorted(INJECTED_DIR.glob("*.json")):
            try:
                with open(path) as f:
                    d = json.load(f)
                moves = [(int(m["x"]), int(m["y"])) for m in d["moves"]]
                winner = int(d["winner"]) if "winner" in d else _game_winner_from_replay(moves)
                if winner is None or winner == 0:
                    continue
                _add_game(moves, winner, path.stem, "injected")
                injected_ok += 1
            except Exception:
                continue
    log.info("loaded_injected_games", count=injected_ok)

    if not all_s:
        return (
            np.empty((0, 18, BOARD_SIZE, BOARD_SIZE), dtype=np.float16),
            np.empty((0, BOARD_SIZE * BOARD_SIZE + 1), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    states   = np.concatenate(all_s, axis=0)
    policies = np.concatenate(all_p, axis=0)
    outcomes = np.concatenate(all_o, axis=0)
    weights  = np.array(all_w, dtype=np.float32)
    return states, policies, outcomes, weights


# ── Trainer ───────────────────────────────────────────────────────────────────

class BootstrapTrainer:
    """Pretraining loop using the Phase 4.0 loss function.

    Matches Trainer._train_on_batch() from python/training/trainer.py:
      - FP16 AMP
      - aux opponent-reply head (aux_weight from config)
      - gradient clipping to 1.0
      - policy valid masking (skip zero-policy rows)
      - label smoothing on policy targets
      - cosine LR schedule
    """

    def __init__(
        self,
        model: HexTacToeNet,
        config: Dict,
        device: torch.device,
        checkpoint_dir: Path,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0

        fp16 = bool(config.get("fp16", True)) and device.type == "cuda"
        self.fp16 = fp16

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(config.get("lr", 0.002)),
            weight_decay=float(config.get("weight_decay", 0.0001)),
        )
        self.scaler = torch.amp.GradScaler(device=device.type, enabled=fp16)

        total_steps = int(config.get("pretrain_total_steps", 50_000))
        eta_min = float(config.get("pretrain_eta_min", 1e-5))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_steps), eta_min=eta_min,
        )

    def train_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        label_smoothing: float = 0.05,
        aux_weight: float = 0.15,
        chain_weight: float = 0.0,
        step_budget: Optional[int] = None,
        start_step: Optional[int] = None,
        log_interval: int = 50,
    ) -> Dict[str, float]:
        """One full pass over the dataloader.

        Args:
            loader:         DataLoader yielding (states, chain_planes, policies, outcomes).
            label_smoothing: ε for policy targets; 0 disables.
            aux_weight:     Weight for opponent-reply auxiliary loss.
            chain_weight:   Weight for Q13-aux chain-length head (0 disables it).
            step_budget:    Stop after this many steps (for smoke tests).
            start_step:     Baseline ``self.step`` when budget counting began. If
                            None, defaults to the value of ``self.step`` on entry.

        Returns:
            Dict with keys loss, policy_loss, value_loss, opp_reply_loss, chain_loss.
        """
        budget_origin = start_step if start_step is not None else self.step
        self.model.train()
        total: Dict[str, float] = {
            "loss": 0.0, "policy_loss": 0.0,
            "value_loss": 0.0, "opp_reply_loss": 0.0,
            "chain_loss": 0.0,
        }
        n_batches = 0

        for batch in loader:
            # Batches are 4-tuple by default; with §169 A3 global-crop wiring
            # the collate appends a 5th element (global_crops).
            if len(batch) == 5:
                states, chain_planes, policies, outcomes, global_crops = batch
                global_crops = global_crops.to(self.device)
            else:
                states, chain_planes, policies, outcomes = batch
                global_crops = None
            states       = states.to(self.device)
            chain_planes = chain_planes.to(self.device)
            policies     = policies.to(self.device)
            outcomes     = outcomes.to(self.device)

            if label_smoothing > 0.0:
                n_actions = policies.shape[1]
                policies = policies * (1.0 - label_smoothing) + label_smoothing / n_actions

            self.optimizer.zero_grad()

            use_chain = chain_weight > 0.0
            fwd_kwargs = {"aux": True, "chain": use_chain}
            if global_crops is not None:
                fwd_kwargs["global_crop"] = global_crops
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.fp16,
            ):
                fwd = self.model(states, **fwd_kwargs)
                log_policy, _value, v_logit, opp_reply = fwd[0], fwd[1], fwd[2], fwd[3]
                chain_pred = fwd[4] if use_chain else None

                policy_valid = policies.sum(dim=1) > 1e-6
                policy_loss = compute_policy_loss(log_policy, policies, policy_valid, self.device)
                value_loss = compute_value_loss(v_logit, outcomes)
                opp_reply_loss = compute_aux_loss(opp_reply, policies, policy_valid, self.device)

                chain_loss = None
                if use_chain and chain_pred is not None:
                    chain_loss = compute_chain_loss(chain_pred, chain_planes.float())

                loss = compute_total_loss(
                    policy_loss,
                    value_loss,
                    opp_reply_loss,
                    aux_weight,
                    chain_loss=chain_loss,
                    chain_weight=chain_weight,
                )

            # Defense-in-depth: skip backward+step when forward produced NaN/inf.
            # §167 B1 retrain hit a single-batch fp16 overflow in KataConvAndGPool
            # at step -22000 (epoch 14). The standard scaler.unscale_ +
            # clip_grad_norm_ + scaler.step chain failed to skip — clip_grad_norm_
            # multiplies grads by NaN clip_coef, optimizer wrote NaN to weights,
            # and all subsequent steps spun NaN. Skipping the step entirely on
            # non-finite forward loss prevents the cascade.
            if not torch.isfinite(loss):
                self.skipped_nonfinite_steps = getattr(
                    self, "skipped_nonfinite_steps", 0
                ) + 1
                if self.skipped_nonfinite_steps <= 5 or self.skipped_nonfinite_steps % 50 == 0:
                    log.warning(
                        "skipped_nonfinite_loss",
                        step=self.step + 1,
                        n_skipped=self.skipped_nonfinite_steps,
                        loss=float(loss.detach().item()),
                        policy_loss=float(policy_loss.detach().item()),
                        value_loss=float(value_loss.detach().item()),
                    )
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.step += 1
                continue

            grad_norm = fp16_backward_step(loss, self.optimizer, self.scaler, self.model, self.fp16)

            self.scheduler.step()
            self.step += 1

            step_loss = loss.item()
            total["loss"]           += step_loss
            total["policy_loss"]    += policy_loss.item()
            total["value_loss"]     += value_loss.item()
            total["opp_reply_loss"] += opp_reply_loss.item()
            if chain_loss is not None:
                total["chain_loss"] += chain_loss.item()
            n_batches += 1

            if log_interval > 0 and self.step % log_interval == 0:
                policy_entropy = -torch.sum(torch.exp(log_policy) * log_policy, dim=1).mean().item()
                value_accuracy = (torch.sign(v_logit.squeeze()) == torch.sign(outcomes)).float().mean().item()
                lr = float(self.optimizer.param_groups[0]["lr"])

                # §169 A3 — surface the global-token gate scalar so the
                # operator can read whether the global branch earned weight
                # or stayed at init. None for non-pma_global runs.
                # §170 P3 — parallel surface for the gpool-bias gate. The two
                # gates are mutually exclusive (disjoint pool_types); separate
                # event keys keep the dashboard schema clean.
                base_model = get_base_model(self.model)
                gate_val: Optional[float] = None
                gpool_bias_gate_val: Optional[float] = None
                cluster_pool = getattr(base_model, "cluster_pool", None)
                if cluster_pool is not None and hasattr(cluster_pool, "gate_value"):
                    gate_val = float(cluster_pool.gate_value())
                if hasattr(base_model, "gpool_bias_gate_value"):
                    gbg = base_model.gpool_bias_gate_value()
                    if gbg is not None:
                        gpool_bias_gate_val = float(gbg)

                # Emit to dashboard — include loss_chain so the C14 dashboard
                # rendering surfaces the Q13-aux head during pretrain runs.
                event = {
                    "event": "training_step",
                    "step": self.step,
                    "loss_total": float(step_loss),
                    "loss_policy": float(policy_loss.item()),
                    "loss_value": float(value_loss.item()),
                    "loss_aux": float(opp_reply_loss.item()),
                    "loss_chain": float(chain_loss.item()) if chain_loss is not None else 0.0,
                    "loss_ownership": 0.0,
                    "loss_threat": 0.0,
                    "policy_entropy": policy_entropy,
                    "value_accuracy": value_accuracy,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "corpus_mix": {"pretrain": 1.0, "self_play": 0.0},
                    "phase": "pretrain",
                }
                if gate_val is not None:
                    event["pool_global_gate"] = gate_val
                if gpool_bias_gate_val is not None:
                    event["gpool_bias_gate"] = gpool_bias_gate_val
                emit_event(event)

                log_kwargs = dict(
                    step=self.step,
                    phase="pretrain",
                    loss=round(step_loss, 4),
                    policy_loss=round(policy_loss.item(), 4),
                    value_loss=round(value_loss.item(), 4),
                    aux_opp_reply_loss=round(opp_reply_loss.item(), 4),
                    policy_entropy=round(policy_entropy, 4),
                    value_accuracy=round(value_accuracy, 4),
                    lr=lr,
                    grad_norm=round(grad_norm, 4),
                    corpus_mix={"pretrain": 1.0, "self_play": 0.0},
                )
                if gate_val is not None:
                    log_kwargs["pool_global_gate"] = round(gate_val, 5)
                if gpool_bias_gate_val is not None:
                    log_kwargs["gpool_bias_gate"] = round(gpool_bias_gate_val, 5)
                log.info("train_step", **log_kwargs)

            if step_budget is not None and (self.step - budget_origin) >= step_budget:
                break

        n = max(n_batches, 1)
        return {k: v / n for k, v in total.items()}

    def save_checkpoint(self, inf_out: Optional[Path] = None) -> Path:
        """Save full checkpoint in self-play-compatible format.

        Also writes a weights-only file (default: checkpoints/bootstrap_model.pt)
        for the eval pipeline. Pass ``inf_out`` to override that path — used by
        --resume runs that should not clobber the canonical bootstrap model
        until eval confirms uplift.
        """
        ckpt_path = self.checkpoint_dir / f"pretrain_{abs(self.step):08d}.pt" if self.step < 0 else self.checkpoint_dir / f"pretrain_{self.step:08d}.pt"
        save_full_checkpoint(
            self.model, self.optimizer, self.scaler, self.scheduler,
            self.step, self.config, ckpt_path,
        )
        inf_path = inf_out if inf_out is not None else Path("checkpoints") / "bootstrap_model.pt"
        save_inference_weights(self.model, inf_path)
        log.info("checkpoint_saved", path=str(ckpt_path), inference=str(inf_path))
        return ckpt_path


# ── Fine-tune freeze (§171 A4 P2-reopen C) ────────────────────────────────────

def _apply_finetune_freeze(
    base_model,
    *,
    freeze_trunk_entry: bool,
    unfreeze_blocks: Optional[set],
) -> Dict[str, int]:
    """Apply §171 A4 fine-tune freeze pattern.

    - `freeze_trunk_entry=True`: requires_grad=False on `trunk.input_conv`
      (PartialConv2d under canvas_realness) + `trunk.input_gn`.
    - `unfreeze_blocks={i,...}`: requires_grad=False on every
      `trunk.tower[k]` where k not in the set. Heads (policy_head /
      opp_reply_head / value_fc1 / value_fc2 / value_var) left
      trainable — they are not touched by this function so their
      requires_grad stays at whatever the model construction set
      (True for KataGo head + linear value, never frozen here).

    Returns counts for logging. AdamW weight_decay drift on frozen params
    is bounded by exp(-lr * wd * steps); at lr=5e-5, wd=1e-4, 3000 steps
    that is ~1.5e-5 — negligible. Optimizer state is not rebuilt; frozen
    params receive zero-gradient Adam updates (m_hat, v_hat → 0).
    """
    trunk = base_model.trunk
    tower = trunk.tower

    if freeze_trunk_entry:
        for p in trunk.input_conv.parameters():
            p.requires_grad = False
        for p in trunk.input_gn.parameters():
            p.requires_grad = False

    if unfreeze_blocks is not None:
        n_blocks = len(tower)
        for idx in unfreeze_blocks:
            if not (0 <= idx < n_blocks):
                raise ValueError(
                    f"--unfreeze-blocks entry {idx} out of [0, {n_blocks}); "
                    f"trunk has {n_blocks} blocks"
                )
        for i, block in enumerate(tower):
            keep = i in unfreeze_blocks
            for p in block.parameters():
                p.requires_grad = keep

    total = sum(p.numel() for p in base_model.parameters())
    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    return {
        "freeze_trunk_entry": int(bool(freeze_trunk_entry)),
        "unfreeze_blocks": sorted(unfreeze_blocks) if unfreeze_blocks else [],
        "total_params": int(total),
        "trainable_params": int(trainable),
    }


# ── Validation ────────────────────────────────────────────────────────────────

def validate(ckpt_path: Path, device: torch.device) -> None:
    """Verify checkpoint round-trip and play 100 greedy games vs RandomBot.

    Uses argmax policy (no MCTS) — suitable for a pretrained but not
    yet self-play-trained checkpoint.

    Encoding-aware (§168): reads `board_size` and `encoding` from the
    saved config and configures the Rust Board with the matching cluster
    window size + threshold + legal-move radius. v6 default at 19×19 /
    threshold 5 / r=5; v6w25 at 25×25 / threshold 8 / r=8; v8 path
    skipped (encoding != "v6"-family — no K-cluster window encoder).
    """
    from hexo_rl.bootstrap.bots.random_bot import RandomBot

    # Round-trip: load checkpoint, rebuild model, run forward pass.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert "model_state" in ckpt and "config" in ckpt, (
        f"Checkpoint missing keys; got {list(ckpt.keys())}"
    )
    cfg = ckpt["config"]
    enc_section = cfg.get("encoding")
    if isinstance(enc_section, str):
        cfg_encoding = enc_section
    elif isinstance(enc_section, dict):
        cfg_encoding = str(enc_section.get("version", "v6"))
    else:
        cfg_encoding = "v6"
    _enc_spec = _lookup_encoding(cfg_encoding)
    cfg_board_size = _enc_spec.trunk_size
    cfg_in_channels = int(cfg.get("in_channels", _enc_spec.n_planes))
    cfg_n_actions = _enc_spec.policy_logit_count
    cfg_half = (cfg_board_size - 1) // 2

    cfg_pool_type = str(cfg.get("pool_type", "min_max"))
    cfg_gpool_bias_active = bool(cfg.get("gpool_bias_active", False))
    loaded_model = HexTacToeNet(
        board_size=cfg_board_size,
        in_channels=cfg_in_channels,
        filters=int(cfg.get("filters", 128)),
        res_blocks=int(cfg.get("res_blocks", 12)),
        se_reduction_ratio=int(cfg.get("se_reduction_ratio", 4)),
        encoding=cfg_encoding,
        pool_type=cfg_pool_type,
        pool_attn_dropout=float(cfg.get("pool_attn_dropout", 0.1)),
        gpool_bias_active=cfg_gpool_bias_active,
    )
    loaded_model.load_state_dict(ckpt["model_state"])
    loaded_model.eval().to(device)

    dummy = torch.zeros(
        1, cfg_in_channels, cfg_board_size, cfg_board_size, device=device,
    )
    fwd_kwargs: Dict = {}
    if cfg_pool_type == "pma_global" or cfg_gpool_bias_active:
        # pma_global / gpool_bias_active need a global crop for the forward;
        # a zeroed canvas is a valid empty-board input. The smoke just checks
        # the wiring round-trips; full play-vs-RandomBot is skipped.
        from hexo_rl.utils.global_crop import (
            CANVAS_SIZE as _GLOBAL_CANVAS_SIZE,
            N_GLOBAL_PLANES as _N_GLOBAL_PLANES,
        )
        fwd_kwargs["global_crop"] = torch.zeros(
            1, _N_GLOBAL_PLANES, _GLOBAL_CANVAS_SIZE, _GLOBAL_CANVAS_SIZE,
            device=device,
        )
    with torch.no_grad():
        log_pol, val, v_logit = loaded_model(dummy.float(), **fwd_kwargs)
    assert log_pol.shape == (1, cfg_n_actions), \
        f"Unexpected policy shape: {log_pol.shape} (expected (1, {cfg_n_actions}))"
    log.info("checkpoint_forward_pass_ok", val=float(val[0, 0]))

    # v8 needs a different game loop (single-bbox encoder + V8ArgmaxBot
    # history tracking). Skip the play-100-greedy step under v8; the
    # round-trip + forward-pass shape check is sufficient at this layer.
    if cfg_encoding == "v8":
        log.info("validation_skipped_v8_path", reason="v8 needs V8ArgmaxBot path")
        return
    # pma_global needs a per-position global crop computed from the live board.
    # The play-100-greedy harness is K=1 cluster-based and would have to call
    # compute_global_crop_from_board per ply; defer that to the §169 A3 eval
    # script instead of doubling the wiring here.
    if cfg_pool_type == "pma_global":
        log.info(
            "validation_skipped_pma_global",
            reason="pma_global validation runs under scripts/eval_a3_pma_global.sh",
        )
        return
    # §170 P3 — gpool_bias_active also needs per-position global crops via
    # compute_global_crop_from_board. Defer to scripts/eval_gpool_bias.sh.
    if cfg_gpool_bias_active:
        log.info(
            "validation_skipped_gpool_bias_active",
            reason="gpool_bias validation runs under scripts/eval_gpool_bias.sh",
        )
        return

    # Play 100 greedy games vs RandomBot — expect high win rate after pretraining.
    # §173 A6: Board constructed via registry (Board.with_encoding_name) — no
    # triple-setter. Encoding params (radius, cluster window/threshold) come
    # from the registry entry for cfg_encoding.
    random_bot = RandomBot()
    wins = 0
    n_games = 100

    for i in range(n_games):
        board = Board.with_encoding_name(cfg_encoding)
        state = GameState.from_board(board)
        model_player = 1 if i % 2 == 0 else -1

        for _ in range(200):
            if board.check_win() or board.legal_move_count() == 0:
                break

            if board.current_player == model_player:
                tensor, centers = state.to_tensor()
                # Aug-only K-aggregation site. Live training/inference forwards ALL
                # K cluster views through the network: min-pool on value, scatter-max
                # on policy (worker_loop.rs:299-401 MCTS forward, 649-682 replay push).
                # This site picks cluster 0 ONLY because the consumer (RandomBot
                # validation) is an aug fixture, not a boundary path. See sprint §164 P1.
                aug_cluster = tensor[0]
                aug_cluster_center = centers[0]
                inp = torch.from_numpy(aug_cluster[list(_enc_spec.kept_plane_indices)]).unsqueeze(0).to(device).float()
                with torch.no_grad():
                    lp, _, _ = loaded_model(inp)
                lp_np = lp[0].cpu().numpy()
                cq, cr = aug_cluster_center
                legal = board.legal_moves()
                best_move, best_score = legal[0], -1e9
                for q, r in legal:
                    wq, wr = q - cq + cfg_half, r - cr + cfg_half
                    if 0 <= wq < cfg_board_size and 0 <= wr < cfg_board_size:
                        score = float(lp_np[wq * cfg_board_size + wr])
                        if score > best_score:
                            best_score, best_move = score, (q, r)
                q, r = best_move
            else:
                q, r = random_bot.get_move(state, board)

            state = state.apply_move(board, q, r)

        if board.winner() == model_player:
            wins += 1

    log.info("validation_complete", wins=wins, games=n_games)
    min_wins = 95
    if wins >= min_wins:
        console.print(f"[green]Validation passed: {wins}/{n_games} wins vs RandomBot[/green]")
    elif wins > 0:
        log.warning("validation_below_threshold", wins=wins, threshold=min_wins, games=n_games)
        console.print(
            f"[yellow]Validation: {wins}/{n_games} wins vs RandomBot "
            f"(below ≥{min_wins} threshold — investigate before proceeding)[/yellow]"
        )
    else:
        console.print(
            "[yellow]Validation: 0 wins vs RandomBot "
            "(expected after very brief training — checkpoint format is correct)[/yellow]"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def pretrain() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap pretrain pipeline")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of full passes over the dataset")
    parser.add_argument("--steps", type=int, default=None,
                        help="Hard step budget (overrides epochs; for smoke tests)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile even if config enables it")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a full pretrain checkpoint (model+optimizer+scaler state). "
                             "Cosine LR schedule is restarted across --epochs at --lr-peak (or config default).")
    parser.add_argument("--lr-peak", type=float, default=None,
                        help="Override peak LR for cosine restart (used with --resume; "
                             "lower than original 2e-3 to fine-tune without disturbing learned weights).")
    parser.add_argument("--inference-out", type=str, default=None,
                        help="Override inference-weights output path "
                             "(default: checkpoints/bootstrap_model.pt). "
                             "Use a v7e30-style filename for --resume runs.")
    parser.add_argument("--eta-min", type=float, default=None,
                        help="Override CosineAnnealingLR eta_min "
                             "(default 1e-5). For 30-epoch full retrains, "
                             "5e-5 avoids the §149-observed LR-floor stall "
                             "in the final 3 epochs.")
    # ── v8 / Phase B variant CLI overrides ─────────────────────────────────
    _registered_encodings = tuple(s.name for s in _all_specs())
    parser.add_argument("--encoding", choices=_registered_encodings, default=None,
                        help="Override encoding.version from configs/model.yaml. "
                             "Registered encodings: " + ", ".join(_registered_encodings) + ". "
                             "Routes corpus NPZ and model construction accordingly.")
    parser.add_argument("--filters", type=int, default=None,
                        help="Override trunk channel count (model.yaml: filters).")
    parser.add_argument("--res-blocks", type=int, default=None,
                        help="Override trunk depth (model.yaml: res_blocks).")
    parser.add_argument("--gpool-sites", type=str, default=None,
                        help="Comma-separated trunk gpool indices for v8 "
                             "(e.g. '6,10' for a 12-block trunk). Empty string "
                             "disables trunk gpool (B0 control). v6 ignores this.")
    parser.add_argument("--head-no-gpool", action="store_true",
                        help="Drop the G branch from the v8 policy / opp_reply head. "
                             "B0 control arm uses this. v6 ignores.")
    parser.add_argument("--pool-type", choices=("min_max", "pma", "pma_global"), default=None,
                        help="K-cluster pool type for v6 / v6w25. 'min_max' "
                             "(default) keeps existing per-cluster heads + "
                             "bot-side scatter-max; 'pma' (§169 A2) replaces "
                             "the value/policy heads with a Set-Transformer "
                             "1×SAB + 2 PMA seeds aggregator. 'pma_global' "
                             "(§169 A3) extends 'pma' with a global summary "
                             "token branch (32×32 cur/opp/canvas-mask crop "
                             "+ KataGo gpool + learned scalar gate); requires "
                             "global_crops in the corpus NPZ. v8 ignores.")
    parser.add_argument("--pool-attn-dropout", type=float, default=None,
                        help="Attention dropout for PMA pool (collapse "
                             "mitigation). Default 0.1; raise to 0.2 if "
                             "the §169 PMA-collapse smoke fires.")
    parser.add_argument("--canvas-realness", action="store_true",
                        help="§169 A4 — invert v8 plane 8 polarity to "
                             "canvas_realness (1 inside, 0 outside) and wire "
                             "PartialConv2d (Innamorati 2018 partial-conv-padding) "
                             "at the trunk entry. v8-only. Requires a corpus "
                             "regenerated with --canvas-realness.")
    parser.add_argument(
        "--gpool-bias-active", action="store_true",
        help="§170 P3 — A1 + gpool-bias side-branch (additive K-invariant "
             "global-pool bias to value/policy heads). Requires --pool-type "
             "min_max + global_crops in the corpus NPZ. v6/v6w25 only.",
    )
    parser.add_argument(
        "--policy-only-bias", action="store_true",
        help="§170 P4 — confine the gpool-bias side-branch to the policy "
             "head (value head structurally frozen at A1: value_bias = 0, "
             "no gradient through value_proj). Requires --gpool-bias-active. "
             "Discriminates whether the §170 P3 MCTS regression is caused by "
             "value-head bias drift specifically.",
    )
    parser.add_argument("--corpus-npz", type=str, default=None,
                        help="Override corpus NPZ path. Default resolved from "
                             "encoding registry (resolve_corpus_path).")
    parser.add_argument(
        "--freeze-trunk-entry", action="store_true",
        help="§171 A4 fine-tune — freeze trunk.input_conv (PartialConv2d when "
             "canvas_realness) + trunk.input_gn so the trunk-entry stays at the "
             "weights learned during canvas_realness pretraining.",
    )
    parser.add_argument(
        "--unfreeze-blocks", type=str, default=None,
        help="§171 A4 fine-tune — CSV indices of trunk.tower blocks to keep "
             "trainable (e.g. '8,9,10,11'). All other blocks freeze; heads "
             "(policy/opp_reply/value/value_var) stay trainable. Unset = all "
             "blocks remain trainable.",
    )
    args = parser.parse_args()

    # Load configs
    from hexo_rl.utils.config import load_config
    config: Dict = load_config("configs/model.yaml", "configs/training.yaml")
    corpus_cfg: Dict = load_config("configs/corpus.yaml")
    if args.batch_size:
        config["batch_size"] = args.batch_size
    batch_size = int(config.get("batch_size", 512))
    label_smoothing = float(corpus_cfg.get("label_smoothing_default", 0.05))
    aux_weight = float(config.get("aux_opp_reply_weight", 0.15))
    source_weights: Dict[str, float] = corpus_cfg.get("source_weights", {})

    # ── v8 / variant overrides — CLI takes precedence over model.yaml ──────
    if args.encoding is not None:
        config["encoding"] = args.encoding
        # §172 A10 T6 — `board_size` / `in_channels` were retired from configs in
        # favor of registry lookup, but `configs/model.yaml` still carries the v6
        # defaults so legacy code paths keep working. CLI override means the
        # registry is canonical; drop scattered keys so resolve_from_config does
        # not raise on the now-stale model.yaml values.
        for stale_key in ("board_size", "in_channels", "n_planes",
                          "cluster_window_size", "cluster_threshold",
                          "legal_move_radius"):
            config.pop(stale_key, None)
    enc_section = config.get("encoding")
    if isinstance(enc_section, str):
        encoding = enc_section
    elif isinstance(enc_section, dict):
        encoding = str(enc_section.get("version", "v6"))
    else:
        encoding = "v6"
    # Validate encoding name against canonical registry.
    _ = _lookup_encoding(encoding)

    # Encoding-specific shape overrides — single source of truth via registry.
    # board_size / in_channels / n_actions scalars retired from configs (§172 A10).
    _enc_spec = _lookup_encoding(encoding)
    config["in_channels"] = _enc_spec.n_planes
    explicit_n_actions = _enc_spec.n_actions

    # Variant overrides — used by Phase B B0..B4 retrains.
    if args.filters is not None:
        config["filters"] = int(args.filters)
    if args.res_blocks is not None:
        config["res_blocks"] = int(args.res_blocks)

    # Parse --gpool-sites into a list of ints (or empty list if explicitly "").
    gpool_indices: Optional[List[int]] = None
    if args.gpool_sites is not None:
        if args.gpool_sites.strip() == "":
            gpool_indices = []
        else:
            gpool_indices = [int(s.strip()) for s in args.gpool_sites.split(",")
                             if s.strip()]
    head_use_gpool: bool = not args.head_no_gpool

    # §169 K-cluster pool selector — defaults to 'min_max' (current behavior).
    pool_type: str = (
        args.pool_type
        if args.pool_type is not None
        else str(config.get("pool_type", "min_max"))
    )
    pool_attn_dropout: float = (
        float(args.pool_attn_dropout)
        if args.pool_attn_dropout is not None
        else float(config.get("pool_attn_dropout", 0.1))
    )

    # §169 A4 — canvas_realness gates inverted plane-8 + PartialConv2d at
    # trunk entry. v8-only; surfaced loudly otherwise.
    canvas_realness: bool = bool(args.canvas_realness or config.get("canvas_realness", False))
    if canvas_realness and encoding != "v8":
        raise ValueError(
            f"--canvas-realness requires --encoding v8; got {encoding!r}"
        )

    # §170 P3 — A1 + gpool-bias. K-cluster-only (v6/v6w25), pool_type='min_max',
    # mutually exclusive w/ canvas_realness + trunk gpool sites. Mirror the
    # model-constructor invariants so YAML typos fail at CLI parse, not silently
    # mid-training. Model double-checks at construction (network.py).
    gpool_bias_active: bool = bool(
        args.gpool_bias_active or config.get("gpool_bias_active", False)
    )
    policy_only_bias: bool = bool(
        args.policy_only_bias or config.get("policy_only_bias", False)
    )
    if policy_only_bias and not gpool_bias_active:
        raise ValueError(
            "--policy-only-bias requires --gpool-bias-active; the policy-only "
            "knob configures the GpoolBiasBranch and has no effect without "
            "the branch being active."
        )
    if gpool_bias_active:
        if encoding == "v8":
            raise ValueError(
                "--gpool-bias-active requires K-cluster encoding (v6/v6w25); "
                f"v8 has no K dim. Got encoding={encoding!r}."
            )
        if pool_type != "min_max":
            raise ValueError(
                "--gpool-bias-active requires --pool-type min_max; got "
                f"pool_type={pool_type!r}. The pma / pma_global pools already "
                "carry a global-token branch; gpool-bias is the A1-only analog."
            )
        if canvas_realness:
            raise ValueError(
                "--gpool-bias-active is incompatible with --canvas-realness "
                "(canvas_realness is v8-only; gpool_bias is K-cluster-only)."
            )
        if gpool_indices:
            raise ValueError(
                "--gpool-bias-active is incompatible with non-empty "
                f"--gpool-sites; got {gpool_indices!r}. Trunk gpool sites are "
                "an in-trunk feature-mixing intervention; gpool-bias is the "
                "additive head-level analog."
            )

    log.info(
        "encoding_resolved",
        encoding=encoding,
        board_size=_registry_resolve_cfg(config).trunk_size,
        in_channels=int(config.get("in_channels", BUFFER_CHANNELS)),
        filters=int(config.get("filters", 128)),
        res_blocks=int(config.get("res_blocks", 12)),
        gpool_indices=gpool_indices,
        head_use_gpool=head_use_gpool,
        pool_type=pool_type,
        pool_attn_dropout=pool_attn_dropout,
        canvas_realness=canvas_realness,
        gpool_bias_active=gpool_bias_active,
        policy_only_bias=policy_only_bias,
    )

    from hexo_rl.utils.device import best_device
    device = best_device()
    console.print(f"[bold]Pretrain — device:[/bold] {device}")

    # Quality scores
    quality_path = Path("data/corpus/quality_scores.json")
    quality_scores: Dict = {}
    if quality_path.exists():
        with open(quality_path) as f:
            quality_scores = json.load(f)
        log.info("loaded_quality_scores", n_games=len(quality_scores))
    else:
        log.warning("no_quality_scores", hint="Run `make corpus.analysis` to generate them")

    # Corpus — prefer mmap'd NPZ to avoid 2× RAM peak from load_corpus()
    console.print("[bold]Loading corpus...[/bold]")
    if args.corpus_npz is not None:
        npz_path = Path(args.corpus_npz)
    else:
        corpus_enc = encoding
        if encoding == "v8" and canvas_realness:
            corpus_enc = "v8_canvas_realness"
        npz_path = _resolve_corpus_path(_lookup_encoding(corpus_enc))
    global_crops_array: Optional[np.ndarray] = None
    if npz_path.exists():
        log.info("loading_corpus_from_npz", path=str(npz_path))
        data = np.load(npz_path, mmap_mode='r')
        states   = data['states']    # memory-mapped, not loaded into RAM
        policies = data['policies']
        outcomes = data['outcomes']
        weights  = data['weights']
        # §169 A3 — opt-in global-crop array. Present iff the NPZ was built
        # with --with-global-crop. Required when pool_type='pma_global'.
        if 'global_crops' in data.files:
            global_crops_array = data['global_crops']
    else:
        log.warning("npz_not_found_falling_back_to_load_corpus", path=str(npz_path))
        states, policies, outcomes, weights = load_corpus(quality_scores, source_weights)

    # §170 P3 — gpool_bias_active also consumes global_crops. Same NPZ as
    # pma_global; the gate=0 init keeps A1 byte-exact at construction.
    needs_global = (pool_type == "pma_global") or gpool_bias_active
    if needs_global and global_crops_array is None:
        raise RuntimeError(
            f"pool_type='{pool_type}' / gpool_bias_active={gpool_bias_active} "
            f"requires a corpus NPZ with global_crops; none found in {npz_path}. "
            f"Regenerate via `python scripts/export_corpus_npz.py --encoding v6w25 "
            f"--with-global-crop --human-only --no-compress`."
        )
    if not needs_global and global_crops_array is not None:
        # Don't waste IO / GPU memory; drop the unused array.
        log.info(
            "ignoring_global_crops_in_npz",
            reason=f"pool_type={pool_type!r} + gpool_bias_active={gpool_bias_active}",
        )
        global_crops_array = None

    if len(outcomes) == 0:
        console.print("[red]No corpus data found. Run `make corpus.fetch` first.[/red]")
        return

    log.info(
        "dataset_built",
        n_positions=int(len(outcomes)),
        quality_scores_loaded=bool(quality_scores),
        source_weights=source_weights,
    )
    console.print(f"Dataset: {len(outcomes):,} positions")

    dataset = AugmentedBootstrapDataset(
        states, policies, outcomes, global_crops=global_crops_array,
    )
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(dataset),
        replacement=True,
    )
    board_size_for_collate = _registry_resolve_cfg(config).trunk_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=make_augmented_collate(
            augment=True,
            board_size=board_size_for_collate,
            encoding=encoding,
            with_global_crop=(global_crops_array is not None),
        ),
    )

    # One-off timing of the Rust aug path so any throughput regression vs the
    # pre-Q13 Python _apply_hex_sym path is visible in the pretrain console log.
    _t0 = time.perf_counter()
    _probe_batches = min(20, len(loader))
    _it = iter(loader)
    for _ in range(_probe_batches):
        next(_it)
    _dt = time.perf_counter() - _t0
    console.print(
        f"[bold]Aug-path probe:[/bold] {_probe_batches} batches of "
        f"{batch_size} via Rust apply_symmetries_batch = {_dt*1000:.1f} ms "
        f"({(_dt/_probe_batches)*1000:.2f} ms/batch)"
    )

    # Model — encoding-aware. v8 wires gpool sites + KataGo policy head.
    model = HexTacToeNet(
        board_size=_registry_resolve_cfg(config).trunk_size,
        in_channels=int(config["in_channels"]),
        filters=int(config["filters"]),
        res_blocks=int(config["res_blocks"]),
        se_reduction_ratio=int(config.get("se_reduction_ratio", 4)),
        encoding=encoding,
        gpool_indices=gpool_indices,
        head_use_gpool=head_use_gpool,
        pool_type=pool_type,
        pool_attn_dropout=pool_attn_dropout,
        canvas_realness=canvas_realness,
        gpool_bias_active=gpool_bias_active,
        policy_only_bias=policy_only_bias,
    )
    use_compile = (
        config.get("torch_compile", True)
        and device.type == "cuda"
        and not args.no_compile
    )
    if use_compile:
        model = compile_model(model, mode="default")

    checkpoint_dir = Path(args.checkpoint_dir)
    # Compute total steps before creating trainer so the scheduler T_max is exact.
    step_budget = args.steps
    total_pretrain_steps = step_budget if step_budget is not None else args.epochs * len(loader)
    config["pretrain_total_steps"] = total_pretrain_steps
    if args.eta_min is not None:
        config["pretrain_eta_min"] = float(args.eta_min)
    # Persist v8 / variant knobs in the saved checkpoint config so post-hoc
    # consumers (eval pipeline, threat probe, viewer) can reconstruct the model
    # without re-deriving from CLI flags.
    config["gpool_indices"] = gpool_indices
    config["head_use_gpool"] = head_use_gpool
    config["pool_type"] = pool_type
    config["pool_attn_dropout"] = pool_attn_dropout
    config["canvas_realness"] = canvas_realness
    config["gpool_bias_active"] = gpool_bias_active
    config["policy_only_bias"] = policy_only_bias
    config["n_actions"] = explicit_n_actions
    trainer = BootstrapTrainer(model, config, device, checkpoint_dir)

    # Resume mode: load model/optimizer/scaler from full checkpoint, restart
    # cosine schedule across the new --epochs window with the requested peak LR.
    # Step counter is reset so the new cosine completes over the new run length.
    if args.resume:
        resume_path = Path(args.resume)
        log.info("resume_loading", path=str(resume_path))
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        weights_only = isinstance(resume_ckpt, dict) and "model_state" not in resume_ckpt
        if weights_only:
            log.info("resume_weights_only_mode", reason="ckpt has no model_state key — treating as inference state_dict; optimizer/scaler reset")
            get_base_model(trainer.model).load_state_dict(resume_ckpt)
        else:
            get_base_model(trainer.model).load_state_dict(resume_ckpt["model_state"])
            trainer.optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            if resume_ckpt.get("scaler_state") is not None:
                trainer.scaler.load_state_dict(resume_ckpt["scaler_state"])
        new_peak = float(args.lr_peak) if args.lr_peak is not None else float(config.get("lr", 0.002))
        new_eta_min = float(args.eta_min) if args.eta_min is not None else 1e-5
        for g in trainer.optimizer.param_groups:
            g["lr"] = new_peak
            g["initial_lr"] = new_peak
        trainer.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=max(1, total_pretrain_steps), eta_min=new_eta_min,
        )
        log.info("resume_complete", new_peak_lr=new_peak,
                 cosine_t_max=total_pretrain_steps,
                 prev_step=int(resume_ckpt.get("step", 0)) if not weights_only else 0,
                 weights_only=weights_only)

    if args.freeze_trunk_entry or args.unfreeze_blocks is not None:
        unfreeze_set: Optional[set[int]] = None
        if args.unfreeze_blocks is not None:
            unfreeze_set = {int(s) for s in args.unfreeze_blocks.split(",") if s.strip()}
        freeze_report = _apply_finetune_freeze(
            get_base_model(trainer.model),
            freeze_trunk_entry=args.freeze_trunk_entry,
            unfreeze_blocks=unfreeze_set,
        )
        log.info("finetune_freeze_applied", **freeze_report)
        console.print(
            f"[yellow]§171 fine-tune freeze:[/yellow] "
            f"trainable_params={freeze_report['trainable_params']:,} / "
            f"{freeze_report['total_params']:,} "
            f"({100.0 * freeze_report['trainable_params'] / freeze_report['total_params']:.1f}%)"
        )

    # Training loop
    console.print(
        f"[bold]Training:[/bold] epochs={args.epochs} batch={batch_size} "
        f"label_smooth={label_smoothing} aux_weight={aux_weight}"
        + (f" RESUME from {args.resume} peak_lr={trainer.optimizer.param_groups[0]['lr']:.1e}" if args.resume else "")
    )
    trainer.step = -total_pretrain_steps
    start_step = trainer.step

    chain_weight = float(config.get("aux_chain_weight", 0.0))
    prev_loss: Optional[float] = None
    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(
            loader,
            label_smoothing=label_smoothing,
            aux_weight=aux_weight,
            chain_weight=chain_weight,
            step_budget=step_budget,
            start_step=start_step,
        )
        log.info("epoch_complete", epoch=epoch, **{k: round(v, 4) for k, v in metrics.items()})
        console.print(
            f"Epoch {epoch}/{args.epochs}  "
            f"loss={metrics['loss']:.4f}  "
            f"policy={metrics['policy_loss']:.4f}  "
            f"value={metrics['value_loss']:.4f}  "
            f"aux={metrics['opp_reply_loss']:.4f}  "
            f"chain={metrics['chain_loss']:.4f}"
        )
        if step_budget is not None and (trainer.step - start_step) >= step_budget:
            break
        prev_loss = metrics["loss"]

    inf_out = Path(args.inference_out) if args.inference_out else None
    ckpt_path = trainer.save_checkpoint(inf_out=inf_out)
    console.print(f"[green]Checkpoint: {ckpt_path}[/green]")

    # validate() walks GameState.to_tensor() / KEPT_PLANE_INDICES — v6 only.
    # v8 validation is deferred to Gate 4 (SealBot WR + threat probe). The
    # v8 retrain harness skips the RandomBot smoke pass with an info log so
    # future v8 self-play (§168 Phase D) can re-introduce a v8-aware probe.
    if encoding == "v8":
        log.info(
            "skipping_validate_v8",
            reason="validate() is v6-only; v8 quality measured via SealBot WR / threat probe",
            ckpt=str(ckpt_path),
        )
    else:
        validate(ckpt_path, device)


if __name__ == "__main__":
    pretrain()
