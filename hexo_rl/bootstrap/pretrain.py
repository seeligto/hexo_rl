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
from hexo_rl.utils.constants import BOARD_SIZE, KEPT_PLANE_INDICES
from hexo_rl.monitoring.events import emit_event
from hexo_rl.augment.luts import get_policy_scatters

log = structlog.get_logger()
console = Console()

POLICY_SIZE = BOARD_SIZE * BOARD_SIZE + 1

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

    Args:
        states:   (N, 18, 19, 19) float16 array (mmap-compatible).
        policies: (N, 362) float32 array.
        outcomes: (N,) float32 array, values in {-1, 0, +1}.
    """

    def __init__(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        outcomes: np.ndarray,
    ) -> None:
        self.states = states
        self.policies = policies
        self.outcomes = outcomes

    def __len__(self) -> int:
        return len(self.outcomes)

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        # Copy out of the (possibly mmapped) backing store so downstream
        # collate can batch-concat without aliasing.
        return (
            self.states[idx].copy(),
            self.policies[idx].copy(),
            float(self.outcomes[idx]),
        )


def make_augmented_collate(augment: bool, board_size: int = BOARD_SIZE):
    """Return a collate_fn that batches triples and applies hex augmentation
    via the Rust `engine.apply_symmetries_batch` kernel.

    With `augment=True`:
      - Stack the batch of raw states into an (N, 18, 19, 19) float32 array
        (upcast from the f16 dataset copies — the Rust binding takes f32).
      - Draw N uniform sym indices in [0, 12).
      - Call `engine.apply_symmetries_batch(states_f32, sym_indices)` →
        (N, 18, 19, 19) f32 augmented states (one PyO3 hop per batch).
      - Scatter the per-sample policies via the precomputed numpy index
        tables (`_get_policy_scatters`).
      - Cast states back to float16 (matches the pre-rewrite tensor dtype).
      - Compute chain_planes from augmented stone planes 0 (cur) and 4 (opp).

    With `augment=False`:
      - Stack as-is, no Rust hop, no policy scatter.
      - Compute chain_planes from raw stone planes.
    """
    scatters_np = get_policy_scatters(board_size) if augment else None

    def _collate(
        batch: List[Tuple[np.ndarray, np.ndarray, float]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = len(batch)
        states = np.stack([b[0] for b in batch], axis=0)          # f16 (N,8,19,19)
        policies = np.stack([b[1] for b in batch], axis=0)        # f32 (N, 362)
        outcomes = np.asarray([b[2] for b in batch], dtype=np.float32)

        if augment and scatters_np is not None:
            states_f32 = states.astype(np.float32, copy=False)
            sym_indices = np.random.randint(0, 12, size=n).astype(np.uint64)
            states_f32 = engine.apply_symmetries_batch(states_f32, sym_indices.tolist())
            # Policy scatter per row via numpy fancy indexing. Each sym has its
            # own permutation index table; build one (N, 362) dst-from-src map.
            scattered = np.empty_like(policies)
            for i in range(n):
                scattered[i] = policies[i][scatters_np[int(sym_indices[i])]]
            policies = scattered
            states = states_f32.astype(np.float16, copy=False)

        # Compute chain planes from stone planes post-augmentation.
        # Recomputing from augmented stones is self-consistent with the chain
        # head supervision — no axis-perm remap required.
        # 8-plane layout: [0,1,2,3,8,9,10,11] from original 18.
        # Cur-player ply-0 = index 0; opp ply-0 = index 4.
        chain_np = np.zeros((n, 6, board_size, board_size), dtype=np.float16)
        for i in range(n):
            chain_np[i] = _compute_chain_planes(
                states[i, 0].astype(np.float32),
                states[i, 4].astype(np.float32),
            ).astype(np.float16) / 6.0

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
            np.empty((0, POLICY_SIZE), dtype=np.float32),
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

        for states, chain_planes, policies, outcomes in loader:
            states       = states.to(self.device)
            chain_planes = chain_planes.to(self.device)
            policies     = policies.to(self.device)
            outcomes     = outcomes.to(self.device)

            if label_smoothing > 0.0:
                n_actions = policies.shape[1]
                policies = policies * (1.0 - label_smoothing) + label_smoothing / n_actions

            self.optimizer.zero_grad()

            use_chain = chain_weight > 0.0
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.fp16,
            ):
                fwd = self.model(states, aux=True, chain=use_chain)
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

                # Emit to dashboard — include loss_chain so the C14 dashboard
                # rendering surfaces the Q13-aux head during pretrain runs.
                emit_event({
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
                })

                log.info(
                    "train_step",
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


# ── Validation ────────────────────────────────────────────────────────────────

def validate(ckpt_path: Path, device: torch.device) -> None:
    """Verify checkpoint round-trip and play 100 greedy games vs RandomBot.

    Uses argmax policy (no MCTS) — suitable for a pretrained but not
    yet self-play-trained checkpoint.
    """
    from hexo_rl.bootstrap.bots.random_bot import RandomBot

    # Round-trip: load checkpoint, rebuild model, run forward pass.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert "model_state" in ckpt and "config" in ckpt, (
        f"Checkpoint missing keys; got {list(ckpt.keys())}"
    )
    cfg = ckpt["config"]
    _ckpt_model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    _ckpt_use_hex = bool(
        _ckpt_model_cfg.get("use_hex_kernel", cfg.get("use_hex_kernel", False))
    )
    loaded_model = HexTacToeNet(
        board_size=int(cfg.get("board_size", 19)),
        in_channels=int(cfg.get("in_channels", 18)),
        filters=int(cfg.get("filters", 128)),
        res_blocks=int(cfg.get("res_blocks", 12)),
        se_reduction_ratio=int(cfg.get("se_reduction_ratio", 4)),
        use_hex_kernel=_ckpt_use_hex,
    )
    loaded_model.load_state_dict(ckpt["model_state"])
    loaded_model.eval().to(device)

    dummy = torch.zeros(
        1, int(cfg.get("in_channels", 18)), BOARD_SIZE, BOARD_SIZE, device=device,
    )
    with torch.no_grad():
        log_pol, val, v_logit = loaded_model(dummy.float())
    assert log_pol.shape == (1, POLICY_SIZE), f"Unexpected policy shape: {log_pol.shape}"
    log.info("checkpoint_forward_pass_ok", val=float(val[0, 0]))

    # Play 100 greedy games vs RandomBot — expect high win rate after pretraining.
    random_bot = RandomBot()
    wins = 0
    n_games = 100

    for i in range(n_games):
        board = Board()
        state = GameState.from_board(board)
        model_player = 1 if i % 2 == 0 else -1

        for _ in range(200):
            if board.check_win() or board.legal_move_count() == 0:
                break

            if board.current_player == model_player:
                tensor, centers = state.to_tensor()
                inp = torch.from_numpy(tensor[0][KEPT_PLANE_INDICES]).unsqueeze(0).to(device).float()
                with torch.no_grad():
                    lp, _, _ = loaded_model(inp)
                lp_np = lp[0].cpu().numpy()
                cq, cr = centers[0]
                legal = board.legal_moves()
                best_move, best_score = legal[0], -1e9
                for q, r in legal:
                    wq, wr = q - cq + 9, r - cr + 9
                    if 0 <= wq < BOARD_SIZE and 0 <= wr < BOARD_SIZE:
                        score = float(lp_np[wq * BOARD_SIZE + wr])
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
    args = parser.parse_args()

    # Load configs
    from hexo_rl.utils.config import load_config
    config: Dict = load_config("configs/model.yaml", "configs/training.yaml")
    corpus_cfg: Dict = load_config("configs/corpus.yaml")
    if args.batch_size:
        config["batch_size"] = args.batch_size

    # Phase B' v9 §153 T2 — flip the engine corner-mask flag so the
    # post-train validation loop's `state.to_tensor()` calls produce
    # tensors that match the model's input expectations. The training
    # data itself comes pre-encoded from the .npz corpus, so corpus-side
    # encoding mismatches are a separate concern (regenerate corpus with
    # corner_mask=True if pretraining a hex-trunk model from scratch).
    _model_cfg_bootstrap = config.get("model") if isinstance(config.get("model"), dict) else {}
    _corner_mask_on_pretrain = bool(
        _model_cfg_bootstrap.get("corner_mask", config.get("corner_mask", False))
    )
    try:
        from engine import set_corner_mask_enabled as _set_mask  # type: ignore[attr-defined]
        _set_mask(_corner_mask_on_pretrain)
    except (ImportError, AttributeError):
        if _corner_mask_on_pretrain:
            log.warning("corner_mask_unavailable_pretrain")
    batch_size = int(config.get("batch_size", 512))
    label_smoothing = float(corpus_cfg.get("label_smoothing_default", 0.05))
    aux_weight = float(config.get("aux_opp_reply_weight", 0.15))
    source_weights: Dict[str, float] = corpus_cfg.get("source_weights", {})

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
    npz_path = Path(corpus_cfg.get("corpus_npz_path", "data/bootstrap_corpus.npz"))
    if npz_path.exists():
        log.info("loading_corpus_from_npz", path=str(npz_path))
        data = np.load(npz_path, mmap_mode='r')
        states   = data['states']    # memory-mapped, not loaded into RAM
        policies = data['policies']
        outcomes = data['outcomes']
        weights  = data['weights']
    else:
        log.warning("npz_not_found_falling_back_to_load_corpus", path=str(npz_path))
        states, policies, outcomes, weights = load_corpus(quality_scores, source_weights)

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

    dataset = AugmentedBootstrapDataset(states, policies, outcomes)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(dataset),
        replacement=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=make_augmented_collate(augment=True),
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

    # Model
    _model_cfg_pretrain = config.get("model") if isinstance(config.get("model"), dict) else {}
    _use_hex_kernel_pretrain = bool(
        _model_cfg_pretrain.get("use_hex_kernel", config.get("use_hex_kernel", False))
    )
    model = HexTacToeNet(
        board_size=int(config["board_size"]),
        in_channels=int(config["in_channels"]),
        filters=int(config["filters"]),
        res_blocks=int(config["res_blocks"]),
        se_reduction_ratio=int(config.get("se_reduction_ratio", 4)),
        use_hex_kernel=_use_hex_kernel_pretrain,
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
    trainer = BootstrapTrainer(model, config, device, checkpoint_dir)

    # Resume mode: load model/optimizer/scaler from full checkpoint, restart
    # cosine schedule across the new --epochs window with the requested peak LR.
    # Step counter is reset so the new cosine completes over the new run length.
    if args.resume:
        resume_path = Path(args.resume)
        log.info("resume_loading", path=str(resume_path))
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
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
                 prev_step=int(resume_ckpt.get("step", 0)))

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

    validate(ckpt_path, device)


if __name__ == "__main__":
    pretrain()
