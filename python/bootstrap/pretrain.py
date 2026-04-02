"""
Bootstrap pretraining pipeline — Phase 4.0 architecture.

Loads human + bot corpus, applies quality/source weighting, and trains the
HexTacToeNet with Phase 4.0 losses:

    L = L_policy + L_value_BCE + aux_weight × L_opp_reply

Saves a checkpoint in the same format as Trainer.save_checkpoint() so that
scripts/train.py can resume from it seamlessly.

Usage:
    python -m python.bootstrap.pretrain [--epochs N] [--steps N] [--batch-size N]
    make pretrain          # 5 epochs (default)
    make pretrain.full     # 15 epochs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import structlog
from rich.console import Console

from native_core import Board
from python.bootstrap.dataset import replay_game_to_triples
from python.bootstrap.generate_corpus import BOT_GAMES_DIR, RAW_HUMAN_DIR
from python.env.game_state import GameState
from python.model.network import HexTacToeNet, compile_model
from python.training.losses import (
    compute_policy_loss, compute_value_loss, compute_aux_loss,
    compute_total_loss, fp16_backward_step,
)
from python.training.checkpoints import save_full_checkpoint, save_inference_weights
from python.utils.constants import BOARD_SIZE

log = structlog.get_logger()
console = Console()

POLICY_SIZE = BOARD_SIZE * BOARD_SIZE + 1

# ── Hex augmentation ──────────────────────────────────────────────────────────

_HEX_SYMS: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None


def _precompute_hex_syms(board_size: int = BOARD_SIZE) -> List[Tuple]:
    """Build 12 hexagonal symmetry scatter tables (one per symmetry).

    Mirrors the Rust SymTables in native_core/src/replay_buffer.rs:
      - 6 rotations of 60° each:  (q,r) → (−r, q+r)
      - 2 reflections (with/without): (q,r) → (r,q) applied first

    Each entry is (src_rows, src_cols, dst_rows, dst_cols) — int32 arrays of
    valid cell pairs. Cells that map outside the 19×19 window are dropped.
    """
    center = board_size // 2
    N = board_size
    syms: List[Tuple] = []

    for sym_idx in range(12):
        do_reflect = sym_idx >= 6
        n_rot = sym_idx % 6

        src_rs: List[int] = []
        src_cs: List[int] = []
        dst_rs: List[int] = []
        dst_cs: List[int] = []

        for src_row in range(N):
            for src_col in range(N):
                q, r = src_col - center, src_row - center
                if do_reflect:
                    q, r = r, q
                for _ in range(n_rot):
                    q, r = -r, q + r
                dst_row = r + center
                dst_col = q + center
                if 0 <= dst_row < N and 0 <= dst_col < N:
                    src_rs.append(src_row)
                    src_cs.append(src_col)
                    dst_rs.append(dst_row)
                    dst_cs.append(dst_col)

        syms.append((
            np.array(src_rs, dtype=np.int32),
            np.array(src_cs, dtype=np.int32),
            np.array(dst_rs, dtype=np.int32),
            np.array(dst_cs, dtype=np.int32),
        ))
    return syms


def _get_hex_syms() -> List[Tuple]:
    global _HEX_SYMS
    if _HEX_SYMS is None:
        _HEX_SYMS = _precompute_hex_syms()
    return _HEX_SYMS


def _apply_hex_sym(
    state: np.ndarray,
    policy: np.ndarray,
    sym: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply one hex symmetry to a (state, policy) pair.

    Args:
        state:  (18, N, N) float16 array.
        policy: (N*N+1,) float32 array.
        sym:    (src_rows, src_cols, dst_rows, dst_cols)

    Returns new (state, policy) arrays — same dtypes as inputs.
    """
    src_rs, src_cs, dst_rs, dst_cs = sym
    N = state.shape[-1]

    new_state = np.zeros_like(state)
    new_state[:, dst_rs, dst_cs] = state[:, src_rs, src_cs]

    new_policy = np.zeros_like(policy)
    new_policy[dst_rs * N + dst_cs] = policy[src_rs * N + src_cs]
    new_policy[N * N] = policy[N * N]   # pass move is invariant under all symmetries

    return new_state, new_policy


# ── Dataset ───────────────────────────────────────────────────────────────────

class AugmentedBootstrapDataset(torch.utils.data.Dataset):
    """Pretrain dataset with optional 12-fold hex augmentation per sample.

    Args:
        states:   (N, 18, 19, 19) float16 array.
        policies: (N, 362) float32 array.
        outcomes: (N,) float32 array, values in {-1, 0, +1}.
        augment:  If True, apply a randomly chosen hex symmetry per sample.
    """

    def __init__(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        outcomes: np.ndarray,
        augment: bool = True,
    ) -> None:
        self.states = states
        self.policies = policies
        self.outcomes = outcomes
        self.augment = augment
        self.syms = _get_hex_syms() if augment else None

    def __len__(self) -> int:
        return len(self.outcomes)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.states[idx].copy()
        p = self.policies[idx].copy()
        v = float(self.outcomes[idx])

        if self.augment and self.syms is not None:
            sym = self.syms[np.random.randint(12)]
            s, p = _apply_hex_sym(s, p, sym)

        return (
            torch.from_numpy(s),                        # float16
            torch.from_numpy(p),                        # float32
            torch.tensor(v, dtype=torch.float32),
        )


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
        s, p, o = replay_game_to_triples(moves, winner)
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

    # Bot d4
    d4_ok = 0
    d4_dir = BOT_GAMES_DIR / "sealbot_d4"
    if d4_dir.exists():
        for path in sorted(d4_dir.glob("*.json")):
            try:
                with open(path) as f:
                    d = json.load(f)
                moves = [(int(m["x"]), int(m["y"])) for m in d["moves"]]
                winner = int(d["winner"]) if "winner" in d else _game_winner_from_replay(moves)
                if winner is None or winner == 0:
                    continue
                _add_game(moves, winner, path.stem, "bot_d4")
                d4_ok += 1
            except Exception:
                continue
    log.info("loaded_d4_games", count=d4_ok)

    # Bot d6
    d6_ok = 0
    d6_dir = BOT_GAMES_DIR / "sealbot_d6"
    if d6_dir.exists():
        for path in sorted(d6_dir.glob("*.json")):
            try:
                with open(path) as f:
                    d = json.load(f)
                moves = [(int(m["x"]), int(m["y"])) for m in d["moves"]]
                winner = int(d["winner"]) if "winner" in d else _game_winner_from_replay(moves)
                if winner is None or winner == 0:
                    continue
                _add_game(moves, winner, path.stem, "bot_d6")
                d6_ok += 1
            except Exception:
                continue
    log.info("loaded_d6_games", count=d6_ok)

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
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_steps), eta_min=1e-5,
        )

    def train_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        label_smoothing: float = 0.05,
        aux_weight: float = 0.15,
        step_budget: Optional[int] = None,
        log_interval: int = 50,
    ) -> Dict[str, float]:
        """One full pass over the dataloader.

        Args:
            loader:         DataLoader yielding (states, policies, outcomes).
            label_smoothing: ε for policy targets; 0 disables.
            aux_weight:     Weight for opponent-reply auxiliary loss.
            step_budget:    Stop after this many steps (for smoke tests).

        Returns:
            Dict with keys loss, policy_loss, value_loss, opp_reply_loss.
        """
        self.model.train()
        total: Dict[str, float] = {
            "loss": 0.0, "policy_loss": 0.0,
            "value_loss": 0.0, "opp_reply_loss": 0.0,
        }
        n_batches = 0

        for states, policies, outcomes in loader:
            states   = states.to(self.device)
            policies = policies.to(self.device)
            outcomes = outcomes.to(self.device)

            if label_smoothing > 0.0:
                n_actions = policies.shape[1]
                policies = policies * (1.0 - label_smoothing) + label_smoothing / n_actions

            self.optimizer.zero_grad()

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.fp16,
            ):
                log_policy, _value, v_logit, opp_reply = self.model(states, aux=True)

                policy_valid = policies.sum(dim=1) > 1e-6
                policy_loss = compute_policy_loss(log_policy, policies, policy_valid, self.device)
                value_loss = compute_value_loss(v_logit, outcomes)
                opp_reply_loss = compute_aux_loss(opp_reply, policies, policy_valid, self.device)
                loss = compute_total_loss(policy_loss, value_loss, opp_reply_loss, aux_weight)

            fp16_backward_step(loss, self.optimizer, self.scaler, self.model, self.fp16)

            self.scheduler.step()
            self.step += 1

            step_loss = loss.item()
            total["loss"]           += step_loss
            total["policy_loss"]    += policy_loss.item()
            total["value_loss"]     += value_loss.item()
            total["opp_reply_loss"] += opp_reply_loss.item()
            n_batches += 1

            if log_interval > 0 and self.step % log_interval == 0:
                log.info(
                    "train_step",
                    step=self.step,
                    loss=round(step_loss, 4),
                    policy_loss=round(policy_loss.item(), 4),
                    value_loss=round(value_loss.item(), 4),
                )

            if step_budget is not None and self.step >= step_budget:
                break

        n = max(n_batches, 1)
        return {k: v / n for k, v in total.items()}

    def save_checkpoint(self) -> Path:
        """Save full checkpoint in self-play-compatible format.

        Also writes checkpoints/bootstrap_model.pt (weights only) for the
        eval pipeline.
        """
        ckpt_path = self.checkpoint_dir / f"pretrain_{self.step:08d}.pt"
        save_full_checkpoint(
            self.model, self.optimizer, self.scaler, self.scheduler,
            self.step, self.config, ckpt_path,
        )
        inf_path = Path("checkpoints") / "bootstrap_model.pt"
        save_inference_weights(self.model, inf_path)
        log.info("checkpoint_saved", path=str(ckpt_path), inference=str(inf_path))
        return ckpt_path


# ── Validation ────────────────────────────────────────────────────────────────

def validate(ckpt_path: Path, device: torch.device) -> None:
    """Verify checkpoint round-trip and play greedy games vs RandomBot.

    Uses argmax policy (no MCTS) — suitable for a pretrained but not
    yet self-play-trained checkpoint.
    """
    from python.bootstrap.bots.random_bot import RandomBot

    # Round-trip: load checkpoint, rebuild model, run forward pass.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert "model_state" in ckpt and "config" in ckpt, (
        f"Checkpoint missing keys; got {list(ckpt.keys())}"
    )
    cfg = ckpt["config"]
    loaded_model = HexTacToeNet(
        board_size=int(cfg.get("board_size", 19)),
        in_channels=int(cfg.get("in_channels", 18)),
        filters=int(cfg.get("filters", 128)),
        res_blocks=int(cfg.get("res_blocks", 12)),
        se_reduction_ratio=int(cfg.get("se_reduction_ratio", 4)),
    )
    loaded_model.load_state_dict(ckpt["model_state"])
    loaded_model.eval().to(device)

    dummy = torch.zeros(1, 18, BOARD_SIZE, BOARD_SIZE, device=device)
    with torch.no_grad():
        log_pol, val, v_logit = loaded_model(dummy.float())
    assert log_pol.shape == (1, POLICY_SIZE), f"Unexpected policy shape: {log_pol.shape}"
    log.info("checkpoint_forward_pass_ok", val=float(val[0, 0]))

    # Play 5 greedy games vs RandomBot — expect >0 wins after even brief training.
    random_bot = RandomBot()
    wins = 0
    n_games = 5

    for i in range(n_games):
        board = Board()
        state = GameState.from_board(board)
        model_player = 1 if i % 2 == 0 else -1

        for _ in range(200):
            if board.check_win() or board.legal_move_count() == 0:
                break

            if board.current_player == model_player:
                tensor, centers = state.to_tensor()
                inp = torch.from_numpy(tensor[0]).unsqueeze(0).to(device).float()
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
    if wins > 0:
        console.print(f"[green]Validation passed: {wins}/{n_games} wins vs RandomBot[/green]")
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
    args = parser.parse_args()

    # Load configs
    from python.utils.config import load_config
    config: Dict = load_config("configs/training.yaml", "configs/model.yaml")
    corpus_cfg: Dict = load_config("configs/corpus_filter.yaml")
    if args.batch_size:
        config["batch_size"] = args.batch_size
    batch_size = int(config.get("batch_size", 512))
    label_smoothing = float(corpus_cfg.get("label_smoothing_default", 0.05))
    aux_weight = float(config.get("aux_opp_reply_weight", 0.15))
    source_weights: Dict[str, float] = corpus_cfg.get("source_weights", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Corpus
    console.print("[bold]Loading corpus...[/bold]")
    states, policies, outcomes, weights = load_corpus(quality_scores, source_weights)

    if len(outcomes) == 0:
        console.print("[red]No corpus data found. Run `make corpus.all` first.[/red]")
        return

    log.info(
        "dataset_built",
        n_positions=int(len(outcomes)),
        quality_scores_loaded=bool(quality_scores),
        source_weights=source_weights,
    )
    console.print(f"Dataset: {len(outcomes):,} positions")

    dataset = AugmentedBootstrapDataset(states, policies, outcomes, augment=True)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(dataset),
        replacement=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = HexTacToeNet(
        board_size=int(model_cfg["board_size"]),
        in_channels=int(model_cfg["in_channels"]),
        filters=int(model_cfg["filters"]),
        res_blocks=int(model_cfg["res_blocks"]),
        se_reduction_ratio=int(model_cfg.get("se_reduction_ratio", 4)),
    )
    use_compile = (
        config.get("torch_compile", True)
        and device.type == "cuda"
        and not args.no_compile
    )
    if use_compile:
        model = compile_model(model, mode="reduce-overhead")

    checkpoint_dir = Path(args.checkpoint_dir)
    trainer = BootstrapTrainer(model, config, device, checkpoint_dir)

    # Training loop
    console.print(
        f"[bold]Training:[/bold] epochs={args.epochs} batch={batch_size} "
        f"label_smooth={label_smoothing} aux_weight={aux_weight}"
    )
    step_budget = args.steps

    prev_loss: Optional[float] = None
    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(
            loader,
            label_smoothing=label_smoothing,
            aux_weight=aux_weight,
            step_budget=step_budget,
        )
        log.info("epoch_complete", epoch=epoch, **{k: round(v, 4) for k, v in metrics.items()})
        console.print(
            f"Epoch {epoch}/{args.epochs}  "
            f"loss={metrics['loss']:.4f}  "
            f"policy={metrics['policy_loss']:.4f}  "
            f"value={metrics['value_loss']:.4f}  "
            f"aux={metrics['opp_reply_loss']:.4f}"
        )
        if step_budget is not None and trainer.step >= step_budget:
            break
        prev_loss = metrics["loss"]

    ckpt_path = trainer.save_checkpoint()
    console.print(f"[green]Checkpoint: {ckpt_path}[/green]")

    validate(ckpt_path, device)


if __name__ == "__main__":
    pretrain()
