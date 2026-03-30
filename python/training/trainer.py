"""
Trainer — FP16 training step with policy and value loss.

Architecture spec (docs/01_architecture.md §2):
    Optimizer:  AdamW, lr=2e-3, weight_decay=1e-4
    Loss:       L = L_policy + L_value
                L_policy = -sum(π_mcts · log π_net)   (cross-entropy)
                L_value  = MSE(v_net, z)   z ∈ {-1, 0, +1}
    Mixed prec: torch.autocast(device_type, dtype=float16) + GradScaler

Checkpointing every N steps:
    checkpoint_<step>.pt     — full state (model + optimizer + meta)
    inference_only.pt        — model weights only (for deployment)
    checkpoint_log.json      — step, loss values
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]

from python.model.network import HexTacToeNet
from python.training.replay_buffer import ReplayBuffer


class Trainer:
    """Manages one training step and checkpoint I/O.

    Args:
        model:      HexTacToeNet instance.
        config:     Config dict (loaded from yaml).  Required keys:
                        lr, weight_decay, batch_size, checkpoint_interval,
                        log_interval, board_size.
        checkpoint_dir: Directory to write checkpoints into.
        device:     torch.device (default: cuda if available, else cpu).
    """

    def __init__(
        self,
        model: HexTacToeNet,
        config: Dict[str, Any],
        checkpoint_dir: str | Path = "checkpoints",
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        self.scheduler = self._build_scheduler(config)

        # GradScaler for FP16 training; no-op on CPU.
        self.scaler = GradScaler(device=self.device.type, enabled=(self.device.type == "cuda"))

        self.step = 0
        self.checkpoint_log: list = []

        # Load log if it already exists (resuming from checkpoint).
        log_path = self.checkpoint_dir / "checkpoint_log.json"
        if log_path.exists():
            with open(log_path) as f:
                self.checkpoint_log = json.load(f)

    def _build_scheduler(self, config: Dict[str, Any]):
        schedule = str(config.get("lr_schedule", "none")).lower()
        if schedule in {"none", "off", "disabled"}:
            return None

        if schedule == "cosine":
            total_steps = int(config.get("scheduler_t_max", config.get("total_steps", 50_000)))
            total_steps = max(1, total_steps)
            min_lr = float(config.get("min_lr", 1e-5))
            return CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=min_lr)

        raise ValueError(f"Unsupported lr_schedule: {schedule}")

    # ── Training step ─────────────────────────────────────────────────────────

    def train_step(
        self, buffer: ReplayBuffer
    ) -> Dict[str, float]:
        """Sample a batch from `buffer` and perform one gradient update.

        Returns:
            dict with keys "loss", "policy_loss", "value_loss".
        """
        batch_size = int(self.config["batch_size"])

        states, policies, outcomes = buffer.sample(batch_size)

        # Move to device; keep float16 states as-is for autocast.
        states_t   = torch.from_numpy(states).to(self.device)       # float16
        policies_t = torch.from_numpy(policies).to(self.device)     # float32
        outcomes_t = torch.from_numpy(outcomes).to(self.device)     # float32

        self.optimizer.zero_grad()

        with autocast(device_type=self.device.type, dtype=torch.float16,
                      enabled=(self.device.type == "cuda")):
            log_policy, value = self.model(states_t)

            # Policy loss: cross-entropy with MCTS visit distribution.
            # log_policy is already log_softmax; policies_t is the target.
            policy_loss = -(policies_t * log_policy).sum(dim=1).mean()

            # Value loss: MSE between predicted and actual game outcome.
            value_loss = nn.functional.mse_loss(value.squeeze(1), outcomes_t)

            loss = policy_loss + value_loss

        if self.device.type == "cuda":
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.step += 1

        result = {
            "loss":        loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss":  value_loss.item(),
        }

        interval = int(self.config.get("checkpoint_interval", 100))
        if self.step % interval == 0:
            self.save_checkpoint(result)

        return result

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_model_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize model state dict keys across compiled/non-compiled models."""
        if not state_dict:
            return state_dict

        has_orig_prefix = any(k.startswith("_orig_mod.") for k in state_dict.keys())
        if not has_orig_prefix:
            return state_dict

        prefix = "_orig_mod."
        return {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }

    def save_checkpoint(self, loss_info: Optional[Dict[str, float]] = None) -> Path:
        """Save full checkpoint and inference-only weights.

        Returns path to the checkpoint file.
        """
        base_model = getattr(self.model, "_orig_mod", self.model)
        model_state = base_model.state_dict()

        ckpt_path = self.checkpoint_dir / f"checkpoint_{self.step:08d}.pt"
        torch.save(
            {
                "step":            self.step,
                "model_state":     model_state,
                "optimizer_state": self.optimizer.state_dict(),
                "scaler_state":    self.scaler.state_dict(),
                "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
                "config":          self.config,
            },
            ckpt_path,
        )

        # Inference-only copy (weights only, no optimizer state).
        inf_path = self.checkpoint_dir / "inference_only.pt"
        torch.save(model_state, inf_path)

        # Update log.
        entry: Dict[str, Any] = {"step": self.step}
        if loss_info:
            entry.update(loss_info)
        self.checkpoint_log.append(entry)
        with open(self.checkpoint_dir / "checkpoint_log.json", "w") as f:
            json.dump(self.checkpoint_log, f, indent=2)

        return ckpt_path

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str | Path,
        checkpoint_dir: Optional[str | Path] = None,
        device: Optional[torch.device] = None,
        fallback_config: Optional[Dict[str, Any]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> "Trainer":
        """Restore a Trainer from a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint_<step>.pt.
            checkpoint_dir:  Where to write future checkpoints (defaults to
                             the same directory as the checkpoint file).
            device:          Override device.
            fallback_config: Config to use if the checkpoint is weights-only.
            config_overrides: Optional config keys to override after loading
                              checkpoint config (useful for controlled resume
                              behavior such as scheduler horizon changes).
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        
        # If it's a full checkpoint, it should have a 'config' or 'model_state' key.
        # If it's a weights-only state_dict, it will have keys like 'resnet.0.weight' etc.
        is_full_ckpt = "model_state" in ckpt and "config" in ckpt
        
        if is_full_ckpt:
            config = dict(ckpt["config"])
            model_state = ckpt["model_state"]
        else:
            if fallback_config is None:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} appears to be weights-only, "
                    "but no fallback_config was provided."
                )
            config = dict(fallback_config)
            model_state = ckpt

        if config_overrides:
            config.update(config_overrides)

        model_state = cls._normalize_model_state_dict_keys(model_state)

        board_size  = int(config.get("board_size", 19))
        res_blocks  = int(config.get("res_blocks", 10))
        filters     = int(config.get("filters", 128))

        model = HexTacToeNet(
            board_size=board_size,
            res_blocks=res_blocks,
            filters=filters,
        )
        model.load_state_dict(model_state)

        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else Path(checkpoint_path).parent
        trainer = cls(model, config, checkpoint_dir=ckpt_dir, device=device)
        
        if is_full_ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
            trainer.scaler.load_state_dict(ckpt["scaler_state"])
            if trainer.scheduler is not None and ckpt.get("scheduler_state") is not None:
                scheduler_state = ckpt["scheduler_state"]
                # Optional: let explicit config overrides update scheduler horizon.
                if config_overrides and (
                    "scheduler_t_max" in config_overrides or "total_steps" in config_overrides
                ):
                    scheduler_state = dict(scheduler_state)
                    scheduler_state["T_max"] = int(
                        config.get("scheduler_t_max", config.get("total_steps", scheduler_state.get("T_max", 1)))
                    )
                trainer.scheduler.load_state_dict(scheduler_state)
            trainer.step = ckpt["step"]

        return trainer
