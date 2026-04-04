"""
Trainer — FP16 training step with policy, value, and auxiliary losses.

Architecture spec (docs/01_architecture.md §2):
    Optimizer:  AdamW, lr=2e-3, weight_decay=1e-4
    Loss:       L = L_policy + L_value + w_aux · L_opp_reply
                L_policy     = -sum(π_mcts · log π_net)   (cross-entropy)
                L_value      = BCE(sigmoid(v_logit), (z+1)/2)
                L_opp_reply  = -sum(π_opp · log π_opp_net)  (auxiliary)
    Mixed prec: torch.autocast(device_type, dtype=float16) + GradScaler

Checkpointing every N steps:
    checkpoint_<step>.pt     — full state (model + optimizer + meta)
    inference_only.pt        — model weights only (for deployment)
    checkpoint_log.json      — step, loss values
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]

import structlog

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.losses import (
    compute_policy_loss, compute_value_loss, compute_aux_loss,
    compute_total_loss, fp16_backward_step,
)
from hexo_rl.training.checkpoints import (
    save_full_checkpoint, save_inference_weights, prune_checkpoints,
    normalize_model_state_dict_keys, get_base_model,
)
from engine import ReplayBuffer

log = structlog.get_logger()


def prune_policy_targets(
    pi: torch.Tensor, threshold_frac: float = 0.02
) -> torch.Tensor:
    """Zero out policy target entries below threshold and renormalize.

    Entries strictly at or below ``threshold_frac * max(row)`` are zeroed,
    then the row is renormalized to sum to 1. This sharpens MCTS visit
    distributions by removing exploration noise on clearly-bad moves.

    Args:
        pi: (B, A) policy target tensor (non-negative, sums to ~1).
        threshold_frac: fraction of per-row max below which entries are pruned.

    Returns:
        Pruned and renormalized tensor of same shape.
    """
    if threshold_frac <= 0.0:
        return pi
    max_vals = pi.max(dim=-1, keepdim=True).values
    mask = pi > (threshold_frac * max_vals)
    pruned = pi * mask
    return pruned / pruned.sum(dim=-1, keepdim=True).clamp(min=1e-8)


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

        fp16_requested = bool(config.get("fp16", True))
        if fp16_requested and self.device.type != "cuda":
            log.warning("fp16_disabled_no_cuda", device=str(self.device))
            fp16_requested = False
        self.fp16 = fp16_requested

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
        self.scaler = GradScaler(device=self.device.type, enabled=self.fp16)

        # torch.compile disabled — Python 3.14 compatibility issues
        # See sprint log §25, §30 for history
        # Re-enable when PyTorch + Python 3.14 CUDA graph support stabilizes
        if config.get("torch_compile", False) and self.device.type == "cuda":
            try:
                self.model = torch.compile(
                    self.model, mode="default", fullgraph=False
                )
                log.info("torch_compile_enabled", mode="default")
            except Exception as exc:
                log.warning("torch_compile_failed", error=str(exc))

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
            return CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=min_lr, last_epoch=-1)

        raise ValueError(f"Unsupported lr_schedule: {schedule}")

    # ── Training step ─────────────────────────────────────────────────────────

    def train_step(
        self, buffer: "ReplayBuffer", augment: bool = True
    ) -> Dict[str, float]:
        """Sample a batch from `buffer` and perform one gradient update.

        Args:
            buffer:  Replay buffer to sample from.
            augment: Whether to apply 12-fold hex augmentation during sampling.
                     Set False in tests that assert on loss convergence to
                     eliminate RNG-dependent variance.

        Returns:
            dict with keys "loss", "policy_loss", "value_loss",
            and optionally "opp_reply_loss".
        """
        batch_size = int(self.config["batch_size"])
        states, policies, outcomes = buffer.sample_batch(batch_size, augment)
        return self._train_on_batch(states, policies, outcomes)

    def train_step_from_tensors(
        self,
        states: "numpy.ndarray",
        policies: "numpy.ndarray",
        outcomes: "numpy.ndarray",
    ) -> Dict[str, float]:
        """Perform one gradient update from pre-built numpy arrays.

        Used by the mixed-buffer training loop (Phase 4.0) where samples
        are drawn from pretrained + self-play buffers externally.
        """
        return self._train_on_batch(states, policies, outcomes)

    def _train_on_batch(
        self,
        states: "numpy.ndarray",
        policies: "numpy.ndarray",
        outcomes: "numpy.ndarray",
    ) -> Dict[str, float]:
        """Core training step: forward, loss, backward, optimizer step."""
        import numpy  # noqa: F811 — deferred import for type alias above
        aux_weight = float(self.config.get("aux_opp_reply_weight", 0.0))

        # Move to device. With FP16/autocast, keep float16 states for the mixed-
        # precision path; without it, upcast to float32 to match model weights.
        states_t = torch.from_numpy(states).to(self.device)
        if not self.fp16:
            states_t = states_t.float()
        policies_t = torch.from_numpy(policies).to(self.device)     # float32
        outcomes_t = torch.from_numpy(outcomes).to(self.device)     # float32

        prune_frac = float(self.config.get("policy_prune_frac", 0.0))
        if prune_frac > 0.0:
            policies_t = prune_policy_targets(policies_t, prune_frac)

        self.optimizer.zero_grad()

        entropy_weight = float(self.config.get("entropy_reg_weight", 0.0))

        with autocast(device_type=self.device.type, dtype=torch.float16,
                      enabled=self.fp16):
            use_aux = aux_weight > 0.0

            if use_aux:
                log_policy, value, v_logit, opp_reply = self.model(states_t, aux=True)
            else:
                log_policy, value, v_logit = self.model(states_t)

            policy_valid = policies_t.sum(dim=1) > 1e-6
            policy_loss = compute_policy_loss(log_policy, policies_t, policy_valid, self.device)
            value_loss = compute_value_loss(v_logit, outcomes_t)

            opp_reply_loss = None
            if use_aux:
                opp_reply_loss = compute_aux_loss(opp_reply, policies_t, policy_valid, self.device)

            # Entropy regularization: subtract entropy bonus to maximize policy entropy.
            entropy_bonus = None
            if entropy_weight > 0.0:
                entropy_bonus = -(log_policy.exp() * log_policy).sum(dim=-1).mean()

            loss = compute_total_loss(policy_loss, value_loss, opp_reply_loss, aux_weight,
                                      entropy_bonus, entropy_weight)

        max_grad_norm = float(self.config.get("grad_clip", 1.0))
        grad_norm = fp16_backward_step(
            loss, self.optimizer, self.scaler, self.model, self.fp16,
            max_grad_norm=max_grad_norm,
        )

        self.step += 1

        # Step scheduler AFTER optimizer.step() (inside fp16_backward_step)
        # and after self.step increment so counters stay in sync.
        # Guard: when GradScaler detects inf/nan grads it skips optimizer.step(),
        # so scheduler must also skip to keep the two in sync and avoid the
        # "lr_scheduler.step() before optimizer.step()" warning.
        if self.scheduler is not None and math.isfinite(grad_norm):
            self.scheduler.step()

        with torch.no_grad():
            # Policy entropy: H = -Σ π log π  (nats). Computed outside autocast
            # to avoid fp16 underflow for near-zero probabilities.
            policy_entropy = (
                -(log_policy.exp() * log_policy).sum(dim=-1).mean()
            ).float().item()

            # Value accuracy: fraction where predicted winner matches actual.
            # v_logit > 0 → predict win (outcome > 0), v_logit ≤ 0 → predict loss.
            pred_win = (v_logit.squeeze(1) > 0).float()
            target_win = (outcomes_t > 0).float()
            value_accuracy = (pred_win == target_win).float().mean().item()

        lr = self.optimizer.param_groups[0]["lr"]

        result = {
            "loss":           loss.item(),
            "policy_loss":    policy_loss.item(),
            "value_loss":     value_loss.item(),
            "policy_entropy": policy_entropy,
            "grad_norm":      grad_norm,
            "value_accuracy": value_accuracy,
            "lr":             lr,
        }
        if use_aux:
            result["opp_reply_loss"] = opp_reply_loss.item()

        interval = int(self.config.get("checkpoint_interval", 100))
        if self.step % interval == 0:
            self.save_checkpoint(result)

        return result

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    @staticmethod
    def _infer_res_blocks_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
        pattern = re.compile(r"^(?:trunk\.)?tower\.(\d+)\.")
        idxs = {
            int(match.group(1))
            for key in state_dict.keys()
            for match in [pattern.search(key)]
            if match is not None
        }
        if not idxs:
            return None
        return max(idxs) + 1

    @staticmethod
    def _infer_model_hparams(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Infer model hyperparameters directly from a checkpoint state_dict."""
        inferred: Dict[str, int] = {}

        conv_w = state_dict.get("trunk.input_conv.weight")
        if conv_w is not None and conv_w.ndim == 4:
            inferred["filters"] = int(conv_w.shape[0])
            inferred["in_channels"] = int(conv_w.shape[1])

        policy_fc_w = state_dict.get("policy_fc.weight")
        if policy_fc_w is not None and policy_fc_w.ndim == 2:
            two_spatial = int(policy_fc_w.shape[1])
            if two_spatial % 2 == 0:
                spatial = two_spatial // 2
                board_size = int(math.isqrt(spatial))
                if board_size * board_size == spatial:
                    inferred["board_size"] = board_size

        res_blocks = Trainer._infer_res_blocks_from_state_dict(state_dict)
        if res_blocks is not None:
            inferred["res_blocks"] = int(res_blocks)

        return inferred

    @staticmethod
    def _extract_model_state(ckpt: Any) -> Dict[str, torch.Tensor]:
        """Extract the model state dict from common checkpoint payload layouts."""
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unsupported checkpoint payload type: {type(ckpt)!r}")

        for key in ("model_state", "model_state_dict", "state_dict"):
            maybe_state = ckpt.get(key)
            if isinstance(maybe_state, dict):
                return maybe_state

        # Weights-only checkpoints are plain state_dict payloads.
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt

        raise ValueError("Unable to locate model state dict in checkpoint payload")

    @staticmethod
    def _resolve_model_hparams(config: Dict[str, Any], model_state: Dict[str, torch.Tensor]) -> Dict[str, int]:
        model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}

        resolved = {
            "board_size": int(model_cfg.get("board_size", config.get("board_size", 19))),
            "res_blocks": int(model_cfg.get("res_blocks", config.get("res_blocks", 12))),
            "filters": int(model_cfg.get("filters", config.get("filters", 128))),
            "in_channels": int(model_cfg.get("in_channels", config.get("in_channels", 18))),
            "se_reduction_ratio": int(model_cfg.get("se_reduction_ratio", config.get("se_reduction_ratio", 4))),
        }

        inferred = Trainer._infer_model_hparams(model_state)
        for key, inferred_value in inferred.items():
            if key in resolved:
                resolved[key] = int(inferred_value)

        # Keep top-level config aligned with inferred model dimensions.
        config["board_size"] = resolved["board_size"]
        config["res_blocks"] = resolved["res_blocks"]
        config["filters"] = resolved["filters"]
        config["in_channels"] = resolved["in_channels"]
        if isinstance(model_cfg, dict):
            model_cfg.update(resolved)
            config["model"] = model_cfg

        return resolved

    def save_checkpoint(self, loss_info: Optional[Dict[str, float]] = None) -> Path:
        """Save full checkpoint and inference-only weights.

        Returns path to the checkpoint file.
        """
        ckpt_path = self.checkpoint_dir / f"checkpoint_{self.step:08d}.pt"
        save_full_checkpoint(
            self.model, self.optimizer, self.scaler, self.scheduler,
            self.step, self.config, ckpt_path,
        )

        # Inference-only copy (weights only, no optimizer state).
        inf_path = self.checkpoint_dir / "inference_only.pt"
        save_inference_weights(self.model, inf_path)

        # Update log.
        entry: Dict[str, Any] = {"step": self.step}
        if loss_info:
            entry.update(loss_info)
        self.checkpoint_log.append(entry)
        with open(self.checkpoint_dir / "checkpoint_log.json", "w") as f:
            json.dump(self.checkpoint_log, f, indent=2)

        # Prune old checkpoints if max_checkpoints_kept is set.
        prune_checkpoints(
            self.checkpoint_dir,
            self.config.get("max_checkpoints_kept"),
        )

        log.info(
            "checkpoint_saved",
            step=self.step,
            checkpoint_path=str(ckpt_path),
        )

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
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unsupported checkpoint payload type: {type(ckpt)!r}")

        is_full_ckpt = all(k in ckpt for k in ("model_state", "config", "optimizer_state", "scaler_state", "step"))

        if "config" in ckpt and isinstance(ckpt["config"], dict):
            config = dict(ckpt["config"])
        elif fallback_config is not None:
            config = dict(fallback_config)
        else:
            raise ValueError(
                f"Checkpoint {checkpoint_path} does not include config and no fallback_config was provided."
            )

        model_state = cls._extract_model_state(ckpt)

        if config_overrides:
            config.update(config_overrides)

        model_state = normalize_model_state_dict_keys(model_state)

        model_hparams = cls._resolve_model_hparams(config, model_state)

        model = HexTacToeNet(
            board_size=model_hparams["board_size"],
            in_channels=model_hparams["in_channels"],
            res_blocks=model_hparams["res_blocks"],
            filters=model_hparams["filters"],
            se_reduction_ratio=model_hparams.get("se_reduction_ratio", 4),
        )
        model.load_state_dict(model_state, strict=False)

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
