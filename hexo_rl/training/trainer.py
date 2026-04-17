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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]

import structlog

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.utils.device import best_device
from hexo_rl.training.aux_decode import decode_ownership, decode_winning_line, mask_aux_rows
from hexo_rl.training.losses import (
    compute_policy_loss, compute_kl_policy_loss, compute_value_loss,
    compute_aux_loss, compute_total_loss, compute_uncertainty_loss,
    compute_chain_loss, fp16_backward_step,
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
        self.device = device or best_device()
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
        self._policy_diag_done = False  # fires once per Trainer instantiation

        # Q19: threat-head BCE positive-class weight. Allocated once per trainer
        # instance so we do not rebuild the tensor every forward pass. Device
        # placement matches `self.device` so autocast does not cross devices.
        _pos_weight_val = float(config.get("threat_pos_weight", 1.0))
        self._threat_pos_weight: Optional[torch.Tensor] = None
        if _pos_weight_val != 1.0:
            self._threat_pos_weight = torch.tensor(
                _pos_weight_val, dtype=torch.float32, device=self.device
            )

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
            t_max = config.get("scheduler_t_max", config.get("total_steps"))
            if t_max is None:
                raise ValueError(
                    "lr_schedule: cosine requires total_steps to be set. "
                    "Either pass --iterations <N> on the CLI or add "
                    "total_steps: <N> to configs/training.yaml."
                )
            total_steps = max(1, int(t_max))
            min_lr_val = config.get("eta_min", config.get("min_lr"))
            if min_lr_val is None:
                raise ValueError(
                    "lr_schedule: cosine requires eta_min to be set in configs/training.yaml."
                )
            min_lr = float(min_lr_val)
            return CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=min_lr, last_epoch=-1)

        raise ValueError(f"Unsupported lr_schedule: {schedule}")

    # ── Training step ─────────────────────────────────────────────────────────

    def train_step(
        self,
        buffer: "ReplayBuffer",
        augment: bool = True,
        recent_buffer: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Sample a batch from `buffer` and perform one gradient update.

        When ``recent_buffer`` is provided and ``recency_weight > 0`` in config,
        a fraction of the batch is drawn from the recent buffer (newest positions)
        and the remainder from the full Rust buffer.  75/25 recent/uniform is the
        default; configurable via ``recency_weight`` in training.yaml.

        Aux targets (ownership + winning_line) are sampled alongside state/policy/
        outcome from both buffers — no separate aux source.

        Args:
            buffer:        Full Rust ReplayBuffer to sample from.
            augment:       Apply 12-fold hex augmentation during Rust buffer sampling.
                           Set False in convergence tests to eliminate RNG variance.
            recent_buffer: Optional RecentBuffer for recency-weighted sampling.

        Returns:
            dict with keys "loss", "policy_loss", "value_loss",
            and optionally "opp_reply_loss".
        """
        batch_size = int(self.config["batch_size"])
        recency_weight = float(self.config.get("recency_weight", 0.0))

        if recent_buffer is not None and recent_buffer.size > 0 and recency_weight > 0.0:
            n_recent = max(1, int(round(batch_size * recency_weight)))
            n_uniform = batch_size - n_recent
            s_r, c_r, p_r, o_r, own_r, wl_r, ifs_r = recent_buffer.sample(n_recent)
            # WHY: RecentBuffer stores aux flat (n, 361); reshape to (n, 19, 19) view
            own_r = own_r.reshape(-1, 19, 19)
            wl_r  = wl_r.reshape(-1, 19, 19)
            s_u, c_u, p_u, o_u, own_u, wl_u, ifs_u = buffer.sample_batch(max(1, n_uniform), augment)
            states          = np.concatenate([s_r, s_u],     axis=0)
            chain_planes    = np.concatenate([c_r, c_u],     axis=0)
            policies        = np.concatenate([p_r, p_u],     axis=0)
            outcomes        = np.concatenate([o_r, o_u],     axis=0)
            ownership       = np.concatenate([own_r, own_u], axis=0)
            winning_line    = np.concatenate([wl_r, wl_u],   axis=0)
            is_full_search  = np.concatenate([ifs_r, ifs_u], axis=0)
        else:
            states, chain_planes, policies, outcomes, ownership, winning_line, is_full_search = \
                buffer.sample_batch(batch_size, augment)

        return self._train_on_batch(
            states, policies, outcomes,
            chain_planes=chain_planes,
            ownership_targets=ownership,
            threat_targets=winning_line,
            is_full_search=is_full_search,
            n_pretrain=0,
        )

    def train_step_from_tensors(
        self,
        states: "numpy.ndarray",
        policies: "numpy.ndarray",
        outcomes: "numpy.ndarray",
        chain_planes: Optional[Any] = None,
        ownership_targets: Optional[Any] = None,
        threat_targets: Optional[Any] = None,
        is_full_search: Optional[Any] = None,
        n_pretrain: int = 0,
    ) -> Dict[str, float]:
        """Perform one gradient update from pre-built numpy arrays.

        Used by the mixed-buffer training loop (Phase 4.0) where samples
        are drawn from pretrained + self-play buffers externally.

        Args:
            chain_planes: (B, 6, 19, 19) float16 array of Q13 chain-length planes,
                          stored separately from state since the 18-plane input
                          no longer includes chain as input channels.
            is_full_search: Optional (B,) uint8 array — 1 = full-search (apply policy
                          loss), 0 = quick-search (value/chain only). None means all
                          positions are treated as full-search (legacy behaviour).
            n_pretrain: Number of rows (from the start of the batch) that came
                        from the pretrained corpus. Used to compute per-stream
                        entropy. 0 means all rows are self-play.
        """
        return self._train_on_batch(states, policies, outcomes,
                                     chain_planes=chain_planes,
                                     ownership_targets=ownership_targets,
                                     threat_targets=threat_targets,
                                     is_full_search=is_full_search,
                                     n_pretrain=n_pretrain)

    def _train_on_batch(
        self,
        states: "numpy.ndarray",
        policies: "numpy.ndarray",
        outcomes: "numpy.ndarray",
        chain_planes: Optional[Any] = None,
        ownership_targets: Optional[Any] = None,
        threat_targets: Optional[Any] = None,
        is_full_search: Optional[Any] = None,
        n_pretrain: int = 0,
    ) -> Dict[str, float]:
        """Core training step: forward, loss, backward, optimizer step."""
        aux_weight         = float(self.config.get("aux_opp_reply_weight", 0.0))
        uncertainty_weight = float(self.config.get("uncertainty_weight", 0.0))
        ownership_weight   = float(self.config.get("ownership_weight", 0.0))
        threat_weight      = float(self.config.get("threat_weight", 0.0))
        chain_weight       = float(self.config.get("aux_chain_weight", 0.0))

        # Move to device. With FP16/autocast, keep float16 states for the mixed-
        # precision path; without it, upcast to float32 to match model weights.
        states_t = torch.from_numpy(states).to(self.device)
        if not self.fp16:
            states_t = states_t.float()

        policies_t = torch.from_numpy(policies).to(self.device)     # float32
        outcomes_t = torch.from_numpy(outcomes).to(self.device)     # float32
        # is_full_search: (B,) uint8 → bool tensor. None = all positions full-search.
        full_search_mask_t: Optional[torch.Tensor] = None
        if is_full_search is not None:
            full_search_mask_t = torch.from_numpy(
                np.asarray(is_full_search, dtype=np.uint8)
            ).to(self.device).bool()

        prune_frac = float(self.config.get("policy_prune_frac", 0.0))
        if prune_frac > 0.0:
            if not self._policy_diag_done:
                _nz_before = (policies_t > 0).float().sum(dim=-1).mean().item()
            policies_t = prune_policy_targets(policies_t, prune_frac)
            if not self._policy_diag_done:
                _nz_after = (policies_t > 0).float().sum(dim=-1).mean().item()
                log.info(
                    "policy_target_nonzero_diag",
                    step=self.step,
                    mean_nonzero_before_prune=round(_nz_before, 2),
                    mean_nonzero_after_prune=round(_nz_after, 2),
                    prune_frac=prune_frac,
                )
                self._policy_diag_done = True
        else:
            if not self._policy_diag_done:
                _nz = (policies_t > 0).float().sum(dim=-1).mean().item()
                log.info(
                    "policy_target_nonzero_diag",
                    step=self.step,
                    mean_nonzero_no_prune=round(_nz, 2),
                    prune_frac=0.0,
                )
                self._policy_diag_done = True

        self.optimizer.zero_grad()

        entropy_weight = float(self.config.get("entropy_reg_weight", 0.0))

        # Prepare ownership/threat target tensors (if provided).
        # Decode is delegated to aux_decode — ships u8 to device (4× smaller
        # H2D transfer vs fp32) then converts in-place on the GPU.
        batch_n = int(states.shape[0])
        assert 0 <= n_pretrain <= batch_n, f"n_pretrain={n_pretrain} out of [0, {batch_n}]"
        own_t: Optional[torch.Tensor] = None
        thr_t: Optional[torch.Tensor] = None
        use_ownership = ownership_weight > 0.0 and ownership_targets is not None
        use_threat    = threat_weight > 0.0    and threat_targets    is not None
        if use_ownership:
            own_t = decode_ownership(ownership_targets, self.device)   # (B, 19, 19) f32
        if use_threat:
            thr_t = decode_winning_line(threat_targets, self.device)   # (B, 19, 19) f32

        with autocast(device_type=self.device.type, dtype=torch.float16,
                      enabled=self.fp16):
            use_aux         = aux_weight > 0.0
            use_uncertainty = uncertainty_weight > 0.0
            use_chain       = chain_weight > 0.0

            fwd_result = self.model(
                states_t,
                aux=use_aux,
                uncertainty=use_uncertainty,
                ownership=use_ownership,
                threat=use_threat,
                chain=use_chain,
            )
            # Unpack in order: log_policy, value, v_logit, [opp_reply], [sigma2], [own_pred], [thr_pred], [chain_pred]
            log_policy, value, v_logit = fwd_result[0], fwd_result[1], fwd_result[2]
            _idx = 3
            opp_reply = fwd_result[_idx] if use_aux else None
            if use_aux: _idx += 1
            sigma2 = fwd_result[_idx] if use_uncertainty else None
            if use_uncertainty: _idx += 1
            own_pred = fwd_result[_idx] if use_ownership else None
            if use_ownership: _idx += 1
            thr_pred = fwd_result[_idx] if use_threat else None
            if use_threat: _idx += 1
            chain_pred = fwd_result[_idx] if use_chain else None

            policy_valid = policies_t.sum(dim=1) > 1e-6
            use_kl = bool(self.config.get("completed_q_values", False))
            if use_kl:
                policy_loss = compute_kl_policy_loss(
                    log_policy, policies_t, policy_valid, self.device,
                    full_search_mask=full_search_mask_t,
                )
            else:
                policy_loss = compute_policy_loss(
                    log_policy, policies_t, policy_valid, self.device,
                    full_search_mask=full_search_mask_t,
                )
            value_loss = compute_value_loss(v_logit, outcomes_t)

            opp_reply_loss = None
            if use_aux:
                opp_reply_loss = compute_aux_loss(
                    opp_reply, policies_t, policy_valid, self.device,
                    full_search_mask=full_search_mask_t,
                )

            # Entropy regularization: subtract entropy bonus to maximize policy entropy.
            entropy_bonus = None
            if entropy_weight > 0.0:
                p_fp32 = torch.exp(log_policy.float())
                entropy_bonus = torch.special.entr(p_fp32).sum(dim=-1).mean()

            # Uncertainty (Gaussian NLL): stop gradient from σ² influencing value head.
            unc_loss = None
            if use_uncertainty:
                value_detached = value.detach()
                unc_loss = compute_uncertainty_loss(sigma2, outcomes_t, value_detached)

            # WHY: pretrain corpus rows carry dummy aux (ones/zeros); slice them
            # out so only self-play rows contribute to the spatial heads.
            aux_skip_full_pretrain = (n_pretrain >= batch_n)
            own_loss = None
            if use_ownership and own_pred is not None and own_t is not None and not aux_skip_full_pretrain:
                own_pred_m, own_t_m = mask_aux_rows(own_pred, own_t, n_pretrain)
                own_loss = nn.functional.mse_loss(own_pred_m.squeeze(1), own_t_m)

            thr_loss = None
            if use_threat and thr_pred is not None and thr_t is not None and not aux_skip_full_pretrain:
                thr_pred_m, thr_t_m = mask_aux_rows(thr_pred, thr_t, n_pretrain)
                thr_loss = nn.functional.binary_cross_entropy_with_logits(
                    thr_pred_m.squeeze(1), thr_t_m,
                    pos_weight=self._threat_pos_weight,
                )

            # Q13-aux chain loss: target is the separately-stored chain_planes
            # sub-buffer (6 planes per position, float16 normalized by /6.0).
            # Computed on ALL batch rows; corpus chain targets are computed at
            # NPZ load (see batch_assembly.load_pretrained_buffer, §102.a) so
            # every row carries a real board-deterministic target.
            chain_loss = None
            if use_chain and chain_pred is not None and chain_planes is not None:
                chain_target = torch.from_numpy(chain_planes).to(self.device).float()
                chain_loss = compute_chain_loss(chain_pred, chain_target)

            loss = compute_total_loss(
                policy_loss, value_loss,
                opp_reply_loss, aux_weight,
                entropy_bonus, entropy_weight,
                unc_loss, uncertainty_weight,
                own_loss, ownership_weight,
                thr_loss, threat_weight,
                chain_loss, chain_weight,
            )

        # Guard: skip step on non-finite loss (safety net for numerical instability).
        if not torch.isfinite(loss):
            log.warning(
                "nan_or_inf_loss_skipped",
                step=self.step,
                policy_loss=policy_loss.item() if torch.isfinite(policy_loss) else float("nan"),
                value_loss=value_loss.item() if torch.isfinite(value_loss) else float("nan"),
            )
            self.optimizer.zero_grad()
            self.scaler.update()  # let scale decay; prevents perpetual scale inflation
            return {
                "loss": float("nan"),
                "policy_loss": float("nan"),
                "value_loss": float("nan"),
                "policy_entropy": float("nan"),
                "policy_entropy_pretrain": float("nan"),
                "policy_entropy_selfplay": float("nan"),
                "policy_target_entropy": 0.0,
                "grad_norm": float("nan"),
                "value_accuracy": float("nan"),
                "lr": self.optimizer.param_groups[0]["lr"],
            }

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
            p_fp32 = torch.exp(log_policy.float())
            policy_entropy = torch.special.entr(p_fp32).sum(dim=-1).mean().item()

            # Per-stream entropy split (pretrain rows first, selfplay rows after).
            # n_pretrain=0 means all rows are self-play (single-buffer path).
            _batch_n = p_fp32.shape[0]
            if n_pretrain > 0 and n_pretrain < _batch_n:
                policy_entropy_pretrain = torch.special.entr(p_fp32[:n_pretrain]).sum(dim=-1).mean().item()
                policy_entropy_selfplay = torch.special.entr(p_fp32[n_pretrain:]).sum(dim=-1).mean().item()
            elif n_pretrain == 0:
                policy_entropy_pretrain = float("nan")
                policy_entropy_selfplay = policy_entropy
            else:  # entire batch is pretrain
                policy_entropy_pretrain = policy_entropy
                policy_entropy_selfplay = float("nan")

            # Value accuracy: fraction where predicted winner matches actual.
            # v_logit > 0 → predict win (outcome > 0), v_logit ≤ 0 → predict loss.
            pred_win = (v_logit.squeeze(1) > 0).float()
            target_win = (outcomes_t > 0).float()
            value_accuracy = (pred_win == target_win).float().mean().item()

            # Policy target entropy: mean entropy (nats) of the MCTS policy target
            # distribution over the batch, computed only over non-zero-policy rows.
            # Mask matches the policy loss mask for consistency.
            if policy_valid.any():
                _tgt = policies_t[policy_valid].float()
                _ent = -(_tgt * _tgt.clamp_min(1e-9).log()).sum(-1).mean().item()
                policy_target_entropy = _ent if math.isfinite(_ent) else 0.0
            else:
                policy_target_entropy = 0.0

        lr = self.optimizer.param_groups[0]["lr"]

        # Fraction of batch positions that actually contributed to policy loss
        # (valid_mask & full_search_mask). 1.0 when full_search_prob=0.0
        # (feature disabled) and all rows have valid policy targets.
        if full_search_mask_t is not None:
            combined_mask = policy_valid & full_search_mask_t.bool()
            full_search_frac = combined_mask.float().mean().item()
        else:
            full_search_frac = policy_valid.float().mean().item()

        result = {
            "loss":                     loss.item(),
            "policy_loss":              policy_loss.item(),
            "value_loss":               value_loss.item(),
            "policy_entropy":           policy_entropy,
            "policy_entropy_pretrain":  policy_entropy_pretrain,
            "policy_entropy_selfplay":  policy_entropy_selfplay,
            "policy_target_entropy":    policy_target_entropy,
            "grad_norm":                grad_norm,
            "value_accuracy":           value_accuracy,
            "lr":                       lr,
            "full_search_frac":         full_search_frac,
        }
        if use_aux:
            result["opp_reply_loss"] = opp_reply_loss.item()
        if use_uncertainty:
            with torch.no_grad():
                avg_sigma = sigma2.float().sqrt().mean().item()
            result["uncertainty_loss"] = unc_loss.item()
            result["avg_sigma"] = avg_sigma
        if use_ownership and own_loss is not None:
            result["ownership_loss"] = own_loss.item()
        if use_threat and thr_loss is not None:
            result["threat_loss"] = thr_loss.item()
        if use_chain and chain_loss is not None:
            result["chain_loss"] = chain_loss.item()
        if use_ownership or use_threat:
            result["aux_loss_rows"] = max(0, batch_n - n_pretrain)

        interval = int(self.config.get("checkpoint_interval", 100))
        if self.step % interval == 0:
            self.save_checkpoint(result)

        log.info(
            "train_step",
            step=self.step,
            grad_norm=result["grad_norm"],
            total_loss=result["loss"],
            policy_loss=result["policy_loss"],
            value_loss=result["value_loss"],
            aux_loss=result.get("opp_reply_loss"),
            uncertainty_loss=result.get("uncertainty_loss"),
            ownership_loss=result.get("ownership_loss"),
            threat_loss=result.get("threat_loss"),
            chain_loss=result.get("chain_loss"),
            full_search_frac=result["full_search_frac"],
            lr=result["lr"],
            fp16_scale=self.scaler.get_scale(),
        )

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
        # Eval-step checkpoints are exempt from rotation when
        # preserve_eval_checkpoints=True (default) and eval_interval is set.
        eval_interval: int = self.config.get("eval_interval", 0)
        preserve_predicate = None
        if self.config.get("preserve_eval_checkpoints", True) and eval_interval > 0:
            preserve_predicate = lambda s: s > 0 and s % eval_interval == 0
        prune_checkpoints(
            self.checkpoint_dir,
            self.config.get("max_checkpoints_kept"),
            preserve_predicate=preserve_predicate,
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
            try:
                trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
            except ValueError as exc:
                log.warning(
                    "optimizer_state_skipped",
                    reason=str(exc),
                    msg="Model architecture changed (new parameters added) — "
                        "optimizer restarted from scratch for new params.",
                )
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
