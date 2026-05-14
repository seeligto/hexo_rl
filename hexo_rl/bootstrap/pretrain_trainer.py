"""BootstrapTrainer (§176 P39 split from pretrain.py).

Pretraining loop using the Phase 4.0 loss function. Matches
Trainer._train_on_batch() from python/training/trainer.py:
  - FP16 AMP
  - aux opponent-reply head (aux_weight from config)
  - gradient clipping to 1.0
  - policy valid masking (skip zero-policy rows)
  - label smoothing on policy targets
  - cosine LR schedule
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import structlog
import torch
import torch.optim as optim

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.monitoring.events import emit_event
from hexo_rl.training.checkpoints import (
    get_base_model,
    save_full_checkpoint,
    save_inference_weights,
)
from hexo_rl.training.losses import (
    compute_aux_loss,
    compute_chain_loss,
    compute_policy_loss,
    compute_total_loss,
    compute_value_loss,
    fp16_backward_step,
)

log = structlog.get_logger()


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

        Negative-step convention (C-004):
            ``self.step`` is initialised to ``-total_pretrain_steps`` at the call
            site (pretrain CLI, line ~1316) and counts up toward 0 across pretrain
            epochs. The ``step_budget`` exit at line 625 uses the *delta*
            ``self.step - budget_origin`` so the contract is sign-independent;
            callers must NOT pre-negate ``step_budget``. Checkpoint filenames at
            line 639 branch on sign and write ``pretrain_{abs(step):08d}.pt`` for
            the negative phase. ``tests/test_pretrain_step_accounting.py`` pins
            this invariant — any refactor that splits ``BootstrapTrainer`` must
            preserve it.
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
