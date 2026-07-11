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
import time
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
# Perf probes use structlog directly (log.info) so they persist to JSONL even
# when --no-dashboard is set. emit_event is dashboard-renderer fan-out only.
from hexo_rl.training.aux_decode import decode_ownership, decode_winning_line, mask_aux_rows
from hexo_rl.training.losses import (
    compute_policy_loss, compute_kl_policy_loss, compute_value_loss,
    compute_aux_loss, compute_ply_index_loss, compute_total_loss, compute_uncertainty_loss,
    compute_chain_loss, fp16_backward_step,
)
from hexo_rl.training.checkpoints import (
    save_full_checkpoint, save_inference_weights, prune_checkpoints,
    get_base_model,
    extract_model_state as _extract_model_state_impl,
    infer_model_hparams as _infer_model_hparams_impl,
    infer_res_blocks_from_state_dict as _infer_res_blocks_impl,
    load_state_dict_strict as _load_state_dict_strict_impl,
)
# §172 A4.3: re-export legacy spec resolver for backward-compat imports
# (e.g. tests/test_training_registry_plumbing.py). Cold-path checkpoint
# loader lives in trainer_ckpt_load (§176 P7); see Trainer.load_checkpoint.
from hexo_rl.encoding import (
    resolve_from_checkpoint as registry_resolve_ckpt,  # noqa: F401  (re-export for §172 A5)
)
from engine import ReplayBuffer


# §176 P14 re-export of `legacy_spec_for_registry_name` retired in cycle 3
# Wave 8 Batch C (FF.10, 2026-05-17) alongside the underlying compat shim.
# Callers needing wire-format scalars read them off
# `hexo_rl.encoding.lookup(name)` directly (returns the registry record).

log = structlog.get_logger()


def _resolve_amp_dtype(config: Dict[str, Any]) -> torch.dtype:
    """Map config ``amp_dtype`` string to a torch.dtype. Default fp16."""
    raw = str(config.get("amp_dtype", "fp16")).lower()
    if raw in ("fp16", "float16", "half"):
        return torch.float16
    if raw in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(
        f"amp_dtype must be 'fp16' or 'bf16', got {raw!r}. "
        "Set in configs/training.yaml or a variant override."
    )


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


_POLICY_TARGET_METRIC_KEYS = (
    "policy_target_entropy_fullsearch", "policy_target_entropy_fastsearch",
    "policy_target_kl_uniform_fullsearch", "policy_target_kl_uniform_fastsearch",
    # B5 (STAT_AUDIT delete-4): frac_fullsearch_in_batch dropped — identical
    # numerator AND denominator to full_search_frac (trainer.py train_step).
    "n_rows_policy_loss", "n_rows_total",
)


def compute_policy_target_metrics(
    target_policy: torch.Tensor,
    policy_valid: torch.Tensor,
    full_search_mask: Optional[torch.Tensor],
) -> Dict[str, float]:
    """§101 D-Gumbel / D-Zeroloss: Shannon H + KL(tgt||uniform), split by is_full_search.

    NaN for empty subsets (first-class signal). ``full_search_mask=None`` ⇒
    all-full (legacy). Packs 7 scalars into one cross-device transfer.
    """
    with torch.no_grad():
        p = target_policy.float()
        H = torch.special.entr(p).sum(dim=-1)                         # (B,), nats
        log_N = math.log(float(max(p.size(-1), 1)))
        KL_uniform = log_N - H

        pvb = policy_valid.bool()
        if full_search_mask is not None:
            fs = full_search_mask.bool()
            mask_full, mask_fast = pvb & fs, pvb & (~fs)
        else:
            mask_full, mask_fast = pvb, torch.zeros_like(pvb)

        mf, mfa, pvf = (m.to(H.dtype) for m in (mask_full, mask_fast, pvb))
        (H_full_s, H_fast_s, KL_full_s, KL_fast_s, n_full_f, n_fast_f, n_valid_f
         ) = torch.stack([
            (H * mf).sum(), (H * mfa).sum(),
            (KL_uniform * mf).sum(), (KL_uniform * mfa).sum(),
            mf.sum(), mfa.sum(), pvf.sum(),
        ]).cpu().tolist()

        n_full, n_fast, n_valid = (int(round(x)) for x in (n_full_f, n_fast_f, n_valid_f))
        batch_n = int(pvb.numel())
        _div = lambda num, cnt: (float("nan") if cnt == 0 else num / cnt)
        return {
            "policy_target_entropy_fullsearch":    _div(H_full_s, n_full),
            "policy_target_entropy_fastsearch":    _div(H_fast_s, n_fast),
            "policy_target_kl_uniform_fullsearch": _div(KL_full_s, n_full),
            "policy_target_kl_uniform_fastsearch": _div(KL_fast_s, n_fast),
            "n_rows_policy_loss":                  n_full,
            "n_rows_total":                        n_valid,
        }


# NaN/0 defaults for the 7 metric keys — used in NaN-loss guard + gate-off.
_ZERO_POLICY_TARGET_METRICS: Dict[str, float] = {
    "policy_target_entropy_fullsearch":    float("nan"),
    "policy_target_entropy_fastsearch":    float("nan"),
    "policy_target_kl_uniform_fullsearch": float("nan"),
    "policy_target_kl_uniform_fastsearch": float("nan"),
    "n_rows_policy_loss": 0, "n_rows_total": 0,
}


def compute_value_metrics_per_source(
    v_logit: torch.Tensor,
    outcomes: torch.Tensor,
    value_mask: Optional[torch.Tensor],
    n_pretrain: int,
    decided_eps: float = 1e-3,
) -> Dict[str, float]:
    """§D-VALCEIL Q3 — per-source + masked value diagnostics (logging-only).

    Resolves the §D-VALPROBE `value_accuracy` ANOMALY (batch 0.66 vs
    component-predicted ~0.725): the headline ``value_accuracy`` is UNMASKED —
    ``target_win = (z > 0)`` scores draw rows (z = draw_value, default −0.5)
    and ply-capped rows (z = ply_cap_value; ``value_target_valid = 0``) as
    "loss" targets, so non-decided rows deflate it relative to decided-row
    winner-calling. This helper decomposes it WITHOUT touching any existing
    key (curve continuity) or any tensor used for backward.

    Source semantics — batch row order is ``[corpus(+bot) | recent |
    uniform_self]``: "corpus" = rows ``[0, n_pretrain)``. §178 bot-corpus rows
    are folded into ``n_pretrain`` upstream (step_coordinator passes
    ``n_pretrain = n_pre + n_bot``) and are NOT separable here — the corpus
    bucket includes them. "selfplay" = rows ``[n_pretrain, B)`` (recent +
    uniform self-play).

    Masking semantics:
      - supervised = ``value_mask`` (DRAW-MASK Phase 6 ``value_target_valid``;
        ``None`` ⇒ all rows, matching ``compute_value_loss``). Currently 0
        only on ply-capped self-play rows; organic draws stay supervised.
      - decided = ``|z| > 1 − decided_eps``: decisive games store exactly
        ±1.0; ``draw_value`` and ``ply_cap_value`` lie strictly inside (−1, 1)
        by config, so draws AND capped rows are excluded.
      - ``value_accuracy_masked`` = accuracy over supervised AND decided rows.
      - per-source accuracies stay UNMASKED so count-weighted recombination
        reproduces the batch ``value_accuracy`` exactly.
      - per-source BCE = mean per-row BCE-with-logits over SUPERVISED rows in
        the slice (exact ``compute_value_loss`` semantics); weighted by the
        ``*_supervised`` counts it recombines to ``value_loss``.

    Empty slices report NaN (first-class signal, §101 convention) + count 0.
    Cost: one per-row BCE + boolean masks on tensors already in memory.
    """
    with torch.no_grad():
        # v_logit is (B, 1) (or already (B,)) — flatten to (B,). reshape(-1)
        # avoids a bare positional plane slice (INV: no_positional_plane_slice).
        logit = v_logit.reshape(-1).float()
        z = outcomes.reshape(-1).float()
        batch_n = int(z.numel())
        n_pre = max(0, min(int(n_pretrain), batch_n))

        correct = ((logit > 0).float() == (z > 0).float()).float()      # (B,)
        per_row_bce = nn.functional.binary_cross_entropy_with_logits(
            logit, (z + 1.0) / 2.0, reduction="none"
        )                                                               # (B,)
        supervised = (
            value_mask.reshape(-1).bool()
            if value_mask is not None
            else torch.ones(batch_n, dtype=torch.bool, device=z.device)
        )
        decided = z.abs() > (1.0 - decided_eps)
        masked_rows = supervised & decided

        is_corpus = torch.zeros(batch_n, dtype=torch.bool, device=z.device)
        is_corpus[:n_pre] = True
        is_selfplay = ~is_corpus

        def _mean_over(values: torch.Tensor, mask: torch.Tensor) -> tuple[float, int]:
            n = int(mask.sum().item())
            if n == 0:
                return float("nan"), 0
            return values[mask].mean().item(), n

        acc_masked, n_masked = _mean_over(correct, masked_rows)
        acc_corpus, _ = _mean_over(correct, is_corpus)
        acc_selfplay, _ = _mean_over(correct, is_selfplay)
        bce_corpus, n_corpus_sup = _mean_over(per_row_bce, is_corpus & supervised)
        bce_selfplay, n_selfplay_sup = _mean_over(per_row_bce, is_selfplay & supervised)

        return {
            "value_accuracy_masked":          acc_masked,
            "value_accuracy_corpus":          acc_corpus,
            "value_accuracy_selfplay":        acc_selfplay,
            "value_bce_corpus":               bce_corpus,
            "value_bce_selfplay":             bce_selfplay,
            "value_rows_corpus":              n_pre,
            "value_rows_selfplay":            batch_n - n_pre,
            "value_rows_masked":              n_masked,
            "value_rows_corpus_supervised":   n_corpus_sup,
            "value_rows_selfplay_supervised": n_selfplay_sup,
        }


def build_param_groups(model: nn.Module, weight_decay: float) -> list:
    """Split params for AdamW weight decay: 1D params + biases skip decay.

    Standard AdamW pattern (nanoGPT / KataGo precedent):
    - 2D+ weights (conv/linear kernels) receive weight decay.
    - 1D params (BN/LN scale & bias, explicit bias terms) get weight_decay=0.
    This prevents decay from shrinking normalisation gains toward zero and
    avoids decaying bias terms which should be free to shift.

    §S181 PR-B: no-decay group added to counter loss-of-plasticity at late
    training when colony attractor manifests.
    """
    decay = []
    no_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


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

        # Autocast dtype selection — fp16 vs bf16. bf16 has wider dynamic
        # range (no GradScaler needed on Ampere+/Ada) at the cost of less
        # mantissa precision. Default fp16 preserves legacy behaviour.
        self.amp_dtype = _resolve_amp_dtype(config)
        # GradScaler is only meaningful for fp16; bf16 has sufficient range.
        scaler_enabled = self.fp16 and self.amp_dtype == torch.float16

        self.config = config
        # CONFRES F1(A) back-prop: the set of keys the resume's F1 defer-to-baked preserved (empty on
        # a fresh run / weights-only resume). load_checkpoint overwrites this with the real set so the
        # orchestrator can back-propagate the preserved baked values into the loop-read config.
        self.f1_deferred_keys: "frozenset[str]" = frozenset()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.AdamW(
            build_param_groups(self.model, float(config["weight_decay"])),
            lr=float(config["lr"]),
        )
        self.scheduler = self._build_scheduler(config)

        # §S181-AUDIT Wave 2 — EMA of model weights for self-play / eval /
        # bot-promotion dispatch. Trainer keeps stepping the raw model; the
        # EMA module is read-only and updated every `ema_update_every`
        # optimizer steps after the optimizer step lands. Default OFF
        # preserves pre-Wave-2 behaviour (see hexo_rl/training/ema.py).
        from hexo_rl.training.ema import build_ema_model, resolve_ema_config
        _ema_enabled, _ema_decay, _ema_update_every = resolve_ema_config(config)
        self.ema_update_every: int = _ema_update_every
        if _ema_enabled:
            _base = getattr(self.model, "_orig_mod", self.model)
            self.ema_model = build_ema_model(_base, decay=_ema_decay)
            log.info("ema_enabled", decay=_ema_decay, update_every=_ema_update_every)
        else:
            self.ema_model = None

        # GradScaler for FP16 training; no-op on CPU or when bf16 selected.
        self.scaler = GradScaler(device=self.device.type, enabled=scaler_enabled)
        self._scaler_enabled = scaler_enabled

        # torch.compile re-enabled §116 (2026-04-23): reduce-overhead GO on
        # Py3.14.2 + PT2.11.0. Mode is config-driven via torch_compile_mode
        # (default | reduce-overhead | max-autotune). Old §32 blockers
        # (Py3.14 TLS crash + 27 GB Triton JIT spike) resolved in PT2.11.
        if config.get("torch_compile", False) and self.device.type == "cuda":
            _compile_mode = str(config.get("torch_compile_mode", "default"))
            try:
                self.model = torch.compile(
                    self.model, mode=_compile_mode, fullgraph=False
                )
                log.info("torch_compile_enabled", mode=_compile_mode)
            except Exception as exc:
                log.warning("torch_compile_failed", mode=_compile_mode, error=str(exc))

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

        # §101: gate D-Gumbel / D-Zeroloss instrumentation (default on).
        _mon = config.get("monitoring") if isinstance(config.get("monitoring"), dict) else {}
        self._log_policy_target_metrics = bool(
            _mon.get("log_policy_target_metrics", config.get("log_policy_target_metrics", True))
        )

        # Perf-investigation probes (docs/perf/instrumentation_notes.md).
        # Cached booleans so hot-path has no dict lookups when disabled.
        _diag = config.get("diagnostics") if isinstance(config.get("diagnostics"), dict) else {}
        self._perf_timing = bool(_diag.get("perf_timing", False))
        self._perf_sync_cuda = bool(_diag.get("perf_sync_cuda", False))
        self._vram_probe_interval = int(_diag.get("vram_probe_interval", 100))

        # §S181-AUDIT Track B — per-source gradient-norm attribution (~3× cost,
        # off by default). Set `track_b_grad_attribution: true` to enable.
        self._track_b_grad_attribution = bool(
            config.get("track_b_grad_attribution", False)
        )
        # §S181-AUDIT Track B — selfplay buffer position-class snapshot at
        # checkpoint cadence. ~5k positions per snapshot, ~few sec per fire.
        self._track_b_buffer_snapshot = bool(
            config.get("track_b_buffer_snapshot", False)
        )
        self._track_b_buffer_snapshot_n = int(
            config.get("track_b_buffer_snapshot_n", 5000)
        )
        if self._perf_sync_cuda and torch.cuda.is_available():
            log.warning(
                "perf_sync_cuda_enabled_serialising_stream",
                impact="expect_30_50_pct_throughput_drop",
                remedy="unset_diagnostics.perf_sync_cuda_in_production_config",
            )
        if self._perf_timing:
            log.info(
                "perf_timing_enabled",
                sync_cuda=self._perf_sync_cuda,
                vram_probe_interval=self._vram_probe_interval,
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

        _perf = self._perf_timing
        _t_sample_start = time.perf_counter() if _perf else 0.0

        n_recent = 0
        position_indices: Optional[np.ndarray] = None
        value_target_valid: Optional[np.ndarray] = None
        if recent_buffer is not None and recent_buffer.size > 0 and recency_weight > 0.0:
            n_recent = max(1, int(round(batch_size * recency_weight)))
            n_uniform = batch_size - n_recent
            s_r, c_r, p_r, o_r, own_r, wl_r, ifs_r, vv_r = recent_buffer.sample(n_recent)
            # WHY: RecentBuffer stores aux flat (n, aux_stride); reshape to (n, board_size, board_size)
            _bs = int(math.isqrt(own_r.shape[1]))
            own_r = own_r.reshape(-1, _bs, _bs)
            wl_r  = wl_r.reshape(-1, _bs, _bs)
            # §S181-AUDIT Wave 4 4B-impl-3 — recent_buffer lacks ply index; fill zeros.
            pos_r = np.zeros(len(s_r), dtype=np.uint16)
            s_u, c_u, p_u, o_u, own_u, wl_u, ifs_u, pos_u, vv_u = buffer.sample_batch_with_pos(max(1, n_uniform), augment)
            states          = np.concatenate([s_r, s_u],     axis=0)
            chain_planes    = np.concatenate([c_r, c_u],     axis=0)
            policies        = np.concatenate([p_r, p_u],     axis=0)
            outcomes        = np.concatenate([o_r, o_u],     axis=0)
            ownership       = np.concatenate([own_r, own_u], axis=0)
            winning_line    = np.concatenate([wl_r, wl_u],   axis=0)
            is_full_search  = np.concatenate([ifs_r, ifs_u], axis=0)
            position_indices = np.concatenate([pos_r, pos_u], axis=0)
            # DRAW-MASK (Phase 6): per-row value-supervision mask.
            value_target_valid = np.concatenate([vv_r, vv_u], axis=0)
        else:
            states, chain_planes, policies, outcomes, ownership, winning_line, is_full_search, position_indices, value_target_valid = \
                buffer.sample_batch_with_pos(batch_size, augment)

        if _perf:
            log.info(
                "buffer_sample_timing",
                step=self.step + 1,  # pre-increment; +1 matches post-increment step in train_step_timing
                sample_us=(time.perf_counter() - _t_sample_start) * 1e6,
                batch_n=int(states.shape[0]),
                used_recent=recent_buffer is not None and recent_buffer.size > 0 and recency_weight > 0.0,
            )

        return self._train_on_batch(
            states, policies, outcomes,
            chain_planes=chain_planes,
            ownership_targets=ownership,
            threat_targets=winning_line,
            is_full_search=is_full_search,
            n_pretrain=0,
            n_recent=n_recent,
            position_indices=position_indices,
            value_target_valid=value_target_valid,
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
        n_recent: int = 0,
        position_indices: Optional[Any] = None,
        value_target_valid: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Perform one gradient update from pre-built numpy arrays.

        Used by the mixed-buffer training loop (Phase 4.0) where samples
        are drawn from pretrained + self-play buffers externally.

        Args:
            chain_planes: (B, 6, T, T) float16 array of Q13 chain-length planes,
                          stored separately from state since the 18-plane input
                          no longer includes chain as input channels. T is the
                          encoding trunk_size (19 for v6, 25 for v6w25/v8).
            is_full_search: Optional (B,) uint8 array — 1 = full-search (apply policy
                          loss), 0 = quick-search (value/chain only). None means all
                          positions are treated as full-search (legacy behaviour).
            n_pretrain: Number of rows (from the start of the batch) that came from
                        the pretrained corpus (first n_pretrain rows). Used for
                        per-stream entropy. 0 means all rows are self-play.
            n_recent: Number of rows from the recent_buffer, immediately following the
                      corpus rows. Batch order: [corpus | recent | uniform_self].
                      Used to compute the 3-way entropy split. 0 = no recent slice.
        """
        return self._train_on_batch(states, policies, outcomes,
                                     chain_planes=chain_planes,
                                     ownership_targets=ownership_targets,
                                     threat_targets=threat_targets,
                                     is_full_search=is_full_search,
                                     n_pretrain=n_pretrain,
                                     n_recent=n_recent,
                                     position_indices=position_indices,
                                     value_target_valid=value_target_valid)

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
        n_recent: int = 0,
        position_indices: Optional[Any] = None,
        value_target_valid: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Core training step: forward, loss, backward, optimizer step."""
        _perf = self._perf_timing
        _sync = self._perf_sync_cuda
        _t0 = time.perf_counter() if _perf else 0.0

        aux_weight         = float(self.config.get("aux_opp_reply_weight", 0.0))
        uncertainty_weight = float(self.config.get("uncertainty_weight", 0.0))
        ownership_weight   = float(self.config.get("ownership_weight", 0.0))
        threat_weight      = float(self.config.get("threat_weight", 0.0))
        chain_weight       = float(self.config.get("aux_chain_weight", 0.0))
        ply_index_weight   = float(self.config.get("ply_index_weight", 0.0))

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
        # DRAW-MASK (Phase 6): value_target_valid (B,) uint8 → bool tensor. 1 =
        # supervise the value head, 0 = ply-capped row → mask out of value loss.
        # None = all rows supervised (backward-compat; matches prior behaviour).
        value_mask_t: Optional[torch.Tensor] = None
        if value_target_valid is not None:
            value_mask_t = torch.from_numpy(
                np.asarray(value_target_valid, dtype=np.uint8)
            ).to(self.device).bool()

        # §S181-AUDIT Wave 2/3 — per-class target temperature on configurable
        # slices (V-B-A `uniform_self` lever per REAL_RUN_RECIPE §3). Softens
        # visit-count CE targets on colony positions to attenuate the gradient-
        # pull asymmetry Track B B1 pinned. Wave 3 default scope per L52 is
        # pretrain+bot rows only (`apply_to_selfplay: false`), preserving the
        # model's own sharp policies on selfplay rows. Applied BEFORE
        # policy_prune_frac so pruning sees the softened distribution.
        # Default OFF preserves pre-Wave-2 behaviour.
        if self.config.get("per_class_target_temperature", {}).get("enabled", False):
            from hexo_rl.training.per_class_target_temperature import (
                apply_per_class_temperature,
            )
            policies_t = apply_per_class_temperature(
                policies_t, states_t, n_pretrain=n_pretrain,
                config=self.config, device=self.device,
            )

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
            own_t = decode_ownership(ownership_targets, self.device)   # (B, T, T) f32
        if use_threat:
            thr_t = decode_winning_line(threat_targets, self.device)   # (B, T, T) f32

        if _perf:
            if _sync and self.device.type == "cuda":
                torch.cuda.synchronize()
            _t_h2d = time.perf_counter()

        with autocast(device_type=self.device.type, dtype=self.amp_dtype,
                      enabled=self.fp16):
            use_aux         = aux_weight > 0.0
            use_uncertainty = uncertainty_weight > 0.0
            use_chain       = chain_weight > 0.0
            use_ply_index   = ply_index_weight > 0.0 and position_indices is not None

            fwd_result = self.model(
                states_t,
                aux=use_aux,
                uncertainty=use_uncertainty,
                ownership=use_ownership,
                threat=use_threat,
                chain=use_chain,
                ply_index=use_ply_index,
            )
            # Unpack in order: log_policy, value, v_logit, [opp_reply], [sigma2], [own_pred], [thr_pred], [chain_pred], [ply_pred]
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
            if use_chain: _idx += 1
            ply_pred = fwd_result[_idx] if use_ply_index else None

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
            value_loss = compute_value_loss(v_logit, outcomes_t, value_mask=value_mask_t)

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

            ply_index_loss = None
            if use_ply_index and ply_pred is not None and position_indices is not None:
                pos_idx_t = torch.from_numpy(np.asarray(position_indices)).to(self.device)
                ply_index_loss = compute_ply_index_loss(ply_pred, pos_idx_t)

            loss = compute_total_loss(
                policy_loss, value_loss,
                opp_reply_loss, aux_weight,
                entropy_bonus, entropy_weight,
                unc_loss, uncertainty_weight,
                own_loss, ownership_weight,
                thr_loss, threat_weight,
                chain_loss, chain_weight,
                ply_index_loss, ply_index_weight,
            )

        if _perf:
            if _sync and self.device.type == "cuda":
                torch.cuda.synchronize()
            _t_forward_loss = time.perf_counter()

        # Guard: skip step on non-finite loss (safety net for numerical instability).
        if not torch.isfinite(loss):
            log.warning(
                "nan_or_inf_loss_skipped",
                step=self.step,
                policy_loss=policy_loss.item() if torch.isfinite(policy_loss) else float("nan"),
                value_loss=value_loss.item() if torch.isfinite(value_loss) else float("nan"),
            )
            self.optimizer.zero_grad()
            # Manually decay the scale. We cannot call `scaler.update()` here
            # because no `scaler.step()` / `scaler.unscale_()` fired this
            # iteration; the internal state machine asserts
            # "No inf checks were recorded prior to update." `update(new_scale=)`
            # bypasses that path (see torch/amp/grad_scaler.py::update). Skip
            # entirely when the scale tensor has not been lazy-initialised yet
            # (NaN on the very first step — nothing to decay).
            if (
                self.fp16
                and self.scaler.is_enabled()
                and getattr(self.scaler, "_scale", None) is not None
            ):
                decayed = float(self.scaler.get_scale() * self.scaler.get_backoff_factor())
                self.scaler.update(new_scale=decayed)
            return {
                "loss": float("nan"),
                "policy_loss": float("nan"),
                "value_loss": float("nan"),
                "policy_entropy": float("nan"),
                "policy_entropy_pretrain": float("nan"),
                "policy_entropy_selfplay": float("nan"),
                "selfplay_model_entropy_batch": float("nan"),  # alias; drop 2026-05-28
                "policy_entropy_recent": float("nan"),
                "policy_entropy_uniform_selfplay": float("nan"),
                "policy_target_entropy": 0.0,
                "grad_norm": float("nan"),
                "value_accuracy": float("nan"),
                "lr": self.optimizer.param_groups[0]["lr"],
                **_ZERO_POLICY_TARGET_METRICS,
            }

        # §S181-AUDIT Track B — per-source gradient-norm attribution.
        # Fires BEFORE the main backward; uses retain_graph so backward
        # still consumes the graph. Skips on any failure (never blocks
        # training). ~3× per-step cost (3 extra slice backwards).
        if self._track_b_grad_attribution:
            try:
                from hexo_rl.training.track_b_attribution import (
                    build_slice_losses,
                    compute_per_source_grad_attribution,
                    select_param_groups,
                )
                slice_losses = build_slice_losses(
                    log_policy=log_policy, v_logit=v_logit,
                    policies_t=policies_t, outcomes_t=outcomes_t,
                    policy_valid=policy_valid, device=self.device,
                    n_pretrain=n_pretrain, n_recent=n_recent,
                    full_search_mask=full_search_mask_t,
                    use_kl=use_kl,
                )
                target_groups = select_param_groups(self.model)
                attribution = compute_per_source_grad_attribution(
                    slice_losses, target_groups,
                )
                log.info(
                    "per_source_grad_norm",
                    step=self.step + 1,
                    n_pretrain=n_pretrain, n_recent=n_recent,
                    n_uniform_self=batch_n - n_pretrain - n_recent,
                    **attribution,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "track_b_attribution_failed",
                    step=self.step, error=str(exc),
                )

        max_grad_norm = float(self.config.get("grad_clip", 1.0))
        grad_norm = fp16_backward_step(
            loss, self.optimizer, self.scaler, self.model, self._scaler_enabled,
            max_grad_norm=max_grad_norm,
        )

        if _perf:
            if _sync and self.device.type == "cuda":
                torch.cuda.synchronize()
            _t_backward_opt = time.perf_counter()

        self.step += 1

        # Step scheduler AFTER optimizer.step() (inside fp16_backward_step)
        # and after self.step increment so counters stay in sync.
        # Guard: when GradScaler detects inf/nan grads it skips optimizer.step(),
        # so scheduler must also skip to keep the two in sync and avoid the
        # "lr_scheduler.step() before optimizer.step()" warning.
        if self.scheduler is not None and math.isfinite(grad_norm):
            self.scheduler.step()

        # §S181-AUDIT Wave 2 — EMA update after optimizer step. Skip when
        # grad_norm is non-finite (matches scheduler skip above so EMA does
        # not absorb a step that the optimizer itself rejected).
        if (self.ema_model is not None
                and math.isfinite(grad_norm)
                and self.step % self.ema_update_every == 0):
            _base = getattr(self.model, "_orig_mod", self.model)
            self.ema_model.update_parameters(_base)

        entropies = self._compute_entropies(
            log_policy=log_policy, n_pretrain=n_pretrain, n_recent=n_recent,
        )

        with torch.no_grad():
            # Value accuracy: fraction where predicted winner matches actual.
            # v_logit > 0 → predict win (outcome > 0), v_logit ≤ 0 → predict loss.
            pred_win = (v_logit.squeeze(1) > 0).float()
            target_win = (outcomes_t > 0).float()
            value_accuracy = (pred_win == target_win).float().mean().item()

            # §D-VALCEIL Q3 — per-source + masked value decomposition
            # (logging-only; reads tensors already in memory, zero effect on
            # training math). Source/masking semantics documented on the helper.
            value_metrics_per_source = compute_value_metrics_per_source(
                v_logit, outcomes_t, value_mask_t, n_pretrain,
            )

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

        # §101: policy-target quality split by is_full_search (~<0.2% step cost).
        policy_target_metrics = (
            compute_policy_target_metrics(policies_t, policy_valid, full_search_mask_t)
            if self._log_policy_target_metrics else dict(_ZERO_POLICY_TARGET_METRICS)
        )

        result = {
            "loss":                     loss.item(),
            "policy_loss":              policy_loss.item(),
            "value_loss":               value_loss.item(),
            "policy_entropy":                 entropies["policy_entropy"],
            "policy_entropy_pretrain":        entropies["policy_entropy_pretrain"],
            "policy_entropy_selfplay":        entropies["policy_entropy_selfplay"],
            "selfplay_model_entropy_batch":   entropies["policy_entropy_selfplay"],  # alias; drop 2026-05-28
            "policy_entropy_recent":          entropies["policy_entropy_recent"],
            "policy_entropy_uniform_selfplay": entropies["policy_entropy_uniform_sp"],
            "policy_target_entropy":    policy_target_entropy,
            "grad_norm":                grad_norm,
            "value_accuracy":           value_accuracy,
            "lr":                       lr,
            "full_search_frac":         full_search_frac,
        }
        result.update(policy_target_metrics)
        self._append_conditional_metrics(
            result,
            use_aux=use_aux, opp_reply_loss=opp_reply_loss,
            use_uncertainty=use_uncertainty, sigma2=sigma2, unc_loss=unc_loss,
            use_ownership=use_ownership, own_loss=own_loss,
            use_threat=use_threat, thr_loss=thr_loss,
            use_chain=use_chain, chain_loss=chain_loss,
            use_ply_index=use_ply_index, ply_index_loss=ply_index_loss,
            batch_n=batch_n, n_pretrain=n_pretrain,
        )

        # §D-VALPROBE Phase 3 — explicit value-axis decomposition (logging-only;
        # derived from already-computed scalars, zero effect on training math).
        # value_loss has always been the main BCE term; the uncertainty/aux keys
        # below are the WEIGHTED contributions as they enter the total, so
        # value_loss_composite == main + uncertainty + aux exactly.
        # CAVEAT (review, §D-VALPROBE Phase 6): opp_reply aux is a POLICY-shaped
        # head and can dominate the composite — read value_loss_composite as
        # "total minus pure-policy accounting", NOT as value-head signal; the
        # value-head signal is value_loss (+ value_loss_uncertainty).
        # B5 (STAT_AUDIT delete-4): redundant value_loss_main alias dropped —
        # it was an exact copy of value_loss (zero marginal signal).
        result["value_loss_uncertainty"] = (
            uncertainty_weight * result["uncertainty_loss"] if use_uncertainty else 0.0
        )
        result["value_loss_aux"] = (
            aux_weight * result["opp_reply_loss"] if use_aux else 0.0
        )
        result["value_loss_composite"] = (
            result["value_loss"]
            + result["value_loss_uncertainty"]
            + result["value_loss_aux"]
        )

        # §D-VALCEIL Q3 — per-source masked value keys (logging-only; NEW keys
        # only, no existing key renamed/altered — curve continuity). Invariant:
        # count-weighted value_accuracy_{corpus,selfplay} recombine to
        # value_accuracy; *_supervised-weighted value_bce_{corpus,selfplay}
        # recombine to value_loss. "corpus" includes §178 bot rows (folded into
        # n_pretrain upstream, not separable here).
        result.update(value_metrics_per_source)

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
            value_loss_uncertainty=result["value_loss_uncertainty"],
            value_loss_aux=result["value_loss_aux"],
            value_loss_composite=result["value_loss_composite"],
            # §D-VALCEIL Q3 — per-source masked value keys (additive).
            value_accuracy=result["value_accuracy"],
            value_accuracy_masked=result["value_accuracy_masked"],
            value_accuracy_corpus=result["value_accuracy_corpus"],
            value_accuracy_selfplay=result["value_accuracy_selfplay"],
            value_bce_corpus=result["value_bce_corpus"],
            value_bce_selfplay=result["value_bce_selfplay"],
            value_rows_corpus=result["value_rows_corpus"],
            value_rows_selfplay=result["value_rows_selfplay"],
            value_rows_masked=result["value_rows_masked"],
            value_rows_corpus_supervised=result["value_rows_corpus_supervised"],
            value_rows_selfplay_supervised=result["value_rows_selfplay_supervised"],
            aux_loss=result.get("opp_reply_loss"),
            uncertainty_loss=result.get("uncertainty_loss"),
            ownership_loss=result.get("ownership_loss"),
            threat_loss=result.get("threat_loss"),
            chain_loss=result.get("chain_loss"),
            full_search_frac=result["full_search_frac"],
            lr=result["lr"],
            fp16_scale=self.scaler.get_scale(),
        )

        # Perf probes (diagnostic mode only) — via structlog so they persist
        # to JSONL independent of dashboard renderers.
        if _perf:
            self._perf_probe_emit(
                t0=_t0, t_h2d=_t_h2d,
                t_forward_loss=_t_forward_loss, t_backward_opt=_t_backward_opt,
                sync=_sync, batch_n=batch_n,
            )

        return result

    # ── _train_on_batch helpers (§176 P8) ─────────────────────────────────────

    def _compute_entropies(
        self, *,
        log_policy: torch.Tensor,
        n_pretrain: int,
        n_recent: int,
    ) -> Dict[str, float]:
        """Per-stream policy entropy split (corpus / recent / uniform_self).

        Returns dict with 5 keys: policy_entropy, policy_entropy_pretrain,
        policy_entropy_selfplay, policy_entropy_recent, policy_entropy_uniform_sp.
        Each is a Python float (.item() called inside).

        Block: torch.no_grad() — fp32 entropies computed outside autocast to
        avoid fp16 underflow for near-zero probabilities.
        """
        with torch.no_grad():
            # Policy entropy: H = -Σ π log π  (nats). Computed outside autocast
            # to avoid fp16 underflow for near-zero probabilities.
            p_fp32 = torch.exp(log_policy.float())
            policy_entropy = torch.special.entr(p_fp32).sum(dim=-1).mean().item()

            # Per-stream entropy split.  Batch order: [corpus | recent | uniform_self].
            # n_pretrain=0 means all rows are self-play (single-buffer path).
            # n_recent=0 means recent_buffer was absent or empty this step.
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

            # 3-way split: corpus / recent / uniform_self
            _sp_start  = n_pretrain
            _rec_end   = n_pretrain + n_recent
            if n_recent > 0 and _rec_end <= _batch_n and _sp_start < _batch_n:
                policy_entropy_recent = (
                    torch.special.entr(p_fp32[_sp_start:_rec_end]).sum(dim=-1).mean().item()
                    if _rec_end > _sp_start else float("nan")
                )
                policy_entropy_uniform_sp = (
                    torch.special.entr(p_fp32[_rec_end:]).sum(dim=-1).mean().item()
                    if _rec_end < _batch_n else float("nan")
                )
            else:
                policy_entropy_recent    = float("nan")
                policy_entropy_uniform_sp = policy_entropy_selfplay

        return {
            "policy_entropy":            policy_entropy,
            "policy_entropy_pretrain":   policy_entropy_pretrain,
            "policy_entropy_selfplay":   policy_entropy_selfplay,
            "policy_entropy_recent":     policy_entropy_recent,
            "policy_entropy_uniform_sp": policy_entropy_uniform_sp,
        }

    def _append_conditional_metrics(
        self, result: Dict[str, float], *,
        use_aux: bool, opp_reply_loss: Optional[torch.Tensor],
        use_uncertainty: bool, sigma2: Optional[torch.Tensor],
        unc_loss: Optional[torch.Tensor],
        use_ownership: bool, own_loss: Optional[torch.Tensor],
        use_threat: bool, thr_loss: Optional[torch.Tensor],
        use_chain: bool, chain_loss: Optional[torch.Tensor],
        use_ply_index: bool = False,
        ply_index_loss: Optional[torch.Tensor] = None,
        batch_n: int = 0, n_pretrain: int = 0,
    ) -> None:
        """Append conditional aux/uncertainty/ownership/threat/chain loss keys
        to the result dict. Mutates ``result`` in place; returns None.

        Field-order preserved from the inline block: opp_reply_loss → unc/sigma →
        own → thr → chain. (B5: aux_loss_rows dropped — == value_rows_selfplay,
        identical batch_n − n_pretrain.)
        """
        if use_aux:
            result["opp_reply_loss"] = opp_reply_loss.item()
        if use_uncertainty:
            with torch.no_grad():
                avg_sigma = sigma2.float().sqrt().mean().item()
            result["uncertainty_loss"] = unc_loss.item()
            result["avg_sigma"] = avg_sigma   # sqrt(predicted squared err) since Wave 4 4B-impl-5
        if use_ownership and own_loss is not None:
            result["ownership_loss"] = own_loss.item()
        if use_threat and thr_loss is not None:
            result["threat_loss"] = thr_loss.item()
        if use_chain and chain_loss is not None:
            result["chain_loss"] = chain_loss.item()
        if use_ply_index and ply_index_loss is not None:
            result["ply_index_loss"] = ply_index_loss.item()
        # B5 (STAT_AUDIT delete-4): aux_loss_rows dropped — exact duplicate of
        # value_rows_selfplay (both == batch_n − n_pretrain). batch_n /
        # n_pretrain params retained for signature stability (unused now).

    def _perf_probe_emit(
        self, *,
        t0: float,
        t_h2d: float,
        t_forward_loss: float,
        t_backward_opt: float,
        sync: bool,
        batch_n: int,
    ) -> None:
        """Emit train_step_timing event + optional VRAM probe.

        Caller-guarded by ``self._perf_timing``; this helper unconditionally
        emits the train_step_timing event and additionally fires a vram_probe
        when on CUDA at the configured interval.
        """
        log.info(
            "train_step_timing",
            step=self.step,
            h2d_us=(t_h2d - t0) * 1e6,
            fwd_loss_us=(t_forward_loss - t_h2d) * 1e6,
            bwd_opt_us=(t_backward_opt - t_forward_loss) * 1e6,
            total_us=(t_backward_opt - t0) * 1e6,
            sync_cuda=sync,
            batch_n=batch_n,
        )
        if self._vram_probe_interval > 0 \
                and self.device.type == "cuda" \
                and self.step % self._vram_probe_interval == 0:
            stats = torch.cuda.memory_stats(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            allocated = torch.cuda.memory_allocated(self.device)
            peak = torch.cuda.max_memory_allocated(self.device)
            log.info(
                "vram_probe",
                step=self.step,
                vram_peak_gb=peak / 1e9,
                vram_allocated_gb=allocated / 1e9,
                vram_reserved_gb=reserved / 1e9,
                vram_frag_gb=max(0, reserved - allocated) / 1e9,
                num_ooms=int(stats.get("num_ooms", 0)),
            )
            torch.cuda.reset_peak_memory_stats(self.device)

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    # §176 P79: helpers lifted to hexo_rl.training.checkpoints so
    # viewer/model_loader.py and probe scripts share one implementation.
    # Staticmethod surface preserved for back-compat callers
    # (our_model_bot, scripts/probe_*, tests/test_analyze_api parity check).
    _infer_res_blocks_from_state_dict = staticmethod(_infer_res_blocks_impl)
    _infer_model_hparams = staticmethod(_infer_model_hparams_impl)
    # §176 P47: lifted to checkpoints.load_state_dict_strict; staticmethod
    # surface preserved for back-compat callers (trainer_ckpt_load.py via
    # `cls._load_state_dict_strict`).
    _load_state_dict_strict = staticmethod(_load_state_dict_strict_impl)

    # §176 P79: see comment near _infer_model_hparams above.
    _extract_model_state = staticmethod(_extract_model_state_impl)

    def inference_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return the state_dict consumed by self-play / eval / promotion.

        §S181-AUDIT Wave 2. When EMA is enabled, downstream weight-sync
        sites (build_inference_model, eval kickoff, best-model promotion,
        anchor fresh-init) must read EMA weights so the training-loop
        decoupling is honoured. Centralising the choice here keeps the
        dispatch-routing single-sourced and prevents drift between the
        sites listed above.
        """
        if self.ema_model is not None:
            return self.ema_model.state_dict()
        base = getattr(self.model, "_orig_mod", self.model)
        return base.state_dict()

    def save_checkpoint(self, loss_info: Optional[Dict[str, float]] = None) -> Path:
        """Save full checkpoint and inference-only weights.

        Returns path to the checkpoint file.
        """
        ckpt_path = self.checkpoint_dir / f"checkpoint_{self.step:08d}.pt"
        # §172 A5.1: stamp encoding metadata at every save site. Resolve via
        # the registry so renames (v6w25 → ...) propagate uniformly.
        try:
            from hexo_rl.encoding.resolvers import resolve_from_config
            _enc_name = resolve_from_config(self.config).name
        except Exception as exc:
            # Resolver failure is itself a bug — surface it loudly. Save still
            # proceeds without metadata so the run isn't lost; load path will
            # fall back to shape inference.
            log.error("checkpoint_encoding_resolve_failed", error=str(exc))
            _enc_name = None
        save_full_checkpoint(
            self.model, self.optimizer, self.scaler, self.scheduler,
            self.step, self.config, ckpt_path,
            encoding_name=_enc_name,
            train_config_path=self.config.get("_config_path"),
            corpus_sha256=self.config.get("corpus_sha256"),
            model_variant=self.config.get("model_variant"),
        )

        # Inference-only copy (weights only, no optimizer state).
        inf_path = self.checkpoint_dir / "inference_only.pt"
        save_inference_weights(self.model, inf_path)

        # §S181-AUDIT Wave 2 — EMA sidecars. When EMA is on, the inference
        # / eval / promotion paths read EMA weights via Trainer.inference_state_dict;
        # the EMA sidecars let post-hoc viewer + analysis scripts compare
        # raw vs EMA trajectories. Naming: `checkpoint_<step>_ema.pt` next to
        # the raw ckpt; `inference_only_ema.pt` next to the inference copy.
        if self.ema_model is not None:
            ema_ckpt_path = self.checkpoint_dir / f"checkpoint_{self.step:08d}_ema.pt"
            save_inference_weights(self.ema_model.module, ema_ckpt_path)
            ema_inf_path = self.checkpoint_dir / "inference_only_ema.pt"
            save_inference_weights(self.ema_model.module, ema_inf_path)

        # Update log.
        entry: Dict[str, Any] = {"step": self.step}
        if loss_info:
            entry.update(loss_info)
        self.checkpoint_log.append(entry)
        with open(self.checkpoint_dir / "checkpoint_log.json", "w") as f:
            json.dump(self.checkpoint_log, f, indent=2)

        # Prune old checkpoints if max_checkpoints_kept is set.
        # Preserved steps: eval_interval boundaries + anchor_every_steps boundaries.
        # keep_all=True disables pruning entirely (for debug runs).
        eval_interval: int = self.config.get("eval_interval", 0)
        anchor_every_steps: Optional[int] = self.config.get("anchor_every_steps") or None
        preserve_eval = self.config.get("preserve_eval_checkpoints", True)

        def _preserve(s: int) -> bool:
            if s <= 0:
                return False
            if preserve_eval and eval_interval > 0 and s % eval_interval == 0:
                return True
            if anchor_every_steps and s % anchor_every_steps == 0:
                return True
            return False

        prune_checkpoints(
            self.checkpoint_dir,
            self.config.get("max_checkpoints_kept"),
            preserve_predicate=_preserve,
            keep_all=bool(self.config.get("keep_all", False)),
        )

        log.info(
            "checkpoint_saved",
            step=self.step,
            checkpoint_path=str(ckpt_path),
        )

        # §S181 PR-A — value-spread colony-capture canary. Fires on every
        # checkpoint save (not in the inner loop). Fire-and-forget: never
        # raises, restores model train mode internally.
        #
        # §S181-AUDIT Wave 2 — when EMA is on, fire the canary on the EMA
        # weights so the dashboard signal tracks what self-play / eval
        # actually see. EmaModel is not an nn.Module, so we materialize
        # the EMA state into the trainer's own model in-place around the
        # fire and restore the training weights after. The canary runs
        # under torch.no_grad on the same thread that owns the optimizer,
        # so the swap window is single-threaded by construction.
        try:
            from hexo_rl.monitoring.value_spread_canary import fire_canary
            base = getattr(self.model, "_orig_mod", self.model)
            saved_state: Optional[Dict[str, torch.Tensor]] = None
            if self.ema_model is not None:
                saved_state = {
                    k: v.detach().clone() for k, v in base.state_dict().items()
                }
                base.load_state_dict(self.ema_model.state_dict())
            try:
                fire_canary(self.model, self.step, self.device,
                            encoding=self.config.get("encoding"))
            finally:
                if saved_state is not None:
                    base.load_state_dict(saved_state)
        except Exception as exc:  # noqa: BLE001 — canary must never break a save
            log.warning("value_spread_canary_dispatch_failed",
                        step=self.step, error=str(exc))

        return ckpt_path

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str | Path,
        checkpoint_dir: Optional[str | Path] = None,
        device: Optional[torch.device] = None,
        fallback_config: Optional[Dict[str, Any]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        declared_keys: Optional["frozenset | set"] = None,
    ) -> "Trainer":
        """Restore a Trainer from a checkpoint file.

        Thin delegate around :func:`hexo_rl.training.trainer_ckpt_load.load_checkpoint`
        (§176 P7 extraction). The cold-path body + encoding-reconciliation
        helpers live in ``trainer_ckpt_load`` so this module stays focused
        on the hot training step.

        ``declared_keys`` threads the CONFRES F1(A) operator-declaration set through to the
        override apply (base-inherited overrides defer to the checkpoint-baked value; declared keys
        still win). ``None`` preserves the pre-6b verbatim-update behaviour.
        """
        from hexo_rl.training import trainer_ckpt_load
        return trainer_ckpt_load.load_checkpoint(
            cls,
            checkpoint_path,
            checkpoint_dir=checkpoint_dir,
            device=device,
            fallback_config=fallback_config,
            config_overrides=config_overrides,
            declared_keys=declared_keys,
        )
