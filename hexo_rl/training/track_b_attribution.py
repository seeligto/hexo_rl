"""§S181-AUDIT Wave 1 Track B — per-source gradient-norm attribution.

Tooling for the instrumented short training run. Given a forward pass'
intermediate tensors (log_policy, v_logit, etc.) and the batch's source
slicing (pretrain / recent / uniform_self), compute per-source gradient
L2 norm on each target parameter group (trunk / value_head / policy_head).

Goal (Track B / L49). Track A localized that NO single corpus-level
source is dominant (multi-source attractor). Track B closes the per-step
pull accounting gap by measuring which source actually carries the
gradient signal under live §S180b mixing weights. Pre-registered
verdict V-B-{A,B,C,D,E} consumes the per-source norms collected here.

Mechanism. For each source slice s and each parameter group g, we
compute the loss `L_s = mean(per-row policy + value loss within slice s)`
and `g_s = ∂L_s / ∂params_g`. The L2 norm `||g_s||₂` is the per-source
pull magnitude on group g. With `retain_graph=True`, three
`torch.autograd.grad` calls (one per source) reuse the original forward
graph — no extra forward passes.

INSPECTION-ONLY at the trainer level: the attribution does NOT step the
optimizer or interfere with the main backward — it is a side-channel
read that is consumed BEFORE the main backward. Skip if any slice is
empty.
"""
from __future__ import annotations

from typing import Optional

import torch


def _l2_norm(grads: tuple) -> float:
    """Sum-of-squares over a tuple of param grads → scalar L2 norm."""
    total = 0.0
    for g in grads:
        if g is None:
            continue
        total += float(g.detach().float().pow(2).sum().item())
    return float(total ** 0.5)


def compute_per_source_grad_attribution(
    slice_losses: dict[str, Optional[torch.Tensor]],
    target_groups: dict[str, list],
) -> dict[str, float]:
    """For each (source, group) combination, compute L2 grad norm.

    Args:
        slice_losses: {source_name → scalar loss tensor (or None)}. A
            None / empty-slice loss is skipped (returns NaN). Each loss
            must be a scalar tensor with grad attached to the model.
        target_groups: {group_name → list of nn.Parameter}.

    Returns:
        Flat dict {f"{group}_{source}": l2_norm_float}. Missing slices
        produce NaN entries so the downstream event has a stable schema.
    """
    out: dict[str, float] = {}
    for source_name, loss in slice_losses.items():
        for group_name, params in target_groups.items():
            key = f"{group_name}_{source_name}"
            if loss is None or not params:
                out[key] = float("nan")
                continue
            grads = torch.autograd.grad(
                loss, params,
                retain_graph=True,
                allow_unused=True,
                create_graph=False,
            )
            out[key] = _l2_norm(grads)
    return out


def select_param_groups(model: torch.nn.Module) -> dict[str, list]:
    """Pick trunk / value / policy parameters from a HexTacToeNet."""
    trunk = list(model.trunk.parameters()) if hasattr(model, "trunk") else []

    # Value head: ``value_fc1`` + ``value_fc2`` are the canonical pair
    # post-§131 P3. ``value_var`` (uncertainty) is excluded — it gates
    # on `use_uncertainty` and its grads have their own attribution dim.
    value: list = []
    for name in ("value_fc1", "value_fc2"):
        mod = getattr(model, name, None)
        if mod is not None:
            value.extend(mod.parameters())

    # Policy head: KataGo head if present, else legacy `policy_conv`+`policy_fc`.
    policy: list = []
    ph = getattr(model, "policy_head", None)
    if ph is not None:
        policy.extend(ph.parameters())
    else:
        for name in ("policy_conv", "policy_fc"):
            mod = getattr(model, name, None)
            if mod is not None:
                policy.extend(mod.parameters())

    return {"trunk": trunk, "value": value, "policy": policy}


def build_slice_losses(
    log_policy: torch.Tensor,
    v_logit: torch.Tensor,
    policies_t: torch.Tensor,
    outcomes_t: torch.Tensor,
    policy_valid: torch.Tensor,
    device: torch.device,
    n_pretrain: int,
    n_recent: int,
    full_search_mask: Optional[torch.Tensor] = None,
    use_kl: bool = False,
) -> dict[str, Optional[torch.Tensor]]:
    """Construct {source → scalar loss tensor} for the three batch slices.

    Slice ordering (matches trainer.py docstring): [corpus | recent |
    uniform_self]. Returns None for any empty slice.
    """
    from hexo_rl.training.losses import (
        compute_kl_policy_loss, compute_policy_loss, compute_value_loss,
    )

    batch_n = int(log_policy.shape[0])
    slice_spans = {
        "pretrain": (0, n_pretrain),
        "recent": (n_pretrain, n_pretrain + n_recent),
        "uniform_self": (n_pretrain + n_recent, batch_n),
    }

    out: dict[str, Optional[torch.Tensor]] = {}
    for name, (a, b) in slice_spans.items():
        if b <= a:
            out[name] = None
            continue
        lp = log_policy[a:b]
        p = policies_t[a:b]
        pv = policy_valid[a:b]
        vl = v_logit[a:b]
        oc = outcomes_t[a:b]
        fsm = None if full_search_mask is None else full_search_mask[a:b]
        if use_kl:
            sl_pol = compute_kl_policy_loss(lp, p, pv, device, full_search_mask=fsm)
        else:
            sl_pol = compute_policy_loss(lp, p, pv, device, full_search_mask=fsm)
        sl_val = compute_value_loss(vl, oc)
        out[name] = sl_pol + sl_val
    return out
