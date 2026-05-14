"""Fine-tune freeze helper (§176 P39 split from pretrain.py).

Contains:
  - _apply_finetune_freeze — §171 A4 P2-reopen C freeze pattern. Sets
    requires_grad on trunk.input_conv / trunk.input_gn and per-block
    trunk.tower entries. Used by pretrain CLI when --freeze-trunk-entry
    or --unfreeze-blocks is passed.
"""

from __future__ import annotations

from typing import Dict, Optional


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
