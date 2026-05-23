"""§S181-AUDIT Wave 2 — per-class target temperature on the selfplay slice.

V-B-A `uniform_self` routing per REAL_RUN_RECIPE §3 + Wave 2 dispatcher
Stage 3. Soften the visit-count CE / KL policy targets on colony
positions inside the selfplay slice to attenuate the gradient pull
asymmetry that Track A A4 measured (1.21× colony-vs-extension grad L2
ratio per `audit/structural/track_a/A4_h_ce_strength.md`) and Track B
B1 pinned as the dominant channel (selfplay-family carries ~91 % of
total gradient pull at near-miss V-B-A under source-group reading; see
`audit/structural/track_b/B_verdict_synthesis.md`).

Mechanism. For each selfplay-slice row classify the position with the
Track A classifier (colony / extension / neither). Build a per-row
temperature vector and apply ``p_scaled = (p ** (1 / T)) / sum`` to the
target distribution before the policy loss. With T_colony > 1.0 the
sharp visit-count peaks on colony positions are softened — the CE
gradient pull on the dominant move is reduced, compensating the L43
mechanism (colony positions yield lower-entropy MCTS targets).

Pretrain slice is left at T = 1.0 unless `apply_to_pretrain: true` is
set; the pretrain corpus is a static distribution that doesn't
participate in the colony attractor feedback loop (see B1: pretrain
slice mean share 0.092, smallest contributor). Bot corpus rows live
in the pretrain slice and are similarly untouched by default.

Cost. classify_state is a numpy loop over hex axes — per-state cost is
dominated by `_find_max_open_run` (3 axes × O(H × W) walks). On a batch
of 256 with selfplay slice ~192 rows the classify call adds a few ms
per train step on top of a ~100 ms forward+backward. Smoke gate
(Stage 4) measures the real overhead; if intolerable the next iteration
can sub-sample the selfplay slice (`subsample_rate < 1.0`).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch


def _resolve_config(config: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Pull `per_class_target_temperature` settings from the trainer config.

    Returns None if disabled OR if every temperature equals 1.0 (no-op).
    Otherwise a dict ``{colony, extension, neither, apply_to_pretrain}``.
    """
    cfg = config.get("per_class_target_temperature") if isinstance(
        config.get("per_class_target_temperature"), dict
    ) else {}
    if not cfg.get("enabled", False):
        return None
    t_col = float(cfg.get("colony_temperature", 1.0))
    t_ext = float(cfg.get("extension_temperature", 1.0))
    t_nei = float(cfg.get("neither_temperature", 1.0))
    if t_col == 1.0 and t_ext == 1.0 and t_nei == 1.0:
        return None
    for label, val in (("colony", t_col), ("extension", t_ext), ("neither", t_nei)):
        if val <= 0.0:
            raise ValueError(
                f"per_class_target_temperature.{label}_temperature must be > 0; got {val}"
            )
    return {
        "colony": t_col,
        "extension": t_ext,
        "neither": t_nei,
        "apply_to_pretrain": bool(cfg.get("apply_to_pretrain", False)),
    }


def _classify_rows(states_np: np.ndarray) -> np.ndarray:
    """Run the Track A classifier over `(n, 8, 19, 19)` states.

    Returns an array of dtype=object whose entries are
    `"colony" | "extension" | "neither"`.
    """
    # Lazy import — `scripts/` lives outside the hexo_rl package; only pulled
    # in on first use (mirrors the buffer-snapshot pattern). Classifier is
    # pure-NumPy and side-effect-free.
    from scripts.structural_diagnosis.track_a.position_classifier import classify_batch
    return classify_batch(states_np)


def apply_per_class_temperature(
    policies_t: torch.Tensor,
    states_t: torch.Tensor,
    n_pretrain: int,
    config: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Apply per-class temperature scaling to the policy target tensor.

    Args:
        policies_t: ``(B, A)`` policy target tensor on device. Modified
                    out-of-place; the original tensor is returned unchanged
                    when the lever is disabled / a no-op.
        states_t:   ``(B, n_planes, H, W)`` state tensor on device. Only the
                    selfplay slice (rows ``n_pretrain..B``) is consulted —
                    the rest is left untouched unless ``apply_to_pretrain``
                    is set on the config.
        n_pretrain: Count of pretrain rows that precede the selfplay slice.
                    Trainer batch order is ``[pretrain | recent | uniform_self]``;
                    the selfplay slice is everything from ``n_pretrain`` onward.
        config:     Trainer config dict (root); reads
                    ``per_class_target_temperature.{enabled,colony_temperature,
                    extension_temperature,neither_temperature,apply_to_pretrain}``.
        device:     Target device for the temperature tensor.

    Returns:
        The temperature-scaled policy target tensor. Bit-identical to the
        input when the lever is disabled or when every temperature
        is 1.0 (the function short-circuits in those cases).
    """
    resolved = _resolve_config(config)
    if resolved is None:
        return policies_t

    batch_n = int(policies_t.shape[0])
    apply_pretrain = resolved["apply_to_pretrain"]
    row_start = 0 if apply_pretrain else min(int(n_pretrain), batch_n)
    if row_start >= batch_n:
        return policies_t  # nothing to scale

    # Snapshot the slice we need to classify; conversion goes via CPU since
    # the classifier is pure numpy. fp16 states get upcast to float32 before
    # the .numpy() call so threshold comparisons (>0.5) behave the same as
    # they would on the dashboard / B2 path.
    slice_states = states_t[row_start:].detach()
    if slice_states.dtype != torch.float32:
        slice_states = slice_states.float()
    states_np = slice_states.cpu().numpy()
    classes = _classify_rows(states_np)

    temps = np.full(states_np.shape[0], resolved["neither"], dtype=np.float32)
    temps[classes == "colony"] = resolved["colony"]
    temps[classes == "extension"] = resolved["extension"]
    if np.all(temps == 1.0):
        return policies_t  # every classified row landed on a T=1.0 class

    inv_temps = torch.from_numpy(1.0 / temps).to(device).unsqueeze(1)
    # Out-of-place clone preserves the original tensor's autograd graph
    # (policies are constructed from torch.from_numpy → no graph, but the
    # caller still expects an unmodified input).
    out = policies_t.clone()
    sub = out[row_start:].clamp(min=1e-9).to(dtype=torch.float32)
    scaled = sub.pow(inv_temps)
    norm = scaled.sum(dim=1, keepdim=True).clamp(min=1e-9)
    out[row_start:] = (scaled / norm).to(dtype=out.dtype)
    return out
