"""§S181-AUDIT Wave 2/3 — per-class target temperature on configurable slices.

V-B-A `uniform_self` routing per REAL_RUN_RECIPE §3. Soften the visit-
count CE / KL policy targets on colony positions to attenuate the
gradient pull asymmetry that Track A A4 measured (1.21× colony-vs-
extension grad L2 ratio per `audit/structural/track_a/A4_h_ce_strength.md`)
and Track B B1 pinned as the dominant channel.

Mechanism. For each in-scope row classify the position with the Track A
classifier (colony / extension / neither). Build a per-row temperature
vector and apply ``p_scaled = (p ** (1 / T)) / sum`` to the target
distribution before the policy loss. With T_colony > 1.0 the sharp
visit-count peaks on colony positions are softened — the CE gradient
pull on the dominant move is reduced, compensating the L43 mechanism
(colony positions yield lower-entropy MCTS targets).

Scope (Wave 2 default vs Wave 3 revision per L52).

Wave 2 default applied per-class temp to the selfplay slice only
(``apply_to_pretrain: false, apply_to_selfplay: true``). Wave 2 main
run (47k steps) revealed that softening tactical CE on selfplay rows
attenuates the model's own best-move signal once selfplay buffer
accumulates late-game shapes — degrading tactical sharpness. L52
banks the lesson: apply the lever only to slices where the model is
NOT learning its own play (pretrain + bot rows; both live in the
``pretrain`` batch slice per the mixing pipeline).

Wave 3 variant sets ``apply_to_pretrain: true, apply_to_selfplay: false``
— targets the static corpus + bot rows, leaves selfplay slice
unmodified so the model's own sharp policies are preserved. Default
remains backward-compatible (selfplay-only), so existing Wave 2
configs continue to work bitwise; opt into Wave 3 scope explicitly via
``apply_to_selfplay: false`` on the variant.

Cost. classify_state is a numpy loop over hex axes — per-state cost is
dominated by `_find_max_open_run` (3 axes × O(H × W) walks). The Wave 2
Stage 4 smoke measured ~16 steps/min vs B4's ~37 steps/min (~2.3× slower)
with full-rate classify on ~192 selfplay rows per 256-batch (see
`audit/structural/wave2_smoke.md` §"Throughput regression"). The
`selfplay_sample_rate` knob sub-samples the selfplay rows per batch —
sample_rate=0.20 means classify + apply to ~38 rows per batch. The
sample-rate applies only when ``apply_to_selfplay: true``; pretrain
rows are always fully classified (much smaller slice — typically 64
rows per 256-batch).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch


def _resolve_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull `per_class_target_temperature` settings from the trainer config.

    Returns None if disabled, if every temperature equals 1.0 (no-op),
    OR if both ``apply_to_pretrain`` and ``apply_to_selfplay`` are
    false (no slice in scope). Otherwise a dict
    ``{colony, extension, neither, apply_to_pretrain,
    apply_to_selfplay, selfplay_sample_rate}``.

    `apply_to_selfplay` defaults to True for backward compatibility with
    the Wave 2 lever (selfplay-only scope). Set false explicitly to
    adopt the Wave 3 L52 scope (pretrain + bot only).

    `selfplay_sample_rate` defaults to 1.0 (classify every selfplay row);
    smaller values uniformly sub-sample selfplay rows per batch to cut
    the `classify_batch` CPU loop cost. Rows outside the sample keep
    T=1.0 (no temperature applied). Only effective when
    ``apply_to_selfplay: true``.
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
    sample_rate = float(cfg.get("selfplay_sample_rate", 1.0))
    if not (0.0 < sample_rate <= 1.0):
        raise ValueError(
            f"per_class_target_temperature.selfplay_sample_rate must be in "
            f"(0.0, 1.0]; got {sample_rate}"
        )
    apply_pretrain = bool(cfg.get("apply_to_pretrain", False))
    apply_selfplay = bool(cfg.get("apply_to_selfplay", True))
    if not apply_pretrain and not apply_selfplay:
        return None  # both scopes off → no-op
    return {
        "colony": t_col,
        "extension": t_ext,
        "neither": t_nei,
        "apply_to_pretrain": apply_pretrain,
        "apply_to_selfplay": apply_selfplay,
        "selfplay_sample_rate": sample_rate,
    }


def _classify_rows(states_np: np.ndarray) -> np.ndarray:
    """Run the Track A classifier over `(n, 8, 19, 19)` states.

    Returns an array of dtype=object whose entries are
    `"colony" | "extension" | "neither"`.
    """
    # Lazy import — `scripts/` lives outside the hexo_rl package; only pulled
    # in on first use (mirrors the buffer-snapshot pattern). Classifier is
    # pure-NumPy and side-effect-free.
    from scripts.diagnosis.track_a.position_classifier import classify_batch
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
        states_t:   ``(B, n_planes, H, W)`` state tensor on device. The
                    rows consulted depend on ``apply_to_pretrain`` (gates
                    ``[0, n_pretrain)``) and ``apply_to_selfplay`` (gates
                    ``[n_pretrain, B)``). At least one must be true for
                    any scaling to occur.
        n_pretrain: Count of pretrain rows that precede the selfplay slice.
                    Trainer batch order is ``[pretrain | recent | uniform_self]``;
                    the selfplay slice is everything from ``n_pretrain`` onward.
        config:     Trainer config dict (root); reads
                    ``per_class_target_temperature.{enabled,colony_temperature,
                    extension_temperature,neither_temperature,apply_to_pretrain,
                    apply_to_selfplay,selfplay_sample_rate}``.
        device:     Target device for the temperature tensor.

    Returns:
        The temperature-scaled policy target tensor. Bit-identical to the
        input when the lever is disabled, when every temperature is 1.0,
        or when both ``apply_to_pretrain`` and ``apply_to_selfplay`` are
        false (the function short-circuits in those cases).
    """
    resolved = _resolve_config(config)
    if resolved is None:
        return policies_t

    batch_n = int(policies_t.shape[0])
    apply_pretrain = resolved["apply_to_pretrain"]
    apply_selfplay = resolved["apply_to_selfplay"]
    pretrain_end = min(int(n_pretrain), batch_n)

    # Build the row-index list per the (pretrain, selfplay) gate.  Pretrain
    # rows are always fully classified (typically 64/256 rows; cheap).
    # Selfplay rows respect ``selfplay_sample_rate`` for the per-row CPU
    # cost optimization Stage 5 smoke validated.
    row_chunks: list[torch.Tensor] = []
    if apply_pretrain and pretrain_end > 0:
        row_chunks.append(torch.arange(0, pretrain_end, device="cpu"))
    if apply_selfplay and batch_n > pretrain_end:
        sample_rate = float(resolved["selfplay_sample_rate"])
        selfplay_rows = torch.arange(pretrain_end, batch_n, device="cpu")
        if sample_rate < 1.0:
            n_sample = max(1, int(round(selfplay_rows.numel() * sample_rate)))
            # torch.randperm is deterministic under torch.manual_seed at run launch.
            perm = torch.randperm(selfplay_rows.numel(), device="cpu")[:n_sample]
            selfplay_rows = selfplay_rows[perm.sort().values]
        row_chunks.append(selfplay_rows)

    if not row_chunks:
        return policies_t  # nothing to scale

    target_rows = torch.cat(row_chunks) if len(row_chunks) > 1 else row_chunks[0]
    target_rows_device = target_rows.to(device)

    # Snapshot only the targeted rows for classification; fp16 states get
    # upcast to float32 before the .numpy() call so threshold comparisons
    # (>0.5) behave the same as they would on the dashboard / B2 path.
    slice_states = states_t.index_select(0, target_rows_device).detach()
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
    sub = out.index_select(0, target_rows_device).clamp(min=1e-9).to(dtype=torch.float32)
    scaled = sub.pow(inv_temps)
    norm = scaled.sum(dim=1, keepdim=True).clamp(min=1e-9)
    out.index_copy_(0, target_rows_device, (scaled / norm).to(dtype=out.dtype))
    return out
