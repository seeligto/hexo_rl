"""GNN fresh-init BC-prefit warm-start seam (WP-5b commit B, P6).

Wires `hexo_rl.model.gnn_net.load_representation_policy_from_bc` (fn EXISTS,
`gnn_net.py:250` — STRICT prefix-match transfer + `torch.allclose` landed-verify
already built, C7 red-team demand) into a FRESH-run launch seam, DECOUPLED from
`Trainer.load_checkpoint`'s `--checkpoint` graph-resume branch.

Why a separate seam (not `--checkpoint`): WP-4's `assert_full_gnn_checkpoint_or_
raise` (`checkpoints.py:156`) correctly REFUSES a BC-prefit-only state dict (e.g.
`checkpoints/probes/gnn_bc/gnn_bc_040000.pt` — representation+policy_head present,
NO `value_head.fc2_bins.weight`) on ANY graph-resume load — that guard exists so a
partial/mismatched checkpoint never silently half-loads. run4's INIT=BC-prefit
(`docs/designs/run4_gnn_design.md` §1) therefore CANNOT route through
`--checkpoint`; commit-B builds this seam to route around the guard on purpose
(delta doc §8 ruling: "WP-4 correctly refuses ... so WP-5 can route around it").

Trigger: a NEW `gnn_warm_start` config section (NOT `--checkpoint`, NOT
`warm_start` — that key is the E1 CNN value-head-only hook,
`hexo_rl/training/warmstart_launch.py`, a different mechanism transferring a
DIFFERENT tensor subset). Call `maybe_warmstart_gnn_from_bc` AFTER `build_net`
constructs the fresh `GnnNet`, BEFORE wrapping it in `Trainer(...)` — i.e. only
on the FRESH (no-`--checkpoint`) launch path (`orchestrator.init_trainer`'s
`else:` branch). Default-OFF (`gnn_warm_start` absent/`enabled: false`) is a
byte-identical no-op for every non-warm-start graph launch and every grid launch.

Value head: NEVER touched here (`load_representation_policy_from_bc` transfers
ONLY `representation.*`/`policy_head.*`) — stays the `build_net`-fresh-init
random dist65 head either way (E1 REVIVE: dist65 warm-starts fine from an
absent value head, `gnn_net.py` module docstring).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import structlog
import torch

from hexo_rl.model.gnn_net import load_representation_policy_from_bc
from hexo_rl.training.checkpoints import extract_model_state

# Mirrors checkpoints.py's own BC-prefit-detection marker key (the presence of
# a trained dist65 bin-logit tail, `assert_full_gnn_checkpoint_or_raise`).
_DIST65_BINS_KEY = "value_head.fc2_bins.weight"

_log = structlog.get_logger("hexo_rl.training.gnn_warmstart")


def maybe_warmstart_gnn_from_bc(
    model: Any,
    combined_config: Dict[str, Any],
    *,
    spec: Any,
    log: Optional[Any] = None,
) -> bool:
    """Seed a FRESH `GnnNet`'s representation+policy_head from a BC-prefit
    checkpoint declared at `gnn_warm_start.checkpoint`.

    Returns True iff the transfer fired (a `gnn_warmstart_loaded` event is
    emitted, and `load_representation_policy_from_bc`'s `torch.allclose`
    landed-verify fired over every transferred tensor — OQ-5). Returns False
    — a byte-identical no-op — when `gnn_warm_start` is disabled/absent.

    Args:
        model:            the freshly-`build_net`-constructed `GnnNet` (NOT
                          yet wrapped in a `Trainer`).
        combined_config:  the flattened launch config (reads `gnn_warm_start`).
        spec:             the resolved registry spec for this run (must be
                          `representation == "graph"` when warm-start is on).
        log:              optional structlog logger; falls back to the module
                          logger.

    Raises:
        ValueError: `gnn_warm_start.enabled` is True but `spec.representation`
            != "graph" (config misuse — this seam is graph-only), or
            `gnn_warm_start.checkpoint` is unset.
        FileNotFoundError: the declared checkpoint does not exist.
        RuntimeError: `load_representation_policy_from_bc` detects a key
            mismatch or a failed landed-verify (F1 guard — never silently
            drops a rep/policy tensor).
    """
    logger = log if log is not None else _log
    ws_cfg = combined_config.get("gnn_warm_start") or {}
    if not isinstance(ws_cfg, dict) or not ws_cfg.get("enabled", False):
        return False

    representation = getattr(spec, "representation", "grid")
    if representation != "graph":
        raise ValueError(
            "gnn_warm_start.enabled is true but the resolved encoding "
            f"{getattr(spec, 'name', '?')!r} has representation={representation!r} "
            "(expected 'graph') — the BC-prefit warm-start seam is graph-only. "
            "Use hexo_rl.training.warmstart_launch (warm_start.*) for the CNN "
            "value-head-only E1 warm-start instead."
        )

    ckpt_path_raw = ws_cfg.get("checkpoint")
    if not ckpt_path_raw:
        raise ValueError(
            "gnn_warm_start.enabled is true but gnn_warm_start.checkpoint is "
            "unset — cannot resolve the BC-prefit source checkpoint."
        )
    ckpt_path = Path(ckpt_path_raw)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"gnn_warm_start.checkpoint not found: {ckpt_path} — check the path "
            "(e.g. checkpoints/probes/gnn_bc/gnn_bc_040000.pt)."
        )

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    bc_state_dict = extract_model_state(raw) if isinstance(raw, dict) else raw

    # Detection-only diagnostic (delta doc P6: "detect a BC-prefit-only sd"):
    # a source that ALSO carries trained dist65 bins looks like a FULL GnnNet
    # checkpoint, not a BC-prefit-only source — harmless either way
    # (`load_representation_policy_from_bc` only ever reads rep+policy), but
    # worth a loud diagnostic since the operator likely meant `--checkpoint`
    # (a genuine resume) instead of this fresh-init seam.
    if _DIST65_BINS_KEY in bc_state_dict:
        logger.warning(
            "gnn_warmstart_source_has_value_head",
            checkpoint=str(ckpt_path),
            msg=(
                "gnn_warm_start.checkpoint carries value_head.fc2_bins.weight "
                "— this looks like a FULL GnnNet checkpoint, not a BC-prefit-"
                "only source. load_representation_policy_from_bc transfers "
                "ONLY representation.*/policy_head.*; the value head stays "
                "fresh either way (E1 REVIVE). If a full resume was intended, "
                "use --checkpoint instead of gnn_warm_start."
            ),
        )

    result = load_representation_policy_from_bc(model, bc_state_dict)

    logger.info(
        "gnn_warmstart_loaded",
        checkpoint=str(ckpt_path),
        loaded_keys=len(result["loaded_keys"]),
        verified_tensors=result["verified_tensors"],
    )
    return True


__all__ = ["maybe_warmstart_gnn_from_bc"]
