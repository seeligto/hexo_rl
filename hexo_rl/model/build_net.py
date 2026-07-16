"""Single construction authority for HexTacToeNet / GnnNet — WP-4 (C4 subset).

Kills the SILENT-CORRUPT construction-family hole named in the ragged
contract audit (`docs/designs/gnn_ragged_contract_v1.md` Part 1, rows
11a-11c): ``orchestrator.py`` / ``lifecycle.py`` / ``anchor.py`` each built
a CNN UNCONDITIONALLY, so a ``representation=graph`` config self-played a
mis-constructed net with no error. ``build_net(spec, config, **kwargs)`` is
the CONFRES ``build_player`` single-construction-authority pattern (commit
``69442e5``) applied to model construction — one call, dispatched on
``spec.representation``, replacing all three sites.

Grid path is a byte-identical passthrough to ``HexTacToeNet(**kwargs)`` —
same args, same defaults, same behavior as every pre-WP-4 call site (verify
via ``tests/model/test_build_net.py``'s parameter-shape-equality pin).

Graph path builds ``GnnNet`` from the spec's graph geometry
(``node_feat_dim`` / ``edge_feat_dim``) plus model-config GNN hparams
(default to the probe-284k class that carries the +414 evidence,
``hexo_rl/model/gnn_net.py`` module docstring "Net-scale ruling") plus
``n_value_bins``, validating that ``value_head_type`` is compatible —
``GnnNet`` ships ONLY ``GnnDist65ValueHead`` (no scalar variant).

Any other ``spec.representation`` value, or an incompatible
``value_head_type`` on the graph path, raises ``RepresentationMismatch`` —
mirrors the Rust seam's raised-``ValueError`` convention
(``engine/src/inference_bridge.rs:492-497``,
``tests/selfplay/test_graph_collate.py:305-307`` match on that literal
prefix) so both sides of the loop fail with the same message shape.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from hexo_rl.model.gnn_net import GnnNet
from hexo_rl.training.binned_value import N_VALUE_BINS


class RepresentationMismatch(ValueError):
    """``spec.representation`` unknown, or incompatible with the requested
    model config (e.g. a non-``dist65`` ``value_head_type`` under
    ``representation="graph"``, or a graph encoding missing its schema-v4
    geometry fields).

    Message is prefixed ``"RepresentationMismatch: "`` to mirror the Rust
    seam's raised-``ValueError`` convention
    (``engine/src/inference_bridge.rs:492-497``).
    """

    def __init__(self, msg: str) -> None:
        super().__init__(f"RepresentationMismatch: {msg}")


def model_representation(model: nn.Module) -> str:
    """``"grid"`` | ``"graph"`` for a LIVE model instance (``HexTacToeNet`` vs
    ``GnnNet``) — the ``isinstance``-based counterpart to ``spec.representation``.

    S7 round-2 fix-class (F5b/F7/F8, ``S7_smoke_gate.md`` "Re-run after blocker
    fixes"): every one of those findings was a dense-only ``.in_channels`` read
    reached on a live ``GnnNet`` object (``AttributeError``, never a clean
    ``RepresentationMismatch``). ``spec.representation`` (the string on the
    resolved ``EncodingSpec``) is the right discriminant when a trustworthy spec
    is already bound to the object in hand (``build_net`` itself, ``lifecycle.py``
    ``cuda_warmup`` via ``_arch.spec``). It is NOT reliable where two
    INDEPENDENTLY-loaded model objects must be compared (e.g. ``anchor.py``
    resolving a possibly-stale/cross-lineage ``best_model.pt`` against the live
    ``inf_model`` — the whole point of that code is defending against an anchor
    that does NOT match the run's own declared config) or where a call site has
    no spec in scope at all (``LocalInferenceEngine.infer_batch`` is one
    exception: it always carries ``self.encoding_spec`` and should prefer that).
    Unwraps ``torch.compile``'s ``_orig_mod`` wrapper, same convention as every
    other call site in this codebase (``anchor.py``, ``inference_server.py``).
    """
    base = getattr(model, "_orig_mod", model)
    return "graph" if isinstance(base, GnnNet) else "grid"


def resolve_value_head_type(spec: Any, config: Optional[Mapping[str, Any]] = None) -> str:
    """Representation-aware ``value_head_type`` default — the ONE shared
    resolver for every merge point that feeds ``build_net`` (WP-4 review
    finding 1, MUST-FIX).

    A declared config value always wins. When the config omits the key (or
    explicitly nulls it), the default follows the representation:
    ``"dist65"`` for a graph spec — ``GnnNet`` ships only
    ``GnnDist65ValueHead``, and the blanket ``"scalar"`` default
    (``MODEL_HPARAM_DEFAULTS``) made every graph launch that didn't declare
    ``value_head_type: dist65`` raise ``RepresentationMismatch`` before
    reaching the net (the "omit it" advice in that error was unreachable
    from any production call site) — else the canonical grid default
    (``"scalar"``), byte-identical to the pre-WP-4 ``config.get(...,
    MODEL_HPARAM_DEFAULTS["value_head_type"])`` at every grid call site.

    Consumers: ``orchestrator.init_trainer`` (fresh run, 11a),
    ``lifecycle.build_inference_model`` (11b; ``build_eval_model`` +
    ``anchor.resolve_anchor`` inherit via ``InfModelArch.value_head_type``),
    ``anchor.resolve_anchor`` (11c, direct-call fallback), and the C7 resume
    graph branch (``trainer_ckpt_load.load_checkpoint``).
    """
    declared = (config or {}).get("value_head_type")
    if declared is not None:
        return str(declared)
    if getattr(spec, "representation", "grid") == "graph":
        return "dist65"
    from hexo_rl.training.model_defaults import MODEL_HPARAM_DEFAULTS
    return str(MODEL_HPARAM_DEFAULTS["value_head_type"])


def amp_dtype_for(representation: str, config: Optional[Mapping[str, Any]] = None) -> torch.dtype:
    """Representation-aware autocast dtype — the ONE resolver both the
    trainer's graph training step (``Trainer._train_on_graph_batch``) and the
    self-play/eval graph inference seam (``InferenceServer.__init__``, which
    ``LocalInferenceEngine``'s offline EVALFAIR/deploy-eval graph leg also
    constructs internally) consult (S7 F9 fix,
    ``reports/probes/gnn_integration/S7_smoke_gate.md`` "Re-run 3" / F9 —
    "fp16 GNN forward is non-finite on production-scale self-play graphs").

    Mechanism: ``_GINEConv``'s sum-aggregation (``agg.index_add_(0, dst,
    msg)``, ``hexo_rl/bots/strix_v1_net.py``) accumulates one ReLU'd message
    per incoming edge into each destination node. On production-scale
    self-play graphs (ply-cap-deep games, ~500-node late-game positions —
    far past the BC-corpus distribution every prior fp16 test leg lived
    under) this sum tips select batches past fp16's 65504 ceiling
    (diagnosed absmax 5.56e4 vs the 6.55e4 max on real run-3 data) ->
    ``inf`` -> ``LayerNorm`` -> NaN through the value/embedding path. Same
    ceiling explained the 26 live ``NonFiniteModelOutput`` self-play
    inference hits in the same report.

    GRAPH representation: **bfloat16, unconditionally** — not config-tunable
    here. bf16 keeps fp32's 8-bit exponent (~1e38 range) at the same 2-byte
    width, eliminating the overflow class outright; native on both the dev
    4060 (Ada, sm_89) and the vast 5080 (Blackwell, sm_120) tensor cores, no
    extra memory. A graph run's ``amp_dtype`` config key (present via the
    shared ``configs/training.yaml`` root default ``amp_dtype: "fp16"``
    unless a variant overrides it) is intentionally NOT consulted for this
    branch: this repo already paid once for the "declared vs. silently
    inherited" ambiguity class (F1 corpus-sha, F5a shared ``best_model.pt``
    — a forgotten/stale yaml default silently reintroducing the exact bug a
    fix just closed). Pinning it in code means F9 cannot come back via a
    dropped or stale variant override. ``Trainer.scaler``'s
    ``GradScaler(enabled=...)`` already keys off ``amp_dtype == float16``
    (``trainer.py`` ``scaler_enabled``), so bf16 here correctly disables/
    bypasses the scaler too — GradScaler is unnecessary (and, per torch
    convention, should not be used) for bf16's full exponent range.

    GRID (dense) representation: delegates to the existing ``amp_dtype``
    config knob (default ``"fp16"``) — BYTE-IDENTICAL to pre-F9 behaviour.
    The live run3 CNN lineage rides this branch untouched.
    """
    if representation == "graph":
        return torch.bfloat16
    raw = str((config or {}).get("amp_dtype", "fp16")).lower()
    if raw in ("fp16", "float16", "half"):
        return torch.float16
    if raw in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(
        f"amp_dtype must be 'fp16' or 'bf16', got {raw!r}. "
        "Set in configs/training.yaml or a variant override."
    )


# GNN hparam config keys, with defaults mirroring GnnNet's own ctor defaults
# (the probe-284k class that carries the +414 [+320,+560] BT-Elo evidence,
# `gnn_net.py` module docstring "Net-scale ruling"). A launch config does
# not need to set any of these — they exist so a deliberate net-scale
# experiment (run4-v2, the NET-CAPACITY plateau falsifier) can override
# without touching this dispatch.
GNN_CONFIG_DEFAULTS: dict[str, int] = {
    "gnn_hidden": 128,
    "gnn_num_layers": 4,
    "gnn_policy_hidden": 128,
    "gnn_value_hidden": 32,
}


def build_net(spec: Any, config: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> nn.Module:
    """Construct the model for ``spec.representation`` — the ONE authority.

    Args:
        spec: resolved registry spec (``hexo_rl.encoding.lookup(name)`` /
            ``hexo_rl.encoding.resolve_from_config(config)``), carrying
            ``.representation`` (``"grid"`` | ``"graph"``) and, for a graph
            encoding, ``.node_feat_dim`` / ``.edge_feat_dim``. ``None`` (or
            any object with no ``.representation`` attribute) defaults to
            ``"grid"`` — preserves byte-identical behavior for callers/tests
            that do not thread a spec.
        config: the run config dict (``combined_config`` / ``trainer.config``)
            — consulted ONLY for the ``gnn_*`` hparam overrides
            (``GNN_CONFIG_DEFAULTS``) on the graph path. May be omitted
            (``None``) for a grid build (unused) or when GNN defaults
            suffice.
        **kwargs: the EXACT keyword args the call site would have passed to
            ``HexTacToeNet(...)`` directly (``board_size``, ``res_blocks``,
            ``filters``, ``in_channels``, ``input_channels``,
            ``se_reduction_ratio``, ``value_head_type``, ``n_value_bins``,
            ...). The grid path forwards these byte-identically. The graph
            path reads only ``value_head_type`` / ``n_value_bins`` off them
            — the rest are grid-only geometry and are ignored (a graph net
            has no ``board_size`` / ``filters`` / ``res_blocks`` meaning,
            contract §"Load-bearing audit finding").

    Returns:
        ``HexTacToeNet`` (``representation="grid"``) or ``GnnNet``
        (``representation="graph"``).

    Raises:
        RepresentationMismatch: ``spec.representation`` is neither
            ``"grid"`` nor ``"graph"``; or (graph only) ``value_head_type``
            is explicitly declared and isn't ``"dist65"``; or (graph only)
            the spec is missing ``node_feat_dim`` / ``edge_feat_dim``.
    """
    representation = getattr(spec, "representation", "grid")
    if representation == "grid":
        from hexo_rl.model.network import HexTacToeNet

        return HexTacToeNet(**kwargs)
    if representation == "graph":
        return _build_gnn_net(spec, config or {}, **kwargs)
    raise RepresentationMismatch(
        f"build_net: spec.representation={representation!r} for encoding "
        f"{getattr(spec, 'name', '?')!r} — expected 'grid' or 'graph'."
    )


def _build_gnn_net(spec: Any, config: Mapping[str, Any], **kwargs: Any) -> GnnNet:
    value_head_type = kwargs.get("value_head_type")
    if value_head_type is not None and value_head_type != "dist65":
        raise RepresentationMismatch(
            f"build_net: representation='graph' (encoding {getattr(spec, 'name', '?')!r}) "
            f"only ships GnnDist65ValueHead; got value_head_type={value_head_type!r}. "
            "Declare value_head_type: dist65 (or omit it) for a graph encoding."
        )
    node_feat_dim = getattr(spec, "node_feat_dim", None)
    edge_feat_dim = getattr(spec, "edge_feat_dim", None)
    if node_feat_dim is None or edge_feat_dim is None:
        raise RepresentationMismatch(
            f"build_net: encoding {getattr(spec, 'name', '?')!r} declares "
            "representation='graph' but is missing node_feat_dim/edge_feat_dim "
            "(schema-v4 graph fields, engine/src/encoding/registry.toml)."
        )
    n_value_bins = kwargs.get("n_value_bins")
    if n_value_bins is None:
        n_value_bins = N_VALUE_BINS
    return GnnNet(
        in_dim=int(node_feat_dim),
        hidden=int(config.get("gnn_hidden", GNN_CONFIG_DEFAULTS["gnn_hidden"])),
        num_layers=int(config.get("gnn_num_layers", GNN_CONFIG_DEFAULTS["gnn_num_layers"])),
        edge_dim=int(edge_feat_dim),
        policy_hidden=int(config.get("gnn_policy_hidden", GNN_CONFIG_DEFAULTS["gnn_policy_hidden"])),
        value_hidden=int(config.get("gnn_value_hidden", GNN_CONFIG_DEFAULTS["gnn_value_hidden"])),
        n_value_bins=int(n_value_bins),
    )
