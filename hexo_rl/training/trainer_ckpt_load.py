"""Cold-path checkpoint loader extracted from Trainer (§176 P7).

`Trainer.load_checkpoint` and its four encoding-reconciliation helpers
(``_detect_encoding_from_state_dict``, ``_resolve_checkpoint_encoding``,
``_propagate_encoding_into_config``, ``_resolve_model_hparams``) ran in a
~480 LOC block inside ``trainer.py`` that only fires on ckpt load. They
were lifted here to shrink ``trainer.py`` without touching behaviour:
the helpers are module-level functions copied byte-identical, and
``Trainer.load_checkpoint`` is a thin delegate that calls
``load_checkpoint(cls, ...)`` from this module.

Hot-path helpers (`_extract_model_state`, `_infer_model_hparams`,
`_load_state_dict_strict`) stay on ``Trainer`` — they are re-used by
``hexo_rl/viewer/model_loader.py`` and ``hexo_rl/bootstrap/bots/our_model_bot.py``.

See docs/refactor-template.md (pure-move rule) and the §176 P7 sprint log.
"""

from __future__ import annotations

import math  # noqa: F401  (kept for parity with trainer module imports)
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING

import torch
import torch.nn as nn  # noqa: F401  (kept for parity)

import structlog

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.checkpoints import (
    normalize_model_state_dict_keys,
)
from hexo_rl.training.model_defaults import MODEL_HPARAM_DEFAULTS
from hexo_rl.utils.constants import BUFFER_CHANNELS
from hexo_rl.encoding import (
    EncodingSpec as RegistrySpec,
    lookup as registry_lookup,
    resolve_from_config as registry_resolve_config,
)
from hexo_rl.encoding.compat import (
    WireFormatSpec,
    WIRE_FORMAT_SPECS,
    legacy_spec_for_registry_name as _legacy_spec_for_registry_name,
)

if TYPE_CHECKING:
    from hexo_rl.training.trainer import Trainer

log = structlog.get_logger()


def _detect_encoding_from_state_dict(
    state_dict: Dict[str, torch.Tensor], ckpt_label: str,
) -> Optional[WireFormatSpec]:
    """Infer a wire-format spec from a bare model state_dict.

    Used for weights-only checkpoints (e.g. bootstrap_model_v6w25.pt)
    that carry no `config` payload. Mirrors the dispatch in
    hexo_rl.eval.checkpoint_loader.detect_encoding_label but returns
    a ``WireFormatSpec`` (§176 P3 — replaces the legacy
    ``hexo_rl.utils.encoding.EncodingSpec`` NamedTuple).

    Returns ``None`` when the state-dict shape does not match a
    canonical (v6 / v6w25 / v8) encoding — the caller should then
    fall through to legacy `_resolve_model_hparams` inference. This
    keeps backward compat for non-canonical test fixtures (e.g. a
    9×9 trainer round-trip).

    ``ckpt_label`` is a short identifier (e.g. the checkpoint filename)
    used for the v6 vs v6w25 filename disambiguator.
    """
    inp_w = state_dict.get("trunk.input_conv.weight")
    if inp_w is None:
        inp_w = state_dict.get("trunk.input_conv.conv.weight")
    if inp_w is None or inp_w.dim() != 4:
        return None
    in_ch = int(inp_w.shape[1])
    # Read the policy-head action count to disambiguate canonical encodings
    # from non-canonical (e.g. 9×9 test) shapes. A canonical match must
    # carry one of the well-known action counts (v6=362, v6w25=626, v8=625).
    n_actions: Optional[int] = None
    for k in ("policy_fc.weight", "cluster_pool.policy_mlp.2.weight"):
        w = state_dict.get(k)
        if w is not None and w.dim() == 2:
            n_actions = int(w.shape[0])
            break
    label = ckpt_label.lower()
    if in_ch == 11 and n_actions == 625:
        return WIRE_FORMAT_SPECS["v8"]
    if in_ch == BUFFER_CHANNELS:
        if n_actions == 626 or "v6w25" in label or "_w25" in label:
            # Filename hint can override action-count when the head is a
            # PMA variant whose output dim differs; in that case we trust
            # the operator's filename labeling.
            return WIRE_FORMAT_SPECS["v6w25"]
        if n_actions == 362:
            return WIRE_FORMAT_SPECS["v6"]
    return None


def _resolve_checkpoint_encoding(
    ckpt: Dict[str, Any],
    in_memory_config: Optional[Dict[str, Any]],
    model_state: Dict[str, torch.Tensor],
    checkpoint_path: Any,
) -> Optional[WireFormatSpec]:
    """Reconcile checkpoint encoding with the in-memory config encoding.

    Resolution rules:
      - Checkpoint encoding: prefer ckpt['config']['encoding'] (full
        ckpt). Otherwise infer from state_dict shape + filename.
      - In-memory encoding: read `in_memory_config['encoding']`. If
        absent, defaults to v6 via the registry resolver.
      - Disagreement on name → ValueError naming both sources.
      - In-memory config also pinning board_size / cluster_window_size /
        cluster_threshold that contradict the ckpt-resolved spec → also
        ValueError.

    Returns the ckpt-resolved ``WireFormatSpec`` on success.
    """
    # Step 1 — resolve the ckpt-side encoding.
    ckpt_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    ckpt_source: str
    ckpt_spec: Optional[WireFormatSpec]
    if isinstance(ckpt_cfg, dict) and ckpt_cfg.get("encoding") is not None:
        ckpt_registry_spec = registry_resolve_config(ckpt_cfg)
        ckpt_spec = _legacy_spec_for_registry_name(ckpt_registry_spec.name)
        ckpt_source = f"checkpoint {checkpoint_path!s} config['encoding']"
    else:
        ckpt_spec = _detect_encoding_from_state_dict(
            model_state, str(checkpoint_path),
        )
        ckpt_source = f"checkpoint {checkpoint_path!s} state_dict shape"
    if ckpt_spec is None:
        # Non-canonical state-dict shape (e.g. test fixture). Defer to
        # the legacy _resolve_model_hparams path entirely.
        return None

    # Step 2 — resolve the in-memory side. If the config carries no
    # explicit `encoding` section AND no explicit board_size/cluster_*
    # pin, treat the in-memory side as "unspecified" and accept the ckpt
    # spec without raising. This preserves backward compat for v6 ckpts
    # loaded against minimal configs.
    if in_memory_config is None:
        in_memory_config = {}
    explicit_encoding = isinstance(in_memory_config.get("encoding"), dict)
    explicit_pins: Dict[str, Any] = {}
    for key in ("board_size", "cluster_window_size", "cluster_threshold"):
        if in_memory_config.get(key) is not None:
            explicit_pins[key] = in_memory_config[key]

    if not explicit_encoding and not explicit_pins:
        return ckpt_spec

    # In-memory side has at least one pin — reconcile.
    if explicit_encoding:
        cfg_registry_spec = registry_resolve_config(in_memory_config)
        cfg_spec = _legacy_spec_for_registry_name(cfg_registry_spec.name)
        cfg_source = "in-memory config['encoding']"
    else:
        # No encoding section but explicit pins → derive a v6/v6w25-ish
        # comparison spec from the pins themselves.
        cfg_spec = ckpt_spec  # start from ckpt; override with pins for compare
        cfg_source = "in-memory config (pins)"

    if cfg_spec.name != ckpt_spec.name:
        raise ValueError(
            f"Encoding version disagrees: "
            f"{cfg_source}.version={cfg_spec.name!r} vs "
            f"{ckpt_source}.version={ckpt_spec.name!r}. "
            "Update the variant config to match the checkpoint encoding "
            "(or load a checkpoint matching the config). Refusing to "
            "silently override either direction."
        )

    # Even when versions match, refuse to silently disagree on
    # numeric pins (board_size / cluster_window_size / cluster_threshold).
    for key, cfg_val in explicit_pins.items():
        if key == "board_size":
            ckpt_val: Optional[int] = registry_lookup(ckpt_spec.name).trunk_size
        elif key == "cluster_window_size":
            ckpt_val = ckpt_spec.cluster_window_size
        elif key == "cluster_threshold":
            ckpt_val = ckpt_spec.cluster_threshold
        else:
            ckpt_val = None
        if ckpt_val is not None and int(cfg_val) != int(ckpt_val):
            raise ValueError(
                f"Encoding pin {key!r} disagrees: "
                f"in-memory config {key}={cfg_val} vs "
                f"{ckpt_source} {key}={ckpt_val} "
                f"(checkpoint encoding={ckpt_spec.name!r}). "
                "Remove the explicit pin from the config or load a "
                "matching checkpoint; refusing to silently override."
            )

    return ckpt_spec


def _propagate_encoding_into_config(
    config: Dict[str, Any], spec: WireFormatSpec,
) -> None:
    """Write the resolved encoding fields into the in-memory config.

    Downstream selfplay surfaces (pool, worker, lifecycle) read
    ``cluster_window_size`` / ``cluster_threshold`` / ``legal_move_radius``
    directly. After ckpt load, those must reflect the encoding the
    model was actually trained under, not whatever the variant YAML
    happened to ship with.

    §172 A10: `config["board_size"]` scalar retired. Readers use
    `resolve_from_config(cfg).trunk_size` instead.
    """
    encoding_section = config.get("encoding")
    if not isinstance(encoding_section, dict):
        encoding_section = {}
    encoding_section["version"] = spec.name
    config["encoding"] = encoding_section

    # board_size scalar retired — §172 A10.
    if spec.cluster_window_size is not None:
        config["cluster_window_size"] = int(spec.cluster_window_size)
    if spec.cluster_threshold is not None:
        config["cluster_threshold"] = int(spec.cluster_threshold)
    # legal_move_radius rides along — used by selfplay legal-mask jitter.
    config["legal_move_radius"] = int(spec.legal_move_radius)


def _resolve_model_hparams(
    trainer_cls: type,
    config: Dict[str, Any],
    model_state: Dict[str, torch.Tensor],
) -> Dict[str, int]:
    """Resolve model architecture hparams from config + state_dict inference.

    Priority (B-007):
      - If config explicitly sets a key AND state_dict inference yields a
        different value → raise ValueError. Do not silently override.
      - If inference yields a value → use it (covers weights-only resume).
      - Otherwise fall back to explicit config, then hard default.

    Sweep variant support: when config carries `input_channels`, the model
    in_channels is derived from len(input_channels). The conv weight shape
    in the checkpoint must agree.
    """
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}

    # Resolve input_channels override before hparam reconciliation so
    # in_channels can be derived consistently.
    input_channels_cfg = config.get("input_channels")
    if input_channels_cfg is None and isinstance(model_cfg, dict):
        input_channels_cfg = model_cfg.get("input_channels")
    if input_channels_cfg is not None:
        from hexo_rl.model.network import _validate_input_channels  # local import to avoid cycle
        input_channels_cfg = _validate_input_channels(input_channels_cfg)
        config["input_channels"] = list(input_channels_cfg)
        if isinstance(model_cfg, dict):
            model_cfg["input_channels"] = list(input_channels_cfg)
        derived_in = len(input_channels_cfg)
        # Override config's in_channels to match the variant's plane count.
        config["in_channels"] = derived_in
        if isinstance(model_cfg, dict):
            model_cfg["in_channels"] = derived_in

    defaults = dict(MODEL_HPARAM_DEFAULTS)

    def _explicit(key: str) -> Optional[int]:
        if isinstance(model_cfg, dict) and key in model_cfg:
            v = model_cfg[key]
            return int(v) if v is not None else None
        if key in config:
            v = config[key]
            return int(v) if v is not None else None
        return None

    inferred = trainer_cls._infer_model_hparams(model_state)

    # Sliced warm-start: when the variant pins `input_channels`, the
    # resulting model's in_channels is len(input_channels) regardless of
    # what the checkpoint's input-conv weight reports. The actual slicing
    # happens later in load_checkpoint when state_dict's input_conv has
    # WIRE_CHANNELS planes; this branch just keeps the hparam reconciler
    # from raising on the full-vs-sliced mismatch.
    if input_channels_cfg is not None and "in_channels" in inferred:
        inferred["in_channels"] = len(input_channels_cfg)

    resolved: Dict[str, int] = {}
    for key, default_val in defaults.items():
        cfg_val = _explicit(key)
        inf_val = inferred.get(key)

        if cfg_val is not None and inf_val is not None and int(cfg_val) != int(inf_val):
            raise ValueError(
                f"Model hparam '{key}' disagrees: config={cfg_val} vs "
                f"checkpoint-inferred={inf_val}. Remove '{key}' from config "
                f"to accept the checkpoint architecture, or load a checkpoint "
                f"matching the config."
            )
        if inf_val is not None:
            resolved[key] = int(inf_val)
        elif cfg_val is not None:
            resolved[key] = int(cfg_val)
        else:
            resolved[key] = int(default_val)

    # Keep top-level config aligned with final model dimensions.
    # board_size scalar retired (§172 A10) — readers use resolve_from_config(cfg).trunk_size.
    config["res_blocks"] = resolved["res_blocks"]
    config["filters"] = resolved["filters"]
    config["in_channels"] = resolved["in_channels"]
    if isinstance(model_cfg, dict):
        model_cfg.update(resolved)
        config["model"] = model_cfg

    return resolved


def load_checkpoint(
    cls: type,
    checkpoint_path: "str | Path",
    checkpoint_dir: Optional["str | Path"] = None,
    device: Optional[torch.device] = None,
    fallback_config: Optional[Dict[str, Any]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> "Trainer":
    """Restore a Trainer from a checkpoint file.

    Args:
        cls:             Trainer class (passed by the thin delegate so
                         hot-path helpers stay on Trainer).
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
    model_state = normalize_model_state_dict_keys(model_state)

    # Reconcile encoding (board_size / cluster window / threshold) BEFORE
    # _resolve_model_hparams so the trainer fails loudly with both source
    # values when the variant YAML and the checkpoint disagree (§171 P3
    # blocker: bootstrap_model_v6w25.pt vs v6-default variant config).
    # The in-memory side is whatever the user explicitly passed via
    # fallback_config / config_overrides — separate from the ckpt-baked
    # config that already populates `config` above.
    in_memory_view: Dict[str, Any] = {}
    if fallback_config is not None:
        in_memory_view.update(fallback_config)
    if config_overrides:
        in_memory_view.update(config_overrides)

    # §172 A4.3: prefer ckpt['metadata']['encoding_name'] (registry
    # source-of-truth) before falling back to shape-inference. A5 will
    # stamp this field on every existing artifact via migration script.
    ckpt_metadata = ckpt.get("metadata") if isinstance(ckpt, dict) else None
    registry_spec_from_meta: Optional[RegistrySpec] = None
    if isinstance(ckpt_metadata, dict) and "encoding_name" in ckpt_metadata:
        enc_name = ckpt_metadata["encoding_name"]
        if isinstance(enc_name, str):
            registry_spec_from_meta = registry_lookup(enc_name)
            log.info(
                "checkpoint_encoding_metadata_found",
                checkpoint_path=str(checkpoint_path),
                encoding_name=enc_name,
                source="metadata['encoding_name']",
            )
        else:
            import warnings
            warnings.warn(
                f"checkpoint {checkpoint_path}: metadata['encoding_name'] is "
                f"{type(enc_name).__name__}, expected str; falling back to "
                f"shape inference.",
                DeprecationWarning,
                stacklevel=2,
            )

    if registry_spec_from_meta is not None:
        # Bridge to wire-format spec (§176 P3 — replaces the legacy
        # NamedTuple). Propagation writes board_size / cluster_* /
        # legal_move_radius into the in-memory config from the wire
        # format mapped to the registry name.
        resolved_spec = _legacy_spec_for_registry_name(
            registry_spec_from_meta.name
        )
        _propagate_encoding_into_config(config, resolved_spec)
        log.info(
            "checkpoint_encoding_resolved",
            checkpoint_path=str(checkpoint_path),
            encoding_version=resolved_spec.name,
            encoding_name=registry_spec_from_meta.name,
            board_size=registry_resolve_config(config).trunk_size,
            cluster_window_size=config.get("cluster_window_size"),
            cluster_threshold=config.get("cluster_threshold"),
            source="registry_metadata",
        )
    else:
        # Backward-compat: legacy ckpts have no metadata['encoding_name'].
        # Emit a DeprecationWarning + fall through to shape inference.
        if isinstance(ckpt, dict) and (ckpt_metadata is None
                                        or not isinstance(ckpt_metadata, dict)
                                        or "encoding_name" not in ckpt_metadata):
            import warnings
            warnings.warn(
                f"checkpoint {checkpoint_path} lacks "
                f"metadata['encoding_name']; A5 migration script will "
                f"stamp existing artifacts. Using shape inference for now.",
                DeprecationWarning,
                stacklevel=2,
            )
        resolved_spec = _resolve_checkpoint_encoding(
            ckpt, in_memory_view, model_state, checkpoint_path,
        )
        if resolved_spec is not None:
            _propagate_encoding_into_config(config, resolved_spec)
            log.info(
                "checkpoint_encoding_resolved",
                checkpoint_path=str(checkpoint_path),
                encoding_version=resolved_spec.name,
                board_size=registry_resolve_config(config).trunk_size,
                cluster_window_size=config.get("cluster_window_size"),
                cluster_threshold=config.get("cluster_threshold"),
                source="shape_inference",
            )

    if config_overrides:
        config.update(config_overrides)
        if resolved_spec is not None:
            # config_overrides won the merge above — re-propagate spec so
            # the encoding pins survive (overrides shouldn't silently
            # re-introduce stale board_size etc.). Spec propagation is
            # idempotent.
            _propagate_encoding_into_config(config, resolved_spec)

    model_hparams = _resolve_model_hparams(cls, config, model_state)

    model = HexTacToeNet(
        board_size=model_hparams["board_size"],
        in_channels=model_hparams["in_channels"],
        res_blocks=model_hparams["res_blocks"],
        filters=model_hparams["filters"],
        se_reduction_ratio=model_hparams.get("se_reduction_ratio", MODEL_HPARAM_DEFAULTS["se_reduction_ratio"]),
        input_channels=config.get("input_channels"),
    )
    # If input_channels was explicitly nulled (e.g. loading a sweep checkpoint
    # as a full-18ch anchor), strip input_channel_index from the state dict so
    # _load_state_dict_strict doesn't reject it as an unexpected key.
    if config.get("input_channels") is None and "input_channel_index" in model_state:
        model_state = {k: v for k, v in model_state.items() if k != "input_channel_index"}

    # Sliced warm-start: when loading a full-18ch checkpoint (e.g. bootstrap)
    # into a sweep variant model with reduced in_channels, slice the input
    # conv weight along the input dim to keep only the variant's channels.
    # All other layers (after the first conv) are architecturally identical
    # so they load directly. Without this branch, the strict load below
    # would reject the (filters, 18, k, k) → (filters, N<18, k, k) shape
    # mismatch. The variant model also registers an `input_channel_index`
    # buffer (initialized in HexTacToeNet.__init__ from input_channels);
    # full-channel checkpoints don't have it, so we copy the model's own
    # buffer into the state dict to satisfy the strict-load missing-key
    # check without disturbing the actual indices.
    input_channels_cfg = config.get("input_channels")
    if input_channels_cfg is not None:
        from hexo_rl.model.network import WIRE_CHANNELS as _WIRE_CHANNELS
        _conv_key = "trunk.input_conv.weight"
        _w = model_state.get(_conv_key)
        _state_was_modified = False
        if (_w is not None
                and _w.dim() == 4
                and _w.shape[1] == _WIRE_CHANNELS
                and _w.shape[1] != len(input_channels_cfg)):
            model_state = dict(model_state)
            model_state[_conv_key] = _w[:, list(input_channels_cfg), :, :].contiguous()
            _state_was_modified = True
            log.info(
                "anchor_input_conv_sliced",
                src_channels=_WIRE_CHANNELS,
                dst_channels=len(input_channels_cfg),
                input_channels=list(input_channels_cfg),
                msg="warm-starting reduced-channel sweep variant from full-channel checkpoint",
            )
        if "input_channel_index" not in model_state and hasattr(model, "input_channel_index"):
            _idx = model.input_channel_index
            if isinstance(_idx, torch.Tensor):
                if not _state_was_modified:
                    model_state = dict(model_state)
                model_state["input_channel_index"] = _idx.detach().clone()

    cls._load_state_dict_strict(model, model_state)

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
        if trainer.scheduler is not None:
            ckpt_scheduler_state = ckpt.get("scheduler_state")
            if ckpt_scheduler_state is None:
                allow_fresh = bool((config_overrides or {}).get("allow_fresh_scheduler", False))
                if not allow_fresh:
                    raise ValueError(
                        "scheduler_state missing from checkpoint; cannot resume. "
                        "Pass --allow-fresh-scheduler (or config_overrides['allow_fresh_scheduler']=True) "
                        "to rebuild the scheduler from scratch."
                    )
                log.warning(
                    "scheduler_state_missing_fresh_start",
                    schedule=trainer.config.get("lr_schedule"),
                )
            else:
                scheduler_state = ckpt_scheduler_state
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
