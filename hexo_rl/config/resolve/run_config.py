"""CONFRES Phase-3 batch 6a-i — the aggregating ResolvedRunConfig builder.

A PURE, cycle-free builder that AGGREGATES the batch-1..5 single-knob resolvers into one frozen
``ResolvedRunConfig`` carrying per-knob provenance. Two invocation surfaces (design §2):

- ``resolve_preload_config`` (Phase-A, I7/F3) — the launch-only knobs consumed BEFORE the
  checkpoint loads (``seed``/``device``/``tf32``): resolved from ``variant_layers`` + ``cli`` with
  NO checkpoint ingestion; flagged ``consumed_pre_resolution: true`` for emission (6a-ii).
- ``resolve_run_config`` (Phase-B) — everything else, post-load, aggregating the ``resolve.*``
  modules (encoding / n_sims-eval / eval-temperature / bootstrap / lr).

Scope note (6a-i): PURE functions + frozen dataclasses only. NO launch wiring, NO consumer
migration, NO precedence change to live paths, NO emission event fired anywhere.
``to_event_payload`` RENDERS the §5 ``resolved_config`` payload but is NOT emitted (that is 6a-ii).

Import constraint (§8 N3): stdlib + ``hexo_rl.config.resolve.*`` + ``hexo_rl.encoding`` +
``hexo_rl.utils.config`` ONLY — no ``hexo_rl.eval`` / ``.selfplay`` / ``.training`` import, to keep
the resolve package cycle-free.

Design: docs/designs/confres_design.md §1, §2, §3, §5, §10, §13.
"""
from __future__ import annotations

import copy
import types
from dataclasses import dataclass, field
from typing import Any, Mapping

import yaml

from hexo_rl.config.resolve.bootstrap import resolve_bootstrap
from hexo_rl.config.resolve.encoding import UNSPECIFIED as _ENC_UNSPECIFIED
from hexo_rl.config.resolve.encoding import EncodingConflictError
from hexo_rl.config.resolve.encoding import normalize_declared as _enc_normalize_declared
from hexo_rl.config.resolve.encoding import normalize_stamp as _enc_normalize_stamp
from hexo_rl.config.resolve.encoding import reconcile_declared_vs_stamp as _enc_reconcile
from hexo_rl.config.resolve.lr import LrProvenance, resolve_lr_provenance
from hexo_rl.config.resolve.nsims import resolve_eval_model_sims
from hexo_rl.config.resolve.temperature import resolve_eval_temperature
from hexo_rl.utils.config import _deep_merge

# 6c: the ABSENT-encoding sentinel now lives in ``resolve.encoding`` (single source, delegated to by
# both surfaces). Re-exported here under the historical name for any importer of this module.
UNSPECIFIED = _ENC_UNSPECIFIED

# The valid vocabularies (design §2 ResolvedValue). Enforced at construction so a typo in a source
# label surfaces as a build error, not a silent mis-labelled provenance row.
_VALID_SOURCES = frozenset(
    {"registry", "variant", "checkpoint", "checkpoint_state", "cli", "derived", "default", "external"}
)
_VALID_FAMILIES = frozenset(
    {"raise-on-conflict", "checkpoint-wins-loud", "variant-wins", "derived", "documented-external"}
)


def _render_event_value(value: Any) -> Any:
    """Render a knob value for the JSONL ``resolved_config`` event.

    The ``lr`` knob's value is an ``LrProvenance`` dataclass — serialise it as a NESTED DICT
    (``{"declared","baked","effective","override_ignored"}``) so the forensic artifact is
    structured, not an opaque repr string. Every other value renders verbatim. The in-memory
    ``lr_provenance()`` accessor still returns the dataclass (it reads ``._values["lr"].value``,
    which this does NOT mutate).
    """
    if isinstance(value, LrProvenance):
        return {
            "declared": value.declared,
            "baked": value.baked,
            "effective": value.effective,
            "override_ignored": value.override_ignored,
        }
    return value


class ConfigConflictError(ValueError):
    """Two PRESENT, differing values for a raise-on-conflict knob (I2, design §2/§3).

    Subclasses ``ValueError`` so existing ``except ValueError`` build-time handlers still catch it.
    The message NAMES the knob and every conflicting (source → value) pair — the F1-forensic
    requirement that a checkpoint may INFORM but never SILENTLY override a declared value.
    """

    def __init__(self, knob: str, sources: Mapping[str, Any]):
        self.knob = knob
        self.sources = dict(sources)
        rendered = ", ".join(f"{s}={v!r}" for s, v in self.sources.items())
        super().__init__(
            f"config conflict on {knob!r}: {rendered} (declared value and checkpoint stamp "
            f"disagree; a checkpoint may INFORM but never silently override a declared value)"
        )


@dataclass(frozen=True)
class ResolvedValue:
    """One resolved knob + its provenance (design §2).

    ``inputs_seen`` records EVERY source that offered a value → its raw value (incl the
    checkpoint-baked config for variant-wins knobs — B3), so emission can never lie.
    """

    value: Any
    source: str
    precedence_family: str
    inputs_seen: Mapping[str, Any] = field(default_factory=dict)
    # Phase-A pre-load knobs (I7) carry this flag so 6a-ii emits ``consumed_pre_resolution: true``.
    consumed_pre_resolution: bool = False

    def __post_init__(self) -> None:
        if self.source not in _VALID_SOURCES:
            raise ValueError(f"invalid ResolvedValue.source {self.source!r}; allowed: {sorted(_VALID_SOURCES)}")
        if self.precedence_family not in _VALID_FAMILIES:
            raise ValueError(
                f"invalid ResolvedValue.precedence_family {self.precedence_family!r}; "
                f"allowed: {sorted(_VALID_FAMILIES)}"
            )
        # M6 frozen guarantee: snapshot inputs_seen so post-build mutation of the caller's dict is
        # inert (previously stored a reference; incidental string immutability was the only guard).
        object.__setattr__(self, "inputs_seen", dict(self.inputs_seen))

    def as_event_dict(self) -> dict:
        d = {
            "value": _render_event_value(self.value),
            "source": self.source,
            "precedence_family": self.precedence_family,
            "inputs_seen": dict(self.inputs_seen),
        }
        if self.consumed_pre_resolution:
            d["consumed_pre_resolution"] = True
        return d


@dataclass(frozen=True)
class ResolvedRunConfig:
    """Frozen bundle of resolved knobs. Accessors read ``_values[knob].value`` (design §2).

    Only the knobs resolvable NOW (batch 6a-i) have typed accessors. window_set/eval_radius/
    planner/loop_horizon/… land in later sub-batches (encoding-registry lookup, radius curriculum,
    planner factory). ``provenance`` + ``to_event_payload`` are generic over whatever is present.
    """

    _values: Mapping[str, ResolvedValue]

    def __post_init__(self) -> None:
        # M6 frozen guarantee: snapshot _values to a read-only MappingProxy so neither the caller's
        # dict nor the internal reference can be mutated after build.
        object.__setattr__(self, "_values", types.MappingProxyType(dict(self._values)))

    # ── generic provenance ───────────────────────────────────────────────────
    def provenance(self, knob: str) -> ResolvedValue:
        try:
            return self._values[knob]
        except KeyError:
            raise KeyError(f"no resolved knob {knob!r}; resolved: {sorted(self._values)}") from None

    def to_event_payload(self) -> dict:
        """Render the §5 ``resolved_config`` event payload. NOT emitted here (6a-ii wires that)."""
        return {
            "event": "resolved_config",
            "knobs": {knob: rv.as_event_dict() for knob, rv in self._values.items()},
        }

    # ── typed accessors (batch 6a-i knobs) ───────────────────────────────────
    def encoding_name(self) -> str:
        return self._values["encoding"].value

    def n_sims_eval(self, opponent: str) -> int:
        return self._values[f"n_sims_eval.{opponent}"].value

    def eval_temperature(self) -> float:
        return self._values["eval_temperature"].value

    def bootstrap_path(self) -> str | None:
        return self._values["bootstrap"].value

    def lr_provenance(self) -> LrProvenance:
        return self._values["lr"].value


# ── raw-layer capture + merge helpers (F2/F4) ────────────────────────────────
def capture_config_layers(
    base_paths: list[str],
    config_override: str | None,
    variant_path: str | None,
) -> list[dict]:
    """Return the RAW pre-merge yaml layer chain in LOAD ORDER (F2/F4).

    Order = base configs → ``--config`` override → variant (later-wins, matching
    ``load_train_config``: base → --config → variant). Each layer is
    ``{"label": <path>, "raw": <parsed dict>}``. ``--config`` is a first-class chain position
    (the round-3 F2 miss was omitting it). The variant is HIGHEST precedence.
    """
    layers: list[dict] = []
    for p in base_paths:
        with open(p) as f:
            raw = yaml.safe_load(f) or {}
        layers.append({"label": p, "kind": "base", "raw": raw})
    if config_override is not None:
        with open(config_override) as f:
            raw = yaml.safe_load(f) or {}
        layers.append({"label": config_override, "kind": "config", "raw": raw})
    if variant_path is not None:
        with open(variant_path) as f:
            raw = yaml.safe_load(f) or {}
        layers.append({"label": variant_path, "kind": "variant", "raw": raw})
    return layers


def merged_layers(layers: list[dict]) -> dict:
    """Deep-merge the raw layer chain, later-wins (reuses ``hexo_rl.utils.config._deep_merge``)."""
    merged: dict = {}
    for layer in layers:
        _deep_merge(merged, copy.deepcopy(layer["raw"]))
    return merged


def assert_layers_reconstruct(layers: list[dict], combined_config: Mapping[str, Any]) -> None:
    """Assert ``merged_layers(layers) == combined_config`` (F2 build-time invariant).

    Raises ``ValueError`` naming the mismatching top-level keys, so the resolver's per-key
    provenance can only ADD to the merged value, never diverge from what the rest of the code reads.
    """
    reconstructed = merged_layers(layers)
    if reconstructed == combined_config:
        return
    keys = sorted(set(reconstructed) | set(combined_config))
    mismatched = [k for k in keys if reconstructed.get(k) != combined_config.get(k)]
    raise ValueError(
        f"layer chain does not reconstruct combined_config; mismatching keys: {mismatched}"
    )


# ── layer-provenance helpers ─────────────────────────────────────────────────
def _variant_layers_only(layers: list[dict]) -> list[dict]:
    """The operator-declaration layers (``--config`` + variant), NOT the base configs.

    A key present in one of THESE is a DECLARATION (incl explicit null); a key present only in a
    base layer is INHERITED, not a declaration (B5a / I4).

    Primary rule: if the layer dict carries a ``"kind"`` field (set by ``capture_config_layers``),
    a declaration layer is one whose kind is ``"config"`` or ``"variant"`` — base layers are
    inherited, not declarations. This is kind-explicit and robust against any filename convention.

    Backward-compat fallback: layers WITHOUT a ``"kind"`` key (e.g. hand-built test dicts) are
    classified by the OLD label heuristic so existing tests that build raw ``{"label","raw"}``
    dicts still classify correctly. When the heuristic matches nothing, fall back to "all but the
    first base layer".
    """
    decl = [ly for ly in layers if _is_declaration_layer(ly)]
    # If nothing matched, fall back to "all but the first base layer".
    return decl if decl else layers[1:]


def _is_declaration_layer(layer: dict) -> bool:
    """Return True iff ``layer`` is an operator-declaration layer (not a base/inherited layer).

    Uses the explicit ``"kind"`` field when present; falls back to the old label heuristic for
    hand-built layers that lack ``"kind"``.
    """
    kind = layer.get("kind")
    if kind is not None:
        return kind in {"config", "variant"}
    # Backward-compat: no "kind" key — use old filename-based heuristic on the label.
    label = layer["label"]
    return ("variants/" in label) or ("variant" in label.rsplit("/", 1)[-1])


def _lookup_in_layers(layers: list[dict], *keys: str):
    """Highest-precedence PRESENT value for a (possibly nested) key path across ``layers``.

    Returns ``(found: bool, value)``. Layers are in load order (later-wins), so we scan reversed.
    """
    for layer in reversed(layers):
        node: Any = layer["raw"]
        ok = True
        for k in keys:
            if isinstance(node, Mapping) and k in node:
                node = node[k]
            else:
                ok = False
                break
        if ok:
            return True, node
    return False, None


def _lookup_baked(baked: Mapping[str, Any] | None, *keys: str):
    if baked is None:
        return False, None
    node: Any = baked
    for k in keys:
        if isinstance(node, Mapping) and k in node:
            node = node[k]
        else:
            return False, None
    return True, node


# ── encoding resolution (I1/I2, raise-on-conflict) — DELEGATES to resolve.encoding ──
def _resolve_encoding(
    variant_layers: list[dict],
    checkpoint_stamps: Mapping[str, Any],
) -> ResolvedValue:
    """Resolve the encoding name (I1 presence-before-normalize, I2 conflict-raise, B5a absent→stamp).

    CONFRES 6c: the raise/normalize DECISION lives in ``resolve.encoding.reconcile_declared_vs_stamp``
    — the ONE rule the eval checkpoint loader ALSO delegates to (design law #1). This function only
    maps the layer/stamp provenance onto a ``ResolvedValue`` + re-raises the shared
    ``EncodingConflictError`` as the builder's ``ConfigConflictError`` (same ``ValueError`` lineage,
    same both-sources-named contract).
    """
    decl_layers = _variant_layers_only(variant_layers)
    # I1: PRESENCE test BEFORE normalization — a key present in a DECLARATION layer is a decl.
    present, raw_decl = _lookup_in_layers(decl_layers, "encoding")
    declared = _enc_normalize_declared(present, raw_decl)
    stamp = _enc_normalize_stamp(checkpoint_stamps)

    inputs: dict[str, Any] = {}
    if declared is not _ENC_UNSPECIFIED:
        inputs["variant"] = declared
    if stamp is not None:
        inputs["checkpoint"] = stamp

    try:
        res = _enc_reconcile(declared, stamp)
    except EncodingConflictError as exc:
        raise ConfigConflictError(
            "encoding", {"variant": exc.declared, "checkpoint": exc.stamp}
        ) from exc
    if res.source == "default":
        inputs["default"] = res.name
    return ResolvedValue(res.name, res.source, "raise-on-conflict", inputs)


# ── variant-wins scalar resolution (I4: cli → variant-layer → ckpt-baked → default) ──
def _resolve_variant_wins_scalar(
    knob: str,
    cli_val,
    cli_present: bool,
    variant_layers: list[dict],
    baked: Mapping[str, Any] | None,
    key_path: tuple[str, ...],
    default,
) -> ResolvedValue:
    """Resolve a variant-wins scalar with full inputs_seen provenance (I4/B3).

    Precedence: cli → variant-layer-present → checkpoint-baked → base default. Every source that
    offered a value is recorded in ``inputs_seen``.
    """
    decl_layers = _variant_layers_only(variant_layers)
    v_present, v_val = _lookup_in_layers(decl_layers, *key_path)
    b_present, b_val = _lookup_baked(baked, *key_path)

    inputs: dict[str, Any] = {}
    if cli_present:
        inputs["cli"] = cli_val
    if v_present:
        inputs["variant"] = v_val
    if b_present:
        inputs["checkpoint"] = b_val

    if cli_present and cli_val is not None:
        return ResolvedValue(cli_val, "cli", "variant-wins", inputs)
    if v_present and v_val is not None:
        return ResolvedValue(v_val, "variant", "variant-wins", inputs)
    if b_present and b_val is not None:
        return ResolvedValue(b_val, "checkpoint", "variant-wins", inputs)
    inputs.setdefault("default", default)
    return ResolvedValue(default, "default", "variant-wins", inputs)


# ── Phase-A: pre-load-consumed knobs (I7/F3) ─────────────────────────────────
_PRELOAD_SEED_DEFAULT = 42


def resolve_preload_config(variant_layers: list[dict], cli: Mapping[str, Any]) -> ResolvedRunConfig:
    """Phase-A (I7/F3): resolve ONLY the launch-consumed-before-checkpoint knobs.

    ``seed`` / ``device`` / ``tf32`` are consumed at launch BEFORE the checkpoint loads, so they
    are resolved from ``variant_layers`` + ``cli`` with NO checkpoint ingestion (their emitted
    value IS what the run used). They are a distinct precedence family (variant-wins, launch-only,
    NO ckpt-baked source — I4 does not apply) and are flagged ``consumed_pre_resolution: true``.
    """
    values: dict[str, ResolvedValue] = {}
    decl_layers = variant_layers  # for the launch surface every layer is a declaration layer

    # seed: cli → variant/base layer → default 42
    seed = _resolve_launch_scalar(
        "seed", cli, decl_layers, key_path=("seed",), default=_PRELOAD_SEED_DEFAULT
    )
    values["seed"] = seed

    # device: OS/host-derived (best_device) — not a config knob. Recorded as derived, launch-only.
    values["device"] = ResolvedValue(
        _lookup_or_default(cli, decl_layers, ("device",), None),
        "cli" if "device" in cli else ("variant" if _lookup_in_layers(decl_layers, "device")[0] else "derived"),
        "variant-wins",
        _launch_inputs(cli, decl_layers, ("device",)),
        consumed_pre_resolution=True,
    )

    # tf32: a block ({tf32_matmul, tf32_cudnn}); variant-wins over base, launch-only.
    tf32_present, tf32_val = _lookup_in_layers(decl_layers, "tf32")
    values["tf32"] = ResolvedValue(
        tf32_val if tf32_present else None,
        "cli" if "tf32" in cli else ("variant" if tf32_present else "default"),
        "variant-wins",
        _launch_inputs(cli, decl_layers, ("tf32",)),
        consumed_pre_resolution=True,
    )

    return ResolvedRunConfig(_values=values)


def _resolve_launch_scalar(knob, cli, decl_layers, *, key_path, default) -> ResolvedValue:
    cli_present = key_path[-1] in cli
    cli_val = cli.get(key_path[-1]) if cli_present else None
    v_present, v_val = _lookup_in_layers(decl_layers, *key_path)
    inputs = _launch_inputs(cli, decl_layers, key_path)
    if cli_present and cli_val is not None:
        return ResolvedValue(cli_val, "cli", "variant-wins", inputs, consumed_pre_resolution=True)
    if v_present and v_val is not None:
        return ResolvedValue(v_val, "variant", "variant-wins", inputs, consumed_pre_resolution=True)
    inputs.setdefault("default", default)
    return ResolvedValue(default, "default", "variant-wins", inputs, consumed_pre_resolution=True)


def _launch_inputs(cli, decl_layers, key_path) -> dict:
    inputs: dict[str, Any] = {}
    if key_path[-1] in cli:
        inputs["cli"] = cli[key_path[-1]]
    v_present, v_val = _lookup_in_layers(decl_layers, *key_path)
    if v_present:
        inputs["variant"] = v_val
    return inputs


def _lookup_or_default(cli, decl_layers, key_path, default):
    if key_path[-1] in cli and cli[key_path[-1]] is not None:
        return cli[key_path[-1]]
    v_present, v_val = _lookup_in_layers(decl_layers, *key_path)
    if v_present and v_val is not None:
        return v_val
    return default


# ── Phase-B: full post-load resolution ───────────────────────────────────────
def resolve_run_config(
    registry: Any,
    variant_layers: list[dict],
    combined_config: Mapping[str, Any],
    checkpoint_stamps: Mapping[str, Any],
    checkpoint_state: Mapping[str, Any],
    cli: Mapping[str, Any],
    *,
    checkpoint_baked: Mapping[str, Any] | None = None,
    require: tuple[str, ...] = (),
) -> ResolvedRunConfig:
    """Phase-B builder — resolve the post-load knobs, aggregating the ``resolve.*`` modules.

    Parameters mirror design §2. ``checkpoint_baked`` is the checkpoint's baked ``config`` blob
    (the ``ckpt["config"]`` position of the variant-wins resume chain, I4/B3); optional so
    fresh-run builds pass nothing. ``checkpoint_state`` is the optimizer/scheduler STATE blob (I3).

    ``require`` names knobs that MUST resolve from a real source (not merely their compat default):
    a required knob whose only source is ``default`` → ``ValueError`` at build (M7/I5).

    Missing a required knob → ``ValueError`` at build (M7/I5). (The F2 invariant
    ``merge(variant_layers) == combined_config`` is asserted at LOAD time by the launch caller
    against the PRISTINE config — NOT here, where ``combined_config`` may reflect legitimate
    downstream transforms like mixing/buffer ``<auto>``-path expansion or resume encoding
    back-prop.)
    """
    values: dict[str, ResolvedValue] = {}

    # encoding (raise-on-conflict, I1/I2/B5a) — REQUIRED (M7).
    values["encoding"] = _resolve_encoding(variant_layers, checkpoint_stamps)

    # eval_temperature (variant-wins; resolver applies the 0.5 default).
    et_present, et_raw = _lookup_in_layers(
        _variant_layers_only(variant_layers), "evaluation", "eval_temperature"
    )
    et_inputs: dict[str, Any] = {}
    if et_present:
        et_inputs["variant"] = et_raw
    baked_et_present, baked_et = _lookup_baked(checkpoint_baked, "evaluation", "eval_temperature")
    if baked_et_present:
        et_inputs["checkpoint"] = baked_et
    et_cfg_value = et_raw if et_present else (baked_et if baked_et_present else None)
    et_source = "variant" if et_present else ("checkpoint" if baked_et_present else "default")
    values["eval_temperature"] = ResolvedValue(
        resolve_eval_temperature(et_cfg_value), et_source, "variant-wins", et_inputs
    )

    # n_sims eval per opponent (variant-wins; nsims resolver applies the per-opponent default).
    for opponent in ("random", "sealbot"):
        key = f"{opponent}_model_sims"
        rv = _resolve_variant_wins_scalar(
            f"n_sims_eval.{opponent}",
            cli.get(key), key in cli, variant_layers, checkpoint_baked,
            ("evaluation", key), default=None,
        )
        cfg_value = rv.value  # None → resolver default fires
        resolved_sims = resolve_eval_model_sims(opponent, cfg_value)
        source = rv.source if cfg_value is not None else "default"
        values[f"n_sims_eval.{opponent}"] = ResolvedValue(
            resolved_sims, source, "variant-wins", rv.inputs_seen
        )

    # bootstrap (variant-wins + conditional existence-validate, P3/I5).
    cli_ckpt = cli.get("checkpoint")
    rb = resolve_bootstrap(cli_ckpt)
    boot_inputs: dict[str, Any] = {}
    if "checkpoint" in cli:
        boot_inputs["cli"] = cli_ckpt
    boot_source = "cli" if rb.source == "cli" else "default"
    values["bootstrap"] = ResolvedValue(rb.path, boot_source, "variant-wins", boot_inputs)

    # lr provenance (checkpoint-wins-loud, I3/B2). Only meaningful on a full resume; on a fresh run
    # declared/baked/effective are all absent → override_ignored False.
    declared_present, declared_lr = _lookup_in_layers(_variant_layers_only(variant_layers), "lr")
    baked_lr_present, baked_lr = _lookup_baked(checkpoint_baked, "lr")
    effective_lr = _effective_lr_from_state(checkpoint_state)
    lr_prov = resolve_lr_provenance(
        declared=declared_lr if declared_present else None,
        baked=baked_lr if baked_lr_present else None,
        effective=effective_lr,
    )
    lr_inputs: dict[str, Any] = {}
    if declared_present:
        lr_inputs["variant"] = declared_lr
    if baked_lr_present:
        lr_inputs["checkpoint"] = baked_lr
    if effective_lr is not None:
        lr_inputs["checkpoint_state.param_groups"] = effective_lr
    lr_source = "checkpoint_state" if effective_lr is not None else (
        "variant" if declared_present else "default"
    )
    values["lr"] = ResolvedValue(lr_prov, lr_source, "checkpoint-wins-loud", lr_inputs)

    # I5/M7: a required knob that resolved only to its compat default (no real source) is missing.
    for knob in require:
        rv = values.get(knob)
        if rv is None:
            raise ValueError(f"required knob {knob!r} was not resolved by the builder")
        if rv.source == "default":
            raise ValueError(
                f"required knob {knob!r} has no source — resolved only to its default "
                f"{rv.value!r} (no cli/variant/checkpoint offered a value)"
            )

    return ResolvedRunConfig(_values=values)


def _effective_lr_from_state(checkpoint_state: Mapping[str, Any]) -> float | None:
    """Read the restored optimizer param_group lr from the state blob (I3/B2), else None."""
    opt = checkpoint_state.get("optimizer_state") if checkpoint_state else None
    if not isinstance(opt, Mapping):
        return None
    pgs = opt.get("param_groups")
    if isinstance(pgs, (list, tuple)) and pgs and isinstance(pgs[0], Mapping) and "lr" in pgs[0]:
        return float(pgs[0]["lr"])
    return None
