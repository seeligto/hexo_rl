"""Encoding resolvers — config-form, checkpoint-form, state-dict validation.

Authored §172 Phase A3 (2026-05-09). The two `resolve_*` functions are
the only A4-blessed paths to construct an `EncodingSpec` outside the
registry itself; consumer call sites should never call `lookup` with a
hard-coded string except for explicit defaults.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from engine import RegistrySpec as EncodingSpec  # type: ignore[attr-defined]

from hexo_rl.encoding import compat
from hexo_rl.encoding._probes import FIRST_CONV_KEYS as _FIRST_CONV_KEYS
from hexo_rl.encoding._probes import POLICY_FC_KEYS as _POLICY_FC_KEYS
from hexo_rl.encoding.registry import EncodingRegistryError, _load as _load_registry, lookup


class ShapeMismatchError(Exception):
    """Raised when state-dict shapes contradict an EncodingSpec."""


# Scattered-key invariant (§172 A4.5 — design doc §10).
# Map config-level scalar keys to the corresponding EncodingSpec field.
# `in_channels` is the legacy buffer-format alias for `n_planes`.
# Sentinel value used by expand_auto_paths to detect unresolved artifact paths.
_AUTO = "<auto>"

def normalize_encoding_name(enc: Any) -> str:
    """Coerce a config encoding value to its registry name string.

    Accepts the four forms that show up at consumer sites:
      - str ``"v6"``                           → returned as-is
      - dict ``{"version": "v6", ...}`` or
             ``{"name": "v6", ...}``           → version/name extracted
      - object with ``.name`` (EncodingSpec)   → ``.name`` returned
      - ``None``                               → default ``"v6"``

    §175 eval-fix: `Trainer._propagate_encoding_into_config` rewrites
    `config["encoding"]` from the initial string form to a
    ``{"version": <name>}`` dict on resume. Downstream sites that call
    ``lookup(config.get("encoding"))`` must funnel through this helper or
    they crash with ``TypeError: unhashable type: 'dict'``.
    """
    if enc is None:
        return "v6"
    if isinstance(enc, str):
        return enc
    if isinstance(enc, Mapping):
        name = enc.get("name", enc.get("version", "v6"))
        if not isinstance(name, str):
            raise EncodingRegistryError(
                f"encoding mapping name/version must be a string; "
                f"got {type(name).__name__}: {name!r}"
            )
        return name
    name = getattr(enc, "name", None)
    if isinstance(name, str):
        return name
    raise EncodingRegistryError(
        f"cannot extract encoding name from {type(enc).__name__}: {enc!r}"
    )


_SCATTERED_KEYS_TO_FIELD: dict[str, str] = {
    "board_size": "board_size",
    "cluster_window_size": "cluster_window_size",
    "cluster_threshold": "cluster_threshold",
    "legal_move_radius": "legal_move_radius",
    "n_planes": "n_planes",
    "in_channels": "n_planes",
}


def _check_scattered_keys(cfg: Mapping[str, Any], spec: EncodingSpec) -> None:
    """Raise EncodingRegistryError if any scattered key disagrees with spec.

    Consistency rule: if a key is present in the config AND the registry spec
    has a non-None value for the corresponding field, the integers must match.
    Keys absent from the config or with `None` registry values are skipped.
    """
    if not cfg:
        return
    disagreements: list[str] = []
    for cfg_key, spec_field in _SCATTERED_KEYS_TO_FIELD.items():
        cfg_val = cfg.get(cfg_key)
        if cfg_val is None:
            continue
        spec_val = getattr(spec, spec_field, None)
        if spec_val is None:
            # Registry says "n/a" for this field on this encoding (e.g.
            # cluster_window_size is None for v6); accept whatever the
            # config says rather than fight legacy scaffolding.
            continue
        try:
            cfg_int = int(cfg_val)
        except (TypeError, ValueError):
            disagreements.append(
                f"  - {cfg_key}={cfg_val!r} (config) is not an int; "
                f"{spec_field}={spec_val} (encoding {spec.name!r})"
            )
            continue
        if cfg_int != int(spec_val):
            disagreements.append(
                f"  - {cfg_key}={cfg_val} (config) vs {spec_field}={spec_val} "
                f"(encoding {spec.name!r} from registry.toml)"
            )
    if disagreements:
        raise EncodingRegistryError(
            f"variant config has scattered key(s) that disagree with the "
            f"declared encoding {spec.name!r}:\n"
            + "\n".join(disagreements)
            + f"\n\nRemove the scattered key(s) and let the registry decide. "
            f"Registered encodings: {sorted(_load_registry())}. "
            f"Schema: docs/designs/encoding_registry_design.md §10."
        )


# ---------------------------------------------------------------------------
# Canonical artifact paths per encoding name.
# Operator-curated; bump when a new encoding ships corpora/anchors.
# Paths are repo-relative (no leading slash).
# ---------------------------------------------------------------------------

_CORPUS_PATHS: dict[str, str] = {
    "v6":                 "data/bootstrap_corpus.npz",
    "v6tp":               "data/bootstrap_corpus_v6tp.npz",  # §P5-CT CF-2 (10-plane, incl. 16/17)
    "v6_live2":           "data/bootstrap_corpus_v6_live2.npz",  # §P5-CT H-PLANE fix (4-plane [0,8,16,17])
    "v6_live2_anchored":  "data/bootstrap_corpus_v6_live2.npz",  # §PRELONG-2A — shares v6_live2 corpus (eval-only; re-pretrain would mint its own)
    "v6w25":              "data/bootstrap_corpus_v6w25.npz",
    "v7full":             "data/bootstrap_corpus.npz",   # shared with v6 (§150)
    "v7mw":               "data/bootstrap_corpus.npz",   # shared with v6/v7full (§176a)
    "v8":                 "data/bootstrap_corpus_v8.npz",
    "v8_canvas_realness": "data/bootstrap_corpus_v8_canvas_realness.npz",
}

_ANCHOR_PATHS: dict[str, str] = {
    "v6":                 "checkpoints/bootstrap_model_v6.pt",
    "v6tp":               "checkpoints/bootstrap_model_v6tp.pt",  # §P5-CT CF-2 self-anchor
    "v6_live2":           "checkpoints/bootstrap_model_v6_live2.pt",  # §P5-CT H-PLANE fix self-anchor
    "v6_live2_anchored":  "checkpoints/bootstrap_model_v6_live2.pt",  # §PRELONG-2A — frozen v6_live2 weights load unchanged
    "v6w25":              "checkpoints/bootstrap_model_v6w25.pt",
    "v7full":             "checkpoints/bootstrap_model_v7full.pt",
    "v7mw":               "checkpoints/bootstrap_model_v7full.pt",   # v7full anchor (same arch §176a)
    "v8":                 "checkpoints/bootstrap_model_v8full_warm.pt",
    "v8_canvas_realness": "checkpoints/v8_variants/A4_canvas_realness.pt",
}


# ---------------------------------------------------------------------------
# Architecture resolver — ONE registry-derived map from an encoding NAME to the
# arch facts consumers used to hardcode (plane count, kept-index list, stone
# slots, policy width). Authored §P5-CT de-hardcoding sweep (2026-05-31, L63/
# L65 follow-through). Every P0/P1 ledger site routes through this instead of a
# literal 18 / module-level v6 `KEPT_PLANE_INDICES` / bare `[:, 4]` slice.
#
# Source-plane semantics fixed by the v6 wire format (game_state.to_tensor):
#   0       → current-player stone, t0          (always kept-slot 0)
#   8       → opponent stone, t0
#   1,2,3   → current-player history t-1..t-3   (H-PLANE class, §P5-CT)
#   9,10,11 → opponent history t-1..t-3
#   16,17   → turn-phase scalars (CF-2 class, §P5-CT)
# `history_planes` / `turn_phase_planes` are the kept-SLOT indices (positions
# within the sliced tensor) of those source planes that the encoding retains.
# ---------------------------------------------------------------------------

_CUR_STONE_SRC_PLANE = 0
_OPP_STONE_SRC_PLANE = 8
# MAGIC-2 (2026-06-02) — named singletons for the turn-phase scalar planes so no
# source consumer (game_state corpus tensor build, registry pin) carries a bare
# 16/17. Mirror of engine core.rs MOVES_REMAINING_PLANE / PLY_PARITY_PLANE; the
# registry kept_plane_indices select from these (registry is canonical).
_MOVES_REMAINING_SRC_PLANE = 16
_PLY_PARITY_SRC_PLANE = 17
_HISTORY_SRC_PLANES = frozenset({1, 2, 3, 9, 10, 11})
_TURN_PHASE_SRC_PLANES = frozenset({_MOVES_REMAINING_SRC_PLANE, _PLY_PARITY_SRC_PLANE})


def _kept_slot_of(kept: list[int], src_plane: int) -> int:
    """Position of ``src_plane`` within the encoding's kept-plane order."""
    return kept.index(src_plane)


@dataclass(frozen=True)
class ArchSpec:
    """Registry-derived architecture facts for a single encoding.

    A thin, typed, immutable view over the registry `EncodingSpec` — every
    field is computed from `lookup(name)`, never hardcoded. This is the single
    place a consumer asks "for encoding X, how many input channels / which
    kept planes / where is the opponent stone" instead of baking v6 literals.
    """

    name: str
    in_channels: int               # = spec.n_planes (model trunk in_channels)
    kept_indices: tuple[int, ...]  # = spec.kept_plane_indices (source→wire slice)
    cur_stone_slot: int            # kept-slot of source plane 0 (always 0)
    opp_stone_slot: int            # kept-slot of source plane 8
    k_max: int                     # = spec.k_max (multi-window cluster cap)
    policy_logit_count: int        # = spec.policy_logit_count
    history_planes: tuple[int, ...]     # kept-slots of source {1,2,3,9,10,11}
    turn_phase_planes: tuple[int, ...]  # kept-slots of source {16,17}


def resolve_arch(name: Any) -> ArchSpec:
    """Resolve an encoding NAME (str / dict / EncodingSpec) to its `ArchSpec`.

    The one registry-derived resolver: never shape-sniff a checkpoint, never
    hardcode a plane count or kept-index list — call this by name. Funnels
    through `normalize_encoding_name` so the dict form
    (`Trainer._propagate_encoding_into_config`) also resolves.
    """
    spec = lookup(normalize_encoding_name(name))
    kept = list(spec.kept_plane_indices)
    history = tuple(i for i, src in enumerate(kept) if src in _HISTORY_SRC_PLANES)
    turn_phase = tuple(i for i, src in enumerate(kept) if src in _TURN_PHASE_SRC_PLANES)
    return ArchSpec(
        name=spec.name,
        in_channels=spec.n_planes,
        kept_indices=tuple(kept),
        cur_stone_slot=_kept_slot_of(kept, _CUR_STONE_SRC_PLANE),
        opp_stone_slot=_kept_slot_of(kept, _OPP_STONE_SRC_PLANE),
        k_max=spec.k_max,
        policy_logit_count=spec.policy_logit_count,
        history_planes=history,
        turn_phase_planes=turn_phase,
    )


def cur_stone_slot(spec: Any) -> int:
    """Slice index of the current-player t0 stone plane (source plane 0).

    Always 0 today (every kept set leads with source plane 0), but derived from
    the registry so a future plane-reordering encoding cannot silently break
    the chain-plane recompute that reads `states[:, cur_stone_slot]`.
    """
    return _kept_slot_of(list(spec.kept_plane_indices), _CUR_STONE_SRC_PLANE)


def opp_stone_slot(spec: Any) -> int:
    """Slice index of the opponent t0 stone plane (source plane 8) within the
    encoding's kept-plane order.

    Chain-plane recompute (`_compute_chain_planes`) reads the current- and
    opponent-player t0 stone planes out of a sliced corpus / buffer array. The
    current-player t0 plane (source 0) is always slot 0 (every kept set leads
    with 0); the opponent plane (source 8) sits at slot 4 for the v6 family
    (kept [0,1,2,3,8,...]) but slot 1 for v6_live2 (kept [0,8,16,17]). Derive it
    from the registry instead of hardcoding the v6-only ``4``.
    """
    return _kept_slot_of(list(spec.kept_plane_indices), _OPP_STONE_SRC_PLANE)


def resolve_corpus_path(spec: Any) -> Path:
    """Canonical corpus npz for an encoding.

    Args:
        spec: Any object with a `.name` attribute (EncodingSpec or compatible).

    Returns:
        Repo-relative Path to the corpus npz.

    Raises:
        EncodingRegistryError: if no canonical path is registered for spec.name.
    """
    p = _CORPUS_PATHS.get(spec.name)
    if p is None:
        raise EncodingRegistryError(
            f"No canonical corpus path registered for encoding {spec.name!r}. "
            "Add an entry to _CORPUS_PATHS in hexo_rl/encoding/resolvers.py."
        )
    return Path(p)


def resolve_anchor_path(spec: Any) -> Path:
    """Canonical bootstrap anchor checkpoint for an encoding.

    Args:
        spec: Any object with a `.name` attribute (EncodingSpec or compatible).

    Returns:
        Repo-relative Path to the checkpoint.

    Raises:
        EncodingRegistryError: if no canonical path is registered for spec.name.
    """
    p = _ANCHOR_PATHS.get(spec.name)
    if p is None:
        raise EncodingRegistryError(
            f"No canonical anchor path registered for encoding {spec.name!r}. "
            "Add an entry to _ANCHOR_PATHS in hexo_rl/encoding/resolvers.py."
        )
    return Path(p)


def expand_auto_paths(config: dict[str, Any], spec: Any) -> None:
    """Expand ``<auto>`` literals in *config* using the canonical artifact paths.

    Mutates *config* in-place. Handles both flat top-level keys and the nested
    keys present in variant YAML files.

    Flat keys expanded:
      - ``corpus_npz``       → resolve_corpus_path(spec)
      - ``bootstrap_anchor`` → resolve_anchor_path(spec)

    Nested keys expanded:
      - ``mixing.pretrained_buffer_path``                → resolve_corpus_path(spec)
      - ``eval_pipeline.opponents.bootstrap_anchor.path`` → resolve_anchor_path(spec)

    Only expands when the current value is the literal string ``"<auto>"``.
    """
    # Flat top-level keys (task description canonical form).
    if config.get("corpus_npz") == _AUTO:
        config["corpus_npz"] = str(resolve_corpus_path(spec))
    if config.get("bootstrap_anchor") == _AUTO:
        config["bootstrap_anchor"] = str(resolve_anchor_path(spec))

    # Nested: mixing.pretrained_buffer_path
    mixing = config.get("mixing")
    if isinstance(mixing, dict) and mixing.get("pretrained_buffer_path") == _AUTO:
        mixing["pretrained_buffer_path"] = str(resolve_corpus_path(spec))

    # Nested: eval_pipeline.opponents.bootstrap_anchor.path
    eval_cfg = config.get("eval_pipeline")
    if isinstance(eval_cfg, dict):
        opponents = eval_cfg.get("opponents")
        if isinstance(opponents, dict):
            anchor_cfg = opponents.get("bootstrap_anchor")
            if isinstance(anchor_cfg, dict) and anchor_cfg.get("path") == _AUTO:
                anchor_cfg["path"] = str(resolve_anchor_path(spec))


def resolve_from_config(cfg: Mapping[str, Any] | None) -> EncodingSpec:
    """Return an `EncodingSpec` from a config mapping.

    Accepts BOTH legacy forms:
      - `cfg['encoding'] = "v6w25"`            (string form, §172 A4.5 canonical)
      - `cfg['encoding'] = {'version': 'v6'}`  (mapping form, accepted for
                                                backward-compat)

    Default: `"v6"` if no encoding section present (preserves byte-exact
    pre-§166 behavior).

    §172 A4.5 — scattered-key invariant: if `cfg` has an explicit `encoding`
    key, all of the legacy scattered scalars (`board_size`, `n_planes`,
    `in_channels`, `cluster_window_size`, `cluster_threshold`,
    `legal_move_radius`) must equal the registry spec on the fields where
    the spec is non-None. Disagreement raises `EncodingRegistryError`.
    Legacy configs with NO `encoding` key downgrade the rejection to a
    `DeprecationWarning` and resolve as v6 (silent default).
    """
    if cfg is None:
        return lookup("v6")
    section = cfg.get("encoding")
    explicit_encoding = section is not None
    spec: EncodingSpec
    if section is None:
        spec = lookup("v6")
    elif isinstance(section, str):
        spec = lookup(section)
    elif isinstance(section, Mapping):
        version = section.get("version", "v6")
        if not isinstance(version, str):
            raise EncodingRegistryError(
                f"encoding.version must be a string; got {type(version).__name__}"
            )
        spec = lookup(version)
    else:
        raise EncodingRegistryError(
            f"encoding section must be str or mapping; got {type(section).__name__}"
        )

    if explicit_encoding:
        # Strict: scattered key disagreement is an error. Forces variant
        # configs to declare the encoding once and let the registry decide
        # everything else.
        _check_scattered_keys(cfg, spec)
    else:
        # Lenient: legacy config without explicit encoding section.
        # Accept silently if scattered keys agree with the v6 default;
        # downgrade disagreement to DeprecationWarning so old call sites
        # (e.g. tests, eval scripts pre-A4) keep loading.
        try:
            _check_scattered_keys(cfg, spec)
        except EncodingRegistryError as e:
            warnings.warn(
                f"legacy config without 'encoding' key has scattered keys "
                f"that disagree with v6 default; resolving as v6 anyway. "
                f"Add an explicit `encoding: <name>` declaration. Detail:\n{e}",
                DeprecationWarning,
                stacklevel=2,
            )
    return spec


def detect_encoding_from_state_dict(
    state: Mapping[str, Any],
    ckpt_label: str,
    strict: bool = False,
) -> EncodingSpec | None:
    """Detect a registry encoding from a model state-dict shape + filename.

    Consolidates two near-duplicate implementations (§176 P6):
      - `hexo_rl.training.trainer_ckpt_load._detect_encoding_from_state_dict`
        (lenient — returned None on no match so trainer fell through to
        legacy hparam inference)
      - `hexo_rl.eval.checkpoint_loader.detect_encoding_label`
        (strict — raised ValueError on no match; defaulted in_ch=8 with
        no n_actions probe to v6)

    Detection logic:
      1. Read `inp_w` from state-dict at `trunk.input_conv.conv.weight`
         (preferred — wraps partial-conv) then `trunk.input_conv.weight`
         (legacy).
      2. If `inp_w` is missing or has wrong dims:
         - strict=True → raise ValueError (eval-side message preserved).
         - strict=False → return None.
      3. `in_ch = inp_w.shape[1]`; probe `n_actions` from
         `policy_fc.weight` then `cluster_pool.policy_mlp.2.weight`.
      4. Dispatch:
         - `in_ch=11 and n_actions=625` → v8.
         - `in_ch=8` (BUFFER_CHANNELS):
             * filename hint (`v6w25` or `_w25` substring) OR
               `n_actions=626` → v6w25.
             * `n_actions=362` → v6.
             * Otherwise: strict defaults to v6 (matches eval-side
               line 99 fallback); lenient returns None (matches
               trainer-side fall-through).
         - `in_ch != 8 and != 11`: strict raises ValueError;
           lenient returns None.

    Args:
        state: Model state-dict (key → tensor).
        ckpt_label: Free-form label/path string used for the v6/v6w25
                    filename disambiguator.
        strict: If True, raise ValueError on no canonical match.
                If False, return None.

    Returns:
        Registry `EncodingSpec` (via `lookup(name)`), or None when
        lenient and no match.
    """
    buffer_channels = lookup("v6").n_planes
    inp_w = state.get("trunk.input_conv.conv.weight")
    if inp_w is None:
        inp_w = state.get("trunk.input_conv.weight")
    if inp_w is None or getattr(inp_w, "dim", lambda: 0)() != 4:
        if strict:
            raise ValueError(
                f"checkpoint {ckpt_label} has no trunk.input_conv(.conv)?.weight; "
                "cannot detect encoding"
            )
        return None
    in_ch = int(inp_w.shape[1])
    n_actions: int | None = None
    for k in ("policy_fc.weight", "cluster_pool.policy_mlp.2.weight"):
        w = state.get(k)
        if w is not None and getattr(w, "dim", lambda: 0)() == 2:
            n_actions = int(w.shape[0])
            break
    label = ckpt_label.lower()
    if in_ch == 11 and n_actions == 625:
        return lookup("v8")
    if in_ch == 11 and strict:
        # Eval-side accepts in_ch=11 alone as v8 (no n_actions probe required).
        return lookup("v8")
    # §P5-CT CF-2 — v6tp keeps turn-phase planes 16/17 → in_ch=10, 362 actions,
    # 19×19 single-window. Distinct plane count from the v6 family (8).
    if in_ch == 10 and (n_actions == 362 or "v6tp" in label):
        return lookup("v6tp")
    # §P5-CT H-PLANE fix — v6_live2 keeps only the live-on-both-paths planes
    # [0,8,16,17] → in_ch=4, 362 actions, 19×19 single-window. Distinct plane
    # count from v6 (8) / v6tp (10).
    if in_ch == 4 and (n_actions == 362 or "v6_live2" in label or "live2" in label):
        return lookup("v6_live2")
    if in_ch == buffer_channels:
        if n_actions == 626 or "v6w25" in label or "_w25" in label:
            # Filename hint can override action-count when the head is a
            # PMA variant whose output dim differs; trust operator labels.
            return lookup("v6w25")
        if n_actions == 362:
            return lookup("v6")
        if strict:
            # Eval-side fallback: in_ch=8 with no n_actions match defaults
            # to v6 (preserves checkpoint_loader.py:99 behaviour).
            return lookup("v6")
        return None
    if strict:
        raise ValueError(
            f"checkpoint {ckpt_label}: unsupported in_channels={in_ch} "
            "(expected 4 for v6_live2, 8 for v6/v6w25, 10 for v6tp, 11 for v8)"
        )
    return None


def resolve_from_checkpoint(path: str | Path) -> EncodingSpec:
    """Return an `EncodingSpec` for a saved checkpoint.

    Reads `ckpt['metadata']['encoding_name']` if present. Otherwise
    falls back to `compat.infer_encoding_from_state_dict` and emits a
    `DeprecationWarning` directing to §172 A5 stamping.
    """
    import torch

    d = torch.load(path, map_location="cpu", weights_only=False)
    meta = d.get("metadata") if isinstance(d, dict) else None
    if isinstance(meta, dict) and "encoding_name" in meta:
        name = meta["encoding_name"]
        if not isinstance(name, str):
            raise EncodingRegistryError(
                f"checkpoint {path}: metadata['encoding_name'] is "
                f"{type(name).__name__}, expected str"
            )
        return lookup(name)

    if isinstance(d, dict) and "model_state" in d:
        sd = d["model_state"]
    else:
        sd = d
    if not isinstance(sd, Mapping):
        raise EncodingRegistryError(
            f"checkpoint {path}: cannot extract state-dict for shape inference"
        )
    name = compat.infer_encoding_from_state_dict(sd, str(path))
    warnings.warn(
        f"checkpoint {path} has no metadata['encoding_name']; "
        f"inferred {name!r} from state-dict + filename. Stamp metadata via "
        f"§172 A5 migration script.",
        DeprecationWarning,
        stacklevel=2,
    )
    return lookup(name)


def resolve_encoding_for_eval(
    checkpoint_path: str | Path,
    encoding_override: str | None = None,
) -> EncodingSpec:
    """Resolve an EncodingSpec for an eval/smoke/probe consumer.

    Priority:
      1. ``encoding_override`` (CLI flag) — direct lookup.
      2. ``resolve_from_checkpoint(checkpoint_path)`` — metadata, then
         shape-inference fallback (DeprecationWarning).
      3. Re-raise with a clearer "pass --encoding explicitly" message
         that lists registered encoding names.
    """
    if isinstance(encoding_override, str) and encoding_override:
        # Direct lookup: do NOT touch the checkpoint.
        return lookup(encoding_override)

    try:
        return resolve_from_checkpoint(checkpoint_path)
    except EncodingRegistryError as e:
        registered = sorted(_load_registry())
        raise EncodingRegistryError(
            f"could not auto-detect encoding for checkpoint {checkpoint_path!s}; "
            f"pass --encoding explicitly. Registered encodings: {registered}. "
            f"Underlying error: {e}"
        ) from e


def validate_against_state_dict(
    spec: EncodingSpec, state_dict: Mapping[str, Any]
) -> None:
    """Cross-check spec.policy_logit_count + spec.n_planes against a state-dict.

    Probes a list of common key names for the policy fc and first conv.
    Silently no-ops for keys that don't appear (caller's responsibility
    to know which architecture they hold). Raises `ShapeMismatchError`
    on disagreement.
    """
    pfc = None
    for k in _POLICY_FC_KEYS:
        if k in state_dict:
            pfc = state_dict[k]
            break
    if pfc is not None:
        out_features = int(pfc.shape[0])
        if out_features != spec.policy_logit_count:
            raise ShapeMismatchError(
                f"policy_fc out_features {out_features} != "
                f"spec.policy_logit_count {spec.policy_logit_count} "
                f"for encoding {spec.name!r}"
            )

    conv = None
    for k in _FIRST_CONV_KEYS:
        if k in state_dict:
            conv = state_dict[k]
            break
    if conv is not None:
        in_channels = int(conv.shape[1])
        if in_channels != spec.n_planes:
            raise ShapeMismatchError(
                f"first conv in_channels {in_channels} != "
                f"spec.n_planes {spec.n_planes} for encoding {spec.name!r}"
            )
