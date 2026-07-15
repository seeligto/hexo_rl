"""Encoding-aware checkpoint loader.

Single entry point that detects the encoding version from a checkpoint
file and instantiates the matching `HexTacToeNet`. Used by the
generalized SealBot eval harness (`scripts/run_sealbot_eval.py`) and any
other downstream code that must accept a checkpoint without knowing the
encoding upfront.

Detection priority:
1. Explicit `encoding` field in the checkpoint dict (preferred for new
   checkpoints — pretrain should write this).
2. Filename heuristic: `v6w25` substring → v6w25.
3. `trunk.input_conv.weight` shape:
   - `in_channels == 11` → v8
   - `in_channels == 8`  → v6 (default; v6w25 falls under v6 by shape but is
     disambiguated via the filename heuristic above).

v6w25 shares wire format with v6 (8 planes, K-cluster) but uses a 25×25
cluster window + R=8 perception. Both fact families resolve at the
inference adapter (V6ArgmaxBot, etc.) — the loader only needs to surface
the EncodingSpec correctly so dispatch downstream picks the right bot.

D-EVALGATE G1 (2026-07-03) — ports the trainer-side D-FORENSIC F1 gate to
this eval-path loader. ``load_model_with_encoding`` now accepts an
optional ``declared_encoding`` (the eval-side analogue of a variant's
``encoding:`` declaration) that is reconciled against the checkpoint's OWN
trusted stamp (``metadata['encoding_name']`` then ``config['encoding']``
then top-level ``raw['encoding']``, string or dict form) BEFORE any of the
"Detection priority" fallback above runs — disagreement raises
:class:`DeclaredEncodingMismatchError` naming both sources, and a present
stamp is never silently overridden by shape/filename inference (the
ambiguity class that let the d1m lineage self-play single-window
``v6_live2`` for 272k+ steps while every variant declared multi-window
``v6_live2_ls`` — the two encodings are state-dict-shape-identical, so
shape/filename inference alone cannot disambiguate them). See
``_resolve_ckpt_stamped_encoding`` / ``_check_declared_vs_stamped_encoding``
and ``tests/test_checkpoint_loader_encoding_gate.py``.

D-EVALGATE fix wave (2026-07-03) — two DISTINCT semantics were conflated in
the G1 cut and are now split:
  - ``declared_encoding`` — an ASSERTION ("this checkpoint IS X"). Mismatch
    vs ANY stamp source raises. Use this for a verdict read that must be
    pinned to a known encoding.
  - ``decode_override`` — a deliberate DECODE-TIME cross-decode (the
    D-DECODE program: same net weights, decode under a different window
    geometry, e.g. re-decoding a stale-stamped ``v6_live2`` checkpoint as
    ``v6_live2_ls``). The override ALWAYS wins; a disagreeing stamp is
    logged loudly (``encoding_decode_override``) but never raises. Passing
    both kwargs together raises ``ValueError`` (mutually exclusive).
Malformed encoding values are symmetric on both sides now (declared,
decode_override, and every stamp source all funnel through
``_normalize_or_raise``, which raises a ``ValueError``-lineage error with
context instead of silently downgrading to ``(None, None)``).

NOTE — the trainer-side gate (``trainer_ckpt_load._resolve_checkpoint_encoding``)
also reconciles NUMERIC pins (``board_size`` / ``cluster_window_size`` /
``cluster_threshold``) against the checkpoint's resolved spec. That
reconciliation is deliberately NOT ported here — the eval path never
constructs a model from a bare hparam dict the way the trainer's resume
path does, so there is no second numeric-pin source to disagree with.

Docstring correction: ``declared_encoding=None`` is NOT byte-for-byte
identical to the pre-gate loader for every checkpoint — it is identical
for UNSTAMPED checkpoints (no metadata/config/raw stamp; shape/filename
inference resolves exactly as before) and for STAMP-CONSISTENT checkpoints.
For a checkpoint whose stamp disagrees with what shape/filename inference
would have picked, the stamp now wins (the intended D-FORENSIC F1
correction) — e.g. ``checkpoints/ws3v3_warmstart_200k.pt`` is stamped
``v6_live2_ls`` but pre-gate shape/filename inference would have said
``v6_live2``; this is the known, intended drift artifact.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from hexo_rl.encoding import (
    EncodingRegistryError,
    EncodingSpec,
    detect_encoding_from_state_dict as _registry_detect_from_state_dict,
    lookup as _registry_lookup,
    normalize_encoding_name as _normalize_encoding_name,
)
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.checkpoints import (
    assert_full_gnn_checkpoint_or_raise,
    infer_gnn_hparams_from_state_dict,
    normalize_model_state_dict_keys,
)

try:
    import structlog
    _log = structlog.get_logger()
except ImportError:  # pragma: no cover - structlog is a hard project dep in prod
    import logging
    logging.basicConfig(level=logging.INFO)
    _log = logging.getLogger("hexo_rl.eval.checkpoint_loader")  # type: ignore[assignment]

BUFFER_CHANNELS: int = _registry_lookup("v6").n_planes


class DeclaredEncodingMismatchError(ValueError):
    """Declared encoding disagrees with the checkpoint's own trusted stamp.

    D-EVALGATE G1 — port of the trainer-side D-FORENSIC F1 gate
    (``hexo_rl.training.trainer_ckpt_load``) to the eval path. Mirrors the
    trainer's rule exactly: an explicitly declared encoding (string OR
    ``{"version": ...}``/``{"name": ...}`` dict form both count) that
    disagrees with the checkpoint's own ``metadata['encoding_name']`` or
    ``config['encoding']`` stamp RAISES, naming both sources. Silent
    override in either direction is the exact failure class that let the
    d1m lineage self-play single-window ``v6_live2`` for 272k+ steps while
    every variant declared multi-window ``v6_live2_ls`` (the two encodings
    are state-dict-shape-identical, so shape/filename inference cannot
    disambiguate them — the declared/stamped names are the only truthful
    source).
    """


def _normalize_or_raise(value: Any, *, side: str, source: str) -> str:
    """``normalize_encoding_name`` wrapper that re-raises a malformed value as
    a ``ValueError``-lineage error naming which side + source it came from.

    Symmetric fix (D-EVALGATE fix wave review point 8): the pre-fix loader
    caught this on the STAMP side only and silently downgraded to
    ``(None, None)`` (falling through to shape/filename inference) while a
    malformed ``declared_encoding`` raised the registry's own
    ``EncodingRegistryError`` uncaught. Both sides now raise the same
    ``ValueError``-lineage error with context.
    """
    try:
        return _normalize_encoding_name(value)
    except EncodingRegistryError as exc:
        raise ValueError(
            f"malformed {side} encoding value from {source}: {value!r} ({exc})"
        ) from exc


def _resolve_ckpt_stamped_encoding(
    raw: Any, checkpoint_path: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(encoding_name, source_description)`` from the checkpoint's
    OWN trusted stamp — never from shape/filename inference.

    Priority (mirrors ``trainer_ckpt_load.load_checkpoint``'s
    metadata-then-config precedence, extended with a third source):
      1. ``raw['metadata']['encoding_name']`` (registry source of truth,
         §172 A4.3 / A5 migration schema).
      2. ``raw['config']['encoding']`` (string or ``{'version': ...}``/
         ``{'name': ...}`` dict form — the §172 A4.5 canonical variant
         declaration, as baked into a full checkpoint's own config).
      3. top-level ``raw['encoding']`` (the legacy/promoted-anchor field
         written by ``hexo_rl.training.anchor.save_best_model_atomic`` —
         D-EVALGATE fix wave review point 2: this used to be consumed as a
         priority-1 label BEFORE reconciliation, silently bypassing the
         whole gate. It is now folded in as a third STAMP source, reconciled
         like the other two.).

    Every stamp source that IS present must resolve to the SAME name.
    ``anchor.py`` writes ``metadata['encoding_name']`` and top-level
    ``encoding`` from the same variable, so a disagreement between present
    sources means the checkpoint is internally inconsistent (corrupt, or
    hand-edited) — this raises rather than picking a side.

    Returns ``(None, None)`` when NO stamp source is present — a bare
    weights-only payload (e.g. a bootstrap ``.pt``) carries no
    independently-trustworthy stamp; shape/filename inference is the only
    remaining source and is treated as last-resort (never compared against
    a declared name — see ``load_model_with_encoding``).
    """
    if not isinstance(raw, dict):
        return None, None

    candidates: list[Tuple[str, str]] = []  # (name, source_description), priority order

    metadata = raw.get("metadata")
    if isinstance(metadata, dict) and "encoding_name" in metadata:
        source = f"checkpoint {checkpoint_path!s} metadata['encoding_name']"
        name = _normalize_or_raise(metadata["encoding_name"], side="checkpoint stamp", source=source)
        candidates.append((name, source))

    cfg = raw.get("config")
    if isinstance(cfg, dict) and cfg.get("encoding") is not None:
        source = f"checkpoint {checkpoint_path!s} config['encoding']"
        name = _normalize_or_raise(cfg["encoding"], side="checkpoint stamp", source=source)
        candidates.append((name, source))

    raw_encoding = raw.get("encoding")
    if raw_encoding is not None:
        source = f"checkpoint {checkpoint_path!s} raw['encoding']"
        name = _normalize_or_raise(raw_encoding, side="checkpoint stamp", source=source)
        candidates.append((name, source))

    if not candidates:
        return None, None

    distinct_names = {name for name, _ in candidates}
    if len(distinct_names) > 1:
        listed = "; ".join(f"{src}={name!r}" for name, src in candidates)
        raise ValueError(
            f"checkpoint {checkpoint_path!s}: stamp sources disagree with each "
            f"other: {listed}. A checkpoint's own stamp fields are written from "
            "ONE variable (see hexo_rl.training.anchor.save_best_model_atomic) "
            "so mutual disagreement means the checkpoint is internally "
            "inconsistent — refusing to silently pick a side; re-stamp or "
            "inspect the checkpoint."
        )
    return candidates[0]


def _check_declared_vs_stamped_encoding(
    declared_encoding: Any,
    ckpt_stamp_name: Optional[str],
    ckpt_stamp_source: Optional[str],
) -> Optional[str]:
    """Reconcile a caller-declared encoding against the checkpoint's stamp.

    ``declared_encoding`` counts as an EXPLICIT declaration whenever it is
    not ``None`` — a canonical string form (``"v6_live2_ls"``) or a dict
    form (``{"version": "v6_live2_ls"}``) both qualify (mirrors the
    trainer-side D-FORENSIC F1 fix, which closed the dict-only
    ``isinstance`` check that silently treated the string form as
    "unspecified").

    Returns the declared name (or ``None`` if no declaration was made).
    Raises :class:`DeclaredEncodingMismatchError` on disagreement with a
    present checkpoint stamp — no silent override in either direction.

    CONFRES 6c: the raise-on-conflict DECISION delegates to the ONE shared rule
    ``hexo_rl.config.resolve.encoding.reconcile_declared_vs_stamp`` (design law #1) so this eval
    Surface-B gate and the launch-builder Surface-A resolver apply provably identical semantics.
    This wrapper keeps the eval-specific ``DeclaredEncodingMismatchError`` type + message (callers
    catch it by name); only the comparison logic is single-sourced.
    """
    if declared_encoding is None:
        return None
    declared_name = _normalize_or_raise(declared_encoding, side="declared", source="declared_encoding")
    from hexo_rl.config.resolve.encoding import EncodingConflictError as _EncConflict
    from hexo_rl.config.resolve.encoding import reconcile_declared_vs_stamp as _reconcile
    try:
        _reconcile(declared_name, ckpt_stamp_name, stamp_source=ckpt_stamp_source or "checkpoint")
    except _EncConflict as exc:
        raise DeclaredEncodingMismatchError(
            f"Encoding version disagrees: declared_encoding={exc.declared!r} vs "
            f"{ckpt_stamp_source}={exc.stamp!r}. The checkpoint stamp would "
            "silently route eval under the wrong action-space/window geometry "
            "(v6_live2 vs v6_live2_ls are state-dict-shape-identical — shape "
            "inference cannot catch this). Re-stamp the checkpoint (weights-only "
            "strip + metadata['encoding_name'], e.g. scripts/make_ws3v3_warmstart.py) "
            "or fix the declared encoding; refusing to silently override either "
            "direction."
        ) from exc
    return declared_name


def validate_arch_against_spec(
    in_channels: int, policy_logit_count: int, spec: EncodingSpec,
) -> None:
    """Raise if the arch inferred from a checkpoint disagrees with the registry spec
    resolved by NAME (D-EVALFOUND C1 — replaces silent shape-sniff trust).

    The §D-FOUNDING driver had to sidestep a loader that hardcoded v6 (8-plane) and
    crashed/mis-loaded the 4-plane v6_live2 net. Resolving the spec by name upstream
    is necessary but not sufficient — this guard makes a plane-count / policy-width
    mismatch a LOUD error instead of a silent corruption.
    """
    if in_channels != spec.n_planes:
        raise ValueError(
            f"checkpoint in_channels={in_channels} != spec '{spec.name}' "
            f"n_planes={spec.n_planes}; encoding/arch mismatch (registry-by-name guard)"
        )
    if policy_logit_count != spec.policy_logit_count:
        raise ValueError(
            f"checkpoint policy logit count={policy_logit_count} != spec '{spec.name}' "
            f"policy_logit_count={spec.policy_logit_count}; encoding/arch mismatch"
        )


def _strip_compile_prefixes(state: dict) -> dict:
    """Strip `_orig_mod.` / `module.` wrapper prefixes from state-dict keys.

    Lighter than `normalize_model_state_dict_keys` — does NOT add
    `tower.*` ↔ `trunk.tower.*` aliases. Aliasing breaks strict-load on
    v8 checkpoints whose state dicts already carry both.
    """
    prefixes = ("_orig_mod.", "module.")
    out: dict = {}
    for key, value in state.items():
        norm_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if norm_key.startswith(prefix):
                    norm_key = norm_key[len(prefix):]
                    changed = True
        out[norm_key] = value
    return out



def detect_encoding_label(ckpt_path: Path, state: dict) -> str:
    """Return the encoding label string: 'v6', 'v6w25', or 'v8'.

    Pure-detection helper — does not load model. Useful for tests and the
    inference-method dispatcher when only the label is needed.

    §176 P6: thin shim over
    ``hexo_rl.encoding.resolvers.detect_encoding_from_state_dict``
    (strict=True). The shared helper raises ValueError on missing keys
    and unsupported in_channels, defaults `in_ch=8` with no n_actions
    probe to v6 (preserves the previous fallback at this site), and
    handles the v6w25 filename-substring disambiguator via the
    ``ckpt_label`` parameter (we pass the basename only to keep the
    historical scoping — full path could match `v6w25` in a parent dir).
    """
    spec = _registry_detect_from_state_dict(
        state, ckpt_path.name, strict=True,
    )
    # strict=True guarantees a non-None spec.
    assert spec is not None
    return spec.name


def load_model_with_encoding(
    ckpt_path: str | Path,
    device: torch.device,
    declared_encoding: Any | None = None,
    require_encoding_source: bool = False,
    decode_override: Any | None = None,
) -> Tuple[torch.nn.Module, EncodingSpec, str]:
    """Load checkpoint, detect encoding, return (model, spec, label).

    The label matches ``spec.name`` (registry-by-name). Use the label to
    drive bot dispatch; use the spec for numeric constants.

    Args:
        declared_encoding: D-EVALGATE G1 — an optional caller-declared
            encoding (string form ``"v6_live2_ls"`` OR dict form
            ``{"version": "v6_live2_ls"}``/``{"name": ...}``; both count as
            an EXPLICIT declaration, mirroring the trainer-side
            D-FORENSIC F1 fix). This is an ASSERTION ("this checkpoint IS
            X") reconciled against the checkpoint's OWN trusted stamp
            (``metadata['encoding_name']`` then ``config['encoding']`` then
            top-level ``raw['encoding']``) — disagreement raises
            :class:`DeclaredEncodingMismatchError` naming both sources.
            Agreement (or no stamp to compare against) makes the declared
            name authoritative for ``label`` — it is NEVER silently
            overridden by shape/filename inference. Mutually exclusive with
            ``decode_override`` (passing both raises ``ValueError``).
        require_encoding_source: D-EVALGATE G1 part 3 — pre-flight hard
            gate for the "no declared encoding + no checkpoint stamp"
            hole (§D-FORENSIC F1 follow-up: ~12/61 variants carry no
            `encoding:` key). When True, raise loudly if none of
            ``declared_encoding``, ``decode_override``, or the checkpoint's
            own stamp resolves a name, instead of silently falling through
            to shape/filename inference. Default False — fully backward
            compatible; opt in at call sites that carry variant/config
            context and want to refuse an unstamped, undeclared load
            outright.
        decode_override: D-EVALGATE fix wave — a deliberate DECODE-TIME
            cross-decode (the D-DECODE program: same net weights, decode
            under a different window geometry than the checkpoint's own
            stamp — e.g. re-decoding a stale-stamped ``v6_live2`` checkpoint
            as ``v6_live2_ls`` to exercise the multi-window no-drop action
            space). The override is ALWAYS authoritative for ``label`` — a
            disagreeing stamp is logged loudly
            (``encoding_decode_override``, naming the checkpoint path, the
            stamp, and the decode-as name) but NEVER raises; this is the
            sanctioned escape hatch the exploit_probe.py /
            run_sealbot_eval.py / gumbel_greedy_bot.py runbooks depend on.
            Mutually exclusive with ``declared_encoding``.
    """
    if declared_encoding is not None and decode_override is not None:
        raise ValueError(
            "load_model_with_encoding: declared_encoding and decode_override "
            "are mutually exclusive — declared_encoding is an ASSERTION "
            "(raises on stamp disagreement) while decode_override is a "
            "deliberate cross-decode (never raises on stamp disagreement, "
            "only logs). Pass exactly one."
        )

    ckpt_path = Path(ckpt_path)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state: dict = raw
    if isinstance(raw, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in raw and isinstance(raw[key], dict):
                state = raw[key]
                break

    # D-EVALGATE G1 — port of the trainer-side D-FORENSIC F1 gate. Resolve
    # the checkpoint's OWN trusted stamp (never shape/filename inference)
    # and reconcile it against any caller-declared encoding BEFORE the
    # shape-sniff fallback below ever runs, mirroring
    # `trainer_ckpt_load.load_checkpoint`'s metadata-then-config
    # precedence and its "refuse to silently override either direction"
    # invariant. Fix wave: the old priority-1 `raw.get("encoding")`
    # shortcut (review point 2 — a silent bypass of the whole gate) is
    # gone; top-level `raw['encoding']` is now the third STAMP source
    # inside `_resolve_ckpt_stamped_encoding`, reconciled like the rest.
    ckpt_stamp_name, ckpt_stamp_source = _resolve_ckpt_stamped_encoding(raw, ckpt_path)
    declared_name = _check_declared_vs_stamped_encoding(
        declared_encoding, ckpt_stamp_name, ckpt_stamp_source,
    )

    override_name: Optional[str] = None
    if decode_override is not None:
        override_name = _normalize_or_raise(
            decode_override, side="decode_override", source="decode_override",
        )
        log_fields = {
            "checkpoint": str(ckpt_path),
            "ckpt_stamp": ckpt_stamp_name,
            "ckpt_stamp_source": ckpt_stamp_source,
            "decode_as": override_name,
        }
        if ckpt_stamp_name is not None and override_name != ckpt_stamp_name:
            _log.warning("encoding_decode_override", **log_fields)
        else:
            _log.info("encoding_decode_override", **log_fields)

    if (
        require_encoding_source
        and declared_name is None
        and ckpt_stamp_name is None
        and override_name is None
    ):
        raise DeclaredEncodingMismatchError(
            f"checkpoint {ckpt_path!s}: no declared encoding, no decode "
            "override, and no usable checkpoint stamp (no "
            "metadata['encoding_name'], no config['encoding'], no "
            "raw['encoding']). require_encoding_source=True refuses to "
            "silently fall through to shape/filename inference — pass an "
            "explicit declared_encoding/decode_override or re-stamp the "
            "checkpoint."
        )
    # decode_override ALWAYS wins when given (declared_encoding and
    # decode_override are mutually exclusive, enforced above).
    stamped_or_declared = override_name or ckpt_stamp_name or declared_name

    # v6 path uses normalize_model_state_dict_keys (handles tower↔trunk.tower
    # aliasing for legacy v7full / v6 / v7 checkpoints saved without the
    # `trunk.` prefix). v8 checkpoints already use `trunk.tower.*` and
    # break under aliasing — they get the lighter strip-prefixes-only path.
    inp_w = state.get("trunk.input_conv.weight")
    if inp_w is None:
        inp_w = state.get("trunk.input_conv.conv.weight")
    if inp_w is None:
        inp_w = state.get("_orig_mod.trunk.input_conv.weight")
    if inp_w is None:
        inp_w = state.get("_orig_mod.trunk.input_conv.conv.weight")
    if inp_w is None:
        # Fall back: try after light strip — may surface a v8 checkpoint
        # under a wrapper prefix.
        light = _strip_compile_prefixes(state)
        inp_w = light.get("trunk.input_conv.weight") \
            or light.get("trunk.input_conv.conv.weight")
    if inp_w is not None and int(inp_w.shape[1]) == 11:
        state = _strip_compile_prefixes(state)
    else:
        state = normalize_model_state_dict_keys(state)
    # Registry-by-name priority: an explicit stamp/declaration is
    # authoritative and is NEVER overridden by shape/filename inference —
    # the latter runs only as a last resort when neither side names an
    # encoding (D-EVALGATE G1 TDD requirement).
    label = stamped_or_declared or detect_encoding_label(ckpt_path, state)
    try:
        _registry_lookup(label)
    except EncodingRegistryError as exc:
        # WP-4 review finding 2 — an UNKNOWN *checkpoint stamp* label (never
        # a declared/override name, which are caller assertions and must
        # stay loud) falls through to state-dict detection instead of dying
        # generic. Concrete case: the banked BC-prefit artifacts
        # (`checkpoints/probes/gnn_bc/gnn_bc_*.pt`) carry a top-level
        # `encoding: "strix_axis_graph"` stamp (`train_bc.py`) that was
        # never registered — the generic "unknown encoding label" raise
        # short-circuited BEFORE `_build_gnn_model`'s actionable
        # `assert_full_gnn_checkpoint_or_raise` diagnostic could fire.
        # Detection is shape-marker-driven (`detect_encoding_label`,
        # strict), so a garbage state dict still dies loud here.
        if (
            declared_name is None
            and override_name is None
            and label == ckpt_stamp_name
        ):
            _log.warning(
                "unknown_ckpt_encoding_stamp_falling_through_to_detection",
                checkpoint=str(ckpt_path),
                stamp=label,
                stamp_source=ckpt_stamp_source,
            )
            label = detect_encoding_label(ckpt_path, state)
        else:
            raise ValueError(
                f"checkpoint {ckpt_path}: unknown encoding label {label!r}"
            ) from exc

    if label == "gnn_axis_v1":
        # GNN-integration WP-4 (C4) — graph representation, no plane/pass-slot
        # geometry; _build_model_from_spec dispatches to _build_gnn_model.
        spec = _registry_lookup("gnn_axis_v1")
    elif label == "v8":
        spec = _registry_lookup("v8")
    elif label == "v6w25":
        spec = _registry_lookup("v6w25")
    elif label == "v6tp":
        # §P5-CT CF-2 — 10-plane (v6 + turn-phase 16/17); builds like v6
        # (min_max head, in_channels read from the conv weight = 10).
        spec = _registry_lookup("v6tp")
    elif label == "v6_live2":
        # §P5-CT H-PLANE fix — 4-plane (v6 minus history); builds like v6
        # (min_max head, in_channels read from the conv weight = 4).
        spec = _registry_lookup("v6_live2")
    elif label == "v6_live2_ls":
        # D-EVALGATE G1 — multi-window legal-set sibling of v6_live2.
        # State-dict-shape-IDENTICAL to v6_live2 (same 4-plane/362 net) so
        # this label is only ever reached via an explicit declared_encoding
        # or a checkpoint's own metadata/config stamp — never via
        # shape/filename inference (see `_resolve_ckpt_stamped_encoding`).
        spec = _registry_lookup("v6_live2_ls")
    else:
        spec = _registry_lookup("v6")
    model = _build_model_from_spec(state, spec)

    model.to(device)
    model.eval()
    return model, spec, label


def _build_model_from_spec(state: dict, spec: EncodingSpec) -> torch.nn.Module:
    """Unified entry point — dispatches to the per-family builder.

    Branch:
      - ``representation="graph"`` → ``_build_gnn_model`` (WP-4 / C4 —
                                  MUST be checked before ``has_pass_slot``:
                                  the graph encoding has ``has_pass_slot=
                                  True`` too, so a naive has_pass_slot-only
                                  dispatch would mis-route it into
                                  ``_build_min_max_model``, which reads
                                  ``trunk.input_conv.weight`` — absent on a
                                  GnnNet state dict).
      - ``has_pass_slot=True``  → ``_build_min_max_model`` (v6 / v6w25 /
                                  v7full / v7 / v7e30 / v7mw family;
                                  min_max policy head with optional
                                  pma / pma_global pool variants and the
                                  §170 P3 gpool_bias side-branch).
      - ``has_pass_slot=False`` → ``_build_kata_model`` (v8 / v8_canvas_realness
                                  family; KataGoPolicyHead + per-block gpool +
                                  optional PartialConv2d canvas_realness wrap).

    Cycle 3 Wave 8 Batch D (2026-05-17): renamed from
    ``_build_v6_model`` / ``_build_v8_model`` (GENERICISE #4 fold);
    bodies preserved architecturally distinct because feature-detection
    + ``strict`` load policy differ per family.
    """
    if getattr(spec, "representation", "grid") == "graph":
        return _build_gnn_model(state, spec)
    if spec.has_pass_slot:
        return _build_min_max_model(state, spec)
    return _build_kata_model(state, spec)


def _build_gnn_model(state: dict, spec: EncodingSpec):
    """C4 graph builder — selected when ``spec.representation == "graph"``
    (WP-4, contract node 11e).

    Ground-truths ``hidden``/``num_layers``/``policy_hidden``/
    ``value_hidden`` from the checkpoint's OWN tensor shapes
    (``infer_gnn_hparams_from_state_dict``, single-sourced with the C7
    resume branch, `hexo_rl.training.trainer_ckpt_load`) rather than
    trusting a hardcoded default — a probe-284k checkpoint and any future
    differently-scaled GNN both load correctly.

    Landed-verify (C7 red-team demand — representation+policy coverage,
    not value-only): mirrors the E1 ``torch.allclose`` post-load guard
    (``_build_min_max_model``, above) across ALL three submodules
    (representation, policy_head, value_head), not just the dist65 bins.
    ``strict=True`` on ``load_state_dict`` already makes a partial/dropped
    load impossible for THIS state dict (unlike the CNN's
    tower.*-duplicate-tolerant ``strict=False``) — the allclose pass below
    is a belt-and-suspenders parity check (F1 defense-in-depth), not the
    only defense.
    """
    from hexo_rl.model.gnn_net import GnnNet

    assert_full_gnn_checkpoint_or_raise(state, checkpoint_label="gnn checkpoint")

    node_feat_dim = getattr(spec, "node_feat_dim", None)
    edge_feat_dim = getattr(spec, "edge_feat_dim", None)
    if node_feat_dim is None or edge_feat_dim is None:
        raise ValueError(
            f"_build_gnn_model: encoding {spec.name!r} declares representation="
            "'graph' but is missing node_feat_dim/edge_feat_dim (schema-v4 "
            "graph fields, engine/src/encoding/registry.toml)."
        )
    inferred = infer_gnn_hparams_from_state_dict(state)
    # node_feat_dim/edge_feat_dim: cross-check the checkpoint's own shape
    # against the registry spec — a mismatch means the checkpoint does not
    # actually speak the declared encoding (D-EVALFOUND C1 class guard,
    # mirrors validate_arch_against_spec above).
    ckpt_node_feat_dim = inferred.get("node_feat_dim", node_feat_dim)
    ckpt_edge_feat_dim = inferred.get("edge_feat_dim", edge_feat_dim)
    if ckpt_node_feat_dim != node_feat_dim or ckpt_edge_feat_dim != edge_feat_dim:
        raise ValueError(
            f"checkpoint node_feat_dim/edge_feat_dim={ckpt_node_feat_dim}/"
            f"{ckpt_edge_feat_dim} != spec {spec.name!r} "
            f"node_feat_dim/edge_feat_dim={node_feat_dim}/{edge_feat_dim}; "
            "encoding/arch mismatch (registry-by-name guard)."
        )
    model = GnnNet(
        in_dim=int(node_feat_dim),
        hidden=int(inferred.get("gnn_hidden", 128)),
        num_layers=int(inferred.get("gnn_num_layers", 4)),
        edge_dim=int(edge_feat_dim),
        policy_hidden=int(inferred.get("gnn_policy_hidden", 128)),
        value_hidden=int(inferred.get("gnn_value_hidden", 32)),
        n_value_bins=int(inferred.get("n_value_bins", 65)),
    )
    model.load_state_dict(state, strict=True)

    # Landed-verify (C7 red-team demand): representation + policy + dist65
    # value tensors, not value-only (E1 precedent, `_build_min_max_model`
    # above). strict=True already guarantees full landing for this state
    # dict (no legacy tower.*-alias tolerance exists on GnnNet); this is a
    # belt-and-suspenders parity re-check, not the only defense.
    # Scope (by design, WP-4 review finding 4): this verifies the LOAD
    # landed what the file said — it does NOT validate the file's own
    # contents; shape-preserving corruption (e.g. a transposed SQUARE
    # tensor) inside the checkpoint passes both strict-load and allclose.
    reloaded = model.state_dict()
    for key, src in state.items():
        dst = reloaded.get(key)
        if dst is None:
            continue
        if not torch.allclose(dst, src.to(device=dst.device, dtype=dst.dtype)):
            raise RuntimeError(
                f"gnn checkpoint: landed-verify FAILED for {key!r} "
                "(load_state_dict did not land this tensor)."
            )
    return model


def _reject_graph_shaped_state_for_grid_spec(state: dict, spec: EncodingSpec, missing_key: str) -> None:
    """WP-4 fix pass (review finding 5) — reverse-F1 diagnosability.

    A GRID spec (declared/stamped/inferred) handed a GRAPH-shaped state dict
    used to die with a raw ``KeyError: 'trunk.input_conv.weight'`` — loud,
    but zero hint that the state dict is a GNN net mis-declared as grid.
    Raise the named ``RepresentationMismatch`` family error instead when the
    graph marker is present; a genuinely-malformed grid state dict keeps its
    original ``KeyError`` behavior (the caller's indexing raises next).
    """
    from hexo_rl.encoding._probes import GNN_GRAPH_MARKER_KEY
    from hexo_rl.model.build_net import RepresentationMismatch

    if missing_key not in state and GNN_GRAPH_MARKER_KEY in state:
        raise RepresentationMismatch(
            f"encoding {spec.name!r} (representation='grid') declared for a "
            f"GRAPH-shaped state dict — no {missing_key!r}, but the GNN "
            f"marker {GNN_GRAPH_MARKER_KEY!r} is present. This checkpoint "
            "is a GnnNet (graph representation, encoding 'gnn_axis_v1'); "
            "fix the declared/stamped encoding instead of loading it as a "
            "grid CNN."
        )


def _build_min_max_model(state: dict, spec: EncodingSpec) -> HexTacToeNet:
    _reject_graph_shaped_state_for_grid_spec(state, spec, "trunk.input_conv.weight")
    inp_w = state["trunk.input_conv.weight"]
    filters = int(inp_w.shape[0])
    in_channels = int(inp_w.shape[1])
    block_indices = sorted({
        int(k.split(".")[2]) for k in state.keys()
        if k.startswith("trunk.tower.") and len(k.split(".")) >= 4
    })
    res_blocks = max(block_indices) + 1 if block_indices else 12
    # §169 A2 / A3 — detect pool type from state dict.
    #   global_encoder.* + cluster_pool.global_gate ⇒ pma_global (A3).
    #   cluster_pool.* without global branch ⇒ pma (A2).
    #   Otherwise ⇒ min_max (A1, default).
    has_global_encoder = any(k.startswith("global_encoder.") for k in state)
    has_cluster_pool = any(k.startswith("cluster_pool.") for k in state)
    if has_global_encoder:
        pool_type = "pma_global"
    elif has_cluster_pool:
        pool_type = "pma"
    else:
        pool_type = "min_max"
    # §170 P3 — detect the gpool-bias side-branch via state-dict keys. Only
    # valid in tandem with pool_type='min_max' (the constructor enforces
    # this); presence of `gpool_bias_branch.*` keys is the unambiguous flag.
    gpool_bias_active = any(
        k.startswith("gpool_bias_branch.") for k in state
    )
    if gpool_bias_active and pool_type != "min_max":
        raise ValueError(
            "checkpoint has both gpool_bias_branch.* and "
            f"cluster_pool.* keys (pool_type={pool_type!r}); the side-branch "
            "is A1-only (pool_type='min_max'). Inspect the checkpoint."
        )
    # D-EVALFOUND C1 — registry-by-name guard: the spec was resolved by NAME upstream
    # (load_model_with_encoding); validate the shape-sniffed arch agrees with it so a
    # plane-count / policy-width mismatch is loud, not a silent corruption.
    policy_w = state.get("policy_fc.weight")
    policy_logit_count = int(policy_w.shape[0]) if policy_w is not None else spec.policy_logit_count
    validate_arch_against_spec(in_channels, policy_logit_count, spec)
    # E1 C1 — detect distributional value head from state dict.  A dist65
    # checkpoint carries `value_fc2_bins.weight`; a scalar checkpoint does
    # not.  Without this detection the ctor defaults to scalar, strict=False
    # silently drops the trained bin weights, and every from-disk instrument
    # (promotion gate, SealBot, probes) runs a RANDOM value head.
    bins_w = state.get("value_fc2_bins.weight")
    if bins_w is not None:
        value_head_type = "dist65"
        n_value_bins = int(bins_w.shape[0])
    else:
        value_head_type = "scalar"
        n_value_bins = 65  # default; unused for scalar head
    model = HexTacToeNet(
        board_size=spec.board_size,
        in_channels=in_channels,
        filters=filters,
        res_blocks=res_blocks,
        encoding=spec.name,
        pool_type=pool_type,
        gpool_bias_active=gpool_bias_active,
        value_head_type=value_head_type,
        n_value_bins=n_value_bins,
    )
    # strict=False because v6 / v6w25 checkpoints may carry tower.* duplicates
    # left over from older save formats (see eval_pipeline._load_anchor_model).
    model.load_state_dict(state, strict=False)
    # E1 C1 post-load guard: for a dist checkpoint, assert the bin weights
    # actually landed (not silently dropped).  strict=False is kept for the
    # tower.* duplicate reason above, so we verify explicitly.
    if value_head_type == "dist65":
        assert model.value_fc2_bins is not None, (
            "Internal error: dist65 HexTacToeNet built without value_fc2_bins"
        )
        loaded_w = model.value_fc2_bins.weight.data
        if not torch.allclose(loaded_w, bins_w.to(device=loaded_w.device).to(dtype=loaded_w.dtype)):
            raise RuntimeError(
                "dist65 checkpoint: value_fc2_bins.weight was NOT loaded "
                "(post-load mismatch despite strict=False). "
                "Inspect the checkpoint state dict for key conflicts."
            )
    return model


def _build_kata_model(state: dict, spec: EncodingSpec) -> HexTacToeNet:
    # §169 A4 — under canvas_realness the trunk-entry conv is wrapped in a
    # PartialConv2d, so the weight key shifts from `trunk.input_conv.weight`
    # to `trunk.input_conv.conv.weight`. Detection is unambiguous because
    # the regular Conv2d path never has a `.conv.weight` sub-key.
    canvas_realness = "trunk.input_conv.conv.weight" in state
    inp_w_key = (
        "trunk.input_conv.conv.weight" if canvas_realness
        else "trunk.input_conv.weight"
    )
    _reject_graph_shaped_state_for_grid_spec(state, spec, inp_w_key)
    inp_w = state[inp_w_key]
    filters = int(inp_w.shape[0])
    in_channels = int(inp_w.shape[1])
    if in_channels != 11:
        raise ValueError(
            f"v8 checkpoint expects in_channels=11; got {in_channels}"
        )
    block_indices = sorted({
        int(k.split(".")[2]) for k in state.keys()
        if k.startswith("trunk.tower.") and len(k.split(".")) >= 4
    })
    res_blocks = max(block_indices) + 1 if block_indices else 0
    gpool_indices = sorted({
        i for i in block_indices
        if f"trunk.tower.{i}.conv1.conv1g.weight" in state
    })
    head_use_gpool = "policy_head.conv1g.weight" in state
    model = HexTacToeNet(
        board_size=spec.board_size,
        in_channels=in_channels,
        filters=filters,
        res_blocks=res_blocks,
        encoding=spec.name,
        gpool_indices=gpool_indices if gpool_indices else None,
        head_use_gpool=head_use_gpool,
        canvas_realness=canvas_realness,
    )
    model.load_state_dict(state, strict=True)
    return model
