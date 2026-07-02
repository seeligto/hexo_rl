#!/usr/bin/env python
"""D-WS3V3 V0 — strip the 200k anchor to a WEIGHTS-ONLY warm-start checkpoint.

v2's confound: `--resume checkpoint_00200000.pt` is a FULL checkpoint (carries
`config`/`optimizer_state`/`scaler_state`), so `trainer_ckpt_load.load_checkpoint`
classifies it `is_full_ckpt=True` and the resumed run inherits the anchor's LR
scheduler STATE — the 1M-step cosine schedule, effective LR 1.8569e-3 at step
200k, not the variant's `lr: 1.0e-4` (RESUME_CHECKPOINT_OWNED_KEYS,
`hexo_rl/training/orchestrator.py:233-252`, includes `lr`/`lr_schedule`/
`eta_min`/`min_lr`/`total_steps`/`scheduler_t_max`). A full-checkpoint resume
CANNOT pin the LR — the only way is to strip the checkpoint down to a
weights-only payload BEFORE the resume, so `is_full_ckpt` reads False
(`trainer_ckpt_load.py:328-336` requires model_state + config + optimizer_state
+ scaler_state + step ALL present) and the variant's `lr`/`lr_schedule` win via
`fallback_config`.

Output: EXACTLY {'model_state', 'metadata', 'step'} — no `config`, no
`optimizer_state`, no `scaler_state` (any one of them present re-activates
full-resume semantics; `config` alone would also make the checkpoint's config
win over the variant, trainer_ckpt_load.py:333-334). `metadata` is kept
otherwise VERBATIM except for `encoding_name`, which is RE-STAMPED to
`--encoding-name` (default `v6_live2_ls`, see FIX1 below) —
`metadata['encoding_name']` drives registry resolution ahead of shape
inference (trainer_ckpt_load.py:361-398).

Usage:
  python scripts/make_ws3v3_warmstart.py \\
      --in reports/d_decide_2026-06-24/checkpoints/checkpoint_00200000.pt \\
      --out checkpoints/ws3v3_warmstart_200k.pt \\
      --encoding-name v6_live2_ls   # default; pass '' to keep the source stamp verbatim

D-WS3V3 FIX1 (BLOCKER — encoding, 2026-07-02): the source checkpoint above
carries STALE `metadata['encoding_name']='v6_live2'` (single-window). Every
downstream consumer prefers checkpoint metadata unconditionally
(`hexo_rl/training/trainer_ckpt_load.py`) and back-propagates the
ckpt-resolved encoding into the variant's combined config
(`hexo_rl/training/orchestrator.py` init_trainer), which `hexo_rl/selfplay/
pool.py` resolves for the Rust self-play runner. Left uncorrected, all three
D-WS3V3 arms would self-play single-window v6_live2 despite the variants
declaring `v6_live2_ls` (multi-window legal-set — the whole point of the
injection's off-window routing). `--encoding-name` re-stamps
`metadata['encoding_name']` in the STRIPPED payload to the workflow's
intended encoding, but ONLY after asserting the override shares the
original's WIRE SIGNATURE (`n_planes`, `board_size`, `policy_logit_count`,
`has_pass_slot`, `sym_table_id` — the tuple that determines whether the
saved model weights load no-reshape; mirrors the Rust
`RegistrySpec::wire_signature()` used for HEXB replay-buffer cross-encoding
checks). v6_live2 vs v6_live2_ls share this tuple exactly (both
`(4, 19, 362, True, "size_19")`) — the two encodings differ only in
self-play/decode MECHANICS (cluster_window_size, is_multi_window,
value_pool, policy_pool), not in anything the model weights care about. A
genuinely different encoding (e.g. v6, v8) would NOT share the tuple and the
script refuses to stamp it.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hexo_rl.encoding import (  # noqa: E402
    lookup as _registry_lookup,
    EncodingRegistryError as _EncodingRegistryError,
)

STRIPPED_KEYS = ("model_state", "metadata", "step")
FORBIDDEN_KEYS = ("config", "optimizer_state", "scaler_state")

DEFAULT_IN = REPO_ROOT / "reports" / "d_decide_2026-06-24" / "checkpoints" / "checkpoint_00200000.pt"
DEFAULT_OUT = REPO_ROOT / "checkpoints" / "ws3v3_warmstart_200k.pt"
DEFAULT_ENCODING_NAME = "v6_live2_ls"


def _wire_signature(spec: Any) -> tuple:
    """Shape-relevant tuple a saved checkpoint's weights actually depend on.

    Mirrors `engine::encoding::RegistrySpec::wire_signature()`
    (`engine/src/encoding/spec/mod.rs`) — every OTHER registry field (cluster
    window/threshold, is_multi_window, value_pool, policy_pool, ...) affects
    self-play/decode mechanics but not the stored tensor shapes, so it is
    deliberately excluded here.
    """
    return (
        spec.n_planes,
        spec.board_size,
        spec.policy_logit_count,
        spec.has_pass_slot,
        spec.sym_table_id,
    )


def restamp_encoding(metadata: Dict[str, Any], new_encoding_name: str) -> Dict[str, Any]:
    """Return a NEW metadata dict with `encoding_name` re-stamped to
    `new_encoding_name`, after validating both the original and the override
    resolve in the registry AND share a wire signature.

    Raises ValueError (loud, both names + signatures in the message) on any
    of: original name missing/unresolvable, override name unresolvable,
    wire-signature mismatch.
    """
    orig_name = metadata.get("encoding_name")
    if not isinstance(orig_name, str):
        raise ValueError(
            f"metadata['encoding_name'] missing or non-str ({orig_name!r}); "
            "cannot validate a wire-signature-preserving re-stamp without a "
            "known original encoding."
        )
    try:
        orig_spec = _registry_lookup(orig_name)
    except _EncodingRegistryError as exc:
        raise ValueError(f"original encoding_name {orig_name!r} does not resolve in the registry: {exc}") from exc
    try:
        new_spec = _registry_lookup(new_encoding_name)
    except _EncodingRegistryError as exc:
        raise ValueError(f"override encoding_name {new_encoding_name!r} does not resolve in the registry: {exc}") from exc

    orig_sig = _wire_signature(orig_spec)
    new_sig = _wire_signature(new_spec)
    if orig_sig != new_sig:
        raise ValueError(
            f"refusing to re-stamp: {orig_name!r} wire_signature={orig_sig} != "
            f"{new_encoding_name!r} wire_signature={new_sig} "
            "(n_planes, board_size, policy_logit_count, has_pass_slot, sym_table_id) "
            "— the override would silently change the shapes the saved weights "
            "load against. Load a checkpoint matching the target encoding, or "
            "pick an override that shares the wire signature."
        )

    new_metadata = dict(metadata)
    new_metadata["encoding_name"] = new_encoding_name
    print(f"  encoding stamp: {orig_name!r} -> {new_encoding_name!r} (wire_signature {new_sig} unchanged)")
    return new_metadata


def strip_checkpoint(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Return a NEW dict carrying exactly STRIPPED_KEYS, verbatim values.

    Raises if any required key is missing (a source checkpoint that is
    already weights-only, or from an unexpected schema, must not silently
    produce a truncated warm-start).
    """
    missing = [k for k in STRIPPED_KEYS if k not in ckpt]
    if missing:
        raise ValueError(f"source checkpoint missing required key(s): {missing}")
    return {k: ckpt[k] for k in STRIPPED_KEYS}


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN,
                     help=f"source FULL checkpoint (default: {DEFAULT_IN})")
    ap.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT,
                     help=f"destination weights-only warm-start (default: {DEFAULT_OUT})")
    ap.add_argument("--force", action="store_true",
                     help="overwrite --out if it already exists")
    ap.add_argument(
        "--encoding-name", dest="encoding_name", default=DEFAULT_ENCODING_NAME,
        help=(
            f"re-stamp metadata['encoding_name'] to this registry name "
            f"(default: {DEFAULT_ENCODING_NAME!r} — the D-WS3V3 multi-window "
            "legal-set encoding the variants declare). Pass '' (empty string) "
            "to keep the source checkpoint's stamp verbatim, unvalidated. "
            "The override must share the original's WIRE SIGNATURE "
            "(n_planes/board_size/policy_logit_count/has_pass_slot/"
            "sym_table_id) — a mismatch is a loud error, not a silent "
            "override."
        ),
    )
    args = ap.parse_args()

    if not args.in_path.exists():
        print(f"ERROR: source checkpoint not found: {args.in_path}", file=sys.stderr)
        sys.exit(2)
    if args.out_path.exists() and not args.force:
        print(f"ERROR: {args.out_path} already exists; pass --force to overwrite.", file=sys.stderr)
        sys.exit(2)

    print(f"Loading: {args.in_path}", file=sys.stderr)
    src = torch.load(args.in_path, map_location="cpu", weights_only=False)
    if not isinstance(src, dict):
        print(f"ERROR: unsupported checkpoint payload type: {type(src)!r}", file=sys.stderr)
        sys.exit(2)

    stripped = strip_checkpoint(src)

    if args.encoding_name:
        try:
            stripped["metadata"] = restamp_encoding(stripped["metadata"], args.encoding_name)
        except ValueError as exc:
            print(f"ERROR: encoding re-stamp refused: {exc}", file=sys.stderr)
            sys.exit(2)
    else:
        print(
            f"  encoding stamp: kept verbatim ({stripped['metadata'].get('encoding_name')!r}), "
            "--encoding-name='' passed",
        )

    present_forbidden = [k for k in FORBIDDEN_KEYS if k in stripped]
    assert not present_forbidden, f"BUG: stripped payload carries forbidden key(s): {present_forbidden}"
    assert set(stripped.keys()) == set(STRIPPED_KEYS), (
        f"BUG: stripped payload key set {sorted(stripped.keys())} != {sorted(STRIPPED_KEYS)}"
    )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, args.out_path)

    # Re-load to assert the on-disk payload is exactly what we intended —
    # catches any torch.save/load key-loss surprise, not just the in-memory dict.
    reloaded = torch.load(args.out_path, map_location="cpu", weights_only=False)
    reloaded_keys = set(reloaded.keys())
    if reloaded_keys != set(STRIPPED_KEYS):
        print(
            f"ERROR: post-save re-load key mismatch: {sorted(reloaded_keys)} != "
            f"{sorted(STRIPPED_KEYS)}",
            file=sys.stderr,
        )
        sys.exit(2)
    present_forbidden_reload = [k for k in FORBIDDEN_KEYS if k in reloaded_keys]
    if present_forbidden_reload:
        print(
            f"ERROR: post-save re-load carries forbidden key(s): {present_forbidden_reload}",
            file=sys.stderr,
        )
        sys.exit(2)

    sha = _sha256_of_file(args.out_path)
    step = reloaded.get("step")
    print(f"OK: wrote {args.out_path}")
    print(f"  step:  {step}")
    print(f"  keys:  {sorted(reloaded_keys)}")
    print(f"  sha256: {sha}")
    print(f"  metadata: {reloaded.get('metadata')}")


if __name__ == "__main__":
    main()
