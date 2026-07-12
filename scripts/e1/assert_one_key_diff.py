#!/usr/bin/env python
"""T8 D5 — E1 one-key-diff assertion.

The two E1 arm variants MUST differ in EXACTLY `value_head_type` (INV-D1 / R5,
one-variable discipline). scripts/e1/run_pair.sh calls this BEFORE launching and
REFUSES (exit nonzero) if the resolved diff is anything other than that single
key.

(The key is FLAT top-level `value_head_type`, not nested `model.value_head_type`:
base configs/model.yaml carries it as a flat key, so a nested `model:` block is a
silent namespace shadow the variant_validator hard-aborts on.)

Resolves each variant through the SAME loader the launch uses
(orchestrator.load_train_config) so the assertion sees the exact merged config
that will run — no re-implemented merge that could drift.

Usage:
  python -m scripts.e1.assert_one_key_diff e1_scalar e1_dist65
  # prints the diff; exit 0 iff diff == {model.value_head_type}, else exit 1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The single key the arms are ALLOWED to differ on (flat top-level — see docstring).
EXPECTED_DIFF_KEY = "value_head_type"


def _resolve_variant_config(variant: str) -> Dict[str, Any]:
    """Resolve a variant's merged config via the launch loader.

    Uses orchestrator.load_train_config with a synthetic args namespace so the
    merged config is byte-identical to what scripts/train.py --variant <name>
    would produce (same base set, same variant path, same deep-merge).
    """
    from hexo_rl.training import orchestrator as _orch

    args = argparse.Namespace(config=None, variant=variant)
    config, _layers = _orch.load_train_config(args)
    return config


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dicts to dotted-key leaves. Lists / scalars are leaves."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def diff_variant_configs(
    variant_a: str, variant_b: str
) -> Dict[str, Tuple[Any, Any]]:
    """Return {dotted_key: (a_value, b_value)} for every differing leaf between
    the two variants' MERGED configs."""
    fa = _flatten(_resolve_variant_config(variant_a))
    fb = _flatten(_resolve_variant_config(variant_b))
    all_keys = set(fa) | set(fb)
    _MISSING = object()
    diff: Dict[str, Tuple[Any, Any]] = {}
    for k in sorted(all_keys):
        av = fa.get(k, _MISSING)
        bv = fb.get(k, _MISSING)
        if av != bv:
            diff[k] = (
                None if av is _MISSING else av,
                None if bv is _MISSING else bv,
            )
    return diff


def assert_one_key_diff(
    variant_a: str, variant_b: str
) -> Dict[str, Tuple[Any, Any]]:
    """Assert the two variants differ in EXACTLY {EXPECTED_DIFF_KEY}.

    Returns the diff on success; raises ValueError otherwise (0-key, 2+-key, or
    a single WRONG key).
    """
    diff = diff_variant_configs(variant_a, variant_b)
    if set(diff) != {EXPECTED_DIFF_KEY}:
        raise ValueError(
            f"E1 one-key-diff INVARIANT VIOLATED: {variant_a!r} vs {variant_b!r} "
            f"must differ in EXACTLY {{{EXPECTED_DIFF_KEY!r}}}, but the resolved "
            f"diff is {sorted(diff)} ({len(diff)} key(s)). Full diff: "
            + "; ".join(f"{k}: {a!r} -> {b!r}" for k, (a, b) in sorted(diff.items()))
            + ". REFUSING to launch — the arms are not a clean one-variable pair."
        )
    return diff


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("variant_a", help="scalar arm variant name (e.g. e1_scalar)")
    ap.add_argument("variant_b", help="dist65 arm variant name (e.g. e1_dist65)")
    args = ap.parse_args()
    try:
        diff = assert_one_key_diff(args.variant_a, args.variant_b)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
    (k, (a, b)) = next(iter(diff.items()))
    print(f"OK: one-key diff — {k}: {a!r} -> {b!r}")
    sys.exit(0)


if __name__ == "__main__":
    main()
