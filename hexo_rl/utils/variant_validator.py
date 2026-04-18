"""Validator for variant YAML configs — catches silent namespace shadows.

A silent namespace shadow occurs when a variant yaml has a top-level key K
whose value is a dict, but K is not a dict in the merged base configs (i.e.
base keys are flat). The deep-merge then creates merged[K] = {...} instead
of overriding the flat base keys, silently dropping the variant's intent.

Example: training.yaml has flat ``max_train_burst: 16``.
Variant with ``training: {max_train_burst: 8}`` creates ``merged['training']``
and never touches ``merged['max_train_burst']``. Desktop ran at 16 instead of 8.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


def validate_variant_against_bases(
    variant_cfg: dict[str, Any],
    base_cfgs: dict[str, dict[str, Any]],
) -> list[str]:
    """Return warning strings for variant keys that silently shadow flat base keys.

    Args:
        variant_cfg: Loaded variant yaml dict.
        base_cfgs: Maps base-config name (e.g. 'training') to its loaded dict.
                   Keys should be the base yaml stem so the error message is useful.

    Returns:
        List of warning strings (empty = clean).
    """
    from hexo_rl.utils.config import _deep_merge  # local import avoids circular

    merged_base: dict[str, Any] = {}
    for content in base_cfgs.values():
        _deep_merge(merged_base, content)

    flat_base_keys = {k for k, v in merged_base.items() if not isinstance(v, dict)}

    warnings: list[str] = []
    for key, val in variant_cfg.items():
        if not isinstance(val, dict):
            continue
        if key in merged_base and isinstance(merged_base[key], dict):
            # Both variant and base have a dict at this key → valid deep-merge.
            continue
        # variant has a dict at 'key' but merged base does NOT have a dict there.
        # Any sub-keys that appear flat in the base will be silently dropped.
        shadowed = sorted(sub for sub in val if sub in flat_base_keys)
        if shadowed:
            warnings.append(
                f"variant key '{key}' is a nested block but base configs have "
                f"{shadowed} as flat keys — sub-keys under '{key}' will be "
                f"silently dropped and base values will be used instead"
            )
    return warnings


def _load_standard_base_cfgs(root: Path) -> dict[str, dict[str, Any]]:
    names = ["model", "training", "selfplay", "game_replay", "monitoring"]
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        p = root / "configs" / f"{name}.yaml"
        if p.exists():
            with open(p) as f:
                result[name] = yaml.safe_load(f) or {}
    return result


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m hexo_rl.utils.variant_validator <variant.yaml>")
        sys.exit(2)

    variant_path = Path(sys.argv[1])
    if not variant_path.exists():
        print(f"error: {variant_path} not found")
        sys.exit(2)

    with open(variant_path) as f:
        variant_cfg = yaml.safe_load(f) or {}

    root = Path(__file__).resolve().parents[2]
    base_cfgs = _load_standard_base_cfgs(root)

    warnings = validate_variant_against_bases(variant_cfg, base_cfgs)
    if warnings:
        for w in warnings:
            print(f"WARNING: {w}")
        sys.exit(1)
    else:
        print(f"OK: {variant_path.name} — no nested namespace shadows found")
        sys.exit(0)


if __name__ == "__main__":
    main()
