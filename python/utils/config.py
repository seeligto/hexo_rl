"""Shared YAML config loading utility."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(*paths: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load and merge YAML configs. Later paths override earlier.

    Args:
        *paths: One or more paths to YAML config files.
        overrides: Optional dict of overrides applied last.

    Returns:
        Merged config dict.
    """
    merged: Dict[str, Any] = {}
    for p in paths:
        with open(p) as f:
            cfg = yaml.safe_load(f) or {}
        _deep_merge(merged, cfg)
    if overrides:
        _deep_merge(merged, overrides)
    return merged


def _deep_merge(base: Dict, override: Dict) -> None:
    """Recursively merge override into base (mutates base)."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
