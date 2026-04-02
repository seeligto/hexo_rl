"""Shared YAML config loading utility."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_log = logging.getLogger(__name__)


def load_config(*paths: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load and merge YAML configs. Later paths override earlier.

    Args:
        *paths: One or more paths to YAML config files.
        overrides: Optional dict of overrides applied last.

    Returns:
        Merged config dict.

    Warnings:
        Logs a warning when the same top-level key appears in multiple files,
        so that accidental override bugs are visible at startup.
    """
    merged: Dict[str, Any] = {}
    key_sources: Dict[str, str] = {}

    for p in paths:
        with open(p) as f:
            cfg = yaml.safe_load(f) or {}
        for key in cfg:
            if key in key_sources:
                _log.warning(
                    "config_key_overlap: key '%s' appears in both '%s' and '%s' — '%s' wins",
                    key, key_sources[key], p, p,
                )
            else:
                key_sources[key] = p
        _log.debug("config_loaded: %s keys from %s", len(cfg), p)
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
