"""Shared state-dict key probes for encoding detection (§176 P64).

Probe keys for the network's first conv (in_channels) and final policy
fc (out_features). Listed in priority order; first match wins.

Canonical single definition — compat.py and resolvers.py import from here.
"""
from __future__ import annotations

FIRST_CONV_KEYS: tuple[str, ...] = (
    "trunk.0.weight",
    "trunk.conv.weight",
    "input_conv.weight",
    "stem.0.weight",
    "conv1.weight",
)

POLICY_FC_KEYS: tuple[str, ...] = (
    "policy_fc.weight",
    "policy_head.fc.weight",
    "policy.fc.weight",
    "policy.weight",
)
