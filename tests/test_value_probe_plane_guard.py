"""§P5-CT P1-3 — value_probe skips (NaN) on fixture/model plane mismatch.

The probe fed a fixed-plane fixture straight into the model with no guard, so a
v6tp (10-ch) model + 8-plane fixture crashed the forward (caught downstream →
probe went dark). Add a plane-count guard that returns NaN means, matching
value_spread_canary's skip behaviour.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from hexo_rl.monitoring.value_probe import ValueProbe


class _StubNet(torch.nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):  # noqa: ANN001
        # Would raise on a plane mismatch — the guard must short-circuit first.
        assert x.shape[1] == self.in_channels, "plane mismatch reached forward"
        n = x.shape[0]
        return torch.zeros(n, 362), torch.zeros(n), torch.zeros(n)


def _write_fixture(path, planes: int) -> None:
    states = np.zeros((4, planes, 19, 19), dtype=np.float16)
    subset = np.array([0, 0, 1, 1], dtype=np.int8)
    cfg = json.dumps({"wire_planes": planes, "board_size": 19, "encoding": "v6"}).encode()
    np.savez_compressed(
        path, states=states, subset=subset,
        config_bytes=np.frombuffer(cfg, dtype=np.uint8),
    )


def test_plane_mismatch_returns_nan_not_crash(tmp_path):
    fix = tmp_path / "fix8.npz"
    _write_fixture(fix, planes=8)
    probe = ValueProbe(fixture_path=fix)
    out = probe.compute(_StubNet(in_channels=10))  # 8-plane fixture, 10-ch model
    assert np.isnan(out["decisive_mean"])
    assert np.isnan(out["draw_mean"])


def test_matched_planes_runs_forward(tmp_path):
    fix = tmp_path / "fix8.npz"
    _write_fixture(fix, planes=8)
    probe = ValueProbe(fixture_path=fix)
    out = probe.compute(_StubNet(in_channels=8))  # matched
    assert out["decisive_mean"] == 0.0
    assert out["decisive_n"] == 2 and out["draw_n"] == 2
