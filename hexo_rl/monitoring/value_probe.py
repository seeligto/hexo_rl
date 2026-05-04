"""Phase B' Class-2 — Value-head drift probe.

Loads ``fixtures/value_probe_50.npz`` and runs the trainer's model forward
on the 50 fixed positions every N training steps. Emits a
``value_probe_drift`` event with per-subset means so we can watch:

  * decisive subset drifting toward −0.5 → Class 2 dominant
    (value head collapses onto the draw equilibrium)
  * draw subset staying flat → Class 2 weak
  * neither drifting → Class 2 weak overall

The probe is cheap: ~50 forward passes, batched as a single (50, 8, 19, 19)
tensor, ~3-8 ms on a 4060 / 5080. Runs under ``torch.inference_mode()``.

Dispatched from `_emit_training_events` only when
``instrumentation.enabled=true``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import structlog
import torch

log = structlog.get_logger(__name__)


class ValueProbe:
    """Run value-head forward pass on a fixed 50-position fixture.

    The fixture stores 18-plane wire-format tensors; the trainer's network
    slices to its configured `in_channels` via index_select internally.
    """

    def __init__(
        self,
        fixture_path: Path | str = Path("fixtures/value_probe_50.npz"),
        device: Optional[torch.device] = None,
    ) -> None:
        path = Path(fixture_path)
        if not path.exists():
            raise FileNotFoundError(
                f"value-probe fixture missing at {path}; run "
                "scripts/build_value_probe_fixture.py first",
            )
        npz = np.load(path, allow_pickle=False)
        self._states_np = np.ascontiguousarray(npz["states"], dtype=np.float16)
        self._subset_np = np.ascontiguousarray(npz["subset"], dtype=np.int8)
        self._n = int(self._states_np.shape[0])
        config: dict = {}
        if "config_bytes" in npz.files:
            try:
                config = json.loads(bytes(npz["config_bytes"]).decode("utf-8"))
            except Exception:
                config = {}
        # Post-§131: production buffer wire format = 8 planes natively.
        self._wire_planes = int(config.get("wire_planes", 8))
        self._board_size = int(config.get("board_size", 19))
        self.fixture_path = str(path)

        self._device = device if device is not None else torch.device("cpu")
        # f16 staging tensor on device — kept for the lifetime of the probe.
        self._states_t = torch.from_numpy(self._states_np).to(
            self._device, non_blocking=True,
        )

    @property
    def n_positions(self) -> int:
        return self._n

    @property
    def n_decisive(self) -> int:
        return int(np.sum(self._subset_np == 0))

    @property
    def n_draw(self) -> int:
        return int(np.sum(self._subset_np == 1))

    @torch.inference_mode()
    def compute(self, model: torch.nn.Module) -> dict[str, Any]:
        """Run forward; return per-subset value statistics.

        The model returns a 3-tuple `(log_policy, value, value_logit)` from
        its forward — we read the post-tanh `value` channel. The probe
        returns NaN-safe means even when a subset happens to be empty.

        Fixture is 8-plane native (post-§131 buffer format) — fed straight
        to the model with no slicing.
        """
        model.eval()
        was_train = model.training
        try:
            outputs = model(self._states_t.float())
        finally:
            if was_train:
                model.train()
        if isinstance(outputs, (tuple, list)):
            value_t = outputs[1]
        else:
            value_t = outputs
        value_arr = value_t.detach().to("cpu").float().reshape(-1).numpy()
        if value_arr.shape[0] != self._n:
            # Some networks return per-cluster values shape (N, K) — mean over K
            value_arr = value_arr.reshape(self._n, -1).mean(axis=1)

        decisive_mask = self._subset_np == 0
        draw_mask = self._subset_np == 1
        decisive_v = value_arr[decisive_mask]
        draw_v = value_arr[draw_mask]

        def _mean(x: np.ndarray) -> float:
            return float(x.mean()) if x.size > 0 else float("nan")

        def _std(x: np.ndarray) -> float:
            return float(x.std(ddof=0)) if x.size > 1 else 0.0

        return {
            "decisive_mean": _mean(decisive_v),
            "decisive_std":  _std(decisive_v),
            "draw_mean":     _mean(draw_v),
            "draw_std":      _std(draw_v),
            "decisive_n":    int(decisive_v.size),
            "draw_n":        int(draw_v.size),
            # Per-position raw values — kept compact, useful for trend traces.
            "decisive_values": [round(v, 4) for v in decisive_v.tolist()],
            "draw_values":     [round(v, 4) for v in draw_v.tolist()],
        }
