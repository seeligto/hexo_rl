"""Canonical defaults for HexTacToeNet hparams.

Single source of repetition (SSR) for the model architecture defaults
that previously appeared inline in trainer.py, lifecycle.py, and loop.py.
Centralising prevents drift between the `_resolve_model_hparams` resolver,
the `build_inference_model` config-fallback chain, the trainer's resolved-
dict ``.get`` fallbacks, and the run_start event payload.

``board_size`` is included for the resolver defaults dict, but call sites
that derive trunk size from the encoding registry (e.g. lifecycle.py) keep
that path — only the ``res_blocks`` / ``filters`` / ``in_channels`` /
``se_reduction_ratio`` fallbacks are SSR-relevant there.

No behavior change: values mirror what was inlined prior to §176 P10.
"""

from __future__ import annotations

from typing import Any, Dict

from hexo_rl.encoding import lookup as _lookup_encoding

_V6 = _lookup_encoding("v6")
BOARD_SIZE: int = _V6.board_size
BUFFER_CHANNELS: int = _V6.n_planes

# Canonical defaults. Keep keys in sync with HexTacToeNet.__init__.
MODEL_HPARAM_DEFAULTS: Dict[str, Any] = {
    "board_size": BOARD_SIZE,
    "res_blocks": 12,
    "filters": 128,
    "in_channels": BUFFER_CHANNELS,
    "se_reduction_ratio": 4,
    # Distributional value head (E1). Default 'scalar' = current BCE head.
    "value_head_type": "scalar",
    "n_value_bins": 65,
}
