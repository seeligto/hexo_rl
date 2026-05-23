"""§S181-AUDIT Wave 2 — exponential moving average of model weights.

The EMA model is updated every `update_every` training steps via a decayed
running mean of the trainer's raw parameters. Self-play inference,
evaluation, and best-model promotion read EMA weights when EMA is
enabled; the trainer's raw weights continue to drive the next gradient
step.

Goal. Anti-colony lever (Wave 1 close-out + REAL_RUN_RECIPE §3 LANDED).
Smooths the value-head signal that self-play workers see, attenuating
high-frequency colony-direction drift in the gradient.

Mechanism. Hand-rolled state_dict-level EMA. We can't use
`torch.optim.swa_utils.AveragedModel` here because it deep-copies the
model on construction and `HexTacToeNet._spec` is a PyO3
`RegistrySpec` that does not implement Python's deep-copy protocol.
The manual implementation keeps a flat `name -> tensor` shadow indexed
by `state_dict()` keys (so all parameters + buffers ride along, matching
`use_buffers=True` semantics) and applies the EMA update in-place.

Backward compat. `enabled=False` (default) means `build_ema_model` is
never called; trainer keeps a None EMA slot, all dispatch routes through
the raw model. Existing checkpoints load fine with no `_ema` sidecar.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable

import torch


DEFAULT_DECAY = 0.999
DEFAULT_UPDATE_EVERY = 10


def _base_of(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap a torch.compile OptimizedModule if present.

    `torch.compile` returns an OptimizedModule whose `.state_dict()`
    prefixes every key with `_orig_mod.`; the EMA storage is keyed off
    the raw module's names so we always train against the unwrapped
    surface.
    """
    return getattr(model, "_orig_mod", model)


class EmaModel:
    """State-dict-level EMA of a wrapped model's parameters and buffers.

    The shadow storage owns its own tensor allocations. Floating-point
    entries are mixed via the standard EMA formula
    ``avg = avg + (1 - decay) * (cur - avg)``; non-floating entries
    (int buffers, e.g. ``num_batches_tracked``) are copied verbatim so
    the EMA state stays compatible with downstream ``load_state_dict``.
    """

    def __init__(self, model: torch.nn.Module, decay: float = DEFAULT_DECAY) -> None:
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"EMA decay must be in [0, 1); got {decay}")
        self.decay: float = float(decay)
        base = _base_of(model)
        self._shadow: Dict[str, torch.Tensor] = {
            name: tensor.detach().clone()
            for name, tensor in base.state_dict().items()
        }

    def update_parameters(self, model: torch.nn.Module) -> None:
        """Apply one EMA mixing step from `model`'s current weights."""
        base = _base_of(model)
        with torch.no_grad():
            for name, cur in base.state_dict().items():
                shadow = self._shadow.get(name)
                if shadow is None:
                    # Architecture change mid-run — re-seed this entry.
                    self._shadow[name] = cur.detach().clone()
                    continue
                if cur.dtype.is_floating_point:
                    # In-place mix preserves device + dtype.
                    shadow.mul_(self.decay).add_(cur.detach(), alpha=1.0 - self.decay)
                else:
                    # Integer / bool buffers — overwrite straight through.
                    shadow.copy_(cur.detach())

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return a shallow-copied view of the shadow state.

        Callers that want to mutate must clone first; the underlying
        tensors are the EMA's own storage and must not be modified
        outside `update_parameters`.
        """
        return dict(self._shadow)

    def load_into(self, model: torch.nn.Module) -> None:
        """Load EMA shadow state into `model` in-place."""
        base = _base_of(model)
        base.load_state_dict(self._shadow)

    # Compatibility shim — some call sites use ``ema.module.state_dict()``
    # by analogy with torch.optim.swa_utils.AveragedModel. Expose `module`
    # as a thin proxy that owns the same state.
    @property
    def module(self) -> "_EmaModuleView":
        return _EmaModuleView(self)


class _EmaModuleView:
    """Module-like proxy that exposes the EMA shadow via state_dict / parameters.

    Provided for compatibility with call sites that want to treat the EMA
    weights as if they came from a real `nn.Module` (e.g. for
    ``save_inference_weights(ema.module, path)``). It is NOT a real
    `nn.Module` — it has no `forward`. Forward inference against EMA
    weights must go through a real model with EMA state materialized,
    e.g. via ``ema.load_into(fresh_model)`` or by calling
    ``Trainer.inference_state_dict()`` and loading downstream.
    """

    def __init__(self, owner: EmaModel) -> None:
        self._owner = owner

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self._owner.state_dict()

    def parameters(self) -> Iterable[torch.Tensor]:
        # Only floating-point shadow entries are "parameter-shaped";
        # integer buffers are skipped to match nn.Module.parameters semantics.
        for t in self._owner._shadow.values():
            if t.dtype.is_floating_point:
                yield t


def build_ema_model(model: torch.nn.Module, decay: float = DEFAULT_DECAY) -> EmaModel:
    """Construct an EMA wrapper around `model`.

    The wrapper allocates fresh shadow storage on construction. Call
    `update_parameters(model)` after each optimizer step (gated by
    `update_every` in the trainer) to mix the current weights in.
    """
    return EmaModel(model, decay=decay)


def resolve_ema_config(config: Dict[str, Any]) -> tuple[bool, float, int]:
    """Read EMA settings from the trainer config block.

    Accepts either nested form `{"ema": {"enabled": ..., ...}}` or flat
    keys `ema_enabled` / `ema_decay` / `ema_update_every` for back-compat
    with variant configs that flatten everything to the root.
    """
    nested = config.get("ema") if isinstance(config.get("ema"), dict) else {}
    enabled = bool(nested.get("enabled", config.get("ema_enabled", False)))
    decay = float(nested.get("decay", config.get("ema_decay", DEFAULT_DECAY)))
    update_every = int(nested.get("update_every", config.get("ema_update_every", DEFAULT_UPDATE_EVERY)))
    if update_every < 1:
        raise ValueError(f"ema.update_every must be >= 1; got {update_every}")
    return enabled, decay, update_every


# Back-compat alias for the `avg_fn`-style helper retained from the
# pre-rewrite ema.py. Tests in `hexo_rl/training/tests/test_ema.py`
# exercise this directly — keep it pure-function so EMA semantics stay
# pinnable in isolation from the EmaModel wrapper.
def _ema_avg_fn(decay: float):
    if not (0.0 <= decay < 1.0):
        raise ValueError(f"EMA decay must be in [0, 1); got {decay}")

    def fn(avg_p: torch.Tensor, cur_p: torch.Tensor, _num_averaged: torch.Tensor) -> torch.Tensor:
        return avg_p + (1.0 - decay) * (cur_p - avg_p)

    return fn
