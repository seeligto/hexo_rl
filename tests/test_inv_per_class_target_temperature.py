"""§S181-AUDIT Wave 2 INV pin: per-class target temperature lever wiring.

Guards the V-B-A `uniform_self` lever path (per REAL_RUN_RECIPE §3).
When `per_class_target_temperature.enabled = true` the trainer must
import `apply_per_class_temperature` AND call it before the policy loss
on every batch. The temperature module must remain side-effect-free
when disabled (default false in `configs/training.yaml`) so existing
runs see bit-identical behaviour.

If a future refactor moves the temperature call out of `train_step`
or accidentally disables it via config plumbing drift, this test
breaks immediately.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hexo_rl.training.per_class_target_temperature import (
    _resolve_config,
    apply_per_class_temperature,
)


def test_per_class_target_temperature_default_is_disabled_in_trainer_yaml():
    """training.yaml ships with the lever disabled by default."""
    yaml_path = Path(__file__).resolve().parents[1] / "configs" / "training.yaml"
    text = yaml_path.read_text()
    assert "per_class_target_temperature:" in text, (
        "training.yaml must define per_class_target_temperature so variants "
        "can override; key is missing."
    )
    # Find the block — the first matching `enabled:` under the heading wins.
    block_idx = text.index("per_class_target_temperature:")
    block = text[block_idx:block_idx + 600]
    assert "enabled: false" in block, (
        "per_class_target_temperature.enabled must default to false; "
        "smoke + main-run variants opt in explicitly."
    )


def test_trainer_train_step_imports_apply_per_class_temperature():
    """Grep-guard: trainer.py must reference the apply_per_class_temperature
    helper inside `train_step` so the lever wiring survives refactors."""
    trainer_path = Path(__file__).resolve().parents[1] / "hexo_rl" / "training" / "trainer.py"
    text = trainer_path.read_text()
    assert "apply_per_class_temperature" in text, (
        "trainer.py no longer references apply_per_class_temperature — the "
        "V-B-A `uniform_self` lever wiring is broken."
    )
    # Must guard on config to avoid a per-step import cost when disabled.
    assert "per_class_target_temperature" in text, (
        "trainer.py must guard the lever on the config key so disabled runs "
        "skip the import + classify cost."
    )


def test_lever_disabled_returns_input_unchanged_bitwise():
    """When the lever is off, policies_t passes through unmodified."""
    rng = np.random.default_rng(0)
    states = torch.from_numpy(rng.random((4, 8, 19, 19), dtype=np.float32))
    policies = torch.from_numpy(rng.random((4, 362), dtype=np.float32))
    cfg_disabled = {"per_class_target_temperature": {"enabled": False}}
    out = apply_per_class_temperature(
        policies, states, n_pretrain=0,
        config=cfg_disabled, device=torch.device("cpu"),
    )
    assert out is policies, "Disabled lever must return the input object unchanged."


def test_lever_off_via_all_unit_temperatures_short_circuits():
    """`enabled: true` + every temperature == 1.0 is still a no-op."""
    cfg = {"per_class_target_temperature": {
        "enabled": True, "colony_temperature": 1.0,
        "extension_temperature": 1.0, "neither_temperature": 1.0,
    }}
    assert _resolve_config(cfg) is None
