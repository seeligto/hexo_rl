"""Variant-config regressions — resolved configs after train.py deep-merge.

Each test loads a base+variant config stack through ``hexo_rl.utils.config.load_config``
(the same path ``scripts/train.py`` takes when ``--variant`` is passed) and asserts
on the resolved values. Guards against silent drift when the base ``selfplay.yaml``
grows new keys that variants inherit by accident.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from hexo_rl.encoding.resolvers import resolve_from_config
from hexo_rl.utils.config import load_config
from hexo_rl.utils.variant_validator import validate_variant_against_bases

# Match scripts/train.py::main `_BASE_CONFIGS` (minus monitoring/game_replay —
# they do not touch selfplay.playout_cap and keep the test surface minimal).
ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIGS = [
    str(ROOT / "configs" / "model.yaml"),
    str(ROOT / "configs" / "training.yaml"),
    str(ROOT / "configs" / "selfplay.yaml"),
]


def _resolve(variant: str) -> dict:
    variant_path = ROOT / "configs" / "variants" / f"{variant}.yaml"
    return load_config(*BASE_CONFIGS, str(variant_path))


def _base_cfgs_for_validator() -> dict:
    names = ["model", "training", "selfplay"]
    result = {}
    for name in names:
        p = ROOT / "configs" / f"{name}.yaml"
        with open(p) as f:
            result[name] = yaml.safe_load(f) or {}
    return result


# ---------------------------------------------------------------------------
# Canonical variant (vast.yaml) — §174 sustained-run validation
# ---------------------------------------------------------------------------


def test_vast_resolves_to_sustained_values() -> None:
    """§174 canonical variant must resolve with the P0-operator-locked knobs.
    Prevents silent drift when base configs change."""
    cfg = _resolve("vast")
    assert cfg.get("training_steps_per_game") == 2.0
    assert cfg.get("max_train_burst") == 8
    assert cfg["lr"] == 1e-3
    assert cfg["grad_clip"] == 1.0
    assert cfg["encoding"] == "v6w25"
    assert cfg["selfplay"]["gumbel_mcts"] is False
    assert cfg["selfplay"]["completed_q_values"] is True
    assert cfg["selfplay"]["n_workers"] == 18
    assert cfg["selfplay"]["inference_batch_size"] == 128
    assert cfg["selfplay"]["max_game_moves"] == 150
    assert cfg["selfplay"]["playout_cap"]["fast_prob"] == 0.0
    assert cfg["mcts"]["n_simulations"] == 400
    assert cfg["eval_interval"] == 10000
    assert cfg["monitors"]["hard_abort_grad_norm"] == 10.0


# ---------------------------------------------------------------------------
# P68 — scripts/train.py wires variant_validator (abort-on-warning)
# ---------------------------------------------------------------------------


def test_validator_fires_on_train_py(tmp_path, monkeypatch):
    """§176 P68: scripts/train.py must invoke variant_validator on --variant
    load and abort when it returns warnings.

    Constructs a deliberately-shadowing variant (nested ``training`` block
    declaring a flat-base key) and invokes scripts/train.py via subprocess.
    Expect non-zero exit + stderr containing ``variant_validator WARNING``.
    """
    import subprocess
    import sys

    # Build a minimal variant that triggers the namespace-shadow warning:
    # base ``training.yaml`` has flat ``max_train_burst``; nesting it under
    # ``training:`` triggers the validator.
    bad_variant_dir = ROOT / "configs" / "variants"
    bad_variant_name = "__p68_validator_test_variant__"
    bad_variant_path = bad_variant_dir / f"{bad_variant_name}.yaml"
    bad_variant_path.write_text(
        "encoding: v6w25\n"
        "training:\n"
        "  max_train_burst: 8\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train.py"),
             "--variant", bad_variant_name,
             "--log-dir", str(tmp_path / "logs")],
            capture_output=True, text=True, timeout=30,
            cwd=str(ROOT),
        )
        # Must have aborted before training starts.
        assert proc.returncode != 0, (
            f"train.py should have aborted on bad variant; stdout={proc.stdout!r} "
            f"stderr={proc.stderr!r}"
        )
        assert "variant_validator WARNING" in proc.stderr, (
            f"missing validator warning in stderr; stderr={proc.stderr!r}"
        )
    finally:
        bad_variant_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# P69 — registry-sourced board_size / in_channels after SSR14 cleanup
# ---------------------------------------------------------------------------


def test_vast_resolves_encoding_from_registry() -> None:
    """P69/SSR14: vast.yaml must NOT carry board_size or in_channels as
    scattered keys; encoding: v6w25 in registry must supply both values."""
    variant_path = ROOT / "configs" / "variants" / "vast.yaml"
    with open(variant_path) as f:
        raw = yaml.safe_load(f) or {}

    # Scattered keys must be absent after P69 cleanup.
    assert "board_size" not in raw, (
        "vast.yaml still has redundant board_size scalar — P69 cleanup missed"
    )
    assert "in_channels" not in raw, (
        "vast.yaml still has redundant in_channels scalar — P69 cleanup missed"
    )

    # Registry must supply the correct values from encoding: v6w25.
    spec = resolve_from_config(raw)
    assert spec.board_size == 25, f"expected board_size=25 from registry, got {spec.board_size}"
    assert spec.n_planes == 8, f"expected n_planes(in_channels)=8 from registry, got {spec.n_planes}"


# ---------------------------------------------------------------------------
# E-001 / E-018 / E-019 / Q29 regression tests
# ---------------------------------------------------------------------------


def test_all_variants_have_no_nested_base_namespaces() -> None:
    """E-001/E-018/E-019: every configs/variants/*.yaml must have no nested
    namespace block that shadows a flat base key (silent-drop class of bugs)."""
    base_cfgs = _base_cfgs_for_validator()
    variant_dir = ROOT / "configs" / "variants"
    failures: list[str] = []
    for variant_path in sorted(variant_dir.glob("*.yaml")):
        with open(variant_path) as f:
            variant_cfg = yaml.safe_load(f) or {}
        warnings = validate_variant_against_bases(variant_cfg, base_cfgs)
        for w in warnings:
            failures.append(f"{variant_path.name}: {w}")
    assert not failures, "Nested namespace shadows found:\n" + "\n".join(failures)


def test_validator_catches_nested_training_block() -> None:
    """Q29: validator must flag a variant with training: {...} when training.yaml
    is flat — this was the E-001 class of bug."""
    base_cfgs = _base_cfgs_for_validator()
    synthetic_variant = {"training": {"training_steps_per_game": 4.0}}
    warnings = validate_variant_against_bases(synthetic_variant, base_cfgs)
    assert warnings, (
        "validator did not flag nested 'training:' block — "
        "E-001 class bug would go undetected"
    )
    assert any("training" in w for w in warnings)
