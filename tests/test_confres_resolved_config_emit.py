"""CONFRES Phase-3 batch 6a-ii — launch wiring of the ResolvedRunConfig builder + emission.

Verifies the STRICTLY-ADDITIVE, BYTE-PURE wiring that emits exactly ONE ``resolved_config`` event
(the F1-forensic provenance artifact) alongside the existing launch logic:

- ``build_and_emit_resolved_config`` emits exactly one ``resolved_config`` event whose payload
  carries the expected knobs with provenance (Phase-B post-load knobs + Phase-A launch-only knobs
  flagged ``consumed_pre_resolution: true``).
- ``assert_layers_reconstruct`` fires (raises) when the raw layer chain does NOT reconstruct the
  merged config, on a REAL base+variant/config load via ``_build_load_paths`` + ``capture_config_layers``.
- ``load_train_config`` returns the kind-tagged raw layer chain from the SINGLE load_paths source,
  and those layers reconstruct the merged config (F2).
- The emission is emission-only: a resolver conflict (I2) does NOT abort the launch (6a-ii byte-pure;
  the precedence change is 6b).

Hermetic — no GPU, no real training loop. The trainer is a ``SimpleNamespace`` stub exposing only
the two attributes the wiring reads (``config`` + ``optimizer.param_groups``).
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

import hexo_rl.monitoring.events as events_mod
from hexo_rl.config.resolve.run_config import (
    assert_layers_reconstruct,
    capture_config_layers,
    merged_layers,
)
from hexo_rl.training import orchestrator


# ── event capture via a registered renderer (project convention) ─────────────
class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def on_event(self, payload: dict) -> None:
        self.calls.append(payload)


@pytest.fixture(autouse=True)
def _reset_renderers():
    with events_mod._lock:
        events_mod._renderers.clear()
    yield
    with events_mod._lock:
        events_mod._renderers.clear()


@pytest.fixture
def recorder():
    r = _Recorder()
    events_mod.register_renderer(r)
    return r


class _StubLog:
    """structlog-BoundLogger-shaped stub: methods accept an event + arbitrary kwargs, swallow them."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict]] = []

    def _rec(self, level, event, **kw):
        self.records.append((level, event, kw))

    def info(self, event, **kw):
        self._rec("info", event, **kw)

    def warning(self, event, **kw):
        self._rec("warning", event, **kw)

    def error(self, event, **kw):
        self._rec("error", event, **kw)


def _log():
    return _StubLog()


def _args(**over) -> argparse.Namespace:
    base = dict(checkpoint=None, config=None, variant=None, iterations=None)
    base.update(over)
    return argparse.Namespace(**base)


def _fake_trainer(config: dict | None = None, lr: float | None = 0.002):
    """Stub exposing ONLY the attributes the wiring reads: config + optimizer.param_groups."""
    param_groups = [{"lr": lr}] if lr is not None else []
    return SimpleNamespace(
        config=config if config is not None else {},
        optimizer=SimpleNamespace(param_groups=param_groups),
        step=0,
    )


def _fresh_layers(encoding="v6"):
    """Minimal base→variant chain; merged_config is its deep-merge (F2 target)."""
    layers = [
        {"label": "configs/model.yaml", "kind": "base", "raw": {"encoding": encoding, "seed": 42}},
        {"label": "configs/variants/x.yaml", "kind": "variant",
         "raw": {"evaluation": {"eval_temperature": 0.7}}},
    ]
    return layers, merged_layers(layers)


# ── EMIT exactly one resolved_config with expected knobs + provenance ────────
def test_emits_exactly_one_resolved_config_event(recorder):
    layers, merged = _fresh_layers()
    trainer = _fake_trainer(config={"encoding": "v6"}, lr=0.002)
    payload = orchestrator.build_and_emit_resolved_config(
        layers, merged, registry_spec={}, trainer=trainer, args=_args(), log=_log(),
    )
    resolved_events = [c for c in recorder.calls if c.get("event") == "resolved_config"]
    assert len(resolved_events) == 1
    emitted = resolved_events[0]
    # returned payload is the emitted one (sans the emitter's ts)
    assert payload["event"] == "resolved_config"
    assert emitted["knobs"] == payload["knobs"]


def test_payload_carries_phase_b_knobs_with_provenance(recorder):
    layers, merged = _fresh_layers()
    trainer = _fake_trainer(config={"encoding": "v6"})
    payload = orchestrator.build_and_emit_resolved_config(
        layers, merged, registry_spec={}, trainer=trainer, args=_args(), log=_log(),
    )
    knobs = payload["knobs"]
    for expect in ("encoding", "eval_temperature", "n_sims_eval.random",
                   "n_sims_eval.sealbot", "bootstrap", "lr"):
        assert expect in knobs, f"missing Phase-B knob {expect}"
    # every knob carries the provenance shape
    for spec in knobs.values():
        assert set(spec) >= {"value", "source", "precedence_family", "inputs_seen"}
    # eval_temperature declared on the variant layer → variant-sourced 0.7
    assert knobs["eval_temperature"]["value"] == 0.7
    assert knobs["eval_temperature"]["source"] == "variant"


def test_phase_a_preload_knobs_flagged_consumed_pre_resolution(recorder):
    layers, merged = _fresh_layers()
    trainer = _fake_trainer(config={"encoding": "v6"})
    payload = orchestrator.build_and_emit_resolved_config(
        layers, merged, registry_spec={}, trainer=trainer, args=_args(), log=_log(),
    )
    knobs = payload["knobs"]
    # seed/device/tf32 are the launch-only knobs merged in from Phase-A
    for k in ("seed", "device", "tf32"):
        assert k in knobs, f"missing Phase-A knob {k}"
        assert knobs[k].get("consumed_pre_resolution") is True
    # seed came from the base layer (42) — no checkpoint source on a pre-load knob
    assert knobs["seed"]["value"] == 42
    assert "checkpoint" not in knobs["seed"]["inputs_seen"]


def test_resume_ckpt_stamp_and_effective_lr_captured(recorder, tmp_path):
    # Resume: no variant encoding declaration → ckpt stamp is authoritative (no conflict). On the
    # live path init_trainer validates the checkpoint exists BEFORE emission; mirror that with a
    # real temp file so the bootstrap existence-validate inside resolve_run_config passes.
    ckpt = tmp_path / "some.pt"
    ckpt.write_bytes(b"x")
    layers = [
        {"label": "configs/model.yaml", "kind": "base", "raw": {"encoding": "v6", "seed": 42}},
        {"label": "configs/variants/x.yaml", "kind": "variant", "raw": {"evaluation": {}}},
    ]
    merged = merged_layers(layers)
    trainer = _fake_trainer(config={"encoding": {"version": "v6_live2_ls"}}, lr=0.0018569)
    payload = orchestrator.build_and_emit_resolved_config(
        layers, merged, registry_spec={}, trainer=trainer,
        args=_args(checkpoint=str(ckpt)), log=_log(),
    )
    enc = payload["knobs"]["encoding"]
    assert enc["value"] == "v6_live2_ls"
    assert enc["source"] == "checkpoint"
    # effective lr flowed from the live optimizer param_group into the LR provenance (a frozen
    # LrProvenance dataclass carried as the knob value).
    lr_prov = payload["knobs"]["lr"]["value"]
    assert lr_prov.effective == pytest.approx(0.0018569)


def test_empty_param_groups_guarded(recorder):
    layers, merged = _fresh_layers()
    trainer = _fake_trainer(config={"encoding": "v6"}, lr=None)  # empty param_groups
    payload = orchestrator.build_and_emit_resolved_config(
        layers, merged, registry_spec={}, trainer=trainer, args=_args(), log=_log(),
    )
    # No crash; lr effective is None (no state blob available).
    assert payload["knobs"]["lr"]["value"].effective is None


# ── BYTE-PURE: resolver conflict does NOT abort the launch (6a-ii, not 6b) ────
def test_resolver_conflict_does_not_abort_launch_emits_nothing(recorder):
    # variant DECLARES encoding v6_live2_ls, ckpt stamp v6_live2 → I2 conflict inside the resolver.
    layers = [
        {"label": "configs/model.yaml", "kind": "base", "raw": {"encoding": "v6"}},
        {"label": "configs/variants/x.yaml", "kind": "variant",
         "raw": {"encoding": "v6_live2_ls"}},
    ]
    merged = merged_layers(layers)
    trainer = _fake_trainer(config={"encoding": {"version": "v6_live2"}})
    # Must NOT raise (byte-pure: the live launch path is unaffected).
    payload = orchestrator.build_and_emit_resolved_config(
        layers, merged, registry_spec={}, trainer=trainer,
        args=_args(checkpoint="checkpoints/some.pt"), log=_log(),
    )
    assert payload == {}
    assert [c for c in recorder.calls if c.get("event") == "resolved_config"] == []


# ── assert_layers_reconstruct fires on a real base+variant load (F2) ──────────
def test_assert_layers_reconstruct_fires_on_real_load(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("encoding: v6\nseed: 42\n")
    variant = tmp_path / "variant.yaml"
    variant.write_text("encoding: v6_live2_ls\nn_sims: 150\n")
    layers = capture_config_layers(
        base_paths=[str(base)], config_override=None, variant_path=str(variant),
    )
    merged = merged_layers(layers)
    # Passes on a faithful merge.
    assert_layers_reconstruct(layers, merged)
    # Tamper the merged target → must raise naming the divergent key.
    merged["encoding"] = "TAMPERED"
    with pytest.raises(ValueError) as ei:
        assert_layers_reconstruct(layers, merged)
    assert "encoding" in str(ei.value)


# ── load_train_config: single-source kind-tagged layers reconstruct (F2) ──────
def test_load_train_config_returns_reconstructing_layers(monkeypatch, tmp_path):
    # Use a real --config override on top of the real base configs (repo-root relative). No variant
    # → no validation dependency. Exercises _build_load_paths → capture_config_layers single source.
    override = tmp_path / "override.yaml"
    override.write_text("seed: 7\n")
    args = _args(config=str(override))
    config, layers = orchestrator.load_train_config(args)
    # layers are kind-tagged (base/config), load-order.
    kinds = [ly["kind"] for ly in layers]
    assert kinds[-1] == "config"
    assert all(k in {"base", "config", "variant"} for k in kinds)
    # F2: the captured layers deep-merge back to exactly the merged config load_train_config returned.
    assert_layers_reconstruct(layers, config)
    # the --config override won (single source honoured the override position).
    assert config["seed"] == 7


def test_build_load_paths_dedups_base_config_and_orders():
    # --config == a base file → the base copy is dropped so config_key_overlap won't double-fire.
    args = _args(config="configs/model.yaml")
    base_paths, config_override, variant_path = orchestrator._build_load_paths(args)
    assert "configs/model.yaml" not in [p for p in base_paths]
    assert config_override == "configs/model.yaml"
    assert variant_path is None


# ── FIX 2: fresh run must not emit encoding.source="checkpoint" ───────────────
def test_fresh_run_encoding_source_is_not_checkpoint(recorder):
    """On a fresh run (args.checkpoint=None) the trainer's yaml-default encoding must NOT be
    attributed to the checkpoint — checkpoint_stamps must be {} so the resolver falls to the
    variant/default precedence family."""
    layers, merged = _fresh_layers(encoding="v6")
    # trainer.config holds the yaml default "v6" — same as _checkpoint_stamps_from_trainer would
    # return IF called unconditionally on a fresh run.
    trainer = _fake_trainer(config={"encoding": "v6"}, lr=0.002)
    payload = orchestrator.build_and_emit_resolved_config(
        layers, merged, registry_spec={}, trainer=trainer, args=_args(checkpoint=None), log=_log(),
    )
    enc = payload["knobs"]["encoding"]
    assert enc["source"] in {"variant", "default"}, (
        f"fresh-run encoding.source must not be 'checkpoint', got {enc['source']!r}"
    )
