"""CONFRES Phase-3 batch 6a-i — ResolvedRunConfig pure builder.

Exercises the aggregating builder ``resolve_run_config`` / ``resolve_preload_config`` and the
raw-layer capture helpers (``capture_config_layers`` / ``merged_layers`` /
``assert_layers_reconstruct``). Everything under test is a PURE function + frozen dataclass —
no launch wiring, no emission, no consumer migration (those are later sub-batches).

Frozen acceptance tests (design §10 mutation map):
- M3: string-form ``encoding`` ≡ dict-form (F1 dead by construction).
- M6: mutating the raw config dict after build does NOT change any accessor (frozen).
- M7: a missing required knob → ValueError at build.
- M2 (partial): ckpt stamp v6_live2 + PRESENT variant v6_live2_ls → ConfigConflictError naming both.
- absent-encoding (M2c/B5a): no ``encoding:`` key + ckpt stamp → resolves to stamp, source=checkpoint, no raise.
- layers invariant (F2): assert_layers_reconstruct passes for a real merge, raises naming the key on mismatch.
- provenance (B3): to_event_payload records BOTH inputs_seen for a variant≠checkpoint-baked knob.
"""
from __future__ import annotations

import pytest

from hexo_rl.config.resolve.run_config import (
    ConfigConflictError,
    ResolvedRunConfig,
    ResolvedValue,
    assert_layers_reconstruct,
    capture_config_layers,
    merged_layers,
    resolve_preload_config,
    resolve_run_config,
)


# ── shared fixtures ──────────────────────────────────────────────────────────
def _registry():
    # The builder only needs the registry as a passthrough context object; a dict suffices.
    return {}


def _base_layers(encoding_value=None):
    """A minimal base→variant layer chain. ``encoding_value`` set on the variant layer if given."""
    variant_raw: dict = {"evaluation": {"eval_temperature": 0.7}}
    if encoding_value is not None:
        variant_raw["encoding"] = encoding_value
    return [
        {"label": "configs/model.yaml", "raw": {"encoding": "v6", "seed": 42}},
        {"label": "configs/variants/x.yaml", "raw": variant_raw},
    ]


def _combined_from(layers):
    return merged_layers(layers)


# ── M3: string ≡ dict encoding ───────────────────────────────────────────────
def test_m3_string_and_dict_encoding_resolve_identically():
    str_layers = _base_layers(encoding_value="v6_live2_ls")
    dict_layers = _base_layers(encoding_value={"version": "v6_live2_ls"})
    str_cfg = resolve_run_config(
        _registry(), str_layers, _combined_from(str_layers),
        checkpoint_stamps={}, checkpoint_state={}, cli={},
    )
    dict_cfg = resolve_run_config(
        _registry(), dict_layers, _combined_from(dict_layers),
        checkpoint_stamps={}, checkpoint_state={}, cli={},
    )
    assert str_cfg.encoding_name() == "v6_live2_ls"
    assert dict_cfg.encoding_name() == "v6_live2_ls"
    assert str_cfg.encoding_name() == dict_cfg.encoding_name()


# ── M6: frozen — post-build mutation of raw dicts is inert ───────────────────
def test_m6_post_build_mutation_does_not_change_accessor():
    layers = _base_layers(encoding_value="v6_live2_ls")
    combined = _combined_from(layers)
    cfg = resolve_run_config(
        _registry(), layers, combined,
        checkpoint_stamps={}, checkpoint_state={}, cli={},
    )
    before = cfg.encoding_name()
    # Mutate every raw source AFTER the build.
    combined["encoding"] = "v8"
    layers[-1]["raw"]["encoding"] = "v8"
    layers[0]["raw"]["encoding"] = "v8"
    assert cfg.encoding_name() == before == "v6_live2_ls"


def test_m6_resolved_value_is_frozen():
    rv = ResolvedValue(value="v6", source="variant", precedence_family="variant-wins", inputs_seen={})
    with pytest.raises((AttributeError, TypeError)):
        rv.value = "v8"  # type: ignore[misc]


# ── M7: missing required knob → ValueError at build ──────────────────────────
def test_m7_missing_required_knob_raises_valueerror():
    # A knob declared REQUIRED that resolves only to its default (no real source: no variant decl,
    # no base, no stamp) → ValueError at build (I5). Encoding absent-everywhere resolves to the
    # "v6" compat default, which is NOT a real source — requiring it makes the missing case loud.
    layers = [{"label": "configs/variants/x.yaml", "raw": {"evaluation": {}}}]
    combined = _combined_from(layers)
    with pytest.raises(ValueError) as ei:
        resolve_run_config(
            _registry(), layers, combined,
            checkpoint_stamps={}, checkpoint_state={}, cli={},
            require=("encoding",),
        )
    assert "encoding" in str(ei.value)


def test_m7_required_knob_present_does_not_raise():
    # Same required set, but the variant DECLARES encoding → resolves from a real source, no raise.
    layers = _base_layers(encoding_value="v6_live2_ls")
    combined = _combined_from(layers)
    cfg = resolve_run_config(
        _registry(), layers, combined,
        checkpoint_stamps={}, checkpoint_state={}, cli={},
        require=("encoding",),
    )
    assert cfg.encoding_name() == "v6_live2_ls"


# ── M2 (partial): present variant ≠ ckpt stamp → ConfigConflictError ─────────
def test_m2_present_variant_conflicts_with_ckpt_stamp():
    layers = _base_layers(encoding_value="v6_live2_ls")
    combined = _combined_from(layers)
    with pytest.raises(ConfigConflictError) as ei:
        resolve_run_config(
            _registry(), layers, combined,
            checkpoint_stamps={"encoding": "v6_live2"}, checkpoint_state={}, cli={},
        )
    msg = str(ei.value)
    assert "v6_live2_ls" in msg and "v6_live2" in msg
    assert "encoding" in msg


def test_m2_conflict_error_is_valueerror_lineage():
    assert issubclass(ConfigConflictError, ValueError)


# ── absent-encoding (B5a / M2c): absent + stamp → stamp wins, no raise ───────
def test_absent_encoding_resolves_to_stamp_source_checkpoint():
    # No `encoding:` key on the VARIANT layer (base declares v6, inherited-not-a-declaration).
    layers = [
        {"label": "configs/model.yaml", "raw": {"seed": 42}},  # NO encoding anywhere in yaml
        {"label": "configs/variants/no_enc.yaml", "raw": {"evaluation": {}}},
    ]
    combined = _combined_from(layers)
    cfg = resolve_run_config(
        _registry(), layers, combined,
        checkpoint_stamps={"encoding": "v6_live2"}, checkpoint_state={}, cli={},
    )
    assert cfg.encoding_name() == "v6_live2"
    prov = cfg.provenance("encoding")
    assert prov.source == "checkpoint"
    assert prov.value == "v6_live2"


def test_absent_encoding_present_base_still_inherits_not_declares():
    # base model.yaml declares encoding: v6 — under variant-presence this is INHERITED, not a
    # variant declaration, so a non-v6 stamp must NOT raise (B5a-caveat).
    layers = [
        {"label": "configs/model.yaml", "raw": {"encoding": "v6"}},
        {"label": "configs/variants/no_enc.yaml", "raw": {"evaluation": {}}},
    ]
    combined = _combined_from(layers)
    cfg = resolve_run_config(
        _registry(), layers, combined,
        checkpoint_stamps={"encoding": "v6_live2"}, checkpoint_state={}, cli={},
    )
    assert cfg.encoding_name() == "v6_live2"
    assert cfg.provenance("encoding").source == "checkpoint"


def test_absent_encoding_fresh_run_falls_to_v6_default():
    # Absent everywhere, NO stamp (fresh run) → the "v6" compat default, source=default.
    layers = [{"label": "configs/model.yaml", "raw": {"encoding": "v6"}},
              {"label": "configs/variants/x.yaml", "raw": {}}]
    combined = _combined_from(layers)
    cfg = resolve_run_config(
        _registry(), layers, combined,
        checkpoint_stamps={}, checkpoint_state={}, cli={},
    )
    assert cfg.encoding_name() == "v6"


# ── layers invariant (F2) ────────────────────────────────────────────────────
def test_assert_layers_reconstruct_passes_on_real_merge():
    layers = _base_layers(encoding_value="v6")
    combined = _combined_from(layers)
    # Must not raise.
    assert_layers_reconstruct(layers, combined)


def test_assert_layers_reconstruct_raises_naming_key_on_mismatch():
    layers = _base_layers(encoding_value="v6")
    combined = _combined_from(layers)
    combined["encoding"] = "TAMPERED"  # deliberate divergence from the merged chain
    with pytest.raises(ValueError) as ei:
        assert_layers_reconstruct(layers, combined)
    assert "encoding" in str(ei.value)


def test_merged_layers_later_wins_deep_merge():
    layers = [
        {"label": "base", "raw": {"a": {"x": 1, "y": 2}, "b": 1}},
        {"label": "variant", "raw": {"a": {"y": 9}, "c": 3}},
    ]
    merged = merged_layers(layers)
    assert merged == {"a": {"x": 1, "y": 9}, "b": 1, "c": 3}


# ── provenance / to_event_payload (B3) ───────────────────────────────────────
def test_to_event_payload_records_both_inputs_for_variant_vs_ckpt_baked():
    # n_sims eval: variant declares 64, checkpoint-baked config carried 128 → both in inputs_seen.
    layers = [
        {"label": "configs/model.yaml", "raw": {"encoding": "v6"}},
        {"label": "configs/variants/x.yaml",
         "raw": {"evaluation": {"sealbot_model_sims": 64}}},
    ]
    combined = _combined_from(layers)
    cfg = resolve_run_config(
        _registry(), layers, combined,
        checkpoint_stamps={},
        checkpoint_state={},
        cli={},
        checkpoint_baked={"evaluation": {"sealbot_model_sims": 128}},
    )
    payload = cfg.to_event_payload()
    assert payload["event"] == "resolved_config"
    knob = payload["knobs"]["n_sims_eval.sealbot"]
    assert knob["value"] == 64
    assert knob["source"] == "variant"
    assert knob["precedence_family"] == "variant-wins"
    assert knob["inputs_seen"]["variant"] == 64
    assert knob["inputs_seen"]["checkpoint"] == 128


def test_to_event_payload_renders_all_resolvable_knobs():
    layers = _base_layers(encoding_value="v6_live2_ls")
    combined = _combined_from(layers)
    cfg = resolve_run_config(
        _registry(), layers, combined,
        checkpoint_stamps={}, checkpoint_state={}, cli={},
    )
    payload = cfg.to_event_payload()
    knobs = payload["knobs"]
    assert "encoding" in knobs
    assert "eval_temperature" in knobs
    for spec in knobs.values():
        assert set(spec) >= {"value", "source", "precedence_family", "inputs_seen"}


# ── typed accessors + aggregation of existing resolvers ──────────────────────
def test_accessors_delegate_to_existing_resolvers():
    layers = [
        {"label": "configs/model.yaml", "raw": {"encoding": "v6"}},
        {"label": "configs/variants/x.yaml",
         "raw": {"evaluation": {"eval_temperature": 0.9, "random_model_sims": 42}}},
    ]
    combined = _combined_from(layers)
    cfg = resolve_run_config(
        _registry(), layers, combined,
        checkpoint_stamps={}, checkpoint_state={}, cli={},
    )
    assert cfg.eval_temperature() == 0.9
    assert cfg.n_sims_eval("random") == 42
    assert cfg.n_sims_eval("sealbot") == 128  # default authority


def test_n_sims_eval_default_when_absent():
    layers = _base_layers(encoding_value="v6")
    combined = _combined_from(layers)
    cfg = resolve_run_config(
        _registry(), layers, combined,
        checkpoint_stamps={}, checkpoint_state={}, cli={},
    )
    assert cfg.n_sims_eval("random") == 96
    assert cfg.eval_temperature() == 0.7  # from variant fixture


def test_bootstrap_path_none_for_fresh_run():
    layers = _base_layers(encoding_value="v6")
    combined = _combined_from(layers)
    cfg = resolve_run_config(
        _registry(), layers, combined,
        checkpoint_stamps={}, checkpoint_state={}, cli={"checkpoint": None},
    )
    assert cfg.bootstrap_path() is None
    assert cfg.provenance("bootstrap").source == "default"


def test_bootstrap_path_from_cli_checkpoint(tmp_path):
    ckpt = tmp_path / "boot.pt"
    ckpt.write_bytes(b"x")
    layers = _base_layers(encoding_value="v6")
    combined = _combined_from(layers)
    cfg = resolve_run_config(
        _registry(), layers, combined,
        checkpoint_stamps={}, checkpoint_state={}, cli={"checkpoint": str(ckpt)},
    )
    assert cfg.bootstrap_path() == str(ckpt)
    assert cfg.provenance("bootstrap").source == "cli"


def test_lr_provenance_accessor_from_checkpoint_state():
    layers = _base_layers(encoding_value="v6")
    combined = _combined_from(layers)
    # variant declares lr 0.001, ckpt baked 0.002 → override ignored on resume.
    layers[-1]["raw"]["lr"] = 0.001
    combined = _combined_from(layers)
    cfg = resolve_run_config(
        _registry(), layers, combined,
        checkpoint_stamps={},
        checkpoint_state={"optimizer_state": {"param_groups": [{"lr": 0.0018569}]}},
        cli={},
        checkpoint_baked={"lr": 0.002},
    )
    prov = cfg.lr_provenance()
    assert prov.declared == 0.001
    assert prov.baked == 0.002
    assert prov.effective == 0.0018569
    assert prov.override_ignored is True


# ── I7 two-phase preload split (F3) ──────────────────────────────────────────
def test_resolve_preload_config_only_launch_knobs():
    layers = [
        {"label": "configs/training.yaml", "raw": {"seed": 42, "tf32": {"tf32_matmul": "auto"}}},
        {"label": "configs/variants/x.yaml", "raw": {"seed": 7}},
    ]
    pre = resolve_preload_config(layers, cli={})
    assert pre.provenance("seed").value == 7  # variant-wins
    assert pre.provenance("seed").precedence_family == "variant-wins"
    # Pre-load knobs carry NO checkpoint source (I7).
    assert "checkpoint" not in pre.provenance("seed").inputs_seen


def test_resolve_preload_config_flags_consumed_pre_resolution():
    layers = [{"label": "configs/training.yaml", "raw": {"seed": 42}}]
    pre = resolve_preload_config(layers, cli={})
    payload = pre.to_event_payload()
    assert payload["knobs"]["seed"]["consumed_pre_resolution"] is True


def test_resolve_preload_config_cli_overrides_variant_seed():
    layers = [{"label": "configs/training.yaml", "raw": {"seed": 42}}]
    pre = resolve_preload_config(layers, cli={"seed": 99})
    assert pre.provenance("seed").value == 99
    assert pre.provenance("seed").source == "cli"


def test_resolve_preload_config_seed_default():
    layers = [{"label": "configs/training.yaml", "raw": {}}]
    pre = resolve_preload_config(layers, cli={})
    assert pre.provenance("seed").value == 42  # default


# ── capture_config_layers (F2/F4): load-order raw chain ──────────────────────
def test_capture_config_layers_load_order(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("encoding: v6\nseed: 42\n")
    variant = tmp_path / "variant.yaml"
    variant.write_text("encoding: v6_live2_ls\n")
    override = tmp_path / "override.yaml"
    override.write_text("seed: 7\n")
    layers = capture_config_layers(
        base_paths=[str(base)], config_override=str(override), variant_path=str(variant),
    )
    assert [ly["label"] for ly in layers] == [str(base), str(override), str(variant)]
    assert layers[0]["raw"] == {"encoding": "v6", "seed": 42}
    assert layers[-1]["raw"] == {"encoding": "v6_live2_ls"}
    # The merged chain reconstructs (F2 invariant).
    merged = merged_layers(layers)
    assert merged == {"encoding": "v6_live2_ls", "seed": 7}


def test_capture_config_layers_no_override_no_variant(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("encoding: v6\n")
    layers = capture_config_layers(base_paths=[str(base)], config_override=None, variant_path=None)
    assert [ly["label"] for ly in layers] == [str(base)]


# ── FIX 1 regression: --config layer IS a declaration layer (timebomb closed) ──
def test_config_override_layer_is_a_declaration():
    """A 3-layer chain [base, --config, variant] where --config declares encoding must be seen.

    The old filename-sniff heuristic would miss the ``--config`` layer (no "variants/" in the
    path), so a ``--config`` file declaring a different encoding than the checkpoint stamp would be
    silently ignored and the stamp would win — a silent override.  The kind-tagged path closes this.
    """
    layers = [
        {"label": "configs/model.yaml", "kind": "base", "raw": {"encoding": "v6"}},
        {"label": "configs/override.yaml", "kind": "config", "raw": {"encoding": "v6_live2_ls"}},
        {"label": "configs/variants/x.yaml", "kind": "variant", "raw": {"n_sims": 150}},
    ]
    combined = merged_layers(layers)
    # checkpoint stamp v6 conflicts with the --config declaration v6_live2_ls → must raise.
    with pytest.raises(ConfigConflictError) as ei:
        resolve_run_config(
            _registry(), layers, combined,
            checkpoint_stamps={"encoding": "v6"},
            checkpoint_state={},
            cli={},
        )
    msg = str(ei.value)
    assert "v6_live2_ls" in msg and "v6" in msg
    assert "encoding" in msg


# ── FIX 2 regression: inputs_seen snapshot is immune to post-build dict mutation ──
def test_inputs_seen_snapshot_immune_to_post_build_mutation():
    """Mutating the dict passed as inputs_seen after construction must NOT alter rv.inputs_seen."""
    original = {"variant": "v6_live2_ls", "checkpoint": "v6"}
    rv = ResolvedValue(
        value="v6_live2_ls",
        source="variant",
        precedence_family="raise-on-conflict",
        inputs_seen=original,
    )
    # Snapshot value before mutation.
    before = dict(rv.inputs_seen)
    # Mutate the original dict passed in.
    original["variant"] = "TAMPERED"
    original["injected"] = "extra"
    # rv.inputs_seen must be unchanged.
    assert rv.inputs_seen == before


# ── FIX 1 regression: variant+stamp agree → source==variant, no raise ──────────
def test_encoding_agree_variant_and_stamp_source_is_variant():
    """variant encoding == stamp → resolves cleanly, source recorded as 'variant'."""
    layers = [
        {"label": "configs/model.yaml", "raw": {"encoding": "v6"}},
        {"label": "configs/variants/x.yaml", "raw": {"encoding": "v6_live2_ls"}},
    ]
    combined = merged_layers(layers)
    cfg = resolve_run_config(
        _registry(), layers, combined,
        checkpoint_stamps={"encoding": "v6_live2_ls"},
        checkpoint_state={},
        cli={},
    )
    assert cfg.encoding_name() == "v6_live2_ls"
    prov = cfg.provenance("encoding")
    assert prov.source == "variant"
