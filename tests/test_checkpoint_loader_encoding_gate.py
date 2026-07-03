"""D-EVALGATE G1 — port of the trainer-side D-FORENSIC F1 encoding gate to
the eval path (``hexo_rl.eval.checkpoint_loader.load_model_with_encoding``).

D-FORENSIC F1: the whole d1m lineage self-played single-window ``v6_live2``
for 272k+ steps while every variant declared multi-window ``v6_live2_ls`` —
two stacked holes on the trainer side: (1) the canonical STRING form
``encoding: "v6_live2_ls"`` was treated as "unspecified" by a dict-only
check, letting filename/shape inference win silently; (2) the checkpoint's
own ``metadata['encoding_name']`` stamp was preferred unconditionally,
self-perpetuating a stale/wrong resolution through every resume. Both holes
were fixed on the trainer side (`hexo_rl/training/trainer_ckpt_load.py`,
see `tests/test_trainer_encoding_load.py`) but the SAME ambiguity class was
still wide open on the eval path — the exact instruments used for SealBot
eval, exploit_probe, and promotion-eval read checkpoints via
``load_model_with_encoding`` with zero declared-vs-stamp reconciliation.

This module pins the ported gate. Mirrors (does not duplicate) the
trainer-side test shapes in ``tests/test_trainer_encoding_load.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hexo_rl.encoding import lookup as registry_lookup
from hexo_rl.eval.checkpoint_loader import (
    DeclaredEncodingMismatchError,
    load_model_with_encoding,
)
from hexo_rl.eval.eval_pipeline import _load_anchor_model
from hexo_rl.model.network import HexTacToeNet

_FAST_RES_BLOCKS = 2
_FAST_FILTERS = 16


def _v6_live2_model() -> HexTacToeNet:
    """4-plane / 19x19 / policy=362 — v6_live2 and v6_live2_ls are
    state-dict-shape-IDENTICAL (window fields only), so shape inference
    cannot disambiguate them; this is the crux of the D-FORENSIC F1 bug."""
    return HexTacToeNet(
        board_size=19,
        in_channels=4,
        filters=_FAST_FILTERS,
        res_blocks=_FAST_RES_BLOCKS,
        encoding="v6_live2",
    )


def _save_weights_only(path: Path) -> Path:
    """Bare state_dict — no config, no metadata. Mirrors a bootstrap
    ``.pt`` (e.g. bootstrap_model_v6_live2_8300.pt): no independently
    trustworthy stamp exists, shape/filename inference is the only source."""
    torch.save(_v6_live2_model().state_dict(), path)
    return path


def _save_full_ckpt_with_metadata(path: Path, encoding_name: str) -> Path:
    """Full checkpoint carrying metadata['encoding_name'] — mirrors the
    real d1m lineage artifact shape (checkpoint_00272357.pt)."""
    payload = {
        "model_state": _v6_live2_model().state_dict(),
        "metadata": {"encoding_name": encoding_name, "schema_version": 1},
    }
    torch.save(payload, path)
    return path


def _save_full_ckpt_with_config_encoding(path: Path, encoding) -> Path:
    """Full checkpoint with config['encoding'] but NO metadata dict —
    the pre-A5-migration shape."""
    payload = {
        "model_state": _v6_live2_model().state_dict(),
        "config": {"encoding": encoding},
    }
    torch.save(payload, path)
    return path


def _save_full_ckpt_with_raw_encoding(
    path: Path, encoding, *, metadata_encoding: str | None = None,
) -> Path:
    """Full checkpoint carrying a top-level ``raw['encoding']`` field —
    mirrors ``hexo_rl.training.anchor.save_best_model_atomic``'s promoted-
    anchor payload shape (D-EVALGATE fix wave review point 2: this field
    used to be consumed as a priority-1 label BEFORE reconciliation). When
    ``metadata_encoding`` is given, also stamps ``metadata['encoding_name']``
    (to test mutual stamp-source agreement/disagreement)."""
    payload: dict = {
        "model_state": _v6_live2_model().state_dict(),
        "encoding": encoding,
    }
    if metadata_encoding is not None:
        payload["metadata"] = {"encoding_name": metadata_encoding}
    torch.save(payload, path)
    return path


DEVICE = torch.device("cpu")


# ── Baseline: declared_encoding=None preserves prior behaviour ──────────


def test_no_declared_no_stamp_falls_back_to_shape_inference(tmp_path: Path) -> None:
    """Backward compat: no declaration, no gate call site opted in — the
    last-resort shape/filename inference still resolves exactly as before."""
    ckpt_path = _save_weights_only(tmp_path / "model_generic_name.pt")
    model, spec, label = load_model_with_encoding(ckpt_path, DEVICE)
    assert label == "v6_live2"
    assert spec.name == "v6_live2"
    assert isinstance(model, HexTacToeNet)


# ── Explicit declaration, string AND dict form ───────────────────────────


@pytest.mark.parametrize(
    "declared", ["v6_live2", {"version": "v6_live2"}, {"name": "v6_live2"}],
    ids=["string-form", "dict-version-form", "dict-name-form"],
)
def test_declared_encoding_agreeing_with_shape_loads(tmp_path: Path, declared) -> None:
    """String form (§172 A4.5 canonical) and dict form both count as an
    EXPLICIT declaration — mirrors the trainer-side D-FORENSIC F1 fix that
    closed the dict-only isinstance check."""
    ckpt_path = _save_weights_only(tmp_path / "model_generic_name.pt")
    model, spec, label = load_model_with_encoding(
        ckpt_path, DEVICE, declared_encoding=declared,
    )
    assert label == "v6_live2"
    assert spec.name == "v6_live2"


def test_declared_v6_live2_ls_with_no_stamp_resolves_ls(tmp_path: Path) -> None:
    """No checkpoint stamp to compare against → declared name is
    authoritative (the documented exploit_probe/run_sealbot_eval
    ``--encoding`` override escape hatch for shape-ambiguous checkpoints
    survives — it is a DIFFERENT case from a present-but-disagreeing
    stamp, which DOES raise below)."""
    ckpt_path = _save_weights_only(tmp_path / "model_generic_name.pt")
    model, spec, label = load_model_with_encoding(
        ckpt_path, DEVICE, declared_encoding="v6_live2_ls",
    )
    assert label == "v6_live2_ls"
    assert spec.name == "v6_live2_ls"


# ── D-FORENSIC F1 regression: metadata stamp vs explicit declaration ─────


def test_d1m_regression_metadata_v6_live2_vs_declared_v6_live2_ls_raises(
    tmp_path: Path,
) -> None:
    """THE regression test: a checkpoint stamped (via metadata) single-
    window v6_live2 — exactly the self-perpetuated d1m lineage stamp —
    loaded with an explicit multi-window v6_live2_ls declaration (the
    variant's canonical string-form intent) must RAISE, not silently pick
    either side."""
    ckpt_path = _save_full_ckpt_with_metadata(
        tmp_path / "checkpoint_00272357.pt", "v6_live2",
    )
    with pytest.raises(DeclaredEncodingMismatchError) as excinfo:
        load_model_with_encoding(
            ckpt_path, DEVICE, declared_encoding="v6_live2_ls",
        )
    msg = str(excinfo.value)
    assert "v6_live2_ls" in msg
    assert "v6_live2" in msg
    assert "metadata" in msg


@pytest.mark.parametrize(
    "declared", ["v6_live2_ls", {"version": "v6_live2_ls"}],
    ids=["string-form", "dict-form"],
)
def test_metadata_mismatch_raises_both_declared_forms(tmp_path: Path, declared) -> None:
    ckpt_path = _save_full_ckpt_with_metadata(
        tmp_path / "checkpoint_stamped.pt", "v6_live2",
    )
    with pytest.raises(DeclaredEncodingMismatchError):
        load_model_with_encoding(ckpt_path, DEVICE, declared_encoding=declared)


def test_metadata_stamp_agreeing_with_declaration_loads_and_resolves_ls(
    tmp_path: Path,
) -> None:
    """Sanctioned re-stamp workflow (mirrors scripts/make_ws3v3_warmstart.py):
    metadata correctly stamped v6_live2_ls + declared v6_live2_ls → loads
    cleanly and resolves the MULTI-window spec, not the shape-identical
    single-window one."""
    ckpt_path = _save_full_ckpt_with_metadata(
        tmp_path / "checkpoint_ls.pt", "v6_live2_ls",
    )
    model, spec, label = load_model_with_encoding(
        ckpt_path, DEVICE, declared_encoding="v6_live2_ls",
    )
    assert label == "v6_live2_ls"
    assert spec.name == "v6_live2_ls"


def test_config_encoding_stamp_vs_declared_mismatch_raises(tmp_path: Path) -> None:
    """No metadata dict, but a baked config['encoding'] stamp (dict form)
    disagreeing with the declaration must also raise — config['encoding']
    is the second trusted-stamp source (mirrors trainer priority order)."""
    ckpt_path = _save_full_ckpt_with_config_encoding(
        tmp_path / "checkpoint_cfg_stamped.pt", {"version": "v6_live2"},
    )
    with pytest.raises(DeclaredEncodingMismatchError) as excinfo:
        load_model_with_encoding(
            ckpt_path, DEVICE, declared_encoding="v6_live2_ls",
        )
    msg = str(excinfo.value)
    assert "v6_live2_ls" in msg
    assert "v6_live2" in msg
    assert "config['encoding']" in msg


# ── Registry-by-name only: shape-sniff never overrides a present name ────


def test_no_declaration_metadata_stamp_still_authoritative_over_shape(
    tmp_path: Path,
) -> None:
    """Even with NO caller declaration, a present checkpoint stamp must
    win over shape/filename inference — shape sniffing alone can never
    distinguish v6_live2 from v6_live2_ls, so silently trusting it (the
    pre-fix eval-path behaviour) is exactly the D-FORENSIC F1 hole."""
    ckpt_path = _save_full_ckpt_with_metadata(
        tmp_path / "checkpoint_ls_no_declare.pt", "v6_live2_ls",
    )
    model, spec, label = load_model_with_encoding(ckpt_path, DEVICE)
    assert label == "v6_live2_ls"
    assert spec.name == "v6_live2_ls"


def test_metadata_wins_when_no_declaration_single_window(tmp_path: Path) -> None:
    """No-declaration backward compat, single-window case (mirrors
    trainer's `test_metadata_encoding_wins_when_config_declares_nothing`)."""
    ckpt_path = _save_full_ckpt_with_metadata(
        tmp_path / "checkpoint_sw_no_declare.pt", "v6_live2",
    )
    model, spec, label = load_model_with_encoding(ckpt_path, DEVICE)
    assert label == "v6_live2"
    assert spec.name == "v6_live2"


# ── Part 3 — pre-flight hard gate for "no declared + no stamp" ──────────


def test_require_encoding_source_raises_with_neither_declared_nor_stamped(
    tmp_path: Path,
) -> None:
    """§D-FORENSIC F1 follow-up: ~12/61 variants declare no `encoding:`.
    Opt-in pre-flight gate refuses the silent shape/filename fallback
    outright when NEITHER side names an encoding."""
    ckpt_path = _save_weights_only(tmp_path / "model_unstamped.pt")
    with pytest.raises(DeclaredEncodingMismatchError) as excinfo:
        load_model_with_encoding(
            ckpt_path, DEVICE, require_encoding_source=True,
        )
    assert "require_encoding_source" in str(excinfo.value)


def test_require_encoding_source_passes_with_declared(tmp_path: Path) -> None:
    ckpt_path = _save_weights_only(tmp_path / "model_unstamped.pt")
    model, spec, label = load_model_with_encoding(
        ckpt_path, DEVICE,
        declared_encoding="v6_live2",
        require_encoding_source=True,
    )
    assert label == "v6_live2"


def test_require_encoding_source_passes_with_stamp_only(tmp_path: Path) -> None:
    ckpt_path = _save_full_ckpt_with_metadata(
        tmp_path / "checkpoint_stamped_only.pt", "v6_live2",
    )
    model, spec, label = load_model_with_encoding(
        ckpt_path, DEVICE, require_encoding_source=True,
    )
    assert label == "v6_live2"


def test_require_encoding_source_default_off_does_not_raise(tmp_path: Path) -> None:
    """Default False — every existing call site (none pass this kwarg
    today) is completely unaffected."""
    ckpt_path = _save_weights_only(tmp_path / "model_unstamped.pt")
    model, spec, label = load_model_with_encoding(ckpt_path, DEVICE)
    assert label == "v6_live2"


# ── Sanity: the new spec branch resolves correctly, not the v6 default ──


def test_v6_live2_ls_spec_matches_registry(tmp_path: Path) -> None:
    ckpt_path = _save_full_ckpt_with_metadata(
        tmp_path / "checkpoint_ls_spec_check.pt", "v6_live2_ls",
    )
    _model, spec, _label = load_model_with_encoding(
        ckpt_path, DEVICE, declared_encoding="v6_live2_ls",
    )
    reg = registry_lookup("v6_live2_ls")
    assert spec.n_planes == reg.n_planes == 4
    assert spec.policy_logit_count == reg.policy_logit_count == 362
    assert spec.board_size == reg.board_size == 19


# ── D-EVALGATE fix wave: decode_override (decode-time cross-decode) ──────


def test_decode_override_cross_decode_no_raise_resolves_override(tmp_path: Path) -> None:
    """The D-DECODE workflow this whole fix wave exists to un-break: a
    checkpoint stamped single-window v6_live2, re-decoded as v6_live2_ls via
    decode_override. Must NOT raise (declared_encoding would); label is the
    override, and a structured warning is emitted for the stamp disagreement."""
    ckpt_path = _save_full_ckpt_with_metadata(
        tmp_path / "checkpoint_decode_override.pt", "v6_live2",
    )
    model, spec, label = load_model_with_encoding(
        ckpt_path, DEVICE, decode_override="v6_live2_ls",
    )
    assert label == "v6_live2_ls"
    assert spec.name == "v6_live2_ls"


def test_decode_override_agreeing_with_stamp_no_warning_needed(tmp_path: Path) -> None:
    """decode_override agreeing with the stamp still resolves cleanly (info,
    not warning — no disagreement to flag)."""
    ckpt_path = _save_full_ckpt_with_metadata(
        tmp_path / "checkpoint_decode_override_agree.pt", "v6_live2_ls",
    )
    model, spec, label = load_model_with_encoding(
        ckpt_path, DEVICE, decode_override="v6_live2_ls",
    )
    assert label == "v6_live2_ls"


def test_decode_override_with_no_stamp_resolves_override(tmp_path: Path) -> None:
    """No stamp to compare against — decode_override is still authoritative
    (same escape hatch as declared_encoding's no-stamp case, just via the
    non-raising kwarg)."""
    ckpt_path = _save_weights_only(tmp_path / "model_generic_name.pt")
    model, spec, label = load_model_with_encoding(
        ckpt_path, DEVICE, decode_override="v6_live2_ls",
    )
    assert label == "v6_live2_ls"


def test_declared_encoding_and_decode_override_together_raises_valueerror(
    tmp_path: Path,
) -> None:
    """Mutually exclusive — passing both is a caller bug, not a reconcilable
    state (one is an assertion, the other a deliberate override)."""
    ckpt_path = _save_weights_only(tmp_path / "model_generic_name.pt")
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_model_with_encoding(
            ckpt_path, DEVICE,
            declared_encoding="v6_live2",
            decode_override="v6_live2_ls",
        )


# ── D-EVALGATE fix wave: raw['encoding'] as a third stamp source ─────────


def test_raw_encoding_stamp_vs_declared_mismatch_raises(tmp_path: Path) -> None:
    """Pins the MEDIUM bypass: the legacy top-level raw['encoding'] field
    (written by hexo_rl.training.anchor's promoted-anchor payload) must be
    folded into stamp reconciliation, not consumed as a priority-1 label
    that bypasses the whole gate."""
    ckpt_path = _save_full_ckpt_with_raw_encoding(
        tmp_path / "checkpoint_raw_encoding.pt", "v6_live2",
    )
    with pytest.raises(DeclaredEncodingMismatchError) as excinfo:
        load_model_with_encoding(
            ckpt_path, DEVICE, declared_encoding="v6_live2_ls",
        )
    msg = str(excinfo.value)
    assert "v6_live2_ls" in msg
    assert "v6_live2" in msg
    assert "raw['encoding']" in msg


def test_raw_encoding_agreeing_with_metadata_loads_cleanly(tmp_path: Path) -> None:
    """metadata['encoding_name'] and raw['encoding'] both present and AGREEING
    (the real anchor.py shape — both written from one variable) loads fine."""
    ckpt_path = _save_full_ckpt_with_raw_encoding(
        tmp_path / "checkpoint_raw_and_meta_agree.pt", "v6_live2_ls",
        metadata_encoding="v6_live2_ls",
    )
    model, spec, label = load_model_with_encoding(ckpt_path, DEVICE)
    assert label == "v6_live2_ls"
    assert spec.name == "v6_live2_ls"


def test_metadata_vs_raw_encoding_mutual_disagreement_raises(tmp_path: Path) -> None:
    """Present stamp sources disagreeing WITH EACH OTHER (not just vs a
    declaration) is checkpoint corruption — anchor.py writes both from one
    variable, so this can only happen via corruption/hand-editing. No caller
    declaration is even needed to trigger this."""
    ckpt_path = _save_full_ckpt_with_raw_encoding(
        tmp_path / "checkpoint_raw_meta_disagree.pt", "v6_live2",
        metadata_encoding="v6_live2_ls",
    )
    with pytest.raises(ValueError) as excinfo:
        load_model_with_encoding(ckpt_path, DEVICE)
    msg = str(excinfo.value)
    assert "disagree" in msg
    assert "v6_live2_ls" in msg
    assert "v6_live2" in msg


# ── D-EVALGATE fix wave: malformed encoding values, symmetric both sides ──


def test_malformed_declared_encoding_raises_valueerror(tmp_path: Path) -> None:
    """A malformed declared_encoding dict (name/version present but not a
    string) used to raise the registry's own EncodingRegistryError uncaught;
    now raises a ValueError-lineage error with context, same as the stamp
    side below."""
    ckpt_path = _save_weights_only(tmp_path / "model_generic_name.pt")
    with pytest.raises(ValueError, match="malformed declared"):
        load_model_with_encoding(
            ckpt_path, DEVICE, declared_encoding={"version": 123},
        )


def test_malformed_config_encoding_stamp_raises_valueerror(tmp_path: Path) -> None:
    """A malformed config['encoding'] stamp used to be silently downgraded to
    (None, None) (falling through to shape/filename inference) — review
    point 8's asymmetry. Now raises the same ValueError-lineage error the
    declared side does."""
    ckpt_path = _save_full_ckpt_with_config_encoding(
        tmp_path / "checkpoint_malformed_cfg_stamp.pt", {"version": 123},
    )
    with pytest.raises(ValueError, match="malformed checkpoint stamp"):
        load_model_with_encoding(ckpt_path, DEVICE)


def test_malformed_decode_override_raises_valueerror(tmp_path: Path) -> None:
    ckpt_path = _save_weights_only(tmp_path / "model_generic_name.pt")
    with pytest.raises(ValueError, match="malformed decode_override"):
        load_model_with_encoding(
            ckpt_path, DEVICE, decode_override={"name": 123},
        )


# ── D-EVALGATE fix wave: ordering — mismatch raise wins over require_encoding_source ──


def test_require_encoding_source_true_with_mismatched_declaration_raises_mismatch(
    tmp_path: Path,
) -> None:
    """require_encoding_source=True + a declared_encoding that DISAGREES with
    a present stamp: the mismatch raise must fire (not the require_encoding_
    source "no source at all" raise, which would be the wrong diagnostic —
    a source WAS given, it just disagrees)."""
    ckpt_path = _save_full_ckpt_with_metadata(
        tmp_path / "checkpoint_ordering.pt", "v6_live2",
    )
    with pytest.raises(DeclaredEncodingMismatchError) as excinfo:
        load_model_with_encoding(
            ckpt_path, DEVICE,
            declared_encoding="v6_live2_ls",
            require_encoding_source=True,
        )
    assert "disagrees" in str(excinfo.value)


# ── D-EVALGATE fix wave: anchor-loader gate (red-team HOLE 3) ─────────────


def test_anchor_loader_unstamped_no_declaration_raises_with_require_source(
    tmp_path: Path,
) -> None:
    """eval_pipeline._load_anchor_model, wired with require_encoding_source=True
    and no declared_encoding (mirrors opponent_runners._run_bootstrap_anchor's
    opt-in path when a config sets require_encoding_source but not encoding):
    an unstamped anchor must fail loudly, not silently sniff shape."""
    ckpt_path = _save_weights_only(tmp_path / "anchor_unstamped.pt")
    with pytest.raises(DeclaredEncodingMismatchError):
        _load_anchor_model(ckpt_path, DEVICE, require_encoding_source=True)


def test_anchor_loader_stamped_loads(tmp_path: Path) -> None:
    """A stamped anchor loads fine with no declaration at all (the common,
    byte-for-byte-compatible default path)."""
    ckpt_path = _save_full_ckpt_with_metadata(
        tmp_path / "anchor_stamped.pt", "v6_live2_ls",
    )
    model, spec, label = _load_anchor_model(ckpt_path, DEVICE)
    assert label == "v6_live2_ls"
    assert spec.name == "v6_live2_ls"


def test_anchor_loader_config_declared_vs_stamp_mismatch_raises(tmp_path: Path) -> None:
    """The anchor's OWN declared-vs-stamp assertion (config `encoding:` key,
    §D-EVALGATE) raises on disagreement — this is F07-preserving: it is
    NEVER a comparison against a candidate's encoding, only the anchor
    checkpoint's declaration against its OWN stamp."""
    ckpt_path = _save_full_ckpt_with_metadata(
        tmp_path / "anchor_declared_mismatch.pt", "v6_live2",
    )
    with pytest.raises(DeclaredEncodingMismatchError):
        _load_anchor_model(ckpt_path, DEVICE, declared_encoding="v6_live2_ls")


def test_base_eval_yaml_bootstrap_anchor_carries_no_encoding_key() -> None:
    """DEEP-MERGE regression pin (§D-EVALGATE fix wave): configs/eval.yaml's
    `opponents.bootstrap_anchor` block must carry NO `encoding:` key.

    The runtime config merge is a recursive key-by-key deep merge
    (hexo_rl/utils/config.py `_deep_merge`), so a variant that overrides
    ONLY `path:` (repointing at a differently-shaped anchor — v6_botmix_s178,
    v6_live2_smoke, longrun_v6_live2_ls_gumbel_m16 all do) would silently
    INHERIT a base-level `encoding:` value, manufacturing a false anchor
    declaration and hard-crashing at anchor load with a declared-vs-stamp
    mismatch. The base file must therefore never declare it; declarations
    belong alongside `path:` in the SAME block."""
    import yaml

    repo_root = Path(__file__).resolve().parent.parent
    with (repo_root / "configs" / "eval.yaml").open() as f:
        cfg = yaml.safe_load(f)
    anchor_cfg = cfg["eval_pipeline"]["opponents"]["bootstrap_anchor"]
    assert "encoding" not in anchor_cfg, (
        "configs/eval.yaml opponents.bootstrap_anchor must NOT declare "
        "`encoding:` — variants overriding only `path` would deep-merge-"
        "inherit it into a false declaration for a differently-shaped anchor"
    )

    # And the merged variant-style view (path-only override) must not
    # manufacture a declaration either.
    from hexo_rl.utils.config import _deep_merge

    merged = dict(anchor_cfg)
    _deep_merge(merged, {"path": "checkpoints/bootstrap_model_v6.pt"})
    assert merged.get("encoding") is None
