"""CONFRES Phase-3 batch 6b — structured lr-knob rendering in the resolved_config event.

The ``lr`` knob's value is an ``LrProvenance`` dataclass. In the JSONL event it was serialised as a
repr string (opaque forensic artifact). Render it as a nested dict
``{"declared","baked","effective","override_ignored"}`` so the provenance is structured. The
in-memory ``lr_provenance()`` accessor still returns the dataclass.
"""
from hexo_rl.config.resolve.lr import LrProvenance
from hexo_rl.config.resolve.run_config import ResolvedValue, ResolvedRunConfig


def _lr_rv(declared=0.002, baked=0.0018569, effective=0.0018, override_ignored=True) -> ResolvedValue:
    prov = LrProvenance(
        declared=declared, baked=baked, effective=effective, override_ignored=override_ignored
    )
    return ResolvedValue(prov, "checkpoint_state", "checkpoint-wins-loud", {"variant": declared})


def test_lr_knob_renders_as_nested_dict():
    rv = _lr_rv()
    d = rv.as_event_dict()
    # value is a nested dict, NOT a repr string.
    assert isinstance(d["value"], dict)
    assert d["value"] == {
        "declared": 0.002,
        "baked": 0.0018569,
        "effective": 0.0018,
        "override_ignored": True,
    }


def test_lr_knob_render_survives_to_event_payload():
    rv = _lr_rv()
    rrc = ResolvedRunConfig(_values={"lr": rv})
    payload = rrc.to_event_payload()
    lr_val = payload["knobs"]["lr"]["value"]
    assert isinstance(lr_val, dict)
    assert lr_val["override_ignored"] is True
    assert lr_val["effective"] == 0.0018


def test_lr_provenance_accessor_still_returns_dataclass():
    rv = _lr_rv()
    rrc = ResolvedRunConfig(_values={"lr": rv})
    prov = rrc.lr_provenance()
    assert isinstance(prov, LrProvenance)
    assert prov.declared == 0.002


def test_non_lr_values_render_unchanged():
    rv = ResolvedValue(0.5, "variant", "variant-wins", {"variant": 0.5})
    d = rv.as_event_dict()
    assert d["value"] == 0.5
