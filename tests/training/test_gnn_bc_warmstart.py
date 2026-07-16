"""GNN-integration WP-5b commit B (P6) — BC-prefit warm-start seam tests.

Two layers:
  1. `load_representation_policy_from_bc` (fn EXISTS, `gnn_net.py:250`) —
     landed-verify FIRES on the real banked BC-prefit artifact (OQ-5), and an
     ADV sd with a dropped representation tensor RAISES (F1 guard).
  2. `maybe_warmstart_gnn_from_bc` (NEW, `gnn_warmstart.py`) — the fresh-init
     launch seam: default-OFF no-op, graph-only misuse-detection, missing-
     checkpoint diagnosability, and the value-head-present diagnostic.

Decoupled from `test_gnn_train_step.py`'s fresh-init train-step smoke (delta
doc §8 T1: the train-step smoke does NOT depend on warm-start).
"""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hexo_rl.encoding import lookup as registry_lookup  # noqa: E402
from hexo_rl.model.gnn_net import (  # noqa: E402
    BC_TRANSFER_PREFIXES,
    GnnNet,
    load_representation_policy_from_bc,
)
from hexo_rl.training.checkpoints import extract_model_state  # noqa: E402
from hexo_rl.training.gnn_warmstart import maybe_warmstart_gnn_from_bc  # noqa: E402

BC_CKPT = Path("checkpoints/probes/gnn_bc/gnn_bc_040000.pt")
GRAPH_SPEC = registry_lookup("gnn_axis_v1")
GRID_SPEC = registry_lookup("v6")


def _skip_if_no_bc_checkpoint():
    if not BC_CKPT.exists():
        pytest.skip(f"{BC_CKPT} not present (banked BC-prefit probe checkpoint)")


# ── Layer 1: load_representation_policy_from_bc (existing fn, OQ-5 proof) ────


def test_landed_verify_fires_on_the_real_bc_checkpoint():
    _skip_if_no_bc_checkpoint()
    raw = torch.load(BC_CKPT, map_location="cpu", weights_only=True)
    bc_sd = extract_model_state(raw)

    net = GnnNet()
    result = load_representation_policy_from_bc(net, bc_sd)

    assert result["verified_tensors"] > 0
    assert result["verified_tensors"] == len(result["loaded_keys"])
    # Every transferred key must actually be present + allclose on the net —
    # re-derive independently of the fn's own internal verify (belt-and-suspenders).
    reloaded = net.state_dict()
    for k in result["loaded_keys"]:
        assert torch.allclose(reloaded[k], bc_sd[k].to(reloaded[k].dtype))
    # value_head untouched (E1 REVIVE) — no key under the transferred set.
    assert not any(k.startswith("value_head.") for k in result["loaded_keys"])


def test_landed_verify_raises_on_a_dropped_rep_tensor_adv():
    """F1 guard: a source sd missing ONE representation tensor must raise —
    a silent strict=False drop is the exact failure class that self-played
    the wrong representation for 272k+ steps undetected
    (d-forensic-f1-lineage-single-window-cascade)."""
    _skip_if_no_bc_checkpoint()
    raw = torch.load(BC_CKPT, map_location="cpu", weights_only=True)
    bc_sd = dict(extract_model_state(raw))
    dropped_key = next(k for k in bc_sd if k.startswith("representation."))
    del bc_sd[dropped_key]

    net = GnnNet()
    with pytest.raises(RuntimeError, match="key mismatch"):
        load_representation_policy_from_bc(net, bc_sd)


def test_landed_verify_raises_on_a_corrupted_transferred_tensor_adv():
    """A source tensor that loads (key present, shape-compatible) but whose
    VALUE was tampered with post-load must still be caught — proves the
    landed-verify actually re-reads the net's own state, not the source dict."""
    net = GnnNet()
    src = {k: v.clone() for k, v in net.state_dict().items() if k.startswith(BC_TRANSFER_PREFIXES)}
    # Sabotage the net AFTER a legit strict=False load would land it, by
    # monkeypatching load_state_dict to silently skip one tensor — simulate
    # via a hand-rolled net whose state_dict lies about one tensor's value.
    key = next(iter(src))
    real_load = net.load_state_dict

    def _lying_load(state_dict, strict=True):  # noqa: ANN001
        state_dict = dict(state_dict)
        state_dict[key] = state_dict[key] + 999.0  # never actually lands this value
        result = real_load(state_dict, strict=strict)
        # Undo the sabotage on the ACTUAL net so state_dict() reads the
        # pre-sabotage value -> landed-verify's re-read must catch the gap.
        with torch.no_grad():
            base_key = key
            mod = net
            for part in base_key.split(".")[:-1]:
                mod = getattr(mod, part)
            getattr(mod, base_key.split(".")[-1]).copy_(state_dict[key] - 999.0)
        return result

    net.load_state_dict = _lying_load  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="landed-verify FAILED"):
        load_representation_policy_from_bc(net, src)


# ── Layer 2: maybe_warmstart_gnn_from_bc (the fresh-init launch seam) ────────


def test_disabled_is_byte_identical_noop():
    net = GnnNet()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    fired = maybe_warmstart_gnn_from_bc(net, {}, spec=GRAPH_SPEC)
    assert fired is False
    after = net.state_dict()
    assert all(torch.equal(v, after[k]) for k, v in before.items())


def test_disabled_flag_explicit_false_is_noop():
    net = GnnNet()
    fired = maybe_warmstart_gnn_from_bc(
        net, {"gnn_warm_start": {"enabled": False}}, spec=GRAPH_SPEC,
    )
    assert fired is False


def test_enabled_on_grid_spec_raises_config_misuse():
    net = GnnNet()
    with pytest.raises(ValueError, match="graph-only"):
        maybe_warmstart_gnn_from_bc(
            net, {"gnn_warm_start": {"enabled": True, "checkpoint": "x.pt"}},
            spec=GRID_SPEC,
        )


def test_enabled_without_checkpoint_raises():
    net = GnnNet()
    with pytest.raises(ValueError, match="checkpoint is unset"):
        maybe_warmstart_gnn_from_bc(
            net, {"gnn_warm_start": {"enabled": True}}, spec=GRAPH_SPEC,
        )


def test_enabled_with_missing_file_raises_file_not_found(tmp_path):
    net = GnnNet()
    missing = tmp_path / "does_not_exist.pt"
    with pytest.raises(FileNotFoundError):
        maybe_warmstart_gnn_from_bc(
            net, {"gnn_warm_start": {"enabled": True, "checkpoint": str(missing)}},
            spec=GRAPH_SPEC,
        )


def test_enabled_on_real_bc_checkpoint_transfers_rep_and_policy_only():
    _skip_if_no_bc_checkpoint()
    net = GnnNet()
    rep_policy_before = {
        k: v.clone() for k, v in net.state_dict().items() if k.startswith(BC_TRANSFER_PREFIXES)
    }
    value_before = {k: v.clone() for k, v in net.state_dict().items() if k.startswith("value_head.")}

    fired = maybe_warmstart_gnn_from_bc(
        net, {"gnn_warm_start": {"enabled": True, "checkpoint": str(BC_CKPT)}},
        spec=GRAPH_SPEC,
    )
    assert fired is True

    after = net.state_dict()
    n_changed = sum(1 for k, v in rep_policy_before.items() if not torch.allclose(v, after[k]))
    assert n_changed > 0, "warm-start must actually change representation/policy tensors"
    assert all(torch.allclose(v, after[k]) for k, v in value_before.items()), (
        "value head must stay fresh (E1 REVIVE) — warm-start transfers ONLY rep+policy"
    )


def test_source_with_dist65_bins_logs_diagnostic_but_still_transfers(monkeypatch, tmp_path):
    """A source that ALSO carries `value_head.fc2_bins.weight` (looks like a
    FULL checkpoint, not BC-prefit-only) is a loud diagnostic, not a hard
    error — the transfer still fires (rep+policy only, value head ignored
    either way)."""
    donor = GnnNet()  # has value_head.fc2_bins.weight (fresh-init, still present as a key)
    ckpt_path = tmp_path / "looks_full.pt"
    torch.save({"model_state_dict": donor.state_dict()}, ckpt_path)

    logged = {}

    class _CaptureLogger:
        def warning(self, event, **kw):
            logged[event] = kw

        def info(self, event, **kw):
            pass

    net = GnnNet()
    fired = maybe_warmstart_gnn_from_bc(
        net, {"gnn_warm_start": {"enabled": True, "checkpoint": str(ckpt_path)}},
        spec=GRAPH_SPEC, log=_CaptureLogger(),
    )
    assert fired is True
    assert "gnn_warmstart_source_has_value_head" in logged
