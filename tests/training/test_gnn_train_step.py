"""GNN-integration WP-5b commit B (P1) — trainer graph training-step tests.

OQ-7 part-3 train-leg (`docs/designs/run4_gnn_design.md` §8 OQ-7): a graph
position round-trips PyO3 -> HEXG push -> rebuild-at-sample -> train-step
(finite losses, step advances) -> checkpoint -> reload, the full ragged
assertion set green.

Fresh-init decoupling (delta doc §8): finite losses + ckpt round-trip are
init-agnostic — `build_net` always makes a fresh `GnnNet` — so this smoke
runs fresh-init; a SEPARATE `test_gnn_bc_warmstart.py` validates the
BC-prefit loader landed-verify (OQ-5).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple

import pytest

torch = pytest.importorskip("torch")

from engine import HexgBuffer  # noqa: E402
from hexo_rl.model.gnn_net import GnnNet  # noqa: E402
from hexo_rl.training import losses as _losses_mod  # noqa: E402
from hexo_rl.training import trainer as _trainer_mod  # noqa: E402
from hexo_rl.training.binned_value import binned_value_loss  # noqa: E402
from hexo_rl.training.trainer import Trainer  # noqa: E402

pytestmark = pytest.mark.integration

ENC = "gnn_axis_v1"
WPA = Path("reports/probes/gnn_integration/wpa_positions.json")
_NEIGHBORS: List[Tuple[int, int]] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


def _empty_neighbor(stones):
    occ = {(q, r) for q, r, _ in stones}
    for q0, r0, _ in stones:
        for dq, dr in _NEIGHBORS:
            c = (q0 + dq, r0 + dr)
            if c not in occ:
                return c
    raise AssertionError("no empty neighbor — degenerate position")


def _filled_buffer(n: int, *, capacity: int = 128) -> HexgBuffer:
    if not WPA.exists():
        pytest.skip(f"{WPA} not present (WP-A frozen position set)")
    data = json.loads(WPA.read_text())
    positions = data["positions"][:n]
    buf = HexgBuffer(capacity, ENC)
    for i, p in enumerate(positions):
        stones = [(int(q), int(r), int(pl)) for q, r, pl in p["stones"]]
        nq, nr = _empty_neighbor(stones)
        buf.push_graph_position(
            stones, [(int(nq), int(nr), 1.0)], int(p["current_player"]),
            int(p["moves_remaining"]), int(p.get("ply", 0)) & 0xFFFF, True,
            1.0 if i % 2 == 0 else -1.0, True, 30, i,
        )
    return buf


def _fast_config(**overrides) -> dict:
    cfg = {
        "encoding": ENC,
        "fp16": False,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "checkpoint_interval": 3,
        "grad_clip": 1.0,
    }
    cfg.update(overrides)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────


def test_dist65_verbatim_import_identity():
    """§3 proof obligation: the graph branch calls the SAME `binned_value_loss`
    symbol the CNN's `_train_on_batch` calls — not a graph-local copy."""
    assert _trainer_mod.binned_value_loss is binned_value_loss
    assert _trainer_mod.ragged_policy_ce is _losses_mod.ragged_policy_ce


@pytest.mark.parametrize(
    "fp16",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="fp16 autocast is disabled off CUDA (Trainer.__init__ "
                       "downgrades it silently) — the True leg needs a real "
                       "CUDA device to exercise the run4 regime.",
            ),
        ),
    ],
)
def test_three_steps_fresh_init_finite_losses_and_step_advances(fp16, tmp_path):
    """The OQ-7 part-3 train-leg: 3 steps on a fresh-init GnnNet, all losses
    + grads finite, `trainer.step` advances by exactly 3, the sampled wire
    is native-built (F7).

    Parametrized over fp16 True/False (WP5b commit-B red-team BREAK-1): the
    prior fp16=False-only coverage never exercised run4's actual launch
    regime (`fp16: True`, inherited from `configs/training.yaml`), which
    crashed `ragged_policy_ce`'s `segment_softmax` scatter on step 0
    (dtype mismatch: `torch.exp` autopromotes to fp32 under CUDA autocast,
    `denom` stayed fp16) — see `hexo_rl/training/losses.py::ragged_policy_ce`
    fp32-cast-at-entry fix. The True leg is the regression guard for that fix
    and requires CUDA (fp16 is a silent no-op on CPU).

    `grad_norm` is checked non-negative/non-NaN rather than strictly finite
    under fp16: a transient `inf` from `GradScaler` detecting an overflow and
    backing off the scale is expected, documented fp16 behaviour (repo
    convention: `tests/test_trainer.py::test_train_step_returns_grad_norm`),
    unrelated to the scatter-dtype-crash/log-underflow-NaN class BREAK-1
    actually names — `loss`/`policy_loss`/`value_loss` (the values that class
    corrupts) stay strictly-finite-checked for both legs."""
    buf = _filled_buffer(48)
    device = torch.device("cuda") if fp16 else torch.device("cpu")
    trainer = Trainer(GnnNet(), _fast_config(fp16=fp16), checkpoint_dir=tmp_path, device=device)

    assert trainer.step == 0
    for expected_step in (1, 2, 3):
        loss_info = trainer.train_step(buf, augment=True)
        assert trainer.step == expected_step
        for key in ("loss", "policy_loss", "value_loss"):
            assert math.isfinite(loss_info[key]), f"step {expected_step}: {key}={loss_info[key]!r} not finite"
        assert loss_info["grad_norm"] >= 0.0
        assert not math.isnan(loss_info["grad_norm"]), (
            f"step {expected_step}: grad_norm is NaN"
        )
        if not fp16:
            for p in trainer.model.parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), "all grads must be finite"

    # checkpoint_interval=3 -> exactly one checkpoint written, at step 3.
    ckpts = sorted(tmp_path.glob("checkpoint_*.pt"))
    assert [p.name for p in ckpts] == ["checkpoint_00000003.pt"]


def test_ragged_policy_ce_casts_fp16_logits_without_scatter_dtype_crash():
    """BREAK-1 unit-level guard, GPU-independent: fp16 `policy_logits` (the
    dtype `GnnNet.forward_batch` emits under CUDA fp16 autocast) must not
    scatter-dtype-crash inside `segment_softmax`, nor silently NaN through
    the `log(clamp(min=1e-12))` fp16-underflow path. Passing fp16 tensors
    directly (no autocast context needed) exercises the entry-point cast
    fix independent of CUDA autocast's op-promotion quirks, so this runs
    everywhere `pytest -m "not slow"` runs."""
    legal_offsets = torch.tensor([0, 3, 5], dtype=torch.int64)
    policy_logits = torch.tensor([2.0, -1.0, 0.5, 3.0, -3.0], dtype=torch.float16)
    policy_target = torch.tensor([0.2, 0.3, 0.5, 0.9, 0.1], dtype=torch.float32)

    loss = _losses_mod.ragged_policy_ce(policy_logits, policy_target, legal_offsets)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the scatter-dtype crash is a CUDA-autocast-specific op-promotion "
           "quirk (torch.exp autopromotes to fp32 under CUDA autocast but not "
           "CPU autocast) — this test faithfully reproduces the production "
           "mechanism and needs a real CUDA device.",
)
def test_ragged_policy_ce_under_real_cuda_autocast_no_scatter_crash():
    """Faithful repro of the BREAK-1 production mechanism (not just a
    dtype-cast unit check): a `nn.Linear` executed inside
    `torch.autocast(cuda, fp16)` emits fp16 logits exactly as
    `GnnNet.forward_batch` does inside `_train_on_graph_batch`'s autocast
    block; pre-fix this raised
    `RuntimeError: scatter(): Expected self.dtype to be equal to src.dtype`
    on step 0 of every run4 launch (`fp16=True`, the base-config default).
    Post-fix this must not raise and must return a finite fp32 loss."""
    device = torch.device("cuda")
    legal_offsets = torch.tensor([0, 3, 5], dtype=torch.int64, device=device)
    policy_target = torch.tensor(
        [0.2, 0.3, 0.5, 0.9, 0.1], dtype=torch.float32, device=device
    )
    lin = torch.nn.Linear(4, 1).to(device)
    x = torch.randn(5, 4, device=device)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        policy_logits = lin(x).squeeze(-1)
        assert policy_logits.dtype == torch.float16, "fixture must reproduce fp16 logits"
        loss = _losses_mod.ragged_policy_ce(policy_logits, policy_target, legal_offsets)

    assert torch.isfinite(loss).all()


def test_sample_wire_is_native_builder_every_step():
    """F7 — every training-step sample must carry `builder_impl==1` (the
    native `hexo_graph::build_axis_graph` rebuild, never the Python-oracle
    fallback trap)."""
    buf = _filled_buffer(32)
    wire, _tg = buf.sample_graph_batch(16, augment=True, recent_frac=0.0)
    assert int(wire.builder_impl) == 1
    assert int(wire.contract_version) == 1


def test_recent_frac_threads_from_recency_weight_config(monkeypatch):
    """§4/§12 — `recency_weight` in config must thread through to
    `sample_graph_batch(recent_frac=...)`, not silently no-op. Monkeypatch
    the CLASS method (PyO3 instances don't support per-instance attribute
    patching) to capture the actual kwarg the trainer passes."""
    buf = _filled_buffer(32)
    captured = {}
    real_sample = HexgBuffer.sample_graph_batch

    def _spy(self, batch_size, augment=False, recent_frac=0.0):
        captured["recent_frac"] = recent_frac
        captured["batch_size"] = batch_size
        return real_sample(self, batch_size, augment=augment, recent_frac=recent_frac)

    monkeypatch.setattr(HexgBuffer, "sample_graph_batch", _spy)
    trainer = Trainer(GnnNet(), _fast_config(recency_weight=0.75),
                       checkpoint_dir="/tmp", device=torch.device("cpu"))
    trainer.train_step(buf, augment=True)
    assert captured["recent_frac"] == pytest.approx(0.75), (
        "recency_weight config must thread through to sample_graph_batch(recent_frac=...) "
        "(a wiring bug here is a SILENT no-op, delta doc §11)"
    )


def test_recent_frac_defaults_to_zero_when_recency_weight_absent(monkeypatch):
    buf = _filled_buffer(32)
    captured = {}
    real_sample = HexgBuffer.sample_graph_batch

    def _spy(self, batch_size, augment=False, recent_frac=0.0):
        captured["recent_frac"] = recent_frac
        return real_sample(self, batch_size, augment=augment, recent_frac=recent_frac)

    monkeypatch.setattr(HexgBuffer, "sample_graph_batch", _spy)
    trainer = Trainer(GnnNet(), _fast_config(), checkpoint_dir="/tmp", device=torch.device("cpu"))
    trainer.train_step(buf, augment=True)
    assert captured["recent_frac"] == 0.0


def test_positive_aux_weight_on_graph_config_raises_loud():
    """Standing §6.3 guard: GnnNet has no aux heads — a positive aux weight
    on a graph config must raise, not silently no-op."""
    buf = _filled_buffer(16)
    trainer = Trainer(GnnNet(), _fast_config(ownership_weight=0.5),
                       checkpoint_dir="/tmp", device=torch.device("cpu"))
    with pytest.raises(ValueError, match="ownership_weight"):
        trainer.train_step(buf, augment=True)


def test_checkpoint_round_trip_encoding_and_schema_stamped(tmp_path):
    """§7 — save -> reload round-trip: `encoding_name=='gnn_axis_v1'` +
    `schema_version==1` stamped (WP-4/existing stamping, no new code in
    commit B — this is the trainer-side ASSERT commit B designs)."""
    from hexo_rl.training.checkpoints import CHECKPOINT_METADATA_SCHEMA_VERSION

    buf = _filled_buffer(32)
    trainer = Trainer(GnnNet(), _fast_config(checkpoint_interval=1),
                       checkpoint_dir=tmp_path, device=torch.device("cpu"))
    trainer.train_step(buf, augment=False)
    ckpt_path = tmp_path / "checkpoint_00000001.pt"
    assert ckpt_path.exists()

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert raw["metadata"]["encoding_name"] == ENC
    assert raw["metadata"]["schema_version"] == CHECKPOINT_METADATA_SCHEMA_VERSION

    reloaded = Trainer.load_checkpoint(str(ckpt_path), checkpoint_dir=tmp_path, device=torch.device("cpu"))
    assert reloaded.step == trainer.step
    base_sd = trainer.model.state_dict()
    reload_sd = reloaded.model.state_dict()
    assert base_sd.keys() == reload_sd.keys()
    for k in base_sd:
        assert torch.allclose(base_sd[k], reload_sd[k]), f"{k} did not round-trip"
