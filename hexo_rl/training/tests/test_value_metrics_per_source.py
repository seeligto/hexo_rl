"""§D-VALCEIL Q3 — per-source masked value logging (logging-only).

Resolves the §D-VALPROBE value_accuracy ANOMALY (reported batch 0.66 vs
component-predicted ~0.725): batch ``value_accuracy`` is UNMASKED — draw rows
(z = draw_value) and ply-capped rows (z = ply_cap_value, value_target_valid=0)
get ``target_win = (z > 0) = 0`` and are scored as "loss" targets, deflating
the headline number relative to decided-row accuracy.

New keys (NO existing key renamed/altered — curve continuity):
  value_accuracy_masked            accuracy over supervised AND decided rows
  value_accuracy_corpus            unmasked accuracy, rows [0, n_pretrain)
  value_accuracy_selfplay          unmasked accuracy, rows [n_pretrain, B)
  value_bce_corpus / value_bce_selfplay   per-source BCE over supervised rows
  value_rows_corpus / value_rows_selfplay row counts (slice sizes)
  value_rows_masked                rows entering value_accuracy_masked
  value_rows_corpus_supervised / value_rows_selfplay_supervised
                                   rows entering the per-source BCE

Source semantics (batch order [corpus(+bot) | recent | uniform_self]):
"corpus" = rows [0, n_pretrain). §178 bot-corpus rows are folded into
n_pretrain upstream (step_coordinator passes n_pretrain = n_pre + n_bot) and
are NOT separable at the trainer — the corpus bucket includes them.

Masking semantics: supervised = value_target_valid (all-ones when None;
0 only on ply-capped self-play rows, DRAW-MASK Phase 6); decided =
|z| > 0.999 (decisive games store exactly ±1.0; draw_value and ply_cap_value
both lie strictly inside (-1, 1) by config).

Run: .venv/bin/pytest hexo_rl/training/tests/test_value_metrics_per_source.py -v
"""
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from engine import ReplayBuffer
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.losses import compute_value_loss
from hexo_rl.training.trainer import Trainer, compute_value_metrics_per_source


FAST_CONFIG = {
    "board_size":          19,
    "res_blocks":          2,
    "filters":             32,
    "batch_size":          12,
    "lr":                  2e-3,
    "weight_decay":        1e-4,
    "checkpoint_interval": 1000,
    "log_interval":        1,
    "torch_compile":       False,
}

NEW_KEYS = [
    "value_accuracy_masked",
    "value_accuracy_corpus",
    "value_accuracy_selfplay",
    "value_bce_corpus",
    "value_bce_selfplay",
    "value_rows_corpus",
    "value_rows_selfplay",
    "value_rows_masked",
    "value_rows_corpus_supervised",
    "value_rows_selfplay_supervised",
]


def _per_row_bce(logit_col, outcome):
    target = (outcome.float() + 1.0) / 2.0
    return F.binary_cross_entropy_with_logits(
        logit_col.squeeze(1).float(), target, reduction="none"
    )


# ── Pure-helper unit tests ────────────────────────────────────────────────────

def test_masked_excludes_draw_and_capped_rows():
    """Draws (z=draw_value) and ply-capped rows (mask=0) are excluded from
    value_accuracy_masked; decided rows are scored with the exact batch
    semantics (pred = logit>0, target = z>0)."""
    # rows: [win correct, loss correct, draw z=-0.5, capped z=0.0, win wrong]
    logit = torch.tensor([[2.0], [-2.0], [2.0], [2.0], [-2.0]])
    z = torch.tensor([1.0, -1.0, -0.5, 0.0, 1.0])
    mask = torch.tensor([1, 1, 1, 0, 1], dtype=torch.uint8)  # capped row masked
    m = compute_value_metrics_per_source(logit, z, mask, n_pretrain=0)
    # decided & supervised rows = {0, 1, 4}: correct, correct, wrong → 2/3
    assert m["value_rows_masked"] == 3
    assert m["value_accuracy_masked"] == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_unmasked_per_source_matches_batch_semantics():
    """Per-source accuracies are UNMASKED — draws/capped count as target=loss,
    exactly like the existing batch value_accuracy."""
    logit = torch.tensor([[2.0], [-2.0], [2.0], [-2.0]])
    z = torch.tensor([1.0, -0.5, 0.0, -1.0])  # win, draw, capped, loss
    mask = torch.tensor([1, 1, 0, 1], dtype=torch.uint8)
    m = compute_value_metrics_per_source(logit, z, mask, n_pretrain=2)
    # corpus rows 0,1: row0 pred win/target win correct; row1 pred loss /
    # target (z>0)=0 → "correct" under unmasked semantics → 2/2
    assert m["value_accuracy_corpus"] == pytest.approx(1.0, abs=1e-6)
    assert m["value_rows_corpus"] == 2
    # selfplay rows 2,3: row2 pred win vs target 0 → wrong; row3 correct → 1/2
    assert m["value_accuracy_selfplay"] == pytest.approx(0.5, abs=1e-6)
    assert m["value_rows_selfplay"] == 2


def test_mask_none_equals_all_ones_mask():
    torch.manual_seed(0)
    logit = torch.randn(8, 1)
    z = torch.tensor([1.0, -1.0, -0.5, 0.0, 1.0, 1.0, -1.0, -0.5])
    a = compute_value_metrics_per_source(logit, z, None, n_pretrain=3)
    b = compute_value_metrics_per_source(
        logit, z, torch.ones(8, dtype=torch.uint8), n_pretrain=3
    )
    for k in NEW_KEYS:
        if isinstance(a[k], float) and np.isnan(a[k]):
            assert np.isnan(b[k])
        else:
            assert a[k] == pytest.approx(b[k], abs=1e-7)


def test_empty_slices_yield_nan_and_zero_counts():
    logit = torch.tensor([[1.0], [-1.0]])
    z = torch.tensor([1.0, -1.0])
    m0 = compute_value_metrics_per_source(logit, z, None, n_pretrain=0)
    assert m0["value_rows_corpus"] == 0
    assert np.isnan(m0["value_accuracy_corpus"])
    assert np.isnan(m0["value_bce_corpus"])
    m_all = compute_value_metrics_per_source(logit, z, None, n_pretrain=2)
    assert m_all["value_rows_selfplay"] == 0
    assert np.isnan(m_all["value_accuracy_selfplay"])
    assert np.isnan(m_all["value_bce_selfplay"])
    # all-capped batch → masked accuracy undefined
    m_capped = compute_value_metrics_per_source(
        logit, torch.tensor([0.0, -0.5]),
        torch.zeros(2, dtype=torch.uint8), n_pretrain=1,
    )
    assert m_capped["value_rows_masked"] == 0
    assert np.isnan(m_capped["value_accuracy_masked"])


def test_reconstruction_identity_hand_tensors():
    """count-weighted per-source accuracies recombine to the batch accuracy."""
    torch.manual_seed(1)
    logit = torch.randn(16, 1)
    z = torch.tensor([1.0, -1.0, -0.5, 0.0] * 4)
    mask = torch.tensor([1, 1, 1, 0] * 4, dtype=torch.uint8)
    for n_pre in (0, 5, 16):
        m = compute_value_metrics_per_source(logit, z, mask, n_pretrain=n_pre)
        pred_win = (logit.squeeze(1) > 0).float()
        target_win = (z > 0).float()
        batch_acc = (pred_win == target_win).float().mean().item()
        parts = 0.0
        if m["value_rows_corpus"] > 0:
            parts += m["value_accuracy_corpus"] * m["value_rows_corpus"]
        if m["value_rows_selfplay"] > 0:
            parts += m["value_accuracy_selfplay"] * m["value_rows_selfplay"]
        recombined = parts / (m["value_rows_corpus"] + m["value_rows_selfplay"])
        assert recombined == pytest.approx(batch_acc, abs=1e-6)


def test_bce_per_source_recombines_to_value_loss():
    """Per-source BCE (supervised rows, compute_value_loss semantics) weighted
    by supervised counts recombines to the trainer's value_loss."""
    torch.manual_seed(2)
    logit = torch.randn(12, 1)
    z = torch.tensor([1.0, -1.0, -0.5, 0.0] * 3)
    mask = torch.tensor([1, 1, 1, 0] * 3, dtype=torch.uint8)
    m = compute_value_metrics_per_source(logit, z, mask, n_pretrain=4)
    loss = compute_value_loss(logit, z, value_mask=mask).item()
    n_c, n_s = m["value_rows_corpus_supervised"], m["value_rows_selfplay_supervised"]
    recombined = (m["value_bce_corpus"] * n_c + m["value_bce_selfplay"] * n_s) / (n_c + n_s)
    assert recombined == pytest.approx(loss, abs=1e-6)
    # cross-check one slice against a reference per-row BCE
    ref = _per_row_bce(logit, z)
    corpus_ref = ref[:4][mask[:4].bool()].mean().item()
    assert m["value_bce_corpus"] == pytest.approx(corpus_ref, abs=1e-6)


# ── Integration through the real train step ──────────────────────────────────

def _mixed_batch(batch_n=12, seed=3):
    """Synthetic batch: decided + draw + capped rows in BOTH source slices."""
    rng = np.random.default_rng(seed)
    states = rng.random((batch_n, 8, 19, 19), dtype=np.float32)
    chain = np.zeros((batch_n, 6, 19, 19), dtype=np.float16)
    policies = rng.dirichlet(np.ones(362), size=batch_n).astype(np.float32)
    # per-slice pattern: win, loss, draw(-0.5), capped(0.0) ...
    outcomes = np.array([1.0, -1.0, -0.5, 0.0] * (batch_n // 4), dtype=np.float32)
    vv = np.array([1, 1, 1, 0] * (batch_n // 4), dtype=np.uint8)  # cap rows masked
    return states, chain, policies, outcomes, vv


def test_train_step_from_tensors_emits_keys_counts_and_reconstructs(tmp_path: Path):
    torch.manual_seed(7)
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, FAST_CONFIG, checkpoint_dir=tmp_path,
                      device=torch.device("cpu"))
    states, chain, policies, outcomes, vv = _mixed_batch()
    n_pre = 4  # rows [0,4) = corpus(+bot); rows [4,12) = self-play
    result = trainer.train_step_from_tensors(
        states, policies, outcomes, chain_planes=chain,
        n_pretrain=n_pre, value_target_valid=vv,
    )
    for k in NEW_KEYS:
        assert k in result, f"missing new key {k}"
    assert result["value_rows_corpus"] == n_pre
    assert result["value_rows_selfplay"] == len(outcomes) - n_pre
    # supervised counts: 3 of every 4 rows (capped row masked)
    assert result["value_rows_corpus_supervised"] == 3
    assert result["value_rows_selfplay_supervised"] == 6
    # masked accuracy rows = supervised AND decided = 2 of every 4 rows
    assert result["value_rows_masked"] == 6
    # RECONSTRUCTION (the §D-VALCEIL proof): per-source components weighted by
    # counts recombine to the reported batch value_accuracy exactly (float tol).
    recombined = (
        result["value_accuracy_corpus"] * result["value_rows_corpus"]
        + result["value_accuracy_selfplay"] * result["value_rows_selfplay"]
    ) / (result["value_rows_corpus"] + result["value_rows_selfplay"])
    assert recombined == pytest.approx(result["value_accuracy"], abs=1e-6)
    # BCE components recombine to the reported value_loss (CPU fp32 path).
    n_cs = result["value_rows_corpus_supervised"]
    n_ss = result["value_rows_selfplay_supervised"]
    bce_recombined = (
        result["value_bce_corpus"] * n_cs + result["value_bce_selfplay"] * n_ss
    ) / (n_cs + n_ss)
    assert bce_recombined == pytest.approx(result["value_loss"], abs=1e-5)


def test_existing_keys_unchanged_and_not_renamed(tmp_path: Path):
    """Curve continuity: pre-change key set still present, values untouched
    (value_accuracy stays the UNMASKED batch number)."""
    torch.manual_seed(11)
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, FAST_CONFIG, checkpoint_dir=tmp_path,
                      device=torch.device("cpu"))
    states, chain, policies, outcomes, vv = _mixed_batch(seed=5)
    result = trainer.train_step_from_tensors(
        states, policies, outcomes, chain_planes=chain,
        n_pretrain=0, value_target_valid=vv,
    )
    for k in ("loss", "policy_loss", "value_loss", "value_accuracy", "grad_norm",
              "lr", "full_search_frac", "value_loss_composite"):
        assert k in result
    assert "value_loss_main" not in result  # redundant alias deleted in B5
    # n_pretrain=0 single-buffer path: corpus slice empty, selfplay == batch
    assert result["value_rows_corpus"] == 0
    assert np.isnan(result["value_accuracy_corpus"])
    assert result["value_accuracy_selfplay"] == pytest.approx(
        result["value_accuracy"], abs=1e-6
    )
