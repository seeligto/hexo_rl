"""§170 P3 — pretrain wiring smoke for the gpool-bias side-branch.

End-to-end glue verification:
  * Build small synthetic v6w25 corpus (50 positions) w/ global_crops column.
  * Wrap in AugmentedBootstrapDataset + make_augmented_collate(with_global_crop=True).
  * Construct HexTacToeNet(gpool_bias_active=True) — small filters/depth for CPU.
  * Run ONE forward+backward step manually.
  * Verify (a) global_crop kwarg flows through, (b) every gpool_bias_branch.*
    parameter receives a non-None finite gradient, (c) gate_value()==0.0 at
    construction (byte-exact-A1 invariant).

Stays on CPU, no DataLoader workers — keeps `make test` fast.
"""
from __future__ import annotations

import numpy as np
import torch

from hexo_rl.bootstrap.dataset_v6w25 import (
    BOARD_SIZE_V6W25,
    N_ACTIONS_V6W25,
    N_PLANES_V6W25,
)
from hexo_rl.bootstrap.pretrain import (
    AugmentedBootstrapDataset,
    make_augmented_collate,
)
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.utils.global_crop import CANVAS_SIZE, N_GLOBAL_PLANES


def _synthetic_corpus(n: int = 50, seed: int = 0):
    """Tiny synthetic v6w25 corpus + global_crops column (no real game logic)."""
    rng = np.random.default_rng(seed)
    states = rng.standard_normal(
        (n, N_PLANES_V6W25, BOARD_SIZE_V6W25, BOARD_SIZE_V6W25),
    ).astype(np.float16)
    # Random one-hot policy targets (cells only; pass slot stays zero).
    policies = np.zeros((n, N_ACTIONS_V6W25), dtype=np.float32)
    cells = BOARD_SIZE_V6W25 * BOARD_SIZE_V6W25
    pol_idx = rng.integers(0, cells, size=n)
    policies[np.arange(n), pol_idx] = 1.0
    outcomes = rng.choice([-1.0, 0.0, 1.0], size=n).astype(np.float32)
    global_crops = rng.standard_normal(
        (n, N_GLOBAL_PLANES, CANVAS_SIZE, CANVAS_SIZE),
    ).astype(np.float16)
    return states, policies, outcomes, global_crops


def test_gpool_bias_pretrain_wiring_one_step():
    """Forward + backward smoke. Confirms global_crop threads + grads reach
    every gpool_bias_branch parameter."""
    states, policies, outcomes, global_crops = _synthetic_corpus()

    dataset = AugmentedBootstrapDataset(
        states, policies, outcomes, global_crops=global_crops,
    )
    assert len(dataset) == len(outcomes)
    # 4-tuple yield (states, policies, outcomes, global_crops).
    sample = dataset[0]
    assert len(sample) == 4

    collate = make_augmented_collate(
        augment=True,
        board_size=BOARD_SIZE_V6W25,
        encoding="v6w25",
        with_global_crop=True,
    )
    batch_size = 8
    batch = collate([dataset[i] for i in range(batch_size)])
    # 5-element batch w/ global_crops appended.
    assert len(batch) == 5
    states_t, chain_t, policies_t, outcomes_t, global_crops_t = batch
    assert states_t.shape == (
        batch_size, N_PLANES_V6W25, BOARD_SIZE_V6W25, BOARD_SIZE_V6W25,
    )
    assert global_crops_t.shape == (
        batch_size, N_GLOBAL_PLANES, CANVAS_SIZE, CANVAS_SIZE,
    )

    # Tiny model: 32 filters × 2 res blocks. CPU-fast; full A1 contract intact.
    torch.manual_seed(0)
    model = HexTacToeNet(
        board_size=BOARD_SIZE_V6W25,
        in_channels=N_PLANES_V6W25,
        filters=32,
        res_blocks=2,
        encoding="v6w25",
        pool_type="min_max",
        gpool_bias_active=True,
    ).train()

    # Gate=0 invariant at construction (byte-exact A1).
    assert model.gpool_bias_gate_value() == 0.0

    # Forward — pass aux=True so opp_reply head is exercised.
    log_policy, _value, v_logit, opp_reply = model(
        states_t.float(),
        aux=True,
        global_crop=global_crops_t.float(),
    )
    assert log_policy.shape == (batch_size, N_ACTIONS_V6W25)
    assert v_logit.shape == (batch_size, 1)
    assert opp_reply.shape == (batch_size, N_ACTIONS_V6W25)

    # Backward on a simple combined loss — exercises bias branch grads.
    nll = -(log_policy * policies_t).sum(dim=1).mean()
    val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        v_logit.squeeze(1), (outcomes_t > 0).float(),
    )
    loss = nll + val_loss
    loss.backward()
    assert torch.isfinite(loss)

    # Every gpool_bias_branch param must receive a finite, non-None grad.
    bias_params = list(model.gpool_bias_branch.named_parameters())
    assert len(bias_params) > 0, "gpool_bias_branch has zero parameters"
    for name, p in bias_params:
        assert p.grad is not None, f"gpool_bias_branch.{name} grad is None"
        assert torch.isfinite(p.grad).all(), (
            f"gpool_bias_branch.{name} grad has non-finite entries"
        )
