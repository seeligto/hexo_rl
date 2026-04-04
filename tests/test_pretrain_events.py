import numpy as np
import torch
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from hexo_rl.bootstrap.pretrain import BootstrapTrainer
from hexo_rl.model.network import HexTacToeNet

def test_pretrain_emits_training_step_event():
    model = HexTacToeNet(board_size=19, in_channels=18, filters=16, res_blocks=2, se_reduction_ratio=4)
    config = {
        "lr": 0.01,
        "pretrain_total_steps": 100,
        "batch_size": 2,
    }
    trainer = BootstrapTrainer(model, config, torch.device("cpu"), Path("tmp_checkpoints"))
    trainer.step = -100

    # dummy loader
    states = torch.zeros(2, 18, 19, 19, dtype=torch.float32)
    policies = torch.zeros(2, 19*19+1, dtype=torch.float32)
    policies[:, 0] = 1.0 # dummy valid policy
    outcomes = torch.zeros(2, dtype=torch.float32)
    loader = [(states, policies, outcomes)]

    with patch("hexo_rl.bootstrap.pretrain.emit_event") as mock_emit:
        trainer.train_epoch(loader, log_interval=1)
        
        assert mock_emit.called
        event = mock_emit.call_args[0][0]
        assert event["event"] == "training_step"
        assert event["step"] == -99  # incremented before emit
        assert "loss_total" in event
        assert "loss_policy" in event
        assert "loss_value" in event
        assert "loss_aux" in event
        assert "policy_entropy" in event
        assert "value_accuracy" in event
        assert "lr" in event
        assert "grad_norm" in event
        assert event["corpus_mix"] == {"pretrain": 1.0, "self_play": 0.0}
        assert event["phase"] == "pretrain"
