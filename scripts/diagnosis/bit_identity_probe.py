"""Cross-commit bit-identity probe (§D-VALPROBE Phase 3 review evidence).

Run from a repo checkout root. Deterministic CPU train_step (identical-rows
buffer, fixed seeds) → prints loss, value keys, and sha256 over post-step
parameter bytes. Identical output across commits == logging-only proof.
"""
import hashlib
import json
import pathlib
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path.cwd()))
from engine import ReplayBuffer  # noqa: E402
from hexo_rl.model.network import HexTacToeNet  # noqa: E402
from hexo_rl.training.trainer import Trainer  # noqa: E402

CFG = {
    "board_size": 19, "res_blocks": 2, "filters": 32, "batch_size": 8,
    "lr": 2e-3, "weight_decay": 1e-4, "checkpoint_interval": 1000,
    "log_interval": 1, "torch_compile": False,
    "uncertainty_weight": 0.1, "aux_opp_reply_weight": 0.15,
}

torch.manual_seed(99)
torch.use_deterministic_algorithms(True)
model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
trainer = Trainer(model, CFG, checkpoint_dir=tempfile.mkdtemp(),
                  device=torch.device("cpu"))

buf = ReplayBuffer(capacity=200)
rng = np.random.default_rng(7)
own = np.ones(361, dtype=np.uint8)
wl = np.zeros(361, dtype=np.uint8)
chain = np.zeros((6, 19, 19), dtype=np.float16)
state = rng.random((8, 19, 19), dtype=np.float32).astype(np.float16)
policy = rng.dirichlet(np.ones(362)).astype(np.float32)
for _ in range(32):
    buf.push(state, chain, policy, 1.0, own, wl)

result = trainer.train_step(buf, augment=False)

h = hashlib.sha256()
for p in trainer.model.parameters():
    h.update(p.detach().cpu().numpy().tobytes())

print(json.dumps({
    "loss": result["loss"],
    "policy_loss": result["policy_loss"],
    "value_loss": result["value_loss"],
    "uncertainty_loss": result.get("uncertainty_loss"),
    "opp_reply_loss": result.get("opp_reply_loss"),
    "value_loss_composite": result.get("value_loss_composite", "ABSENT(pre-change)"),
    "param_sha256": h.hexdigest(),
}, indent=2))
