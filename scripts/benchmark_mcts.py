
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import time
import numpy as np
from native_core import Board
from python.model.network import HexTacToeNet
from python.selfplay.worker import SelfPlayWorker

def benchmark_mcts():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on: {device}")
    
    # Use a standard sized model
    model = HexTacToeNet(board_size=19, res_blocks=10, filters=128).to(device)
    model.eval()
    
    config = {
        'mcts': {
            'n_simulations': 400,
            'c_puct': 1.5,
            'temperature_threshold_ply': 30,
            'dirichlet_alpha': 0.3,
            'epsilon': 0.25
        }
    }
    worker = SelfPlayWorker(model, config, device)
    board = Board()
    
    # Warmup
    _ = worker._run_mcts_with_sims(board, n_sims=40, batch_size=1)
    
    n_sims = 400
    
    # 1. Sequential (Batch 1)
    t0 = time.time()
    _ = worker._run_mcts_with_sims(board, n_sims=n_sims, batch_size=1)
    seq_time = time.time() - t0
    print(f"Sequential (batch=1): {seq_time:.4f}s ({n_sims/seq_time:.1f} sims/s)")
    
    # 2. Batched (Batch 8)
    t0 = time.time()
    _ = worker._run_mcts_with_sims(board, n_sims=n_sims, batch_size=8)
    batch_time = time.time() - t0
    print(f"Batched    (batch=8): {batch_time:.4f}s ({n_sims/batch_time:.1f} sims/s)")
    
    # 3. Batched (Batch 16)
    t0 = time.time()
    _ = worker._run_mcts_with_sims(board, n_sims=n_sims, batch_size=16)
    batch16_time = time.time() - t0
    print(f"Batched   (batch=16): {batch16_time:.4f}s ({n_sims/batch16_time:.1f} sims/s)")

    print(f"\nSpeedup (8 vs 1): {seq_time / batch_time:.2f}x")
    print(f"Speedup (16 vs 1): {seq_time / batch16_time:.2f}x")

if __name__ == "__main__":
    benchmark_mcts()
