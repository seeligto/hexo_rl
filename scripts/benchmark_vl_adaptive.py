#!/usr/bin/env python3
"""
Final A/B Test Conclusion: Constant vs. Adaptive Virtual Loss.
"""

import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from native_core import Board, MCTSTree
from python.model.network import HexTacToeNet
from python.env.game_state import GameState

# Final Benchmark Settings
N_POSITIONS = 50
SIMS_PER_MOVE = 1024
BATCH_SIZE = 16 

def load_positions(n: int = 50) -> List[Board]:
    corpus_dir = Path("data/corpus/generated_bot")
    game_files = list(corpus_dir.glob("*.json"))
    if not game_files:
        raise FileNotFoundError("No bot games found in data/corpus/generated_bot")
    random.seed(42)
    selected_files = random.sample(game_files, min(n, len(game_files)))
    boards = []
    for p in selected_files:
        with open(p, 'r') as f:
            data = json.load(f)
            moves = data['moves']
            target_ply = min(30, len(moves) - 1)
            board = Board()
            for i in range(target_ply):
                m = moves[i]
                board.apply_move(m['x'], m['y'])
            boards.append(board)
    return boards

def load_bootstrap_model(device: torch.device) -> HexTacToeNet:
    model_path = Path("checkpoints/bootstrap_model.pt")
    model = HexTacToeNet(board_size=19, res_blocks=10, filters=128)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def evaluate_batch(model: HexTacToeNet, device: torch.device, boards: List[Board]) -> Tuple[List[List[float]], List[float]]:
    all_tensors = []
    metadata = []
    for board in boards:
        state = GameState.from_board(board)
        tensor, centers = state.to_tensor(board)
        all_tensors.append(torch.from_numpy(tensor.astype(np.float32)).to(device))
        metadata.append((len(centers), centers))
    if not all_tensors: return [], []
    batch_tensor = torch.cat(all_tensors, dim=0)
    log_policies, values = model(batch_tensor)
    policies = torch.exp(log_policies).cpu().numpy()
    values = values.cpu().numpy()
    final_policies = []
    final_values = []
    curr = 0
    for k, centers in metadata:
        final_policies.append(np.mean(policies[curr:curr+k], axis=0).tolist())
        final_values.append(float(np.mean(values[curr:curr+k])))
        curr += k
    return final_policies, final_values

def run_benchmark(variant_name: str, vl_val: float, adaptive: bool, positions: List[Board], model: HexTacToeNet, device: torch.device) -> Dict[str, Any]:
    total_sims = 0
    total_time = 0
    depths = []
    overlaps = []
    v_histories = []
    for board in positions:
        tree = MCTSTree(c_puct=1.5, virtual_loss=vl_val, vl_adaptive=adaptive)
        tree.new_game(board)
        start = time.perf_counter()
        sims_done = 0
        v_h = []
        while sims_done < SIMS_PER_MOVE:
            leaves = tree.select_leaves(BATCH_SIZE)
            if not leaves: break
            policies, values = evaluate_batch(model, device, leaves)
            tree.expand_and_backup(policies, values)
            sims_done += len(leaves)
            v_h.append(float(values[0]))
        elapsed = time.perf_counter() - start
        total_sims += sims_done
        total_time += elapsed
        depths.append(tree.max_depth_observed)
        overlaps.append(tree.selection_overlap_count)
        v_histories.append(v_h)
    avg_nps = total_sims / total_time
    avg_depth = np.mean(depths)
    avg_overlap_pct = (sum(overlaps) / total_sims) * 100.0 if total_sims > 0 else 0
    stabilities = [np.std(h[len(h)//2:]) for h in v_histories if len(h) > 1]
    avg_stability = np.mean(stabilities) if stabilities else 0.0
    return {
        "depth": avg_depth,
        "nps": avg_nps,
        "overlap": avg_overlap_pct,
        "instability": avg_stability,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_bootstrap_model(device)
    positions = load_positions(N_POSITIONS)
    
    print(f"Running final A/B test on {len(positions)} positions...")
    
    # Variant A: Baseline
    res_a = run_benchmark("Static 3.0", 3.0, False, positions, model, device)
    
    # Variant B: Optimum
    res_b = run_benchmark("Adaptive Sqrt 3.0", 3.0, True, positions, model, device)
    
    print("\n| Metric | Variant A (Static 3.0) | Variant B (Adaptive 3.0) | % Change |")
    print("| :--- | :---: | :---: | :---: |")
    
    metrics = [
        ("Max Search Depth", "depth", ".2f"),
        ("Nodes per Second", "nps", ",.0f"),
        ("% Selection Overlap", "overlap", ".2f"),
        ("Avg. Value Stability", "instability", ".4f"),
    ]
    
    for label, key, fmt in metrics:
        val_a = res_a[key]
        val_b = res_b[key]
        diff = ((val_b - val_a) / val_a * 100.0) if val_a != 0 else 0
        print(f"| {label} | {val_a:{fmt}} | {val_b:{fmt}} | {diff:+.1f}% |")

if __name__ == "__main__":
    main()
