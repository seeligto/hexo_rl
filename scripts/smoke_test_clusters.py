import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from python.model.network import HexTacToeNet
from python.env.game_state import GameState
from native_core import Board
from python.selfplay.worker import SelfPlayWorker

def test_multi_window():
    print("--- Multi-Window Colony Threat Detection Smoke Test ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HexTacToeNet(board_size=19, res_blocks=10, filters=128)
    
    ckpt_path = "checkpoints/bootstrap_model.pt"
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded trained checkpoint from {ckpt_path}.")
    except FileNotFoundError:
        print("WARNING: No checkpoint found. Running with random weights. Assertions on value/policy may fail.")
        
    model.to(device)
    model.eval()

    config = {
        'mcts': {
            'n_simulations': 50,
            'c_puct': 1.5,
        }
    }
    worker = SelfPlayWorker(model, config, device)

    board = Board()
    
    # Construct a valid game sequence to create the scenario:
    # P1 builds a colony at q=25.
    # P2 builds a lethal 5-in-a-row threat at the center (q=0..4, r=0).
    
    # Turn 0: P1 (1 move)
    board.apply_move(25, 0)
    
    # Turn 1: P2 (2 moves)
    board.apply_move(0, 0)
    board.apply_move(1, 0)
    
    # Turn 2: P1 (2 moves)
    board.apply_move(25, 1)
    board.apply_move(25, 2)
    
    # Turn 3: P2 (2 moves)
    board.apply_move(2, 0)
    board.apply_move(3, 0)
    
    # Turn 4: P1 (2 moves)
    board.apply_move(25, 3)
    board.apply_move(25, 4)
    
    # Turn 5: P2 (2 moves) - completes the 5-in-a-row threat and plays a filler
    board.apply_move(4, 0)
    board.apply_move(-5, 5) # filler
    
    # Now it is P1's turn. P2 threatens to win on the next turn by playing (5,0) or (-1,0).
    # P1 MUST respond to the threat at the center, despite having a 5-stone colony at q=25.
    
    # 1. Verify Rust core generates 2 separate tensors
    views, centers = board.get_cluster_views()
    print(f"\nNumber of clusters detected: {len(centers)}")
    print(f"Cluster centers: {centers}")
    assert len(centers) >= 2, f"Expected at least 2 clusters, got {len(centers)}"
    
    # 2. Verify Value Head output and Policy
    policy, value = worker._infer(board)
    print(f"\nAggregated Value (P1 perspective): {value:.4f}")
    if value > -0.5:
        print("Note: Value is not strongly negative. (Expected if using random weights)")
    else:
        print("Success: Value head correctly detects the lethal threat (-1.0).")
    
    # 3. Verify the top Policy moves target the (0,0) threat
    policy_np = np.array(policy)
    
    # Get top 5 legal moves
    legal_moves = board.legal_moves()
    move_probs = []
    for q, r in legal_moves:
        idx = board.to_flat(q, r)
        if idx < len(policy_np):
            move_probs.append(((q, r), policy_np[idx]))
            
    move_probs.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 policy moves:")
    for (q, r), prob in move_probs[:5]:
        print(f"  Move ({q:2d}, {r:2d}): {prob:.4f}")
        
    top_q, top_r = move_probs[0][0]
    is_near_threat = (abs(top_q) <= 6 and abs(top_r) <= 6)
    
    if is_near_threat:
        print("\nSuccess: Top policy move targets the threat colony!")
    else:
        print("\nWarning: Top policy move does NOT target the threat colony. (Expected if using random weights)")

    print("\nSmoke test complete.")

if __name__ == "__main__":
    test_multi_window()
