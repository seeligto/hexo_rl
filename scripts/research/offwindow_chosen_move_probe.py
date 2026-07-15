"""WP-3 §1.4 pre-registered falsifier: off-window CHOSEN-move fraction.

Banked gnn_bc_040000 deploy-argmax over 320 real self-play positions; the
chosen legal node's policy_dst_slot == -1 (OFF_WINDOW_SLOT) => the dense-362
seam would have dropped the move the model actually plays.
"""
import json, sys
import torch
sys.path.insert(0, ".")
from hexo_rl.bots.strix_v1_graph import build_axis_graph_raw
from hexo_rl.probes.gnn_bc.gnn_bc_net import GnnBcNet

rust = json.load(open("/home/timmy/.claude/jobs/7d6e8877/tmp/wpa_graphs_rust.json"))
positions = json.load(open("reports/probes/gnn_integration/wpa_positions.json"))["positions"]
graphs = rust["graphs"] if isinstance(rust, dict) else rust
assert len(graphs) == len(positions), (len(graphs), len(positions))

net = GnnBcNet()
sd = torch.load("checkpoints/probes/gnn_bc/gnn_bc_040000.pt", map_location="cpu", weights_only=True)
net.load_state_dict(sd["model_state_dict"], strict=True)
net.eval()

off_chosen = 0; off_mass_total = 0.0; n = 0; off_cells = 0; legal_cells = 0
for rec, rg in zip(positions, graphs):
    stone_map = {(q, r): p for (q, r, p) in rec["stones"]}
    g = build_axis_graph_raw(stone_map, rec["current_player"], rec["moves_remaining"],
                             win_length=6, radius=6, prune_empty_edges=True,
                             threat_features=True, relative_stones=True)
    nn_, fdim = g["num_nodes"], g["fdim"]
    x = torch.tensor(g["features"], dtype=torch.float32).reshape(nn_, fdim)
    e = len(g["edge_src"])
    ei = torch.tensor([g["edge_src"], g["edge_dst"]], dtype=torch.int64) if e else torch.zeros((2,0), dtype=torch.int64)
    ea = torch.tensor(g["edge_attr"], dtype=torch.float32).reshape(e, 5) if e else torch.zeros((0,5), dtype=torch.float32)
    lm = torch.tensor(g["legal_mask"], dtype=torch.bool)
    sm = torch.tensor(g["stone_mask"], dtype=torch.bool)
    with torch.no_grad():
        logits, _ = net.policy_logits_for_graph(x, ei, ea, lm, sm)
    slots = rg["policy_dst_slot"]
    assert len(slots) == logits.shape[0], (len(slots), logits.shape)
    probs = torch.softmax(logits, dim=0)
    am = int(torch.argmax(logits))
    if slots[am] == -1:
        off_chosen += 1
    off_mass_total += float(probs[[i for i, s in enumerate(slots) if s == -1]].sum()) if any(s == -1 for s in slots) else 0.0
    off_cells += sum(1 for s in slots if s == -1)
    legal_cells += len(slots)
    n += 1

print(f"positions: {n}")
print(f"off-window CELL fraction (sanity vs WP-1 43.55%): {off_cells/legal_cells:.4f}")
print(f"off-window CHOSEN-move fraction (deploy argmax): {off_chosen}/{n} = {off_chosen/n:.4f}")
print(f"mean off-window policy MASS: {off_mass_total/n:.4f}")
