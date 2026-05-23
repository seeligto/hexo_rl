"""§S181 Track A — A4 H-CE-STRENGTH: per-class CE gradient asymmetry.

Quantifies the L43 entropy asymmetry (colony H_frac 0.484 vs extension
H_frac 0.805) as actual gradient strength on the trunk's output.

Mechanism: lower target entropy = sharper target = larger per-sample CE
gradient. If colony positions push the trunk harder per sample, equal
sample counts produce unequal effective training pressure → colony
positions effectively oversampled per unit of corpus content.

Method:
  1. T3 40-position bank → realize Boards.
  2. For each position, run production MCTS (n_sims=400, c_puct=1.5,
     dirichlet on) using bootstrap_model_v6 to generate a visit-count
     policy target.
  3. Forward each board through bootstrap_model_v6; compute CE loss
     against the MCTS target.
  4. Compute gradient L2 norm of the CE loss w.r.t. the trunk's output
     tensor (cheap — no backward through weights).
  5. Bucket by class. Report ratio colony / extension.

Pre-registered verdict (LITERAL L13):
  ASYMMETRY-CONFIRMED   ratio > 1.5
  NEUTRAL               ratio in [0.85, 1.15]
  REVERSE-ASYMMETRY     ratio < 0.67
  INCONCLUSIVE          otherwise

Outputs:
  audit/structural/track_a/A4_h_ce_strength.json
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from engine import Board, MCTSTree
from hexo_rl.viewer.model_loader import load_model
from hexo_rl.selfplay.inference import LocalInferenceEngine
from hexo_rl.encoding import lookup as lookup_encoding

BANK = REPO / "tests" / "fixtures" / "value_spread_bank.json"
ANCHOR = REPO / "checkpoints" / "bootstrap_model_v6.pt"
BANK_SHA = "934204713620d171743820aea6907cf4e117ca97c69e50052b991a3fdcc23991"

# Production MCTS settings (configs/selfplay.yaml mirror).
MCTS_N_SIMS = 400
MCTS_C_PUCT = 1.5
MCTS_DIR_ALPHA = 0.05
MCTS_EPS = 0.10
MCTS_LEAF_BATCH = 8
SEED = 20260523


def realize_state_v6(spec: dict, v6_kept: list[int]) -> tuple[Board, np.ndarray] | None:
    b = Board()
    for q, r in spec["moves"]:
        try:
            b.apply_move(int(q), int(r))
        except Exception:
            return None
    full = b.to_tensor().reshape(18, 19, 19)
    state8 = full[v6_kept].astype(np.float32).copy()
    return b, state8


def bank_sha(positions: list[dict]) -> str:
    h = hashlib.sha256()
    for spec in positions:
        h.update(spec["name"].encode())
        h.update(spec["pos_class"].encode())
        for q, r in spec["moves"]:
            h.update(f"{int(q)},{int(r)};".encode())
    return h.hexdigest()


def run_mcts_target(board: Board, eng: LocalInferenceEngine,
                     rng: np.random.Generator) -> np.ndarray:
    """Run production MCTS; return visit-count target (n_actions,) normalised
    to sum to 1."""
    tree = MCTSTree(c_puct=MCTS_C_PUCT)
    tree.new_game(board.clone())
    # expand root
    leaves = tree.select_leaves(1)
    if leaves:
        pols, vals = eng.infer_batch(leaves)
        tree.expand_and_backup(pols, vals)
    # root Dirichlet
    nch = tree.root_n_children()
    if nch > 0:
        noise = rng.dirichlet([MCTS_DIR_ALPHA] * nch).tolist()
        tree.apply_dirichlet_to_root(noise, MCTS_EPS)
    # remaining sims
    done = 1
    while done < MCTS_N_SIMS:
        b = min(MCTS_LEAF_BATCH, MCTS_N_SIMS - done)
        leaves = tree.select_leaves(b)
        if not leaves:
            break
        pols, vals = eng.infer_batch(leaves)
        tree.expand_and_backup(pols, vals)
        done += len(leaves)
    # extract visit counts from top moves
    visits = tree.get_top_visits(361)  # exhaustive top
    target = np.zeros(362, dtype=np.float32)  # v6 has 362 (361 cells + pass)
    total = 0
    for (q, r), v, prior, qv in visits:
        fi = board.to_flat(q, r)
        if 0 <= fi < 362:
            target[fi] = float(v)
            total += float(v)
    if total > 0:
        target /= total
    return target


def main():
    t0 = time.time()
    print(f"loading bank {BANK} ...")
    data = json.loads(BANK.read_text())
    positions = data["positions"]
    sha = bank_sha(positions)
    if sha != BANK_SHA:
        raise SystemExit(f"bank SHA {sha} != pinned {BANK_SHA} — STOP")
    print(f"  bank SHA verified: {sha[:16]}…")

    device = torch.device("cpu")
    print(f"loading model {ANCHOR} ...")
    net, meta, _ = load_model(ANCHOR, device=device)
    net.eval()
    eng = LocalInferenceEngine(net, device)

    v6 = lookup_encoding("v6")
    v6_kept = list(v6.kept_plane_indices)
    rng = np.random.default_rng(SEED)

    per_pos: list[dict[str, Any]] = []
    print(f"running MCTS + grad on {len(positions)} positions "
          f"(n_sims={MCTS_N_SIMS}, c_puct={MCTS_C_PUCT}, dir={MCTS_DIR_ALPHA}) …")
    for k, spec in enumerate(positions):
        res = realize_state_v6(spec, v6_kept)
        if res is None:
            print(f"  SKIP illegal: {spec['name']}")
            continue
        board, state8 = res

        # 1) generate MCTS target (numpy)
        target = run_mcts_target(board, eng, rng)
        target_entropy_nats = float(-(target[target > 0] * np.log(target[target > 0])).sum())
        n_legal = int((target > 0).sum())
        h_uniform = float(np.log(n_legal)) if n_legal > 1 else 1.0
        target_h_frac = target_entropy_nats / h_uniform if h_uniform > 0 else 0.0

        # 2) forward with grad tracking on trunk output
        x = torch.from_numpy(state8).unsqueeze(0).float().to(device)  # (1, 8, 19, 19)
        x.requires_grad_(False)
        net.zero_grad()
        # Manual forward — replicate net.forward but capture trunk_out
        # and skip aux heads. For v6 (has_pass_slot=True), mask=None.
        trunk_out = net.trunk(x, mask=None, mask_sum_hw=None)  # (1, C, H, W)
        trunk_out.retain_grad()
        # v6 policy head: policy_conv + flatten + policy_fc
        p = net.policy_conv(trunk_out)  # (1, 2, H, W)
        p_flat = p.reshape(1, -1)
        policy_logits = net.policy_fc(p_flat)  # (1, n_actions=362)
        log_policy = torch.nn.functional.log_softmax(policy_logits, dim=-1)
        # CE loss (target is sum-to-1 distribution)
        tgt_t = torch.from_numpy(target).unsqueeze(0).float().to(device)  # (1, 362)
        ce_loss = -(tgt_t * log_policy).sum(dim=-1).mean()

        # gradient L2 to trunk output (no weight backward)
        trunk_grad = torch.autograd.grad(ce_loss, trunk_out, retain_graph=False)[0]
        grad_l2 = float(trunk_grad.norm().item())

        per_pos.append(dict(
            name=spec["name"],
            pos_class=spec["pos_class"],
            ce_loss=round(float(ce_loss.item()), 4),
            target_entropy_nats=round(target_entropy_nats, 4),
            target_h_frac=round(target_h_frac, 4),
            n_legal=n_legal,
            trunk_grad_l2=round(grad_l2, 6),
        ))
        if (k + 1) % 10 == 0:
            print(f"  done {k+1}/{len(positions)}")

    # Aggregate
    def agg(filt):
        sub = [p for p in per_pos if filt(p["pos_class"])]
        if not sub:
            return None
        return dict(
            n=len(sub),
            mean_ce_loss=round(float(np.mean([p["ce_loss"] for p in sub])), 4),
            mean_target_h_frac=round(float(np.mean([p["target_h_frac"] for p in sub])), 4),
            mean_trunk_grad_l2=round(float(np.mean([p["trunk_grad_l2"] for p in sub])), 6),
            std_trunk_grad_l2=round(float(np.std([p["trunk_grad_l2"] for p in sub])), 6),
        )

    by_class = dict(
        colony=agg(lambda c: c == "colony"),
        extension=agg(lambda c: "extension" in c),
    )

    col = by_class["colony"]
    ext = by_class["extension"]
    if col and ext and ext["mean_trunk_grad_l2"] > 0:
        ratio = round(col["mean_trunk_grad_l2"] / ext["mean_trunk_grad_l2"], 4)
    else:
        ratio = None

    # Pre-registered verdict (LITERAL L13)
    if ratio is None:
        verdict = "INCONCLUSIVE"
    elif ratio > 1.5:
        verdict = "ASYMMETRY-CONFIRMED"
    elif 0.85 <= ratio <= 1.15:
        verdict = "NEUTRAL"
    elif ratio < 0.67:
        verdict = "REVERSE-ASYMMETRY"
    else:
        verdict = "INCONCLUSIVE"

    result = dict(
        meta=dict(
            bank_sha256=sha,
            anchor=str(ANCHOR.name),
            anchor_step=meta.get("step"),
            n_positions=len(per_pos),
            mcts_settings=dict(n_sims=MCTS_N_SIMS, c_puct=MCTS_C_PUCT,
                               dir_alpha=MCTS_DIR_ALPHA, epsilon=MCTS_EPS,
                               leaf_batch=MCTS_LEAF_BATCH),
            seed=SEED,
            wall_s=round(time.time() - t0, 1),
        ),
        per_position=per_pos,
        by_class=by_class,
        ratio_colony_over_extension_grad_l2=ratio,
        verdict=verdict,
    )

    out = REPO / "audit" / "structural" / "track_a" / "A4_h_ce_strength.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}  ({result['meta']['wall_s']}s)")
    print(json.dumps({
        "by_class": by_class,
        "ratio_grad_l2": ratio,
        "verdict": verdict,
    }, indent=2))


if __name__ == "__main__":
    main()
