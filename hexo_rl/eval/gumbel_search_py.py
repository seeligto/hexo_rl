"""D-GUMBELSIMS — matched-position search driver (faithful Gumbel-SH + PUCT).

Drives the REAL Rust MCTS primitives (PyO3 MCTSTree) on a FIXED board to produce the
completed-Q improved-policy TARGET under a given search config, so two configs can be
compared on the SAME position (reads A/B/C/D, DESIGN §3). The TARGET is the identical
Rust `MCTSTree::get_improved_policy` production self-play records; only the search
STEERING is Python and it mirrors production exactly (DESIGN §1 parity table + §10):

  * effective_m = min(m, n_sims, n_children)            (inner.rs:757)
  * sims_per    = max(1, rem // (rem_phases * n_cand))  (inner.rs:777)
  * q̂ clamped to [-1,1] in halving score               (gumbel_search.rs:123)
  * max_n over ALL root children (not just candidates)  (gumbel_search.rs:105)
  * len<=1 break at the BOTTOM of the phase loop        (inner.rs:802)
  * Dirichlet root noise iff !is_intermediate_ply       (inner.rs:747)  [configurable]
  * FULL improved-policy vector (no top-10 truncation)
  * played move = SH winner (argmax score over survivors) (gumbel_search.rs:154)

NOT a hot path (offline analysis). The companion `gumbel_search_py_golden` test asserts
bit-equality vs a fixed Rust seed on one position before any sweep is trusted.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from engine import Board, MCTSTree
from hexo_rl.selfplay.inference import LocalInferenceEngine


def _infer_and_expand(tree: MCTSTree, engine: LocalInferenceEngine,
                      batch_size: int, n_sims: int, legal_set: bool = False) -> int:
    """Drive n_sims through the tree via synchronous inference. Returns sims done.

    ``legal_set`` (§D-DECODE Track 2): when True, expand via the MULTI-WINDOW no-drop
    legal-set action space (the action space the net trains under in self-play) instead
    of the single-window dense head. Per-cluster RAW NN outputs are pooled in Rust
    (``expand_and_backup_ls``: ``aggregate_policy_ls`` + value min-pool) so off-global-
    window cells COVERED by a cluster get a child — the structural fix for the off-window
    deploy hole. Spec scalars (stride / pass-slot / trunk) come from the engine's bound
    ``EncodingSpec``; Rust cross-checks ``trunk == board.cluster_window_size()``."""
    done = 0
    if legal_set:
        spec = engine.encoding_spec
        stride = int(spec.policy_logit_count)
        pass_slot = bool(spec.has_pass_slot)
        trunk = int(spec.cluster_window_size)
    while done < n_sims:
        batch = min(batch_size, n_sims - done)
        leaves = tree.select_leaves(batch)
        if not leaves:
            break
        if legal_set:
            policies, values, leaf_k = engine.infer_batch_per_cluster(leaves)
            tree.expand_and_backup_ls(policies, values, leaf_k, stride, pass_slot, trunk)
        else:
            policies, values = engine.infer_batch(leaves)
            tree.expand_and_backup(policies, values)
        done += len(leaves)
    return done


def _is_intermediate_ply(board: Board) -> bool:
    """Production gate (inner.rs:744): the 2nd stone of a compound turn / not the opener."""
    return bool(board.moves_remaining == 1 and board.ply > 0)


def _apply_root_dirichlet(tree: MCTSTree, alpha: float, epsilon: float,
                          rng: np.random.Generator) -> None:
    n_ch = tree.root_n_children()
    if n_ch <= 0:
        return
    noise = rng.dirichlet([alpha] * n_ch).astype(np.float32).tolist()
    tree.apply_dirichlet_to_root(noise, epsilon)


def _make_tree(c_puct: float, virtual_loss: float, fpu_reduction: float,
               quiescence_enabled: bool, quiescence_blend_2: float) -> MCTSTree:
    return MCTSTree(c_puct=c_puct, virtual_loss=virtual_loss, fpu_reduction=fpu_reduction,
                    quiescence_enabled=quiescence_enabled, quiescence_blend_2=quiescence_blend_2)


def _collect_witnesses(tree: MCTSTree, board: Board, improved: np.ndarray) -> Dict:
    """v_mix-reliance (DESIGN §3D) + root value + per-move Q (for value-regret) witness."""
    children = tree.get_root_children_info()   # ((q,r), pool_idx, prior, visits, q_value)
    unvisited_mass = 0.0
    child_q: Dict[Tuple[int, int], float] = {}      # coord -> root-perspective Q (visited only)
    child_visits: Dict[Tuple[int, int], int] = {}
    for (q, r), _pool, _prior, visits, q_value in children:
        coord = (int(q), int(r))
        child_visits[coord] = int(visits)
        if visits > 0:
            child_q[coord] = float(q_value)
        else:
            idx = board.to_flat(q, r)
            if 0 <= idx < len(improved):
                unvisited_mass += float(improved[idx])
    return {
        "root_value": float(tree.root_value()),
        "unvisited_mass": unvisited_mass,
        "n_children": len(children),
        "child_q": child_q,
        "child_visits": child_visits,
    }


def run_puct_on_board(engine: LocalInferenceEngine, board: Board, *, n_sims: int,
                      c_puct: float = 1.5, dirichlet: bool = False, alpha: float = 0.05,
                      epsilon: float = 0.10, leaf_batch: int = 8,
                      c_visit: float = 50.0, c_scale: float = 1.0,
                      virtual_loss: float = 1.0, fpu_reduction: float = 0.25,
                      quiescence_enabled: bool = True, quiescence_blend_2: float = 0.3,
                      rng: Optional[np.random.Generator] = None) -> Dict:
    """Standard PUCT search; returns the completed-Q improved-policy (the target the
    golong completed_q_values=true regime records) + visit-policy witness + played move."""
    rng = rng if rng is not None else np.random.default_rng(0)
    tree = _make_tree(c_puct, virtual_loss, fpu_reduction, quiescence_enabled, quiescence_blend_2)
    tree.new_game(board)
    _infer_and_expand(tree, engine, batch_size=1, n_sims=1)   # expand root
    if dirichlet and not _is_intermediate_ply(board):
        _apply_root_dirichlet(tree, alpha, epsilon, rng)
    _infer_and_expand(tree, engine, batch_size=leaf_batch, n_sims=max(0, n_sims - 1))

    improved = np.asarray(tree.get_improved_policy(c_visit=c_visit, c_scale=c_scale), dtype=np.float64)
    visit = np.asarray(tree.get_policy(temperature=1.0), dtype=np.float64)
    top = tree.get_top_visits(1)
    played = (int(top[0][0][0]), int(top[0][0][1])) if top else None
    out = {"improved_policy": improved, "visit_policy": visit, "played_move": played, "mode": "puct"}
    out.update(_collect_witnesses(tree, board, improved))
    return out


def run_gumbel_on_board(engine: LocalInferenceEngine, board: Board, *, n_sims: int,
                        m: int = 16, c_visit: float = 50.0, c_scale: float = 1.0,
                        c_puct: float = 1.5, dirichlet: bool = False, alpha: float = 0.05,
                        epsilon: float = 0.10, leaf_batch: int = 8,
                        virtual_loss: float = 1.0, fpu_reduction: float = 0.25,
                        quiescence_enabled: bool = True, quiescence_blend_2: float = 0.3,
                        gumbel_scale: float = 1.0, legal_set: bool = False,
                        rng: Optional[np.random.Generator] = None) -> Dict:
    """Gumbel Sequential-Halving search (production-parity steering). Returns the
    completed-Q improved-policy TARGET (identical Rust fn), the SH-winner played move
    (read C), the visit witness, and the v_mix-reliance witness (read D).

    ``gumbel_scale`` multiplies the Gumbel(0,1) root noise (mctx convention).
    ``1.0`` = production self-play (training-time root exploration). ``0.0`` = the
    canonical STRENGTH-EVAL head: the root noise is zeroed so candidate selection +
    the SH winner are a DETERMINISTIC argmax over (log_prior + completed-Q sigma) — the
    AGZ/mctx/LightZero deterministic deploy head. The interior PUCT descent and the
    completed-Q SH math are byte-identical to self-play either way; only the +g_c root
    term changes. D-LOCALIZE P4: the in-loop strength gate runs this at scale 0.0."""
    rng = rng if rng is not None else np.random.default_rng(0)
    tree = _make_tree(c_puct, virtual_loss, fpu_reduction, quiescence_enabled, quiescence_blend_2)
    tree.new_game(board)
    _infer_and_expand(tree, engine, batch_size=1, n_sims=1, legal_set=legal_set)   # expand root

    if dirichlet and not _is_intermediate_ply(board):
        _apply_root_dirichlet(tree, alpha, epsilon, rng)   # BEFORE candidate select + target

    children = tree.get_root_children_info()                # priors now dirichlet-noised
    n_children = len(children)
    if n_children == 0:
        improved = np.asarray(tree.get_improved_policy(c_visit=c_visit, c_scale=c_scale), dtype=np.float64)
        out = {"improved_policy": improved,
               "visit_policy": np.asarray(tree.get_policy(temperature=1.0), dtype=np.float64),
               "played_move": None, "mode": "gumbel"}
        out.update(_collect_witnesses(tree, board, improved))
        return out

    eff_m = min(int(m), int(n_sims), n_children)             # effective_m parity (inner.rs:757)

    # Gumbel(0,1) noise + log-prior → top-m candidates.
    # gumbel_scale=0.0 (eval) zeroes the root noise → deterministic deploy head
    # (mctx gumbel_scale=0): candidate select + SH winner = argmax(log_prior + sigma).
    u = rng.uniform(1e-10, 1.0 - 1e-7, size=n_children)
    gumbels = float(gumbel_scale) * (-np.log(-np.log(u)))
    log_priors = np.log(np.maximum([c[2] for c in children], 1e-8))
    order = np.argsort(-(gumbels + log_priors))             # descending
    candidates = list(order[:eff_m])

    num_phases = 1 if eff_m <= 1 else math.ceil(math.log2(eff_m))
    budget = max(0, n_sims - 1)                             # root expansion ate one
    sims_used = 0

    def _max_n_all() -> int:
        # max visits over ALL root children (parity: gumbel_search.rs:105), not candidates.
        return max((c[3] for c in tree.get_root_children_info()), default=0)

    def _score(info_by_pool, max_n) -> "list":
        out = []
        for ci in candidates:
            pool_idx = children[ci][1]
            info = info_by_pool.get(pool_idx)
            visits = info[3] if info else 0
            q_hat = float(info[4]) if (info and visits > 0) else 0.0
            q_hat = max(-1.0, min(1.0, q_hat))             # q̂ clamp parity (gumbel_search.rs:123)
            sigma = (c_visit + max_n) * c_scale * q_hat
            out.append((ci, float(gumbels[ci] + log_priors[ci] + sigma)))
        return out

    for phase in range(num_phases):
        if sims_used >= budget:
            break
        remaining = budget - sims_used
        remaining_phases = num_phases - phase
        sims_per = max(1, remaining // (remaining_phases * len(candidates)))
        for ci in candidates:
            if sims_used >= budget:
                break
            tree.forced_root_child = int(children[ci][1])
            done = _infer_and_expand(tree, engine, batch_size=leaf_batch,
                                     n_sims=min(sims_per, budget - sims_used),
                                     legal_set=legal_set)
            sims_used += done
        tree.forced_root_child = None

        if len(candidates) <= 1:                            # break at BOTTOM (inner.rs:802)
            break
        refreshed = tree.get_root_children_info()
        info_by_pool = {c[1]: c for c in refreshed}
        scored = _score(info_by_pool, _max_n_all())
        scored.sort(key=lambda t: t[1], reverse=True)
        keep = (len(scored) + 1) // 2                       # div_ceil(2) parity
        candidates = [ci for ci, _ in scored[:keep]]

    tree.forced_root_child = None

    # Played move = SH winner: argmax score over surviving candidates (gumbel_search.rs:154).
    info_by_pool = {c[1]: c for c in tree.get_root_children_info()}
    final_scored = _score(info_by_pool, _max_n_all())
    win_ci = max(final_scored, key=lambda t: t[1])[0] if final_scored else candidates[0]
    wq, wr = children[win_ci][0]
    played = (int(wq), int(wr))

    improved = np.asarray(tree.get_improved_policy(c_visit=c_visit, c_scale=c_scale), dtype=np.float64)
    visit = np.asarray(tree.get_policy(temperature=1.0), dtype=np.float64)
    out = {"improved_policy": improved, "visit_policy": visit, "played_move": played,
           "mode": "gumbel", "effective_m": eff_m, "sims_used": sims_used + 1}
    out.update(_collect_witnesses(tree, board, improved))
    return out
