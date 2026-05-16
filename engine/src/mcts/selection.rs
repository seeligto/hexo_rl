/// PUCT selection and tree traversal.

use crate::board::{Board, MoveDiff};
use fxhash::FxHashSet;
use super::MCTSTree;

/// §P8: single-pass argmax over `[first..first+n_ch)` by PUCT score, computing
/// each child's score exactly once. Replaces the prior `.max_by()` closures
/// that re-evaluated `puct_score(a)` and `puct_score(b)` for every comparator
/// pair (`2·(K-1)` scores per descent level for K=192 children). Tie-break
/// follows `partial_cmp(...).unwrap_or(Equal)` semantics — a NaN score never
/// displaces the running best (Equal → keep current).
#[inline]
fn pick_best_puct(
    tree: &MCTSTree,
    first: usize,
    n_ch: usize,
    parent_idx: u32,
    parent_n: f32,
    fpu_value: f32,
) -> u32 {
    debug_assert!(n_ch > 0, "pick_best_puct called on a node with no children");
    let mut best_idx: u32 = first as u32;
    let mut best_score: f32 = tree.puct_score(best_idx, parent_idx, parent_n, fpu_value);
    for i in (first + 1)..(first + n_ch) {
        let score = tree.puct_score(i as u32, parent_idx, parent_n, fpu_value);
        // Strict `>` matches `max_by` Greater semantics: first equal score
        // wins, NaN comparisons preserve the running best (Equal fallback).
        if score.partial_cmp(&best_score).unwrap_or(std::cmp::Ordering::Equal)
            == std::cmp::Ordering::Greater
        {
            best_idx = i as u32;
            best_score = score;
        }
    }
    best_idx
}

impl MCTSTree {
    /// PUCT score for `child_idx`, evaluated from `parent_idx`'s player perspective.
    ///
    /// `fpu_value`: pre-computed first-play urgency value for unvisited children.
    /// For visited children this is ignored — their actual Q is used instead.
    #[inline]
    pub(crate) fn puct_score(&self, child_idx: u32, parent_idx: u32, parent_n: f32, fpu_value: f32) -> f32 {
        let child  = &self.pool[child_idx  as usize];
        let parent = &self.pool[parent_idx as usize];

        let q = if child.n_visits == 0 && child.virtual_loss_count == 0 {
            // Unvisited node: use dynamic FPU value.
            // fpu_value is already negated relative to parent when moves_remaining == 1,
            // so we use it directly (the caller computes it from the parent's Q).
            fpu_value
        } else if parent.moves_remaining == 1 {
            -child.q_value_vl(self.virtual_loss)
        } else {
            child.q_value_vl(self.virtual_loss)
        };

        let u = self.c_puct * child.prior * parent_n.sqrt()
            / (1.0 + child.n_visits as f32 + child.virtual_loss_count as f32);
        q + u
    }

    /// Walk the tree via PUCT until an unexpanded (or terminal) leaf is found.
    /// Applies virtual loss to every node on the path.
    /// Returns `(leaf_node_index, leaf_depth)`.
    pub(crate) fn select_one_leaf(&mut self, board: &mut Board, diffs: &mut Vec<MoveDiff>) -> (u32, u32) {
        let mut cur: u32 = 0;
        let mut depth = 0;
        loop {
            self.pool[cur as usize].virtual_loss_count += 1;

            let node = &self.pool[cur as usize];
            if node.is_terminal || !node.is_expanded() {
                if depth > self.max_depth_observed {
                    self.max_depth_observed = depth;
                }
                return (cur, depth);
            }

            let parent_n = (node.n_visits + node.virtual_loss_count) as f32;
            let first    = node.first_child as usize;
            let n_ch     = node.n_children  as usize;

            // KataGo-style dynamic FPU: value estimate for unvisited children.
            // explored_mass = sum of priors for all children that have been visited.
            // fpu_value = parent_q - fpu_reduction * sqrt(explored_mass)
            // When fpu_reduction == 0.0 this collapses to 0.0 (legacy behaviour).
            let fpu_value = if self.fpu_reduction > 0.0 {
                let parent_q = if node.n_visits > 0 {
                    node.w_value / node.n_visits as f32
                } else {
                    0.0
                };
                let explored_mass: f32 = (first..first + n_ch)
                    .filter(|&i| {
                        let c = &self.pool[i];
                        c.n_visits > 0 || c.virtual_loss_count > 0
                    })
                    .map(|i| self.pool[i].prior)
                    .sum();
                parent_q - self.fpu_reduction * explored_mass.sqrt()
            } else {
                0.0
            };

            // Gumbel MCTS: at root (cur==0), skip PUCT and use the forced child.
            let best = if cur == 0 {
                if let Some(forced) = self.forced_root_child {
                    forced
                } else {
                    // §P8: max_by closure invoked `puct_score` twice per
                    // comparator pair (sa, sb), costing ~2·(K-1) PUCT scores
                    // per descent level. Manual for-loop evaluates each
                    // child's score exactly once and tracks the running best
                    // with the same NaN→Ordering::Equal fallback semantics
                    // (`partial_cmp(...).unwrap_or(Equal)`).
                    pick_best_puct(self, first, n_ch, cur, parent_n, fpu_value)
                }
            } else {
                pick_best_puct(self, first, n_ch, cur, parent_n, fpu_value)
            };

            let val = self.pool[best as usize].action_idx;
            let q = (val >> 16) as i32 - 32768;
            let r = (val & 0xFFFF) as i32 - 32768;

            let diff = board
                .apply_move_tracked(q, r)
                .expect("selected move should always be legal");
            diffs.push(diff);

            cur = best;
            depth += 1;
        }
    }

    /// Select up to `n` distinct leaves for evaluation.
    pub fn select_leaves(&mut self, n: usize) -> Vec<Board> {
        self.pending.clear();
        let mut boards = Vec::with_capacity(n);
        // §P36: O(1) overlap dedup via FxHashSet<u32> on leaf pool indices.
        // Prior code scanned `self.pending` linearly (`.iter().any(...)`) for
        // every selected leaf — O(N²) in batch size. Set lives only for the
        // duration of this call; capacity matches batch hint to avoid rehash.
        let mut pending_ids: FxHashSet<u32> = FxHashSet::default();
        pending_ids.reserve(n);
        let mut board = self.root_board.clone();
        let mut diffs: Vec<MoveDiff> = Vec::with_capacity(32);

        let mut i = 0;
        let mut attempts = 0;
        let max_attempts = n * 4;

        while i < n && attempts < max_attempts {
            attempts += 1;
            diffs.clear();
            let (leaf_idx, leaf_depth) = self.select_one_leaf(&mut board, &mut diffs);
            self.depth_accum += leaf_depth as u64;
            self.sim_count += 1;

            if pending_ids.contains(&leaf_idx) {
                self.undo_virtual_loss(leaf_idx);
                self.selection_overlap_count += 1;
                while let Some(diff) = diffs.pop() {
                    board.undo_move(diff);
                }
                continue;
            }

            // §P7: TT-hit clone of 1448 B policy vector eliminated via
            // `Arc::clone` (refcount bump). `expand_and_backup_single` reads
            // the policy through `&[f32]`, so we dereference the Arc once at
            // the call site. Value is `Copy`. Reading `entry` then dropping
            // the borrow before the `&mut self` call satisfies the borrow
            // checker because `expand_and_backup_single` touches `self.pool`
            // / `self.transposition_table` (insert is a no-op for re-hits)
            // disjointly from the read-only fetch.
            if let Some(entry) = self.transposition_table.get(&board.zobrist_hash) {
                let policy = std::sync::Arc::clone(&entry.policy);
                let value = entry.value;
                self.expand_and_backup_single(leaf_idx, &board, &policy, value);
                while let Some(diff) = diffs.pop() {
                    board.undo_move(diff);
                }
                continue;
            }

            // §P6+§P9: pending now owns the fully-replayed leaf `Board`
            // instead of a `Vec<MoveDiff>`. `expand_and_backup` no longer
            // clones `root_board` + replays `apply_move(q, r)` per leaf —
            // saving ~depth board mutations per leaf (avg depth ~30 ×
            // leaf_batch=8 = ~240 mutations/sim). `boards.push(board.clone())`
            // still produces the NN-input board; the pending clone is a
            // sibling: cycle 1 verified `Board::Clone` skips `legal_cache`
            // copy, so the per-leaf cost is small relative to the eliminated
            // re-walk.
            boards.push(board.clone());
            self.pending.push((leaf_idx, board.clone()));
            pending_ids.insert(leaf_idx);

            while let Some(diff) = diffs.pop() {
                board.undo_move(diff);
            }
            i += 1;
        }

        debug_assert_eq!(board.zobrist_hash, self.root_board.zobrist_hash);
        debug_assert_eq!(board.ply, self.root_board.ply);

        boards
    }

    /// Reverse virtual loss on all nodes from `node_idx` to the root.
    pub(crate) fn undo_virtual_loss(&mut self, mut node_idx: u32) {
        loop {
            let node = &mut self.pool[node_idx as usize];
            if node.virtual_loss_count > 0 {
                node.virtual_loss_count -= 1;
            }
            let parent = node.parent;
            if parent == u32::MAX {
                break;
            }
            node_idx = parent;
        }
    }
}
