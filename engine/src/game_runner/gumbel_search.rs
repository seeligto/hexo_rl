//! Gumbel MCTS Sequential-Halving search state.
//!
//! Per-search state for Gumbel-Top-k root sampling with Sequential Halving
//! (Danihelka et al., "Policy improvement by planning with Gumbel", ICLR 2022).
//!
//! Created once at the start of each MCTS search call (after root expansion).
//! Not stored in the node pool or transposition table. Visibility is
//! `pub(super)` so `worker_loop.rs` can construct and drive it, while
//! everything else in the crate stays out.

use rand::RngExt;

/// Per-search state for Gumbel-Top-k + Sequential Halving.
pub struct GumbelSearchState {
    /// Gumbel(0,1) noise values, one per root child (indexed by child offset).
    pub gumbel_values: Vec<f32>,
    /// Log-prior for each root child.
    pub log_priors: Vec<f32>,
    /// Child offsets (relative to first_child) still in the candidate set.
    pub candidates: Vec<usize>,
    /// ceil(log2(m)) — number of halving phases.
    pub num_phases: usize,
    /// Sigma scaling constants (same as get_improved_policy).
    pub c_visit: f32,
    pub c_scale: f32,
    /// Pool index of root's first child (for converting offsets to pool indices).
    pub first_child: u32,
    /// moves_remaining of root at construction time — used to flip child Q perspective.
    pub root_mr: u8,
    /// Cached (n_visits, w_value) per root child, refreshed each halving phase.
    pub cached_children: Vec<(u32, f32)>,
}

impl GumbelSearchState {
    /// Create a new Gumbel search state after root has been expanded.
    ///
    /// Generates Gumbel(0,1) noise for all root children, computes
    /// `g(a) + log_prior(a)` scores, and selects the top `m` candidates.
    pub fn new(
        tree: &crate::mcts::MCTSTree,
        m: usize,
        c_visit: f32,
        c_scale: f32,
        rng: &mut impl rand::Rng,
    ) -> Self {
        let children = tree.get_root_children_info();
        let n_children = children.len();

        // Generate Gumbel(0,1) = -log(-log(U)), U ~ Uniform(0,1)
        let gumbel_values: Vec<f32> = (0..n_children)
            .map(|_| {
                let u: f32 = rng.random::<f32>().clamp(1e-10, 1.0 - 1e-7);
                -(-u.ln()).ln()
            })
            .collect();

        let log_priors: Vec<f32> = children
            .iter()
            .map(|(_, prior)| prior.max(1e-8).ln())
            .collect();

        // Score = g(a) + log_prior(a); select top m
        let effective_m = m.min(n_children);
        let mut scored: Vec<(usize, f32)> = (0..n_children)
            .map(|i| (i, gumbel_values[i] + log_priors[i]))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let candidates: Vec<usize> = scored.iter().take(effective_m).map(|(i, _)| *i).collect();

        let num_phases = if effective_m <= 1 { 1 } else { (effective_m as f64).log2().ceil() as usize };

        let first_child = tree.pool[0].first_child;
        let root_mr = tree.pool[0].moves_remaining;
        let n_ch = tree.pool[0].n_children as usize;
        let cached_children: Vec<(u32, f32)> = (0..n_ch)
            .map(|j| {
                let c = &tree.pool[first_child as usize + j];
                (c.n_visits, c.w_value)
            })
            .collect();

        GumbelSearchState {
            gumbel_values,
            log_priors,
            candidates,
            num_phases,
            c_visit,
            c_scale,
            first_child,
            root_mr,
            cached_children,
        }
    }

    /// Refresh cached child stats from the tree. Call once per halving phase.
    pub fn refresh_cache(&mut self, tree: &crate::mcts::MCTSTree) {
        let n_ch = self.cached_children.len();
        for j in 0..n_ch {
            let c = &tree.pool[self.first_child as usize + j];
            self.cached_children[j] = (c.n_visits, c.w_value);
        }
    }

    /// Max visit count across all root children (from cache).
    pub fn max_n(&self) -> u32 {
        self.cached_children.iter().map(|(n, _)| *n).max().unwrap_or(0)
    }

    /// Compute the Gumbel + log_prior + sigma(Q) score for a candidate.
    /// `child_offset` is relative to `first_child`.
    /// `max_n` is the max visit count across all root children (cached per phase).
    ///
    /// Unvisited candidates (n_visits=0) return score = gumbel + log_prior with
    /// no Q contribution (q_hat=0 → sigma=0). This is correct per the paper:
    /// before any simulations, score is just the Gumbel perturbation of the prior.
    pub fn score(&self, child_offset: usize, max_n: u32) -> f32 {
        let child = &self.cached_children[child_offset];
        let q_hat = if child.0 > 0 {
            // child.1 is w_value in child's own perspective; negate when root is at
            // moves_remaining==1 (children belong to the opponent).
            let raw = child.1 / child.0 as f32;
            let root_perspective = if self.root_mr == 1 { -raw } else { raw };
            root_perspective.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // sigma(q) = (c_visit + max_b N(b)) * c_scale * q
        let sigma = (self.c_visit + max_n as f32) * self.c_scale * q_hat;

        self.gumbel_values[child_offset] + self.log_priors[child_offset] + sigma
    }

    /// Rank candidates by score, keep top half. Refreshes cache from tree first.
    pub fn halve_candidates(&mut self, tree: &crate::mcts::MCTSTree) {
        if self.candidates.len() <= 1 {
            return;
        }
        self.refresh_cache(tree);
        let max_n = self.max_n();
        // Sort by descending Gumbel+log_prior+sigma(Q) score.
        // Pre-compute into scored pairs to avoid self-borrow in sort closure.
        let mut scored: Vec<(usize, f32)> = self.candidates
            .iter()
            .map(|&c| (c, self.score(c, max_n)))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep = (scored.len() + 1) / 2;
        scored.truncate(keep);
        self.candidates = scored.into_iter().map(|(c, _)| c).collect();
    }

    /// Return the pool index of the best candidate (Sequential Halving winner).
    pub fn best_action_pool_idx(&mut self, tree: &crate::mcts::MCTSTree) -> u32 {
        self.refresh_cache(tree);
        let max_n = self.max_n();
        let best_offset = self.candidates
            .iter()
            .max_by(|&&a, &&b| {
                let sa = self.score(a, max_n);
                let sb = self.score(b, max_n);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(0);
        self.first_child + best_offset as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, BOARD_SIZE};

    fn setup_tree_for_gumbel() -> crate::mcts::MCTSTree {
        let mut tree = crate::mcts::MCTSTree::new(1.5);
        // Use mr=2 (start of compound turn) so child w_values are already in
        // root's perspective and score() applies no Q negation.
        let mut board = Board::new();
        board.apply_move(0, 0).expect("(0,0) must be legal on fresh board");
        assert_eq!(board.moves_remaining, 2);
        tree.new_game(board);

        // Expand root with uniform priors.
        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let policy = vec![1.0 / n_actions as f32; n_actions];
        let _leaves = tree.select_leaves(1);
        tree.expand_and_backup(&[policy], &[0.0]);
        tree
    }

    #[test]
    fn test_gumbel_topk_selection() {
        let tree = setup_tree_for_gumbel();
        let n_children = tree.root_n_children();
        assert!(n_children > 16, "fresh board should have many legal moves");

        let mut rng = rand::rng();
        let gs = GumbelSearchState::new(&tree, 16, 50.0, 1.0, &mut rng);

        assert_eq!(gs.candidates.len(), 16, "should select m=16 candidates");
        assert_eq!(gs.gumbel_values.len(), n_children);
        assert_eq!(gs.log_priors.len(), n_children);

        // All candidates should be unique and within valid range.
        let mut seen = std::collections::HashSet::new();
        for &c in &gs.candidates {
            assert!(c < n_children, "candidate offset {c} out of range");
            assert!(seen.insert(c), "duplicate candidate {c}");
        }
    }

    #[test]
    fn test_gumbel_topk_few_legal_moves() {
        // When legal moves < m, all moves should be candidates.
        let tree = setup_tree_for_gumbel();
        let n_children = tree.root_n_children();
        let mut rng = rand::rng();

        // Request m = 1000, but there are only n_children legal moves.
        let gs = GumbelSearchState::new(&tree, 1000, 50.0, 1.0, &mut rng);
        assert_eq!(gs.candidates.len(), n_children,
            "with m > legal moves, all {} moves should be candidates", n_children);
    }

    #[test]
    fn test_sequential_halving_phases_count() {
        let tree = setup_tree_for_gumbel();
        let mut rng = rand::rng();

        let gs8 = GumbelSearchState::new(&tree, 8, 50.0, 1.0, &mut rng);
        assert_eq!(gs8.num_phases, 3, "ceil(log2(8)) = 3");

        let gs16 = GumbelSearchState::new(&tree, 16, 50.0, 1.0, &mut rng);
        assert_eq!(gs16.num_phases, 4, "ceil(log2(16)) = 4");

        let gs1 = GumbelSearchState::new(&tree, 1, 50.0, 1.0, &mut rng);
        assert_eq!(gs1.num_phases, 1, "ceil(log2(1)) should be 1");
        assert_eq!(gs1.candidates.len(), 1);
    }

    #[test]
    fn test_halve_candidates_reduces_count() {
        let mut tree = setup_tree_for_gumbel();
        let mut rng = rand::rng();
        let mut gs = GumbelSearchState::new(&tree, 8, 50.0, 1.0, &mut rng);
        assert_eq!(gs.candidates.len(), 8);

        // Give some candidates visits so they have different Q values.
        let first = tree.pool[0].first_child as usize;
        for (i, &cand) in gs.candidates.iter().enumerate() {
            let pool_idx = first + cand;
            tree.pool[pool_idx].n_visits = (i + 1) as u32;
            tree.pool[pool_idx].w_value = (i as f32) * 0.1;
        }

        gs.halve_candidates(&tree);
        assert_eq!(gs.candidates.len(), 4, "8 → 4 after one halving");

        gs.halve_candidates(&tree);
        assert_eq!(gs.candidates.len(), 2, "4 → 2 after two halvings");

        gs.halve_candidates(&tree);
        assert_eq!(gs.candidates.len(), 1, "2 → 1 after three halvings");
    }

    #[test]
    fn test_gumbel_score_uses_sigma() {
        let mut tree = setup_tree_for_gumbel();
        let mut rng = rand::rng();
        let mut gs = GumbelSearchState::new(&tree, 8, 50.0, 1.0, &mut rng);

        let first = tree.pool[0].first_child as usize;
        let c0 = gs.candidates[0];
        let c1 = gs.candidates[1];

        // Give c0 a high Q, c1 a low Q.
        tree.pool[first + c0].n_visits = 10;
        tree.pool[first + c0].w_value = 8.0; // Q = 0.8
        tree.pool[first + c1].n_visits = 10;
        tree.pool[first + c1].w_value = -8.0; // Q = -0.8

        gs.refresh_cache(&tree);
        let max_n = gs.max_n();
        let s0 = gs.score(c0, max_n);
        let s1 = gs.score(c1, max_n);
        // Higher Q should lead to higher score (sigma term dominates at high visit counts).
        assert!(s0 > s1, "higher Q should give higher score: {s0} vs {s1}");
    }

    #[test]
    fn test_best_action_pool_idx() {
        let mut tree = setup_tree_for_gumbel();
        let mut rng = rand::rng();
        let mut gs = GumbelSearchState::new(&tree, 4, 50.0, 1.0, &mut rng);

        let first = tree.pool[0].first_child;

        // Give the last candidate a very high Q.
        let best_cand = *gs.candidates.last().unwrap();
        tree.pool[(first as usize) + best_cand].n_visits = 100;
        tree.pool[(first as usize) + best_cand].w_value = 90.0; // Q = 0.9

        let best_pool = gs.best_action_pool_idx(&tree);
        assert_eq!(best_pool, first + best_cand as u32,
            "best action should be the high-Q candidate");
    }
}
