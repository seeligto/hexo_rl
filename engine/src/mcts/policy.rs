//! Policy extraction for MCTSTree — temperature-applied policy,
//! Gumbel completed-Q improved policy, root children info,
//! Dirichlet noise application at root, top-visits selection.
//!
//! Extracted from mcts/mod.rs in §163.

use super::MCTSTree;
use crate::game_runner::records::{self, LegalSetPolicy};
use fxhash::FxHashMap;

impl MCTSTree {
    /// §P2 — `n_actions` is the policy stride supplied by the caller (=
    /// `spec.policy_stride()`). v6/v6w25/v7full = bs²+1 (with pass slot);
    /// v8/v8_canvas_realness = bs² (no pass slot). Pre-P2 this method
    /// computed `bs * bs + 1` unconditionally, which produced phantom
    /// pass-slot vectors for v8 (audit FD.4 / FE.3).
    pub fn get_policy(&self, temperature: f32, n_actions: usize) -> Vec<f32> {
        let mut policy = vec![0.0f32; n_actions];

        let root = &self.pool[0];
        if !root.is_expanded() {
            return policy;
        }

        let first = root.first_child as usize;
        let n_ch  = root.n_children  as usize;

        if temperature == 0.0 {
            if let Some(best) = (first..first + n_ch)
                .max_by_key(|&i| self.pool[i].n_visits)
            {
                let val = self.pool[best].action_idx;
                let q = (val >> 16) as i32 - 32768;
                let r = (val & 0xFFFF) as i32 - 32768;
                let action = self.root_board.window_flat_idx(q, r);

                if action < n_actions {
                    policy[action] = 1.0;
                }
            }
        } else {
            let visits: Vec<f32> = (first..first + n_ch)
                .map(|i| (self.pool[i].n_visits as f32).powf(1.0 / temperature))
                .collect();
            let total: f32 = visits.iter().sum();
            if total > 0.0 {
                for (j, &v) in visits.iter().enumerate() {
                    let val = self.pool[first + j].action_idx;
                    let q = (val >> 16) as i32 - 32768;
                    let r = (val & 0xFFFF) as i32 - 32768;
                    let action = self.root_board.window_flat_idx(q, r);

                    if action < n_actions {
                        policy[action] = v / total;
                    }
                }
            }
        }

        policy
    }

    /// Compute improved policy targets using Gumbel completed Q-values
    /// (Danihelka et al., Gumbel AlphaZero, ICLR 2022 §4, Appendix D Eq. 33).
    ///
    /// §P2 — `n_actions` is the policy stride supplied by the caller (=
    /// `spec.policy_stride()`); see `get_policy` for the rationale. Pre-P2
    /// this method computed `board_size² + 1` unconditionally.
    ///
    /// §P33 (Wave 4): the separate `logits: Vec<f32; n_actions>` buffer is
    /// retired. `child_data` already carries (action, visits, prior, q_val)
    /// and the softmax is sparse (only entries in `child_data` are non-zero
    /// before exp). The max-logit / sum-exp passes now iterate `child_data`
    /// directly instead of scanning a full n_actions-wide sentinel vector,
    /// and the final exp/scatter writes straight into `policy[action]`.
    /// SmallVec swap for `child_data` skipped: `smallvec` is not a dep at
    /// HEAD; per Wave 4 launch prompt we do NOT add a new dep just for this
    /// proposal. `Vec::with_capacity(n_ch)` keeps the single-allocation
    /// child_data path.
    ///
    /// Returns an `n_actions`-dim probability distribution that incorporates
    /// MCTS Q-values into the prior, giving useful policy signal even at
    /// low simulation counts.
    pub fn get_improved_policy(
        &self,
        n_actions: usize,
        c_visit: f32,
        c_scale: f32,
    ) -> Vec<f32> {
        let mut policy = vec![0.0f32; n_actions];

        let root = &self.pool[0];
        if !root.is_expanded() {
            return policy;
        }

        let first = root.first_child as usize;
        let n_ch = root.n_children as usize;

        // Children store w_value in their own player-to-move perspective (backup.rs negamax).
        // When root.moves_remaining==1 the children belong to the opponent, so negate their Q
        // to bring them into root's perspective before computing completed-Q targets.
        let q_sign: f32 = if self.pool[0].moves_remaining == 1 { -1.0 } else { 1.0 };

        // Collect per-child data: (flat_action_idx, visits, prior, q_value)
        let mut child_data: Vec<(usize, u32, f32, f32)> = Vec::with_capacity(n_ch);
        let mut sum_n: u32 = 0;
        let mut max_n: u32 = 0;
        let mut visited_prior_sum: f32 = 0.0;
        let mut policy_weighted_q: f32 = 0.0;

        for j in 0..n_ch {
            let child = &self.pool[first + j];
            let val = child.action_idx;
            let q = (val >> 16) as i32 - 32768;
            let r = (val & 0xFFFF) as i32 - 32768;
            let action = self.root_board.window_flat_idx(q, r);
            if action >= n_actions {
                continue;
            }

            let visits = child.n_visits;
            let prior = child.prior;
            let q_val = if visits > 0 {
                q_sign * child.w_value / visits as f32
            } else {
                0.0
            };

            sum_n += visits;
            if visits > max_n {
                max_n = visits;
            }
            if visits > 0 {
                visited_prior_sum += prior;
                policy_weighted_q += prior * q_val;
            }

            child_data.push((action, visits, prior, q_val));
        }

        // Edge case: no visits at all — return prior distribution
        if sum_n == 0 {
            let mut total_prior = 0.0f32;
            for &(action, _, prior, _) in &child_data {
                policy[action] = prior;
                total_prior += prior;
            }
            if total_prior > 0.0 {
                for p in &mut policy {
                    *p /= total_prior;
                }
            }
            return policy;
        }

        // v_hat: root value estimate (W/N from root node)
        let v_hat = root.w_value / root.n_visits as f32;

        // v_mix: mixed value estimate for unvisited actions (paper Eq. 33).
        // Note: child.prior values come from softmax and are assumed to sum to
        // ~1.0 over legal actions. No normalization step needed unless expansion
        // pruning is added in the future.
        let v_mix = if visited_prior_sum > 1e-8 {
            let sum_n_f = sum_n as f32;
            // §F2: `v_hat + (sum_n_f / visited_prior_sum) * policy_weighted_q`
            // → fused FMA on the inner mul-add.
            (1.0 / (1.0 + sum_n_f))
                * (sum_n_f / visited_prior_sum).mul_add(policy_weighted_q, v_hat)
        } else {
            v_hat
        };

        // Build completed Q-values and compute pi_improved = softmax(log_prior + sigma(completedQ))
        // §P33: logits are sparse over legal actions — every illegal slot is
        // -inf and contributes 0 to the softmax sum. Iterate child_data
        // twice (max + sum) and once for the final scatter, instead of
        // allocating an n_actions-wide -inf vector. Two-pass keeps numerical
        // stability identical to the pre-P33 max_logit / sum_exp pair.
        let sigma_scale = (c_visit + max_n as f32) * c_scale;

        // Pass 1: find max_logit over child_data only (illegal slots are -inf
        // which never wins a max).
        let mut max_logit = f32::NEG_INFINITY;
        for &(_, visits, prior, q_val) in &child_data {
            let completed_q = if visits > 0 {
                q_val.clamp(-1.0, 1.0)
            } else {
                v_mix.clamp(-1.0, 1.0)
            };
            let log_prior = (prior.max(1e-8)).ln();
            // §F2: `log_prior + sigma_scale * completed_q` → fused FMA.
            let l = sigma_scale.mul_add(completed_q, log_prior);
            if l > max_logit { max_logit = l; }
        }
        if max_logit == f32::NEG_INFINITY {
            return policy;
        }

        // Pass 2: sum-exp over child_data.
        let mut sum_exp = 0.0f32;
        for &(_, visits, prior, q_val) in &child_data {
            let completed_q = if visits > 0 {
                q_val.clamp(-1.0, 1.0)
            } else {
                v_mix.clamp(-1.0, 1.0)
            };
            let log_prior = (prior.max(1e-8)).ln();
            // §F2: `log_prior + sigma_scale * completed_q` → fused FMA.
            let l = sigma_scale.mul_add(completed_q, log_prior);
            sum_exp += (l - max_logit).exp();
        }
        if sum_exp <= 0.0 {
            return policy;
        }

        // Pass 3: scatter the softmax mass back into policy at each
        // child's flat action index.
        for &(action, visits, prior, q_val) in &child_data {
            let completed_q = if visits > 0 {
                q_val.clamp(-1.0, 1.0)
            } else {
                v_mix.clamp(-1.0, 1.0)
            };
            let log_prior = (prior.max(1e-8)).ln();
            // §F2: `log_prior + sigma_scale * completed_q` → fused FMA.
            let l = sigma_scale.mul_add(completed_q, log_prior);
            policy[action] = (l - max_logit).exp() / sum_exp;
        }

        // Note: policy pruning is applied only in the Python training loop
        // (prune_policy_targets in trainer.py) to avoid double-pruning.

        policy
    }

    /// §D-MULTICLUSTER-S0 legal-set counterpart of `get_policy`. Keys each root
    /// child by board coord into a ragged `LegalSetPolicy` (in-window → `dense`,
    /// covered off-window → `overflow`). Off-window children with NO cluster
    /// coverage are dropped (today's `get_policy` behaviour). §9.5.
    pub fn get_policy_ls(&self, temperature: f32, n_actions: usize) -> LegalSetPolicy {
        let mut dense = vec![0.0f32; n_actions];
        let mut overflow: FxHashMap<(i32, i32), f32> = FxHashMap::default();

        let root = &self.pool[0];
        if !root.is_expanded() {
            return LegalSetPolicy { dense, overflow };
        }
        let first = root.first_child as usize;
        let n_ch = root.n_children as usize;

        // §9.2a coverage geometry (once per move — export is not the per-sim hot path).
        let (_, centers) = self.root_board.get_cluster_views();
        let trunk_sz = self.root_board.cluster_window_size() as i32;
        let half = (trunk_sz - 1) / 2;

        if temperature == 0.0 {
            if let Some(best) = (first..first + n_ch).max_by_key(|&i| self.pool[i].n_visits) {
                let val = self.pool[best].action_idx;
                let q = (val >> 16) as i32 - 32768;
                let r = (val & 0xFFFF) as i32 - 32768;
                let flat = self.root_board.window_flat_idx(q, r);
                if flat < n_actions {
                    dense[flat] = 1.0;
                } else if records::is_covered(q, r, &centers, trunk_sz, half) {
                    overflow.insert((q, r), 1.0);
                }
            }
        } else {
            let visits: Vec<f32> = (first..first + n_ch)
                .map(|i| (self.pool[i].n_visits as f32).powf(1.0 / temperature))
                .collect();
            let total: f32 = visits.iter().sum();
            if total > 0.0 {
                for (j, &v) in visits.iter().enumerate() {
                    let val = self.pool[first + j].action_idx;
                    let q = (val >> 16) as i32 - 32768;
                    let r = (val & 0xFFFF) as i32 - 32768;
                    let flat = self.root_board.window_flat_idx(q, r);
                    if flat < n_actions {
                        dense[flat] = v / total;
                    } else if records::is_covered(q, r, &centers, trunk_sz, half) {
                        overflow.insert((q, r), v / total);
                    }
                }
            }
        }
        LegalSetPolicy { dense, overflow }
    }

    /// §D-MULTICLUSTER-S0 legal-set counterpart of `get_improved_policy`. The
    /// completed-Q softmax math is FROZEN; the differences are (1) off-window
    /// COVERED children are retained (keyed into `overflow`) instead of dropped,
    /// (2) off-window NO-COVERAGE children are dropped (today's `if action >=
    /// n_actions continue`), and (3) the output is the ragged `LegalSetPolicy`.
    /// §9.5 / §9.2a.
    pub fn get_improved_policy_ls(
        &self,
        n_actions: usize,
        c_visit: f32,
        c_scale: f32,
    ) -> LegalSetPolicy {
        let mut dense = vec![0.0f32; n_actions];
        let mut overflow: FxHashMap<(i32, i32), f32> = FxHashMap::default();

        let root = &self.pool[0];
        if !root.is_expanded() {
            return LegalSetPolicy { dense, overflow };
        }
        let first = root.first_child as usize;
        let n_ch = root.n_children as usize;
        let q_sign: f32 = if self.pool[0].moves_remaining == 1 { -1.0 } else { 1.0 };

        let (_, centers) = self.root_board.get_cluster_views();
        let trunk_sz = self.root_board.cluster_window_size() as i32;
        let half = (trunk_sz - 1) / 2;

        // child_data: (q, r, flat, visits, prior, q_val). flat >= n_actions ⇒ off-window.
        let mut child_data: Vec<(i32, i32, usize, u32, f32, f32)> = Vec::with_capacity(n_ch);
        let mut sum_n: u32 = 0;
        let mut max_n: u32 = 0;
        let mut visited_prior_sum: f32 = 0.0;
        let mut policy_weighted_q: f32 = 0.0;

        for j in 0..n_ch {
            let child = &self.pool[first + j];
            let val = child.action_idx;
            let q = (val >> 16) as i32 - 32768;
            let r = (val & 0xFFFF) as i32 - 32768;
            let flat = self.root_board.window_flat_idx(q, r);
            // §9.2a: drop an off-window child only when NO cluster covers it.
            if flat >= n_actions && !records::is_covered(q, r, &centers, trunk_sz, half) {
                continue;
            }

            let visits = child.n_visits;
            let prior = child.prior;
            let q_val = if visits > 0 {
                q_sign * child.w_value / visits as f32
            } else {
                0.0
            };

            sum_n += visits;
            if visits > max_n {
                max_n = visits;
            }
            if visits > 0 {
                visited_prior_sum += prior;
                policy_weighted_q += prior * q_val;
            }

            child_data.push((q, r, flat, visits, prior, q_val));
        }

        // Edge case: no visits — return prior distribution (covered cells included).
        if sum_n == 0 {
            let mut total_prior = 0.0f32;
            for &(q, r, flat, _, prior, _) in &child_data {
                if flat < n_actions {
                    dense[flat] = prior;
                } else {
                    overflow.insert((q, r), prior);
                }
                total_prior += prior;
            }
            if total_prior > 0.0 {
                for p in &mut dense {
                    *p /= total_prior;
                }
                for v in overflow.values_mut() {
                    *v /= total_prior;
                }
            }
            return LegalSetPolicy { dense, overflow };
        }

        let v_hat = root.w_value / root.n_visits as f32;
        let v_mix = if visited_prior_sum > 1e-8 {
            let sum_n_f = sum_n as f32;
            (1.0 / (1.0 + sum_n_f))
                * (sum_n_f / visited_prior_sum).mul_add(policy_weighted_q, v_hat)
        } else {
            v_hat
        };
        let sigma_scale = (c_visit + max_n as f32) * c_scale;

        let mut max_logit = f32::NEG_INFINITY;
        for &(_, _, _, visits, prior, q_val) in &child_data {
            let completed_q = if visits > 0 {
                q_val.clamp(-1.0, 1.0)
            } else {
                v_mix.clamp(-1.0, 1.0)
            };
            let log_prior = (prior.max(1e-8)).ln();
            let l = sigma_scale.mul_add(completed_q, log_prior);
            if l > max_logit {
                max_logit = l;
            }
        }
        if max_logit == f32::NEG_INFINITY {
            return LegalSetPolicy { dense, overflow };
        }

        let mut sum_exp = 0.0f32;
        for &(_, _, _, visits, prior, q_val) in &child_data {
            let completed_q = if visits > 0 {
                q_val.clamp(-1.0, 1.0)
            } else {
                v_mix.clamp(-1.0, 1.0)
            };
            let log_prior = (prior.max(1e-8)).ln();
            let l = sigma_scale.mul_add(completed_q, log_prior);
            sum_exp += (l - max_logit).exp();
        }
        if sum_exp <= 0.0 {
            return LegalSetPolicy { dense, overflow };
        }

        for &(q, r, flat, visits, prior, q_val) in &child_data {
            let completed_q = if visits > 0 {
                q_val.clamp(-1.0, 1.0)
            } else {
                v_mix.clamp(-1.0, 1.0)
            };
            let log_prior = (prior.max(1e-8)).ln();
            let l = sigma_scale.mul_add(completed_q, log_prior);
            let mass = (l - max_logit).exp() / sum_exp;
            if flat < n_actions {
                dense[flat] = mass;
            } else {
                overflow.insert((q, r), mass);
            }
        }

        LegalSetPolicy { dense, overflow }
    }

    /// Returns (child_pool_index, prior) for each root child.
    /// Used by Gumbel MCTS to build the candidate list after root expansion.
    pub fn get_root_children_info(&self) -> Vec<(u32, f32)> {
        let root = &self.pool[0];
        if !root.is_expanded() {
            return Vec::new();
        }
        let first = root.first_child as usize;
        let n_ch = root.n_children as usize;
        (first..first + n_ch)
            .map(|i| (i as u32, self.pool[i].prior))
            .collect()
    }

    pub fn apply_dirichlet_to_root(&mut self, noise: &[f32], epsilon: f32) {
        let root = &self.pool[0];
        if !root.is_expanded() {
            return;
        }
        let n_ch = root.n_children as usize;
        if noise.len() < n_ch {
            return;
        }
        let first = root.first_child as usize;

        // Snapshot pre-priors for the debug_prior_trace feature. Compiled out
        // in default builds; zero cost.
        #[cfg(feature = "debug_prior_trace")]
        let pre_priors: Vec<f32> = (0..n_ch)
            .map(|j| self.pool[first + j].prior)
            .collect();

        for j in 0..n_ch {
            let child = &mut self.pool[first + j];
            // §F2: `(1.0 - epsilon) * prior + epsilon * noise` → fused FMA.
            child.prior = epsilon.mul_add(noise[j], (1.0 - epsilon) * child.prior);
        }

        #[cfg(feature = "debug_prior_trace")]
        {
            let post_priors: Vec<f32> = (0..n_ch)
                .map(|j| self.pool[first + j].prior)
                .collect();
            crate::debug_trace::record_dirichlet(
                epsilon,
                &pre_priors,
                &noise[..n_ch],
                &post_priors,
            );
        }
    }

    /// Top-N children of root by visit count.
    /// Returns Vec<((q, r), visits, prior, q_value)> sorted by visits descending.
    ///
    /// §P34 — returns raw `(i32, i32)` axial coords instead of pre-formatted
    /// `"(q,r)"` strings. Drops the per-child `format!(...)` heap String
    /// allocation; Python callers format with f-strings at the call site
    /// (`hexo_rl/monitoring/analyze_api.py`, `hexo_rl/viewer/engine.py`).
    pub fn get_top_visits(&self, n: usize) -> Vec<((i32, i32), u32, f32, f32)> {
        let root = &self.pool[0];
        if !root.is_expanded() {
            return Vec::new();
        }
        let first = root.first_child as usize;
        let n_ch = root.n_children as usize;

        let mut children: Vec<(usize, u32)> = (first..first + n_ch)
            .map(|i| (i, self.pool[i].n_visits))
            .collect();
        children.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        children.truncate(n);

        let q_sign: f32 = if root.moves_remaining == 1 { -1.0 } else { 1.0 };
        children.into_iter().map(|(i, visits)| {
            let node = &self.pool[i];
            let val = node.action_idx;
            let q = (val >> 16) as i32 - 32768;
            let r = (val & 0xFFFF) as i32 - 32768;
            let q_value = if visits > 0 { q_sign * node.w_value / visits as f32 } else { 0.0 };
            ((q, r), visits, node.prior, q_value)
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::tests::{setup_two_child_tree, setup_expanded_root};
    use crate::board::{Board, BOARD_SIZE};
    use crate::mcts::node::Node;

    #[test]
    fn test_get_policy_proportional_to_visits() {
        let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
        tree.pool[0].n_visits = 10;
        tree.pool[child_a as usize].n_visits = 7;
        tree.pool[child_b as usize].n_visits = 3;

        let policy = tree.get_policy(1.0, BOARD_SIZE * BOARD_SIZE + 1);
        let pa = policy[180];
        let pb = policy[181];
        assert!((pa - 0.7).abs() < 1e-5, "action 0 should get 70%: {pa}");
        assert!((pb - 0.3).abs() < 1e-5, "action 1 should get 30%: {pb}");
        assert!((pa + pb - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_get_policy_argmax_temperature_zero() {
        let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
        tree.pool[0].n_visits = 10;
        tree.pool[child_a as usize].n_visits = 7;
        tree.pool[child_b as usize].n_visits = 3;

        let policy = tree.get_policy(0.0, BOARD_SIZE * BOARD_SIZE + 1);
        assert_eq!(policy[180], 1.0);
        assert_eq!(policy[181], 0.0);
    }

    #[test]
    fn test_policy_sums_to_one_after_search() {
        let mut tree = MCTSTree::new(1.5);
        let board = Board::new();
        tree.new_game(board);

        let n_sims = 10;
        let uniform = vec![1.0 / (BOARD_SIZE * BOARD_SIZE + 1) as f32; BOARD_SIZE * BOARD_SIZE + 1];
        for _ in 0..n_sims {
            let leaves = tree.select_leaves(1);
            let n = leaves.len();
            let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
            let values = vec![0.0f32; n];
            tree.expand_and_backup(&policies, &values);
        }

        let policy = tree.get_policy(1.0, BOARD_SIZE * BOARD_SIZE + 1);
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "policy should sum to 1.0, got {sum}");
    }

    #[test]
    fn test_dirichlet_ignored_before_root_expanded() {
        // Serialize against test_dirichlet_trace_roundtrip (see debug_trace::TEST_TRACE_LOCK).
        #[cfg(feature = "debug_prior_trace")]
        let _guard = crate::debug_trace::TEST_TRACE_LOCK.lock().unwrap();
        let mut tree = MCTSTree::new(1.5);
        tree.new_game(Board::new());
        tree.apply_dirichlet_to_root(&[0.5, 0.5], 0.25);
        assert_eq!(tree.root_n_children(), 0);
    }

    #[test]
    fn test_dirichlet_mixes_priors_correctly() {
        // Serialize against test_dirichlet_trace_roundtrip (see debug_trace::TEST_TRACE_LOCK).
        #[cfg(feature = "debug_prior_trace")]
        let _guard = crate::debug_trace::TEST_TRACE_LOCK.lock().unwrap();
        let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
        assert!(tree.pool[0].is_expanded());

        let noise   = [0.9f32, 0.1f32];
        let epsilon = 0.25f32;
        tree.apply_dirichlet_to_root(&noise, epsilon);

        let expected_a = (1.0 - epsilon) * 0.7 + epsilon * 0.9;
        let expected_b = (1.0 - epsilon) * 0.3 + epsilon * 0.1;

        let prior_a = tree.pool[child_a as usize].prior;
        let prior_b = tree.pool[child_b as usize].prior;

        assert!((prior_a - expected_a).abs() < 1e-6,
            "child_a prior: expected {expected_a:.6}, got {prior_a:.6}");
        assert!((prior_b - expected_b).abs() < 1e-6,
            "child_b prior: expected {expected_b:.6}, got {prior_b:.6}");
    }

    /// Verifies the compile-time-gated JSONL trace wrapper around
    /// `apply_dirichlet_to_root` writes exactly one well-formed record.
    /// Only compiled with `--features debug_prior_trace`.
    #[cfg(feature = "debug_prior_trace")]
    #[test]
    fn test_dirichlet_trace_roundtrip() {
        use std::fs;
        use std::io::{BufRead, BufReader};

        // Serialize against other tests that call apply_dirichlet_to_root —
        // they share the same global TRACE_FILE sink, so parallel execution
        // would cause cross-test writes into our JSONL file.
        let _guard = crate::debug_trace::TEST_TRACE_LOCK.lock().unwrap();

        let path = std::env::temp_dir().join(format!(
            "hexo_dbg_trace_{}.jsonl", std::process::id()
        ));
        let _ = fs::remove_file(&path);
        let path_str = path.to_str().expect("tmp path utf8");

        crate::debug_trace::set_sink_for_test(path_str);
        crate::debug_trace::reset_counters_for_test();

        let (mut tree, _, _) = setup_two_child_tree(1.5);
        assert!(tree.pool[0].is_expanded());
        let noise = [0.9f32, 0.1f32];
        tree.apply_dirichlet_to_root(&noise, 0.25);

        // Tear the sink down BEFORE releasing the guard so no subsequent
        // test (from either this file or another) can write into our tmp.
        crate::debug_trace::clear_sink_for_test();

        let file = fs::File::open(&path).expect("trace file should exist");
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().filter_map(Result::ok).collect();

        assert_eq!(lines.len(), 1, "expected exactly one JSONL record, got {}", lines.len());
        let record = &lines[0];
        assert!(record.contains(r#""site":"apply_dirichlet_to_root""#),
            "record missing site marker: {record}");
        assert!(record.contains(r#""n_children":2"#),
            "record missing n_children=2: {record}");
        assert!(record.contains(r#""epsilon":0.250000"#),
            "record missing epsilon=0.25: {record}");
        assert!(record.contains(r#""noise":[0.900000,0.100000]"#),
            "record missing noise vector: {record}");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_get_root_children_info_returns_correct_count() {
        let tree = setup_expanded_root();
        let info = tree.get_root_children_info();
        assert_eq!(info.len(), tree.root_n_children());
        // All priors should be > 0 (from uniform policy over full action space).
        for &(idx, prior) in &info {
            assert!(prior > 0.0, "prior for child {idx} should be > 0");
        }
        assert!(!info.is_empty(), "root should have children after expansion");
    }

    // ── get_improved_policy tests ────────────────────────────────────────────

    /// Helper: set up a tree with N root children having specified visits, w_values, and priors.
    /// Each child is placed at action (0, j) for j in 0..N.
    fn setup_improved_policy_tree(
        children: &[(u32, f32, f32)], // (visits, w_value, prior)
    ) -> MCTSTree {
        let mut tree = MCTSTree::new(1.5);
        let board = Board::new();
        tree.root_board = board;
        let n = children.len();
        tree.pool[0].first_child = 1;
        tree.pool[0].n_children = n as u16;
        tree.pool[0].moves_remaining = 2;

        let mut total_visits = 0u32;
        let mut total_w = 0.0f32;
        for (j, &(visits, w_value, prior)) in children.iter().enumerate() {
            let q = 0i32;
            let r = j as i32;
            let action_idx = ((q as u32).wrapping_add(32768) << 16) | (r as u32).wrapping_add(32768);
            tree.pool[1 + j] = Node {
                parent: 0,
                action_idx,
                n_visits: visits,
                w_value,
                prior,
                first_child: u32::MAX,
                n_children: 0,
                moves_remaining: 1,
                is_terminal: false,
                terminal_value: 0.0,
                virtual_loss_count: 0,
            };
            total_visits += visits;
            total_w += w_value;
        }
        tree.pool[0].n_visits = total_visits;
        tree.pool[0].w_value = total_w;
        tree.next_free = 1 + n as u32;
        tree
    }

    #[test]
    fn test_improved_policy_sums_to_one() {
        // Three children with different visits and Q values.
        let tree = setup_improved_policy_tree(&[
            (10, 5.0, 0.5),   // Q=0.5
            (8, -2.0, 0.3),   // Q=-0.25
            (2, 0.4, 0.2),    // Q=0.2
        ]);
        let policy = tree.get_improved_policy(BOARD_SIZE * BOARD_SIZE + 1, 50.0, 1.0);
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "policy should sum to 1.0, got {sum}");
    }

    #[test]
    fn test_improved_policy_no_visits_returns_prior() {
        // All children unvisited — should return normalized priors.
        let tree = setup_improved_policy_tree(&[
            (0, 0.0, 0.6),
            (0, 0.0, 0.4),
        ]);
        let policy = tree.get_improved_policy(BOARD_SIZE * BOARD_SIZE + 1, 50.0, 1.0);
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "prior fallback should sum to 1.0, got {sum}");

        // The two non-zero entries should roughly reflect priors.
        let nonzero: Vec<f32> = policy.iter().copied().filter(|&p| p > 0.0).collect();
        assert_eq!(nonzero.len(), 2);
        assert!(nonzero[0] > nonzero[1], "higher prior should get higher prob");
    }

    #[test]
    fn test_improved_policy_q_ordering() {
        // Two children: one clearly winning (Q=+0.9), one losing (Q=-0.9).
        // Equal priors — the improved policy should favor the winning child.
        let tree = setup_improved_policy_tree(&[
            (50, 45.0, 0.5),   // Q=+0.9
            (50, -45.0, 0.5),  // Q=-0.9
        ]);
        let policy = tree.get_improved_policy(BOARD_SIZE * BOARD_SIZE + 1, 50.0, 1.0);

        // Find the two non-zero actions.
        let (cq, cr) = tree.root_board.window_center();
        let idx_good = Board::window_flat_idx_at(0, 0, cq, cr);
        let idx_bad = Board::window_flat_idx_at(0, 1, cq, cr);

        assert!(policy[idx_good] > policy[idx_bad],
            "Q=+0.9 child should get more probability than Q=-0.9: {} vs {}",
            policy[idx_good], policy[idx_bad]);
    }

    // Note: policy_prune_frac test removed — pruning now lives only in
    // Python's prune_policy_targets (trainer.py) to avoid double-pruning.

    #[test]
    fn test_improved_policy_illegal_actions_stay_zero() {
        let tree = setup_improved_policy_tree(&[
            (10, 5.0, 0.7),
            (5, 1.0, 0.3),
        ]);
        let policy = tree.get_improved_policy(BOARD_SIZE * BOARD_SIZE + 1, 50.0, 1.0);

        // Only 2 actions should be non-zero out of board_size*board_size+1.
        let nonzero_count = policy.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(nonzero_count, 2, "only legal actions should have non-zero prob, got {nonzero_count}");
    }
}
