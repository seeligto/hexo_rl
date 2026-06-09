/// Node, TTEntry, and pool constants for the MCTS tree.

use std::sync::Arc;
use crate::game_runner::records::LegalSetPolicy;

/// Pre-allocated pool size per worker. 1 M nodes ~ 32 MB; 24 workers = 768 MB total.
pub const MAX_NODES: usize = 1_000_000;

/// Virtual-loss penalty applied per unresolved selection.
pub const VIRTUAL_LOSS_PENALTY: f32 = 1.0;

/// Cached MCTS prior for a board state — either the dense scatter_max policy
/// (the existing path, byte-identical) or the §D-MULTICLUSTER-S0 ragged
/// legal-set policy. One run uses exactly one variant (selected by the
/// encoding's `policy_pool`), so the TT never mixes them within a tree.
#[derive(Clone)]
pub enum CachedPolicy {
    Dense(Arc<Vec<f32>>),
    Ls(Arc<LegalSetPolicy>),
}

/// Cached Neural Network evaluation for a board state.
///
/// §P7: the policy is wrapped in `Arc` so TT-hit reads in `select_leaves` are
/// refcount bumps instead of a 1448 B (362 floats × 4 B) Vec clone per hit.
/// Insertion still allocates a fresh `Arc` once per first-touch.
#[derive(Clone)]
pub struct TTEntry {
    pub policy: CachedPolicy,
    pub value: f32,
}

/// One node in the MCTS tree.
#[derive(Clone, Copy)]
pub struct Node {
    pub parent: u32,
    pub action_idx: u32,
    pub n_visits: u32,
    pub w_value: f32,
    pub prior: f32,
    pub first_child: u32,
    pub n_children: u16,
    pub moves_remaining: u8,
    pub is_terminal: bool,
    pub terminal_value: f32,
    pub virtual_loss_count: u32,
}

impl Node {
    pub fn uninit() -> Self {
        Node {
            parent: u32::MAX,
            action_idx: u32::MAX,
            n_visits: 0,
            w_value: 0.0,
            prior: 0.0,
            first_child: u32::MAX,
            n_children: 0,
            moves_remaining: 1,
            is_terminal: false,
            terminal_value: 0.0,
            virtual_loss_count: 0,
        }
    }

    /// Mean value Q(s,a) adjusted for outstanding virtual losses.
    #[inline]
    pub fn q_value_vl(&self, penalty: f32) -> f32 {
        let effective_n = self.n_visits + self.virtual_loss_count;
        if effective_n == 0 {
            0.0
        } else {
            let total_penalty = self.virtual_loss_count as f32 * penalty;
            (self.w_value - total_penalty) / effective_n as f32
        }
    }

    #[inline]
    pub fn is_expanded(&self) -> bool {
        self.first_child != u32::MAX
    }
}
