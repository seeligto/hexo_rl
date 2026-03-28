/// Single-threaded PUCT MCTS tree with a flat pre-allocated node pool.
///
/// Design notes:
/// - Virtual loss (Phase 2): when `select_one_leaf` descends through a node
///   it increments `virtual_loss_count`. This decreases the effective Q seen
///   by subsequent selections on the same path, so batched calls to
///   `select_leaves(n)` naturally spread across different branches. The
///   penalty is reversed during `backup`, leaving `w_value` / `n_visits`
///   correct after the full round-trip.
/// - No per-node heap allocation: all nodes live in `pool: Vec<Node>`.
/// - Negamax value convention: `node.w_value` accumulates values from THAT
///   node's player-to-move perspective. Backup flips sign when the player
///   changes (which happens when `parent.moves_remaining == 1`).
/// - PUCT formula (AlphaZero):
///   `Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a))`
///   where Q is from the *parent's* perspective.
///
/// Phase 2 usage (Python side):
/// ```python
/// tree = MCTSTree(c_puct=1.5)
/// tree.new_game(board)
/// # Select a batch of leaves in one shot — virtual loss spreads them:
/// boards = tree.select_leaves(64)
/// policies, values = model(batch_tensor(boards))
/// tree.expand_and_backup(policies, values)
/// move_policy = tree.get_policy(temperature=1.0, board_size=19)
/// ```

use crate::board::{Board, coords, idx as board_idx, BOARD_SIZE};

/// Pre-allocated pool size. 200 k nodes ≈ 6.4 MB.
pub const MAX_NODES: usize = 200_000;

/// Virtual-loss penalty applied per unresolved selection.
/// 1.0 is the standard AlphaZero value (equivalent to one loss).
pub const VIRTUAL_LOSS_PENALTY: f32 = 1.0;

// ── Node ─────────────────────────────────────────────────────────────────────

/// One node in the MCTS tree.
#[derive(Clone, Copy)]
pub struct Node {
    /// Index of this node's parent. `u32::MAX` means "root (no parent)".
    pub parent: u32,
    /// Flat board-cell index of the move that created this node.
    /// `u16::MAX` means "root (no incoming move)".
    pub action_idx: u16,
    /// Visit count N(s,a). Incremented during `backup`, not during selection.
    pub n_visits: u32,
    /// Sum of backed-up values, from *this node's* player-to-move perspective.
    pub w_value: f32,
    /// Prior probability P(s,a) assigned by the network when the parent was expanded.
    pub prior: f32,
    /// Index of the first child in the pool. `u32::MAX` means "not yet expanded".
    pub first_child: u32,
    /// Number of children (= number of legal moves at this node).
    pub n_children: u16,
    /// `moves_remaining` value at this node's board state.
    /// Used to determine whether the player changes when descending to a child:
    /// if `moves_remaining == 1` the player changes after the next move.
    pub moves_remaining: u8,
    /// True when this is a terminal board state (win or draw).
    pub is_terminal: bool,
    /// Terminal outcome from this node's player's perspective. Valid only if `is_terminal`.
    pub terminal_value: f32,
    /// Number of in-flight virtual losses currently applied to this node.
    /// Incremented in `select_one_leaf` when the node is on the selected path;
    /// decremented in `backup` when the real value is propagated back.
    pub virtual_loss_count: u32,
}

impl Node {
    fn uninit() -> Self {
        Node {
            parent: u32::MAX,
            action_idx: u16::MAX,
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

    /// Mean value Q(s,a) from this node's player's perspective, adjusted for
    /// any outstanding virtual losses.
    ///
    /// Effective formula:
    ///   Q_eff = (w_value − vl_count × penalty) / (n_visits + vl_count)
    ///
    /// When `vl_count == 0` and `n_visits == 0` this returns 0.0 (unvisited).
    #[inline]
    pub fn q_value_vl(&self, penalty: f32) -> f32 {
        let effective_n = self.n_visits + self.virtual_loss_count;
        if effective_n == 0 {
            0.0
        } else {
            (self.w_value - self.virtual_loss_count as f32 * penalty) / effective_n as f32
        }
    }

    /// True once `expand_and_backup` has populated children.
    #[inline]
    pub fn is_expanded(&self) -> bool {
        self.first_child != u32::MAX
    }
}

// ── Tree ─────────────────────────────────────────────────────────────────────

pub struct MCTSTree {
    /// Flat node pool. Root is always at index 0.
    pub(crate) pool: Vec<Node>,
    /// Next free slot in the pool.
    pub(crate) next_free: u32,
    /// The board state at the root of the tree (snapshot when `new_game` was called).
    root_board: Board,
    /// PUCT exploration constant.
    pub(crate) c_puct: f32,
    /// Virtual-loss penalty (default 1.0).
    pub(crate) virtual_loss: f32,
    /// Leaves selected by the most recent `select_leaves` call, waiting for
    /// `expand_and_backup`. Contains `(node_index, reconstructed_board)`.
    pending: Vec<(u32, Board)>,
}

impl MCTSTree {
    pub fn new(c_puct: f32) -> Self {
        MCTSTree::new_with_vl(c_puct, VIRTUAL_LOSS_PENALTY)
    }

    pub fn new_with_vl(c_puct: f32, virtual_loss: f32) -> Self {
        let pool = vec![Node::uninit(); MAX_NODES];
        MCTSTree {
            pool,
            next_free: 1,
            root_board: Board::new(),
            c_puct,
            virtual_loss,
            pending: Vec::new(),
        }
    }

    // ── Game lifecycle ────────────────────────────────────────────────────────

    /// Reset the tree for a new game position.
    pub fn new_game(&mut self, board: Board) {
        let mr = board.moves_remaining;
        self.root_board = board;
        self.pool[0] = Node::uninit();
        self.pool[0].moves_remaining = mr;
        self.next_free = 1;
        self.pending.clear();
    }

    // ── Board reconstruction ─────────────────────────────────────────────────

    /// Reconstruct the board at `node_idx` by replaying moves from the root.
    fn reconstruct_board(&self, node_idx: u32) -> Board {
        // Collect path from node to root by following parent pointers.
        let mut path: Vec<u16> = Vec::new();
        let mut cur = node_idx;
        while cur != 0 {
            let node = &self.pool[cur as usize];
            path.push(node.action_idx);
            cur = node.parent;
        }
        // Replay in root-to-leaf order.
        let mut board = self.root_board.clone();
        for &action in path.iter().rev() {
            let (q, r) = coords(action as usize);
            board
                .apply_move(q, r)
                .expect("reconstructed move should always be legal");
        }
        board
    }

    // ── PUCT ─────────────────────────────────────────────────────────────────

    /// PUCT score for `child_idx`, evaluated from `parent_idx`'s player perspective.
    ///
    /// Q is virtual-loss-adjusted. `parent_n` is the effective visit count of the
    /// parent (n_visits + virtual_loss_count), supplied by the caller to avoid
    /// re-reading the parent node.
    ///
    /// If `parent.moves_remaining == 1` the move passes the turn to the other
    /// player, so we negate the child's Q value to get the parent's view.
    #[inline]
    fn puct_score(&self, child_idx: u32, parent_idx: u32, parent_n: f32) -> f32 {
        let child  = &self.pool[child_idx  as usize];
        let parent = &self.pool[parent_idx as usize];

        let q = if parent.moves_remaining == 1 {
            -child.q_value_vl(self.virtual_loss) // turn changed → flip perspective
        } else {
            child.q_value_vl(self.virtual_loss)  // same player's second move in a turn
        };

        let u = self.c_puct * child.prior * parent_n.sqrt()
            / (1.0 + child.n_visits as f32 + child.virtual_loss_count as f32);
        q + u
    }

    // ── Selection ────────────────────────────────────────────────────────────

    /// Walk the tree via PUCT until an unexpanded (or terminal) leaf is found.
    ///
    /// Applies virtual loss to every node on the path (including the leaf).
    /// This causes subsequent calls to `select_one_leaf` to prefer different
    /// branches, enabling effective batched leaf evaluation.
    ///
    /// Returns `(leaf_node_idx, reconstructed_board_at_leaf)`.
    fn select_one_leaf(&mut self) -> (u32, Board) {
        let mut cur: u32 = 0;
        loop {
            // Apply virtual loss to this node before we inspect it.
            self.pool[cur as usize].virtual_loss_count += 1;

            let node = &self.pool[cur as usize];
            if node.is_terminal || !node.is_expanded() {
                return (cur, self.reconstruct_board(cur));
            }

            // Effective parent visit count includes outstanding virtual losses.
            let parent_n = (node.n_visits + node.virtual_loss_count) as f32;
            let first    = node.first_child as usize;
            let n_ch     = node.n_children  as usize;

            let best = (first..first + n_ch)
                .max_by(|&a, &b| {
                    let sa = self.puct_score(a as u32, cur, parent_n);
                    let sb = self.puct_score(b as u32, cur, parent_n);
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap() as u32;

            cur = best;
        }
    }

    /// Select up to `n` distinct leaves for evaluation.
    ///
    /// Virtual loss is applied as each leaf is selected, so subsequent calls
    /// naturally diverge to different branches. Deduplication is retained as a
    /// safety measure (can still occur early before the tree is expanded).
    ///
    /// Caller must call `expand_and_backup` with exactly `boards.len()` entries.
    pub fn select_leaves(&mut self, n: usize) -> Vec<Board> {
        self.pending.clear();
        let mut boards = Vec::with_capacity(n);
        for _ in 0..n {
            let (leaf_idx, board) = self.select_one_leaf();
            // Skip duplicates — dedup is a safety net; VL normally prevents this.
            if self.pending.iter().any(|(i, _)| *i == leaf_idx) {
                // Undo the VL we just applied on this redundant path.
                self.undo_virtual_loss(leaf_idx);
                continue;
            }
            boards.push(board.clone());
            self.pending.push((leaf_idx, board));
        }
        boards
    }

    /// Reverse virtual loss on all nodes from `node_idx` to the root.
    /// Used when a duplicate leaf is detected in `select_leaves`.
    fn undo_virtual_loss(&mut self, mut node_idx: u32) {
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

    // ── Expansion and backup ─────────────────────────────────────────────────

    /// Given network outputs for the last `select_leaves` batch, expand each
    /// leaf and backup values to the root.
    ///
    /// Also reverses the virtual loss applied during selection.
    ///
    /// * `policies[i]`: flat probability vector of length `board_size² + 1`;
    ///   index corresponds to flat board cell index.
    /// * `values[i]`: scalar value in `[-1, 1]` from the *current player's*
    ///   perspective at leaf `i`.
    pub fn expand_and_backup(&mut self, policies: &[Vec<f32>], values: &[f32]) {
        // Take ownership of pending to avoid borrow conflicts during mutation.
        let pending: Vec<(u32, Board)> = std::mem::take(&mut self.pending);
        let n = pending.len().min(policies.len()).min(values.len());

        for i in 0..n {
            let (leaf_idx, ref board) = pending[i];
            let policy = &policies[i];
            let value  = values[i];

            // ── Already marked terminal (e.g. expanded in a prior batch) ──
            if self.pool[leaf_idx as usize].is_terminal {
                let tv = self.pool[leaf_idx as usize].terminal_value;
                self.backup(leaf_idx, tv);
                continue;
            }
            // ── Skip already-expanded nodes (shouldn't happen single-threaded) ──
            if self.pool[leaf_idx as usize].is_expanded() {
                self.backup(leaf_idx, value);
                continue;
            }

            // ── Terminal: win ──
            // Someone just moved and reached this board state. `board.check_win()`
            // is true when the *previous* player won. The current player (to move)
            // is therefore the loser → value = -1 from their perspective.
            if board.check_win() {
                self.pool[leaf_idx as usize].is_terminal    = true;
                self.pool[leaf_idx as usize].terminal_value = -1.0;
                self.backup(leaf_idx, -1.0);
                continue;
            }

            // ── Terminal: draw (board full) ──
            let legal_moves = board.legal_moves();
            if legal_moves.is_empty() {
                self.pool[leaf_idx as usize].is_terminal    = true;
                self.pool[leaf_idx as usize].terminal_value = 0.0;
                self.backup(leaf_idx, 0.0);
                continue;
            }

            // ── Expand: allocate children ──
            let n_ch         = legal_moves.len();
            let first_child  = self.next_free;
            if first_child as usize + n_ch > self.pool.len() {
                // Pool exhausted — graceful degradation: backup without expanding.
                self.backup(leaf_idx, value);
                continue;
            }
            self.next_free += n_ch as u32;

            let leaf_mr      = self.pool[leaf_idx as usize].moves_remaining;
            // After a move: if leaf had 1 move left, player changes and gets 2;
            // otherwise same player still has 1 move left.
            let child_mr: u8 = if leaf_mr == 1 { 2 } else { 1 };

            self.pool[leaf_idx as usize].first_child = first_child;
            self.pool[leaf_idx as usize].n_children  = n_ch as u16;

            for (j, &(q, r)) in legal_moves.iter().enumerate() {
                let ci          = first_child as usize + j;
                let action_flat = board_idx(q, r) as u16;
                // Look up prior from the policy vector; fall back to uniform.
                let prior = if (action_flat as usize) < policy.len() {
                    policy[action_flat as usize]
                } else {
                    1.0 / n_ch as f32
                };
                self.pool[ci] = Node {
                    parent:              leaf_idx,
                    action_idx:          action_flat,
                    n_visits:            0,
                    w_value:             0.0,
                    prior,
                    first_child:         u32::MAX,
                    n_children:          0,
                    moves_remaining:     child_mr,
                    is_terminal:         false,
                    terminal_value:      0.0,
                    virtual_loss_count:  0,
                };
            }

            self.backup(leaf_idx, value);
        }
    }

    // ── Backup ───────────────────────────────────────────────────────────────

    /// Propagate `value` from `node_idx` to the root.
    ///
    /// `value` is from `node_idx`'s player-to-move perspective. It is negated
    /// each time we ascend a level where the player changes (i.e. when the
    /// parent's `moves_remaining` was 1).
    ///
    /// Also reverses one unit of virtual loss at each node on the path.
    fn backup(&mut self, mut node_idx: u32, mut value: f32) {
        loop {
            let node = &mut self.pool[node_idx as usize];
            node.n_visits += 1;
            node.w_value  += value;
            // Reverse the virtual loss applied during selection.
            if node.virtual_loss_count > 0 {
                node.virtual_loss_count -= 1;
            }

            let parent = node.parent;
            if parent == u32::MAX {
                break; // reached root
            }
            // Flip sign when ascending across a turn boundary.
            if self.pool[parent as usize].moves_remaining == 1 {
                value = -value;
            }
            node_idx = parent;
        }
    }

    // ── Policy extraction ─────────────────────────────────────────────────────

    /// Return the policy vector at the root after all simulations.
    ///
    /// With `temperature == 0` this is a one-hot on the most-visited action.
    /// Otherwise proportional to `N(s,a)^(1/temperature)`.
    ///
    /// Output length: `board_size * board_size + 1` (last element = pass move).
    pub fn get_policy(&self, temperature: f32, board_size: usize) -> Vec<f32> {
        let n_actions = board_size * board_size + 1;
        let mut policy = vec![0.0f32; n_actions];

        let root = &self.pool[0];
        if !root.is_expanded() {
            return policy; // no simulations — caller should handle this
        }

        let first = root.first_child as usize;
        let n_ch  = root.n_children  as usize;

        if temperature == 0.0 {
            // Argmax visit count.
            if let Some(best) = (first..first + n_ch)
                .max_by_key(|&i| self.pool[i].n_visits)
            {
                let action = self.pool[best].action_idx as usize;
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
                    let action = self.pool[first + j].action_idx as usize;
                    if action < n_actions {
                        policy[action] = v / total;
                    }
                }
            }
        }

        policy
    }

    /// Total visit count at the root.
    pub fn root_visits(&self) -> u32 {
        self.pool[0].n_visits
    }

    /// Reset the tree without changing the root board (for benchmarking).
    pub fn reset(&mut self) {
        let mr = self.pool[0].moves_remaining;
        let board = self.root_board.clone();
        self.new_game(board);
        self.pool[0].moves_remaining = mr;
    }

    /// Run `n` simulations using uniform priors and a value of 0.0 (no neural
    /// network). Used for CPU-only throughput benchmarking.
    pub fn run_simulations_cpu_only(&mut self, n: usize) {
        let uniform_prior = 1.0 / (BOARD_SIZE * BOARD_SIZE + 1) as f32;
        let uniform_policy = vec![uniform_prior; BOARD_SIZE * BOARD_SIZE + 1];
        for _ in 0..n {
            let boards = self.select_leaves(1);
            if boards.is_empty() {
                continue;
            }
            let policies: Vec<Vec<f32>> = (0..boards.len())
                .map(|_| uniform_policy.clone())
                .collect();
            let values: Vec<f32> = vec![0.0; boards.len()];
            self.expand_and_backup(&policies, &values);
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    /// Build a tiny manual tree:
    ///   root  (mr=2, n=0)
    ///     ├─ child_a (action=0, prior=0.7, n=0)
    ///     └─ child_b (action=1, prior=0.3, n=0)
    fn setup_two_child_tree(c_puct: f32) -> (MCTSTree, u32, u32) {
        let mut tree = MCTSTree::new(c_puct);
        // Override root to have mr=2 so children have same player
        tree.pool[0].moves_remaining = 2;

        // Manually allocate two children
        let first_child = 1u32;
        tree.next_free = 3;
        tree.pool[0].first_child = first_child;
        tree.pool[0].n_children  = 2;

        tree.pool[1] = Node {
            parent: 0, action_idx: 0, n_visits: 0, w_value: 0.0,
            prior: 0.7, first_child: u32::MAX, n_children: 0,
            moves_remaining: 1, is_terminal: false, terminal_value: 0.0,
            virtual_loss_count: 0,
        };
        tree.pool[2] = Node {
            parent: 0, action_idx: 1, n_visits: 0, w_value: 0.0,
            prior: 0.3, first_child: u32::MAX, n_children: 0,
            moves_remaining: 1, is_terminal: false, terminal_value: 0.0,
            virtual_loss_count: 0,
        };
        (tree, 1, 2)
    }

    #[test]
    fn test_puct_prefers_higher_prior_when_unvisited() {
        // Both children unvisited → U term dominates → higher prior wins.
        let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
        tree.pool[0].n_visits = 1; // root visited once (needed for sqrt(N))
        let score_a = tree.puct_score(child_a, 0, 1.0);
        let score_b = tree.puct_score(child_b, 0, 1.0);
        assert!(
            score_a > score_b,
            "child with prior 0.7 should score higher than 0.3: {score_a:.4} vs {score_b:.4}"
        );
    }

    #[test]
    fn test_puct_visits_reduce_exploration() {
        // After many visits to child_a, child_b (lower prior) should overtake.
        let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
        tree.pool[0].n_visits = 100;
        // Give child_a 99 visits with neutral Q
        tree.pool[child_a as usize].n_visits = 99;
        tree.pool[child_a as usize].w_value  = 0.0; // Q = 0

        let score_a = tree.puct_score(child_a, 0, 100.0);
        let score_b = tree.puct_score(child_b, 0, 100.0);
        assert!(
            score_b > score_a,
            "less-visited child_b should be preferred after child_a is over-visited: \
             {score_b:.4} vs {score_a:.4}"
        );
    }

    #[test]
    fn test_backup_single_value_reaches_root() {
        let mut tree = MCTSTree::new(1.5);
        // Manually create root → child → grandchild chain.
        // root: mr=1 (will flip at this level)
        tree.pool[0].moves_remaining = 1;

        // child at index 1: mr=2, parent=root
        tree.pool[1] = Node {
            parent: 0, action_idx: 0, n_visits: 0, w_value: 0.0,
            prior: 1.0, first_child: u32::MAX, n_children: 0,
            moves_remaining: 2, is_terminal: false, terminal_value: 0.0,
            virtual_loss_count: 0,
        };
        // grandchild at index 2: mr=1, parent=child
        tree.pool[2] = Node {
            parent: 1, action_idx: 1, n_visits: 0, w_value: 0.0,
            prior: 1.0, first_child: u32::MAX, n_children: 0,
            moves_remaining: 1, is_terminal: false, terminal_value: 0.0,
            virtual_loss_count: 0,
        };
        tree.next_free = 3;

        // Backup +1.0 from grandchild (grandchild's player).
        tree.backup(2, 1.0);

        // grandchild: w = +1.0 (its own player)
        assert_eq!(tree.pool[2].w_value, 1.0);
        assert_eq!(tree.pool[2].n_visits, 1);

        // child: parent of grandchild has mr=2 → no flip → w = +1.0
        // (child has same player as grandchild because child.mr=2)
        assert_eq!(tree.pool[1].w_value, 1.0, "child should not flip (mr=2)");
        assert_eq!(tree.pool[1].n_visits, 1);

        // root: parent of child has mr=1 → flip → w = -1.0
        assert_eq!(tree.pool[0].w_value, -1.0, "root should flip (mr=1)");
        assert_eq!(tree.pool[0].n_visits, 1);
    }

    #[test]
    fn test_backup_negamax_player_change() {
        // Simple root → child where root has mr=1 (player changes at child).
        let mut tree = MCTSTree::new(1.5);
        tree.pool[0].moves_remaining = 1; // turn passes after root's move

        tree.pool[1] = Node {
            parent: 0, action_idx: 0, n_visits: 0, w_value: 0.0,
            prior: 1.0, first_child: u32::MAX, n_children: 0,
            moves_remaining: 2, is_terminal: false, terminal_value: 0.0,
            virtual_loss_count: 0,
        };
        tree.next_free = 2;

        // Value = +0.6 from child's player perspective (child's player is winning).
        tree.backup(1, 0.6);

        assert!((tree.pool[1].w_value - 0.6).abs() < 1e-6, "child w = 0.6");
        // Root sees this as -0.6 (child's player winning = root's player losing)
        assert!(
            (tree.pool[0].w_value - (-0.6)).abs() < 1e-6,
            "root w should be -0.6, got {}",
            tree.pool[0].w_value
        );
    }

    #[test]
    fn test_get_policy_proportional_to_visits() {
        let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
        tree.pool[0].n_visits = 10;
        tree.pool[child_a as usize].n_visits = 7;
        tree.pool[child_b as usize].n_visits = 3;

        let policy = tree.get_policy(1.0, BOARD_SIZE);
        let pa = policy[0]; // action_idx=0
        let pb = policy[1]; // action_idx=1

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

        let policy = tree.get_policy(0.0, BOARD_SIZE);
        assert_eq!(policy[0], 1.0); // child_a has more visits
        assert_eq!(policy[1], 0.0);
    }

    #[test]
    fn test_select_leaves_returns_root_when_empty() {
        let mut tree = MCTSTree::new(1.5);
        let board = Board::new();
        tree.new_game(board.clone());

        let leaves = tree.select_leaves(1);
        assert_eq!(leaves.len(), 1);
        // The returned board should equal the root board.
        assert_eq!(leaves[0].ply, board.ply);
        assert_eq!(leaves[0].moves_remaining, board.moves_remaining);
    }

    #[test]
    fn test_expand_and_backup_creates_children() {
        let mut tree = MCTSTree::new(1.5);
        let board = Board::new();
        tree.new_game(board);

        let leaves = tree.select_leaves(1);
        let n_legal = leaves[0].legal_move_count();

        // Feed a uniform prior and neutral value.
        let policy = vec![1.0 / (BOARD_SIZE * BOARD_SIZE + 1) as f32; BOARD_SIZE * BOARD_SIZE + 1];
        tree.expand_and_backup(&[policy], &[0.0]);

        let root = &tree.pool[0];
        assert!(root.is_expanded(), "root should be expanded after first backup");
        assert_eq!(root.n_children as usize, n_legal, "should have one child per legal move");
        assert_eq!(root.n_visits, 1);
    }

    #[test]
    fn test_full_search_runs_n_simulations() {
        // Run a small number of MCTS simulations on a real board and
        // verify root visit count matches.
        let mut tree = MCTSTree::new(1.5);
        let board = Board::new();
        tree.new_game(board);

        let n_sims = 20;
        let uniform = vec![1.0 / (BOARD_SIZE * BOARD_SIZE + 1) as f32; BOARD_SIZE * BOARD_SIZE + 1];

        for _ in 0..n_sims {
            let leaves = tree.select_leaves(1);
            let n = leaves.len();
            let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
            let values: Vec<f32> = (0..n).map(|_| 0.0).collect();
            tree.expand_and_backup(&policies, &values);
        }

        assert_eq!(
            tree.root_visits(), n_sims as u32,
            "root visit count should equal number of simulations"
        );
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

        let policy = tree.get_policy(1.0, BOARD_SIZE);
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "policy should sum to 1.0, got {sum}");
    }

    // ── Virtual loss tests ────────────────────────────────────────────────────

    #[test]
    fn test_virtual_loss_applied_during_select() {
        // After select_leaves(1), the nodes on the selected path (just root here,
        // since root is unexpanded) should have virtual_loss_count > 0 UNTIL backup.
        let mut tree = MCTSTree::new(1.5);
        tree.new_game(Board::new());

        // Root is not yet expanded — select will return root as leaf.
        let _leaves = tree.select_leaves(1);

        // VL should be on root now (backup hasn't happened yet).
        assert_eq!(
            tree.pool[0].virtual_loss_count, 1,
            "root should have virtual_loss_count=1 after select before backup"
        );
    }

    #[test]
    fn test_virtual_loss_reversed_after_backup() {
        // After a full select → expand_and_backup round-trip, VL should be 0.
        let mut tree = MCTSTree::new(1.5);
        tree.new_game(Board::new());

        let leaves = tree.select_leaves(1);
        let n = leaves.len();
        let uniform = vec![1.0 / (BOARD_SIZE * BOARD_SIZE + 1) as f32; BOARD_SIZE * BOARD_SIZE + 1];
        let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
        tree.expand_and_backup(&policies, &vec![0.0; n]);

        // All nodes should have VL cleared.
        for i in 0..tree.next_free as usize {
            assert_eq!(
                tree.pool[i].virtual_loss_count, 0,
                "node {i} should have virtual_loss_count=0 after backup"
            );
        }
    }

    #[test]
    fn test_virtual_loss_causes_path_divergence() {
        // With VL, selecting the same expanded tree twice should visit DIFFERENT
        // children, not the same one twice.
        //
        // We use setup_two_child_tree (priors 0.7 and 0.3). Without VL, both
        // selects would always return child_a (prior dominates at N=0). With VL,
        // child_a gets a -1.0 Q penalty after the first select, causing the second
        // select to prefer child_b.
        //
        // Trace (c_puct=1.5, VL penalty=1.0):
        //   Select 1: root.vl=1, parent_n=2, PUCT(A)=+1.485, PUCT(B)=+0.636 → A
        //   Select 2: root.vl=2, parent_n=3, PUCT(A)=-0.091, PUCT(B)=+0.779 → B
        let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
        // Root needs at least 1 real visit so sqrt(parent_n) is meaningful.
        tree.pool[0].n_visits = 1;

        let batch = tree.select_leaves(2);
        assert_eq!(batch.len(), 2, "VL should cause selections to diverge to 2 distinct leaves");

        // Backup both with 0.0 to reverse virtual losses.
        let dummy = vec![0.5f32; BOARD_SIZE * BOARD_SIZE + 1];
        let policies: Vec<Vec<f32>> = (0..2).map(|_| dummy.clone()).collect();
        tree.expand_and_backup(&policies, &vec![0.0; 2]);

        // All VL counts should be zero after backup.
        assert_eq!(tree.pool[0].virtual_loss_count, 0, "root VL not reversed");
        assert_eq!(tree.pool[child_a as usize].virtual_loss_count, 0, "child_a VL not reversed");
        assert_eq!(tree.pool[child_b as usize].virtual_loss_count, 0, "child_b VL not reversed");
    }

    #[test]
    fn test_virtual_loss_q_adjustment() {
        // Verify that q_value_vl correctly penalises a node with outstanding VL.
        let node = Node {
            parent: u32::MAX, action_idx: u16::MAX,
            n_visits: 4, w_value: 2.0, // Q = 0.5 without VL
            prior: 0.5, first_child: u32::MAX, n_children: 0,
            moves_remaining: 1, is_terminal: false, terminal_value: 0.0,
            virtual_loss_count: 2,
        };
        // Q_eff = (2.0 - 2*1.0) / (4 + 2) = 0.0 / 6 = 0.0
        let q = node.q_value_vl(VIRTUAL_LOSS_PENALTY);
        assert!(
            q.abs() < 1e-6,
            "Q should be 0.0 with 2 VL on node with w=2.0, n=4: got {q}"
        );
    }
}
