use std::collections::{VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use pyo3::prelude::*;
use crate::board::{Board, BOARD_SIZE, HALF};
use crate::mcts::MCTSTree;
use crate::inference_bridge::InferenceBatcher;
use rand::prelude::IndexedRandom;
use rand::rng;

/// One recorded position from a self-play game.
#[derive(Clone)]
pub struct Position {
    pub features: Vec<f32>,
    pub policy: Vec<f32>,
    pub player: i8,
}

#[pyclass(name = "SelfPlayRunner")]
pub struct SelfPlayRunner {
    batcher: InferenceBatcher,
    n_workers: usize,
    max_moves_per_game: usize,
    leaf_batch_size: usize,
    c_puct: f32,
    fast_prob: f32,
    fast_sims: usize,
    standard_sims: usize,
    temp_threshold_compound_moves: usize,
    running: Arc<AtomicBool>,
    games_completed: Arc<AtomicUsize>,
    positions_generated: Arc<AtomicUsize>,
    x_wins: Arc<AtomicU64>,
    o_wins: Arc<AtomicU64>,
    draws: Arc<AtomicU64>,
    results: Arc<Mutex<VecDeque<(Vec<f32>, Vec<f32>, f32)>>>,
    /// Ring-buffer of recent (plies, winner_code) pairs for Python logging.
    /// winner_code: 1 = Player One, 2 = Player Two, 0 = draw.
    recent_game_results: Arc<Mutex<VecDeque<(usize, u8)>>>,
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

#[pymethods]
impl SelfPlayRunner {
    #[new]
    #[pyo3(signature = (n_workers = 4, max_moves_per_game = 128, n_simulations = 50, leaf_batch_size = 8, c_puct = 1.5, feature_len = 18 * 19 * 19, policy_len = 19 * 19 + 1, fast_prob = 0.0, fast_sims = 50, standard_sims = 0, temp_threshold_compound_moves = 15))]
    pub fn new(
        n_workers: usize,
        max_moves_per_game: usize,
        n_simulations: usize,
        leaf_batch_size: usize,
        c_puct: f32,
        feature_len: usize,
        policy_len: usize,
        fast_prob: f32,
        fast_sims: usize,
        standard_sims: usize,
        temp_threshold_compound_moves: usize,
    ) -> Self {
        // If standard_sims is 0, fall back to n_simulations.
        let effective_standard = if standard_sims == 0 { n_simulations } else { standard_sims };
        Self {
            batcher: InferenceBatcher::new(feature_len, policy_len),
            n_workers,
            max_moves_per_game,
            leaf_batch_size,
            c_puct,
            fast_prob,
            fast_sims,
            standard_sims: effective_standard,
            temp_threshold_compound_moves,
            running: Arc::new(AtomicBool::new(false)),
            games_completed: Arc::new(AtomicUsize::new(0)),
            positions_generated: Arc::new(AtomicUsize::new(0)),
            x_wins: Arc::new(AtomicU64::new(0)),
            o_wins: Arc::new(AtomicU64::new(0)),
            draws: Arc::new(AtomicU64::new(0)),
            results: Arc::new(Mutex::new(VecDeque::new())),
            recent_game_results: Arc::new(Mutex::new(VecDeque::new())),
            handles: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }

        let mut handles = self.handles.lock().expect("runner handles lock poisoned");
        for _ in 0..self.n_workers {
            let running = self.running.clone();
            let games_completed = self.games_completed.clone();
            let positions_generated = self.positions_generated.clone();
            let x_wins = self.x_wins.clone();
            let o_wins = self.o_wins.clone();
            let draws = self.draws.clone();
            let batcher = self.batcher.clone();
            let max_moves = self.max_moves_per_game;
            let leaf_batch_size = self.leaf_batch_size;
            let c_puct = self.c_puct;
            let fast_prob = self.fast_prob;
            let fast_sims = self.fast_sims;
            let standard_sims = self.standard_sims;
            let temp_threshold = self.temp_threshold_compound_moves;
            let results_queue = self.results.clone();
            let recent_game_results = self.recent_game_results.clone();

            let handle = thread::spawn(move || {
                let mut tree = MCTSTree::new(c_puct);
                let mut rng = rng();

                while running.load(Ordering::SeqCst) {
                    let mut board = Board::new();
                    let mut records = Vec::new();

                    // KataGo-style playout cap randomisation.
                    let is_fast_game = fast_prob > 0.0 && rand::Rng::random::<f32>(&mut rng) < fast_prob;
                    let game_sims = if is_fast_game { fast_sims } else { standard_sims };

                    for _ in 0..max_moves {
                        if !running.load(Ordering::SeqCst) || board.check_win() || board.legal_move_count() == 0 {
                            break;
                        }

                        // ── MCTS Search ──
                        tree.new_game(board.clone());
                        let mut sims_done = 0;

                        while sims_done < game_sims {
                            if !running.load(Ordering::SeqCst) { break; }
                            
                            let leaves = tree.select_leaves(leaf_batch_size);
                            if leaves.is_empty() { break; }
                            
                            let mut all_batch_features = Vec::new();
                            let mut leaf_metadata = Vec::with_capacity(leaves.len());
                            
                            for leaf in &leaves {
                                let (views, centers) = leaf.get_cluster_views();
                                let k = views.len();
                                leaf_metadata.push((k, centers));

                                for view in views {
                                    let mut buffer = batcher.get_feature_buffer();
                                    // get_cluster_views returns 2-plane views (722 floats).
                                    // Expand to 18 planes in-place for the network;
                                    // history planes 1-7 and 9-15 are zeros (Rust has no
                                    // Python move_history — see get_cluster_views doc).
                                    leaf.encode_18_planes_to_buffer(&view, &mut buffer);
                                    all_batch_features.push(buffer);
                                }
                            }
                            if all_batch_features.is_empty() { break; }
                            
                            let total_clusters: usize = leaf_metadata.iter().map(|(k, _)| *k).sum();
                            
                            // Batch inference via bridge - submit all clusters from all leaves
                            let (all_policies, all_values) = match batcher.submit_batch_and_wait_rust(all_batch_features) {
                                Ok(results) => {
                                    let mut ps = Vec::with_capacity(results.len());
                                    let mut vs = Vec::with_capacity(results.len());
                                    for (p, v) in results {
                                        ps.push(p);
                                        vs.push(v);
                                    }
                                    (ps, vs)
                                }
                                Err(_) => break,
                            };
                            
                            if all_policies.len() < total_clusters { break; }
                            
                            // Aggregate cluster-based policies and values back into one per leaf
                            let mut aggregated_policies = Vec::with_capacity(leaves.len());
                            let mut aggregated_values = Vec::with_capacity(leaves.len());
                            let mut curr = 0;
                            
                            for (i, (k, centers)) in leaf_metadata.iter().enumerate() {
                                let leaf_policies = &all_policies[curr..curr+k];
                                let leaf_values = &all_values[curr..curr+k];
                                curr += k;
                                
                                // Pessimistic value aggregation: use the minimum value among clusters
                                let mut min_v = leaf_values[0];
                                for &v in leaf_values {
                                    if v < min_v { min_v = v; }
                                }
                                aggregated_values.push(min_v);
                                
                                // Map local cluster policies to global policy vector
                                aggregated_policies.push(Self::aggregate_policy(&leaves[i], centers, leaf_policies));
                            }
                            
                            tree.expand_and_backup(&aggregated_policies, &aggregated_values);
                            sims_done += leaves.len();
                        }

                        if !running.load(Ordering::SeqCst) { break; }

                        // ── MCTS Policy with temperature schedule ──
                        let compound_move = if board.ply == 0 { 0 } else { (board.ply as usize + 1) / 2 };
                        let temperature = if is_fast_game {
                            1.0  // fast games: always exploratory
                        } else if compound_move < temp_threshold {
                            1.0
                        } else {
                            0.0  // argmax after threshold
                        };
                        let policy = tree.get_policy(temperature, BOARD_SIZE);

                        // ── Sample and apply move ──
                        let legal = board.legal_moves();
                        if legal.is_empty() { break; }

                        // Sample from policy
                        let move_idx = match Self::sample_policy(&policy, &legal, &board) {
                            Some(idx) => idx,
                            None => *legal.choose(&mut rng).unwrap(),
                        };
                        
                        // ── Record position ──
                        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
                        let (views, centers) = board.get_cluster_views();
                        for (k, center) in centers.iter().enumerate() {
                            let mut feat = batcher.get_feature_buffer();
                            // get_cluster_views returns 2-plane views; expand to 18 for storage.
                            board.encode_18_planes_to_buffer(&views[k], &mut feat);
                            // Fast games: zero-policy marks value-only targets.
                            let projected_policy = if is_fast_game {
                                vec![0.0; n_actions]
                            } else {
                                Self::aggregate_policy_to_local(&board, center, &policy)
                            };
                            records.push((feat, projected_policy, board.current_player));
                        }

                        if board.apply_move(move_idx.0, move_idx.1).is_err() {
                            break;
                        }
                        positions_generated.fetch_add(1, Ordering::SeqCst);
                    }

                    // ── Game End: determine outcome ──
                    let winner = board.winner();
                    let plies = board.ply as usize;
                    let winner_code: u8 = match winner {
                        Some(crate::board::Player::One) => 1,
                        Some(_)                         => 2,
                        None                            => 0,
                    };
                    let mut games_results = results_queue.lock().expect("results lock poisoned");
                    for (feat, pol, player) in records {
                        let outcome = match winner {
                            Some(p) => if p as i8 == player as i8 { 1.0 } else { -1.0 },
                            None => 0.0,
                        };
                        games_results.push_back((feat, pol, outcome));
                    }
                    games_completed.fetch_add(1, Ordering::Relaxed);
                    match winner {
                        Some(crate::board::Player::One) => { x_wins.fetch_add(1, Ordering::Relaxed); }
                        Some(_)                         => { o_wins.fetch_add(1, Ordering::Relaxed); }
                        None                            => { draws.fetch_add(1, Ordering::Relaxed); }
                    }
                    // Record (plies, winner_code) for Python game_complete logging.
                    {
                        let mut rg = recent_game_results.lock().expect("recent_game_results lock poisoned");
                        rg.push_back((plies, winner_code));
                        // Cap at 2000 entries to avoid unbounded growth if Python is slow.
                        if rg.len() > 2000 {
                            rg.pop_front();
                        }
                    }

                    // Cap the results queue to avoid memory explosion if Python is slow
                    if games_results.len() > 10000 {
                        let to_drop = games_results.len() - 10000;
                        for _ in 0..to_drop {
                            games_results.pop_front();
                        }
                    }
                }
            });
            handles.push(handle);
        }
    }

    pub fn collect_data(&self) -> Vec<(Vec<f32>, Vec<f32>, f32)> {
        let mut results = self.results.lock().expect("results lock poisoned");
        let mut out = Vec::with_capacity(results.len());
        while let Some(data) = results.pop_front() {
            out.push(data);
        }
        out
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.batcher.close_rust();

        let mut handles = self.handles.lock().expect("runner handles lock poisoned");
        while let Some(handle) = handles.pop() {
            let _ = handle.join();
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    #[getter]
    pub fn batcher(&self) -> InferenceBatcher {
        self.batcher.clone()
    }

    #[getter]
    pub fn games_completed(&self) -> usize {
        self.games_completed.load(Ordering::Relaxed)
    }

    #[getter]
    pub fn positions_generated(&self) -> usize {
        self.positions_generated.load(Ordering::Relaxed)
    }

    #[getter]
    pub fn x_wins(&self) -> u64 {
        self.x_wins.load(Ordering::Relaxed)
    }

    #[getter]
    pub fn o_wins(&self) -> u64 {
        self.o_wins.load(Ordering::Relaxed)
    }

    #[getter]
    pub fn draws(&self) -> u64 {
        self.draws.load(Ordering::Relaxed)
    }

    pub fn get_win_stats(&self) -> (u64, u64, u64) {
        (
            self.x_wins.load(Ordering::Relaxed),
            self.o_wins.load(Ordering::Relaxed),
            self.draws.load(Ordering::Relaxed),
        )
    }

    /// Drain and return all buffered (plies, winner_code) pairs since the last call.
    /// winner_code: 1 = Player One, 2 = Player Two, 0 = draw.
    /// Called by Python pool._stats_loop() to emit game_complete structlog events.
    pub fn drain_game_results(&self) -> Vec<(usize, u8)> {
        let mut rg = self.recent_game_results.lock().expect("recent_game_results lock poisoned");
        let out: Vec<(usize, u8)> = rg.drain(..).collect();
        out
    }
}

impl SelfPlayRunner {
    fn aggregate_policy(board: &Board, centers: &[(i32, i32)], cluster_policies: &[Vec<f32>]) -> Vec<f32> {
        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let mut global_policy = vec![0.0; n_actions];
        let legal = board.legal_moves();
        
        for &(q, r) in &legal {
            let mcts_idx = board.window_flat_idx(q, r);
            if mcts_idx >= n_actions - 1 { continue; }
            
            let mut max_prob = 0.0;
            for (k, &(cq, cr)) in centers.iter().enumerate() {
                let wq = q - cq + HALF;
                let wr = r - cr + HALF;
                if wq >= 0 && wq < BOARD_SIZE as i32 && wr >= 0 && wr < BOARD_SIZE as i32 {
                    let local_idx = wq as usize * BOARD_SIZE + wr as usize;
                    if cluster_policies[k][local_idx] > max_prob {
                        max_prob = cluster_policies[k][local_idx];
                    }
                }
            }
            global_policy[mcts_idx] = max_prob;
        }
        
        // Pass move is always copied from the first cluster (should be consistent)
        if !cluster_policies.is_empty() {
            global_policy[n_actions - 1] = cluster_policies[0][n_actions - 1];
        }
        
        let sum: f32 = global_policy.iter().sum();
        if sum > 1e-9 {
            for p in &mut global_policy { *p /= sum; }
        } else {
            let uniform = 1.0 / n_actions as f32;
            for p in &mut global_policy { *p = uniform; }
        }
        global_policy
    }

    fn aggregate_policy_to_local(board: &Board, center: &(i32, i32), global_policy: &[f32]) -> Vec<f32> {
        let (cq, cr) = *center;
        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let mut local_policy = vec![0.0; n_actions];
        let legal = board.legal_moves();
        
        for &(q, r) in &legal {
            let wq = q - cq + HALF;
            let wr = r - cr + HALF;
            if wq >= 0 && wq < BOARD_SIZE as i32 && wr >= 0 && wr < BOARD_SIZE as i32 {
                let local_idx = wq as usize * BOARD_SIZE + wr as usize;
                let mcts_idx = board.window_flat_idx(q, r);
                if mcts_idx < global_policy.len() {
                    local_policy[local_idx] = global_policy[mcts_idx];
                }
            }
        }
        
        // Pass move (the last element) is always copied from the global policy
        if n_actions > 0 && global_policy.len() >= n_actions {
            local_policy[n_actions - 1] = global_policy[n_actions - 1];
        }
        
        let sum: f32 = local_policy.iter().sum();
        if sum > 1e-9 {
            for p in &mut local_policy { *p /= sum; }
        } else {
            let uniform = 1.0 / n_actions as f32;
            for p in &mut local_policy { *p = uniform; }
        }
        local_policy
    }

    fn sample_policy(policy: &[f32], legal_moves: &[(i32, i32)], board: &Board) -> Option<(i32, i32)> {
        let mut probs = Vec::with_capacity(legal_moves.len());
        let mut sum = 0.0;
        for &(q, r) in legal_moves {
            let idx = board.window_flat_idx(q, r);
            let p = if idx < policy.len() { policy[idx] } else { 0.0 };
            probs.push(p);
            sum += p;
        }

        if sum < 1e-9 {
            return None;
        }

        let mut rng = rng();
        let mut r: f32 = rand::Rng::random(&mut rng);
        r *= sum;

        let mut current = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            current += p;
            if r <= current {
                return Some(legal_moves[i]);
            }
        }
        Some(legal_moves[legal_moves.len() - 1])
    }
}

impl Drop for SelfPlayRunner {
    fn drop(&mut self) {
        self.stop();
    }
}
