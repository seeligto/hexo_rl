use std::collections::{VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use pyo3::prelude::*;
use crate::board::{Board, Player, BOARD_SIZE, TOTAL_CELLS};
use crate::mcts::MCTSTree;
use crate::inference_bridge::RustInferenceBatcher;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// One recorded position from a self-play game.
#[derive(Clone)]
pub struct Position {
    pub features: Vec<f32>,
    pub policy: Vec<f32>,
    pub player: i8,
}

#[pyclass(name = "RustSelfPlayRunner")]
pub struct RustSelfPlayRunner {
    batcher: RustInferenceBatcher,
    n_workers: usize,
    max_moves_per_game: usize,
    n_simulations: usize,
    c_puct: f32,
    running: Arc<AtomicBool>,
    games_completed: Arc<AtomicUsize>,
    positions_generated: Arc<AtomicUsize>,
    results: Arc<Mutex<VecDeque<(Vec<f32>, Vec<f32>, f32)>>>,
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

#[pymethods]
impl RustSelfPlayRunner {
    #[new]
    #[pyo3(signature = (n_workers = 4, max_moves_per_game = 128, n_simulations = 50, c_puct = 1.5, feature_len = 18 * 19 * 19, policy_len = 19 * 19 + 1))]
    pub fn new(
        n_workers: usize,
        max_moves_per_game: usize,
        n_simulations: usize,
        c_puct: f32,
        feature_len: usize,
        policy_len: usize,
    ) -> Self {
        Self {
            batcher: RustInferenceBatcher::new(feature_len, policy_len),
            n_workers,
            max_moves_per_game,
            n_simulations,
            c_puct,
            running: Arc::new(AtomicBool::new(false)),
            games_completed: Arc::new(AtomicUsize::new(0)),
            positions_generated: Arc::new(AtomicUsize::new(0)),
            results: Arc::new(Mutex::new(VecDeque::new())),
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
            let batcher = self.batcher.clone();
            let max_moves = self.max_moves_per_game;
            let n_sims = self.n_simulations;
            let c_puct = self.c_puct;
            let results_queue = self.results.clone();

            let handle = thread::spawn(move || {
                let mut tree = MCTSTree::new(c_puct);
                
                while running.load(Ordering::SeqCst) {
                    let mut board = Board::new();
                    let mut records = Vec::new();
                    
                    for _ in 0..max_moves {
                        if !running.load(Ordering::SeqCst) || board.check_win() || board.legal_move_count() == 0 {
                            break;
                        }

                        // ── MCTS Search ──
                        tree.new_game(board.clone());
                        let mut sims_done = 0;
                        let batch_size = 8;
                        
                        while sims_done < n_sims {
                            if !running.load(Ordering::SeqCst) { break; }
                            
                            let leaves = tree.select_leaves(batch_size);
                            if leaves.is_empty() { break; }
                            
                            let mut batch_features = Vec::with_capacity(leaves.len());
                            for leaf in &leaves {
                                batch_features.push(Self::encode_18_planes(leaf));
                            }
                            
                            // Batch inference via bridge
                            let mut policies = Vec::with_capacity(leaves.len());
                            let mut values = Vec::with_capacity(leaves.len());
                            
                            for feat in batch_features {
                                match batcher.submit_request_and_wait_rust(feat) {
                                    Ok((p, v)) => {
                                        policies.push(p);
                                        values.push(v);
                                    }
                                    Err(_) => break,
                                }
                            }
                            
                            if policies.len() < leaves.len() { break; }
                            
                            tree.expand_and_backup(&policies, &values);
                            sims_done += leaves.len();
                        }

                        if !running.load(Ordering::SeqCst) { break; }

                        // ── MCTS Policy ──
                        let policy = tree.get_policy(1.0, BOARD_SIZE);

                        // ── Sample and apply move ──
                        let legal = board.legal_moves();
                        if legal.is_empty() { break; }
                        
                        // Sample from policy
                        let mut rng = thread_rng();
                        let move_idx = match Self::sample_policy(&policy, &legal, &board) {
                            Some(idx) => idx,
                            None => *legal.choose(&mut rng).unwrap(),
                        };
                        
                        // ── Record position ──
                        records.push((Self::encode_18_planes(&board), policy, board.current_player));

                        if board.apply_move(move_idx.0, move_idx.1).is_err() {
                            break;
                        }
                        positions_generated.fetch_add(1, Ordering::SeqCst);
                    }

                    // ── Game End: determine outcome ──
                    let winner = board.winner();
                    let mut games_results = results_queue.lock().expect("results lock poisoned");
                    for (feat, pol, player) in records {
                        let outcome = match winner {
                            Some(p) => if p as i8 == player as i8 { 1.0 } else { -1.0 },
                            None => 0.0,
                        };
                        games_results.push_back((feat, pol, outcome));
                    }
                    games_completed.fetch_add(1, Ordering::SeqCst);
                    
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
    pub fn batcher(&self) -> RustInferenceBatcher {
        self.batcher.clone()
    }

    #[getter]
    pub fn games_completed(&self) -> usize {
        self.games_completed.load(Ordering::SeqCst)
    }

    #[getter]
    pub fn positions_generated(&self) -> usize {
        self.positions_generated.load(Ordering::SeqCst)
    }
}

impl RustSelfPlayRunner {
    fn encode_18_planes(board: &Board) -> Vec<f32> {
        let mut out = vec![0.0f32; 18 * TOTAL_CELLS];
        let planes = board.to_planes(); // returns [2, 19, 19]
        
        // Plane 0: my stones
        for i in 0..TOTAL_CELLS {
            out[i] = planes[i];
        }
        // Plane 8: opp stones
        for i in 0..TOTAL_CELLS {
            out[8 * TOTAL_CELLS + i] = planes[TOTAL_CELLS + i];
        }
        // Plane 16: moves_remaining == 2 ? 1.0 : 0.0
        let mr_val = if board.moves_remaining == 2 { 1.0 } else { 0.0 };
        for i in 0..TOTAL_CELLS {
            out[16 * TOTAL_CELLS + i] = mr_val;
        }
        // Plane 17: ply % 2
        let ply_val = (board.ply % 2) as f32;
        for i in 0..TOTAL_CELLS {
            out[17 * TOTAL_CELLS + i] = ply_val;
        }
        
        out
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

        let mut rng = thread_rng();
        let mut r: f32 = rand::Rng::gen(&mut rng);
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

impl Drop for RustSelfPlayRunner {
    fn drop(&mut self) {
        self.stop();
    }
}
