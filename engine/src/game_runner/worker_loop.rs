//! Self-play worker loop — spawns `n_workers` threads that each play full
//! games and push recorded (feature, policy, outcome, aux) rows into the
//! shared results queue.
//!
//! This is the hottest code path in the engine. The PyO3 `start()` facade in
//! `mod.rs` is a thin wrapper over `SelfPlayRunner::start_impl` defined here.
//! End-of-game ownership / winning-line reprojection has been extracted to
//! `records::reproject_game_end_row` so the inner loop stays under the
//! per-file size budget.

/// Quarter-cosine temperature schedule used by the self-play worker loop.
///
/// Returns 1.0 at compound_move=0, decays via cos(π/2·progress) toward
/// temp_min, then clamps at temp_min for compound_move ≥ temp_threshold.
///
/// # Arguments
/// * `compound_move`  — zero-indexed compound move number in the current game
/// * `temp_threshold` — compound move at which the floor kicks in (CLAUDE.md: 15)
/// * `temp_min`       — minimum temperature floor (CLAUDE.md: 0.05)
pub fn compute_move_temperature(
    compound_move: usize,
    temp_threshold: usize,
    temp_min: f32,
) -> f32 {
    if compound_move < temp_threshold {
        let progress = compound_move as f32 / temp_threshold as f32;
        f32::max(temp_min, (std::f32::consts::FRAC_PI_2 * progress).cos())
    } else {
        temp_min
    }
}

use std::sync::atomic::Ordering;
use std::thread;

use rand::prelude::IndexedRandom;
use rand::{rng, RngExt};

use crate::board::{Board, BOARD_SIZE, TOTAL_CELLS, hex_distance};
use crate::board::encode_chain_planes;
use crate::mcts::MCTSTree;

use super::SelfPlayRunner;
use super::gumbel_search::GumbelSearchState;
use super::records;

impl SelfPlayRunner {
    /// Spawn `n_workers` self-play threads. Idempotent: a second call while
    /// already running is a no-op.
    ///
    /// Each worker owns its own `MCTSTree`, RNG, and per-game state. Shared
    /// state is accessed through the `Arc` fields on `SelfPlayRunner`:
    /// `running` (kill switch), `games_completed` / `positions_generated` /
    /// `x_wins` / `o_wins` / `draws` / `results` / `recent_game_results` /
    /// `mcts_*_accum` (stats dashboards). All workers are joined by `stop()`.
    pub(crate) fn start_impl(&self) {
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }

        // Defense-in-depth mutex (§100): game-level (`fast_prob`) and
        // move-level (`full_search_prob`) playout-cap randomisers must not
        // both be active.  Python pool init raises first; this panic only
        // fires if `SelfPlayRunner` is driven from Rust or a path that
        // bypasses the Python validator.
        assert!(
            !(self.fast_prob > 0.0 && self.full_search_prob > 0.0),
            "playout-cap mutex violated: fast_prob={} and full_search_prob={} \
             are both > 0 (§100 — game-level and move-level caps are mutually \
             exclusive)",
            self.fast_prob,
            self.full_search_prob,
        );

        let mut handles = self.handles.lock().expect("runner handles lock poisoned");
        for worker_id in 0..self.n_workers {
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
            let fpu_reduction = self.fpu_reduction;
            let quiescence_enabled = self.quiescence_enabled;
            let quiescence_blend_2 = self.quiescence_blend_2;
            let fast_prob = self.fast_prob;
            let fast_sims = self.fast_sims;
            let standard_sims = self.standard_sims;
            let temp_threshold = self.temp_threshold_compound_moves;
            let temp_min = self.temp_min;
            let draw_reward = self.draw_reward;
            let zoi_enabled = self.zoi_enabled;
            let zoi_lookback = self.zoi_lookback;
            let zoi_margin = self.zoi_margin;
            let completed_q_values = self.completed_q_values;
            let c_visit = self.c_visit;
            let c_scale = self.c_scale;
            let gumbel_mcts = self.gumbel_mcts;
            let gumbel_m = self.gumbel_m;
            let gumbel_explore_moves = self.gumbel_explore_moves;
            let dirichlet_alpha   = self.dirichlet_alpha;
            let dirichlet_epsilon = self.dirichlet_epsilon;
            let dirichlet_enabled = self.dirichlet_enabled;
            let results_queue_cap = self.results_queue_cap;
            let full_search_prob  = self.full_search_prob;
            let n_sims_quick      = self.n_sims_quick;
            let n_sims_full       = self.n_sims_full;
            let results_queue = self.results.clone();
            let positions_dropped = self.positions_dropped.clone();
            let recent_game_results = self.recent_game_results.clone();
            let mcts_depth_accum = self.mcts_depth_accum.clone();
            let mcts_conc_accum = self.mcts_conc_accum.clone();
            let mcts_stat_count = self.mcts_stat_count.clone();
            let mcts_quiescence_fires = self.mcts_quiescence_fires.clone();
            let cluster_value_std_accum = self.cluster_value_std_accum.clone();
            let cluster_policy_disagreement_accum = self.cluster_policy_disagreement_accum.clone();
            let cluster_variance_samples = self.cluster_variance_samples.clone();

            let handle = thread::spawn(move || {
                let mut tree = MCTSTree::new_full(c_puct, crate::mcts::VIRTUAL_LOSS_PENALTY, fpu_reduction);
                tree.quiescence_enabled = quiescence_enabled;
                tree.quiescence_blend_2 = quiescence_blend_2;
                let mut rng = rng();
                // Per-worker game counter for the debug_prior_trace feature.
                // Increments at each game end so the trace records can be
                // grouped by (worker_id, game_index). Not read in default
                // builds — tracked unconditionally to avoid a cfg on every
                // increment site.
                #[cfg(feature = "debug_prior_trace")]
                let mut dbg_game_idx: u32 = 0;

                while running.load(Ordering::SeqCst) {
                    let mut board = Board::new();
                    let mut records_vec = Vec::new();
                    let mut move_history: Vec<(i32, i32)> = Vec::new();

                    // KataGo-style playout cap randomisation.
                    let is_fast_game = fast_prob > 0.0 && rng.random::<f32>() < fast_prob;
                    let game_sims = if is_fast_game { fast_sims } else { standard_sims };

                    for _ in 0..max_moves {
                        if !running.load(Ordering::SeqCst) || board.check_win() || board.legal_move_count() == 0 {
                            break;
                        }

                        // Move-level playout cap (orthogonal to game-level fast_prob above).
                        // When full_search_prob > 0.0, each move independently draws full or quick
                        // search. is_full_search tags the position so Python can gate policy loss.
                        // When disabled (full_search_prob == 0.0), all moves use game_sims and are
                        // tagged as full-search so the downstream mask is a no-op.
                        let (move_is_full_search, move_sims) = if full_search_prob > 0.0 {
                            let full = rng.random::<f32>() < full_search_prob;
                            let sims = if full { n_sims_full } else { n_sims_quick };
                            (full, sims)
                        } else {
                            (true, game_sims)
                        };

                        // ── MCTS Search ──
                        tree.new_game(board.clone());

                        // Helper: select leaves, run NN inference, expand+backup.
                        // Returns number of leaves processed, or 0 on failure.
                        let infer_and_expand = |tree: &mut MCTSTree, batch_size: usize| -> usize {
                            let leaves = tree.select_leaves(batch_size);
                            if leaves.is_empty() { return 0; }

                            let mut all_batch_features = Vec::new();
                            let mut leaf_metadata = Vec::with_capacity(leaves.len());

                            for leaf in &leaves {
                                let (views, centers) = leaf.get_cluster_views();
                                let k = views.len();
                                leaf_metadata.push((k, centers));
                                for view in views {
                                    let mut buffer = batcher.get_feature_buffer();
                                    leaf.encode_planes_to_buffer(&view, &mut buffer);
                                    all_batch_features.push(buffer);
                                }
                            }
                            if all_batch_features.is_empty() { return 0; }

                            let total_clusters: usize = leaf_metadata.iter().map(|(k, _)| *k).sum();

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
                                Err(_) => return 0,
                            };

                            if all_policies.len() < total_clusters { return 0; }

                            let mut aggregated_policies = Vec::with_capacity(leaves.len());
                            let mut aggregated_values = Vec::with_capacity(leaves.len());
                            let mut curr = 0;

                            for (i, (k, centers)) in leaf_metadata.iter().enumerate() {
                                let leaf_policies = &all_policies[curr..curr+k];
                                let leaf_values = &all_values[curr..curr+k];
                                curr += k;

                                // I2 investigation metric: per-cluster value/policy
                                // variance. Attention hijacking probe (Q2, Q27). Only
                                // meaningful when K≥2 clusters exist for this position.
                                if *k >= 2 {
                                    let mean_v: f32 =
                                        leaf_values.iter().sum::<f32>() / *k as f32;
                                    let var_v: f32 = leaf_values.iter()
                                        .map(|&v| (v - mean_v).powi(2))
                                        .sum::<f32>() / *k as f32;
                                    let std_v = var_v.sqrt();
                                    let mut top1 = Vec::with_capacity(*k);
                                    for p in leaf_policies {
                                        let mut bi = 0usize;
                                        let mut bv = p[0];
                                        for (ii, &pv) in p.iter().enumerate() {
                                            if pv > bv { bi = ii; bv = pv; }
                                        }
                                        top1.push(bi);
                                    }
                                    let mut max_c = 1usize;
                                    for &a in &top1 {
                                        let c = top1.iter().filter(|&&x| x == a).count();
                                        if c > max_c { max_c = c; }
                                    }
                                    let disagree = 1.0f32 - (max_c as f32 / *k as f32);
                                    cluster_value_std_accum.fetch_add(
                                        (std_v * 1_000_000.0) as u64, Ordering::Relaxed);
                                    cluster_policy_disagreement_accum.fetch_add(
                                        (disagree * 1_000_000.0) as u64, Ordering::Relaxed);
                                    cluster_variance_samples.fetch_add(1, Ordering::Relaxed);
                                }

                                let mut min_v = leaf_values[0];
                                for &v in leaf_values {
                                    if v < min_v { min_v = v; }
                                }
                                aggregated_values.push(min_v);
                                aggregated_policies.push(records::aggregate_policy(&leaves[i], centers, leaf_policies));
                            }

                            let n = leaves.len();
                            tree.expand_and_backup(&aggregated_policies, &aggregated_values);
                            n
                        };

                        let mut gumbel_state: Option<GumbelSearchState> = None;

                        if gumbel_mcts {
                            // ── Gumbel MCTS with Sequential Halving ──
                            // Phase 1: Expand root to get children and priors.
                            let root_sims = infer_and_expand(&mut tree, 1);
                            if root_sims == 0 || !tree.pool[0].is_expanded() {
                                // Root expansion failed — skip this move.
                                continue;
                            }
                            let mut sims_used = root_sims;

                            // Apply Dirichlet noise to root priors after expansion.
                            // Skip at intermediate ply (second stone of compound turn) —
                            // mirrors hexo_rl/selfplay/worker.py:107-111.
                            // ply==0 is P1's single opening stone, which IS a turn boundary.
                            let is_intermediate_ply = board.moves_remaining == 1 && board.ply > 0;
                            if dirichlet_enabled && !is_intermediate_ply {
                                let n_ch = tree.pool[0].n_children as usize;
                                if n_ch > 0 {
                                    let noise = crate::mcts::dirichlet::sample_dirichlet(
                                        dirichlet_alpha, n_ch, &mut rng,
                                    );
                                    tree.apply_dirichlet_to_root(&noise, dirichlet_epsilon);
                                }
                            }

                            // Phase 2: Gumbel-Top-k candidate selection.
                            // Guard: if effective_m is 0 (no budget or no children),
                            // fall back to the standard PUCT path for this move.
                            let effective_m = gumbel_m.min(move_sims).min(tree.root_n_children());
                            if effective_m == 0 {
                                // No candidates — use standard PUCT search instead.
                                // Subtract root expansion sims already spent from budget.
                                let mut sims_done = sims_used;
                                while sims_done < move_sims {
                                    if !running.load(Ordering::SeqCst) { break; }
                                    let n = infer_and_expand(&mut tree, leaf_batch_size);
                                    if n == 0 { break; }
                                    sims_done += n;
                                }
                                gumbel_state = None;
                            } else {

                            let mut gs = GumbelSearchState::new(
                                &tree, effective_m, c_visit, c_scale, &mut rng,
                            );

                            // Phase 3: Sequential Halving — allocate budget across phases.
                            let num_phases = gs.num_phases;
                            for phase in 0..num_phases {
                                if sims_used >= move_sims { break; }
                                let remaining_budget = move_sims.saturating_sub(sims_used);
                                let remaining_phases = num_phases - phase;
                                let sims_per = (remaining_budget / (remaining_phases * gs.candidates.len())).max(1);

                                let cands = gs.candidates.clone();
                                for &cand_offset in &cands {
                                    if sims_used >= move_sims { break; }
                                    if !running.load(Ordering::SeqCst) { break; }

                                    let child_pool_idx = gs.first_child + cand_offset as u32;
                                    tree.forced_root_child = Some(child_pool_idx);

                                    let mut cand_sims = 0;
                                    while cand_sims < sims_per && sims_used < move_sims {
                                        if !running.load(Ordering::SeqCst) { break; }
                                        // Cap batch to remaining budget for this candidate so we
                                        // don't overshoot sims_per (leaf_batch_size can be 8 when
                                        // only 1-3 sims are budgeted, biasing early candidates).
                                        let batch = leaf_batch_size.min(sims_per.saturating_sub(cand_sims));
                                        let n = infer_and_expand(&mut tree, batch.max(1));
                                        if n == 0 { break; }
                                        cand_sims += n;
                                        sims_used += n;
                                    }
                                    tree.forced_root_child = None;
                                }

                                if gs.candidates.len() <= 1 { break; }
                                gs.halve_candidates(&tree);
                            }
                            tree.forced_root_child = None;
                            gumbel_state = Some(gs);
                            } // end effective_m > 0
                        } else {
                            // ── Standard PUCT search with Dirichlet root noise ──
                            // Expand root first so priors are available before noise injection.
                            // Uses batch=1 to guarantee only the root leaf is processed,
                            // matching the Gumbel branch pattern.
                            let root_n = infer_and_expand(&mut tree, 1);
                            if root_n == 0 {
                                continue;
                            }
                            let mut sims_done = root_n;

                            // Apply Dirichlet noise after root expansion, before simulation loop.
                            // Skip at intermediate ply (second stone of compound turn) —
                            // mirrors hexo_rl/selfplay/worker.py:107-111.
                            // ply==0 is P1's single opening stone, which IS a turn boundary.
                            let is_intermediate_ply = board.moves_remaining == 1 && board.ply > 0;
                            if dirichlet_enabled && !is_intermediate_ply {
                                if tree.pool[0].is_expanded() {
                                    let n_ch = tree.pool[0].n_children as usize;
                                    if n_ch > 0 {
                                        let noise = crate::mcts::dirichlet::sample_dirichlet(
                                            dirichlet_alpha, n_ch, &mut rng,
                                        );
                                        tree.apply_dirichlet_to_root(&noise, dirichlet_epsilon);
                                    }
                                }
                            }

                            while sims_done < move_sims {
                                if !running.load(Ordering::SeqCst) { break; }
                                let n = infer_and_expand(&mut tree, leaf_batch_size);
                                if n == 0 { break; }
                                sims_done += n;
                            }
                        }

                        if !running.load(Ordering::SeqCst) { break; }

                        // ── MCTS Policy with cosine-annealed temperature schedule ──
                        let compound_move = if board.ply == 0 { 0 } else { (board.ply as usize + 1) / 2 };
                        let temperature = if is_fast_game {
                            1.0  // fast games: always exploratory
                        } else {
                            compute_move_temperature(compound_move, temp_threshold, temp_min)
                        };
                        let policy = tree.get_policy(temperature, BOARD_SIZE);

                        // ── debug_prior_trace: snapshot root priors + visit counts ──
                        // Compile-time gated; zero cost in default builds. Runtime
                        // gated a second time by HEXO_PRIOR_TRACE_PATH (see
                        // engine/src/debug_trace.rs). Cap-enforced per site.
                        #[cfg(feature = "debug_prior_trace")]
                        {
                            let root = &tree.pool[0];
                            if root.is_expanded() {
                                let first = root.first_child as usize;
                                let n_ch = root.n_children as usize;
                                let mut priors = Vec::with_capacity(n_ch);
                                let mut visits = Vec::with_capacity(n_ch);
                                for j in 0..n_ch {
                                    priors.push(tree.pool[first + j].prior);
                                    visits.push(tree.pool[first + j].n_visits);
                                }
                                let legal_count = board.legal_move_count() as u32;
                                crate::debug_trace::record_game_runner(
                                    dbg_game_idx,
                                    worker_id as u32,
                                    compound_move as u32,
                                    board.ply as u32,
                                    legal_count,
                                    n_ch as u32,
                                    game_sims as u32,
                                    &priors,
                                    &visits,
                                    temperature,
                                    is_fast_game,
                                );
                            }
                        }

                        // Accumulate MCTS health stats once per search (not in inner sim loop).
                        {
                            let (depth, conc) = tree.last_search_stats();
                            mcts_depth_accum.fetch_add((depth * 1_000_000.0) as u64, Ordering::Relaxed);
                            mcts_conc_accum.fetch_add((conc * 1_000_000.0) as u64, Ordering::Relaxed);
                            mcts_stat_count.fetch_add(1, Ordering::Relaxed);
                            // Accumulate quiescence fires for this search. tree.new_game() resets
                            // the counter before each search, so this captures per-search fires.
                            mcts_quiescence_fires.fetch_add(
                                tree.quiescence_fire_count.load(Ordering::Relaxed),
                                Ordering::Relaxed,
                            );
                        }

                        // Completed Q-values: compute improved policy for training target.
                        // Move selection still uses temperature-scaled visit counts above.
                        let target_policy = if completed_q_values {
                            tree.get_improved_policy(BOARD_SIZE, c_visit, c_scale)
                        } else {
                            policy.clone()
                        };

                        // ── Sample and apply move (ZOI-filtered legal set) ──
                        let full_legal = board.legal_moves();
                        if full_legal.is_empty() { break; }

                        // ZOI filtering: restrict move sampling to cells near recent moves.
                        let legal = if zoi_enabled && move_history.len() >= 3 {
                            let filtered: Vec<_> = full_legal.iter()
                                .filter(|(q, r)| {
                                    move_history.iter().rev().take(zoi_lookback).any(|(q0, r0)| {
                                        hex_distance(*q, *r, *q0, *r0) <= zoi_margin
                                    })
                                })
                                .cloned()
                                .collect();
                            if filtered.len() < 3 { full_legal } else { filtered }
                        } else {
                            full_legal
                        };

                        // Move selection: Gumbel winner or visit-count sampling.
                        // Early in the game (< gumbel_explore_moves plies), always use
                        // visit-count sampling for exploration (paper Figure 8b).
                        // Note: threshold is in plies, not turns. With compound moves
                        // (2 per turn after the first), ply 10 ≈ turn 6. Adjust the
                        // gumbel_explore_moves default if more early exploration is desired.
                        let use_gumbel_winner = gumbel_state.is_some()
                            && board.ply as usize >= gumbel_explore_moves;
                        let move_idx = if use_gumbel_winner {
                            let gs = gumbel_state.as_mut().unwrap();
                            let best_pool = gs.best_action_pool_idx(&tree);
                            let val = tree.pool[best_pool as usize].action_idx;
                            let mq = (val >> 16) as i32 - 32768;
                            let mr = (val & 0xFFFF) as i32 - 32768;
                            // Ensure chosen move is legal (it always should be).
                            if legal.contains(&(mq, mr)) {
                                (mq, mr)
                            } else {
                                match records::sample_policy(&policy, &legal, &board) {
                                    Some(idx) => idx,
                                    None => *legal.choose(&mut rng).unwrap(),
                                }
                            }
                        } else {
                            match records::sample_policy(&policy, &legal, &board) {
                                Some(idx) => idx,
                                None => *legal.choose(&mut rng).unwrap(),
                            }
                        };

                        // ── Record position ──
                        let (views, centers) = board.get_cluster_views();
                        for (k, center) in centers.iter().enumerate() {
                            let mut feat = batcher.get_feature_buffer();
                            // get_cluster_views returns 2-plane views; expand to 18 for storage.
                            board.encode_planes_to_buffer(&views[k], &mut feat);
                            // Compute Q13 chain-length planes separately (not in state).
                            let mut chain = vec![0.0f32; 6 * TOTAL_CELLS];
                            encode_chain_planes(
                                &views[k][..TOTAL_CELLS],
                                &views[k][TOTAL_CELLS..2 * TOTAL_CELLS],
                                &mut chain,
                            );
                            // Fast games: zero-policy marks value-only targets (unless
                            // completed Q-values are enabled, which give signal even at 50 sims).
                            let projected_policy = if is_fast_game && !completed_q_values {
                                vec![0.0; BOARD_SIZE * BOARD_SIZE + 1]
                            } else {
                                records::aggregate_policy_to_local(&board, center, &target_policy)
                            };
                            // Capture the per-row window centre so the aux targets
                            // (computed at game end) can be reprojected into the same
                            // coordinate frame as this row's state planes.
                            records_vec.push((feat, chain, projected_policy, board.current_player, center.0, center.1, move_is_full_search));
                        }

                        if board.apply_move(move_idx.0, move_idx.1).is_err() {
                            break;
                        }
                        move_history.push((move_idx.0, move_idx.1));
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
                    // Snapshot the final-board cell list and winning line once;
                    // each row reprojects them into its own per-cluster window centre.
                    let final_cells: Vec<((i32, i32), crate::board::Cell)> = board.cells
                        .iter()
                        .map(|(&qr, &c)| (qr, c))
                        .collect();
                    let winning_cells: Vec<(i32, i32)> = board.find_winning_line();

                    let mut games_results = results_queue.lock().expect("results lock poisoned");
                    for (feat, chain, pol, player, cq, cr, is_full_search) in records_vec {
                        let outcome = match winner {
                            Some(p) => if p as i8 == player as i8 { 1.0 } else { -1.0 },
                            None => draw_reward,
                        };

                        // Per-row aux reprojection (ownership + winning_line) into this
                        // row's per-cluster window centre. See records::reproject_game_end_row.
                        let aux_u8 = records::reproject_game_end_row(
                            &final_cells, &winning_cells, cq, cr,
                        );

                        games_results.push_back((feat, chain, pol, outcome, plies, aux_u8, is_full_search));
                    }
                    games_completed.fetch_add(1, Ordering::Relaxed);
                    match winner {
                        Some(crate::board::Player::One) => { x_wins.fetch_add(1, Ordering::Relaxed); }
                        Some(_)                         => { o_wins.fetch_add(1, Ordering::Relaxed); }
                        None                            => { draws.fetch_add(1, Ordering::Relaxed); }
                    }
                    // Per-row aux is already pushed into game_results above. The
                    // game-end record here only carries metadata for Python's
                    // game_complete logging — no spatial aux fields.
                    {
                        let mut rg = recent_game_results.lock().expect("recent_game_results lock poisoned");
                        rg.push_back((plies, winner_code, move_history, worker_id));
                        // Cap at 2000 entries to avoid unbounded growth if Python is slow.
                        if rg.len() > 2000 {
                            rg.pop_front();
                        }
                    }

                    // Cap the results queue to avoid memory explosion if Python is slow.
                    // Drop count is tracked on `positions_dropped` for dashboard visibility.
                    if games_results.len() > results_queue_cap {
                        let to_drop = games_results.len() - results_queue_cap;
                        for _ in 0..to_drop {
                            games_results.pop_front();
                        }
                        positions_dropped.fetch_add(to_drop as u64, Ordering::Relaxed);
                    }

                    #[cfg(feature = "debug_prior_trace")]
                    { dbg_game_idx += 1; }
                }
            });
            handles.push(handle);
        }
    }
}
