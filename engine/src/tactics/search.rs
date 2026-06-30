//! Search core — iterative AND-OR threat-space proof over HTTT compound turns.
//!
//! Ported from `scripts/dtactical/solver.py::solve`, onto the native `Board`
//! with **zero clone per node** (`apply_move_tracked`/`undo_move`). Flip-aware
//! negamax: a child result is negated ONLY when the to-move player flipped
//! (HTTT places 2 stones/turn — the side-to-move flips every 2 plies, not every
//! ply). The engine's own turn-flip drives the min/max alternation.
//!
//! # DEFERRED (the load-bearing quiet-move work — see `ordering.rs`/`eval.rs`)
//! This is the threat/double-threat proof core = the measured **8% ceiling**.
//! The quiet-move alpha-beta BODY + threat-quiescence tail + PVS/LMR/aspiration
//! are NOT built here. The narrow threat-guided candidate set finds FORCING
//! wins/losses (mate-in-≤2-turns band); a position whose only escape/win is a
//! quiet developmental move returns UNKNOWN (never a false proof).

use fxhash::FxHashSet;

use crate::board::Board;

use super::eval::heuristic_leaf;
use super::ordering::{candidates, order_moves, OrderingState};
use super::tt::{Bound, ProofTt};
use super::{outcome_of, Budget, Outcome, TacticalConfig, MATE, NEG_INF, POS_INF, WIN_THRESHOLD};

/// A scored node value (D-PFIT P2 increment 1). `score` is a mate-distance-aware
/// value: a proven mate is `±(MATE - ply)` (magnitude >= `WIN_THRESHOLD`); a
/// heuristic / unresolved node is a bounded score (magnitude < `WIN_THRESHOLD`).
/// `line` is the principal variation, populated for WIN — `line[0]` is the move
/// to play; for a 2-stone-turn forcing win `line[0]`/`line[1]` are the
/// side-to-move's two stones (the A1 override caches `line[1]`). Empty otherwise.
pub struct Scored {
    pub score: i32,
    pub line: Vec<(i32, i32)>,
}

impl Scored {
    /// A bounded (non-proof) leaf value — clamped strictly inside the proof
    /// region so `outcome_of` can NEVER read it as a WIN/LOSS proof.
    #[inline]
    fn heuristic(score: i32) -> Self {
        Scored { score: clamp_heuristic(score), line: Vec::new() }
    }
}

/// Clamp a heuristic score strictly inside `(-WIN_THRESHOLD, WIN_THRESHOLD)` so a
/// non-proof leaf can never masquerade as a mate (the soundness invariant: only
/// the proof paths in `solve` may emit a mate-magnitude score).
#[inline]
pub(crate) fn clamp_heuristic(score: i32) -> i32 {
    score.clamp(-(WIN_THRESHOLD - 1), WIN_THRESHOLD - 1)
}

/// Scored α-β AND-OR threat-space proof for `board.current_player` (negamax over
/// HTTT compound turns). Returns a mate-distance-aware score; the 3-valued
/// verdict is `outcome_of(score)` at the ROOT (full-window, exact).
///
/// # Soundness (the load-bearing property — α-β is pruning ONLY)
/// NET-FREE: a mate-magnitude score is produced ONLY by a sound proof path —
/// terminal CF-1 backup (`terminal_value_to_move`), the stone-count shortcuts, an
/// all-candidates-lose node guarded by the R3 completeness check, or the recall
/// verify. The value head / `eval.rs` heuristic is NEVER a proof: a heuristic
/// leaf is `Scored::heuristic` (clamped below `WIN_THRESHOLD`). α-β changes
/// neither: (a) the ROOT runs a FULL window so its value is EXACT and the verdict
/// is sound; (b) a proven LOSS is concluded ONLY when the candidate loop ran to
/// completion with NO β-cutoff, so the all-lose `best` is the exact node value;
/// (c) the proven-WIN cutoff (`best >= WIN_THRESHOLD`) fires before any sibling is
/// searched with `β <= -WIN_THRESHOLD`, so no fail-high LOSS bound can propagate
/// or corrupt a winning PV. The `#[cfg(test)]` 3-valued reference (`solve_3valued`)
/// + the brute oracle cross-check every verdict.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve(
    board: &mut Board,
    depth_left: i32,
    ply: i32,
    mut alpha: i32,
    beta: i32,
    budget: &mut Budget,
    cfg: &TacticalConfig,
    tt: &mut ProofTt,
    ordering: &mut OrderingState,
) -> Scored {
    if !budget.tick() {
        return Scored::heuristic(0); // budget out => UNKNOWN, never a proof
    }

    // (1) Terminal: the engine-owned CF-1 sign is the ONLY proof sign. Mate
    //     distance = `ply` (shorter mates score higher in magnitude).
    if board.check_win() {
        let score = if board.terminal_value_to_move() > 0.0 { MATE - ply } else { -(MATE - ply) };
        return Scored { score, line: Vec::new() };
    }

    let stm = board.current_player;
    let opp = stm.other();

    // (2) Immediate-win shortcut: side-to-move completes 6 with its next stone.
    //     Sound stone-count proof (no net). Yields the winning cell for the PV.
    if board.count_winning_moves(stm) >= 1 {
        let line = board.first_winning_move(stm).map_or_else(Vec::new, |m| vec![m]);
        return Scored { score: MATE - ply, line };
    }

    // (3) Double-threat LOSS shortcut (mr==1 only — provably sound): stm has NO
    //     immediate win and places exactly ONE stone before the turn flips; opp
    //     then has >=2 standing win-in-1 cells -> stm blocks <=1 -> LOSS.
    if board.moves_remaining == 1 && board.count_winning_moves(opp) >= 2 {
        return Scored { score: -(MATE - ply), line: Vec::new() };
    }

    // (4) TT probe — only PROVEN LOSS is trusted as a proof (see `tt.rs`): a WIN
    //     hit would return an empty PV and could truncate the override line. A
    //     cached LOSS is game-theoretic; `get_loss_proof` decodes the mate distance
    //     at THIS node's ply.
    let key = (board.zobrist_hash, stm as i8, board.moves_remaining);
    if let Some(score) = tt.get_loss_proof(key, ply) {
        debug_assert!(score <= -WIN_THRESHOLD, "TT proof probe must decode a mate-magnitude LOSS");
        return Scored { score, line: Vec::new() };
    }

    if depth_left <= 0 {
        budget.hit_horizon = true; // a real depth truncation — deepening could help
        return Scored::heuristic(heuristic_leaf(board)); // horizon => non-proof leaf
    }

    // `in_check` (opp threatens an immediate win) is the prune-symmetry premise of
    // the LOSS conclusion below — captured at the node state, before descent.
    let in_check = board.count_winning_moves(opp) >= 1;
    let mut moves = candidates(board, stm, opp, cfg.cand_cap, cfg.neighbor_dist);
    if moves.is_empty() {
        return Scored::heuristic(heuristic_leaf(board)); // quiet => cannot prove
    }
    let moves_len = moves.len();
    // Remember the reduced set so the recall verify (below) searches only the
    // DROPPED legal moves, not the ones already explored. Built BEFORE ordering so
    // it is order-independent (ordering is a permutation: same set, same guard).
    let searched: FxHashSet<(i32, i32)> = moves.iter().copied().collect();
    // Best-first ORDERING (permutation only — never changes the set or a verdict):
    // TT best move, killers, history. Earlier α-β cutoffs, identical conclusions.
    order_moves(&mut moves, tt.get_best_move(key), ply, ordering);

    // Negamax-α-β OR-node. `best` is the side-to-move's best value over the
    // candidate set; it stays `NEG_INF` until a child is examined (the all-Err
    // guard below). A proven-WIN cutoff fires the instant `best >= WIN_THRESHOLD`
    // (= the old "return on first WIN", PV-identical); the generic `alpha >= beta`
    // β-cutoff prunes once a heuristic value dominates (the lever the eval/ordering
    // increments exploit). Either cutoff returns `best` as a BOUND and never
    // reaches the LOSS-proof logic.
    let mut best = NEG_INF;
    let mut best_line: Vec<(i32, i32)> = Vec::new();
    let mut cutoff = false;

    for (idx, &(q, r)) in moves.iter().enumerate() {
        let node_player = board.current_player; // == stm
        let diff = match board.apply_move_tracked(q, r) {
            Ok(d) => d,
            Err(_) => continue,
        };
        // Flip-aware negamax: negate the child value (and flip the window) ONLY
        // when the to-move side flipped (turn-final stone).
        let flipped = board.current_player != node_player;
        let dchild = depth_left - 1;

        // PVS + LMR (pruning/ordering ONLY — verdict-exact; see the soundness note
        // above and `verdict_invariance_fuzz_*`). idx 0 = principal variation:
        // full window, full depth. Later moves: a null-window scout (LMR-reduced
        // depth for late quiet not-in-check moves), re-searched at FULL depth+window
        // whenever the scout could affect the node value.
        let (rc, child_line) = if idx == 0 {
            let (ca, cb) = if flipped { (-beta, -alpha) } else { (alpha, beta) };
            let c = solve(board, dchild, ply + 1, ca, cb, budget, cfg, tt, ordering);
            (if flipped { -c.score } else { c.score }, c.line)
        } else {
            let reduce = lmr_reduction(idx, depth_left, in_check);
            let (na, nb) = if flipped { (-alpha - 1, -alpha) } else { (alpha, alpha + 1) };
            let c = solve(board, dchild - reduce, ply + 1, na, nb, budget, cfg, tt, ordering);
            let mut rc = if flipped { -c.score } else { c.score };
            let mut line = c.line;
            // Re-search at full depth + full window when the scout could matter:
            //   - PVS scout (reduce==0): the move beats α inside the window
            //     (alpha < rc < beta) so the null result is not exact.
            //   - LMR (reduce>0): the reduced search beat α OR was inconclusive
            //     (UNKNOWN) — either way the reduced value cannot be trusted to
            //     EXCLUDE a deeper proof, so confirm at full depth. The ONLY case
            //     we accept the reduced value is a reduced-PROVEN result that does
            //     not beat α (a sub-α loss proof) — verdict-irrelevant, so skipping
            //     its full re-search leaves the node value/verdict unchanged.
            let need_full = if reduce > 0 {
                rc > alpha || outcome_of(rc) == Outcome::Unknown
            } else {
                rc > alpha && rc < beta
            };
            if need_full {
                let (fa, fb) = if flipped { (-beta, -alpha) } else { (alpha, beta) };
                let c = solve(board, dchild, ply + 1, fa, fb, budget, cfg, tt, ordering);
                rc = if flipped { -c.score } else { c.score };
                line = c.line;
            }
            (rc, line)
        };

        if rc > best {
            best = rc;
            // PV: same player continued (mr was 2) => append the child's line so
            // it carries stm's 2nd stone; flipped (turn-final) => the move alone.
            let mut line = vec![(q, r)];
            if !flipped {
                line.extend(child_line.iter().copied());
            }
            best_line = line;
        }
        board.undo_move(diff);

        if best > alpha {
            alpha = best;
        }
        if best >= WIN_THRESHOLD || alpha >= beta {
            // Cutoff move (WIN or β): reward it for ordering future siblings.
            ordering.record_cutoff(ply, (q, r), depth_left);
            cutoff = true;
            break;
        }
    }

    // No candidate could be applied (degenerate) => UNKNOWN, never a false LOSS.
    if best == NEG_INF {
        return Scored::heuristic(0);
    }

    // Proven WIN (`best` is a mate magnitude): exact lower bound is already a win.
    if best >= WIN_THRESHOLD {
        // ORDERING hint only (is_proof=false; never returned as a verdict — a WIN
        // is always reconstructed). The winning move ordered first next visit.
        tt.store_bound(key, best, ply, Bound::Lower, best_line.first().copied(), depth_left);
        return Scored { score: best, line: best_line };
    }
    // β-cutoff with a non-win `best`: `best` is a fail-high BOUND, not the exact
    // value, so the LOSS-proof logic below must NOT run (it requires the exact
    // node value from a fully-examined loop). Returning the bound is sound — it is
    // strictly above `-WIN_THRESHOLD` (every node has `beta > -WIN_THRESHOLD`; see
    // the soundness note), so it can never be misread as a proven LOSS.
    if cutoff {
        tt.store_bound(key, best, ply, Bound::Lower, best_line.first().copied(), depth_left);
        return Scored { score: best, line: best_line };
    }

    // Loop ran to completion with no cutoff => `best` is the EXACT node value.
    //   best >  -WIN_THRESHOLD  : some candidate did not lose => UNKNOWN.
    //   best <= -WIN_THRESHOLD  : EVERY candidate loses => candidate (R3) logic.
    if outcome_of(best) != Outcome::Loss {
        return Scored::heuristic(best);
    }

    // R3 LOSS-COMPLETENESS GUARD (load-bearing for sound z-LOSS labels).
    // "Every CANDIDATE loses" only proves "every LEGAL move loses" when the
    // candidate set provably covered all escapes:
    //   - `in_check && moves_len < cand_cap`: opp threatens an immediate win, so a
    //     non-block/non-counter loses to the standing threat (prune symmetry) —
    //     valid ONLY if the set was not truncated. The `< cand_cap` boundary is
    //     deliberately PESSIMISTIC (a natural-size-==-cand_cap set is treated as
    //     truncated -> UNKNOWN, a recall false-negative, never a soundness break).
    //     Do NOT relax to `<=`.
    //   - `moves_len >= legal_move_count()`: the candidate set IS the full legal
    //     set (covers not-in-check / quiet-move nodes where prune symmetry fails).
    // Otherwise -> the recall verify (if enabled) or conservative UNKNOWN.
    let loss_complete =
        (in_check && moves_len < cfg.cand_cap) || moves_len >= board.legal_move_count();
    if loss_complete {
        tt.store_loss_proof(key, best, ply, depth_left); // proven, game-theoretic — cache it
        return Scored { score: best, line: Vec::new() };
    }

    // RECALL-PRESERVING VERIFY (quiet-move body, `neighbor_dist` set). The reduced
    // candidate set drove the search; certifying a LOSS needs the DROPPED legal
    // moves too. Search them with a FULL window (NO α-β pruning — preserve exact
    // recall): any WIN is an escape (return it); any UNKNOWN leaves it unresolved;
    // only when EVERY dropped move also loses is the LOSS certified. `vbest` tracks
    // the slowest (least-negative) loss for a correct mate distance.
    if cfg.neighbor_dist.is_some() {
        let mut vbest = best;
        for (q, r) in board.legal_moves() {
            if searched.contains(&(q, r)) {
                continue;
            }
            let node_player = board.current_player;
            let diff = match board.apply_move_tracked(q, r) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let flipped = board.current_player != node_player;
            // Full window, full depth, NO PVS/LMR — the verify preserves EXACT
            // recall for the LOSS certification (the soundness-critical path).
            let child = solve(board, depth_left - 1, ply + 1, NEG_INF, POS_INF, budget, cfg, tt, ordering);
            let rc = if flipped { -child.score } else { child.score };
            if rc >= WIN_THRESHOLD {
                let mut line = vec![(q, r)];
                if !flipped {
                    line.extend(child.line.iter().copied());
                }
                board.undo_move(diff);
                return Scored { score: rc, line };
            }
            board.undo_move(diff);
            if outcome_of(rc) != Outcome::Loss {
                return Scored::heuristic(0); // a dropped move escapes the loss
            }
            if rc > vbest {
                vbest = rc;
            }
        }
        // Every reduced candidate AND every dropped legal move loses => certified.
        tt.store_loss_proof(key, vbest, ply, depth_left);
        return Scored { score: vbest, line: Vec::new() };
    }

    Scored::heuristic(0) // candidate set incomplete, verify disabled -> cannot prove LOSS
}

/// LMR depth reduction for a non-PV candidate. Reduce late (`idx >= 6`) moves at
/// sufficient depth (`depth_left >= 4`) by one ply; never reduce in check (every
/// candidate is a forced defense). Matches SealBot's `search.h:715` index/depth
/// gate. SOUNDNESS: the reduction is VERDICT-EXACT — `solve` re-searches at full
/// depth whenever a reduced child could affect the node value, so a reduction can
/// only SKIP the deep re-search of a child already proved losing below α (a
/// verdict-irrelevant short-cut). It never reduces depth on a move that matters.
#[inline]
fn lmr_reduction(idx: usize, depth_left: i32, in_check: bool) -> i32 {
    if !in_check && idx >= 6 && depth_left >= 4 {
        1
    } else {
        0
    }
}

/// Iterative-deepening + aspiration ROOT driver (D-PFIT P2 increment 4). Deepens
/// `1..=max_depth`, reusing the TT + ordering state across iterations for earlier
/// cutoffs; stops as soon as a depth proves a mate (a proof is final) or the node
/// budget is exhausted, keeping the deepest COMPLETED result.
///
/// SOUNDNESS: every accepted iteration's root window resolves to an EXACT value
/// (`aspiration_search` widens to ±∞ on a fail), so `outcome_of(score)` is a sound
/// verdict — the same premise as the single full-window root search.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_root(
    board: &mut Board,
    max_depth: i32,
    budget: &mut Budget,
    cfg: &TacticalConfig,
    tt: &mut ProofTt,
    ordering: &mut OrderingState,
) -> Scored {
    let mut result = Scored::heuristic(0);
    let mut have = false;
    let mut last = 0i32;
    let mut depth = 1;
    while depth <= max_depth {
        budget.hit_horizon = false; // track whether THIS iteration was depth-truncated
        let s = aspiration_search(board, depth, last, have, budget, cfg, tt, ordering);
        if budget.exhausted {
            if !have {
                result = s; // first iteration starved: best effort
            }
            break;
        }
        result = s;
        last = result.score;
        have = true;
        if outcome_of(result.score) != Outcome::Unknown {
            break; // proven WIN/LOSS — final; deeper search cannot change it
        }
        if !budget.hit_horizon {
            break; // tree fully resolved within depth — deepening cannot change it
        }
        depth += 1;
    }
    result
}

/// One aspiration-windowed root search at `depth`. A narrow window around the
/// previous score yields faster cutoffs; a fail-low/high widens that side to ±∞
/// and re-searches so the RETURNED value is always exact (never a clipped bound).
#[allow(clippy::too_many_arguments)]
fn aspiration_search(
    board: &mut Board,
    depth: i32,
    last: i32,
    have: bool,
    budget: &mut Budget,
    cfg: &TacticalConfig,
    tt: &mut ProofTt,
    ordering: &mut OrderingState,
) -> Scored {
    const W: i32 = 64; // aspiration half-width (heuristic units)
    let (mut alpha, mut beta) = (NEG_INF, POS_INF);
    if have && last.abs() < WIN_THRESHOLD {
        alpha = (last - W).max(NEG_INF);
        beta = (last + W).min(POS_INF);
    }
    loop {
        let s = solve(board, depth, 0, alpha, beta, budget, cfg, tt, ordering);
        if budget.exhausted {
            return s; // starved: caller keeps the last completed result
        }
        if s.score <= alpha && alpha > NEG_INF {
            alpha = NEG_INF; // fail low: widen down, re-search exact
            continue;
        }
        if s.score >= beta && beta < POS_INF {
            beta = POS_INF; // fail high: widen up, re-search exact
            continue;
        }
        return s; // strictly in-window => exact value
    }
}

/// 3-VALUED REFERENCE ORACLE (test-only). The pre-increment-1 proof core,
/// verbatim, kept as the verdict-invariance oracle for the scored α-β `solve`:
/// the scored search MUST reproduce this oracle's every WIN/LOSS conclusion (α-β
/// is a pruning/ordering optimisation, never a verdict change). See
/// `verdict_invariance_scored_matches_3valued`.
#[cfg(test)]
pub(crate) struct Solved3 {
    pub outcome: Outcome,
    pub line: Vec<(i32, i32)>,
}

#[cfg(test)]
impl Solved3 {
    #[inline]
    fn unknown() -> Self {
        Solved3 { outcome: Outcome::Unknown, line: Vec::new() }
    }
}

#[cfg(test)]
pub(crate) fn solve_3valued(
    board: &mut Board,
    depth_left: i32,
    budget: &mut Budget,
    cfg: &TacticalConfig,
    tt: &mut ProofTt,
) -> Solved3 {
    if !budget.tick() {
        return Solved3::unknown();
    }
    if board.check_win() {
        let outcome =
            if board.terminal_value_to_move() > 0.0 { Outcome::Win } else { Outcome::Loss };
        return Solved3 { outcome, line: Vec::new() };
    }
    let stm = board.current_player;
    let opp = stm.other();
    if board.count_winning_moves(stm) >= 1 {
        let line = board.first_winning_move(stm).map_or_else(Vec::new, |m| vec![m]);
        return Solved3 { outcome: Outcome::Win, line };
    }
    if board.moves_remaining == 1 && board.count_winning_moves(opp) >= 2 {
        return Solved3 { outcome: Outcome::Loss, line: Vec::new() };
    }
    let key = (board.zobrist_hash, stm as i8, board.moves_remaining);
    if tt.get_loss_proof(key, 0).is_some() {
        return Solved3 { outcome: Outcome::Loss, line: Vec::new() };
    }
    if depth_left <= 0 {
        return Solved3::unknown();
    }
    let in_check = board.count_winning_moves(opp) >= 1;
    let moves = candidates(board, stm, opp, cfg.cand_cap, cfg.neighbor_dist);
    if moves.is_empty() {
        return Solved3::unknown();
    }
    let moves_len = moves.len();
    let searched: FxHashSet<(i32, i32)> = moves.iter().copied().collect();
    let mut saw_unknown = false;
    for &(q, r) in &moves {
        let node_player = board.current_player;
        let diff = match board.apply_move_tracked(q, r) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let child = solve_3valued(board, depth_left - 1, budget, cfg, tt);
        let flipped = board.current_player != node_player;
        let rc = if flipped { child.outcome.negate() } else { child.outcome };
        if rc == Outcome::Win {
            let mut line = vec![(q, r)];
            if !flipped {
                line.extend(child.line.iter().copied());
            }
            board.undo_move(diff);
            return Solved3 { outcome: Outcome::Win, line };
        }
        board.undo_move(diff);
        if rc == Outcome::Unknown {
            saw_unknown = true;
        }
    }
    if saw_unknown {
        return Solved3::unknown();
    }
    let loss_complete =
        (in_check && moves_len < cfg.cand_cap) || moves_len >= board.legal_move_count();
    if loss_complete {
        tt.store_loss_proof(key, -MATE, 0, depth_left);
        return Solved3 { outcome: Outcome::Loss, line: Vec::new() };
    }
    if cfg.neighbor_dist.is_some() {
        for (q, r) in board.legal_moves() {
            if searched.contains(&(q, r)) {
                continue;
            }
            let node_player = board.current_player;
            let diff = match board.apply_move_tracked(q, r) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let child = solve_3valued(board, depth_left - 1, budget, cfg, tt);
            let flipped = board.current_player != node_player;
            let rc = if flipped { child.outcome.negate() } else { child.outcome };
            if rc == Outcome::Win {
                let mut line = vec![(q, r)];
                if !flipped {
                    line.extend(child.line.iter().copied());
                }
                board.undo_move(diff);
                return Solved3 { outcome: Outcome::Win, line };
            }
            board.undo_move(diff);
            if rc == Outcome::Unknown {
                return Solved3::unknown();
            }
        }
        tt.store_loss_proof(key, -MATE, 0, depth_left);
        return Solved3 { outcome: Outcome::Loss, line: Vec::new() };
    }
    Solved3::unknown()
}

// ── Tests ──────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Cell, Player};

    const WIN: Outcome = Outcome::Win;
    const LOSS: Outcome = Outcome::Loss;

    fn solver() -> super::super::TacticalSolver {
        // window_half=None for the core proof tests (offense-guard tested
        // separately); cand_cap matches solver.py.
        super::super::TacticalSolver::new(TacticalConfig { cand_cap: 40, window_half: None, neighbor_dist: None })
    }

    // ── Independent exhaustive oracle (port of solver.py::_brute_solve) ─────────
    // ALL legal moves, no TT, same AND-OR + flip-aware logic. Early-exits on
    // WIN so finding a defender's escape (the soundness refutation) is cheap.
    fn brute_solve(board: &mut Board, depth: i32, budget: &mut Budget) -> Outcome {
        if !budget.tick() {
            return Outcome::Unknown;
        }
        if board.check_win() {
            return if board.terminal_value_to_move() > 0.0 { WIN } else { LOSS };
        }
        let stm = board.current_player;
        if board.count_winning_moves(stm) >= 1 {
            return WIN;
        }
        if depth <= 0 {
            return Outcome::Unknown;
        }
        let mut saw_unknown = false;
        for (q, r) in board.legal_moves() {
            let node_player = board.current_player;
            let diff = match board.apply_move_tracked(q, r) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let raw = brute_solve(board, depth - 1, budget);
            let flipped = board.current_player != node_player;
            let rc = if flipped { raw.negate() } else { raw };
            board.undo_move(diff);
            if rc == WIN {
                return WIN;
            }
            if rc == Outcome::Unknown {
                saw_unknown = true;
            }
        }
        if saw_unknown { Outcome::Unknown } else { LOSS }
    }

    // Deterministic, dependency-free PRNG (matches board/mod.rs test style).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            // splitmix64
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn range(&mut self, lo: usize, hi: usize) -> usize {
            lo + (self.next() as usize) % (hi - lo + 1)
        }
    }

    /// Build the crossing-open-fives fork (port of solver.py::_build_fork):
    /// two crossing open-fives for P2 (attacker) centred at (3,3) -> 4 winning
    /// cells; P1 to move, mr=2 -> can block only 2 -> proven LOSS. Strict legal
    /// cadence (P1 opener, alternating 2-stone turns).
    fn build_fork() -> Board {
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1 opener
        let p2_order = [
            (3, 3), (2, 3), (4, 3), (1, 3), (5, 3), (3, 2), (3, 4), (3, 1), (3, 5),
        ];
        let p1_fillers = [
            (-3, -3), (-3, -2), (-2, -3), (-3, -4), (-4, -3), (-2, -2),
            (-4, -4), (-4, -2), (-2, -4), (-5, -3), (-3, -5), (-5, -4),
        ];
        let (mut p2i, mut p1i) = (0usize, 0usize);
        let mut turn: i32 = -1;
        let mut guard = 0;
        while p2i < p2_order.len() && guard < 50 {
            guard += 1;
            let mut placed = 0;
            if turn == -1 {
                while placed < 2 && p2i < p2_order.len() {
                    let (q, r) = p2_order[p2i];
                    b.apply_move(q, r).unwrap();
                    p2i += 1;
                    placed += 1;
                }
            } else {
                while placed < 2 && p1i < p1_fillers.len() {
                    let (q, r) = p1_fillers[p1i];
                    b.apply_move(q, r).unwrap();
                    p1i += 1;
                    placed += 1;
                }
            }
            turn *= -1;
        }
        // Normalise to P1-to-move (mr=2) if a P2 partial turn left P2 on move.
        let extra = [(-3, 8), (-3, 7), (-4, 8), (-4, 7)];
        let mut ei = 0;
        while b.current_player == Player::Two && ei < extra.len() {
            let (q, r) = extra[ei];
            b.apply_move(q, r).unwrap();
            ei += 1;
        }
        b
    }

    /// Compact double-threat: two parallel P2 open-fives -> 4 distinct winning
    /// cells; P1 to move, mr=2 -> blocks <=2 -> proven LOSS. `legal_move_radius`
    /// is shrunk to 2 so the EXHAUSTIVE brute oracle (no threat pruning)
    /// confirms the LOSS cheaply — the spread-out `build_fork` has a radius-5
    /// legal set (~300 cells) that makes the brute branching ~300^2 (the source
    /// of the slow full-tree confirmation). Both solver and brute see the same
    /// (small) legal set, so the comparison stays apples-to-apples.
    fn compact_double_threat(off_q: i32, off_r: i32, vertical: bool) -> Board {
        let mut stones: Vec<((i32, i32), Cell)> = Vec::new();
        for i in 0..5i32 {
            if vertical {
                stones.push(((off_q, off_r + i), Cell::P2));
                stones.push(((off_q + 2, off_r + i), Cell::P2));
            } else {
                stones.push(((off_q + i, off_r), Cell::P2));
                stones.push(((off_q + i, off_r + 2), Cell::P2));
            }
        }
        let mut b = static_board(&stones, Player::One, 2);
        b.set_legal_move_radius(2);
        b
    }

    /// NOT-IN-CHECK tactical position: two parallel P2 open-FOURS (length-4 runs,
    /// not 5) -> P2 has `threat_moves` (a 5th stone makes a win-in-1) but NO
    /// `winning_moves` yet, so P1 (to move, mr=2) is NOT in check. This is the
    /// not-in-check analogue of `compact_double_threat` and the surface the R3
    /// guard protects: the prune-symmetry premise (`in_check`) is FALSE here, so a
    /// LOSS may only be concluded when the candidate set is the full legal set.
    /// Radius 2 keeps the brute oracle cheap.
    fn compact_double_open_four(off_q: i32, off_r: i32, vertical: bool) -> Board {
        let mut stones: Vec<((i32, i32), Cell)> = Vec::new();
        for i in 0..4i32 {
            if vertical {
                stones.push(((off_q, off_r + i), Cell::P2));
                stones.push(((off_q + 2, off_r + i), Cell::P2));
            } else {
                stones.push(((off_q + i, off_r), Cell::P2));
                stones.push(((off_q + i, off_r + 2), Cell::P2));
            }
        }
        let mut b = static_board(&stones, Player::One, 2);
        b.set_legal_move_radius(2);
        b
    }

    /// Direct-construction static board (mirrors moves.rs `fwm_board`): set
    /// cells + bbox + turn-phase explicitly. Used for the off-window guard test
    /// where a far stone shifts the window center.
    fn static_board(stones: &[((i32, i32), Cell)], player: Player, mr: u8) -> Board {
        let mut b = Board::new();
        let (mut lq, mut hq, mut lr, mut hr) = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
        for &((q, r), c) in stones {
            b.cells.insert((q, r), c);
            lq = lq.min(q);
            hq = hq.max(q);
            lr = lr.min(r);
            hr = hr.max(r);
        }
        b.has_stones = true;
        b.min_q = lq;
        b.max_q = hq;
        b.min_r = lr;
        b.max_r = hr;
        b.cache_dirty.set(true);
        b.current_player = player;
        b.moves_remaining = mr;
        b.ply = stones.len() as u32;
        b
    }

    #[test]
    fn test1_immediate_win_is_win() {
        // P1 builds 0..4 on r=0 via strict legal cadence; P1 to move with an
        // immediate win available -> WIN (NOT a proven loss).
        let mut c = Board::new();
        for &(q, r) in &[
            (0, 0),
            (0, 9), (0, 8),
            (1, 0), (2, 0),
            (-3, 9), (-3, 8),
            (3, 0), (4, 0),
            (-1, 9), (-2, 9),
        ] {
            c.apply_move(q, r).unwrap();
        }
        assert_eq!(c.current_player, Player::One, "expected P1 to move");
        assert!(c.count_winning_moves(Player::One) >= 1, "P1 should have an immediate win");
        let r = solver().prove(&c, 20, 10_000);
        assert_eq!(r.result, WIN, "P1-with-immediate-win must be WIN, got {:?}", r.result);
        assert!(!r.line.is_empty(), "WIN must carry the move line");
        // line[0] must actually complete a 6.
        let mut c2 = c.clone();
        c2.apply_move(r.line[0].0, r.line[0].1).unwrap();
        assert!(c2.check_win(), "line[0] must complete 6, got {:?}", r.line[0]);
    }

    #[test]
    fn test2_quiet_position_not_loss() {
        // A quiet early position must NOT be a proven LOSS via the threat search.
        let mut e = Board::new();
        for &(q, r) in &[(0, 0), (3, 3), (3, 4), (0, 1), (1, 0)] {
            e.apply_move(q, r).unwrap();
        }
        let r = solver().prove(&e, 20, 5_000);
        assert_ne!(r.result, LOSS, "quiet position must not be a proven LOSS, got {:?}", r.result);
    }

    #[test]
    fn test4_fork_is_proven_loss() {
        // The crossing-open-fives fork (spread-out, radius-5 legal set): P1 to
        // move, 4 P2 winning cells, mr=2. Assert the (threat-pruned) solver
        // proves LOSS at two depths. The independent exhaustive brute
        // confirmation lives in `test5` on the COMPACT loss (radius-2) — the
        // spread fork's ~300-cell legal set makes the full brute tree ~300^2 and
        // needlessly slow (~60 s); the solver itself is fast here.
        let f = build_fork();
        assert_eq!(f.current_player, Player::One, "fork: expected P1 to move");
        assert_eq!(f.count_winning_moves(Player::Two), 4, "fork: expected 4 P2 threats");

        let rt = solver().prove(&f, 30, 300_000);
        let rm = solver().prove(&f, 12, 300_000);
        assert_eq!(rt.result, LOSS, "depth-30 must prove the fork LOSS, got {:?}", rt);
        assert_eq!(rm.result, LOSS, "depth-12 must prove the fork LOSS, got {:?}", rm);
    }

    #[test]
    fn test5_compact_loss_brute_confirmed() {
        // POSITIVE forced-loss detection, independently confirmed. Several varied
        // compact double-threats (offsets + orientations). For each: the solver
        // proves LOSS (true positive, threat-pruned), the exhaustive brute oracle
        // AGREES it is a LOSS (soundness), and the TSS is far cheaper than brute.
        let s = solver();
        let cases = [(0, 0, false), (-3, 4, false), (2, -2, true), (-6, -1, true)];
        for &(q, r, vert) in &cases {
            let b = compact_double_threat(q, r, vert);
            assert_eq!(
                b.count_winning_moves(Player::Two),
                4,
                "compact case {:?}: expected 4 P2 threats",
                (q, r, vert)
            );
            let res = s.prove(&b, 12, 200_000);
            assert_eq!(res.result, LOSS, "solver must prove compact LOSS, case {:?}", (q, r, vert));

            let mut bb = Budget::new(2_000_000);
            let brute = brute_solve(&mut b.clone(), 12, &mut bb);
            assert_eq!(brute, LOSS, "brute oracle disagrees (soundness), case {:?}", (q, r, vert));
            assert!(
                res.nodes < bb.nodes,
                "TSS ({}) not cheaper than brute ({}), case {:?}",
                res.nodes,
                bb.nodes,
                (q, r, vert)
            );
        }
    }

    #[test]
    fn soundness_fuzz_zero_false_loss() {
        // SOUNDNESS: the solver must NEVER claim LOSS on a position that is not a
        // forced loss. Every LOSS claim is cross-checked by the independent
        // exhaustive brute_solve; a brute WIN (defender escape) refutes the LOSS
        // -> unsound. Two streams feed the check so the LOSS path is genuinely
        // exercised (random near-terminal play alone yields ~0 forced losses):
        //   (A) random near-terminal positions  — the soundness CONTROL.
        //   (B) constructed compact double-threats — guarantee real LOSS claims.
        // Assert 0 false-LOSS AND that the fuzz is non-vacuous (claims > 0).
        let mut rng = Lcg(0x0D_D5_01_5E_12_34_56_78);
        // Stream (C) below adds NOT-IN-CHECK open-four doubles: the surface the R3
        // guard protects (no prune symmetry). NOTE this `solver()` is threat-only
        // (neighbor_dist=None), so the recall VERIFY is disabled and a not-in-check
        // position falls through to UNKNOWN — i.e. stream C exercises the
        // not-in-check candidate-generation SURFACE (asserting nic_checked > 0) and
        // cross-checks any LOSS it does claim, but it does NOT exercise a
        // verify-certified not-in-check LOSS. The verify path is covered separately
        // by the `verify_*` tests (which force it via cand_cap=1 truncation).
        let s = solver();
        let mut checked = 0usize;
        let mut bad = 0usize;
        let mut nic_checked = 0usize;

        let cross_check = |bd: &Board| -> (bool, bool) {
            // returns (claimed_loss, refuted_by_brute)
            let res = s.prove(bd, 20, 4_000);
            if res.result != LOSS {
                return (false, false);
            }
            let mut bb = Budget::new(40_000);
            (true, brute_solve(&mut bd.clone(), 12, &mut bb) == WIN)
        };

        // (A) random near-terminal control.
        for _ in 0..60 {
            let mut bd = Board::new();
            let mut ok = true;
            let plies = rng.range(8, 24);
            for _ in 0..plies {
                let lm = bd.legal_moves();
                if lm.is_empty() {
                    ok = false;
                    break;
                }
                let (q, r) = lm[rng.range(0, lm.len() - 1)];
                if bd.apply_move(q, r).is_err() || bd.check_win() {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            let (claimed, refuted) = cross_check(&bd);
            if claimed {
                checked += 1;
                if refuted {
                    bad += 1;
                }
            }
        }

        // (B) constructed double-threats over a grid of offsets/orientations —
        //     these reliably produce (true) LOSS claims to validate the path.
        for off_q in -8..=4i32 {
            for &(off_r, vert) in &[(0, false), (3, true), (-2, false)] {
                let bd = compact_double_threat(off_q, off_r, vert);
                let (claimed, refuted) = cross_check(&bd);
                if claimed {
                    checked += 1;
                    if refuted {
                        bad += 1;
                    }
                }
            }
        }

        // (C) constructed NOT-IN-CHECK open-four doubles. Each is a genuine
        //     not-in-check input (opp has no win-in-1); any LOSS claim is
        //     cross-checked, and the count proves the not-in-check surface ran.
        for off_q in -8..=4i32 {
            for &(off_r, vert) in &[(0, false), (3, true), (-2, false)] {
                let bd = compact_double_open_four(off_q, off_r, vert);
                assert_eq!(
                    bd.count_winning_moves(Player::Two),
                    0,
                    "stream C must be NOT-IN-CHECK (no P2 win-in-1), case {:?}",
                    (off_q, off_r, vert)
                );
                nic_checked += 1;
                let (claimed, refuted) = cross_check(&bd);
                if claimed {
                    checked += 1;
                    if refuted {
                        bad += 1;
                    }
                }
            }
        }

        assert_eq!(bad, 0, "SOUNDNESS: {bad}/{checked} LOSS claims refuted by exhaustive oracle");
        assert!(checked > 0, "soundness fuzz vacuous: no LOSS claims exercised");
        assert!(nic_checked > 0, "not-in-check surface not exercised (R3 guard untested)");
        eprintln!(
            "soundness fuzz: {checked} LOSS claims (all brute-confirmed), {nic_checked} not-in-check positions exercised"
        );
    }

    #[test]
    fn in_window_guard_suppresses_offwindow_win() {
        // P1 has 0..4 on r=0 (immediate win at (-1,0)/(5,0)); a far P2 stone at
        // (20,20) shifts the window center to ~(10,10) so the winning cell is
        // off-window (cheb > 9). Guard ON -> suppressed (UNKNOWN); OFF -> WIN.
        let mut stones: Vec<((i32, i32), Cell)> =
            (0..5).map(|q| ((q, 0), Cell::P1)).collect();
        stones.push(((20, 20), Cell::P2));
        let b = static_board(&stones, Player::One, 2);
        assert!(b.count_winning_moves(Player::One) >= 1, "P1 should have an immediate win");

        let off = super::super::TacticalSolver::new(TacticalConfig { cand_cap: 40, window_half: Some(9), neighbor_dist: None });
        let on = super::super::TacticalSolver::new(TacticalConfig { cand_cap: 40, window_half: None, neighbor_dist: None });

        let guarded = off.prove(&b, 8, 10_000);
        let unguarded = on.prove(&b, 8, 10_000);
        assert_eq!(unguarded.result, WIN, "no guard: must be WIN");
        assert!(super::super::is_off_window(&b, unguarded.line[0], 9), "test setup: win cell must be off-window");
        assert_eq!(guarded.result, Outcome::Unknown, "in-window guard must suppress the off-window WIN");
        assert!(guarded.line.is_empty(), "suppressed proof carries no line");
    }

    /// Build a position that is genuinely a P1 WIN but whose only winning move is
    /// a `threat_move` counter that `cand_cap` truncation drops:
    ///   - P2 cluster: a compact double-threat (4 P2 win-in-1 cells) = on its own
    ///     a forced P1 LOSS (see `test5`). This puts P1 IN CHECK.
    ///   - P1 line: 0..3 on r=0, far from the P2 cluster. P1 has NO win-in-1 (only
    ///     4 in a row), but (4,0) is a `threat_move`; at the root (mr=2) P1 plays
    ///     (4,0) then (5,0) -> 0..5 = six -> P1 WINS on its own turn.
    /// With a full candidate set the search finds (4,0) -> WIN. With `cand_cap=1`
    /// the must-block cells come first and (4,0) is truncated out, so the pruned
    /// search is forced down the losing block sequence and (UNGUARDED) concludes a
    /// FALSE LOSS. The R3 loss-completeness guard must refuse that LOSS (the
    /// candidate set was truncated: `in_check && moves_len == cand_cap`, and
    /// `moves_len < legal_move_count`) and report UNKNOWN instead.
    fn fork_with_p1_counter() -> Board {
        // P2 compact double-threat far from the origin (radius-2 legal set).
        let mut stones: Vec<((i32, i32), Cell)> = Vec::new();
        for i in 0..5i32 {
            stones.push(((10 + i, 10), Cell::P2));
            stones.push(((10 + i, 12), Cell::P2));
        }
        // P1 four-in-a-row on r=0 (the counter line): (4,0)+(5,0) completes six.
        for q in 0..4i32 {
            stones.push(((q, 0), Cell::P1));
        }
        let mut b = static_board(&stones, Player::One, 2);
        b.set_legal_move_radius(2);
        b
    }

    #[test]
    fn neighbor_dist_widens_not_in_check_candidates_to_full_legal() {
        // Z0-B mechanism: at a NOT-IN-CHECK node, `neighbor_dist=Some(d)` with d
        // covering the legal radius makes the candidate set the FULL legal set
        // (the quiet developmental moves the threat-only set omits). This is what
        // both raises the 8% ceiling AND lets the R3 guard prove not-in-check
        // LOSSes (moves_len >= legal_move_count). `None` stays threat-only.
        let b = compact_double_open_four(0, 0, false); // P2 open-fours, P1 to move
        let (stm, opp) = (Player::One, Player::Two);
        assert_eq!(b.count_winning_moves(opp), 0, "setup: P1 is NOT in check");
        let nlegal = b.legal_move_count();

        let threat_only = super::candidates(&b, stm, opp, 1000, None);
        let widened = super::candidates(&b, stm, opp, 1000, Some(5));

        assert!(
            threat_only.len() < nlegal,
            "threat-only set ({}) must be a strict subset of legal ({nlegal})",
            threat_only.len()
        );
        assert_eq!(
            widened.len(),
            nlegal,
            "neighbor_dist covering the radius must yield the full legal set"
        );
        // Widening is additive: every threat-only candidate is still present.
        for m in &threat_only {
            assert!(widened.contains(m), "widened set dropped threat candidate {m:?}");
        }
    }

    #[test]
    fn off_window_completing_stone_suppressed() {
        // §D-COHERENCE (the COMPLETING cell, not the first, is reachability-
        // relevant). The A1 override PLACES line[0] AND the cached completing
        // line[1]; both must be in-window. A P2 block at (-1,0) forces P1's win
        // rightward: line=[(4,0),(5,0)] — line[0]=(4,0) IN-window (cheb 3) but the
        // COMPLETING line[1]=(5,0) OFF-window (cheb 4) about center (1,0)/half 3.
        // The old first-stone-only guard PASSED this (line[0] in) and would drop
        // the off-window completing stone; the completing-stone guard suppresses
        // it. (With the current lex move-order line[0] is normally the outermost
        // stone so line[0] already catches it — the block engineers the in-window-
        // setup / off-window-completion case that net-policy ordering will make
        // common; the guard must be correct for it.)
        let stones = vec![
            ((0, 0), Cell::P1), ((1, 0), Cell::P1), ((2, 0), Cell::P1), ((3, 0), Cell::P1),
            ((-1, 0), Cell::P2),
        ];
        let mut b = static_board(&stones, Player::One, 2);
        b.set_legal_move_radius(3);

        let raw = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 40, window_half: None, neighbor_dist: None,
        });
        let ru = raw.prove(&b, 12, 80_000);
        assert_eq!(ru.result, WIN, "unguarded must find the rightward win");
        assert!(ru.line.len() >= 2, "expected a 2-stone win, got {:?}", ru.line);
        assert!(!super::super::is_off_window(&b, ru.line[0], 3), "setup: line[0] must be IN-window, got {:?}", ru.line[0]);
        assert!(super::super::is_off_window(&b, ru.line[1], 3), "setup: completing line[1] must be OFF-window, got {:?}", ru.line[1]);

        let guarded = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 40, window_half: Some(3), neighbor_dist: None,
        });
        assert_eq!(
            guarded.prove(&b, 12, 80_000).result,
            Outcome::Unknown,
            "an off-window COMPLETING stone must suppress the override"
        );
    }

    #[test]
    fn spread_multicluster_no_false_proof() {
        // IMMUNITY (measured, not asserted): SealBot's phantom mates came from a
        // flat [140][140]+70 array that OOBs past |coord|~63 and on multi-cluster
        // geometry. The native solver is HashMap/run-length based — no flat array,
        // no windowing/aliasing — so it must emit NO false proof in that exact
        // regime. Each verdict is cross-checked against the exhaustive brute oracle
        // at coord magnitudes > 63 and across disjoint clusters.
        let s = solver();

        // (a) a real forced P1 LOSS translated PAST the OOB boundary (coord 64-90).
        // The compact double-threat is a brute-CONFIRMED forced loss at the origin
        // (test5); proving the SAME verdict at |coord|>63 is the translation-
        // invariance check that rules out any flat-array/aliasing corruption (the
        // SealBot OOB failure mode) — cheap, no deep oracle needed.
        for &(oq, orr) in &[(70, 70), (-80, 5), (64, -88)] {
            let b = compact_double_threat(oq, orr, false);
            assert!(
                b.cells.keys().any(|&(q, r)| q.abs().max(r.abs()) > 63),
                "setup: cluster must exceed the |coord|>63 OOB boundary"
            );
            assert_eq!(
                s.prove(&b, 12, 400_000).result,
                LOSS,
                "native must prove the forced loss unchanged past the OOB boundary, coord {:?}",
                (oq, orr)
            );
        }

        // (b) a genuinely MULTI-CLUSTER, non-winning board at large coords (3
        //     disjoint 3-stone clusters, no 5-run anywhere). Kept within a BOUNDED
        //     bbox (the legal set is O(stones·ball) only when the bbox is bounded;
        //     scattering clusters 150 cells apart makes the brute oracle's per-node
        //     legal rebuild O(bbox) and pathological — a test artifact, not the
        //     solver's deploy regime). The solver must never fabricate a proof the
        //     oracle refutes.
        let multi: Vec<((i32, i32), Cell)> = vec![
            ((70, 70), Cell::P1), ((71, 70), Cell::P1), ((70, 71), Cell::P2),
            ((78, 72), Cell::P2), ((79, 72), Cell::P2), ((78, 73), Cell::P1),
            ((73, 78), Cell::P1), ((74, 78), Cell::P2), ((73, 79), Cell::P1),
        ];
        let mut mb = static_board(&multi, Player::One, 2);
        mb.set_legal_move_radius(1); // tight legal set -> the brute oracle stays cheap
        let res = s.prove(&mb, 6, 100_000);
        let mut bb = Budget::new(80_000);
        let brute = brute_solve(&mut mb.clone(), 6, &mut bb);
        if res.result == WIN {
            assert_ne!(brute, LOSS, "native FALSE WIN on multi-cluster board");
        }
        if res.result == LOSS {
            assert_ne!(brute, WIN, "native FALSE LOSS on multi-cluster board");
        }
    }

    #[test]
    fn widened_solver_stays_sound() {
        // SOUNDNESS of the quiet-move widening: with full neighbour coverage the
        // search visits not-in-check interior nodes with the WIDE candidate set
        // (the path the body adds). Its verdict must stay consistent with the
        // exhaustive oracle — a LOSS claim is brute-confirmed LOSS, and a forced-
        // loss position is NEVER reported a WIN. Non-vacuous (>=1 LOSS proven).
        let s = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 1000,
            window_half: None,
            neighbor_dist: Some(5),
        });
        let cases = [(0, 0, false), (-3, 4, false), (2, -2, true), (-6, -1, true)];
        let mut loss_claims = 0;
        for &(q, r, vert) in &cases {
            let b = compact_double_threat(q, r, vert);
            let res = s.prove(&b, 12, 1_000_000);
            assert_ne!(res.result, WIN, "widened FALSE WIN on a forced loss, case {:?}", (q, r, vert));
            if res.result == LOSS {
                loss_claims += 1;
                let mut bb = Budget::new(2_000_000);
                assert_eq!(
                    brute_solve(&mut b.clone(), 12, &mut bb),
                    LOSS,
                    "widened FALSE LOSS (brute disagrees), case {:?}",
                    (q, r, vert)
                );
            }
        }
        assert!(loss_claims > 0, "widened solver proved 0 LOSSes (vacuous) — raise budget/depth");
    }

    #[test]
    fn verify_recovers_truncated_win() {
        // Recall-preserving verify (Z0-C): the reduced candidate set drove the
        // search; when all reduced candidates lose, the dropped legal moves are
        // searched. Here cand_cap=1 truncates away P1's winning counter — the
        // verify must search the dropped set, find it, and return WIN (the same
        // position the bare R3 guard could only call UNKNOWN).
        let b = fork_with_p1_counter();
        let no_verify = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 1,
            window_half: None,
            neighbor_dist: None,
        });
        let verify = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 1,
            window_half: None,
            neighbor_dist: Some(2),
        });
        assert_eq!(no_verify.prove(&b, 20, 200_000).result, Outcome::Unknown, "no-verify: conservative UNKNOWN");
        let res = verify.prove(&b, 20, 400_000);
        assert_eq!(res.result, WIN, "verify must recover the truncated winning counter");
        // The recovered line's first move must actually win the position.
        assert!(!res.line.is_empty(), "WIN carries a line");
    }

    #[test]
    fn verify_certifies_truncated_loss() {
        // The other verify branch: a real forced loss whose reduced set was
        // truncated. The dropped legal moves are searched and ALSO all lose, so
        // the verify certifies the LOSS (brute-confirmed) where the bare guard,
        // unable to trust the truncated set, returned UNKNOWN.
        let b = compact_double_threat(0, 0, false);
        let no_verify = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 1,
            window_half: None,
            neighbor_dist: None,
        });
        let verify = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 1,
            window_half: None,
            neighbor_dist: Some(2),
        });
        assert_eq!(no_verify.prove(&b, 12, 400_000).result, Outcome::Unknown, "no-verify: conservative UNKNOWN");
        assert_eq!(verify.prove(&b, 12, 1_000_000).result, LOSS, "verify must certify the truncated LOSS");
        let mut bb = Budget::new(2_000_000);
        assert_eq!(brute_solve(&mut b.clone(), 12, &mut bb), LOSS, "brute confirms the LOSS (soundness)");
    }

    #[test]
    fn verify_path_is_sound_over_grid() {
        // SOUNDNESS of the recall verify across a grid: cand_cap=1 forces the
        // reduced set to truncate at EVERY node, so the LOSS conclusion always
        // routes through the full-legal verify. Every LOSS it certifies must be a
        // real LOSS (brute-confirmed); a forced loss is NEVER reported a WIN.
        let s = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 1,
            window_half: None,
            neighbor_dist: Some(2),
        });
        let cases = [(0, 0, false), (-3, 4, false), (2, -2, true), (-6, -1, true)];
        for &(q, r, vert) in &cases {
            let b = compact_double_threat(q, r, vert);
            let res = s.prove(&b, 12, 1_500_000);
            assert_ne!(res.result, WIN, "verify FALSE WIN on a forced loss, case {:?}", (q, r, vert));
            if res.result == LOSS {
                let mut bb = Budget::new(2_000_000);
                assert_eq!(
                    brute_solve(&mut b.clone(), 12, &mut bb),
                    LOSS,
                    "verify FALSE LOSS (brute disagrees), case {:?}",
                    (q, r, vert)
                );
            }
        }
    }

    // ── RED-TEAM adversarial soundness attacks (throwaway; #[ignore]-marked
    //    heavy ones run explicitly). Goal: produce a FALSE proof (solver WIN/LOSS
    //    contradicting the full-width brute oracle) or prove it cannot. ───────────

    /// RED-TEAM (a)+(c): the verify path on NOT-IN-CHECK positions with a small,
    /// truncating cand_cap + neighbor widening — the exact surface the handoff
    /// claims is sound. Run the VERIFY config over a grid of not-in-check open-four
    /// doubles AND in-check double threats, varying cand_cap (truncation stress)
    /// and neighbor_dist, cross-checking EVERY proof against the full-width brute
    /// oracle. Definitive contradictions:
    ///   solver LOSS  but brute WIN  => FALSE LOSS (a real defender escape exists)
    ///   solver WIN   but brute LOSS => FALSE WIN
    ///
    /// EXHAUSTIVE / SLOW (`#[ignore]`): the not-in-check verify is a full-width
    /// expansion (no alpha-beta yet — the perf layer is deferred), so the 96
    /// budget-1.5M solves run for minutes (≈2.4 h on the dev laptop, ~145 min on
    /// the vast 5080). This is the on-demand deep soundness sweep — it is the test
    /// that actually exercises a not-in-check ROOT LOSS certified by the verify
    /// (the surface the fast in-check truncation tests cover only by code-path).
    /// Run before promotion / on the perf box:
    ///   `cargo test --lib tactics::search::tests::redteam_verify_grid_no_false_proof -- --ignored`
    /// VERIFIED 0 false proofs on the vast 5080, 2026-06-29 (19/19, this sweep incl).
    #[test]
    #[ignore = "exhaustive full-width verify sweep — minutes; run on-demand (--ignored)"]
    fn redteam_verify_grid_no_false_proof() {
        let mut loss_claims = 0usize;
        let mut win_claims = 0usize;
        let mut nic_loss = 0usize;
        let mut brute_unknown_on_loss = 0usize;
        for &cap in &[1usize, 3] {
            for &nd in &[2i32] {
                let s = super::super::TacticalSolver::new(TacticalConfig {
                    cand_cap: cap,
                    window_half: None,
                    neighbor_dist: Some(nd),
                });
                for off_q in -4..=3i32 {
                    for &(off_r, vert) in &[(0, false), (3, true), (-2, false)] {
                        for builder in 0..2 {
                            let b = if builder == 0 {
                                compact_double_open_four(off_q, off_r, vert)
                            } else {
                                compact_double_threat(off_q, off_r, vert)
                            };
                            let in_check = b.count_winning_moves(Player::Two) >= 1;
                            let res = s.prove(&b, 12, 1_500_000);
                            if res.result == LOSS {
                                loss_claims += 1;
                                if !in_check {
                                    nic_loss += 1;
                                }
                                let mut bb = Budget::new(1_500_000);
                                let brute = brute_solve(&mut b.clone(), 12, &mut bb);
                                assert_ne!(
                                    brute, WIN,
                                    "FALSE LOSS: solver LOSS but brute finds an escape-to-WIN \
                                     (cap={cap}, nd={nd}, in_check={in_check}, case {:?})",
                                    (off_q, off_r, vert, builder)
                                );
                                if brute == Outcome::Unknown {
                                    brute_unknown_on_loss += 1;
                                }
                            } else if res.result == WIN {
                                win_claims += 1;
                                let mut bb = Budget::new(1_500_000);
                                let brute = brute_solve(&mut b.clone(), 12, &mut bb);
                                assert_ne!(
                                    brute, LOSS,
                                    "FALSE WIN: solver WIN but brute proves LOSS \
                                     (cap={cap}, nd={nd}, case {:?})",
                                    (off_q, off_r, vert, builder)
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(loss_claims > 0, "vacuous: no LOSS claims exercised");
        eprintln!(
            "redteam grid: {loss_claims} LOSS ({nic_loss} not-in-check, {brute_unknown_on_loss} \
             brute-unresolved), {win_claims} WIN — all brute-consistent, 0 false proofs"
        );
    }

    /// RED-TEAM (a) random stream: random COMPACT positions (radius-2 legal set so
    /// brute is exhaustive-cheap), VERIFY config with a truncating cand_cap. Every
    /// LOSS cross-checked against brute; a brute WIN refutes. Bidirectional (WIN
    /// claims checked too). The not-in-check counter proves the attack surface ran.
    ///
    /// EXHAUSTIVE / SLOW (`#[ignore]`): 250 verify-config solves + per-claim brute.
    /// On-demand companion to the grid sweep. VERIFIED 0 false proofs on vast 5080,
    /// 2026-06-29. Run: `... redteam_verify_random_compact_no_false_proof -- --ignored`.
    #[test]
    #[ignore = "exhaustive random verify sweep — run on-demand (--ignored)"]
    fn redteam_verify_random_compact_no_false_proof() {
        let mut rng = Lcg(0xBADC_0FFE_E0DD_F00D);
        let s = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 2,
            window_half: None,
            neighbor_dist: Some(2),
        });
        let mut loss_claims = 0usize;
        let mut nic_seen = 0usize;
        let mut win_claims = 0usize;
        let mut samples = 0usize;
        let mut attempt = 0usize;
        // Generate compact random positions: play random legal moves but keep the
        // board in a small coordinate box so the radius-2 legal set stays small.
        while samples < 250 && attempt < 4000 {
            attempt += 1;
            let mut bd = Board::new();
            let plies = rng.range(6, 18);
            let mut ok = true;
            for _ in 0..plies {
                // restrict to a compact box so brute stays cheap + dense tactics
                let lm: Vec<(i32, i32)> = bd
                    .legal_moves()
                    .into_iter()
                    .filter(|&(q, r)| q.abs() <= 3 && r.abs() <= 3)
                    .collect();
                if lm.is_empty() {
                    ok = false;
                    break;
                }
                let (q, r) = lm[rng.range(0, lm.len() - 1)];
                if bd.apply_move(q, r).is_err() || bd.check_win() {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            bd.set_legal_move_radius(2);
            if bd.legal_move_count() > 30 {
                continue; // keep brute exhaustive-cheap
            }
            samples += 1;
            let in_check = bd.count_winning_moves(bd.current_player.other()) >= 1;
            let res = s.prove(&bd, 12, 1_000_000);
            if res.result == LOSS {
                loss_claims += 1;
                if !in_check {
                    nic_seen += 1;
                }
                let mut bb = Budget::new(1_500_000);
                let brute = brute_solve(&mut bd.clone(), 12, &mut bb);
                assert_ne!(
                    brute, WIN,
                    "FALSE LOSS (random): solver LOSS but brute escapes-to-WIN, in_check={in_check}"
                );
            } else if res.result == WIN {
                win_claims += 1;
                let mut bb = Budget::new(1_500_000);
                let brute = brute_solve(&mut bd.clone(), 12, &mut bb);
                assert_ne!(brute, LOSS, "FALSE WIN (random): solver WIN but brute proves LOSS");
            }
        }
        eprintln!(
            "redteam random: {samples} compact positions, {loss_claims} LOSS ({nic_seen} \
             not-in-check), {win_claims} WIN — 0 false proofs"
        );
    }

    /// RED-TEAM (b): flip-aware negamax SIGN. A genuine 2-stone-turn forcing WIN
    /// (P1 has 0..3 on r=0; plays (4,0) then completes six). A flip-sign bug would
    /// mislabel this WIN as LOSS/UNKNOWN. Assert WIN, then REALIZE the returned
    /// same-turn line and confirm it produces a real 6-in-a-row (independent of the
    /// brute oracle, which shares the flip logic and could not catch a flip bug).
    #[test]
    fn redteam_flip_sign_two_stone_win_realized() {
        let stones: Vec<((i32, i32), Cell)> =
            (0..4).map(|q| ((q, 0), Cell::P1)).collect();
        let b = static_board(&stones, Player::One, 2);
        assert_eq!(b.count_winning_moves(Player::One), 0, "setup: no single-stone win (needs 2)");
        let res = solver().prove(&b, 12, 200_000);
        assert_eq!(res.result, WIN, "2-stone forcing win must be WIN (flip-sign), got {:?}", res.result);
        // Realize the same-turn line: both stones are P1's (mr=2 -> 1, not flipped),
        // so line carries P1's two placements. Replaying must complete a 6.
        assert!(res.line.len() >= 2, "same-turn win must carry both P1 stones, got {:?}", res.line);
        let mut c = b.clone();
        c.apply_move(res.line[0].0, res.line[0].1).unwrap();
        c.apply_move(res.line[1].0, res.line[1].1).unwrap();
        assert!(c.check_win(), "realized WIN line must produce a real 6, got line {:?}", res.line);
        // And the SAME position must never be called a LOSS by the verify config.
        let v = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 2,
            window_half: None,
            neighbor_dist: Some(2),
        });
        assert_ne!(v.prove(&b, 12, 500_000).result, LOSS, "winnable position must never be a LOSS");
    }

    /// RED-TEAM (d): budget exhaustion must NEVER manufacture a proof. A genuine
    /// not-in-check verify target starved of budget must return UNKNOWN, never a
    /// (possibly-false) LOSS/WIN. Sweep tiny budgets across the boundary.
    #[test]
    fn redteam_budget_exhaustion_no_false_proof() {
        let v = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 1,
            window_half: None,
            neighbor_dist: Some(2),
        });
        // A real forced LOSS (brute-confirmed) so the full-budget result is LOSS;
        // every STARVED result must be UNKNOWN (proof requires completing the search).
        let b = compact_double_threat(0, 0, false);
        let mut bb = Budget::new(2_000_000);
        assert_eq!(brute_solve(&mut b.clone(), 12, &mut bb), LOSS, "setup: truly a forced LOSS");
        // Find a budget large enough to prove it, then starve below that.
        let full = v.prove(&b, 12, 1_000_000);
        assert_eq!(full.result, LOSS, "full budget proves the LOSS");
        for budget in 1u64..=full.nodes.saturating_sub(1).min(300) {
            let r = v.prove(&b, 12, budget);
            assert_ne!(
                r.result, WIN,
                "budget {budget}: starved search manufactured a WIN"
            );
            // A LOSS is permitted ONLY if the search actually completed (not exhausted).
            if r.result == LOSS {
                assert!(
                    !r.budget_exhausted,
                    "budget {budget}: EXHAUSTED search still returned a LOSS proof (unsound)"
                );
            }
        }
    }

    #[test]
    fn neighbor_dist_does_not_widen_in_check_nodes() {
        // IN CHECK the threat-only set is already complete; widening would only
        // bloat it (and risk truncating a real block past cand_cap). The compact
        // double-threat (4 P2 win-in-1 cells) puts P1 in check; the candidate set
        // must be identical with and without neighbor_dist.
        let b = compact_double_threat(0, 0, false);
        let (stm, opp) = (Player::One, Player::Two);
        assert!(b.count_winning_moves(opp) >= 1, "setup: P1 IS in check");
        let plain = super::candidates(&b, stm, opp, 1000, None);
        let widened = super::candidates(&b, stm, opp, 1000, Some(5));
        assert_eq!(plain, widened, "in-check candidate set must not widen");
    }

    #[test]
    fn r3_guard_suppresses_truncated_false_loss() {
        // SOUNDNESS (R3): an incomplete candidate set must NEVER yield a LOSS.
        // Here `cand_cap=1` truncates away P1's winning counter; the unguarded
        // search would conclude a FALSE LOSS. The guard must report UNKNOWN.
        let b = fork_with_p1_counter();
        // Test-setup invariants: P1 is in check (4 P2 threats), has no win-in-1,
        // but the position is truly a WIN (full-legal brute finds the counter).
        assert_eq!(b.count_winning_moves(Player::Two), 4, "setup: 4 P2 threats (P1 in check)");
        assert_eq!(b.count_winning_moves(Player::One), 0, "setup: P1 has no immediate win");
        let mut bb = Budget::new(2_000_000);
        assert_eq!(brute_solve(&mut b.clone(), 12, &mut bb), WIN, "setup: position is truly a P1 WIN");

        let trunc = super::super::TacticalSolver::new(TacticalConfig { cand_cap: 1, window_half: None, neighbor_dist: None });
        let res = trunc.prove(&b, 20, 200_000);
        assert_ne!(
            res.result, LOSS,
            "R3: truncated candidate set must not yield a false LOSS (got {:?})",
            res.result
        );
    }

    // ── D-PFIT P2 increment 1: scored α-β + mate distance ───────────────────────

    /// Drive the scored α-β core directly (full root window) and return the
    /// verdict + raw score + PV. (The public `prove` surface exposes only the
    /// verdict; the score is the increment-1 deliverable under test.)
    fn run_scored(
        b: &Board,
        cfg: &TacticalConfig,
        depth: i32,
        budget: u64,
    ) -> (Outcome, i32, Vec<(i32, i32)>) {
        let mut board = b.clone();
        let mut bud = Budget::new(budget);
        let mut tt = super::super::tt::ProofTt::new();
        let mut ordering = super::OrderingState::new();
        let s = super::solve(&mut board, depth, 0, NEG_INF, POS_INF, &mut bud, cfg, &mut tt, &mut ordering);
        (outcome_of(s.score), s.score, s.line)
    }

    /// The pre-increment-1 3-valued reference oracle's verdict (the invariance
    /// target — α-β must never change it).
    fn run_3valued(b: &Board, cfg: &TacticalConfig, depth: i32, budget: u64) -> Outcome {
        let mut board = b.clone();
        let mut bud = Budget::new(budget);
        let mut tt = super::super::tt::ProofTt::new();
        super::solve_3valued(&mut board, depth, &mut bud, cfg, &mut tt).outcome
    }

    #[test]
    fn clamp_heuristic_never_reaches_proof_region() {
        // SOUNDNESS: a heuristic leaf can NEVER masquerade as a mate. Clamp pins
        // any eval (incl. ±∞-ish) strictly inside (-WIN_THRESHOLD, WIN_THRESHOLD).
        for &v in &[i32::MIN, -MATE, -WIN_THRESHOLD, -1, 0, 1, WIN_THRESHOLD, MATE, i32::MAX] {
            let c = super::clamp_heuristic(v);
            assert!(c.abs() < WIN_THRESHOLD, "clamp({v}) = {c} leaked into the proof region");
            assert_ne!(outcome_of(c), WIN, "clamped heuristic must never read as WIN");
            assert_ne!(outcome_of(c), LOSS, "clamped heuristic must never read as LOSS");
        }
    }

    #[test]
    fn scored_mate_distance_prefers_shorter_win() {
        // Mate-distance encoding: a SHORTER forced win scores higher. An immediate
        // (1-stone) win at the root is exactly `MATE - 0 = MATE`; a 2-stone forcing
        // win must score strictly less (it lands a ply deeper) but still proven.
        let cfg = TacticalConfig { cand_cap: 40, window_half: None, neighbor_dist: None };

        // Immediate win: P1 has 0..4 on r=0 (a single stone completes six).
        let imm: Vec<((i32, i32), Cell)> = (0..5).map(|q| ((q, 0), Cell::P1)).collect();
        let imm_b = static_board(&imm, Player::One, 2);
        assert!(imm_b.count_winning_moves(Player::One) >= 1, "setup: immediate win");
        let (o1, s1, _) = run_scored(&imm_b, &cfg, 12, 50_000);
        assert_eq!(o1, WIN, "immediate win must be WIN");
        assert_eq!(s1, MATE, "immediate win scores MATE - 0 = MATE, got {s1}");

        // 2-stone forcing win: P1 has 0..3 on r=0 (needs (4,0) then (5,0)).
        let two: Vec<((i32, i32), Cell)> = (0..4).map(|q| ((q, 0), Cell::P1)).collect();
        let two_b = static_board(&two, Player::One, 2);
        assert_eq!(two_b.count_winning_moves(Player::One), 0, "setup: no 1-stone win");
        let (o2, s2, _) = run_scored(&two_b, &cfg, 12, 200_000);
        assert_eq!(o2, WIN, "2-stone forcing win must be WIN");
        assert!(s2 >= WIN_THRESHOLD, "2-stone win must be proven, got {s2}");
        assert!(s2 < s1, "deeper mate must score lower: 2-stone {s2} !< immediate {s1}");
    }

    #[test]
    fn scored_loss_is_mate_magnitude_negative() {
        // A proven forced LOSS carries a mate-magnitude NEGATIVE score (the
        // mate-distance loss encoding), and the verdict matches the oracle.
        let cfg = TacticalConfig { cand_cap: 40, window_half: None, neighbor_dist: None };
        let b = compact_double_threat(0, 0, false);
        let (o, s, _) = run_scored(&b, &cfg, 12, 200_000);
        assert_eq!(o, LOSS, "compact double-threat is a forced LOSS");
        assert!(s <= -WIN_THRESHOLD, "LOSS must carry a mate-magnitude negative score, got {s}");
    }

    #[test]
    fn verdict_invariance_scored_matches_3valued() {
        // THE INVARIANCE GATE (red-team): the scored α-β `solve` must reproduce
        // the pre-change 3-valued core's verdict on EVERY test position — across
        // WIN / LOSS / UNKNOWN, threat-only and verify (neighbor_dist) configs,
        // and the truncating cand_cap=1 surfaces. α-β is pruning/ordering ONLY.
        type Case = (Board, TacticalConfig, i32, u64, &'static str);
        let cfg = |cand_cap, neighbor_dist| TacticalConfig {
            cand_cap,
            window_half: None,
            neighbor_dist,
        };
        let imm: Vec<((i32, i32), Cell)> = (0..5).map(|q| ((q, 0), Cell::P1)).collect();
        let two: Vec<((i32, i32), Cell)> = (0..4).map(|q| ((q, 0), Cell::P1)).collect();
        let mut quiet = Board::new();
        for &(q, r) in &[(0, 0), (3, 3), (3, 4), (0, 1), (1, 0)] {
            quiet.apply_move(q, r).unwrap();
        }
        let cases: Vec<Case> = vec![
            (static_board(&imm, Player::One, 2), cfg(40, None), 12, 50_000, "immediate-win"),
            (static_board(&two, Player::One, 2), cfg(40, None), 12, 200_000, "2-stone-win"),
            (quiet, cfg(40, None), 20, 20_000, "quiet-not-loss"),
            (build_fork(), cfg(40, None), 30, 300_000, "fork-loss-d30"),
            (build_fork(), cfg(40, None), 12, 300_000, "fork-loss-d12"),
            (compact_double_threat(0, 0, false), cfg(40, None), 12, 200_000, "compact-loss-a"),
            (compact_double_threat(-3, 4, false), cfg(40, None), 12, 200_000, "compact-loss-b"),
            (compact_double_threat(2, -2, true), cfg(40, None), 12, 200_000, "compact-loss-c"),
            (compact_double_open_four(0, 0, false), cfg(40, None), 12, 80_000, "open4-threatonly-unknown"),
            // verify (neighbor_dist) + truncating cand_cap surfaces:
            (fork_with_p1_counter(), cfg(1, None), 20, 200_000, "trunc-r3-unknown"),
            (fork_with_p1_counter(), cfg(1, Some(2)), 20, 400_000, "trunc-verify-win"),
            (compact_double_threat(0, 0, false), cfg(1, Some(2)), 12, 1_000_000, "trunc-verify-loss"),
            (compact_double_threat(0, 0, false), cfg(1, None), 12, 400_000, "trunc-noverify-unknown"),
        ];
        for (b, c, depth, budget, name) in cases {
            let (scored, _, _) = run_scored(&b, &c, depth, budget);
            let reference = run_3valued(&b, &c, depth, budget);
            assert_eq!(
                scored, reference,
                "VERDICT INVARIANCE BROKEN ({name}): scored α-β = {scored:?}, 3-valued oracle = {reference:?}"
            );
        }
    }

    #[test]
    fn scored_win_pv_is_realizable() {
        // α-β must not corrupt the winning PV (the A1 override plays line[0..2]).
        // The 2-stone forcing win's line must replay to a real 6-in-a-row.
        let cfg = TacticalConfig { cand_cap: 40, window_half: None, neighbor_dist: None };
        let two: Vec<((i32, i32), Cell)> = (0..4).map(|q| ((q, 0), Cell::P1)).collect();
        let b = static_board(&two, Player::One, 2);
        let (o, _, line) = run_scored(&b, &cfg, 12, 200_000);
        assert_eq!(o, WIN, "must be WIN");
        assert!(line.len() >= 2, "2-stone win carries both stones, got {line:?}");
        let mut c = b.clone();
        c.apply_move(line[0].0, line[0].1).unwrap();
        c.apply_move(line[1].0, line[1].1).unwrap();
        assert!(c.check_win(), "PV must realize a real 6, got {line:?}");
    }

    // ── D-PFIT P2 increment 4: PVS / LMR / aspiration + killers/history ─────────

    #[test]
    fn verdict_invariance_fuzz_scored_matches_3valued() {
        // REVIEW-REQUESTED HARDENING. The fixed 13-case `verdict_invariance_*` is
        // widened to a RANDOMIZED stream so the fail-soft mate-bound corners (PVS
        // null-window, LMR reduction, ordering permutation) are stressed, not just
        // the hand-picked cases. The scored α-β `solve` (PVS + LMR + killers /
        // history / TT ordering) is VERDICT-EXACT vs the plain 3-valued oracle —
        // they share the candidate set + budget, so:
        //   - they must NEVER contradict (one WIN, other LOSS) — the hard invariant;
        //   - when BOTH are conclusive they must be EQUAL;
        //   - any scored verdict is additionally brute-confirmed (compact => cheap).
        // (A scored-conclusive / oracle-UNKNOWN split is an allowed α-β pruning win,
        // never a soundness break — and is asserted brute-sound.)
        let mut rng = Lcg(0xF17E_55ED_2026_0629);
        // THREAT-ONLY configs: PVS + LMR + killers/history/TT ordering (the
        // increment-4 additions under test) ALL run on the threat-only path; the
        // recall VERIFY (neighbor_dist=Some) deliberately uses NO PVS/LMR (kept
        // full-window for exact recall) so this fuzz needs no verify config — and a
        // depth-deep full-width verify on a random non-loss board is a minutes-long
        // full-legal expansion, not the fast-suite budget. The verify path's
        // soundness is covered by `verify_path_is_sound_over_grid` + the #[ignore]
        // red-team sweeps. Varied cand_cap (incl. cand_cap=1 truncation) stresses
        // the PVS/LMR mate-bound corners + the R3 guard interaction.
        let cfgs = [(1usize, None), (2, None), (40, None)];
        let (mut win, mut loss, mut unknown, mut checked, mut exact_agree) = (0, 0, 0, 0, 0);

        let check = |b: &Board, cand: usize, nd: Option<i32>, depth: i32, budget: u64,
                     win: &mut i32, loss: &mut i32, unknown: &mut i32, checked: &mut i32, exact: &mut i32| {
            let cfg = TacticalConfig { cand_cap: cand, window_half: None, neighbor_dist: nd };
            let (scored, _, _) = run_scored(b, &cfg, depth, budget);
            let reference = run_3valued(b, &cfg, depth, budget);
            assert!(
                !(scored == WIN && reference == LOSS) && !(scored == LOSS && reference == WIN),
                "FUZZ CONTRADICTION: scored {scored:?} vs oracle {reference:?} (cand={cand}, nd={nd:?})"
            );
            if scored != Outcome::Unknown && reference != Outcome::Unknown {
                assert_eq!(scored, reference, "FUZZ INVARIANCE: both conclusive but unequal (cand={cand}, nd={nd:?})");
                *exact += 1;
            }
            // brute-confirm any scored verdict (the ultimate soundness gate). Boards
            // are radius-2 compact so the full-width brute terminates cheaply.
            if scored == LOSS || scored == WIN {
                let mut bb = Budget::new(200_000);
                let brute = brute_solve(&mut b.clone(), 12, &mut bb);
                if scored == LOSS {
                    assert_ne!(brute, WIN, "FUZZ FALSE LOSS: scored LOSS but brute escapes to WIN");
                } else {
                    assert_ne!(brute, LOSS, "FUZZ FALSE WIN: scored WIN but brute proves LOSS");
                }
            }
            match scored {
                Outcome::Win => *win += 1,
                Outcome::Loss => *loss += 1,
                Outcome::Unknown => *unknown += 1,
            }
            *checked += 1;
        };

        // (A) random COMPACT boards (tight ±2 box + radius-2 legal set so the
        //     full-width brute + oracle stay cheap), one random threat-only config
        //     each.
        let (mut samples, mut attempt) = (0, 0);
        while samples < 36 && attempt < 4000 {
            attempt += 1;
            let mut bd = Board::new();
            let mut ok = true;
            let plies = rng.range(5, 10);
            for _ in 0..plies {
                let lm: Vec<(i32, i32)> = bd
                    .legal_moves()
                    .into_iter()
                    .filter(|&(q, r)| q.abs() <= 2 && r.abs() <= 2)
                    .collect();
                if lm.is_empty() {
                    ok = false;
                    break;
                }
                let (q, r) = lm[rng.range(0, lm.len() - 1)];
                if bd.apply_move(q, r).is_err() || bd.check_win() {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            bd.set_legal_move_radius(2);
            if bd.legal_move_count() > 36 {
                continue;
            }
            samples += 1;
            let (cand, nd) = cfgs[rng.range(0, cfgs.len() - 1)];
            check(&bd, cand, nd, 6, 40_000, &mut win, &mut loss, &mut unknown, &mut checked, &mut exact_agree);
        }

        // (B) constructed double-threats (in-check + not-in-check) across every
        //     config — guarantee a stream of forced-LOSS verdicts under truncation.
        for off_q in -4..=3i32 {
            for &(off_r, vert) in &[(0, false), (3, true)] {
                for builder in 0..2 {
                    let b = if builder == 0 {
                        compact_double_threat(off_q, off_r, vert)
                    } else {
                        compact_double_open_four(off_q, off_r, vert)
                    };
                    let (cand, nd) = cfgs[rng.range(0, cfgs.len() - 1)];
                    check(&b, cand, nd, 8, 100_000, &mut win, &mut loss, &mut unknown, &mut checked, &mut exact_agree);
                }
            }
        }

        // (C) constructed WIN positions translated over a grid + every config —
        //     WIN-side coverage. P1 holds 0..4 on r=off (an open FIVE: an immediate
        //     win via the pre-candidate shortcut, so WIN is proven instantly for ANY
        //     cand_cap incl. truncation, and the brute returns WIN at its first
        //     count_winning_moves check — keeps this stream cheap + deterministic).
        //     Radius-2 capped so the rare truncation path can't expand a wide tree.
        for off in -4..=3i32 {
            let stones: Vec<((i32, i32), Cell)> =
                (0..5).map(|q| ((q, off), Cell::P1)).collect();
            let mut b = static_board(&stones, Player::One, 2);
            b.set_legal_move_radius(2);
            let (cand, nd) = cfgs[rng.range(0, cfgs.len() - 1)];
            check(&b, cand, nd, 8, 80_000, &mut win, &mut loss, &mut unknown, &mut checked, &mut exact_agree);
        }

        assert!(checked > 40, "fuzz too small ({checked} positions)");
        assert!(win > 0, "fuzz vacuous: no WIN verdict exercised");
        assert!(loss > 0, "fuzz vacuous: no LOSS verdict exercised");
        assert!(unknown > 0, "fuzz should include UNKNOWN positions (mate-bound corners)");
        eprintln!(
            "verdict-invariance fuzz: {checked} positions ({win} WIN, {loss} LOSS, {unknown} UNKNOWN), \
             {exact_agree} both-conclusive agreements, 0 contradictions, 0 brute-refuted verdicts"
        );
    }

    #[test]
    fn iterative_deepening_proves_short_mate_and_is_idempotent() {
        // The ID + aspiration root driver (public `prove`) must prove a forced mate
        // and — mate-early-stop — return the SAME verdict regardless of max_depth
        // (a deeper cap cannot change a proven mate). Also a generous cap must not
        // blow the node budget on the shallow proof (early-stop bounds the work).
        let s = solver();
        let b = compact_double_threat(0, 0, false); // forced P1 LOSS in a few plies
        let shallow = s.prove(&b, 6, 400_000);
        let deep = s.prove(&b, 40, 400_000);
        assert_eq!(shallow.result, LOSS, "ID must prove the short forced LOSS at a small cap");
        assert_eq!(deep.result, LOSS, "ID verdict idempotent under a deeper cap");
        assert!(
            deep.nodes < 400_000 && !deep.budget_exhausted,
            "mate-early-stop must prove the shallow mate well within budget (nodes={}, exhausted={})",
            deep.nodes,
            deep.budget_exhausted
        );
    }

    #[test]
    fn iterative_deepening_win_pv_is_realizable() {
        // ID/aspiration must not corrupt the root WIN PV (deploy override plays
        // line[0..2]). The 2-stone forcing win replayed must complete a real 6.
        let s = solver();
        let two: Vec<((i32, i32), Cell)> = (0..4).map(|q| ((q, 0), Cell::P1)).collect();
        let b = static_board(&two, Player::One, 2);
        let r = s.prove(&b, 20, 200_000);
        assert_eq!(r.result, WIN, "ID must prove the 2-stone forcing WIN");
        assert!(r.line.len() >= 2, "WIN carries both stones, got {:?}", r.line);
        let mut c = b.clone();
        c.apply_move(r.line[0].0, r.line[0].1).unwrap();
        c.apply_move(r.line[1].0, r.line[1].1).unwrap();
        assert!(c.check_win(), "ID WIN PV must realize a real 6, got {:?}", r.line);
    }

    // ── D-PFIT P2 increment 5: net-policy candidate ordering (the proof stays net-free) ──

    /// Drive the scored core with an OPTIONAL `PolicyPrior` wired into ordering.
    fn run_scored_pol(
        b: &Board,
        cfg: &TacticalConfig,
        depth: i32,
        budget: u64,
        policy: Option<Box<dyn super::super::ordering::PolicyPrior>>,
    ) -> (Outcome, Vec<(i32, i32)>) {
        let mut board = b.clone();
        let mut bud = Budget::new(budget);
        let mut tt = super::super::tt::ProofTt::new();
        let mut ordering = match policy {
            Some(p) => super::OrderingState::with_policy(p),
            None => super::OrderingState::new(),
        };
        let s = super::solve(&mut board, depth, 0, NEG_INF, POS_INF, &mut bud, cfg, &mut tt, &mut ordering);
        (outcome_of(s.score), s.line)
    }

    #[test]
    fn verdict_invariant_to_net_policy_ordering() {
        // Increment 5 — the CRITICAL property: wiring a net policy into candidate
        // ORDERING must NEVER change a WIN/LOSS verdict (the proof core reads the
        // net nowhere). An ADVERSARIAL prior (a deterministic per-move pseudo-random
        // value) maximally permutes the candidate order; the verdict + the WIN PV
        // realizability must be invariant vs the no-policy run — across WIN / LOSS /
        // UNKNOWN, threat-only + verify (neighbor_dist) configs, and cand_cap=1
        // truncation. (The proof core stays net-free; ordering is a permutation.)
        use super::super::ordering::PolicyPrior;
        struct Adversarial;
        impl PolicyPrior for Adversarial {
            fn prior(&self, mv: (i32, i32)) -> f32 {
                // splitmix-style hash of the coords -> a scattered value in [-1, 1].
                let mut z = (mv.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (mv.1 as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z ^= z >> 31;
                ((z & 0xFFFF) as f32) / 32768.0 - 1.0
            }
        }
        let cfg = |cand_cap, neighbor_dist| TacticalConfig { cand_cap, window_half: None, neighbor_dist };
        let imm: Vec<((i32, i32), Cell)> = (0..5).map(|q| ((q, 0), Cell::P1)).collect();
        let two: Vec<((i32, i32), Cell)> = (0..4).map(|q| ((q, 0), Cell::P1)).collect();
        let mut quiet = Board::new();
        for &(q, r) in &[(0, 0), (3, 3), (3, 4), (0, 1), (1, 0)] {
            quiet.apply_move(q, r).unwrap();
        }
        type Case = (Board, TacticalConfig, i32, u64, &'static str);
        let cases: Vec<Case> = vec![
            (static_board(&imm, Player::One, 2), cfg(40, None), 12, 50_000, "immediate-win"),
            (static_board(&two, Player::One, 2), cfg(40, None), 12, 200_000, "2-stone-win"),
            (quiet, cfg(40, None), 12, 40_000, "quiet-unknown"),
            (build_fork(), cfg(40, None), 12, 300_000, "fork-loss"),
            (compact_double_threat(0, 0, false), cfg(40, None), 12, 200_000, "compact-loss"),
            (compact_double_threat(-3, 4, false), cfg(40, None), 12, 200_000, "compact-loss-b"),
            (fork_with_p1_counter(), cfg(1, Some(2)), 20, 400_000, "trunc-verify-win"),
            (compact_double_threat(0, 0, false), cfg(1, Some(2)), 12, 1_000_000, "trunc-verify-loss"),
            (compact_double_threat(0, 0, false), cfg(1, None), 12, 400_000, "trunc-noverify-unknown"),
        ];
        for (b, c, depth, budget, name) in cases {
            let (base, base_line) = run_scored_pol(&b, &c, depth, budget, None);
            let (perm, perm_line) = run_scored_pol(&b, &c, depth, budget, Some(Box::new(Adversarial)));
            assert_eq!(base, perm, "net-policy ordering CHANGED the verdict ({name}): {base:?} -> {perm:?}");
            // A WIN PV must still realize a real win under the reordered search.
            if perm == WIN {
                let mut bd = b.clone();
                for &(q, r) in perm_line.iter().take(2) {
                    bd.apply_move(q, r).unwrap();
                }
                assert!(bd.check_win(), "policy-reordered WIN PV must realize a real 6 ({name})");
            }
            let _ = &base_line;
        }
    }
}

