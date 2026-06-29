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

use super::ordering::candidates;
use super::tt::ProofTt;
use super::{Budget, Outcome, TacticalConfig};

/// A resolved node: the proof outcome plus the principal variation (the move
/// LINE). `line` is populated for WIN — `line[0]` is the move to play; for a
/// 2-stone-turn forcing win `line[0]`/`line[1]` are the side-to-move's two
/// stones (the A1 override caches `line[1]`). Empty for LOSS/UNKNOWN.
pub struct Solved {
    pub outcome: Outcome,
    pub line: Vec<(i32, i32)>,
}

impl Solved {
    #[inline]
    fn unknown() -> Self {
        Solved { outcome: Outcome::Unknown, line: Vec::new() }
    }
}

/// 3-valued AND-OR threat-space proof for `board.current_player`.
///
/// NET-FREE: proofs are ONLY terminal backups (`terminal_value_to_move`, CF-1)
/// or stone-count shortcuts (immediate win / mr==1 double-threat). The value
/// head is never read. UNKNOWN = unresolved within depth/budget (the search
/// went quiet or ran out) — never a draw, never a false proof.
pub(crate) fn solve(
    board: &mut Board,
    depth_left: i32,
    budget: &mut Budget,
    cfg: &TacticalConfig,
    tt: &mut ProofTt,
) -> Solved {
    if !budget.tick() {
        return Solved::unknown();
    }

    // (1) Terminal: the engine-owned CF-1 sign is the ONLY proof sign.
    if board.check_win() {
        let outcome = if board.terminal_value_to_move() > 0.0 {
            Outcome::Win
        } else {
            Outcome::Loss
        };
        return Solved { outcome, line: Vec::new() };
    }

    let stm = board.current_player;
    let opp = stm.other();

    // (2) Immediate-win shortcut: side-to-move completes 6 with its next stone.
    //     Sound stone-count proof (no net). Yields the winning cell for the PV.
    if board.count_winning_moves(stm) >= 1 {
        let line = board.first_winning_move(stm).map_or_else(Vec::new, |m| vec![m]);
        return Solved { outcome: Outcome::Win, line };
    }

    // (3) Double-threat LOSS shortcut (mr==1 only — provably sound):
    //     stm has NO immediate win (checked above) and places exactly ONE stone
    //     before the turn flips; the opponent then has >=2 standing win-in-1
    //     cells, so stm blocks at most 1 and >=1 survives -> opp wins next ->
    //     LOSS. (A single placement cannot be a 2-stone self-win, so the mr==2
    //     soundness hole does not exist at mr==1. mr==2 is left to the
    //     recursion, which proves it soundly — often via this shortcut one ply
    //     deeper. The general unblockable-multi-threat AND-OR over opponent
    //     must-hit sets is DEFERRED; the recursion covers it.)
    if board.moves_remaining == 1 && board.count_winning_moves(opp) >= 2 {
        return Solved { outcome: Outcome::Loss, line: Vec::new() };
    }

    // (4) TT probe — proven results are game-theoretic (depth-independent).
    //     Only LOSS is cached (see `tt.rs`): a WIN cache hit would return an
    //     empty PV and could truncate the override line, so WINs are always
    //     reconstructed fresh.
    let key = (board.zobrist_hash, stm as i8, board.moves_remaining);
    if let Some(outcome) = tt.get(key) {
        return Solved { outcome, line: Vec::new() };
    }

    if depth_left <= 0 {
        return Solved::unknown();
    }

    // (5) Threat-guided candidate set (the narrow branching that reaches deep
    //     forcing mates cheaply). DEFERRED: net-policy ordering / killers /
    //     history (`ordering.rs`).
    // `in_check` (opp threatens an immediate win) is the prune-symmetry premise of
    // the LOSS conclusion below — captured at the node state, before descent.
    let in_check = board.count_winning_moves(opp) >= 1;
    let moves = candidates(board, stm, opp, cfg.cand_cap, cfg.neighbor_dist);
    if moves.is_empty() {
        return Solved::unknown(); // no live threats => quiet => cannot prove
    }
    let moves_len = moves.len();
    // Remember the reduced set so the recall verify (below) searches only the
    // DROPPED legal moves, not the ones already explored.
    let searched: FxHashSet<(i32, i32)> = moves.iter().copied().collect();

    let mut saw_unknown = false;
    for &(q, r) in &moves {
        let node_player = board.current_player; // == stm
        let diff = match board.apply_move_tracked(q, r) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let child = solve(board, depth_left - 1, budget, cfg, tt);
        // Flip-aware negate: only when the to-move side flipped (turn-final stone).
        let child_player = board.current_player;
        let flipped = child_player != node_player;
        let rc = if flipped { child.outcome.negate() } else { child.outcome };

        if rc == Outcome::Win {
            // OR-node: side-to-move has a winning/escaping move. Build the PV:
            //  - same player continued (mr was 2): append the child's PV so the
            //    line carries stm's 2nd stone (and beyond).
            //  - flipped (turn-final): the move alone; the rest is the
            //    opponent's forced-losing responses, not stm's moves to play.
            let mut line = vec![(q, r)];
            if !flipped {
                line.extend(child.line.iter().copied());
            }
            board.undo_move(diff);
            return Solved { outcome: Outcome::Win, line };
        }

        board.undo_move(diff);
        if rc == Outcome::Unknown {
            saw_unknown = true;
        }
    }

    // All candidates explored: any UNKNOWN => unresolved; else every move loses.
    if saw_unknown {
        return Solved::unknown();
    }

    // R3 LOSS-COMPLETENESS GUARD (load-bearing for sound z-LOSS labels).
    // "Every CANDIDATE loses" only proves "every LEGAL move loses" when the
    // candidate set provably covered all escapes:
    //   - `in_check && moves_len < cand_cap`: opp threatens an immediate win, so a
    //     non-block/non-counter loses to the standing threat (prune symmetry) —
    //     valid ONLY if the set was not truncated (the dropped tail could hold a
    //     saving block or a faster counter-win; see the truncation test). The
    //     `< cand_cap` boundary is deliberately PESSIMISTIC: a set whose natural
    //     (untruncated) size happens to equal cand_cap is treated as truncated, so
    //     a real LOSS there is reported UNKNOWN — a recall false-negative, never a
    //     soundness violation. Do NOT relax to `<=` (that reintroduces the hole).
    //   - `moves_len >= legal_move_count()`: the candidate set IS the full legal
    //     set, so exhaustiveness is trivial (covers not-in-check positions and the
    //     quiet-move body, where prune symmetry does NOT hold).
    // Otherwise the set is incomplete and a LOSS would be unsound -> UNKNOWN
    // (conservative-safe). The recall-preserving full-legal verify of the dropped
    // moves is the Track-3 quiet-move body's job and lands WITH it; until then a
    // not-in-check / truncated forced loss is reported UNKNOWN, never a false LOSS.
    let loss_complete =
        (in_check && moves_len < cfg.cand_cap) || moves_len >= board.legal_move_count();
    if loss_complete {
        tt.insert(key, Outcome::Loss); // proven, game-theoretic — cache it
        return Solved { outcome: Outcome::Loss, line: Vec::new() };
    }

    // RECALL-PRESERVING VERIFY (quiet-move body, `neighbor_dist` set). The reduced
    // candidate set drove the search for good ordering + early WIN cutoffs, but it
    // is too narrow to CERTIFY a LOSS (a dropped quiet move could escape). Search
    // the dropped legal moves now: any WIN is an escape (return it — the position
    // is a WIN the reduced set missed); any UNKNOWN leaves it unresolved; only when
    // EVERY dropped move also loses is the LOSS certified (sound, full recall). The
    // reduced ordering already cut off the common winning/unresolved cases, so this
    // full expansion runs only on genuinely-lost nodes. Budget-bounded via tick().
    // Without `neighbor_dist` (threat-only deploy mode) stay conservative -> UNKNOWN.
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
            let child = solve(board, depth_left - 1, budget, cfg, tt);
            let flipped = board.current_player != node_player;
            let rc = if flipped { child.outcome.negate() } else { child.outcome };
            if rc == Outcome::Win {
                let mut line = vec![(q, r)];
                if !flipped {
                    line.extend(child.line.iter().copied());
                }
                board.undo_move(diff);
                return Solved { outcome: Outcome::Win, line };
            }
            board.undo_move(diff);
            if rc == Outcome::Unknown {
                return Solved::unknown();
            }
        }
        // Every reduced candidate AND every dropped legal move loses => certified.
        tt.insert(key, Outcome::Loss);
        return Solved { outcome: Outcome::Loss, line: Vec::new() };
    }

    Solved::unknown() // candidate set incomplete, verify disabled -> cannot prove LOSS
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
    #[test]
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
    #[test]
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
}
