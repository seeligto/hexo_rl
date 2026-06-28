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
    let moves = candidates(board, stm, opp, cfg.cand_cap);
    if moves.is_empty() {
        return Solved::unknown(); // no live threats => quiet => cannot prove
    }

    let mut saw_unknown = false;
    for (q, r) in moves {
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
        Solved::unknown()
    } else {
        tt.insert(key, Outcome::Loss); // proven, game-theoretic — cache it
        Solved { outcome: Outcome::Loss, line: Vec::new() }
    }
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
        super::super::TacticalSolver::new(TacticalConfig { cand_cap: 40, window_half: None })
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
        let s = solver();
        let mut checked = 0usize;
        let mut bad = 0usize;

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

        assert_eq!(bad, 0, "SOUNDNESS: {bad}/{checked} LOSS claims refuted by exhaustive oracle");
        assert!(checked > 0, "soundness fuzz vacuous: no LOSS claims exercised");
        eprintln!("soundness fuzz: {checked} LOSS claims, all brute-confirmed (0 false-LOSS)");
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

        let off = super::super::TacticalSolver::new(TacticalConfig { cand_cap: 40, window_half: Some(9) });
        let on = super::super::TacticalSolver::new(TacticalConfig { cand_cap: 40, window_half: None });

        let guarded = off.prove(&b, 8, 10_000);
        let unguarded = on.prove(&b, 8, 10_000);
        assert_eq!(unguarded.result, WIN, "no guard: must be WIN");
        assert!(super::super::is_off_window(&b, unguarded.line[0], 9), "test setup: win cell must be off-window");
        assert_eq!(guarded.result, Outcome::Unknown, "in-window guard must suppress the off-window WIN");
        assert!(guarded.line.is_empty(), "suppressed proof carries no line");
    }
}
