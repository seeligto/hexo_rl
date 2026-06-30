//! Candidate generation + ordering for the threat-space proof.
//!
//! FOUNDATION: the threat-guided narrow candidate set (port of
//! `solver.py::_candidates`) — keeps branching ~4-15 so deep FORCING mates are
//! reachable cheaply. Built on the V4-fuzzed engine primitives
//! (`winning_moves`/`threat_moves`), in-engine so each call is one pass over the
//! legal set, not a clone per cell.
//!
//! # Best-first pipeline (`NATIVE_RUST_SOLVER_design.md` §3.4)
//!   1. TT best move                          <- present (`order_moves`)
//!   2. winning_moves (immediate)             <- present (via in-check must_block)
//!   3. threat_moves + responses (forcing)    <- present (`candidates`)
//!   4. NET-POLICY prior (the learned ordering, the SealBot-lacks lever) <- present
//!      (`PolicyPrior`, INERT/OFF by default — D-TACTICAL gate; post-training)
//!   5. killers, then history                 <- present (`OrderingState`)
//!   6. static `_move_delta` tie-break (eval.rs 3^6 table) <- DEFERRED (eval.rs)
//! Net-policy ordering only ACCELERATES (earlier cutoffs); the threat
//! enumeration here GUARANTEES a 0-prior refuter is still searched (T0: 67% of
//! refuters have ~0 deploy prior). Ordering never affects soundness.
//!
//! Crucially this is the THREAT-ONLY set = the measured **8% ceiling**: it omits
//! the QUIET developmental moves that start most mates. Adding the quiet-move
//! alpha-beta body (the hard, load-bearing part) is its own focused effort.

use fxhash::{FxHashMap, FxHashSet};

use crate::board::{Board, Player};

/// Learned move-ordering prior (D-PFIT P2 increment 5) — the AlphaZero ingredient
/// SealBot lacks (`NATIVE_RUST_SOLVER_design.md` §3.4 step 4). A higher prior
/// ⇒ the move is tried earlier (earlier α-β cutoffs).
///
/// # SOUNDNESS — ORDERING ONLY, NEVER A PROOF
/// The proof core (`search::solve`) reads the net NOWHERE: a `PolicyPrior` enters
/// only through `order_moves`, which is a strict PERMUTATION of the
/// already-generated candidate set. It cannot add/drop a candidate (so the
/// LOSS-completeness guard is unchanged) and cannot change a verdict — the
/// threat/forcing enumeration still SEARCHES every candidate regardless of prior
/// (the T0 "67% of refuters have ~0 prior" guarantee). `verdict_invariant_to_net_policy_ordering`
/// asserts the WIN/LOSS verdict is invariant to an adversarial prior.
pub(crate) trait PolicyPrior {
    /// Ordering prior for `mv` (higher ⇒ searched earlier). Ordering hint ONLY.
    fn prior(&self, mv: (i32, i32)) -> f32;
}

/// Per-search move-ordering state (D-PFIT P2 increment 4 + 5): killer moves (per
/// ply) + a history-heuristic table + an OPTIONAL learned policy prior. ORDERING
/// ONLY — never read as a proof. Reordering the candidate set changes neither the
/// SET (the LOSS-completeness guard reads `moves_len` / `legal_move_count`, both
/// order-independent) nor any verdict; the scored-vs-3valued invariance fuzz +
/// `verdict_invariant_to_net_policy_ordering` assert exactly this.
pub(crate) struct OrderingState {
    /// Two killer moves per ply (moves that caused a β / WIN cutoff at that ply).
    killers: Vec<[Option<(i32, i32)>; 2]>,
    /// History bonus per move — accumulates `depth²` on each cutoff.
    history: FxHashMap<(i32, i32), i32>,
    /// Learned net-policy prior — INERT (`None`) BY DEFAULT. Per D-TACTICAL the
    /// current blind net policy-guides 0/14 (chicken-and-egg), so the hook is built
    /// but GATED OFF: it is a post-training search speedup, not active now. The
    /// production solver (`prove`) always uses `new()` (net-free), preserving the
    /// "proof core reads the net nowhere" invariant.
    policy: Option<Box<dyn PolicyPrior>>,
}

impl OrderingState {
    pub(crate) fn new() -> Self {
        OrderingState { killers: Vec::new(), history: FxHashMap::default(), policy: None }
    }

    /// Build with a net-policy prior WIRED into candidate ordering (the post-
    /// training speedup). The prior orders candidates only; the proof core stays
    /// net-free. Unused by `prove` today (the gate is OFF by default — see the
    /// field note); exercised by `verdict_invariant_to_net_policy_ordering`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn with_policy(policy: Box<dyn PolicyPrior>) -> Self {
        OrderingState { killers: Vec::new(), history: FxHashMap::default(), policy: Some(policy) }
    }

    #[inline]
    fn killers_at(&self, ply: i32) -> [Option<(i32, i32)>; 2] {
        self.killers.get(ply.max(0) as usize).copied().unwrap_or([None, None])
    }

    /// Record a cutoff move: refresh the ply's killer slots + add a depth-weighted
    /// history bonus. Called on the WIN / β cutoff in `search::solve`.
    pub(crate) fn record_cutoff(&mut self, ply: i32, mv: (i32, i32), depth_left: i32) {
        let p = ply.max(0) as usize;
        if self.killers.len() <= p {
            self.killers.resize(p + 1, [None, None]);
        }
        let k = &mut self.killers[p];
        if k[0] != Some(mv) {
            k[1] = k[0];
            k[0] = Some(mv);
        }
        let d = depth_left.max(0);
        *self.history.entry(mv).or_insert(0) += (d * d).max(1);
    }
}

/// Best-first PERMUTATION of an already-generated candidate set (D-PFIT P2
/// increment 4 + 5). Tier order: TT best move, then the two killers, then the
/// quiet moves ordered by (history bonus + net-policy prior); ties keep
/// `candidates`' deterministic order (stable sort). NEVER adds or drops a move —
/// the set (hence every proof conclusion) is invariant; only the search ORDER
/// changes, for earlier α-β cutoffs. The net policy (when present) nudges ONLY the
/// tier-0 (quiet) ordering — the forcing/threat moves are still all searched.
pub(crate) fn order_moves(
    moves: &mut [(i32, i32)],
    tt_move: Option<(i32, i32)>,
    ply: i32,
    state: &OrderingState,
) {
    let killers = state.killers_at(ply);
    // Sort key = (tier, intra-tier score). Tier dominates so TT/killers always
    // lead; within the quiet tier, history + the optional net-policy prior decide.
    let key = |m: &(i32, i32)| -> (i64, i64) {
        let tier = if Some(*m) == tt_move {
            3
        } else if Some(*m) == killers[0] {
            2
        } else if Some(*m) == killers[1] {
            1
        } else {
            0
        };
        let hist = state.history.get(m).copied().unwrap_or(0) as i64;
        // Net-policy prior (INERT unless a policy is wired). Scaled to integer units
        // and ADDED to history — a tie-break that reorders quiet moves only; it can
        // never promote a move across a tier, so TT/killers/forcing order holds.
        let pol = match &state.policy {
            Some(p) => (p.prior(*m).clamp(-1.0, 1.0) * 1_000_000.0) as i64,
            None => 0,
        };
        (tier, hist.saturating_add(pol))
    };
    // Stable sort by descending key — equal-key moves keep `candidates()` order.
    moves.sort_by(|a, b| key(b).cmp(&key(a)));
}

/// Threat-guided candidate set for the side-to-move `stm` (`solver.py::_candidates`).
///
/// - `stm` in check (opponent threatens an immediate win): block the threat(s)
///   (`winning_moves(opp)` — the must-hit cells) ∪ `stm`'s counter-threats. A
///   quiet move loses to the standing threat, so omitting non-responses cannot
///   turn a real escape into a false LOSS.
/// - `stm` not in check: `stm`'s threat-creating moves (attack) ∪ the
///   opponent's threat-creating cells (defensive pre-emption). Restricting to
///   the threat region is the standard TSS prune — guarded by the soundness
///   fuzz.
///
/// Deterministic ordering: must-block (sorted) first, then threats in
/// legal-sorted order. Capped at `cand_cap` (must-block kept first).
pub(crate) fn candidates(
    board: &Board,
    stm: Player,
    opp: Player,
    cand_cap: usize,
    neighbor_dist: Option<i32>,
) -> Vec<(i32, i32)> {
    let must_block = board.winning_moves(opp);
    if !must_block.is_empty() {
        // IN CHECK: the threat-only set (blocks ∪ counter-threats) is already
        // complete (a non-response loses to the standing threat) — NOT widened.
        let mut seen: FxHashSet<(i32, i32)> = must_block.iter().copied().collect();
        let mut out = must_block;
        for m in board.threat_moves(stm) {
            if seen.insert(m) {
                out.push(m);
            }
        }
        out.truncate(cand_cap);
        return out;
    }

    // NOT IN CHECK. Threat-creating moves first (forcing — searched regardless of
    // any prior, the 0-prior-refuter guarantee), then — under the quiet-move body
    // — the developmental neighbour cells the threat-only set omits.
    let mut seen: FxHashSet<(i32, i32)> = FxHashSet::default();
    let mut out: Vec<(i32, i32)> = Vec::new();
    for m in board.threat_moves(stm).into_iter().chain(board.threat_moves(opp)) {
        if seen.insert(m) {
            out.push(m);
        }
    }

    // Quiet-move widening (Track 3, the lever past the 8% ceiling). Append every
    // empty legal cell within cheb-distance `d` of a stone, in sorted order
    // (deterministic). When `d` covers the legal radius the set becomes the FULL
    // legal set, so a not-in-check LOSS satisfies the R3 guard's exhaustiveness
    // branch (`moves_len >= legal_move_count`) and is proven soundly. Quiet cells
    // come AFTER threats so forcing moves are still ordered first.
    if let Some(d) = neighbor_dist {
        for c in board.legal_moves() {
            if !seen.contains(&c)
                && board
                    .cells
                    .keys()
                    .any(|&(sq, sr)| (c.0 - sq).abs().max((c.1 - sr).abs()) <= d)
            {
                seen.insert(c);
                out.push(c);
            }
        }
    }

    out.truncate(cand_cap);
    out
}

// ── Tests ──────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(mut v: Vec<(i32, i32)>) -> Vec<(i32, i32)> {
        v.sort();
        v
    }

    #[test]
    fn order_moves_is_a_strict_permutation() {
        // SOUNDNESS-CRITICAL: ordering must never add/drop a candidate (the
        // LOSS-completeness guard reads the SET). Reordering preserves the multiset.
        let original = vec![(0, 0), (1, 0), (2, 0), (3, 3), (-1, -1), (5, 2), (7, 7), (-4, 1)];
        let mut state = OrderingState::new();
        state.record_cutoff(2, (5, 2), 7); // history bonus for (5,2)
        state.record_cutoff(2, (7, 7), 3); // killer[0] at ply 2
        let mut moves = original.clone();
        order_moves(&mut moves, Some((3, 3)), 2, &state);
        assert_eq!(moves.len(), original.len(), "ordering changed the candidate count");
        assert_eq!(sorted(moves.clone()), sorted(original.clone()), "ordering is not a permutation of the set");
    }

    #[test]
    fn order_moves_prioritizes_tt_then_killers_then_history() {
        // The TT best move leads, then killers, then a history-bonus move, then the
        // rest in `candidates()` order (stable).
        let original = vec![(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)];
        let mut state = OrderingState::new();
        state.record_cutoff(0, (5, 0), 6); // killer[0]
        state.record_cutoff(0, (1, 0), 2); // killer[0] now (1,0); (5,0) -> killer[1]
        state.record_cutoff(1, (3, 0), 8); // history bonus only (different ply's killer)
        let mut moves = original.clone();
        order_moves(&mut moves, Some((4, 0)), 0, &state);
        assert_eq!(moves[0], (4, 0), "TT best move must lead");
        assert_eq!(moves[1], (1, 0), "killer[0] next");
        assert_eq!(moves[2], (5, 0), "killer[1] next");
        // remaining must still be the full set (permutation).
        assert_eq!(sorted(moves.clone()), sorted(original), "permutation preserved");
    }

    #[test]
    fn empty_killers_history_keeps_candidate_order() {
        // With no ordering signal the set is returned in its original (candidates)
        // order — a stable no-op permutation.
        let original = vec![(0, 0), (1, 0), (2, 0), (3, 3)];
        let state = OrderingState::new();
        let mut moves = original.clone();
        order_moves(&mut moves, None, 0, &state);
        assert_eq!(moves, original, "no ordering signal must leave the order unchanged");
    }

    #[test]
    fn net_policy_reorders_quiet_tier_but_never_above_tt_or_killers() {
        // Increment 5: a `PolicyPrior` orders the QUIET (tier-0) moves only — it can
        // never promote a move above the TT move or a killer, and it never adds /
        // drops a move (permutation). Non-vacuous: it visibly changes the quiet
        // order vs `None`.
        struct PreferHighQ;
        impl PolicyPrior for PreferHighQ {
            fn prior(&self, mv: (i32, i32)) -> f32 {
                mv.0 as f32 / 10.0 // higher q ⇒ earlier
            }
        }
        let original = vec![(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)];
        let mut state = OrderingState::with_policy(Box::new(PreferHighQ));
        state.record_cutoff(0, (5, 0), 6); // (5,0) is a killer at ply 0

        let mut moves = original.clone();
        order_moves(&mut moves, Some((1, 0)), 0, &state);
        assert_eq!(moves[0], (1, 0), "TT move must lead regardless of the policy prior");
        assert_eq!(moves[1], (5, 0), "killer must precede the quiet tier regardless of policy");
        // The remaining quiet moves are ordered by the prior (higher q first).
        assert_eq!(&moves[2..], &[(4, 0), (3, 0), (2, 0), (0, 0)], "quiet tier must follow the net-policy prior");
        assert_eq!(sorted(moves.clone()), sorted(original.clone()), "policy ordering must stay a permutation");

        // Non-vacuity: with no policy the quiet tier keeps candidates() order.
        let mut plain = original.clone();
        let mut nostate = OrderingState::new();
        nostate.record_cutoff(0, (5, 0), 6);
        order_moves(&mut plain, Some((1, 0)), 0, &nostate);
        assert_ne!(plain, moves, "policy must actually change the quiet ordering (else vacuous)");
    }
}
