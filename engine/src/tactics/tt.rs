//! Transposition table — generation-aged 2-slot bucket (D-PFIT P2 increment 3).
//!
//! FOUNDATION: a game-theoretic cache of PROVEN outcomes keyed by
//! `(zobrist_hash, side_to_move, moves_remaining)` — depth-INDEPENDENT (a proven
//! win/loss is a proven win/loss regardless of the depth it was found at). Ports
//! the SealBot perf TT (`bot.h:183-216` 2-bucket gen-aged; `bot.h:50-78` int16
//! mate-distance score quantization), onto the engine's native u128 zobrist.
//!
//! # SOUNDNESS INVARIANT — the load-bearing property (unchanged from the foundation)
//! **Only a proven LOSS is trusted as a proof.** `get_loss_proof` returns a
//! verdict ONLY for an entry flagged `is_proof` whose decoded score is a
//! mate-magnitude LOSS; everything else (bounds, best moves, heuristic scores) is
//! ORDERING data and is NEVER read as a proof conclusion. A WIN is never returned
//! as a cached proof — a WIN cache hit would carry an empty principal variation
//! and could truncate the A1 override line, so WINs are always reconstructed fresh
//! (the `store_bound` best-move hint only ORDERS the fresh re-search).
//!
//! Eviction is sound by construction: the table is a CACHE. A bucket collision is
//! resolved by full-key comparison (never a false hit), and evicting a proven-LOSS
//! entry only forces a re-proof later (a miss, more nodes) — never a false proof.
//! The replacement policy (depth-preferred + always-replace + generation aging) is
//! a pure recall/throughput heuristic; it ranks `is_proof` above any non-proof
//! entry so a heuristic bound can never evict a proof from the depth-preferred slot.
//!
//! # DEFERRED (bench-gate / perf-box only — NOT built here)
//! The byte-level `#[repr(C)]` 16-byte packing + 64-byte bucket alignment
//! (`engine_types.h:65-93`) is a memory-layout micro-opt gated on `make bench`
//! (laptop thermal cutoff → deferred to vast). This module uses readable fields
//! (int16 mate-distance `score`, an `EXACT/LOWER/UPPER` enum, an `Option` best
//! move); the algorithmic wins (2-slot aged bucket, mate-distance quantization,
//! best-move ordering) are all present.

use super::{MATE, WIN_THRESHOLD};

/// Key: (zobrist u128, side-to-move as i8, moves_remaining u8).
pub type TtKey = (u128, i8, u8);

/// α-β bound flag (standard TT semantics). A proven LOSS is stored `Exact`; the
/// non-proof ordering/bound entries (increment 4) carry `Lower`/`Upper`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bound {
    /// Exact node value (PV node, or a game-theoretic proof).
    Exact,
    /// Fail-high: `score` is a LOWER bound on the true value (β-cutoff).
    Lower,
    /// Fail-low: `score` is an UPPER bound on the true value.
    Upper,
}

/// Bucket count (power of two so the index is a mask). Proof-LOSS entries are
/// SPARSE (one write per distinct proven loss, not per node), so this comfortably
/// holds every proof a within-budget search produces without eviction.
const N_BUCKETS: usize = 1 << 16;

/// int16 mate band. A mate score is `±(MATE_I16 - dist_from_node)`; heuristic
/// scores clamp strictly inside `±(WIN_THRESHOLD_I16 - 1)`. The 1000-ply gap
/// matches `super::WIN_THRESHOLD` and keeps every realisable mate distance encodable.
const MATE_I16: i16 = 30_000;
const WIN_THRESHOLD_I16: i16 = MATE_I16 - 1000;

/// Encode a root-relative i32 search score into a NODE-relative int16 TT score so
/// a probe at a DIFFERENT ply decodes a correct mate distance. Mate scores keep
/// their distance-from-node; heuristic scores clamp into the i16 heuristic band.
#[inline]
pub(crate) fn encode_score(score: i32, ply: i32) -> i16 {
    // i64 arithmetic so extreme/defensive inputs (i32::MIN) clamp, never overflow.
    let (s, p, mate, mate16) = (score as i64, ply as i64, MATE as i64, MATE_I16 as i64);
    if score >= WIN_THRESHOLD {
        // mate WIN: absolute mate ply = MATE - score; distance from node = − ply.
        let dist = (mate - s - p).clamp(0, 1000);
        (mate16 - dist) as i16
    } else if score <= -WIN_THRESHOLD {
        let dist = (mate + s - p).clamp(0, 1000); // mate - (-score) = mate + score
        -((mate16 - dist) as i16)
    } else {
        let bound = (WIN_THRESHOLD_I16 - 1) as i64;
        s.clamp(-bound, bound) as i16
    }
}

/// Decode a NODE-relative int16 TT score at the probing node's `ply` back to a
/// root-relative i32 search score (inverse of `encode_score`).
#[inline]
pub(crate) fn decode_score(score: i16, ply: i32) -> i32 {
    let s = score as i32; // widen before any negate so the i16 corners can't overflow
    if score >= WIN_THRESHOLD_I16 {
        let dist = MATE_I16 as i32 - s;
        MATE - (ply + dist)
    } else if score <= -WIN_THRESHOLD_I16 {
        let dist = MATE_I16 as i32 + s; // MATE_I16 - (-score) = MATE_I16 + score
        -(MATE - (ply + dist))
    } else {
        s
    }
}

/// One bucket slot. `Copy` + a `used` flag so the bucket array inits cheaply.
#[derive(Clone, Copy)]
struct Slot {
    key: TtKey,
    /// int16 mate-distance-encoded score (node-relative; see `encode_score`).
    score: i16,
    /// EXACT/LOWER/UPPER flag. Stored now; READ by the increment-4 PVS bound
    /// cutoffs (pruning only, never a verdict).
    #[allow(dead_code)]
    bound: Bound,
    /// Best move for ordering (the move that produced `score`); `None` = no hint.
    best: Option<(i32, i32)>,
    /// `depth_left` at store time — the depth-preferred replacement key.
    depth: i16,
    /// Generation written — an entry from an older generation is replaceable.
    gen: u8,
    /// Game-theoretic proof (a proven LOSS). The ONLY thing trusted as a verdict.
    is_proof: bool,
    used: bool,
}

impl Slot {
    const EMPTY: Slot = Slot {
        key: (0, 0, 0),
        score: 0,
        bound: Bound::Exact,
        best: None,
        depth: 0,
        gen: 0,
        is_proof: false,
        used: false,
    };

    /// Replacement rank: a proof outranks any non-proof; ties break on depth. The
    /// depth-preferred slot keeps the highest-rank entry, so a heuristic bound can
    /// never evict a proof.
    #[inline]
    fn rank(&self) -> i64 {
        (self.is_proof as i64) << 32 | (self.depth as i64 & 0xFFFF_FFFF)
    }
}

/// Generation-aged 2-slot bucket transposition table. Slot 0 is depth-preferred,
/// slot 1 is always-replace. See the module soundness note.
pub struct ProofTt {
    buckets: Vec<[Slot; 2]>,
    generation: u8,
}

impl ProofTt {
    pub fn new() -> Self {
        ProofTt { buckets: vec![[Slot::EMPTY; 2]; N_BUCKETS], generation: 0 }
    }

    /// Bucket index — fold the u128 zobrist low word with the turn-structure key.
    #[inline]
    fn index(key: TtKey) -> usize {
        let h = (key.0 as u64) ^ ((key.1 as u64) << 1) ^ ((key.2 as u64) << 3);
        (h as usize) & (N_BUCKETS - 1)
    }

    /// Bump the generation (deploy-time reuse across `prove` calls): entries from
    /// the prior generation become preferentially replaceable.
    #[inline]
    pub fn new_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    /// Find the slot matching `key` exactly (full-key verify — never a false hit
    /// on a bucket collision).
    #[inline]
    fn find(&self, key: TtKey) -> Option<&Slot> {
        let b = &self.buckets[Self::index(key)];
        b.iter().find(|s| s.used && s.key == key)
    }

    /// PROOF probe (the trusted path). Returns the decoded mate-LOSS score iff a
    /// trusted proven-LOSS entry exists at `key`. NEVER returns a WIN (WINs are
    /// reconstructed; see the module note) and NEVER a non-proof entry.
    #[inline]
    pub fn get_loss_proof(&self, key: TtKey, ply: i32) -> Option<i32> {
        let s = self.find(key)?;
        if s.is_proof && s.score <= -WIN_THRESHOLD_I16 {
            Some(decode_score(s.score, ply))
        } else {
            None
        }
    }

    /// ORDERING probe (never a verdict). Best move hint for any entry at `key`.
    #[inline]
    pub fn get_best_move(&self, key: TtKey) -> Option<(i32, i32)> {
        self.find(key).and_then(|s| s.best)
    }

    /// Store a PROVEN LOSS (game-theoretic). `score` is the root-relative exact
    /// node value; `ply`/`depth` are the node's. Trusted by `get_loss_proof`.
    #[inline]
    pub fn store_loss_proof(&mut self, key: TtKey, score: i32, ply: i32, depth: i32) {
        debug_assert!(score <= -WIN_THRESHOLD, "store_loss_proof needs a mate-magnitude LOSS score");
        self.put(Slot {
            key,
            score: encode_score(score, ply),
            bound: Bound::Exact,
            best: None,
            depth: depth.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            gen: self.generation,
            is_proof: true,
            used: true,
        });
    }

    /// Store a NON-PROOF ordering/bound entry (best move + α-β bound). NEVER
    /// trusted as a verdict — only `get_best_move` reads it. (Increment 4 wiring.)
    #[inline]
    pub fn store_bound(
        &mut self,
        key: TtKey,
        score: i32,
        ply: i32,
        bound: Bound,
        best: Option<(i32, i32)>,
        depth: i32,
    ) {
        self.put(Slot {
            key,
            score: encode_score(score, ply),
            bound,
            best,
            depth: depth.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            gen: self.generation,
            is_proof: false,
            used: true,
        });
    }

    /// 2-slot replacement: update-in-place on a key match (keeping the more
    /// authoritative of old/new); else depth-preferred slot 0 / always-replace
    /// slot 1, with stale-generation entries always yielding.
    fn put(&mut self, entry: Slot) {
        let gen = self.generation;
        let b = &mut self.buckets[Self::index(entry.key)];
        for s in b.iter_mut() {
            if s.used && s.key == entry.key {
                // Same position: keep the higher-rank record (a proof, or deeper).
                if entry.rank() >= s.rank() {
                    *s = entry;
                }
                return;
            }
        }
        // Depth-preferred slot 0: replace if empty, stale-generation, or the new
        // entry is at least as authoritative (rank). Demote the displaced entry to
        // the always-replace slot 1 so a recently-deep record survives one round.
        let replace_depth =
            !b[0].used || b[0].gen != gen || entry.rank() >= b[0].rank();
        if replace_depth {
            b[1] = b[0];
            b[0] = entry;
        } else {
            b[1] = entry;
        }
    }
}

impl Default for ProofTt {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn loss_score(mate_abs_ply: i32) -> i32 {
        -(MATE - mate_abs_ply)
    }
    fn win_score(mate_abs_ply: i32) -> i32 {
        MATE - mate_abs_ply
    }

    #[test]
    fn int16_encode_decode_round_trips_mate_distance() {
        // A LOSS landing at absolute ply `m`, observed at node ply `p`, must decode
        // (at the SAME probe ply) to the SAME mate-distance-aware i32 score, for a
        // range of node plies and mate depths — node-relative encoding is reusable.
        for &p in &[0, 1, 4, 7, 20] {
            for &k in &[0, 1, 3, 8, 25] {
                let m = p + k; // mate lands k plies below the node
                let s = loss_score(m);
                let e = encode_score(s, p);
                assert!(e <= -WIN_THRESHOLD_I16, "encoded LOSS {e} not in mate band");
                assert_eq!(decode_score(e, p), s, "LOSS round-trip failed (p={p}, k={k})");
                let sw = win_score(m);
                let ew = encode_score(sw, p);
                assert!(ew >= WIN_THRESHOLD_I16, "encoded WIN {ew} not in mate band");
                assert_eq!(decode_score(ew, p), sw, "WIN round-trip failed (p={p}, k={k})");
            }
        }
    }

    #[test]
    fn int16_encoding_reusable_at_a_different_probe_ply() {
        // Node-relative encoding: a LOSS proven at node ply 3 (mate at abs ply 5,
        // i.e. 2 from node) decodes at ANY probe ply p' to a mate 2 plies below p'.
        let stored = encode_score(loss_score(5), 3); // dist-from-node = 2
        for &pp in &[0, 2, 6, 11] {
            assert_eq!(decode_score(stored, pp), loss_score(pp + 2), "reuse at ply {pp} wrong");
        }
    }

    #[test]
    fn heuristic_score_clamps_into_int16_band_never_a_proof() {
        // A heuristic (sub-threshold, non-mate) score must clamp strictly inside the
        // i16 mate band so a decoded value can NEVER read as a proof magnitude. The
        // boundary values just inside the proof region (`±(WIN_THRESHOLD-1)`, far
        // larger than i16) exercise the clamp. (Inputs AT/over the mate magnitude —
        // `i32::MIN/MAX` — are genuine mate scores and correctly encode to the band;
        // the search never feeds an out-of-band heuristic: leaves are pre-clamped by
        // `clamp_heuristic`.)
        for &v in &[-WIN_THRESHOLD + 1, -123, 0, 77, WIN_THRESHOLD - 1] {
            assert!(v.abs() < WIN_THRESHOLD, "test input {v} must be a genuine heuristic (sub-mate)");
            let e = encode_score(v, 0);
            assert!(e.abs() < WIN_THRESHOLD_I16, "heuristic {v} -> {e} leaked into the mate band");
            let d = decode_score(e, 0);
            assert!(d.abs() < WIN_THRESHOLD, "decoded heuristic {d} leaked into the proof region");
        }
    }

    #[test]
    fn proof_loss_round_trips_through_the_table() {
        let mut tt = ProofTt::new();
        let key = (0x1234_5678_9abc_def0_u128, 1i8, 2u8);
        assert_eq!(tt.get_loss_proof(key, 0), None, "empty table: no proof");
        tt.store_loss_proof(key, loss_score(6), 4, 10);
        // probe re-encodes at the probe ply (dist-from-node = 6 - 4 = 2)
        assert_eq!(tt.get_loss_proof(key, 4), Some(loss_score(6)), "store/probe at same ply");
        assert_eq!(tt.get_loss_proof(key, 9), Some(loss_score(11)), "reuse at deeper ply (2 from node)");
    }

    #[test]
    fn non_proof_bound_is_never_returned_as_a_verdict() {
        // SOUNDNESS: only proven LOSS is trusted. A non-proof bound entry (even one
        // whose stored score sits in the loss band) must NOT be returned by
        // get_loss_proof — only its best move is exposed (for ordering).
        let mut tt = ProofTt::new();
        let key = (42u128, -1i8, 1u8);
        tt.store_bound(key, loss_score(3), 0, Bound::Upper, Some((2, 5)), 8);
        assert_eq!(tt.get_loss_proof(key, 0), None, "non-proof entry must not be a verdict");
        assert_eq!(tt.get_best_move(key), Some((2, 5)), "ordering hint must still be readable");
    }

    #[test]
    fn full_key_verification_no_false_hit_on_bucket_collision() {
        // Two DISTINCT keys that collide on the bucket index must not cross-hit:
        // the index folds only the low word + turn bits, so toggling a HIGH bit of
        // the zobrist keeps the same bucket but a different key.
        let mut tt = ProofTt::new();
        let k1 = (0x0000_0000_0000_0001_u128, 1i8, 2u8);
        let k2 = (k1.0 | (1u128 << 100), 1i8, 2u8); // same low word/turn -> same bucket
        assert_eq!(ProofTt::index(k1), ProofTt::index(k2), "test setup: keys must collide");
        tt.store_loss_proof(k1, loss_score(2), 0, 5);
        assert_eq!(tt.get_loss_proof(k1, 0), Some(loss_score(2)), "k1 stored");
        assert_eq!(tt.get_loss_proof(k2, 0), None, "k2 must NOT false-hit k1's entry");
    }

    #[test]
    fn depth_preferred_slot_keeps_the_deeper_proof() {
        // Two distinct keys in one bucket: a deeper entry takes the depth-preferred
        // slot and survives a shallower store (which lands in always-replace), then
        // a third distinct key (always-replace) does not displace the deep slot.
        let mut tt = ProofTt::new();
        let base = 0u128;
        let deep = (base, 1i8, 2u8);
        let shallow = (base | (1u128 << 100), 1i8, 2u8);
        let third = (base | (1u128 << 101), 1i8, 2u8);
        assert_eq!(ProofTt::index(deep), ProofTt::index(shallow));
        assert_eq!(ProofTt::index(deep), ProofTt::index(third));

        tt.store_loss_proof(deep, loss_score(2), 0, /*depth*/ 30);
        tt.store_loss_proof(shallow, loss_score(2), 0, /*depth*/ 3);
        assert_eq!(tt.get_loss_proof(deep, 0), Some(loss_score(2)), "deep proof survives in slot 0");
        assert_eq!(tt.get_loss_proof(shallow, 0), Some(loss_score(2)), "shallow proof in always-replace");

        tt.store_loss_proof(third, loss_score(2), 0, /*depth*/ 4);
        assert_eq!(tt.get_loss_proof(deep, 0), Some(loss_score(2)), "deep slot retained vs always-replace churn");
    }

    #[test]
    fn proof_is_never_evicted_by_a_non_proof_bound() {
        // The rank rule: a heuristic bound must not displace a proof from the
        // depth-preferred slot, even when the bound is "deeper".
        let mut tt = ProofTt::new();
        let proof = (0u128, 1i8, 2u8);
        let bound_a = (1u128 << 100, 1i8, 2u8);
        let bound_b = (1u128 << 101, 1i8, 2u8);
        assert_eq!(ProofTt::index(proof), ProofTt::index(bound_a));
        tt.store_loss_proof(proof, loss_score(2), 0, /*depth*/ 5);
        // two deep non-proof bounds churn the bucket; the proof must remain.
        tt.store_bound(bound_a, 100, 0, Bound::Lower, Some((1, 1)), /*depth*/ 40);
        tt.store_bound(bound_b, 100, 0, Bound::Lower, Some((2, 2)), /*depth*/ 40);
        assert_eq!(tt.get_loss_proof(proof, 0), Some(loss_score(2)), "proof must survive non-proof churn");
    }

    #[test]
    fn stale_generation_entry_is_replaceable() {
        // Aging: an entry from a prior generation yields the depth slot to a NEW
        // generation store even when the newcomer is shallower — so a reused TT
        // does not pin stale deep entries forever.
        let mut tt = ProofTt::new();
        let old = (0u128, 1i8, 2u8);
        let new = (1u128 << 100, 1i8, 2u8);
        let filler = (1u128 << 101, 1i8, 2u8);
        assert_eq!(ProofTt::index(old), ProofTt::index(new));
        tt.store_loss_proof(old, loss_score(2), 0, /*depth*/ 50); // gen 0, depth slot
        tt.new_generation();
        tt.store_loss_proof(new, loss_score(2), 0, /*depth*/ 1); // gen 1: stale gen-0 yields slot 0
        // old demoted to always-replace; a second gen-1 store overwrites it.
        tt.store_loss_proof(filler, loss_score(2), 0, /*depth*/ 1);
        assert_eq!(tt.get_loss_proof(old, 0), None, "stale deep entry evicted after aging + churn");
        assert_eq!(tt.get_loss_proof(new, 0), Some(loss_score(2)), "new-generation entry retained in slot 0");
    }

    #[test]
    #[cfg(debug_assertions)] // debug_assert-backed contract — meaningless (and failing) under --release
    fn store_loss_proof_rejects_non_loss_in_debug() {
        // The store API asserts its contract (mate-magnitude LOSS) in debug.
        let r = std::panic::catch_unwind(|| {
            let mut tt = ProofTt::new();
            tt.store_loss_proof((0u128, 1i8, 2u8), 0 /* heuristic, not a loss */, 0, 5);
        });
        assert!(r.is_err(), "store_loss_proof must debug-assert a mate-magnitude LOSS score");
    }
}
