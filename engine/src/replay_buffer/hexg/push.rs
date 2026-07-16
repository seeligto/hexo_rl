//! HEXG ring push — write one compact `GraphRecord` into the `head` slot,
//! handling ring overwrite + weight-bucket bookkeeping. Fixed-slot writes mirror
//! the HEXB push discipline (weight from the game-length schedule, bucket
//! histogram decrement-on-overwrite / increment-on-write).

use std::sync::atomic::Ordering;

use half::f16;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{weight_bucket, GraphRecord, HexgBuffer, MAX_STONES, MAX_VISITS};

/// N2+N3 fix (WP-5a red-team re-verification) — pure check, no PyO3 involved
/// (unit-testable without a GIL/interpreter, same rationale as
/// `sample::mass_drop_check`). The sample-time `mass_drop_check` guard is
/// NaN-blind (`NaN > x` is false on both branches → silently passes, NaN
/// reaches `policy_target`/the loss — N2) and sign-blind (a +/- pair on legal
/// cells aligns/sums to 0.0 → guard passes, a NEGATIVE entry reaches the CE
/// target — N3). Reject both here, at push time — the earliest point that can
/// see the raw, unaligned value — naming the offending coord + value.
/// Hostile-push-only: the legit producer (`record_position_graph`'s
/// `if p > 0.0` filter) can never construct a NaN or negative prob.
pub(crate) fn validate_visit_prob(q: i16, r: i16, prob: f32) -> Result<(), String> {
    if prob.is_finite() && prob >= 0.0 {
        Ok(())
    } else {
        Err(format!(
            "push_graph_position: visit ({q}, {r}) has invalid prob {prob} \
             (must be finite and >= 0.0)"
        ))
    }
}

/// Commit-A red-team ADV-A fix (`WP5b_commitA_redteam.md` #1) — same rationale
/// as `validate_visit_prob`: `outcome` lives in a separate `GraphTargets`
/// object the 18-assertion collate contract never inspects, so a NaN/inf
/// value target sails through untouched and poisons `binned_value_loss` on
/// any `value_valid=1` row. Reject at push time, naming the offending value.
/// Hostile-push-only: the live producer's `finalize_graph_outcome` always
/// returns one of {+1, −1, draw_reward, ply_cap_value}, all finite by config
/// contract.
pub(crate) fn validate_outcome(outcome: f32) -> Result<(), String> {
    if outcome.is_finite() {
        Ok(())
    } else {
        Err(format!(
            "push_graph_position: outcome {outcome} is not finite (must be a finite f32 \
             value target)"
        ))
    }
}

/// Commit-A red-team ADV-B fix (`WP5b_commitA_redteam.md` #1) — mirrors the
/// existing `current_player` range check but applied per-stone: an
/// out-of-range stone player (e.g. `0`, `5`, `-3`) is accepted today and
/// silently rebuilds a structurally-valid-but-wrong-feature graph (collate
/// check #14 recomputes `src_player` from the corrupt node identity, so it
/// passes). Reject at push time, naming the offending coord + value.
/// Hostile-push-only: the live producer reads `Cell as i8` (P1=1/P2=−1),
/// always ±1.
pub(crate) fn validate_stone_player(q: i16, r: i16, player: i8) -> Result<(), String> {
    if player == 1 || player == -1 {
        Ok(())
    } else {
        Err(format!(
            "push_graph_position: stone ({q}, {r}) has invalid player {player} \
             (must be +1 or -1)"
        ))
    }
}

impl HexgBuffer {
    /// Write `rec` into the ring at `head`, advancing `head`/`size`. LOUD-FAILs
    /// if the record exceeds the fixed slot geometry (`MAX_STONES` /
    /// `MAX_VISITS`) — die loud, never silently truncate on push.
    pub(crate) fn push_record_impl(&mut self, rec: &GraphRecord, game_id: i64) -> PyResult<()> {
        if rec.stones.len() > MAX_STONES {
            return Err(PyValueError::new_err(format!(
                "push_graph_position: {} stones exceeds MAX_STONES {} (raise the slot cap or \
                 truncate at record time)",
                rec.stones.len(),
                MAX_STONES
            )));
        }
        if rec.visits.len() > MAX_VISITS {
            return Err(PyValueError::new_err(format!(
                "push_graph_position: {} visit cells exceeds MAX_VISITS {} \
                 (record_position_graph must top-k-truncate before push)",
                rec.visits.len(),
                MAX_VISITS
            )));
        }
        if rec.current_player != 1 && rec.current_player != -1 {
            return Err(PyValueError::new_err(format!(
                "push_graph_position: current_player {} out of range (expected +1 / -1)",
                rec.current_player
            )));
        }
        // FIX (ADV-A, WP5b commit-A red-team): validate outcome finiteness before
        // any mutation of `self` — see `validate_outcome` above.
        validate_outcome(rec.outcome).map_err(PyValueError::new_err)?;
        // FIX (N2+N3, WP-5a red-team re-verification): validate every visit prob
        // before any mutation of `self` — see `validate_visit_prob` above.
        for &(q, r, prob) in &rec.visits {
            validate_visit_prob(q, r, prob).map_err(PyValueError::new_err)?;
        }
        // FIX (ADV-B, WP5b commit-A red-team): validate every stone player
        // before any mutation of `self` — see `validate_stone_player` above.
        for &(q, r, player) in &rec.stones {
            validate_stone_player(q, r, player).map_err(PyValueError::new_err)?;
        }

        let slot = self.head;

        // Ring overwrite: decrement the outgoing slot's weight bucket.
        if self.size == self.capacity {
            let old_bucket = weight_bucket(self.weights[slot]);
            self.weight_buckets[old_bucket].fetch_sub(1, Ordering::Relaxed);
        }

        // ── stones (fixed slot, zero-pad tail) ──
        let stone_base = slot * MAX_STONES * 2;
        let player_base = slot * MAX_STONES;
        // zero the whole slot so a shorter record never inherits stale tail data.
        self.stones_qr[stone_base..stone_base + MAX_STONES * 2].fill(0);
        self.stone_players[player_base..player_base + MAX_STONES].fill(0);
        for (i, &(q, r, p)) in rec.stones.iter().enumerate() {
            self.stones_qr[stone_base + i * 2] = q;
            self.stones_qr[stone_base + i * 2 + 1] = r;
            self.stone_players[player_base + i] = p;
        }
        self.n_stones[slot] = rec.stones.len() as u16;

        // ── visits (fixed slot, zero-pad tail) ──
        let visit_base = slot * MAX_VISITS * 2;
        let prob_base = slot * MAX_VISITS;
        self.visit_qr[visit_base..visit_base + MAX_VISITS * 2].fill(0);
        self.visit_probs[prob_base..prob_base + MAX_VISITS].fill(0.0);
        for (i, &(q, r, prob)) in rec.visits.iter().enumerate() {
            self.visit_qr[visit_base + i * 2] = q;
            self.visit_qr[visit_base + i * 2 + 1] = r;
            self.visit_probs[prob_base + i] = prob;
        }
        self.n_visits[slot] = rec.visits.len() as u16;

        // ── scalars ──
        self.current_player[slot] = rec.current_player;
        self.moves_remaining[slot] = rec.moves_remaining;
        self.ply_index[slot] = rec.ply_index;
        self.is_full_search[slot] = u8::from(rec.is_full_search);
        self.outcomes[slot] = rec.outcome;
        self.value_valid[slot] = u8::from(rec.value_valid);
        self.game_length[slot] = rec.game_length;
        self.game_ids[slot] = game_id;
        self.weights[slot] = if rec.game_length == 0 {
            f16::from_f32(1.0).to_bits()
        } else {
            self.weight_schedule.weight_for(rec.game_length)
        };

        let new_bucket = weight_bucket(self.weights[slot]);
        self.weight_buckets[new_bucket].fetch_add(1, Ordering::Relaxed);

        self.head = (self.head + 1) % self.capacity;
        self.size = (self.size + 1).min(self.capacity);
        Ok(())
    }

    /// Read record at logical slot `slot` back into a `GraphRecord` (test / drain
    /// helper; exact inverse of `push_record_impl`).
    pub(crate) fn record_at(&self, slot: usize) -> GraphRecord {
        let ns = self.n_stones[slot] as usize;
        let nv = self.n_visits[slot] as usize;
        let stone_base = slot * MAX_STONES * 2;
        let player_base = slot * MAX_STONES;
        let stones: Vec<(i16, i16, i8)> = (0..ns)
            .map(|i| {
                (
                    self.stones_qr[stone_base + i * 2],
                    self.stones_qr[stone_base + i * 2 + 1],
                    self.stone_players[player_base + i],
                )
            })
            .collect();
        let visit_base = slot * MAX_VISITS * 2;
        let prob_base = slot * MAX_VISITS;
        let visits: Vec<(i16, i16, f32)> = (0..nv)
            .map(|i| {
                (
                    self.visit_qr[visit_base + i * 2],
                    self.visit_qr[visit_base + i * 2 + 1],
                    self.visit_probs[prob_base + i],
                )
            })
            .collect();
        GraphRecord {
            stones,
            visits,
            current_player: self.current_player[slot],
            moves_remaining: self.moves_remaining[slot],
            ply_index: self.ply_index[slot],
            is_full_search: self.is_full_search[slot] != 0,
            outcome: self.outcomes[slot],
            value_valid: self.value_valid[slot] != 0,
            game_length: self.game_length[slot],
        }
    }
}
