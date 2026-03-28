/// Bitboard for a 19×19 hex grid (361 cells, stored in 6 × u64 = 384 bits).
///
/// Flat index layout: index = (q + 9) * 19 + (r + 9)
///   word  = index / 64,  bit = index % 64
///
/// Win detection: sliding AND along each hex axis for 6 consecutive cells.
///
/// Three axes and their flat-index strides:
///   E  / W  : (dq=+1, dr= 0) → stride = +19
///   NE / SW : (dq= 0, dr=+1) → stride = + 1
///   SE / NW : (dq=+1, dr=-1) → stride = +18
///
/// Row-wrap guards
/// ───────────────
/// For the E axis (stride 19) the row width exactly equals the stride, so
/// every adjacent flat-index pair genuinely shares the same r value.  No guard
/// needed.
///
/// For the NE axis (stride 1) a right-shift by 1 moves the bit from
/// (q+1, r=-9) [index%19=0] into position (q, r=9) [index%19=18].  Those two
/// cells are NOT NE-neighbours.  Fix: after shifting, zero out the last column
/// (r=9, index%19=18) of the shifted copy before ANDing.
///
/// For the SE axis (stride 18) a right-shift by 18 moves the bit from
/// (q, r=9) [index%19=18] into position (q, r=-9) [index%19=0].  Those two
/// cells are NOT SE-neighbours.  Fix: after shifting, zero out the first column
/// (r=-9, index%19=0) of the shifted copy before ANDing.

use super::BOARD_SIZE;

const WORDS: usize = 6; // ⌈361/64⌉ = 6

// ── 384-bit right-shift ──────────────────────────────────────────────────────

fn shr_384(words: &[u64; WORDS], shift: usize) -> [u64; WORDS] {
    debug_assert!(shift > 0 && shift < 384);
    let word_shift = shift / 64;
    let bit_shift  = shift % 64;
    let mut out = [0u64; WORDS];
    for i in 0..WORDS {
        let src = i + word_shift;
        if src >= WORDS { break; }
        out[i] = words[src] >> bit_shift;
        if bit_shift > 0 && src + 1 < WORDS {
            out[i] |= words[src + 1] << (64 - bit_shift);
        }
    }
    out
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn any_set(words: &[u64; WORDS]) -> bool {
    words.iter().any(|&w| w != 0)
}

/// Bitwise AND of two word arrays.
fn and_words(a: &[u64; WORDS], b: &[u64; WORDS]) -> [u64; WORDS] {
    let mut out = [0u64; WORDS];
    for i in 0..WORDS { out[i] = a[i] & b[i]; }
    out
}

/// Bitwise AND-NOT: `a & !mask`.
fn and_not(a: &[u64; WORDS], mask: &[u64; WORDS]) -> [u64; WORDS] {
    let mut out = [0u64; WORDS];
    for i in 0..WORDS { out[i] = a[i] & !mask[i]; }
    out
}

// ── Column masks (computed at compile time via const fn) ─────────────────────

/// A bitmask with 1 in every cell whose r == r_val.
/// `r_val` must be in [-9, 9].
const fn column_mask(r_val: i32) -> [u64; WORDS] {
    let mut mask = [0u64; WORDS];
    let mut q: i32 = -9;
    while q <= 9 {
        let i = ((q + 9) as usize) * BOARD_SIZE + ((r_val + 9) as usize);
        mask[i / 64] |= 1u64 << (i % 64);
        q += 1;
    }
    mask
}

// Precomputed column masks for the two boundary columns.
const LAST_COL_MASK:  [u64; WORDS] = column_mask( 9); // r = +9
const FIRST_COL_MASK: [u64; WORDS] = column_mask(-9); // r = -9

// ── Bitboard ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bitboard {
    words: [u64; WORDS],
}

impl Bitboard {
    pub fn empty() -> Self {
        Bitboard { words: [0u64; WORDS] }
    }

    /// Set bit at flat index `i`.
    #[inline]
    pub fn set(&mut self, i: usize) {
        self.words[i / 64] |= 1u64 << (i % 64);
    }

    /// Clear bit at flat index `i`.
    #[inline]
    #[allow(dead_code)]
    pub fn clear(&mut self, i: usize) {
        self.words[i / 64] &= !(1u64 << (i % 64));
    }

    /// Test bit at flat index `i`.
    #[inline]
    #[allow(dead_code)]
    pub fn get(&self, i: usize) -> bool {
        (self.words[i / 64] >> (i % 64)) & 1 != 0
    }

    pub fn is_empty(&self) -> bool {
        !any_set(&self.words)
    }

    /// Check for 6-in-a-row along any of the three hex axes.
    pub fn has_six_in_row(&self) -> bool {
        self.six_in_row_e()
            || self.six_in_row_ne()
            || self.six_in_row_se()
    }

    // ── E axis (stride = 19) ─────────────────────────────────────────────────
    //
    // Row width exactly equals stride → no column wrap possible.
    fn six_in_row_e(&self) -> bool {
        const STRIDE: usize = BOARD_SIZE; // 19
        let mut run = self.words;
        for _ in 0..5 {
            let shifted = shr_384(&run, STRIDE);
            run = and_words(&run, &shifted);
            if !any_set(&run) { return false; }
        }
        true
    }

    // ── NE axis (stride = 1) ─────────────────────────────────────────────────
    //
    // Wrap: after shr by 1, the bit from (q+1, r=-9) [index%19=0] lands at
    // (q, r=9) [index%19=18].  Guard: zero out the last column (r=9) of the
    // shifted copy.
    fn six_in_row_ne(&self) -> bool {
        const STRIDE: usize = 1;
        let mut run = self.words;
        for _ in 0..5 {
            let shifted = shr_384(&run, STRIDE);
            // Remove invalid bits that wrapped from the next row's r=-9 into r=9.
            let shifted_clean = and_not(&shifted, &LAST_COL_MASK);
            run = and_words(&run, &shifted_clean);
            if !any_set(&run) { return false; }
        }
        true
    }

    // ── SE axis (stride = 18) ────────────────────────────────────────────────
    //
    // Wrap: after shr by 18, the bit from (q, r=9) [index%19=18] lands at
    // (q, r=-9) [index%19=0].  Guard: zero out the first column (r=-9) of the
    // shifted copy.
    fn six_in_row_se(&self) -> bool {
        const STRIDE: usize = BOARD_SIZE - 1; // 18
        let mut run = self.words;
        for _ in 0..5 {
            let shifted = shr_384(&run, STRIDE);
            // Remove invalid bits that wrapped from r=9 into r=-9.
            let shifted_clean = and_not(&shifted, &FIRST_COL_MASK);
            run = and_words(&run, &shifted_clean);
            if !any_set(&run) { return false; }
        }
        true
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::idx;

    fn set_row_e(bb: &mut Bitboard, q_start: i32, r: i32, len: usize) {
        for i in 0..len as i32 {
            bb.set(idx(q_start + i, r));
        }
    }

    fn set_col_ne(bb: &mut Bitboard, q: i32, r_start: i32, len: usize) {
        for i in 0..len as i32 {
            bb.set(idx(q, r_start + i));
        }
    }

    fn set_diag_se(bb: &mut Bitboard, q_start: i32, r_start: i32, len: usize) {
        for i in 0..len as i32 {
            bb.set(idx(q_start + i, r_start - i));
        }
    }

    // ── E axis ───────────────────────────────────────────────────────────────

    #[test]
    fn empty_no_win() {
        assert!(!Bitboard::empty().has_six_in_row());
    }

    #[test]
    fn five_e_no_win() {
        let mut bb = Bitboard::empty();
        set_row_e(&mut bb, 0, 0, 5);
        assert!(!bb.has_six_in_row());
    }

    #[test]
    fn six_e_win() {
        let mut bb = Bitboard::empty();
        set_row_e(&mut bb, 0, 0, 6);
        assert!(bb.has_six_in_row());
    }

    #[test]
    fn six_e_at_low_edge() {
        // q from -9 to -4
        let mut bb = Bitboard::empty();
        set_row_e(&mut bb, -9, 0, 6);
        assert!(bb.has_six_in_row());
    }

    #[test]
    fn six_e_at_high_edge() {
        // q from 4 to 9
        let mut bb = Bitboard::empty();
        set_row_e(&mut bb, 4, 0, 6);
        assert!(bb.has_six_in_row());
    }

    #[test]
    fn seven_e_wins() {
        let mut bb = Bitboard::empty();
        set_row_e(&mut bb, -1, 0, 7);
        assert!(bb.has_six_in_row());
    }

    #[test]
    fn six_with_gap_no_win() {
        let mut bb = Bitboard::empty();
        set_row_e(&mut bb, 0, 0, 3);
        set_row_e(&mut bb, 4, 0, 3); // gap at q=3
        assert!(!bb.has_six_in_row());
    }

    // ── NE axis ──────────────────────────────────────────────────────────────

    #[test]
    fn six_ne_win_center() {
        let mut bb = Bitboard::empty();
        set_col_ne(&mut bb, 0, 0, 6); // q=0, r=0..5
        assert!(bb.has_six_in_row());
    }

    #[test]
    fn six_ne_at_low_r_edge() {
        // r from -9 to -4 — tests that detection works at the boundary
        let mut bb = Bitboard::empty();
        set_col_ne(&mut bb, 3, -9, 6);
        assert!(bb.has_six_in_row());
    }

    #[test]
    fn six_ne_at_high_r_edge() {
        // r from 4 to 9
        let mut bb = Bitboard::empty();
        set_col_ne(&mut bb, 3, 4, 6);
        assert!(bb.has_six_in_row());
    }

    /// r=5..9 in one row and r=-9..-8 in the NEXT row must NOT form 6-in-a-row.
    /// (Those cells are not NE-neighbours across the row boundary.)
    #[test]
    fn no_wrap_ne_axis() {
        let mut bb = Bitboard::empty();
        set_col_ne(&mut bb, 0, 5, 5); // q=0: r=5,6,7,8,9
        set_col_ne(&mut bb, 1, -9, 2); // q=1: r=-9,-8
        assert!(
            !bb.has_six_in_row(),
            "wrap across row boundary must not count as 6-in-a-row"
        );
    }

    // ── SE axis ──────────────────────────────────────────────────────────────

    #[test]
    fn six_se_win_center() {
        let mut bb = Bitboard::empty();
        set_diag_se(&mut bb, -3, 3, 6); // (-3,3)→(-2,2)→...→(2,-2)
        assert!(bb.has_six_in_row());
    }

    #[test]
    fn six_se_starting_at_r9() {
        // SE run starting at r=9: (0,9),(1,8),(2,7),(3,6),(4,5),(5,4)
        let mut bb = Bitboard::empty();
        set_diag_se(&mut bb, 0, 9, 6);
        assert!(bb.has_six_in_row());
    }

    /// 5-cell SE run reaching r=-9, plus two cells in the SAME ROW at r=9 and r=8,
    /// must NOT form 6-in-a-row. (r=9 wraps to r=-9 of the same row in a stride-18
    /// shift, so we must guard against this.)
    #[test]
    fn no_wrap_se_axis() {
        let mut bb = Bitboard::empty();
        // Valid SE run of 5: (-4,-5),(-3,-6),(-2,-7),(-1,-8),(0,-9)
        set_diag_se(&mut bb, -4, -5, 5);
        // Two cells that would continue a false SE run via row-wrap:
        // stride-18 from (0,-9)=171 → 189=(0,9); then 207=(1,8)
        bb.set(idx(0, 9));
        bb.set(idx(1, 8));
        assert!(
            !bb.has_six_in_row(),
            "wrap from r=9 to r=-9 of same row must not form SE 6-in-a-row"
        );
    }

    // ── Misc ─────────────────────────────────────────────────────────────────

    #[test]
    fn single_stone_no_win() {
        let mut bb = Bitboard::empty();
        bb.set(idx(0, 0));
        assert!(!bb.has_six_in_row());
    }
}
