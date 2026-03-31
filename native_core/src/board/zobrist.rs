/// Zobrist hash table for the 19×19 board.
///
/// Two tables: one per player. Both are seeded deterministically using
/// splitmix64 with distinct seeds, so the same position always hashes
/// to the same value regardless of how it was reached.
///
/// `ZobristTable::get(cell_index, player_index)` returns the XOR key to
/// apply when placing (or removing) a stone of `player_index` (0=P1, 1=P2)
/// at flat index `cell_index`.

use super::TOTAL_CELLS; // 361

/// Deterministic splitmix64 PRNG. Produces a pseudo-random u64 from a seed.
const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

/// Produce a 128-bit pseudo-random value from a 64-bit seed using two
/// independent splitmix64 streams (different additive constants).
const fn splitmix128(seed: u64) -> u128 {
    // Low 64 bits: standard splitmix64
    let lo = splitmix64(seed);
    // High 64 bits: second pass with a different additive so the streams are independent
    let hi = splitmix64(seed.wrapping_add(0x9e3779b97f4a7c15));
    ((hi as u128) << 64) | (lo as u128)
}

/// Generate a table of `N` pseudo-random u128 values from a given seed.
const fn gen_table<const N: usize>(seed: u64) -> [u128; N] {
    let mut table = [0u128; N];
    let mut state = seed;
    let mut i = 0;
    while i < N {
        state = splitmix64(state);
        table[i] = splitmix128(state);
        i += 1;
    }
    table
}

/// Zobrist keys for player 1 (one per cell).
const KEYS_P1: [u128; TOTAL_CELLS] = gen_table::<TOTAL_CELLS>(0xa02bdbf7bb3c0195);
/// Zobrist keys for player 2 (one per cell).
const KEYS_P2: [u128; TOTAL_CELLS] = gen_table::<TOTAL_CELLS>(0x3f84d5b5b5470917);

pub struct ZobristTable;

impl ZobristTable {
    /// Return the Zobrist key for placing a stone of `player` (0=P1, 1=P2)
    /// at flat cell index `cell` (for the 19×19 absolute grid).
    #[inline(always)]
    pub fn get(cell: usize, player: usize) -> u128 {
        if player == 0 {
            KEYS_P1[cell]
        } else {
            KEYS_P2[cell]
        }
    }

    /// Return the Zobrist key for placing a stone of `player` (0=P1, 1=P2)
    /// at absolute axial coordinate (q, r).
    ///
    /// For coordinates within [-9, 9] uses the pre-generated table.
    /// For out-of-range coordinates uses splitmix128-based hashing.
    ///
    /// This is position-independent: the same (q, r, player) always returns
    /// the same key regardless of board history or window position.
    #[inline]
    pub fn get_for_pos(q: i32, r: i32, player: usize) -> u128 {
        const HALF: i32 = 9;
        const SIZE: i32 = 19;
        if q >= -HALF && q <= HALF && r >= -HALF && r <= HALF {
            let cell = ((q + HALF) as usize) * (SIZE as usize) + ((r + HALF) as usize);
            Self::get(cell, player)
        } else {
            // Mixing primes for arbitrary coordinates
            const M1: u64 = 0x6c62272e07bb0142;
            const M2: u64 = 0x62b821756295c58d;
            const M3: u64 = 0x94d049bb133111eb;
            let seed = (q as i64 as u64).wrapping_mul(M1)
                ^ (r as i64 as u64).wrapping_mul(M2)
                ^ (player as u64).wrapping_mul(M3);
            splitmix128(seed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_nonzero() {
        for i in 0..TOTAL_CELLS {
            assert_ne!(KEYS_P1[i], 0, "P1 key at {i} is zero");
            assert_ne!(KEYS_P2[i], 0, "P2 key at {i} is zero");
        }
    }

    #[test]
    fn keys_are_distinct_within_player() {
        use std::collections::HashSet;
        let p1: HashSet<u128> = KEYS_P1.iter().copied().collect();
        let p2: HashSet<u128> = KEYS_P2.iter().copied().collect();
        assert_eq!(p1.len(), TOTAL_CELLS, "P1 keys have duplicates");
        assert_eq!(p2.len(), TOTAL_CELLS, "P2 keys have duplicates");
    }

    #[test]
    fn keys_are_distinct_across_players() {
        use std::collections::HashSet;
        let all: HashSet<u128> = KEYS_P1.iter().chain(KEYS_P2.iter()).copied().collect();
        assert_eq!(all.len(), TOTAL_CELLS * 2, "P1 and P2 keys overlap");
    }

    #[test]
    fn xor_incremental_matches_bulk() {
        // Place three stones and verify that XOR-ing the keys gives a consistent hash.
        let cells = [10usize, 50, 200];
        let players = [0usize, 1, 0];

        let mut hash = 0u128;
        for (&c, &p) in cells.iter().zip(players.iter()) {
            hash ^= ZobristTable::get(c, p);
        }

        // Recompute by XOR-ing in a different order (XOR is commutative)
        let mut hash2 = 0u128;
        for (&c, &p) in cells.iter().rev().zip(players.iter().rev()) {
            hash2 ^= ZobristTable::get(c, p);
        }

        assert_eq!(hash, hash2, "hash must be order-independent");
    }
}
