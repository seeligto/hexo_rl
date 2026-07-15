//! HEXG v1 on-disk format (design §3) — a SEPARATE format from dense HEXB.
//!
//! Header (little-endian native):
//!   [magic:   u32 = 0x48455847]  ("HEXG" — distinct from HEXB 0x48455842)
//!   [version: u32 = 1]
//!   [max_stones: u32]            (slot geometry; reject on mismatch — HEXG analog
//!   [max_visits: u32]             of HEXB's stride-sig check)
//!   [capacity: u64]
//!   [size: u64]
//!   [encoding_name_len: u32]
//!   [encoding_name: [u8; N]]     (UTF-8; must match the buffer's encoding)
//!   For each of `size` records (oldest → newest):
//!     n_stones: u16, n_visits: u16, current_player: i8, moves_remaining: u8,
//!     ply_index: u16, is_full_search: u8, value_valid: u8, outcome: f32,
//!     game_length: u16, game_id: i64, weight: u16,
//!     stones:  n_stones × (q:i16, r:i16, p:i8),
//!     visits:  n_visits × (q:i16, r:i16, prob:f32)
//!
//! Cross-format safety (both directions): a HEXG file handed to
//! `ReplayBuffer::load_from_path` fails the HEXB magic check; a HEXB (or any
//! non-HEXG) file handed here fails the HEXG magic check. Neither silently
//! mis-parses. Version / slot-geometry / encoding mismatches all LOUD-FAIL.

use std::io::{BufWriter, Read, Write};
use std::sync::atomic::Ordering;

use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

use super::{weight_bucket, HexgBuffer, HEXG_MAGIC, HEXG_VERSION, MAX_STONES, MAX_VISITS};

impl HexgBuffer {
    /// Save all records (oldest → newest) to `path` in HEXG v1 format.
    pub(crate) fn save_to_path_impl(&self, path: &str) -> PyResult<()> {
        let file = std::fs::File::create(path)
            .map_err(|e| PyIOError::new_err(format!("cannot create {path}: {e}")))?;
        let mut w = BufWriter::new(file);
        let io = |e: std::io::Error| PyIOError::new_err(e.to_string());

        w.write_all(&HEXG_MAGIC.to_le_bytes()).map_err(io)?;
        w.write_all(&HEXG_VERSION.to_le_bytes()).map_err(io)?;
        w.write_all(&(MAX_STONES as u32).to_le_bytes()).map_err(io)?;
        w.write_all(&(MAX_VISITS as u32).to_le_bytes()).map_err(io)?;
        w.write_all(&(self.capacity as u64).to_le_bytes()).map_err(io)?;
        w.write_all(&(self.size as u64).to_le_bytes()).map_err(io)?;
        let name = self.encoding.name.as_bytes();
        w.write_all(&(name.len() as u32).to_le_bytes()).map_err(io)?;
        w.write_all(name).map_err(io)?;

        for i in 0..self.size {
            let slot = (self.head + self.capacity - self.size + i) % self.capacity;
            let ns = self.n_stones[slot];
            let nv = self.n_visits[slot];
            w.write_all(&ns.to_le_bytes()).map_err(io)?;
            w.write_all(&nv.to_le_bytes()).map_err(io)?;
            w.write_all(&self.current_player[slot].to_le_bytes()).map_err(io)?;
            w.write_all(&self.moves_remaining[slot].to_le_bytes()).map_err(io)?;
            w.write_all(&self.ply_index[slot].to_le_bytes()).map_err(io)?;
            w.write_all(&[self.is_full_search[slot]]).map_err(io)?;
            w.write_all(&[self.value_valid[slot]]).map_err(io)?;
            w.write_all(&self.outcomes[slot].to_le_bytes()).map_err(io)?;
            w.write_all(&self.game_length[slot].to_le_bytes()).map_err(io)?;
            w.write_all(&self.game_ids[slot].to_le_bytes()).map_err(io)?;
            w.write_all(&self.weights[slot].to_le_bytes()).map_err(io)?;

            let stone_base = slot * MAX_STONES * 2;
            let player_base = slot * MAX_STONES;
            for j in 0..ns as usize {
                w.write_all(&self.stones_qr[stone_base + j * 2].to_le_bytes()).map_err(io)?;
                w.write_all(&self.stones_qr[stone_base + j * 2 + 1].to_le_bytes()).map_err(io)?;
                w.write_all(&self.stone_players[player_base + j].to_le_bytes()).map_err(io)?;
            }
            let visit_base = slot * MAX_VISITS * 2;
            let prob_base = slot * MAX_VISITS;
            for j in 0..nv as usize {
                w.write_all(&self.visit_qr[visit_base + j * 2].to_le_bytes()).map_err(io)?;
                w.write_all(&self.visit_qr[visit_base + j * 2 + 1].to_le_bytes()).map_err(io)?;
                w.write_all(&self.visit_probs[prob_base + j].to_le_bytes()).map_err(io)?;
            }
        }
        w.flush().map_err(io)?;
        Ok(())
    }

    /// Load records written by `save_to_path`. Returns the number loaded. Missing
    /// file → 0. LOUD-FAILs (Err(String)) on magic / version / slot-geometry /
    /// encoding mismatch.
    ///
    /// ## Failure-atomicity contract (WP-5a red-team B1 fix)
    /// A load that fails for ANY reason (truncation, bound-check, geometry)
    /// leaves `self` EXACTLY as it was before the call — only a fully-parsed
    /// payload commits. Mechanism: two-pass parse-then-apply. Pass 1 reads the
    /// whole payload into a local `Vec<ParsedRecord>`; any `?` failure inside
    /// that loop returns immediately WITHOUT having written a single byte to
    /// `self` (the old code zeroed the weight-bucket histogram and wrote slots
    /// in place *during* the read loop, so a mid-payload truncation left `size`
    /// stale against a partially-repopulated histogram and partially-overwritten
    /// slots — SILENT-CORRUPT). Pass 2 (the commit) only runs once parsing has
    /// fully succeeded, and is a sequence of infallible writes.
    ///
    /// Transient-memory tradeoff: pass 1 buffers the whole parsed payload a
    /// second time (once as raw bytes in `buf`, once as `parsed`) before
    /// committing to `self`. Bounded by the FILE's actual record sizes (`ns`/
    /// `nv` per record), not by the fixed `MAX_STONES`/`MAX_VISITS` slot the
    /// committed buffer reserves per record — so peak overhead during a load is
    /// proportional to the file on disk, not to the ring's worst-case slot size.
    pub(crate) fn load_from_path_impl(&mut self, path: &str) -> Result<usize, String> {
        let mut file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(format!("cannot open {path}: {e}")),
        };
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        let mut cur = Cursor { buf: &buf, pos: 0 };

        let magic = cur.u32()?;
        if magic != HEXG_MAGIC {
            return Err(format!(
                "invalid magic: expected {HEXG_MAGIC:#010x} (HEXG), got {magic:#010x} \
                 (a dense HEXB or foreign file handed to the graph loader)"
            ));
        }
        let version = cur.u32()?;
        if version != HEXG_VERSION {
            return Err(format!(
                "HEXG version {version} not supported (this build reads v{HEXG_VERSION})"
            ));
        }
        let max_stones = cur.u32()? as usize;
        let max_visits = cur.u32()? as usize;
        if max_stones != MAX_STONES || max_visits != MAX_VISITS {
            return Err(format!(
                "HEXG slot-geometry mismatch: file (max_stones {max_stones}, max_visits \
                 {max_visits}) != build (max_stones {MAX_STONES}, max_visits {MAX_VISITS})"
            ));
        }
        let _capacity = cur.u64()? as usize;
        let size = cur.u64()? as usize;
        let name_len = cur.u32()? as usize;
        let name = cur.bytes(name_len)?;
        let name = std::str::from_utf8(name).map_err(|e| e.to_string())?;
        if name != self.encoding.name {
            return Err(format!(
                "HEXG encoding mismatch: file '{name}' != buffer '{}'",
                self.encoding.name
            ));
        }
        if size > self.capacity {
            return Err(format!(
                "HEXG load: file has {size} records but buffer capacity is {}",
                self.capacity
            ));
        }

        // ── PASS 1 (parse, no mutation of `self`) ──
        // Any `?` inside `parse_records` returns Err immediately; `self` has
        // not been touched, so the caller sees a clean pre-call buffer on any
        // failure (a free function, not a `&mut self` method — structurally
        // cannot mutate `self` even by accident).
        let parsed = parse_records(&mut cur, size)?;

        // ── PASS 2 (commit) ──
        // The payload parsed completely — `commit_records` is a sequence of
        // infallible writes, so it cannot itself leave a partially-applied
        // state.
        self.commit_records(&parsed);
        self.size = size;
        self.head = size % self.capacity;

        // FIX (game_id re-base, WP-5a red-team T1): a fresh self-play game id
        // must never collide with a just-loaded record's game_id (the
        // Multi-Window correlation-guard dedup in `sample_indices` keys on
        // `game_id`, so a collision would mis-fire it). Monotonic: never LOWER
        // an already-advanced counter, only raise it past the loaded max.
        // `parsed` empty (zero-entry file) → `max()` is `None` → guarded no-op.
        // FIX (N1, WP-5a red-team re-verification): `max_gid + 1` overflow-panics
        // on `game_id == i64::MAX` in debug (unlabeled `PanicException` through
        // PyO3) / silently wraps to `i64::MIN` in release (no-rebase). Hostile
        // input only (the legit producer's ids are small and monotonic), but a
        // loaded file should never crash or silently skip the rebase.
        if let Some(max_gid) = parsed.iter().map(|r| r.game_id).max() {
            self.next_game_id = self.next_game_id.max(max_gid.saturating_add(1));
        }

        Ok(size)
    }

    /// PASS 2 helper: write every parsed record into its slot and rebuild the
    /// weight-bucket histogram. Only called once `parse_records` has fully
    /// succeeded (see the atomicity contract on `load_from_path_impl`) — every
    /// write here is infallible.
    fn commit_records(&mut self, parsed: &[ParsedRecord]) {
        for b in &self.weight_buckets {
            b.store(0, Ordering::Relaxed);
        }
        for (slot, rec) in parsed.iter().enumerate() {
            self.current_player[slot] = rec.current_player;
            self.moves_remaining[slot] = rec.moves_remaining;
            self.ply_index[slot] = rec.ply_index;
            self.is_full_search[slot] = rec.is_full_search;
            self.value_valid[slot] = rec.value_valid;
            self.outcomes[slot] = rec.outcome;
            self.game_length[slot] = rec.game_length;
            self.game_ids[slot] = rec.game_id;
            self.weights[slot] = rec.weight;
            self.n_stones[slot] = rec.ns as u16;
            self.n_visits[slot] = rec.nv as u16;

            let stone_base = slot * MAX_STONES * 2;
            let player_base = slot * MAX_STONES;
            self.stones_qr[stone_base..stone_base + MAX_STONES * 2].fill(0);
            self.stone_players[player_base..player_base + MAX_STONES].fill(0);
            self.stones_qr[stone_base..stone_base + rec.ns * 2].copy_from_slice(&rec.stones_qr);
            self.stone_players[player_base..player_base + rec.ns]
                .copy_from_slice(&rec.stone_players);

            let visit_base = slot * MAX_VISITS * 2;
            let prob_base = slot * MAX_VISITS;
            self.visit_qr[visit_base..visit_base + MAX_VISITS * 2].fill(0);
            self.visit_probs[prob_base..prob_base + MAX_VISITS].fill(0.0);
            self.visit_qr[visit_base..visit_base + rec.nv * 2].copy_from_slice(&rec.visit_qr);
            self.visit_probs[prob_base..prob_base + rec.nv].copy_from_slice(&rec.visit_probs);

            let bucket = weight_bucket(rec.weight);
            self.weight_buckets[bucket].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// PASS 1 helper: parse `size` records from `cur` into an owned staging Vec.
/// A free function (no `&self`/`&mut self`) — structurally cannot touch a
/// `HexgBuffer`, which is the point: any `?` failure here is guaranteed to
/// leave the caller's buffer untouched (WP-5a red-team B1 fix, see the
/// atomicity contract on `load_from_path_impl`).
fn parse_records(cur: &mut Cursor, size: usize) -> Result<Vec<ParsedRecord>, String> {
    let mut parsed: Vec<ParsedRecord> = Vec::with_capacity(size);
    for slot in 0..size {
        let ns = cur.u16()? as usize;
        let nv = cur.u16()? as usize;
        if ns > MAX_STONES || nv > MAX_VISITS {
            return Err(format!(
                "HEXG load: record {slot} declares {ns} stones / {nv} visits over cap"
            ));
        }
        let current_player = cur.i8()?;
        let moves_remaining = cur.u8()?;
        let ply_index = cur.u16()?;
        let is_full_search = cur.u8()?;
        let value_valid = cur.u8()?;
        let outcome = cur.f32()?;
        let game_length = cur.u16()?;
        let game_id = cur.i64()?;
        let weight = cur.u16()?;

        let mut stones_qr = Vec::with_capacity(ns * 2);
        let mut stone_players = Vec::with_capacity(ns);
        for _ in 0..ns {
            stones_qr.push(cur.i16()?);
            stones_qr.push(cur.i16()?);
            stone_players.push(cur.i8()?);
        }
        let mut visit_qr = Vec::with_capacity(nv * 2);
        let mut visit_probs = Vec::with_capacity(nv);
        for _ in 0..nv {
            visit_qr.push(cur.i16()?);
            visit_qr.push(cur.i16()?);
            visit_probs.push(cur.f32()?);
        }

        parsed.push(ParsedRecord {
            ns,
            nv,
            current_player,
            moves_remaining,
            ply_index,
            is_full_search,
            value_valid,
            outcome,
            game_length,
            game_id,
            weight,
            stones_qr,
            stone_players,
            visit_qr,
            visit_probs,
        });
    }
    Ok(parsed)
}

/// Parsed-but-not-yet-committed record (pass 1 staging — see the atomicity
/// contract on `load_from_path_impl`).
struct ParsedRecord {
    ns: usize,
    nv: usize,
    current_player: i8,
    moves_remaining: u8,
    ply_index: u16,
    is_full_search: u8,
    value_valid: u8,
    outcome: f32,
    game_length: u16,
    game_id: i64,
    weight: u16,
    stones_qr: Vec<i16>,
    stone_players: Vec<i8>,
    visit_qr: Vec<i16>,
    visit_probs: Vec<f32>,
}

/// Minimal little-endian cursor over the loaded byte buffer (no external dep).
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.buf.len() {
            return Err(format!(
                "HEXG load: truncated file (needed {n} bytes at offset {}, have {})",
                self.pos,
                self.buf.len()
            ));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        self.take(n)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn i8(&mut self) -> Result<i8, String> {
        Ok(self.take(1)?[0] as i8)
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn i16(&mut self) -> Result<i16, String> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(std::mem::size_of::<u64>())?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64, String> {
        Ok(i64::from_le_bytes(self.take(std::mem::size_of::<i64>())?.try_into().unwrap()))
    }
}
