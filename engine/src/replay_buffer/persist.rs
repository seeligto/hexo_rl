//! HEXB v7 on-disk format for `ReplayBuffer` — `save_to_path_impl` and
//! `load_from_path_impl`.
//!
//! Format (little-endian native):
//!   [magic: u32 = 0x48455842]  ("HEXB")
//!   [version: u32 = 7]
//!   [n_planes: u32]            (redundant sanity field, must equal encoding.n_planes)
//!   [capacity: u64]
//!   [size: u64]
//!   [encoding_name_len: u32]   (NEW in v7)
//!   [encoding_name: [u8; N]]   (UTF-8, no null terminator)
//!   For each of `size` positions (oldest → newest):
//!     state:          STATE_STRIDE × u16    (n_planes × n_cells)
//!     chain_planes:   CHAIN_STRIDE × u16    (6 planes × n_cells)
//!     policy:         POLICY_STRIDE × f32
//!     outcome:        f32
//!     game_id:        i64
//!     weight:         u16
//!     ownership:      AUX_STRIDE × u8
//!     winning_line:   AUX_STRIDE × u8
//!     is_full_search: u8
//!
//! v6 backward compatibility:
//!   v6 files lack the encoding_name field. On load, version==6 → assumed
//!   encoding "v6" with a deprecation warning. The buffer's runtime encoding
//!   must also be "v6"; cross-encoding loads are hard-rejected. Re-save
//!   writes v7.
//!
//! v5 and earlier buffers are HARD-REJECTED at load — the §122 B4 verdict
//! committed to a wire-format channel-drop (18 → 8 planes), making any v5
//! row schema-incompatible at the n_planes field. Existing buffers must be
//! regenerated; the cost is bounded (~1.5 GPU-hr at 4090S throughput per
//! buffer_compat_20260429 §5).

use std::sync::atomic::Ordering;

use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

use super::ReplayBuffer;

pub(crate) const HEXB_MAGIC: u32 = 0x4845_5842; // "HEXB"
pub(crate) const HEXB_VERSION: u32 = 7;

impl ReplayBuffer {
    /// Save buffer contents to a binary file in HEXB v7 format.
    pub(crate) fn save_to_path_impl(&self, path: &str) -> PyResult<()> {
        use std::io::{BufWriter, Write};

        let file = std::fs::File::create(path)
            .map_err(|e| PyIOError::new_err(format!("cannot create {path}: {e}")))?;
        let mut w = BufWriter::new(file);

        // Header
        w.write_all(&HEXB_MAGIC.to_le_bytes())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        w.write_all(&HEXB_VERSION.to_le_bytes())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        // Redundant plane-count field for sanity checking.
        w.write_all(&(self.encoding.n_planes as u32).to_le_bytes())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        w.write_all(&(self.capacity as u64).to_le_bytes())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        w.write_all(&(self.size as u64).to_le_bytes())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        // v7: encoding name
        let name_bytes = self.encoding.name.as_bytes();
        w.write_all(&(name_bytes.len() as u32).to_le_bytes())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        w.write_all(name_bytes)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        let state_stride  = self.encoding.state_stride();
        let chain_stride  = self.encoding.chain_stride();
        let policy_stride = self.encoding.policy_stride();
        let aux_stride    = self.encoding.aux_stride();

        // Positions in logical order (oldest → newest)
        for i in 0..self.size {
            let slot = (self.head + self.capacity - self.size + i) % self.capacity;

            // state: u16 slice → bytes
            let state_start = slot * state_stride;
            // SAFETY: &[u16] is layout-compatible with &[u8] (byte_len = 2 × elem_len);
            // resulting slice lifetime is bounded by this match arm and consumed by
            // w.write_all before any aliasing &mut to self.states can be created.
            let state_bytes = unsafe {
                std::slice::from_raw_parts(
                    self.states[state_start..state_start + state_stride].as_ptr().cast::<u8>(),
                    state_stride * 2,
                )
            };
            w.write_all(state_bytes)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;

            // chain_planes: u16 slice → bytes
            let chain_start = slot * chain_stride;
            // SAFETY: &[u16] is layout-compatible with &[u8] (byte_len = 2 × elem_len);
            // resulting slice lifetime is bounded by this match arm and consumed by
            // w.write_all before any aliasing &mut to self.chain_planes can be created.
            let chain_bytes = unsafe {
                std::slice::from_raw_parts(
                    self.chain_planes[chain_start..chain_start + chain_stride].as_ptr().cast::<u8>(),
                    chain_stride * 2,
                )
            };
            w.write_all(chain_bytes)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;

            // policy: f32 slice → bytes
            let pol_start = slot * policy_stride;
            // SAFETY: &[f32] is layout-compatible with &[u8] (byte_len = 4 × elem_len);
            // resulting slice lifetime is bounded by this match arm and consumed by
            // w.write_all before any aliasing &mut to self.policies can be created.
            let pol_bytes = unsafe {
                std::slice::from_raw_parts(
                    self.policies[pol_start..pol_start + policy_stride].as_ptr().cast::<u8>(),
                    policy_stride * 4,
                )
            };
            w.write_all(pol_bytes)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;

            // outcome: f32
            w.write_all(&self.outcomes[slot].to_le_bytes())
                .map_err(|e| PyIOError::new_err(e.to_string()))?;

            // game_id: i64
            w.write_all(&self.game_ids[slot].to_le_bytes())
                .map_err(|e| PyIOError::new_err(e.to_string()))?;

            // weight: u16
            w.write_all(&self.weights[slot].to_le_bytes())
                .map_err(|e| PyIOError::new_err(e.to_string()))?;

            // ownership: aux_stride × u8
            let aux_start = slot * aux_stride;
            w.write_all(&self.ownership[aux_start..aux_start + aux_stride])
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            // winning_line: aux_stride × u8
            w.write_all(&self.winning_line[aux_start..aux_start + aux_stride])
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            // is_full_search: u8
            w.write_all(&[self.is_full_search[slot]])
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }

        w.flush().map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Load buffer contents from a binary file written by `save_to_path_impl`.
    ///
    /// Returns the number of positions loaded.  If the file does not exist,
    /// returns 0 (not an error — supports first-run case).
    ///
    /// If the saved buffer has more positions than `self.capacity`, only the
    /// most recent `self.capacity` positions are loaded.
    ///
    /// Returns `String` errors so unit tests (no Python interpreter) can
    /// exercise error paths.  The PyO3 façade in `mod.rs` maps these to
    /// `PyValueError`.
    // cycle 3 P68: module split — extract header/payload reader helpers
    #[allow(clippy::too_many_lines)]
    pub(crate) fn load_from_path_impl(&mut self, path: &str) -> Result<usize, String> {
        use std::io::{BufReader, Read};

        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(format!("cannot open {path}: {e}")),
        };
        let mut r = BufReader::new(file);

        // Read header
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        r.read_exact(&mut buf4)
            .map_err(|e| format!("{e}"))?;
        let magic = u32::from_le_bytes(buf4);
        if magic != HEXB_MAGIC {
            return Err(format!(
                "invalid magic: expected 0x48455842 (HEXB), got 0x{magic:08X}"
            ));
        }

        r.read_exact(&mut buf4)
            .map_err(|e| format!("{e}"))?;
        let version = u32::from_le_bytes(buf4);

        // ── Version dispatch ──────────────────────────────────────────────
        let file_encoding_name: String;
        let saved_n_planes: usize;
        let _saved_capacity: usize;
        let saved_size: usize;

        if version == 7 {
            // v7 header: n_planes, capacity, size, encoding_name_len, encoding_name
            r.read_exact(&mut buf4)
                .map_err(|e| format!("{e}"))?;
            saved_n_planes = u32::from_le_bytes(buf4) as usize;

            r.read_exact(&mut buf8)
                .map_err(|e| format!("{e}"))?;
            _saved_capacity = u64::from_le_bytes(buf8) as usize;

            r.read_exact(&mut buf8)
                .map_err(|e| format!("{e}"))?;
            saved_size = u64::from_le_bytes(buf8) as usize;

            r.read_exact(&mut buf4)
                .map_err(|e| format!("{e}"))?;
            let name_len = u32::from_le_bytes(buf4) as usize;
            if name_len > 256 {
                return Err(format!(
                    "HEXB v7 encoding_name_len={name_len} exceeds maximum 256"
                ));
            }
            let mut name_buf = vec![0u8; name_len];
            r.read_exact(&mut name_buf)
                .map_err(|e| format!("{e}"))?;
            file_encoding_name = String::from_utf8(name_buf)
                .map_err(|e| format!(
                    "HEXB v7 encoding name is not valid UTF-8: {e}"
                ))?;
        } else if version == 6 {
            // v6 backward compat: no encoding field — assume "v6"
            eprintln!(
                "warning: loading deprecated HEXB v6 file ({path}). \
                 Assuming encoding 'v6'. Re-save to upgrade to v7."
            );
            r.read_exact(&mut buf4)
                .map_err(|e| format!("{e}"))?;
            saved_n_planes = u32::from_le_bytes(buf4) as usize;

            r.read_exact(&mut buf8)
                .map_err(|e| format!("{e}"))?;
            _saved_capacity = u64::from_le_bytes(buf8) as usize;

            r.read_exact(&mut buf8)
                .map_err(|e| format!("{e}"))?;
            saved_size = u64::from_le_bytes(buf8) as usize;

            file_encoding_name = "v6".to_string();
        } else {
            return Err(format!(
                "HEXB version {version} not supported (this build only reads v{HEXB_VERSION}). \
                 v5 and earlier are deprecated by the §122 B4 verdict — wire-format \
                 channel-drop (18 → 8 planes per D17 ablation Set A) is incompatible \
                 with their plane count. Regenerate the buffer with v{HEXB_VERSION} \
                 (see reports/audits/buffer_compat_20260429.md §5 for cost estimate)."
            ));
        }

        // ── Encoding validation ───────────────────────────────────────────
        let Some(file_spec) = crate::encoding::registry::lookup(&file_encoding_name) else {
            return Err(format!(
                "HEXB file declares unknown encoding '{file_encoding_name}'. \
                 Registered encodings: {:?}",
                {
                    let mut known: Vec<&str> = crate::encoding::registry::all_specs()
                        .map(|s| s.name).collect();
                    known.sort_unstable();
                    known
                }
            ));
        };

        if saved_n_planes != file_spec.n_planes {
            return Err(format!(
                "HEXB file header has n_planes={saved_n_planes}, \
                 but encoding '{file_encoding_name}' expects {}. \
                 File may be corrupted or written with a mismatched registry.",
                file_spec.n_planes
            ));
        }

        // Cross-encoding mismatch guard (§P13).
        //
        // Compare wire-format signature instead of the encoding name string.
        // Two encodings producing byte-identical HEXB rows must auto-cross-load
        // even when their names differ — v6 / v7full / v7 / v7e30 / v7mw all
        // share `(8, 19, 362, true, "size_19")`; v8 / v8_canvas_realness share
        // `(11, 25, 625, false, "size_25")`; v6w25 stays distinct at
        // `(8, 25, 626, true, "size_25")`. Strict shape rejection is preserved
        // — any signature drift (different n_planes, board_size,
        // policy_logit_count, has_pass_slot, or sym_table_id) still hard-errors.
        let buffer_sig = self.encoding.wire_signature();
        let file_sig = file_spec.wire_signature();
        if buffer_sig != file_sig {
            return Err(format!(
                "HEXB encoding mismatch: buffer encoding '{}' wire_signature \
                 {:?} differs from file encoding '{}' wire_signature {:?}",
                self.encoding.name, buffer_sig, file_encoding_name, file_sig
            ));
        }

        // How many to actually load — cap at our capacity
        let to_load = saved_size.min(self.capacity);
        // How many to skip if saved_size > capacity (skip oldest)
        let to_skip = saved_size - to_load;

        let state_stride  = self.encoding.state_stride();
        let chain_stride  = self.encoding.chain_stride();
        let policy_stride = self.encoding.policy_stride();
        let aux_stride    = self.encoding.aux_stride();

        // Per-entry byte sizes.
        let state_bytes = state_stride * 2;
        let chain_bytes = chain_stride * 2;
        let policy_bytes = policy_stride * 4;
        let entry_bytes = state_bytes + chain_bytes + policy_bytes + 4 + 8 + 2
            + aux_stride + aux_stride + 1;

        // Skip oldest entries
        if to_skip > 0 {
            let skip_bytes = to_skip * entry_bytes;
            let mut remaining = skip_bytes;
            let mut skip_buf = vec![0u8; 8192.min(skip_bytes)];
            while remaining > 0 {
                let chunk = remaining.min(skip_buf.len());
                r.read_exact(&mut skip_buf[..chunk])
                    .map_err(|e| format!("{e}"))?;
                remaining -= chunk;
            }
        }

        // Reset weight histogram
        for bucket in &self.weight_buckets {
            bucket.store(0, Ordering::Relaxed);
        }

        // Read positions directly into storage
        let mut state_buf = vec![0u8; state_bytes];
        let mut chain_buf = vec![0u8; chain_bytes];
        let mut pol_buf = vec![0u8; policy_bytes];

        for i in 0..to_load {
            let slot = i; // write sequentially from slot 0

            // state
            r.read_exact(&mut state_buf)
                .map_err(|e| format!("{e}"))?;
            let dst_state = &mut self.states[slot * state_stride..(slot + 1) * state_stride];
            for (j, d) in dst_state.iter_mut().enumerate() {
                *d = u16::from_le_bytes([state_buf[j * 2], state_buf[j * 2 + 1]]);
            }

            // chain_planes
            r.read_exact(&mut chain_buf)
                .map_err(|e| format!("{e}"))?;
            let dst_chain = &mut self.chain_planes[slot * chain_stride..(slot + 1) * chain_stride];
            for (j, d) in dst_chain.iter_mut().enumerate() {
                *d = u16::from_le_bytes([chain_buf[j * 2], chain_buf[j * 2 + 1]]);
            }

            // policy
            r.read_exact(&mut pol_buf)
                .map_err(|e| format!("{e}"))?;
            let dst_pol = &mut self.policies[slot * policy_stride..(slot + 1) * policy_stride];
            for (j, d) in dst_pol.iter_mut().enumerate() {
                *d = f32::from_le_bytes([
                    pol_buf[j * 4], pol_buf[j * 4 + 1],
                    pol_buf[j * 4 + 2], pol_buf[j * 4 + 3],
                ]);
            }

            // outcome
            r.read_exact(&mut buf4)
                .map_err(|e| format!("{e}"))?;
            self.outcomes[slot] = f32::from_le_bytes(buf4);

            // game_id
            r.read_exact(&mut buf8)
                .map_err(|e| format!("{e}"))?;
            self.game_ids[slot] = i64::from_le_bytes(buf8);

            // weight
            let mut buf2 = [0u8; 2];
            r.read_exact(&mut buf2)
                .map_err(|e| format!("{e}"))?;
            let w_bits = u16::from_le_bytes(buf2);
            self.weights[slot] = w_bits;

            // ownership + winning_line
            let aux_dst_start = slot * aux_stride;
            r.read_exact(&mut self.ownership[aux_dst_start..aux_dst_start + aux_stride])
                .map_err(|e| format!("{e}"))?;
            r.read_exact(&mut self.winning_line[aux_dst_start..aux_dst_start + aux_stride])
                .map_err(|e| format!("{e}"))?;

            // is_full_search: u8
            let mut buf1 = [0u8; 1];
            r.read_exact(&mut buf1)
                .map_err(|e| format!("{e}"))?;
            self.is_full_search[slot] = buf1[0];

            // Update weight histogram
            let bucket = Self::weight_bucket(w_bits);
            self.weight_buckets[bucket].fetch_add(1, Ordering::Relaxed);
        }

        self.size = to_load;
        self.head = to_load % self.capacity;
        Ok(to_load)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use std::sync::atomic::AtomicU64;

    /// Position-0 offset within a slot — documents intent of `0 * stride` in
    /// round-trip tests (avoids `clippy::erasing_op` while preserving the
    /// "first stored position" annotation).
    const POS_0: usize = 0;

    /// Per-test unique path under `temp_dir()`.
    /// Pre-existing pattern used hardcoded filenames shared across tests; cargo
    /// runs test binaries in parallel, so the integration test in
    /// `engine/tests/replay_buffer_v6_roundtrip.rs` and these unit tests would
    /// race on the same path under sufficient load (§178-class concurrency
    /// flake observed by Wave 4 Batch C reviewer). Per-test unique paths
    /// dispatch the race entirely. Triaged cycle 2 wave 5 pre-flight (Flake 3).
    fn unique_test_path(stem: &str) -> std::path::PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("hexb_persist_{stem}_{pid}_{nanos}_{n}.hexb"))
    }

    /// HEXB v7 round-trip — verify aux columns (ownership, winning_line, chain_planes, is_full_search) survive save/load.
    #[test]
    fn test_aux_hexb_v7_roundtrip() {
        let mut buf = ReplayBuffer::new(8, "v6");
        let slot = 0;
        let aux_stride   = buf.encoding.aux_stride();
        let chain_stride = buf.encoding.chain_stride();
        let a_start = slot * aux_stride;
        buf.ownership[a_start + 10] = 2;  // P1
        buf.ownership[a_start + 20] = 0;  // P2
        buf.ownership[a_start + 30] = 1;  // empty
        for i in 0..6 { buf.winning_line[a_start + 100 + i] = 1; }
        // Write non-zero chain_planes so the round-trip test is meaningful.
        let c_start = slot * chain_stride;
        buf.chain_planes[c_start + 0]   = f16::from_f32(0.5).to_bits();
        buf.chain_planes[c_start + 100] = f16::from_f32(1.0).to_bits();
        buf.chain_planes[c_start + buf.encoding.n_cells()] = f16::from_f32(0.25).to_bits();  // plane 1
        buf.outcomes[slot]       = 1.0;
        buf.weights[slot]        = f16::from_f32(1.0).to_bits();
        buf.is_full_search[slot] = 0;  // quick-search
        buf.head = 1;
        buf.size = 1;

        let path = unique_test_path("aux_v7_roundtrip");
        buf.save_to_path(path.to_str().unwrap()).unwrap();

        let mut buf2 = ReplayBuffer::new(8, "v6");
        let n = buf2.load_from_path_impl(path.to_str().unwrap()).unwrap();
        assert_eq!(n, 1);

        let aux_stride2   = buf2.encoding.aux_stride();
        let chain_stride2 = buf2.encoding.chain_stride();
        let a2 = POS_0 * aux_stride2;
        assert_eq!(buf2.ownership[a2 + 10], 2);
        assert_eq!(buf2.ownership[a2 + 20], 0);
        assert_eq!(buf2.ownership[a2 + 30], 1);
        for i in 0..6 {
            assert_eq!(buf2.winning_line[a2 + 100 + i], 1);
        }
        let c2 = POS_0 * chain_stride2;
        assert_eq!(buf2.chain_planes[c2 + 0],   f16::from_f32(0.5).to_bits());
        assert_eq!(buf2.chain_planes[c2 + 100], f16::from_f32(1.0).to_bits());
        assert_eq!(buf2.chain_planes[c2 + buf2.encoding.n_cells()], f16::from_f32(0.25).to_bits());
        assert_eq!(buf2.is_full_search[0], 0, "is_full_search must survive round-trip");

        let _ = std::fs::remove_file(path);
    }

    /// HEXB v7 round-trip with v6w25 encoding.
    #[test]
    fn test_hexb_v7_round_trip_v6w25() {
        let mut buf = ReplayBuffer::new(100, "v6w25");
        for i in 0..100 {
            buf.push_for_test(i as f32 / 100.0, (i % 50) as u16, i % 3 == 0);
        }
        assert_eq!(buf.size(), 100);

        let path = unique_test_path("v7_v6w25_roundtrip");
        buf.save_to_path(path.to_str().unwrap()).unwrap();

        let mut buf2 = ReplayBuffer::new(100, "v6w25");
        let n = buf2.load_from_path_impl(path.to_str().unwrap()).unwrap();
        assert_eq!(n, 100);
        assert_eq!(buf2.size(), 100);

        for slot in 0..100 {
            assert_eq!(buf2.outcomes[slot], (slot as f32) / 100.0, "outcome mismatch at slot {slot}");
            assert_eq!(buf2.is_full_search_at(slot), (slot % 3 == 0) as u8, "is_full_search mismatch at slot {slot}");
        }

        let _ = std::fs::remove_file(path);
    }

    /// Parametrize round-trip over all 7 registered encodings.
    #[test]
    fn test_hexb_v7_round_trip_all_encodings() {
        let names: Vec<&str> = crate::encoding::registry::all_specs()
            .map(|s| s.name).collect();
        for name in names {
            let mut buf = ReplayBuffer::new(10, name);
            for i in 0..10 {
                buf.push_for_test(i as f32, 10, true);
            }
            let path = unique_test_path(&format!("v7_all_enc_{name}"));
            buf.save_to_path(path.to_str().unwrap()).unwrap();

            let mut buf2 = ReplayBuffer::new(10, name);
            let n = buf2.load_from_path_impl(path.to_str().unwrap()).unwrap();
            assert_eq!(n, 10, "encoding {name}: expected 10 positions loaded");
            assert_eq!(buf2.size(), 10, "encoding {name}: size mismatch");

            // Verify outcome values survived.
            for slot in 0..10 {
                assert_eq!(buf2.outcomes[slot], slot as f32,
                    "encoding {name}: outcome mismatch at slot {slot}");
            }

            let _ = std::fs::remove_file(&path);
        }
    }

    /// v6 backward compat: manually write a v6-format file, load with v7 code,
    /// verify assumed "v6" + deprecation warning.
    #[test]
    fn test_hexb_v6_backward_compat() {
        use std::io::Write;

        let path = unique_test_path("legacy_v6_no_encoding");
        {
            let mut file = std::fs::File::create(&path).unwrap();
            // v6 header: magic, version=6, n_planes=8, capacity=10, size=5
            file.write_all(&HEXB_MAGIC.to_le_bytes()).unwrap();
            file.write_all(&6u32.to_le_bytes()).unwrap();
            file.write_all(&8u32.to_le_bytes()).unwrap();
            file.write_all(&10u64.to_le_bytes()).unwrap();
            file.write_all(&5u64.to_le_bytes()).unwrap();

            // Write 5 dummy entries (state + chain + policy + outcome + game_id + weight + own + wl + ifs)
            let state_stride = 8 * 361;
            let chain_stride = 6 * 361;
            let policy_stride = 362;
            let aux_stride = 361;
            let entry_bytes = state_stride * 2 + chain_stride * 2 + policy_stride * 4 + 4 + 8 + 2
                + aux_stride + aux_stride + 1;
            for i in 0..5 {
                let mut entry = vec![0u8; entry_bytes];
                // outcome at offset state*2 + chain*2 + policy*4
                let off = state_stride * 2 + chain_stride * 2 + policy_stride * 4;
                entry[off..off+4].copy_from_slice(&(i as f32).to_le_bytes());
                file.write_all(&entry).unwrap();
            }
        }

        let mut buf = ReplayBuffer::new(10, "v6");
        let n = buf.load_from_path_impl(path.to_str().unwrap()).unwrap();
        assert_eq!(n, 5, "v6 backward compat: expected 5 positions");
        assert_eq!(buf.size(), 5);
        for slot in 0..5 {
            assert_eq!(buf.outcomes[slot], slot as f32,
                "v6 backward compat: outcome mismatch at slot {slot}");
        }

        let _ = std::fs::remove_file(path);
    }

    /// §P13 — wire-identical crossload SUCCEEDS in-process (mirror of the
    /// integration test in `engine/tests/test_p13_wire_signature_crossload.rs`).
    /// v6 → v7full share `(8, 19, 362, true, "size_19")` so the load guard
    /// accepts despite the name string mismatch.
    #[test]
    fn test_hexb_v7_wire_signature_v6_to_v7full_crossload_succeeds() {
        let mut writer = ReplayBuffer::new(8, "v6");
        for i in 0..6 {
            writer.push_for_test(i as f32, 10, i % 2 == 0);
        }

        let path = unique_test_path("p13_v6_to_v7full");
        writer.save_to_path(path.to_str().unwrap()).unwrap();

        let mut reader = ReplayBuffer::new(8, "v7full");
        let loaded = reader
            .load_from_path_impl(path.to_str().unwrap())
            .expect("wire-identical v6 -> v7full crossload must succeed");
        assert_eq!(loaded, 6);

        let _ = std::fs::remove_file(path);
    }

    /// §P13 — v6 → v8 wire-mismatched crossload REJECTS. Full wire-signature
    /// drift (n_planes 8→11, board_size 19→25, policy_logit_count 362→625,
    /// has_pass_slot true→false, sym_table_id "size_19"→"size_25"). The
    /// `saved_n_planes != file_spec.n_planes` guard CANNOT fire here because
    /// the file header is internally consistent for v6 (saved_n_planes=8 and
    /// file_spec='v6' n_planes=8); the wire_signature compare is the only
    /// thing rejecting the load.
    #[test]
    fn test_hexb_v7_wire_signature_v6_to_v8_rejects() {
        let mut writer = ReplayBuffer::new(8, "v6");
        writer.push_for_test(1.0, 10, true);

        let path = unique_test_path("p13_v6_to_v8_rejects");
        writer.save_to_path(path.to_str().unwrap()).unwrap();

        let mut reader = ReplayBuffer::new(8, "v8");
        let err = reader
            .load_from_path_impl(path.to_str().unwrap())
            .expect_err("v6 -> v8 crossload must reject (full wire-signature drift)");
        assert!(
            err.contains("encoding mismatch") && err.contains("wire_signature"),
            "expected wire_signature framing, got: {err}"
        );

        let _ = std::fs::remove_file(path);
    }

    /// Cross-encoding mismatch: v7 file with encoding="v6", load into v6w25 buffer → hard error.
    #[test]
    fn test_hexb_v7_encoding_mismatch_rejects() {
        use std::io::Write;

        let path = unique_test_path("v7_mismatch_v6_into_v6w25");
        {
            let mut file = std::fs::File::create(&path).unwrap();
            // v7 header declaring encoding "v6"
            file.write_all(&HEXB_MAGIC.to_le_bytes()).unwrap();
            file.write_all(&7u32.to_le_bytes()).unwrap();
            file.write_all(&8u32.to_le_bytes()).unwrap(); // n_planes=8 (v6)
            file.write_all(&10u64.to_le_bytes()).unwrap();
            file.write_all(&1u64.to_le_bytes()).unwrap();
            file.write_all(&2u32.to_le_bytes()).unwrap(); // name_len = 2
            file.write_all(b"v6").unwrap();

            // Dummy entry
            let state_stride = 8 * 361;
            let chain_stride = 6 * 361;
            let policy_stride = 362;
            let aux_stride = 361;
            let entry_bytes = state_stride * 2 + chain_stride * 2 + policy_stride * 4 + 4 + 8 + 2
                + aux_stride + aux_stride + 1;
            file.write_all(&vec![0u8; entry_bytes]).unwrap();
        }

        let mut buf = ReplayBuffer::new(10, "v6w25");
        let err = buf.load_from_path_impl(path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("encoding mismatch"), "expected 'encoding mismatch' error, got: {err}");
        assert!(err.contains("v6w25"), "error should mention buffer encoding v6w25: {err}");
        assert!(err.contains("v6"), "error should mention file encoding v6: {err}");

        let _ = std::fs::remove_file(path);
    }

    /// Unknown encoding in v7 header → hard error.
    #[test]
    fn test_hexb_v7_unknown_encoding_rejects() {
        use std::io::Write;

        let path = unique_test_path("v7_unknown_encoding");
        {
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(&HEXB_MAGIC.to_le_bytes()).unwrap();
            file.write_all(&7u32.to_le_bytes()).unwrap();
            file.write_all(&8u32.to_le_bytes()).unwrap();
            file.write_all(&10u64.to_le_bytes()).unwrap();
            file.write_all(&1u64.to_le_bytes()).unwrap();
            let name = b"nonexistent";
            file.write_all(&(name.len() as u32).to_le_bytes()).unwrap();
            file.write_all(name).unwrap();

            let state_stride = 8 * 361;
            let chain_stride = 6 * 361;
            let policy_stride = 362;
            let aux_stride = 361;
            let entry_bytes = state_stride * 2 + chain_stride * 2 + policy_stride * 4 + 4 + 8 + 2
                + aux_stride + aux_stride + 1;
            file.write_all(&vec![0u8; entry_bytes]).unwrap();
        }

        let mut buf = ReplayBuffer::new(10, "v6");
        let err = buf.load_from_path_impl(path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("unknown encoding"), "expected 'unknown encoding' error, got: {err}");

        let _ = std::fs::remove_file(path);
    }

    /// v5 hard-reject unchanged.
    #[test]
    fn test_hexb_v5_hard_reject() {
        use std::io::Write;

        let path = unique_test_path("v5_rejected");
        {
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(&HEXB_MAGIC.to_le_bytes()).unwrap();
            file.write_all(&5u32.to_le_bytes()).unwrap();
        }

        let mut buf = ReplayBuffer::new(10, "v6");
        let err = buf.load_from_path_impl(path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("not supported"), "expected 'not supported' for v5, got: {err}");

        let _ = std::fs::remove_file(path);
    }

    /// Max-length encoding name round-trips (v8_canvas_realness = 18 bytes).
    #[test]
    fn test_hexb_v7_header_max_name() {
        let mut buf = ReplayBuffer::new(10, "v8_canvas_realness");
        for i in 0..10 {
            buf.push_for_test(i as f32, 10, true);
        }
        let path = unique_test_path("v7_max_name");
        buf.save_to_path(path.to_str().unwrap()).unwrap();

        let mut buf2 = ReplayBuffer::new(10, "v8_canvas_realness");
        let n = buf2.load_from_path_impl(path.to_str().unwrap()).unwrap();
        assert_eq!(n, 10, "max-name round-trip: expected 10 positions");
        assert_eq!(buf2.size(), 10);

        for slot in 0..10 {
            assert_eq!(buf2.outcomes[slot], slot as f32,
                "max-name round-trip: outcome mismatch at slot {slot}");
        }

        let _ = std::fs::remove_file(path);
    }
}
