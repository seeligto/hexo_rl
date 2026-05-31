//! HEXB v8 on-disk format for `ReplayBuffer` — `save_to_path_impl` and
//! `load_from_path_impl`.
//!
//! Format (little-endian native):
//!   [magic: u32 = 0x48455842]  ("HEXB")
//!   [version: u32 = 8]
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
//!     position_index: u16                    (NEW in v8; §S181 Wave 4 4B-impl-1)
//!
//! v7 backward compatibility:
//!   v7 files lack position_index. On load, version==7 → defaults position_index to 0
//!   for every row. Aux loss masks pretrain rows so the dummy zeros don't pollute
//!   training. Re-save writes v8.
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

use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

use super::ReplayBuffer;

mod load;

pub(crate) const HEXB_MAGIC: u32 = 0x4845_5842; // "HEXB"
pub(crate) const HEXB_VERSION: u32 = 8;

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
            // §S181-AUDIT Wave 4 4B-impl-1 (HEXB v8): position_index u16
            w.write_all(&self.position_indices[slot].to_le_bytes())
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }

        w.flush().map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use std::sync::atomic::{AtomicU64, Ordering};

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
        buf.position_indices[slot] = 42;  // §S181 Wave 4 4B-impl-1
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
        assert_eq!(buf2.position_indices[0], 42, "position_index must survive v8 round-trip");

        let _ = std::fs::remove_file(path);
    }

    /// DRAW-MASK (Phase 6) — per-row `value_target_valid` column round-trip
    /// (in-memory; the column is intentionally NOT persisted — documented
    /// shortcut). Pushes a capped row (value_target_valid=0) plus a default
    /// row, then asserts the accessor reads back the stamped values. Mirrors
    /// the `is_full_search` round-trip contract.
    #[test]
    fn test_value_target_valid_column_in_memory() {
        let mut buf = ReplayBuffer::new(8, "v6");
        // Default rows: push_for_test leaves value_target_valid at its push_raw
        // default of 1 (supervise value).
        buf.push_for_test(0.5, 30, true);
        assert_eq!(buf.value_target_valid_at(0), 1,
            "default pushed row must supervise value (value_target_valid=1)");

        // Stamp slot 1 as a ply-capped row (masked from value loss).
        buf.push_for_test(-0.5, 30, true);
        buf.value_target_valid[1] = 0;
        assert_eq!(buf.value_target_valid_at(1), 0,
            "ply-capped row must read value_target_valid=0 via accessor");
        // is_full_search is independent — the capped row still contributes policy.
        assert_eq!(buf.is_full_search_at(1), 1,
            "value-mask must not touch is_full_search (policy target kept)");

        // Documented persist shortcut: the column is NOT written to disk and
        // defaults to 1 on load. Confirm a fresh buffer inits to all-ones.
        let buf2 = ReplayBuffer::new(8, "v6");
        assert_eq!(buf2.value_target_valid_at(0), 1,
            "fresh buffer must default value_target_valid to 1 (supervise)");
    }

    /// §S181-AUDIT Wave 4 4B-impl-1 — explicit per-row position_index round-trip
    /// across multiple rows, verifying v8 file format wire-correctness.
    #[test]
    fn test_position_index_v8_roundtrip_multirow() {
        let mut buf = ReplayBuffer::new(8, "v6");
        for i in 0..5u16 {
            buf.push_for_test(i as f32, 30, true);
        }
        for i in 0..5u16 {
            buf.position_indices[i as usize] = i * 7 + 3;
        }

        let path = unique_test_path("pos_idx_v8_roundtrip");
        buf.save_to_path(path.to_str().unwrap()).unwrap();

        let mut buf2 = ReplayBuffer::new(8, "v6");
        let n = buf2.load_from_path_impl(path.to_str().unwrap()).unwrap();
        assert_eq!(n, 5);
        for i in 0..5u16 {
            assert_eq!(
                buf2.position_indices[i as usize], i * 7 + 3,
                "position_index mismatch at row {i}",
            );
        }

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
