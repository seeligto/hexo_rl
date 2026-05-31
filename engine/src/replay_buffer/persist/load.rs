//! HEXB v8 + v7 + v6 load path for `ReplayBuffer::load_from_path_impl`.
//! v8 added `position_index: u16` per-row (§S181 Wave 4 4B-impl-1); v7 files
//! load with position_index defaulted to 0.
//!
//! Extracted from `engine/src/replay_buffer/persist.rs` at cycle 3 P68 Wave 7
//! Batch E as a pure module split. The save path + HEXB constants + module
//! docstring stay in `super` (`persist/mod.rs`); the version-dispatch +
//! encoding-validation + payload-read body lives here.

use std::sync::atomic::Ordering;

use super::{ReplayBuffer, HEXB_MAGIC, HEXB_VERSION};

impl ReplayBuffer {
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
    // cycle 3 P68: hosts version dispatch + payload loop; `#[allow]` preserved
    // because the v7/v6 header readers + payload reader run >100 LOC by design
    // (SD4 vs PREP §J).
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

        if version == 8 || version == 7 {
            // v7/v8 header: n_planes, capacity, size, encoding_name_len, encoding_name
            // v8 adds per-row position_index; header layout otherwise identical.
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
                    "HEXB v{version} encoding_name_len={name_len} exceeds maximum 256"
                ));
            }
            let mut name_buf = vec![0u8; name_len];
            r.read_exact(&mut name_buf)
                .map_err(|e| format!("{e}"))?;
            file_encoding_name = String::from_utf8(name_buf)
                .map_err(|e| format!(
                    "HEXB v{version} encoding name is not valid UTF-8: {e}"
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
        // v6/v7 entry: state + chain + policy + outcome(4) + game_id(8) + weight(2)
        //              + ownership + winning_line + is_full_search(1)
        // v8 entry: above + position_index(2)
        let v7_entry_bytes = state_bytes + chain_bytes + policy_bytes + 4 + 8 + 2
            + aux_stride + aux_stride + 1;
        let entry_bytes = if version == 8 { v7_entry_bytes + 2 } else { v7_entry_bytes };

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

            // DRAW-MASK (Phase 6) — value_target_valid is NOT persisted to the
            // on-disk HEXB format (no format bump). On load it defaults to 1
            // (supervise value). ACCEPTABLE SHORTCUT: the 30k re-measure is a
            // FRESH run (no buffer resume), so a saved+reloaded buffer's pre-cap
            // masking is never relied on in practice. `new()` already inits the
            // column to all-ones, so no explicit write is needed here.

            // §S181-AUDIT Wave 4 4B-impl-1 — position_index per row (v8+ only).
            // v6/v7 files default position_index to 0 (aux loss masks pretrain rows).
            if version == 8 {
                let mut buf2 = [0u8; 2];
                r.read_exact(&mut buf2)
                    .map_err(|e| format!("{e}"))?;
                self.position_indices[slot] = u16::from_le_bytes(buf2);
            } else {
                self.position_indices[slot] = 0;
            }

            // Update weight histogram
            let bucket = Self::weight_bucket(w_bits);
            self.weight_buckets[bucket].fetch_add(1, Ordering::Relaxed);
        }

        self.size = to_load;
        self.head = to_load % self.capacity;
        Ok(to_load)
    }
}
