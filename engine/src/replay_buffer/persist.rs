//! HEXB v2 on-disk format for `ReplayBuffer` — `save_to_path_impl` and
//! `load_from_path_impl`.
//!
//! Format (little-endian native):
//!   [magic: u32 = 0x48455842]  ("HEXB")
//!   [version: u32 = 2]
//!   [capacity: u64]
//!   [size: u64]
//!   For each of `size` positions (oldest → newest):
//!     state:        STATE_STRIDE × u16
//!     policy:       POLICY_STRIDE × f32
//!     outcome:      f32
//!     game_id:      i64
//!     weight:       u16
//!     ownership:    AUX_STRIDE × u8   (encoding 0/1/2)
//!     winning_line: AUX_STRIDE × u8   (binary mask)
//!
//! v1 (pre A1 aux refactor) buffers are NOT readable; `load_from_path_impl`
//! returns a clear error if encountered — the on-disk layout no longer
//! matches after per-row ownership + winning_line columns were added.

use std::sync::atomic::Ordering;

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

use super::sym_tables::*;
use super::ReplayBuffer;

pub(crate) const HEXB_MAGIC: u32 = 0x4845_5842; // "HEXB"
pub(crate) const HEXB_VERSION: u32 = 2;

impl ReplayBuffer {
    /// Save buffer contents to a binary file in HEXB v2 format.
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
        w.write_all(&(self.capacity as u64).to_le_bytes())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        w.write_all(&(self.size as u64).to_le_bytes())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        // Positions in logical order (oldest → newest)
        for i in 0..self.size {
            let slot = (self.head + self.capacity - self.size + i) % self.capacity;

            // state: u16 slice → bytes
            let state_start = slot * STATE_STRIDE;
            let state_bytes = unsafe {
                std::slice::from_raw_parts(
                    self.states[state_start..state_start + STATE_STRIDE].as_ptr() as *const u8,
                    STATE_STRIDE * 2,
                )
            };
            w.write_all(state_bytes)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;

            // policy: f32 slice → bytes
            let pol_start = slot * POLICY_STRIDE;
            let pol_bytes = unsafe {
                std::slice::from_raw_parts(
                    self.policies[pol_start..pol_start + POLICY_STRIDE].as_ptr() as *const u8,
                    POLICY_STRIDE * 4,
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

            // ownership: AUX_STRIDE × u8
            let aux_start = slot * AUX_STRIDE;
            w.write_all(&self.ownership[aux_start..aux_start + AUX_STRIDE])
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            // winning_line: AUX_STRIDE × u8
            w.write_all(&self.winning_line[aux_start..aux_start + AUX_STRIDE])
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
    pub(crate) fn load_from_path_impl(&mut self, path: &str) -> PyResult<usize> {
        use std::io::{BufReader, Read};

        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(PyIOError::new_err(
                format!("cannot open {path}: {e}")
            )),
        };
        let mut r = BufReader::new(file);

        // Read header
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        r.read_exact(&mut buf4)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let magic = u32::from_le_bytes(buf4);
        if magic != HEXB_MAGIC {
            return Err(PyValueError::new_err(format!(
                "invalid magic: expected 0x48455842 (HEXB), got 0x{magic:08X}"
            )));
        }

        r.read_exact(&mut buf4)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let version = u32::from_le_bytes(buf4);
        if version != HEXB_VERSION {
            return Err(PyValueError::new_err(format!(
                "HEXB version {version} not supported. v1 buffers were invalidated by the A1 \
                 aux target alignment refactor (per-row ownership + winning_line columns added). \
                 Regenerate the buffer from self-play."
            )));
        }

        r.read_exact(&mut buf8)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let _saved_capacity = u64::from_le_bytes(buf8) as usize;

        r.read_exact(&mut buf8)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let saved_size = u64::from_le_bytes(buf8) as usize;

        // How many to actually load — cap at our capacity
        let to_load = saved_size.min(self.capacity);
        // How many to skip if saved_size > capacity (skip oldest)
        let to_skip = saved_size - to_load;

        // Per-entry byte sizes (v2: state + policy + outcome + game_id + weight + own + wl)
        let state_bytes = STATE_STRIDE * 2;
        let policy_bytes = POLICY_STRIDE * 4;
        let entry_bytes = state_bytes + policy_bytes + 4 + 8 + 2 + AUX_STRIDE + AUX_STRIDE;

        // Skip oldest entries
        if to_skip > 0 {
            let skip_bytes = to_skip * entry_bytes;
            let mut remaining = skip_bytes;
            let mut skip_buf = vec![0u8; 8192.min(skip_bytes)];
            while remaining > 0 {
                let chunk = remaining.min(skip_buf.len());
                r.read_exact(&mut skip_buf[..chunk])
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
                remaining -= chunk;
            }
        }

        // Reset weight histogram
        for bucket in &self.weight_buckets {
            bucket.store(0, Ordering::Relaxed);
        }

        // Read positions directly into storage
        let mut state_buf = vec![0u8; state_bytes];
        let mut pol_buf = vec![0u8; policy_bytes];

        for i in 0..to_load {
            let slot = i; // write sequentially from slot 0

            // state
            r.read_exact(&mut state_buf)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            let dst_state = &mut self.states[slot * STATE_STRIDE..(slot + 1) * STATE_STRIDE];
            for (j, d) in dst_state.iter_mut().enumerate() {
                *d = u16::from_le_bytes([state_buf[j * 2], state_buf[j * 2 + 1]]);
            }

            // policy
            r.read_exact(&mut pol_buf)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            let dst_pol = &mut self.policies[slot * POLICY_STRIDE..(slot + 1) * POLICY_STRIDE];
            for (j, d) in dst_pol.iter_mut().enumerate() {
                *d = f32::from_le_bytes([
                    pol_buf[j * 4], pol_buf[j * 4 + 1],
                    pol_buf[j * 4 + 2], pol_buf[j * 4 + 3],
                ]);
            }

            // outcome
            r.read_exact(&mut buf4)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            self.outcomes[slot] = f32::from_le_bytes(buf4);

            // game_id
            r.read_exact(&mut buf8)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            self.game_ids[slot] = i64::from_le_bytes(buf8);

            // weight
            let mut buf2 = [0u8; 2];
            r.read_exact(&mut buf2)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            let w_bits = u16::from_le_bytes(buf2);
            self.weights[slot] = w_bits;

            // ownership + winning_line (v2)
            let aux_dst_start = slot * AUX_STRIDE;
            r.read_exact(&mut self.ownership[aux_dst_start..aux_dst_start + AUX_STRIDE])
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            r.read_exact(&mut self.winning_line[aux_dst_start..aux_dst_start + AUX_STRIDE])
                .map_err(|e| PyIOError::new_err(e.to_string()))?;

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

    /// HEXB v2 round-trip — verify aux columns survive save/load.
    #[test]
    fn test_aux_hexb_v2_roundtrip() {
        use std::env::temp_dir;

        let mut buf = ReplayBuffer::new(8);
        let slot = 0;
        let a_start = slot * AUX_STRIDE;
        buf.ownership[a_start + 10] = 2;  // P1
        buf.ownership[a_start + 20] = 0;  // P2
        buf.ownership[a_start + 30] = 1;  // empty
        for i in 0..6 { buf.winning_line[a_start + 100 + i] = 1; }
        buf.outcomes[slot] = 1.0;
        buf.weights[slot]  = f16::from_f32(1.0).to_bits();
        buf.head = 1;
        buf.size = 1;

        let path = temp_dir().join("aux_v2_roundtrip.hexb");
        buf.save_to_path(path.to_str().unwrap()).unwrap();

        let mut buf2 = ReplayBuffer::new(8);
        let n = buf2.load_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(n, 1);

        let a2 = 0 * AUX_STRIDE;
        assert_eq!(buf2.ownership[a2 + 10], 2);
        assert_eq!(buf2.ownership[a2 + 20], 0);
        assert_eq!(buf2.ownership[a2 + 30], 1);
        for i in 0..6 {
            assert_eq!(buf2.winning_line[a2 + 100 + i], 1);
        }

        let _ = std::fs::remove_file(path);
    }
}
