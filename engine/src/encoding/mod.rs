//! Encoding spec — bundles per-Board encoding parameters.
//! Used by `Board::with_encoding` to construct a Board with non-default
//! cluster window / threshold / legal-move radius. v6w25 sustained
//! self-play (§171) is the first consumer.
//!
//! Note: serde derives intentionally omitted — serde is not a workspace
//! dep on `engine`. Add behind a feature flag if Python-side YAML round-trip
//! ever needs Rust-native (de)serialization.

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EncodingSpec {
    pub cluster_window_size: usize,
    pub cluster_threshold: i32,
    pub legal_move_radius: i32,
    pub board_size: usize,
}

impl EncodingSpec {
    /// v6 default (window=19, threshold=5, radius=5). board_size=19 = canvas dim.
    pub const V6: EncodingSpec = EncodingSpec {
        cluster_window_size: 19,
        cluster_threshold: 5,
        legal_move_radius: 5,
        board_size: 19,
    };

    /// v6w25 (§168 Gate 3): wider 25×25 cluster window, threshold=8, radius=8.
    pub const V6W25: EncodingSpec = EncodingSpec {
        cluster_window_size: 25,
        cluster_threshold: 8,
        legal_move_radius: 8,
        board_size: 19, // canvas still 19; cluster window is the perception widening
    };

    /// Validate field invariants. Returns Err(message) on bad input.
    pub fn validate(&self) -> Result<(), String> {
        if self.cluster_window_size < 7 {
            return Err(format!(
                "cluster_window_size must be >= 7; got {}",
                self.cluster_window_size
            ));
        }
        if self.cluster_window_size % 2 == 0 {
            return Err(format!(
                "cluster_window_size must be odd; got {}",
                self.cluster_window_size
            ));
        }
        if self.cluster_threshold <= 0 {
            return Err(format!(
                "cluster_threshold must be > 0; got {}",
                self.cluster_threshold
            ));
        }
        if self.legal_move_radius <= 0 {
            return Err(format!(
                "legal_move_radius must be > 0; got {}",
                self.legal_move_radius
            ));
        }
        if (self.cluster_window_size as i32) < self.cluster_threshold {
            return Err(format!(
                "cluster_window_size ({}) must be >= cluster_threshold ({})",
                self.cluster_window_size, self.cluster_threshold
            ));
        }
        if self.board_size < 7 {
            return Err(format!("board_size must be >= 7; got {}", self.board_size));
        }
        Ok(())
    }
}
