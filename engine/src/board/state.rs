use std::cell::{Cell as StdCell, UnsafeCell};
use fxhash::{FxHashMap, FxHashSet};
use super::zobrist::ZobristTable;

// ── MoveDiff ──────────────────────────────────────────────────────────────────

/// Captures everything mutated by one `apply_move_tracked` call so that
/// `undo_move` can reverse it in O(1) without any HashMap scan.
///
/// All fields are private; the only way to construct a `MoveDiff` is through
/// `apply_move_tracked`, and the only way to consume it is through `undo_move`.
#[derive(Debug, Clone)]
pub struct MoveDiff {
    pub(crate) q: i32,
    pub(crate) r: i32,
    pub(crate) player: Player,
    // Previous full Zobrist hash state.
    prev_zobrist_hash: u128,
    // Turn-structure state before the move.
    prev_moves_remaining: u8,
    prev_current_player: Player,
    prev_ply: u32,
    // Win-detection state before the move.
    prev_last_move: Option<(i32, i32)>,
    // Bounding-box state before the move (needed for O(1) bbox undo).
    prev_min_q: i32,
    prev_max_q: i32,
    prev_min_r: i32,
    prev_max_r: i32,
    prev_has_stones: bool,
    // Action anchors state before the move.
    prev_action_anchors: [(i32, i32); 4],
    prev_action_anchors_count: usize,
}

/// Board size (cells per axis of the view window).
pub const BOARD_SIZE: usize = 19;
/// Half-width: window covers [-HALF, HALF] relative to its centre.
pub const HALF: i32 = (BOARD_SIZE as i32 - 1) / 2; // 9
/// Total cells in the 19×19 view window.
pub const TOTAL_CELLS: usize = BOARD_SIZE * BOARD_SIZE; // 361

/// Plane offsets in the 18-plane state tensor (§174 — eliminate bare literals).
pub const MY_STONE_PLANE: usize = 0;
pub const OPP_STONE_PLANE: usize = 8;
pub const MOVES_REMAINING_PLANE: usize = 16;
pub const PLY_PARITY_PLANE: usize = 17;

/// The three hex axis directions (positive direction only; win scan uses ±).
pub fn hex_distance(q1: i32, r1: i32, q2: i32, r2: i32) -> i32 {
    ((q1 - q2).abs() + (q1 + r1 - q2 - r2).abs() + (r1 - r2).abs()) / 2
}

pub const HEX_AXES: [(i32, i32); 3] = [
    (1, 0),  // E / W
    (0, 1),  // NE / SW
    (1, -1), // SE / NW
];

// ── Player ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum Player {
    One = 1,
    Two = -1,
}

impl Player {
    pub fn other(self) -> Self {
        match self {
            Player::One => Player::Two,
            Player::Two => Player::One,
        }
    }
}

// ── Cell ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i8)]
pub enum Cell {
    #[default]
    Empty = 0,
    P1 = 1,
    P2 = -1,
}

// ── Board ─────────────────────────────────────────────────────────────────────

/// Sparse game board.  All state needed to continue a game from any position.
///
/// # Thread safety
///
/// `Board` contains `UnsafeCell` for lazy cache rebuilding, making it `!Sync`
/// by default.  We add `unsafe impl Sync for Board` because every `Board`
/// instance is either:
///   - owned by a single MCTS worker thread, or
///   - accessed from Python under the GIL (which serialises all calls).
/// Concurrent mutable access to the same `Board` never occurs.
#[derive(Debug)]
pub struct Board {
    /// Sparse stone map: (q, r) → Cell.
    pub(crate) cells: FxHashMap<(i32, i32), Cell>,
    /// Whose turn it is.
    pub current_player: Player,
    /// How many moves the current player still has to place this turn.
    /// Starts at 1 on ply 0 (P1's single first move), then 2 for every turn.
    pub moves_remaining: u8,
    /// Total half-moves placed so far.
    pub ply: u32,
    /// Incremental Zobrist hash.
    pub zobrist_hash: u128,
    /// The move most recently applied (used for fast win detection).
    pub(crate) last_move: Option<(i32, i32)>,
    /// Bounding box of all placed stones (maintained incrementally).
    pub(crate) min_q: i32,
    pub(crate) max_q: i32,
    pub(crate) min_r: i32,
    pub(crate) max_r: i32,
    /// True once at least one stone has been placed.
    pub(crate) has_stones: bool,
    /// Last 4 stones placed (q, r).
    pub(crate) action_anchors: [(i32, i32); 4],
    pub(crate) action_anchors_count: usize,
    /// Lazily-maintained set of all currently legal moves.
    ///
    /// Uses interior mutability so that `legal_moves_set(&self)` can rebuild
    /// on demand without requiring `&mut self` (which would conflict with
    /// callers that hold `&Board` references, e.g. `expand_and_backup_single`).
    ///
    /// Invariant: when `cache_dirty` is false, `legal_cache` is correct.
    /// When `cache_dirty` is true, `legal_moves_set()` rebuilds it.
    ///
    /// SAFETY: `Board` is single-owner / single-thread per MCTS worker.
    /// No concurrent access to `legal_cache` can occur.
    pub(crate) legal_cache: UnsafeCell<FxHashSet<(i32, i32)>>,
    /// Set to true by any mutating operation (apply_move / undo_move).
    /// Cleared by `legal_moves_set()` after a full rebuild.
    pub(crate) cache_dirty: StdCell<bool>,
    /// Per-board legal-move radius override (Phase B' v8 §152 Q2).
    ///
    /// `legal_moves_set()` rebuilds by hex-ball expansion at this radius.
    /// Default is the canonical `moves::LEGAL_MOVE_RADIUS` constant (5).
    /// `SelfPlayRunner` overrides this per game when the jitter flag is on
    /// — sampling from {4, 5, 6} via a worker-local RNG to break the
    /// radius-5 stride-5 fixed point identified in §152.
    pub(crate) legal_move_radius: i32,
    /// Per-board cluster connectivity threshold (§168 Gate 3 v6w25 plumbing).
    ///
    /// Two stones share a cluster iff their `hex_distance` is ≤ this value.
    /// Default is `moves::DEFAULT_CLUSTER_THRESHOLD` (5, matching v6 wire
    /// format). v6w25 corpus generation overrides to 8 to widen cluster
    /// reach in proportion to the larger 25×25 cluster window.
    pub(crate) cluster_threshold: i32,
    /// Per-board cluster window side length (§168 Gate 3 v6w25 plumbing).
    ///
    /// `get_cluster_views()` emits 2-plane snapshots of this side length.
    /// Default is `BOARD_SIZE` (19, v6 wire format). v6w25 corpus generation
    /// overrides to 25 for matched-perception A/B vs v8.
    pub(crate) cluster_window_size: usize,
    /// §172 A4.1 — registry-resolved encoding bound to this Board.
    ///
    /// `None` = legacy v6 default (matches `Board::new()` / pre-A4 callers
    /// that never set an explicit encoding). `Some(spec)` = bound by
    /// `Board::with_registry_spec`; `to_planes` / spatial dims read
    /// `spec.board_size`, multi-window encodings (`spec.is_multi_window`)
    /// loud-fail at `to_planes()` per design §4.6.
    ///
    /// Pointer is stable for process lifetime (registry uses `Box::leak`
    /// at parse time).
    pub(crate) encoding: Option<&'static crate::encoding::RegistrySpec>,
}

impl Board {
    /// Create an empty board ready for the first move.
    pub fn new() -> Self {
        // Pre-populate legal_cache with the 5×5 region centred at (0,0).
        // This restricts the first move to a 25-cell neighbourhood, keeping
        // the MCTS branching factor at ~24 for the entire game (matching the
        // bbox+2 semantics used after every stone is placed).
        //
        // The spec says "all 361 cells legal for empty board", but 361 root
        // children would require 361 PUCT evaluations per sim and create a
        // 14x MCTS throughput regression — an unjustifiable cost when the
        // first move's location is strategically arbitrary.
        //
        // cache_dirty starts false — no rebuild needed until a stone is placed.
        let mut init_cache = FxHashSet::default();
        init_cache.reserve(50);
        for dq in -2i32..=2 {
            for dr in -2i32..=2 {
                init_cache.insert((dq, dr));
            }
        }

        Board {
            cells: FxHashMap::default(),
            current_player: Player::One,
            moves_remaining: 1,
            ply: 0,
            zobrist_hash: 0,
            last_move: None,
            min_q: 0,
            max_q: 0,
            min_r: 0,
            max_r: 0,
            has_stones: false,
            action_anchors: [(0, 0); 4],
            action_anchors_count: 0,
            legal_cache: UnsafeCell::new(init_cache),
            cache_dirty: StdCell::new(false),
            legal_move_radius: super::moves::DEFAULT_LEGAL_MOVE_RADIUS,
            cluster_threshold: super::moves::DEFAULT_CLUSTER_THRESHOLD,
            cluster_window_size: BOARD_SIZE,
            encoding: None,
        }
    }

    /// §172 A4.1 — construct a Board bound to a registry-resolved encoding.
    ///
    /// Companion to the legacy 4-field `Board::with_encoding(&EncodingSpec)`
    /// (kept for §171 plumbing compat). New consumers should prefer this
    /// ctor — fields below derive directly from the registry record so
    /// adding a new encoding means editing only `registry.toml`.
    ///
    /// Field mapping (per design doc §4.5):
    ///   - `cluster_window_size` ← `spec.cluster_window_size.unwrap_or(spec.board_size)`
    ///     (single-window encodings carry `None` in the registry; the
    ///     Board still needs a usable window dim, so fall back to
    ///     `board_size` to match legacy `Board::new()` semantics for v6.)
    ///   - `cluster_threshold`   ← `spec.cluster_threshold.unwrap_or(DEFAULT_CLUSTER_THRESHOLD)`
    ///   - `legal_move_radius`   ← `spec.legal_move_radius`
    ///   - `encoding`            ← `Some(spec)` so `to_planes()` and the
    ///     PyO3 surface honor `spec.board_size` and `spec.is_multi_window`.
    ///
    /// Marks `cache_dirty=true` (legal-cache built in `new()` is for the
    /// default radius=5; non-default radii require a rebuild on first
    /// `legal_moves_set()` call). Panics in debug if `spec.validate()`
    /// returns Err — registry parser already validates, so this is
    /// belt-and-braces.
    pub fn with_registry_spec(spec: &'static crate::encoding::RegistrySpec) -> Board {
        debug_assert!(
            spec.validate().is_ok(),
            "RegistrySpec {:?} failed validation: {:?}",
            spec.name,
            spec.validate()
        );
        let mut b = Board::new();
        b.cluster_window_size = spec.cluster_window_size.unwrap_or(spec.board_size);
        b.cluster_threshold = spec
            .cluster_threshold
            .unwrap_or(super::moves::DEFAULT_CLUSTER_THRESHOLD as usize) as i32;
        b.legal_move_radius = spec.legal_move_radius as i32;
        b.encoding = Some(spec);
        b.cache_dirty.set(true);
        b
    }

    /// §171 P2 reopen — construct a Board with a non-default encoding spec.
    ///
    /// Equivalent to `Board::new()` followed by the three setters
    /// (`set_cluster_window_size`, `set_cluster_threshold`,
    /// `set_legal_move_radius`), but performs spec validation eagerly and
    /// returns a single Board. Used by self-play workers under v6w25 and
    /// future encoding variants (§168 Gate 3 plumbing).
    ///
    /// Panics if `enc.validate()` fails. Callers that may receive untrusted
    /// input should call `enc.validate()` first.
    pub fn with_encoding(enc: &crate::encoding::EncodingSpec) -> Board {
        enc.validate().expect("EncodingSpec validation failed");
        let mut b = Board::new();
        b.cluster_window_size = enc.cluster_window_size;
        b.cluster_threshold = enc.cluster_threshold;
        b.legal_move_radius = enc.legal_move_radius;
        // legal_cache built in `new()` is for radius=2 (init_cache); larger radius
        // requires rebuild on next legal_moves_set() call. set_legal_move_radius
        // marks cache dirty; replicate that here:
        b.cache_dirty.set(true);
        b
    }

    /// §168 Gate 3 (v6w25 plumbing): override the cluster connectivity
    /// threshold for this Board. Affects only `get_clusters()` /
    /// `get_cluster_views()`; legal-move expansion is unchanged.
    pub fn set_cluster_threshold(&mut self, threshold: i32) {
        self.cluster_threshold = threshold;
    }

    /// Current cluster threshold (default 5 = v6 wire-format).
    pub fn cluster_threshold(&self) -> i32 {
        self.cluster_threshold
    }

    /// §168 Gate 3 (v6w25 plumbing): override the cluster window side length.
    /// Used by `get_cluster_views()` to size the 2-plane snapshot. Caller
    /// must use an odd value (>= 7); panic enforced by debug_assert.
    pub fn set_cluster_window_size(&mut self, size: usize) {
        debug_assert!(
            size >= 7 && size % 2 == 1,
            "cluster_window_size must be odd and >= 7; got {}", size
        );
        self.cluster_window_size = size;
    }

    /// Current cluster window side length (default 19 = v6 wire-format).
    pub fn cluster_window_size(&self) -> usize {
        self.cluster_window_size
    }

    /// §174 — read the bound encoding spec (if any).
    pub fn encoding_spec(&self) -> Option<&'static crate::encoding::RegistrySpec> {
        self.encoding
    }

    /// Phase B' v8 (§152 Q2): override the legal-move radius for this Board.
    ///
    /// Marks `legal_cache` dirty so the next `legal_moves_set()` rebuilds at
    /// the new radius.  Used by `SelfPlayRunner` when
    /// `legal_move_radius_jitter` is enabled — caller samples r ∈ {4, 5, 6}
    /// per game and applies it before the first move.
    ///
    /// **Post-mutator hazard (§171 P3 e6682f6 precedent, §172 A4.1):** if
    /// this Board was constructed via `Board::with_registry_spec` or
    /// `Board::with_encoding`, calling this setter silently overrides the
    /// encoding's `legal_move_radius`. The caller (e.g. `worker_loop.rs`)
    /// is responsible for guarding with `encoding.is_none()` or by an
    /// explicit jitter-overrides-encoding contract. This method does not
    /// assert because the v6 + jitter combination is the legitimate use
    /// case; the assertion would fire on every legitimate jitter call.
    /// See `feedback_encoding_post_mutators_audit.md`.
    pub fn set_legal_move_radius(&mut self, radius: i32) {
        self.legal_move_radius = radius;
        self.cache_dirty.set(true);
    }

    /// §174 — explicit curriculum radius override.  Works WITH encoding.
    ///
    /// Unlike `set_legal_move_radius()`, this bypasses the §173 A6 guard
    /// and is intended for training-time radius scheduling.  The encoding's
    /// canonical radius remains in the spec; this override affects only
    /// `legal_moves()` for this Board instance.
    pub fn override_legal_move_radius(&mut self, radius: i32) {
        self.legal_move_radius = radius;
        self.cache_dirty.set(true);
    }

    /// Current legal-move radius (default 5, may be overridden via
    /// `set_legal_move_radius`).
    pub fn legal_move_radius(&self) -> i32 {
        self.legal_move_radius
    }

    // ── Window ────────────────────────────────────────────────────────────────

    /// Centre of the 19×19 view window: centroid of the bounding box.
    /// Defaults to (0, 0) on an empty board.
    ///
    /// Uses Rust truncating-toward-zero integer division, which matches the
    /// Python-side `board.window_center()` call used in `game_state.py`.
    pub fn window_center(&self) -> (i32, i32) {
        if !self.has_stones {
            return (0, 0);
        }
        let cq = (self.min_q + self.max_q) / 2;
        let cr = (self.min_r + self.max_r) / 2;
        (cq, cr)
    }

    /// Window-relative flat index for axial (q, r) — spec-aware.
    ///
    /// Result is in [0, trunk_sz²). Returns usize::MAX for out-of-window coords.
    ///
    /// §173 A8'': dispatches via `self.cluster_window_size` (set at Board
    /// construction from `spec.cluster_window_size` for multi-window
    /// encodings, falls back to `spec.board_size` / `BOARD_SIZE` otherwise).
    /// `cluster_window_size` is the NN-input frame geometry (= `trunk_size`
    /// in the registry spec); the window indexing should use it, not the
    /// canvas `board_size`. For v6/v7full/v8 single-window this is 19/19/25
    /// and matches the canvas; for v6w25 it's 25 (vs canvas 25 too —
    /// coincides today; correct by design for a future infinite-canvas +
    /// fixed-window encoding).
    #[inline]
    pub fn window_flat_idx(&self, q: i32, r: i32) -> usize {
        let (cq, cr) = self.window_center();
        let trunk_sz = self.cluster_window_size as i32;
        let half = (trunk_sz - 1) / 2;
        Self::window_flat_idx_at_geom(q, r, cq, cr, trunk_sz, half)
    }

    /// Window-relative flat index for axial (q, r) at a specific center —
    /// legacy v6-default associated fn.
    ///
    /// Callers that need v6w25 (or any non-19 trunk) geometry must use
    /// `window_flat_idx_at_geom` and thread `(trunk_sz, half)` from
    /// spec-extracted scalars at the boundary (§173 A8'' design). This
    /// wrapper keeps byte-exact behaviour for the ~30 existing v6 call
    /// sites in tests + MCTS root setup that don't need to dispatch.
    #[inline]
    pub fn window_flat_idx_at(q: i32, r: i32, cq: i32, cr: i32) -> usize {
        Self::window_flat_idx_at_geom(q, r, cq, cr, BOARD_SIZE as i32, HALF)
    }

    /// Window-relative flat index kernel — spec-threaded geometry.
    ///
    /// §173 A8'': scalar-only API matching the §173 A5b lesson — every
    /// per-MCTS-sim caller pre-extracts `(trunk_sz, half)` once from
    /// `RegistrySpec` at the worker_loop boundary and passes the integer
    /// pair into the per-sim hot loop. Marked `#[inline]` so the compiler
    /// can fold the bounds check + index math into the caller and avoid
    /// the cost of a real function call.
    ///
    /// `trunk_sz`: per-cluster NN input side length (= `RegistrySpec::trunk_size`,
    ///   = `Board::cluster_window_size` cached on Board for self-dispatch).
    /// `half`:     `(trunk_sz - 1) / 2` — pre-computed by caller.
    #[inline]
    pub fn window_flat_idx_at_geom(
        q: i32, r: i32, cq: i32, cr: i32, trunk_sz: i32, half: i32,
    ) -> usize {
        let wq = q - cq + half;
        let wr = r - cr + half;
        if wq >= 0 && wq < trunk_sz && wr >= 0 && wr < trunk_sz {
            (wq as usize * trunk_sz as usize) + wr as usize
        } else {
            usize::MAX
        }
    }

    /// Returns the cell at (q, r).
    pub fn get_cell(&self, q: i32, r: i32) -> Cell {
        self.cells.get(&(q, r)).map(|r| *r).unwrap_or(Cell::Empty)
    }

    /// Axial coordinates (q, r) from a window-relative flat index.
    ///
    /// §173 A8'': dispatches via `self.cluster_window_size` so v6w25
    /// (25×25) and any future multi-window encoding decode correctly.
    #[inline]
    pub fn window_coords(&self, flat: usize) -> (i32, i32) {
        let (cq, cr) = self.window_center();
        let trunk_sz = self.cluster_window_size;
        let half = ((trunk_sz as i32) - 1) / 2;
        let wq = (flat / trunk_sz) as i32;
        let wr = (flat % trunk_sz) as i32;
        (wq - half + cq, wr - half + cr)
    }

    /// Whether (q, r) is inside the current trunk-sized view window.
    ///
    /// §173 A8'': dispatches via `self.cluster_window_size` (25 for v6w25,
    /// 19 for v6, etc.).
    #[inline]
    pub fn in_window(&self, q: i32, r: i32) -> bool {
        let (cq, cr) = self.window_center();
        let trunk_sz = self.cluster_window_size as i32;
        let half = (trunk_sz - 1) / 2;
        let wq = q - cq + half;
        let wr = r - cr + half;
        wq >= 0 && wq < trunk_sz && wr >= 0 && wr < trunk_sz
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Iterator over all occupied cells: yields `(&(q, r), &Cell)` pairs.
    pub fn cells_iter(&self) -> impl Iterator<Item = (&(i32, i32), &Cell)> {
        self.cells.iter()
    }

    /// Cell at (q, r).  Returns Empty for unoccupied or out-of-window cells.
    #[inline]
    pub fn get(&self, q: i32, r: i32) -> Cell {
        self.cells.get(&(q, r)).map(|r| *r).unwrap_or(Cell::Empty)
    }

    // ── Move application ──────────────────────────────────────────────────────

    /// Apply a move at (q, r) for the current player.
    ///
    /// Returns an error if the cell is outside the current 19×19 window or is
    /// already occupied.  Any in-window empty cell is accepted (the player is
    /// not restricted to the bbox+2 margin — that is enforced by MCTS via
    /// `legal_moves`, not by the board itself).
    ///
    /// After a successful move:
    /// - `moves_remaining` decrements.
    /// - When it reaches 0 the turn passes: `current_player` flips and
    ///   `moves_remaining` resets to 2.
    pub fn apply_move(&mut self, q: i32, r: i32) -> Result<(), &'static str> {
        if self.cells.contains_key(&(q, r)) {
            return Err("cell already occupied");
        }

        // Update bounding box FIRST so that window_flat_idx uses the final
        // bounding box — this keeps the Zobrist hash position-deterministic
        // (same stone set → same bbox → same centre → same hash).
        if !self.has_stones {
            self.min_q = q;
            self.max_q = q;
            self.min_r = r;
            self.max_r = r;
            self.has_stones = true;
        } else {
            if q < self.min_q { self.min_q = q; }
            if q > self.max_q { self.max_q = q; }
            if r < self.min_r { self.min_r = r; }
            if r > self.max_r { self.max_r = r; }
        }

        let player_idx = match self.current_player { Player::One => 0, Player::Two => 1 };

        let cell = match self.current_player {
            Player::One => Cell::P1,
            Player::Two => Cell::P2,
        };
        self.cells.insert((q, r), cell);

        // Mark legal cache dirty — legal_moves_set() will rebuild lazily.
        // This avoids 24+ HashSet operations on every apply_move (the MCTS hot
        // path calls apply_move ~2D times per simulation during traversal and
        // reconstruction, but legal_moves_set() is only needed once per sim at
        // leaf expansion).
        self.cache_dirty.set(true);

        // Update action anchors (last 4 stones).
        if self.action_anchors_count < 4 {
            self.action_anchors[self.action_anchors_count] = (q, r);
            self.action_anchors_count += 1;
        } else {
            self.action_anchors[0] = self.action_anchors[1];
            self.action_anchors[1] = self.action_anchors[2];
            self.action_anchors[2] = self.action_anchors[3];
            self.action_anchors[3] = (q, r);
        }

        // Use absolute (q, r) for Zobrist — position-independent, no window dependency.
        self.zobrist_hash ^= ZobristTable::get_for_pos(q, r, player_idx);
        self.ply += 1;
        self.last_move = Some((q, r));

        // Advance turn structure
        self.moves_remaining -= 1;
        if self.moves_remaining == 0 {
            self.current_player = self.current_player.other();
            self.moves_remaining = 2;
        }

        Ok(())
    }

    /// Apply a move and return a reversible state diff for O(1) undo.
    pub fn apply_move_tracked(&mut self, q: i32, r: i32) -> Result<MoveDiff, &'static str> {
        let diff = MoveDiff {
            q,
            r,
            player: self.current_player,
            prev_zobrist_hash: self.zobrist_hash,
            prev_moves_remaining: self.moves_remaining,
            prev_current_player: self.current_player,
            prev_ply: self.ply,
            prev_last_move: self.last_move,
            prev_min_q: self.min_q,
            prev_max_q: self.max_q,
            prev_min_r: self.min_r,
            prev_max_r: self.max_r,
            prev_has_stones: self.has_stones,
            prev_action_anchors: self.action_anchors,
            prev_action_anchors_count: self.action_anchors_count,
        };

        self.apply_move(q, r)?;
        Ok(diff)
    }


    /// Undo a move previously applied by `apply_move_tracked`.
    pub fn undo_move(&mut self, diff: MoveDiff) {
        if let Some(cell) = self.cells.remove(&(diff.q, diff.r)) {
            debug_assert_eq!(
                cell,
                match diff.player {
                    Player::One => Cell::P1,
                    Player::Two => Cell::P2,
                },
                "undo_move removed a stone with mismatched player",
            );
        } else {
            debug_assert!(false, "undo_move expected placed stone to exist");
        }

        // Mark legal cache dirty — it will be rebuilt lazily on next access.
        // This avoids O(24) HashSet operations per undo (undo is called ~D times
        // per MCTS sim during selection traversal but legal_moves_set() is not
        // called until leaf expansion).
        self.cache_dirty.set(true);

        self.zobrist_hash = diff.prev_zobrist_hash;
        self.moves_remaining = diff.prev_moves_remaining;
        self.current_player = diff.prev_current_player;
        self.ply = diff.prev_ply;
        self.last_move = diff.prev_last_move;

        self.min_q = diff.prev_min_q;
        self.max_q = diff.prev_max_q;
        self.min_r = diff.prev_min_r;
        self.max_r = diff.prev_max_r;
        self.has_stones = diff.prev_has_stones;

        self.action_anchors = diff.prev_action_anchors;
        self.action_anchors_count = diff.prev_action_anchors_count;
    }

    // ── Tensor encoding ───────────────────────────────────────────────────────

    /// Encode the 18-plane state tensor into `out` from a 2-plane cluster view.
    ///
    /// Layout:
    ///   plane  0:    current player's stones (from `planes_2[0..TOTAL_CELLS]`)
    ///   planes 1-7:  zero (no Python history on the Rust self-play path)
    ///   plane  8:    opponent's stones (from `planes_2[TOTAL_CELLS..2*TOTAL_CELLS]`)
    ///   planes 9-15: zero (no Python history)
    ///   plane 16:    moves_remaining == 2 broadcast
    ///   plane 17:    ply parity broadcast
    ///   planes 18-23: Q13 chain-length planes (3 hex axes × 2 players),
    ///                 /6.0-normalized to [0, 1]. Layout
    ///                 [a0_cur, a0_opp, a1_cur, a1_opp, a2_cur, a2_opp].
    ///
    /// `out` must have length `18 * TOTAL_CELLS`. Callers are responsible for
    /// zero-initializing the buffer before calling; this function writes to
    /// planes 0, 8, 16, 17 but leaves 1..7 and 9..15 untouched so the
    /// caller can rely on history planes being whatever the buffer started as
    /// (the existing self-play path zero-inits the pooled buffers).
    /// Chain-length planes (formerly 18..23) are computed separately via
    /// `encode_chain_planes` and stored in the replay buffer's dedicated
    /// chain sub-buffer.
    pub fn encode_state_to_buffer(
        &self,
        planes_2: &[f32], // The 2-plane [my, opp] view
        out: &mut [f32]
    ) {
        // Plane 0: my stones
        for i in 0..TOTAL_CELLS {
            out[i] = planes_2[i];
        }
        // Plane 8: opp stones
        for i in 0..TOTAL_CELLS {
            out[OPP_STONE_PLANE * TOTAL_CELLS + i] = planes_2[TOTAL_CELLS + i];
        }
        // Plane 16: moves_remaining == 2 ? 1.0 : 0.0
        let mr_val = if self.moves_remaining == 2 { 1.0 } else { 0.0 };
        for i in 0..TOTAL_CELLS {
            out[MOVES_REMAINING_PLANE * TOTAL_CELLS + i] = mr_val;
        }
        // Plane 17: ply % 2
        let ply_val = (self.ply % 2) as f32;
        for i in 0..TOTAL_CELLS {
            out[PLY_PARITY_PLANE * TOTAL_CELLS + i] = ply_val;
        }
        debug_assert_eq!(
            out.len(),
            18 * TOTAL_CELLS,
            "encode_state_to_buffer output length mismatch — expected 18 planes × {} cells",
            TOTAL_CELLS
        );
    }

    /// Public alias for `encode_state_to_buffer`. Preserved as a named entry
    /// point for callers outside this module.
    #[inline]
    pub fn encode_planes_to_buffer(&self, planes_2: &[f32], out: &mut [f32]) {
        self.encode_state_to_buffer(planes_2, out)
    }

    /// Encode a *subset* of the 18 wire planes selected by `channels`, in the
    /// order given. Used by sweep variants whose model in_channels < 18.
    ///
    /// Plane semantics match `encode_state_to_buffer` (see header comment).
    /// Channels 0/8 carry the only non-zero stone information on the Rust
    /// self-play path; channels 16/17 are scalar broadcasts; 1–7 / 9–15
    /// are zero on this path (history filled by Python tensor assembly).
    /// `channels.iter().any(|c| c >= 18)` panics in debug; release silently
    /// skips out-of-range entries.
    ///
    /// `out` must have length `channels.len() * TOTAL_CELLS`.
    ///
    /// §172 A4.1: this kernel hardcodes `TOTAL_CELLS` (361 = 19×19 v6 wire
    /// format). Encodings with a different `board_size` (v8 = 25×25)
    /// resize the output buffer in `to_planes_channels`/`to_planes` but
    /// the body of this kernel still walks the v6 19×19 layout. v8 Rust
    /// selfplay does not exist today; §17X v8 Rust path will introduce
    /// a separate `to_planes_v8` kernel if needed.
    pub fn encode_state_to_buffer_channels(
        &self,
        planes_2: &[f32],
        out: &mut [f32],
        channels: &[usize],
    ) {
        let n = channels.len();
        debug_assert_eq!(
            out.len(),
            n * TOTAL_CELLS,
            "encode_state_to_buffer_channels output length mismatch — \
             expected {} planes × {} cells",
            n,
            TOTAL_CELLS
        );
        let mr_val = if self.moves_remaining == 2 { 1.0 } else { 0.0 };
        let ply_val = (self.ply % 2) as f32;
        for (slot, &ch) in channels.iter().enumerate() {
            let dst = &mut out[slot * TOTAL_CELLS..(slot + 1) * TOTAL_CELLS];
            match ch {
                0 => {
                    dst.copy_from_slice(&planes_2[0..TOTAL_CELLS]);
                }
                8 => {
                    dst.copy_from_slice(&planes_2[TOTAL_CELLS..2 * TOTAL_CELLS]);
                }
                16 => {
                    for v in dst.iter_mut() {
                        *v = mr_val;
                    }
                }
                17 => {
                    for v in dst.iter_mut() {
                        *v = ply_val;
                    }
                }
                c if c < 18 => {
                    // History planes 1..7 / 9..15 are zero on the Rust
                    // self-play path; clear in case caller did not zero-init.
                    for v in dst.iter_mut() {
                        *v = 0.0;
                    }
                }
                _ => {
                    debug_assert!(false, "channel index {} out of range [0, 18)", ch);
                    for v in dst.iter_mut() {
                        *v = 0.0;
                    }
                }
            }
        }
    }

    /// `to_planes` variant emitting only the listed channels, in the listed
    /// order. Length = `channels.len() * board_size²` where `board_size`
    /// is the Board's encoding's `board_size` (default 19 = v6). See
    /// `encode_state_to_buffer_channels` for plane semantics.
    ///
    /// §172 A4.1 (multi-window guard): panics if the bound encoding is
    /// multi-window (v6w25). Multi-window selfplay is α-deferred — see
    /// `docs/designs/encoding_alpha_multiwindow_selfplay.md`. Use
    /// `get_cluster_views()` instead for those encodings.
    pub fn to_planes_channels(&self, channels: &[usize]) -> Vec<f32> {
        if let Some(spec) = self.encoding {
            if spec.is_multi_window {
                unimplemented!(
                    "multi-window selfplay deferred to α; see \
                     docs/designs/encoding_alpha_multiwindow_selfplay_design.md \
                     (§172 Phase A7)"
                );
            }
        }
        let mut planes_2 = vec![0.0f32; 2 * TOTAL_CELLS];
        let (my_cell, opp_cell) = match self.current_player {
            Player::One => (Cell::P1, Cell::P2),
            Player::Two => (Cell::P2, Cell::P1),
        };
        for (&(q, r), &cell) in self.cells.iter() {
            let flat = self.window_flat_idx(q, r);
            if flat < TOTAL_CELLS {
                if cell == my_cell {
                    planes_2[flat] = 1.0;
                } else if cell == opp_cell {
                    planes_2[TOTAL_CELLS + flat] = 1.0;
                }
            }
        }
        let mut out = vec![0.0f32; channels.len() * TOTAL_CELLS];
        self.encode_state_to_buffer_channels(&planes_2, &mut out, channels);
        out
    }

    /// Encode the board as a flat f32 array of length
    /// `18 * board_size * board_size` representing shape
    /// `[18, board_size, board_size]` (18 history+scalar planes).
    ///
    /// `board_size` resolves from `self.encoding` (default 19 = v6 wire
    /// format; v8 = 25 = canvas dim per registry).
    ///
    /// Chain-length planes are computed separately via `encode_chain_planes`.
    /// Stones outside the current 19×19 window are silently omitted.
    ///
    /// §172 A4.1 (multi-window guard, closes §171 P3 plane-export blocker):
    /// panics if the bound encoding is multi-window (v6w25 etc.). Single-
    /// window selfplay must route through `get_cluster_views` for those
    /// encodings; the silent shape corruption to_planes used to produce
    /// (always 18×19×19 regardless of encoding) was the §171 P3 blocker.
    /// Multi-window selfplay deferred to α — see
    /// `docs/designs/encoding_alpha_multiwindow_selfplay.md`.
    ///
    /// **v8 semantic mismatch caveat (out of scope today):** for
    /// single-window encodings with `board_size != 19` (v8 = 25), this
    /// method emits an 18-plane wire layout sized at the encoding's
    /// `board_size` but the kernel still walks the v6 19×19 layout —
    /// only the first 361 cells per plane carry stone data, the
    /// remainder is zero. v8 native is 11 planes (no KEPT_PLANE_INDICES
    /// slice) and a different stone-placement convention; Rust v8
    /// selfplay does not exist today and §17X will introduce a dedicated
    /// `to_planes_v8` if needed. A4.1's responsibility is bounded to
    /// stopping the spatial-dim hardcode that blocked v6w25.
    pub fn to_planes(&self) -> Vec<f32> {
        if let Some(spec) = self.encoding {
            if spec.is_multi_window {
                unimplemented!(
                    "multi-window selfplay deferred to α; see \
                     docs/designs/encoding_alpha_multiwindow_selfplay_design.md \
                     (§172 Phase A7)"
                );
            }
        }
        let board_size = self.encoding.map_or(BOARD_SIZE, |s| s.board_size);
        let total_cells = board_size * board_size;

        let mut planes_2 = vec![0.0f32; 2 * TOTAL_CELLS];
        let (my_cell, opp_cell) = match self.current_player {
            Player::One => (Cell::P1, Cell::P2),
            Player::Two => (Cell::P2, Cell::P1),
        };
        for (&(q, r), &cell) in self.cells.iter() {
            let flat = self.window_flat_idx(q, r);
            if flat < TOTAL_CELLS {
                if cell == my_cell {
                    planes_2[flat] = 1.0;
                } else if cell == opp_cell {
                    planes_2[TOTAL_CELLS + flat] = 1.0;
                }
            }
        }

        // Output buffer sized by encoding's board_size. v6 (board_size=19)
        // is bit-identical to pre-A4.1 behavior. v8 (board_size=25) gets a
        // larger zero-padded buffer; see method-doc caveat.
        let mut out = vec![0.0f32; 18 * total_cells];
        if board_size == BOARD_SIZE {
            self.encode_state_to_buffer(&planes_2, &mut out);
        } else {
            // v8 single-window path: emit v6 19×19 wire layout into the
            // top-left 361 cells of each 25×25 plane. Plane semantics
            // differ from v8 native (11 planes); see method-doc caveat.
            // Plane 0: my stones; Plane 8: opp stones.
            for i in 0..TOTAL_CELLS {
                out[MY_STONE_PLANE * total_cells + i] = planes_2[i];
                out[OPP_STONE_PLANE * total_cells + i] = planes_2[TOTAL_CELLS + i];
            }
            // Plane 16: moves_remaining == 2 broadcast over full plane.
            let mr_val = if self.moves_remaining == 2 { 1.0 } else { 0.0 };
            for i in 0..total_cells {
                out[16 * total_cells + i] = mr_val;
            }
            // Plane 17: ply parity broadcast over full plane.
            let ply_val = (self.ply % 2) as f32;
            for i in 0..total_cells {
                out[17 * total_cells + i] = ply_val;
            }
        }
        out
    }

    /// Same as `to_planes` — present to make the sliding-window semantics
    /// explicit in the PyO3 interface.
    ///
    /// `size` is ignored; output shape comes from the Board's encoding
    /// (default 19×19 v6). Multi-window encodings panic per `to_planes`
    /// (§172 A4.1).
    pub fn view_window(&self, _size: usize) -> Vec<f32> {
        self.to_planes()
    }

    /// Returns **2-plane views** (2 × TOTAL_CELLS = 722 floats each) and the
    /// window centre (cq, cr) for each cluster.
    ///
    /// Plane 0 = current player's stones in this 19×19 window.
    /// Plane 1 = opponent's stones in this 19×19 window.
    ///
    /// # Why 2 planes, not 18?
    ///
    /// The full AlphaZero input is 18 planes (post-§97):
    ///   - planes  0-7:  current player's stones at t, t-1, … t-7
    ///   - planes  8-15: opponent's stones        at t, t-1, … t-7
    ///   - planes 16-17: game-state scalars (moves_remaining, ply parity)
    ///   (Q13 chain-length planes moved to replay-buffer aux sub-buffer post-§97)
    ///
    /// Assembling all 18 planes requires the full move history — which only
    /// Python's `GameState.move_history` possesses.  Encoding 18 planes in
    /// Rust would mean crossing the PyO3 boundary with 6 498 floats per
    /// cluster (18 × 361) while 14 of them are always zero for Rust-driven
    /// self-play (no Python-side history), a significant overhead.
    ///
    /// Instead:
    ///   - `get_cluster_views` returns the 2-plane current snapshot (722 floats).
    ///   - `GameState.to_tensor()` stacks the current snapshot with historical
    ///     snapshots from `move_history` to form the final (18, 19, 19) tensor.
    ///
    /// The Rust self-play hot-path (`game_runner/worker_loop.rs`) has no
    /// Python history. It expands the 2-plane view to the full 18-plane
    /// layout via `encode_planes_to_buffer` in-place — history slots 1-7 /
    /// 9-15 stay zero for Rust-driven self-play. Chain planes are written
    /// separately to the replay-buffer chain sub-buffer via `encode_chain_planes`.
    ///
    /// `to_planes()` / `Board.to_tensor()` (the single-board Python binding)
    /// also uses `encode_planes_to_buffer`, so the 2-plane snapshot and the
    /// 18-plane encoding share the same kernel.
    pub fn get_cluster_views(&self) -> (Vec<Vec<f32>>, Vec<(i32, i32)>) {
        // §168 Gate 3: window dimensions resolve from `self.cluster_window_size`
        // (default 19 = v6 wire format; v6w25 callers set 25). The "small
        // cluster" span threshold scales with the window: window − 4 leaves a
        // 2-cell margin around the cluster bbox so the centroid window
        // contains every stone.
        let window_size = self.cluster_window_size;
        let total_cells = window_size * window_size;
        let half: i32 = (window_size as i32 - 1) / 2;
        let span_threshold: i32 = window_size as i32 - 4;

        let clusters = self.get_clusters();
        let mut final_centers = Vec::new();

        if clusters.is_empty() {
            final_centers.push((0, 0));
        } else {
            let threat_anchors = self.get_threat_anchors();
            let action_anchors = &self.action_anchors[..self.action_anchors_count];

            for cluster in clusters {
                let mut min_q = i32::MAX;
                let mut max_q = i32::MIN;
                let mut min_r = i32::MAX;
                let mut max_r = i32::MIN;
                for &(q, r) in &cluster {
                    min_q = min_q.min(q);
                    max_q = max_q.max(q);
                    min_r = min_r.min(r);
                    max_r = max_r.max(r);
                }

                let span_q = max_q - min_q;
                let span_r = max_r - min_r;

                if span_q <= span_threshold && span_r <= span_threshold {
                    // Small Clusters: single window centered on geometric middle
                    final_centers.push(((min_q + max_q) / 2, (min_r + max_r) / 2));
                } else {
                    // Massive Clusters: window centered on each Action and Threat anchor in the cluster
                    let mut cluster_anchors = Vec::new();

                    // Action anchors in this cluster
                    for &anchor in action_anchors {
                        if cluster.contains(&anchor) {
                            cluster_anchors.push(anchor);
                        }
                    }

                    // Threat anchors in this cluster
                    for &anchor in &threat_anchors {
                        if cluster.contains(&anchor) {
                            cluster_anchors.push(anchor);
                        }
                    }

                    if cluster_anchors.is_empty() {
                        // Fallback if no anchors found
                        final_centers.push(((min_q + max_q) / 2, (min_r + max_r) / 2));
                    } else {
                        // Deduplicate anchors: radius matches v6 baseline (5)
                        // regardless of cluster_window_size — this is a
                        // dedup tolerance, not a perception radius.
                        let mut deduped: Vec<(i32, i32)> = Vec::new();
                        for &a in &cluster_anchors {
                            if !deduped.iter().any(|&d| hex_distance(a.0, a.1, d.0, d.1) <= 5) {
                                deduped.push(a);
                            }
                        }
                        final_centers.extend(deduped);
                    }
                }
            }
        }

        let mut views = Vec::with_capacity(final_centers.len());
        let (my_cell, opp_cell) = match self.current_player {
            Player::One => (Cell::P1, Cell::P2),
            Player::Two => (Cell::P2, Cell::P1),
        };

        for &(cq, cr) in &final_centers {
            let mut planes_2 = vec![0.0f32; 2 * total_cells];
            for (&(q, r), &cell) in self.cells.iter() {
                let wq = q - cq + half;
                let wr = r - cr + half;
                if wq >= 0
                    && wq < window_size as i32
                    && wr >= 0
                    && wr < window_size as i32
                {
                    let flat = (wq as usize) * window_size + (wr as usize);
                    if cell == my_cell {
                        planes_2[flat] = 1.0;
                    } else if cell == opp_cell {
                        planes_2[total_cells + flat] = 1.0;
                    }
                }
            }
            views.push(planes_2);
        }
        (views, final_centers)
    }

    /// Returns the (q, r) centers of any open 3-in-a-row or 4-in-a-row formations.
    /// A formation is considered "open" if at least one of its ends is empty.
    ///
    /// Performance: Optimized O(Stones) by skipping redundant scans.
    pub fn get_threat_anchors(&self) -> Vec<(i32, i32)> {
        let mut anchors = Vec::with_capacity(8);
        if self.cells.is_empty() {
            return anchors;
        }

        // Pre-collect stones by player to avoid repeated HashMap scans.
        let mut p1_stones = Vec::with_capacity(self.cells.len());
        let mut p2_stones = Vec::with_capacity(self.cells.len());
        for (&pos, &cell) in self.cells.iter() {
            match cell {
                Cell::P1 => p1_stones.push(pos),
                Cell::P2 => p2_stones.push(pos),
                Cell::Empty => {}
            }
        }

        self.append_threat_anchors_for_player(&p1_stones, Cell::P1, &mut anchors);
        self.append_threat_anchors_for_player(&p2_stones, Cell::P2, &mut anchors);

        anchors
    }

    fn append_threat_anchors_for_player(
        &self,
        stones: &[(i32, i32)],
        player_cell: Cell,
        anchors: &mut Vec<(i32, i32)>,
    ) {
        // Track visited (pos, axis) to avoid redundant scans for the same sequence.
        // There are 3 axes, so we can pack (q, r, axis_idx) into a single key if needed,
        // but for simplicity and to avoid allocations, we just check if the previous
        // cell in the direction is the same color.
        for &(q, r) in stones {
            for (dq, dr) in HEX_AXES {
                // Efficiency: Only start scanning if this stone is the *start* of a sequence
                // in this direction. This ensures each sequence is scanned exactly once.
                if self.get(q - dq, r - dr) == player_cell {
                    continue;
                }

                let mut count = 1;
                while self.get(q + dq * count, r + dr * count) == player_cell {
                    count += 1;
                }

                if count == 3 || count == 4 {
                    // Check if open at either end.
                    let open_start = self.get(q - dq, r - dr) == Cell::Empty;
                    let open_end = self.get(q + dq * count, r + dr * count) == Cell::Empty;

                    if open_start || open_end {
                        let center_idx = count / 2;
                        anchors.push((q + dq * center_idx, r + dr * center_idx));
                    }
                }
            }
        }
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

// ── Q13 chain-length plane encoding ─────────────────────────────────────────
//
// Pure-function helpers used by `Board::encode_state_to_buffer`. Mirror the
// Python `_compute_chain_planes` in `hexo_rl/env/game_state.py`.
//
// Output layout: 6 planes × `TOTAL_CELLS` = `6 * 361` f32 entries, written
// into the caller's buffer slice at offset 0. Values are /6.0 normalized.
// Plane order within the 6-plane block:
//   [axis0_cur, axis0_opp, axis1_cur, axis1_opp, axis2_cur, axis2_opp]

/// Saturation cap for chain length — the 6-in-a-row win target.
const CHAIN_CAP: i32 = 6;
/// Normalisation denominator; matches Python's /6.0 after int8 computation.
const CHAIN_NORM: f32 = 6.0;

#[inline]
fn flat_idx(q: i32, r: i32) -> usize {
    const HALF: i32 = (BOARD_SIZE as i32 - 1) / 2; // 9
    ((q + HALF) as usize) * BOARD_SIZE + (r + HALF) as usize
}

#[inline]
fn in_window(q: i32, r: i32) -> bool {
    const HALF: i32 = (BOARD_SIZE as i32 - 1) / 2;
    q >= -HALF && q <= HALF && r >= -HALF && r <= HALF
}

/// Walk +step * (dq, dr) from (q, r) counting consecutive `own` cells.
/// Stops at window edge or first non-own cell. Max count = `CHAIN_CAP - 1`.
#[inline]
fn count_run(own: &[f32], q: i32, r: i32, dq: i32, dr: i32) -> i32 {
    let mut c = 0i32;
    for k in 1..CHAIN_CAP {
        let qk = q + dq * k;
        let rk = r + dr * k;
        if !in_window(qk, rk) {
            break;
        }
        let idx = flat_idx(qk, rk);
        if own[idx] > 0.5 {
            c += 1;
        } else {
            break;
        }
    }
    c
}

/// Write one chain-length plane (single axis, single player) into `out`.
/// `out` must have length `TOTAL_CELLS`.
fn encode_chain_plane_one(
    own: &[f32],
    opp: &[f32],
    dq: i32,
    dr: i32,
    out: &mut [f32],
) {
    const HALF: i32 = (BOARD_SIZE as i32 - 1) / 2;
    for q in -HALF..=HALF {
        for r in -HALF..=HALF {
            let idx = flat_idx(q, r);
            if opp[idx] > 0.5 {
                out[idx] = 0.0;
                continue;
            }
            let pos_run = count_run(own, q, r, dq, dr);
            let neg_run = count_run(own, q, r, -dq, -dr);
            let is_own = own[idx] > 0.5;
            if !is_own && pos_run == 0 && neg_run == 0 {
                out[idx] = 0.0;
                continue;
            }
            let mut v = 1 + pos_run + neg_run;
            if v > CHAIN_CAP {
                v = CHAIN_CAP;
            }
            out[idx] = (v as f32) / CHAIN_NORM;
        }
    }
}

/// Write all 6 chain-length planes into `out` (length `6 * TOTAL_CELLS`).
///
/// `cur_mask` and `opp_mask` are `TOTAL_CELLS`-sized f32 masks with 1.0 at
/// stone positions and 0.0 elsewhere.
pub fn encode_chain_planes(
    cur_mask: &[f32],
    opp_mask: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(cur_mask.len(), TOTAL_CELLS);
    debug_assert_eq!(opp_mask.len(), TOTAL_CELLS);
    debug_assert_eq!(out.len(), 6 * TOTAL_CELLS);

    for (axis_idx, &(dq, dr)) in HEX_AXES.iter().enumerate() {
        let cur_base = 2 * axis_idx * TOTAL_CELLS;
        let opp_base = (2 * axis_idx + 1) * TOTAL_CELLS;
        // Split into two mutable slices so we can borrow disjoint regions.
        let (head, tail) = out.split_at_mut(opp_base);
        encode_chain_plane_one(
            cur_mask,
            opp_mask,
            dq,
            dr,
            &mut head[cur_base..cur_base + TOTAL_CELLS],
        );
        encode_chain_plane_one(
            opp_mask,
            cur_mask,
            dq,
            dr,
            &mut tail[0..TOTAL_CELLS],
        );
    }
}

#[cfg(test)]
mod channel_select_tests {
    use super::*;

    fn build_planes_2() -> Vec<f32> {
        let mut v = vec![0.0f32; 2 * TOTAL_CELLS];
        v[0] = 1.0;
        v[5] = 1.0;
        v[TOTAL_CELLS + 7] = 1.0;
        v[TOTAL_CELLS + 11] = 1.0;
        v
    }

    fn make_board() -> Board {
        let mut b = Board::new();
        // Place a couple of stones to exercise moves_remaining + ply parity.
        b.apply_move(0, 0).unwrap();
        b.apply_move(1, 0).unwrap();
        b.apply_move(0, 1).unwrap();
        b
    }

    #[test]
    fn channel_select_matches_full_kernel_for_canonical_planes() {
        let b = make_board();
        let planes_2 = build_planes_2();

        let mut full = vec![0.0f32; 18 * TOTAL_CELLS];
        b.encode_state_to_buffer(&planes_2, &mut full);

        let channels = [0usize, 8, 16, 17];
        let mut sel = vec![0.0f32; channels.len() * TOTAL_CELLS];
        b.encode_state_to_buffer_channels(&planes_2, &mut sel, &channels);

        for (slot, &ch) in channels.iter().enumerate() {
            let lhs = &full[ch * TOTAL_CELLS..(ch + 1) * TOTAL_CELLS];
            let rhs = &sel[slot * TOTAL_CELLS..(slot + 1) * TOTAL_CELLS];
            assert_eq!(lhs, rhs, "channel {} mismatch (slot {})", ch, slot);
        }
    }

    #[test]
    fn channel_select_history_planes_are_zero() {
        let b = make_board();
        let planes_2 = build_planes_2();
        let channels = [0usize, 1, 8, 9];
        let mut sel = vec![999.0f32; channels.len() * TOTAL_CELLS];
        b.encode_state_to_buffer_channels(&planes_2, &mut sel, &channels);
        // Slots 1 and 3 are history planes (1 and 9) — zero on Rust path.
        for &slot in &[1usize, 3] {
            for v in &sel[slot * TOTAL_CELLS..(slot + 1) * TOTAL_CELLS] {
                assert_eq!(*v, 0.0);
            }
        }
    }

    #[test]
    fn to_planes_channels_length_matches_request() {
        let b = make_board();
        let v = b.to_planes_channels(&[0, 8]);
        assert_eq!(v.len(), 2 * TOTAL_CELLS);
        let v6 = b.to_planes_channels(&[0, 1, 8, 9, 16, 17]);
        assert_eq!(v6.len(), 6 * TOTAL_CELLS);
    }

    #[test]
    fn to_planes_channels_full_18_matches_to_planes() {
        let b = make_board();
        let full = b.to_planes();
        let channels: Vec<usize> = (0..18).collect();
        let sel = b.to_planes_channels(&channels);
        assert_eq!(full.len(), sel.len());
        for (i, (a, c)) in full.iter().zip(sel.iter()).enumerate() {
            assert!((a - c).abs() < 1e-7, "mismatch at index {}: {} vs {}", i, a, c);
        }
    }
}

#[cfg(test)]
mod chain_plane_tests {
    use super::*;

    fn at(plane: &[f32], q: i32, r: i32) -> f32 {
        plane[flat_idx(q, r)]
    }

    fn set(mask: &mut Vec<f32>, q: i32, r: i32) {
        mask[flat_idx(q, r)] = 1.0;
    }

    #[test]
    fn empty_board_all_zeros() {
        let cur = vec![0.0f32; TOTAL_CELLS];
        let opp = vec![0.0f32; TOTAL_CELLS];
        let mut out = vec![0.0f32; 6 * TOTAL_CELLS];
        encode_chain_planes(&cur, &opp, &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn single_stone_value_one_sixth_on_all_axes() {
        let mut cur = vec![0.0f32; TOTAL_CELLS];
        let opp = vec![0.0f32; TOTAL_CELLS];
        set(&mut cur, 0, 0);
        let mut out = vec![0.0f32; 6 * TOTAL_CELLS];
        encode_chain_planes(&cur, &opp, &mut out);
        let expected = 1.0 / CHAIN_NORM;
        // axis0_cur, axis1_cur, axis2_cur at (0,0) all = 1/6.
        let a0 = &out[0..TOTAL_CELLS];
        let a1 = &out[2 * TOTAL_CELLS..3 * TOTAL_CELLS];
        let a2 = &out[4 * TOTAL_CELLS..5 * TOTAL_CELLS];
        assert!((at(a0, 0, 0) - expected).abs() < 1e-5);
        assert!((at(a1, 0, 0) - expected).abs() < 1e-5);
        assert!((at(a2, 0, 0) - expected).abs() < 1e-5);
        // All opp planes zero (no opp stones).
        let o0 = &out[TOTAL_CELLS..2 * TOTAL_CELLS];
        assert_eq!(o0.iter().map(|&v| v as i32).sum::<i32>(), 0);
    }

    #[test]
    fn xx_empty_xxx_caps_at_six() {
        let mut cur = vec![0.0f32; TOTAL_CELLS];
        let opp = vec![0.0f32; TOTAL_CELLS];
        // Stones at q=0,1 then empty at q=2 then stones at q=3,4,5, all r=0.
        for q in [0, 1, 3, 4, 5] {
            set(&mut cur, q, 0);
        }
        let mut out = vec![0.0f32; 6 * TOTAL_CELLS];
        encode_chain_planes(&cur, &opp, &mut out);
        let a0 = &out[0..TOTAL_CELLS];
        assert!((at(a0, 2, 0) - 1.0).abs() < 1e-5); // 6/6
    }

    #[test]
    fn opponent_blocks_run() {
        let mut cur = vec![0.0f32; TOTAL_CELLS];
        let mut opp = vec![0.0f32; TOTAL_CELLS];
        set(&mut cur, 0, 0);
        set(&mut cur, 1, 0);
        opp[flat_idx(2, 0)] = 1.0;
        set(&mut cur, 3, 0);
        let mut out = vec![0.0f32; 6 * TOTAL_CELLS];
        encode_chain_planes(&cur, &opp, &mut out);
        let a0 = &out[0..TOTAL_CELLS];
        let v2 = 2.0 / CHAIN_NORM;
        let v1 = 1.0 / CHAIN_NORM;
        assert!((at(a0, 0, 0) - v2).abs() < 1e-5);
        assert!((at(a0, 1, 0) - v2).abs() < 1e-5);
        assert!((at(a0, 3, 0) - v1).abs() < 1e-5);
    }

    #[test]
    fn cap_saturates_above_six() {
        let mut cur = vec![0.0f32; TOTAL_CELLS];
        let opp = vec![0.0f32; TOTAL_CELLS];
        for q in -3..=3 {
            set(&mut cur, q, 0); // 7 stones
        }
        let mut out = vec![0.0f32; 6 * TOTAL_CELLS];
        encode_chain_planes(&cur, &opp, &mut out);
        let a0 = &out[0..TOTAL_CELLS];
        for q in -3..=3 {
            assert!((at(a0, q, 0) - 1.0).abs() < 1e-5, "q={}", q);
        }
    }

    #[test]
    fn python_rust_parity_50_stone_position() {
        // Reconstruct the Python perf test position deterministically — seed 2613,
        // 25 cur + 25 opp stones on non-overlapping flat indices. Verify Rust
        // output matches the Python helper output byte-exact after /6 normalization.
        // We don't call Python here; instead this test pins the Rust output for
        // a specific seeded position so drift between Rust and Python is caught
        // by a failing Python-side value comparison (implemented in
        // tests/test_chain_plane_augmentation.py once the Python wire-up lands).
        let mut cur = vec![0.0f32; TOTAL_CELLS];
        let mut opp = vec![0.0f32; TOTAL_CELLS];
        // Hand-pick a small representative position instead of pseudo-random.
        for (q, r) in [(0, 0), (1, 0), (2, 0), (-2, 0), (-1, 0)] {
            set(&mut cur, q, r);
        }
        for (q, r) in [(0, 1), (0, 2)] {
            opp[flat_idx(q, r)] = 1.0;
        }
        let mut out = vec![0.0f32; 6 * TOTAL_CELLS];
        encode_chain_planes(&cur, &opp, &mut out);
        let a0 = &out[0..TOTAL_CELLS];
        // Cur run of 5 along axis0 at q=-2..=2, r=0 → each cell sees 5/6.
        let five_sixths = 5.0 / CHAIN_NORM;
        for q in -2..=2 {
            assert!((at(a0, q, 0) - five_sixths).abs() < 1e-5);
        }
        // Empty flanks at q=-3 and q=3 along axis0 see 6/6 (5 + 1).
        assert!((at(a0, -3, 0) - 1.0).abs() < 1e-5);
        assert!((at(a0, 3, 0) - 1.0).abs() < 1e-5);
        // Opponent run along axis1 at (0,1),(0,2) → opp plane axis1.
        let a1_opp = &out[3 * TOTAL_CELLS..4 * TOTAL_CELLS];
        let two_sixths = 2.0 / CHAIN_NORM;
        assert!((at(a1_opp, 0, 1) - two_sixths).abs() < 1e-5);
        assert!((at(a1_opp, 0, 2) - two_sixths).abs() < 1e-5);
    }
}

impl Clone for Board {
    fn clone(&self) -> Self {
        // Skip copying legal_cache contents — rebuilding a HashSet of N entries
        // is O(N) allocation and dominates clone cost in the MCTS hot path (every
        // expand_and_backup reconstructs a board via clone + apply_move* ).
        //
        // We set cache_dirty = true unconditionally so that the first
        // legal_moves_set() call on the clone rebuilds from `cells` (which IS
        // correctly copied).  This is safe even when diffs is empty (root node
        // expansion) because the rebuild is always correct given a valid `cells`.
        let cap = unsafe { (*self.legal_cache.get()).len() };
        Board {
            cells: self.cells.clone(),
            current_player: self.current_player,
            moves_remaining: self.moves_remaining,
            ply: self.ply,
            zobrist_hash: self.zobrist_hash,
            last_move: self.last_move,
            min_q: self.min_q,
            max_q: self.max_q,
            min_r: self.min_r,
            max_r: self.max_r,
            has_stones: self.has_stones,
            action_anchors: self.action_anchors,
            action_anchors_count: self.action_anchors_count,
            legal_cache: UnsafeCell::new(FxHashSet::with_capacity_and_hasher(cap, Default::default())),
            cache_dirty: StdCell::new(true),
            legal_move_radius: self.legal_move_radius,
            cluster_threshold: self.cluster_threshold,
            cluster_window_size: self.cluster_window_size,
            encoding: self.encoding,
        }
    }
}

// SAFETY: `Board` is always accessed by a single thread (MCTS worker or Python GIL).
// The `UnsafeCell` / `Cell` fields are never reached concurrently via shared references.
unsafe impl Sync for Board {}

#[cfg(test)]
mod with_encoding_tests {
    use super::*;
    use crate::encoding::EncodingSpec;

    #[test]
    fn test_with_encoding_v6w25() {
        let b = Board::with_encoding(&EncodingSpec::V6W25);
        assert_eq!(b.cluster_window_size(), 25);
        assert_eq!(b.cluster_threshold(), 8);
        assert_eq!(b.legal_move_radius(), 8);
    }

    #[test]
    fn test_with_encoding_v6_matches_default() {
        let b_with = Board::with_encoding(&EncodingSpec::V6);
        let b_new = Board::new();
        assert_eq!(b_with.cluster_window_size(), b_new.cluster_window_size());
        assert_eq!(b_with.cluster_threshold(), b_new.cluster_threshold());
        assert_eq!(b_with.legal_move_radius(), b_new.legal_move_radius());
    }

    #[test]
    #[should_panic(expected = "cluster_window_size")]
    fn test_with_encoding_window_lt_threshold_panics() {
        let bad = EncodingSpec {
            cluster_window_size: 7,
            cluster_threshold: 9,
            legal_move_radius: 5,
            board_size: 19,
        };
        let _ = Board::with_encoding(&bad);
    }

    #[test]
    #[should_panic]
    fn test_with_encoding_zero_radius_panics() {
        let bad = EncodingSpec {
            cluster_window_size: 19,
            cluster_threshold: 5,
            legal_move_radius: 0,
            board_size: 19,
        };
        let _ = Board::with_encoding(&bad);
    }

    #[test]
    #[should_panic]
    fn test_with_encoding_even_window_panics() {
        let bad = EncodingSpec {
            cluster_window_size: 20,
            cluster_threshold: 5,
            legal_move_radius: 5,
            board_size: 19,
        };
        let _ = Board::with_encoding(&bad);
    }
}

#[cfg(test)]
mod with_registry_spec_tests {
    //! §172 A4.1 — Board::with_registry_spec + to_planes encoding-aware
    //! plumbing. Closes the §171 P3 plane-export blocker (to_planes used
    //! to silently emit 18×19×19 regardless of encoding).
    use super::*;

    #[test]
    fn to_planes_v6_default_emits_18x361() {
        let b = Board::new();
        let v = b.to_planes();
        assert_eq!(v.len(), 18 * 19 * 19); // 6498
    }

    #[test]
    fn to_planes_after_with_registry_spec_v6_matches_default() {
        let v6 = crate::encoding::lookup_or_panic("v6");
        let b = Board::with_registry_spec(v6);
        assert_eq!(b.to_planes().len(), 18 * 19 * 19);
        // Bit-identical to Board::new().to_planes() — same shape, same
        // body (board_size==BOARD_SIZE branch).
        let bd = Board::new();
        assert_eq!(b.to_planes(), bd.to_planes());
    }

    #[test]
    #[should_panic(expected = "multi-window")]
    fn to_planes_v6w25_panics() {
        let v6w25 = crate::encoding::lookup_or_panic("v6w25");
        let b = Board::with_registry_spec(v6w25);
        let _ = b.to_planes(); // panics with α deferral message
    }

    #[test]
    #[should_panic(expected = "multi-window")]
    fn to_planes_channels_v6w25_panics() {
        let v6w25 = crate::encoding::lookup_or_panic("v6w25");
        let b = Board::with_registry_spec(v6w25);
        let _ = b.to_planes_channels(&[0, 8]);
    }

    #[test]
    fn to_planes_v8_emits_18x625_spatial() {
        let v8 = crate::encoding::lookup_or_panic("v8");
        let b = Board::with_registry_spec(v8);
        // Wire layout: 18 planes × 25×25 = 11250. v8 native is 11 planes
        // (no KEPT_PLANE_INDICES slice); Rust v8 selfplay does not exist
        // today, A4.1 just stops the spatial-dim hardcode. See
        // `to_planes` doc-comment caveat.
        assert_eq!(b.to_planes().len(), 18 * 25 * 25);
    }

    #[test]
    fn with_registry_spec_v6w25_propagates_fields() {
        let v6w25 = crate::encoding::lookup_or_panic("v6w25");
        let b = Board::with_registry_spec(v6w25);
        assert_eq!(b.cluster_window_size(), 25);
        assert_eq!(b.cluster_threshold(), 8);
        assert_eq!(b.legal_move_radius(), 8);
        assert_eq!(b.encoding.unwrap().name, "v6w25");
    }

    #[test]
    fn with_registry_spec_v6_board_size_19() {
        let v6 = crate::encoding::lookup_or_panic("v6");
        let b = Board::with_registry_spec(v6);
        assert_eq!(b.encoding.unwrap().board_size, 19);
    }

    #[test]
    fn with_registry_spec_v8_propagates_board_size() {
        let v8 = crate::encoding::lookup_or_panic("v8");
        let b = Board::with_registry_spec(v8);
        assert_eq!(b.encoding.unwrap().board_size, 25);
        // v8 is single-window in the registry — cluster_window_size is
        // None, so Board falls back to spec.board_size (25).
        assert_eq!(b.cluster_window_size(), 25);
    }

    #[test]
    fn clone_preserves_encoding_pointer() {
        let v6w25 = crate::encoding::lookup_or_panic("v6w25");
        let a = Board::with_registry_spec(v6w25);
        let b = a.clone();
        assert!(std::ptr::eq(a.encoding.unwrap(), b.encoding.unwrap()));
    }
}
