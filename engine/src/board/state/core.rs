use std::cell::{Cell as StdCell, UnsafeCell};
use fxhash::{FxHashMap, FxHashSet};
use super::super::zobrist::ZobristTable;

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
            legal_move_radius: super::super::moves::DEFAULT_LEGAL_MOVE_RADIUS,
            cluster_threshold: super::super::moves::DEFAULT_CLUSTER_THRESHOLD,
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
            .unwrap_or(super::super::moves::DEFAULT_CLUSTER_THRESHOLD as usize) as i32;
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
            "cluster_window_size must be odd and >= 7; got {size}"
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
        let cq = i32::midpoint(self.min_q, self.max_q);
        let cr = i32::midpoint(self.min_r, self.max_r);
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
        self.cells.get(&(q, r)).copied().unwrap_or(Cell::Empty)
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
        self.cells.get(&(q, r)).copied().unwrap_or(Cell::Empty)
    }

    // ── Move application ──────────────────────────────────────────────────────

    /// Apply a move at (q, r) for the current player.
    ///
    /// Returns `Err` only if the cell is already occupied. Post-§142 the
    /// board is conceptually infinite (`docs/rules/board-representation.md`)
    /// — `apply_move` performs no window or radius check, and any
    /// previously-empty (q, r) is accepted. Window / radius / bbox-margin
    /// constraints are the caller's responsibility (`legal_moves_set` for
    /// MCTS, `legal_move_radius` for self-play, etc.); this entry point is
    /// the unconditional cell-write primitive.
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
        if self.has_stones {
            if q < self.min_q { self.min_q = q; }
            if q > self.max_q { self.max_q = q; }
            if r < self.min_r { self.min_r = r; }
            if r > self.max_r { self.max_r = r; }
        } else {
            self.min_q = q;
            self.max_q = q;
            self.min_r = r;
            self.max_r = r;
            self.has_stones = true;
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
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
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
