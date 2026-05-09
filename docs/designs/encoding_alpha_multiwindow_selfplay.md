# α — Multi-window K-cluster Selfplay: Scope Memo

*Phase 4.5+ scope · Design doc: §172 Phase A7 · Implementation: §173+*

Date: 2026-05-09

---

### 1. Why α exists

§171 P3 pre-flight surfaced that v6w25 sustained selfplay is blocked at the Rust plane-projection layer. `Board::to_planes()` (engine/src/board/state.rs:641) hardcodes 18×19×19 output regardless of `cluster_window_size`. v6w25 pretrain works via `get_cluster_views()` (state.rs:703, honors cluster_window_size=25); selfplay calls `state.to_tensor()` (inference.py:53), the 19×19 single-window path only.

v6w25 models (policy_fc.weight = (626, 1250), trunk 25×25) cannot receive selfplay positions encoded at 19×19. α is the structural fix: replicate the pretrain multi-window K-cluster encoding path inside selfplay.

Cross-reference: §171 P3 blocker entry in docs/07_PHASE4_SPRINT_LOG.md, §170 P4 P1 NULL verdict (K-invariant value head), §169 A1 canonical pick (K-cluster + min_max pooling, v6w25).

---

### 2. High-level design shape

Three options were considered:

**Option i — K independent MCTS trees:** Run K full MCTS trees, one per cluster window. Policy prior from each tree's own NN call. Merge at root via vote or average. Problem: K × compute cost, no tree reuse, policy priors from different windows not combinable cleanly.

**Option ii — Single-pool aggregation (pretrain-compatible):** Single tree; policy logits averaged across K NN calls at each node expansion. Simpler but conflates distinct spatial contexts into one averaged prior — loses per-cluster signal.

**Option iii — Single tree, per-move cluster dispatch (chosen):**
- Single MCTS tree, unchanged PUCT backups.
- At each node expansion, each legal move is dispatched to its cluster's window NN call.
- Policy prior for move m = logit from the K-th window that owns move m's cluster.
- Value head: K-invariant (min-pool across K cluster forwards at root) — matches §170 P4 P1 lesson: K-invariant global signal does not add measurable lift; value path stays frozen, no new freezing needed.
- Buffer stores board state; K windows recomputed on read (avoids K×storage).

Why Option iii: one tree = unmodified PUCT backups. Per-move dispatch = spatially coherent policy priors. Value min-pool = §170 P4 P1 lesson respected. Buffer schema minimal change.

---

### 3. Required changes (estimated scope)

| Component | Change |
|---|---|
| Rust `engine/src/board/state.rs` | New `to_planes_windowed(window_idx, cluster_window_size)` API; existing `to_planes()` unchanged |
| Rust PyO3 `engine/src/lib.rs` | Expose `get_cluster_views()` path to selfplay workers |
| `hexo_rl/selfplay/inference.py` | K-per-position fan-out in inference batcher |
| `hexo_rl/selfplay/worker_loop.py` | Emit K cluster views per position; label each move with cluster id |
| `hexo_rl/selfplay/pool.py` | Results queue: K-aware batching |
| `hexo_rl/replay_buffer/` | Buffer schema: store board state, not encoded tensor; recompute on sample |
| `hexo_rl/mcts/` | Per-move cluster dispatch in node expansion |
| Tests | K-window round-trip, cluster dispatch correctness, value-head K-invariance regression |

Estimated effort: 1–2 weeks on laptop. Rust changes are modest (new API surface, existing `get_cluster_views()` logic reused). Python MCTS dispatch is the largest non-trivial piece.

---

### 4. Sequencing rationale

α implementation is sequenced AFTER §172 P3 v7full sustained smoke produces a value-drift fingerprint dataset. Rationale: derisk Phase D infrastructure on the encoder-agnostic surface (v7full, 19×19) before beginning the α-specific multi-week work. §170 P4 P1 showed value-path changes require careful ablation; the v7full smoke data provides a clean pre-α baseline fingerprint.

Sequence:
1. §172 P1–P2: encoding registry + `EncodingSpec` clean propagation (removes the scattered-encoding technical debt that made Layer 1 of the §171 P3 blocker possible).
2. §172 P3: v7full sustained smoke under clean encoding propagation.
3. §172 A7: α design doc (full spec, not scope memo).
4. §173+: α implementation sprints.

---

### 5. Forward pointers

- §172 Phase A7 design doc: `docs/designs/encoding_alpha_multiwindow_selfplay_design.md` (to be written in §172).
- §173+ implementation sprint: TBD branch.
- §171 P3 blocker: `docs/07_PHASE4_SPRINT_LOG.md` § §171 P3.
- §170 P4 P1 NULL verdict: same file § §170 P4 P1 (K-invariant global signal null result — motivates value min-pool invariant).
- §169 A1 canonical: same file § §169 (K-cluster + min_max pooling, v6w25 anchor).
