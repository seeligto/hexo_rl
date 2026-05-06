# Augmentation Audit — §119 Scoping

**Date:** 2026-04-24  
**Scope:** Replay buffer → batch assembly → dataloader. Self-play buffer, corpus buffer, recent buffer.

---

## Findings

### 1. Hex rotations

**Yes — all 6 rotations implemented.**

`engine/src/replay_buffer/sym_tables.rs:22` defines `N_SYMS = 12` covering the full
dihedral-6 group: identity + 5 non-trivial rotations (60°/120°/180°/240°/300°), each
paired with a reflected variant. `sample.rs:268` draws `sym_idx` uniformly from
`[0, 12)` per sample when `augment=true`:

```rust
let sym_idx = if augment { self.rng.random_range(0..N_SYMS) } else { 0 };
```

Each non-identity rotation occurs with probability **5/12 ≈ 42%** per Rust-buffer sample.

### 2. Axial reflections

**Yes — one reflection axis implemented.**

`sym_idx` 6–11 apply a reflection (swap of hex axes `(q,r) → (r,q)`) before the
rotation, covering the full dihedral group. Reflection probability per Rust-buffer
sample = **6/12 = 50%**.

### 3. Symmetry group coverage

For samples drawn from the Rust replay buffer (self-play uniform draw + corpus):
**12/12 — full dihedral-6 coverage in distribution.** Each sample draws one uniform
random element; in expectation across a batch all 12 symmetries are equally represented.

---

## The gap: `RecentBuffer` is never augmented

`hexo_rl/training/recency_buffer.py` is a Python ring buffer. Its `.sample()` method
returns raw tensors with no transformation. `batch_assembly.py:264–266` routes
`recency_weight` of each self-play slice through `recent_buffer.sample()` with no
`augment` parameter.

Config defaults: `recency_weight: 0.75`, `initial_pretrained_weight: 0.8` decaying to
`min_pretrained_weight: 0.1` over `decay_steps`.

**Effective un-augmented fraction per 256-row batch:**

| Training phase | n_corpus | n_self | n_recent (no aug) | Unaugmented fraction |
|---|---|---|---|---|
| Early (pretrain_weight=0.8) | ~205 | ~51 | ~38 | 38/256 ≈ **15%** |
| Late (pretrain_weight=0.1) | ~26 | ~230 | ~172 | 172/256 ≈ **67%** |

At steady-state self-play the majority of the batch is identity-only. The model sees
the correct symmetry distribution only for corpus rows and the 25% uniform self-play
slice; the 75% recent slice is always unrotated/unreflected.

**Effective symmetries per sample (late training):**
- 33% of batch: uniform over 12 → expected symmetry diversity = full
- 67% of batch: identity only → 1/12 of group
- Batch-weighted effective coverage ≈ **0.33 × 1.0 + 0.67 × (1/12) ≈ 0.39**,
  i.e., the average sample "sees" the equivalent of ~4.7/12 group elements.

---

## Minimal implementation spec (if §119 decides to close the gap)

Goal: extend augmentation to `RecentBuffer` rows. Implementation is surgical — no
changes to the Rust buffer or symmetry table definitions needed.

### Option A — Augment at `recent_buffer.sample()` call site (Python, simplest)

1. **Policy scatter LUTs** (Python): 12 precomputed index arrays of length 362 already
   exist in `hexo_rl/bootstrap/pretrain.py:73–116` (`_get_policy_scatters`). Extract
   to a shared module (e.g., `hexo_rl/augment/luts.py`).

2. **State permutation**: call `engine.apply_symmetries_batch(states_f32, sym_indices)`
   (Rust kernel, already exposed via PyO3 for the pretrain collate). Draw `sym_indices`
   uniformly from `[0, 12)` for the `n_recent` rows.

3. **Chain planes**: recompute from augmented stone planes (planes 0 and 8) using
   existing Python logic in `pretrain.py:179–195`.

4. **Ownership / winning_line**: scatter via same LUT as policy (361-cell bijection +
   pass cell identity).

5. **Call site**: `batch_assembly.py:266` — after `s_r, … = recent_buffer.sample(n_r)`,
   apply augmentation before concatenation.

### Option B — Move `RecentBuffer` into Rust

Reimplement as a Rust `RecentBuffer` struct alongside the existing `ReplayBuffer`, expose
via PyO3 with the same `sample_batch(n, augment)` signature. Augmentation is then free
(same scatter path as main buffer). Higher upfront cost, cleaner long-term.

### Cost estimate

| Option | Effort | Risk |
|---|---|---|
| A (Python augment at call site) | **2–3 hrs** | Low — reuses existing Rust kernel and LUTs |
| B (Rust RecentBuffer) | **6–10 hrs** | Medium — new Rust struct + PyO3 bindings + migration |

Option A is the 2026 near-term path. Extract LUTs → apply kernel → recompute chain
planes → wire into `batch_assembly.py`. No new tests beyond confirming loss-convergence
tests still pass with `augment=False`.

---

## Verdict for §119

Full 12-fold augmentation **is implemented** for corpus and uniform self-play rows.
The gap is `RecentBuffer`: 0% augmentation, rising to ~67% of the batch at late
self-play. This biases the network toward unrotated positions during the highest-
recency phase of training. Whether this meaningfully hurts generalisation is an open
question — it may be masked by the corpus rows in early training and self-corrected by
diverse game trajectories later. Option A closes the gap at ~2–3 hours; recommend
including it in Phase 4.5 if a symmetry-generalisation failure mode surfaces.
