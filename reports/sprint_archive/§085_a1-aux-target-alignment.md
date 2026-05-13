<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §85 — A1 aux target alignment (Python side) (2026-04-13)

Companion landing to Rust commit `faafc43` (`feat(replay_buffer): per-row aux target alignment (A1 Rust side)`). Rips out the legacy ring-buffer aux path and threads per-row ownership + winning_line u8 columns end-to-end.

### A1 root cause — three compounding subproblems

1. **Indexing.** `pool.get_aux_targets()` pulled aux from a 200-entry game-level ring with independent random indices, fully decoupled from `buffer.sample_batch` batch indices. Aux targets had no relation to the states they were paired with.
2. **Cardinality.** One aux map per *game* broadcast across ~60 replay rows. Even index-aligned, one-per-game cannot match per-row state windows.
3. **Frame.** Aux maps were projected to the **game-end bbox centroid** while each replay row's state was projected to that row's **own cluster window centre at recording time**. Offsets up to ±9 cells in any multi-cluster game.

### Fix shape (Option A from `/tmp/A1_aux_alignment_spike.md`)

- **Rust side (commit faafc43):** `ReplayBuffer` gained per-row `ownership` + `winning_line` u8 columns; `game_runner.rs` reprojects them at game end using each row's own `(cq, cr)` cluster centre via `Board::window_flat_idx_at`. `apply_sym` extended in `replay_buffer/sampling.rs` so the 12-fold hex symmetry table applies to both new planes consistently with state + policy. `sample_batch` now returns a 5-tuple; `push` and `push_game` grew two positional args; `collect_data` now returns a 6-tuple.
- **Python side (this commit):**
  - `pool.get_aux_targets`, `_ownership_ring`, `_threat_ring` deleted from `hexo_rl/selfplay/pool.py`.
  - `_stats_loop` unpacks the new `collect_data` 6-tuple and threads per-row u8 aux into both `replay_buffer.push` and `recent_buffer.push`.
  - `RecentBuffer` (`hexo_rl/training/recency_buffer.py`) gained `_ownership` / `_winning_line` u8 columns; `push` and `sample` now carry aux. Existing 3-arg push callers fall back to ones/zeros defaults.
  - `Trainer._train_on_batch` decodes ownership u8 `{0=P2, 1=empty, 2=P1}` → float `{-1, 0, +1}` via `astype(f32) - 1.0`; winning_line u8 → f32 directly. Accepts `n_pretrain` row count and slices `[n_pretrain:]` before computing ownership MSE + threat BCE so pretrain corpus rows do not poison the aux heads.
  - `Trainer.train_step` no longer accepts the legacy `ownership_targets` / `threat_targets` kwargs; aux now flows from the buffers themselves.
  - `scripts/train.py` corpus loader pads `bootstrap_corpus.npz` with `ownership=ones` (decoded 0.0, neutral MSE) and `winning_line=zeros`; the `n_pretrain` row slice masks them out of aux loss.
  - Pre-allocated batch buffers extended with `_own_buf` + `_wl_buf` (uint8, `(batch, 19, 19)`).

### Empirical baseline (ckpt_19500, 20-position threat-logit probe, `/tmp/colony_spam_diagnosis.md` §2)

| metric                                 | bootstrap_model.pt | checkpoint_00019500.pt |
|----------------------------------------|--------------------|------------------------|
| threat logit @ extension cell          | −0.14 ± 0.74       | **−3.25 ± 0.46**       |
| threat logit @ random empty cell       | −0.52 ± 0.39       | **−5.11 ± 1.40**       |
| contrast (extension − random empty)    | **+0.38**          | **+1.86**              |

ckpt_19500 had a *higher* contrast than bootstrap — the symptom of the head learning a marginal-class shortcut against a stale, mis-aligned label rather than the true spatial signal.

### Kill criterion for next sustained run (REVISED §91 2026-04-14)

**Original §85 criterion was over-indexed on ckpt_19500's specific collapse
signature.** ckpt_00014344 (the next sustained run) hit a different failure
mode: contrast_mean grew TO **10× bootstrap (+3.94)** while absolute logits
drifted globally negative (ext_logit_mean = −6.2). That is the OPPOSITE of
ckpt_19500, where contrast grew only 5× and both logits collapsed by the
same amount. Old C1 (`ext_logit_mean >= baseline − 1.0`) FAILed; old C2 and
C3 PASSed. The pattern is consistent with BCE-on-imbalanced-labels driving
logits globally negative while position-conditional sharpness IMPROVES —
i.e. the policy head is doing exactly what we wanted (not colony-spamming),
just with a global bias shift in the threat head.

The original C1 was therefore not a colony-spam detector — it was a BCE
scale-drift detector, and gating on it would have incorrectly killed a
healthy run. C1 is replaced; the colony-spam intent is preserved by adding
a top-10 condition. The full revision is in §91. The current criterion is:

| # | condition | threshold |
|---|-----------|-----------|
| 1 | contrast_mean (ext − ctrl) | ≥ max(0.38, 0.8 × bootstrap_contrast) |
| 2 | extension cell in policy top-5 | ≥ 40% |
| 3 | extension cell in policy top-10 | ≥ 60% |
| 4 (warning) | abs(ext_logit_mean − bootstrap_ext_mean) < 5.0 | warning only — never gates |

`make probe.latest` enforces C1-C3; C4 prints a WARNING line in the report
but does not flip the exit code. Bootstrap baseline numbers come from
`fixtures/threat_probe_baseline.json` (schema v2).

If not met, the aux fix did not materially land; investigate before continuing.
If met, the colony-spam loop is a separate failure mode and the threat head is
free to do its job.

### Corpus aux shortcut

Chose **option (b)** from the spike doc: pad corpus rows with neutral aux (ones/zeros) at load time, mask via `n_pretrain` row slice. The alternative (extending `bootstrap_corpus.npz` with ownership/winning_line columns + reworking `scripts/export_corpus_npz.py`) is parked as a separate corpus refactor — orthogonal to the colony-spam fix and not blocking the next sustained run.

### Telemetry

`Trainer._train_on_batch` now emits `aux_loss_rows = batch_n - n_pretrain` in the `train_step` result dict whenever aux losses run. Stuck `n_pretrain == batch_n` (no rows contributing to aux) becomes visible in dashboards.

### Dead code left behind

Rust `drain_game_results()` still emits the legacy float32 `ownership_flat` / `winning_line_flat` (game-end frame) tuple fields. Pool unpacks and discards them via underscore variables. **TODO:** strip from the Rust drain path in a follow-up patch — pure dead stripe, zero runtime cost, no urgency.

### Memory delta

ReplayBuffer adds `2 × 361 = 722 B/row` for the new u8 columns. At capacity=1M: **+722 MB**. RecentBuffer (capacity ≈ 500K) adds another **+360 MB**. Headroom on the 48 GB box is fine but should be re-checked before any future capacity bump.

### Files touched

- `engine/src/replay_buffer/mod.rs`, `engine/src/replay_buffer/sampling.rs`, `engine/src/replay_buffer/sym_tables.rs`, `engine/src/game_runner.rs` (Rust, prior commit faafc43)
- `hexo_rl/selfplay/pool.py`
- `hexo_rl/training/trainer.py`
- `hexo_rl/training/recency_buffer.py`
- `scripts/train.py`
- `tests/test_rust_replay_buffer.py`, `tests/test_trainer.py`, `tests/test_phase4_smoke.py`, `tests/test_dashboard_events.py`, `tests/test_weight_schedule_wiring.py`, `tests/test_buffer_shutdown.py`, `tests/test_worker_pool.py`
- `tests/test_aux_target_alignment.py` (new)

**Commit:** `feat(training): align aux targets with state batch (A1 Python side)`

---

