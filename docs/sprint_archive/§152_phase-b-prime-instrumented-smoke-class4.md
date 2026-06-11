<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §152 — Phase B' instrumented smoke: Class-4 dominant, priority order v7→v8 — 2026-05-04

**Trigger:** Tracks A + C falsified single-knob and γ-knob hypotheses for
the Phase B draw plateau (92–94 % cap-rate at smoke step 10k). Open
question after T2 v7full baseline (3 % draws): which of four hypothesis
classes (1=stale dispatch, 2=value-head feedback, 3=buffer composition,
4=horizon-edge policy spam) drives the plateau.

**Run:** `w4c_smoke_v6_instrumented_5080` on RTX 5080 vast.ai. Aborted at
**step 2560 / 5000** (51 % complete) after the four-class signal
saturated — ~3 h walltime, 14.3 steps/min, 640 games, 86.7 % draw rate.
Bootstrap v7full unchanged; engine constants unchanged; production
config defaults unchanged.

### Verdict — MIXED, with Class 4 as the base

| Class | Strength | Dispositive |
|---|---|---|
| **4 — q-axis stride-5 (distance-5) spam** | **DOMINANT (base)** | ρ(stride5_run, is_ply_cap) = +0.50, p = 5e-42 |
| **3 — buffer composition** | STRONGLY ACTIVE (loop) | drawT = 0.979 from step ≤ 500 |
| **2 — value-head drift** | ACTIVE (downstream) | dec = −0.690 ± 0.031, locked below `draw_value=−0.5` |
| **1 — stale dispatch** | NOT TESTED | `eval_interval=5000` zeroed `model_version`; methodology gap |

**Causal story:** v7full mildly prefers stride-5 east-west extensions
(cap games at T2 baseline: row_max median 14, stride5_run median 3 vs
smoke 42 / 30 — **10× amplification**). Smoke conditions (γ knobs,
Dirichlet, playout_cap full=0.5 sims=600, training temperature schedule)
amplify the preference into the dominant policy mode. 87 % cap-rate →
98 % draw-coded buffer → value head trains to overshoot draw_value to
−0.69 ("side-to-move loses" prior) → reinforces cap-prone policy.
Class 4 is the root; Classes 2-3 are downstream loops.

### Class-4 detail (user-flagged)

Pattern: mixed-color stones along a single hex row at distance-5 spacing
— `x____o____o____x_` form (4 empty cells between consecutive stones,
hex_dist = 5). Both `LEGAL_MOVE_RADIUS = 5` (§146) and `CLUSTER_THRESHOLD
= 5` (§151 δ.c) are inclusive at exactly 5 — the policy fixed-points on
the boundary. The pattern persists DESPITE per-game uniform rotation
across the 12-element hex dihedral group (§130, `selfplay_rotation_enabled
= true`): macro `axis_distribution` reads q ≈ r ≈ s ≈ 0.33 (uniform across
rotations) but per-game stride-5 chains concentrate on whichever axis
the rotation puts on the parallelogram's long diagonal.

**Existing macro detectors miss it by construction**:
`colony_extension_fraction` (§107) gates at hex_dist > 6 — skips the
stride-5 boundary entirely. `axis_distribution` measures distance-1
adjacency, not distance-5. Both have stayed quiet through the entire
plateau.

### Phase B' priority order — v7 → v8

v7 (Tracks A + C synthesis) prescribed: draw_value −0.5 → −1.0; cap
150 → 300; pretrained_weight 0.8 → 0.4. **None of those address Class 4**
because Class 4 is in v7full's policy itself, not the training
dynamics.

v8 (post-instrumented-smoke):

1. **Policy- / window-side fix for Class 4 (must lead).** Candidates:
   asymmetric or per-turn-jittered `LEGAL_MOVE_RADIUS` ∈ {4, 5, 6};
   stride-5 anti-spam policy regulariser; `CLUSTER_THRESHOLD` re-test at
   6 or 7 (re-opens §147 v5 colony question — guarded smoke required);
   **hex-shaped window** (corner-mask cells at hex_dist > 9 in the
   19×19 parallelogram — restores C6 symmetry on input + augmentation).
2. **Class 3 buffer surgery.** Cap `draw_target_fraction` at 0.5 via
   subsampling on push; `draw_value −0.5 → −1.0` is secondary.
3. **`initial_pretrained_weight 0.8 → 0.4`** — Track C's flip-inflection
   mechanism. Secondary; does not address Class 4.
4. **`max_game_moves 150 → 300`** — symptom-only fix.
5. **Defer Gumbel re-enable** until items 1-3 ship.

**Explicitly excluded from Phase B':** γ-knob retuning (Track A
falsified); single-knob `draw_value` tweaks alone (won't fix Class 4);
sims-budget bumps (Track A A4_s mildly worsens).

### Required follow-ups before any sustained run

* **Class 1 closure:** same variant config but `eval_interval=500`,
  `iterations=2500` (~3 h). Required to rule Class 1 in or out before
  any sustained run.
* **Live Class-4 metric:** add `stride5_run_max_per_game` and
  `row_max_density_per_game` as dashboard signals (rolling last 50,
  P90 alarm at row_max > 30). Existing macro detectors are blind to
  stride-5; ship the new metric alongside the v8 fixes so they can be
  A/B'd.
* **Regenerate value-probe fixture** with `--cap-source smoke_jsonl`
  against `reports/phase_b_prime/instrumented/events.jsonl` so future
  probe runs measure on actual draw-equilibrium states (current fixture
  uses long-colony proxies because v7full produces only 6 ply_caps in
  200).

### Hex-window question (raised post-diagnosis)

19×19 axial parallelogram windows are anisotropic in hex distance:
corners reach hex_dist 18 along (q+r) diagonal vs hex_dist 9 along
the perpendicular. A regular hexagonal window (cells within hex_dist
≤ 9, 271 cells vs 361) would restore exact C6 symmetry on the input
plane structure and on the 12-fold dihedral augmentation. Possible
contributor to Class 4's per-game axis bias; not investigated in
this audit. Cheapest test: zero-mask the 90 corner cells where
hex_dist > 9 — 1-line change to `Board::encode_state_to_buffer_channels`.
Tracked under v8 item 1.

### Artifacts

* `reports/phase_b_prime/instrumented/diagnosis.md` — full verdict
* `reports/phase_b_prime/instrumented/events.jsonl` — 10 VP / 5 BC /
  5 MV / 5 WDR readings, 640 game records
* `reports/phase_b_prime/instrumented/run.log`
* `reports/phase_b_prime/instrumented/checkpoint_log.json`
* `scripts/phase_b_prime_diagnose.py`, `scripts/phase_b_prime_monitor.py`
* `configs/variants/w4c_smoke_v6_instrumented_5080.yaml`
* `fixtures/value_probe_50.npz`
* `hexo_rl/monitoring/value_probe.py`
* `hexo_rl/selfplay/pool.py` (`buffer_composition`,
  `model_version_summary`, `per_worker_draw_rates`)
* Engine instrumentation: `engine/src/inference_bridge.rs`,
  `engine/src/game_runner/{mod,worker_loop}.rs`,
  `engine/src/replay_buffer/{mod,storage}.rs`

Commits: `ea4b4cc` (engine instrumentation), `24fb0f5` (Python hooks),
`06f4663` (dashboard rendering), `a171d1f` (variant n_workers=8 fix),
`9767509` (diagnosis report).

---

