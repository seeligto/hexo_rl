<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §171 P0 — sprint scoping + entry-point validation — 2026-05-09

**Branch:** `phase4/p171_selfplay_smoke` (off master @ `ebacabd`,
post-§170 close-out merge).
**Status:** P0 closeout note. P1 (variant-config commit `888064a`) +
P2 (cold-smoke pre-flight) + P3 (sustained smoke launch) tracked in
separate sprint-log entries / memory notes.

### §171 P0.1 — corpus-sha manifest transposition correction

While preparing the §171 A4 distribution-shift fine-tune side-arm
(Gate 6 item c) the §170 P4 P2 manifest entry-point was audited
against the on-disk corpus files. **Transposition discovered:**

- §170 P4 P2 manifest + sprint-log entry cited
  `data/bootstrap_corpus_v8.npz sha256 110ea6b2…` as the §171 A4
  fine-tune base corpus.
- Actual on-disk file at `data/bootstrap_corpus_v8.npz` is the
  **vanilla v8 baseline corpus** with sha
  `adb884122bc4744b771ca95f30419ba067764eacbae428fa27c5117db8a0dd77`
  (plane-8 polarity off→outside; produced by §167 / §169
  `export_corpus_npz.py` *without* `--canvas-realness`). Used by §167
  B1 / §169 A2 / A3 retrains.
- The sha `110ea6b2…` belongs to the **canvas_realness variant**
  produced as `data/bootstrap_corpus_v8_canvas_realness.npz` per
  `reports/ablation_169/A4_corpus_export.log` (5,382 MB, 347,142
  positions, plane-8 polarity inside→canvas, used to pretrain
  `checkpoints/ablation_169/A4_canvas_realness.pt`).

**Why it matters.** `A4_canvas_realness.pt` carries a `PartialConv2d`
trunk-entry whose plane-8 polarity invariant is canvas_realness=True.
Resuming fine-tune from that checkpoint with the vanilla v8 corpus
(adb88412…) flips plane-8 polarity at the trunk entry — silently
breaking the renormalisation invariant the PartialConv2d module was
trained under. The corrected entry-point uses
`bootstrap_corpus_v8_canvas_realness.npz` (sha `110ea6b2…`) so plane-
8 polarity matches the resume checkpoint.

**Resolution.** This sprint corrects every §170 P4 P2 + §171 A4
entry-point reference across `reports/gpool_bias/adversarial_manifest.md`
and `docs/07_PHASE4_SPRINT_LOG.md` (§170 P4 P2 entry, §170 P4 close-
out aggregation, Gate 6 item-c scope note). References to
`bootstrap_corpus_v8.npz` as the **vanilla v8 baseline** (e.g. §169
A4's "(v8 corpus, n/a)" matrix cell, §167 B1 retrain entries) are
unchanged — those correctly cite the sha `adb88412…` file at that
path.

History is annotated, not rewritten: §169 P4 / §170 P3 / §170 P4 P1
narratives stand. The §170 P4 P2 corpus-prep work itself is correct
(adversarial_corpus_v8.npz schema-parity is canvas_realness; the
manifest just labelled the partner file with the wrong path). No data
file changes; no code changes.

**Status:** entry-point manifest corrected on
`phase4/p171_selfplay_smoke`. §171 A4 fine-tune (if Gate 6 item c
opens) consumes
`data/bootstrap_corpus_v8_canvas_realness.npz` + `data/adversarial_corpus_v8.npz`
+ `checkpoints/ablation_169/A4_canvas_realness.pt`.

---


## §171 P3 — BLOCKED (handed to §172 architectural sprint) — 2026-05-09

### TL;DR

Two-layer blocker halts v6w25 sustained self-play. Layer 1: canvas vs trunk
`board_size` semantics inconsistency between A2.1 (trainer, trunk=25) and A2.2
(pool guard, canvas=19). Layer 2 (load-bearing): `Board::to_planes()` is
hardcoded to 18×19×19 regardless of `cluster_window_size`; selfplay's
`state.to_tensor()` calls this single-window path. v6w25 pretrain worked because
`dataset_v6w25.py` uses `get_cluster_views()` (honors `cluster_window_size=25`);
selfplay has no equivalent multi-window path. §172 architectural sprint required
to resolve the structural debt.

Pin scope clarification: v6w25 is canonical for pretrain+eval+matched-MCTS per
§170 P4 P1 mechanism. v7full is canonical for selfplay pending α. Neither pin is
silently retracted; both are explicit with scope constraints documented here.

α scope statement: α (multi-window K-cluster selfplay) is Phase 4.5+ scope;
design doc lands in §172, implementation in §173+.

---

### §171 P3.1 — What A1+A2 delivered (correct-in-scope)

Commits `9e4876f..2121069` correctly implemented encoding-awareness at the Python
layer:

- ✓ Trainer encoding-aware checkpoint reconciliation
- ✓ Workers construct `Board::with_encoding(spec)` per game (`cluster_window_size`
  correctly set to 25 for v6w25)
- ✗ But planes emitted are still 19×19 — `to_planes()` ignores `cluster_window_size`

This work is necessary but not sufficient. Not wasted — §172 builds on it.

---

### §171 P3.2 — Surface enumeration (blocker grep findings)

| Site | Finding |
|---|---|
| `engine/src/board/state.rs:41` | `TOTAL_CELLS = 19*19 = 361` — hardcoded |
| `engine/src/board/state.rs:641` | `Board::to_planes()` emits 18 × TOTAL_CELLS — hardcoded 19×19 |
| `engine/src/lib.rs:217-219` | PyO3 `Board.to_tensor()` calls `inner.to_planes()` → 18×19×19 regardless of `cluster_window_size` |
| `engine/src/lib.rs:225` | `view_window` docstring: *"`size` is ignored (always 19×19)"* |
| `engine/src/board/state.rs:703` | `get_cluster_views()` honors `cluster_window_size` — pretrain path, not selfplay |
| `hexo_rl/selfplay/inference.py:53` | calls `state.to_tensor()` — single-window 19×19 path |
| `bootstrap_model_v6w25.pt` | `policy_fc.weight = (626, 1250)`: 626 = 25×25+1 (action space), 1250 = 2ch × 625 (trunk 25×25) |

Inference batcher expects (B, 8, 25, 25) from the model. selfplay delivers
(B, 8, 19, 19). Either crash or silent shape corruption.

---

### §171 P3.3 — Layer 1 detail (canvas vs trunk semantics)

| Site | Convention |
|---|---|
| `_V6W25_SPEC.board_size` (encoding.py:108) | 19 (canvas) |
| `Trainer._model_board_size_for(spec)` (trainer.py:1051) | 25 (trunk) |
| `Trainer._propagate_encoding_into_config` writes to `config["board_size"]` | 25 (trunk) |
| `WorkerPool` guard `model.board_size != spec.board_size` (pool.py:127) | compares against 19 (canvas) |
| Established v6w25 ablation configs | `board_size: 25` (trunk) |

A2.1 picked trunk semantics, A2.2 picked canvas semantics. No test loaded a real
v6w25 checkpoint into a real `WorkerPool` with a real `HexTacToeNet`.

---

### §171 P3.4 — Options and recommendation

- **(α) Multi-window K-cluster selfplay** — replicate pretrain path in selfplay.
  Workers emit K cluster views via new Rust API. Inference batcher handles
  K-per-position fan-out. Pool / results queue / replay buffer all need K-aware
  batching. Large effort. Touches Rust + Python + Buffer.
- **(β) Single-window 25×25 selfplay** — extend `to_planes()` to honor
  `cluster_window_size`. Smaller change but changes v6w25 semantics: trains on
  different distribution than pretrain. May invalidate the v6w25 anchor.
- **(γ) Re-anchor §171 P3 to v7full** — v7full (`board_size=19`, v6 encoding)
  works on existing 19×19 selfplay path. Rolls back §170 close-out canonical pin.
- **(δ) Pivot to A4 fine-tune side-arm** — pretrain only; doesn't exercise the
  broken selfplay plane projection.

**Recommendation:** (α) is the right structural fix but multi-day effort. (γ) and
(δ) are cheap tactical pivots. §172 architectural sprint implements (α) correctly,
using §171 A1+A2 Python plumbing as the foundation.

---

### §171 P3.5 — State preserved

- **5080:** workspace pulled to `2121069`, engine rebuilt clean, pre-flight tmux
  killed, 0 GPU memory in use.
- **Laptop:** branch `phase4/p171_selfplay_smoke` HEAD = `2121069`, pushed.
- **Pre-flight log:** `5080:$REPO_ROOT/logs/sprint_171_p3_preflight/run.log`.
- **Pre-flight ckpt-dir:** empty (crashed before first ckpt).
- **Cross-reference:** raw report at `/tmp/p171_p3_preflight_blocker.md`.

---

### §171 P3.6 — Pin scope clarification

- **v6w25 canonical for:** pretrain, eval, matched-MCTS (per §170 P4 P1
  mechanism). Scope: single-window 25×25 inference from a pre-trained checkpoint.
- **v7full canonical for:** selfplay (sustained smoke), pending α delivery.
- Neither pin is silently retracted. Both are explicit with scope constraints.
- **α (multi-window K-cluster selfplay)** is Phase 4.5+ scope. Design doc in §172
  Phase A7. Implementation in §173+.
- The §172 sprint header will document α scope formally; this entry records the
  blocker that necessitated it.

**Status:** BLOCKED → §172 architectural sprint. `phase4/encoding_registry`
branch cut. §171 P3 will not resume on `phase4/p171_selfplay_smoke`.

---


## §171 A4 P2-reopen C — distribution-shift fine-tune side-arm — 2026-05-11

**Branch:** `phase4/encoding_registry`
**Verdict:** **DEAD** (E2 confirmed cleanly).
**Investigation report:** `reports/investigations/sprint_171_a4_finetune_p2reopen_2026-05-11.md` (gitignored — local-only per project convention; pre-registered hypotheses + thresholds locked before launch).

### Pre-registered scope (locked 2026-05-11 before launch)

- **Hypothesis E1 (ALIVE):** the §169 P0 SPATIAL_RICH reframing is correct; the matched-MCTS collapse is a head-calibration problem that a small adversarial-mix fine-tune can fix. Threshold: MCTS-64 WR > 8% AND Wilson-95 lower > 5%.
- **Hypothesis E2 (DEAD):** spatial features ARE the bottleneck; head + top-4-block fine-tune cannot recover MCTS signal. Threshold: MCTS-64 WR ≤ 2% AND Wilson-95 upper < 4%.
- **MARGINAL bin:** WR in [2%, 8%]; would require deeper unfreeze before condemning A4.

### Recipe

| Knob | Value |
|---|---|
| Resume ckpt | `checkpoints/ablation_169/A4_canvas_realness.pt` (inference state-dict; new weights-only resume path) |
| Mixed corpus | 95% bootstrap_v8_canvas_realness (347,142) + 5% adversarial_v8 (12,781) via `WeightedRandomSampler` rescale (no physical replication; `scale_adv = 0.022712`); n=359,923 |
| Steps | 3000 (--epochs 3, --steps caps before epoch 3 ends) |
| Batch | 256 |
| LR | peak 5e-5, eta_min 5e-6 (cosine restart), AdamW fresh state |
| Freeze | trunk.input_conv (PartialConv2d) + trunk.input_gn + trunk.tower[0..7]; **trainable: trunk.tower[8,9,10,11] + policy/opp_reply/value heads** = 1,354,316 / 3,846,668 (35.2%) |
| compile | OFF (consistency with local 30-step smoke) |

Implementation: 3 commits on `phase4/encoding_registry`:
- `ee8032a` — `scripts/build_mixed_corpus_a4.py`, `--freeze-trunk-entry`/`--unfreeze-blocks` CLI, weights-only `--resume` mode, `tests/test_pretrain_finetune_freeze.py` (4 unit tests).
- `47c2f29` — `scripts/eval_sprint_171_a4.sh` with the verdict-bin classifier.
- `000f6ac` — fix(eval): `spec.version → spec.name` (unrelated §172 A10 stale attr surfaced at eval boot).

### Smoke evidence (laptop, 30 steps)

- Trainable: 1,354,316 / 3,846,668 (35.2%, matches §169 P4 model size).
- Loss: 3.84 → 3.47 over 30 steps (decreasing, finite).
- Frozen-surface invariant: `trunk.input_conv` + `tower[0..7]` max|Δ|=0 pre/post — bit-identical.
- Trainable-surface deltas: `tower[8/9/11]`, `policy_head`, `value_fc1` show ~7e-4 max|Δ|.

### Vast 5080 results

| Metric | Value | §169 P4 baseline | Δ |
|---|---|---|---|
| Fine-tune wall | 5 min 20 s | — | — |
| Final loss | 3.50 (epoch 3 early-cut at step 3000) | 3.47 baseline (post-pretrain) | flat |
| argmax @ r=8 n=200 | **0/200 = 0.0%** [0.000%, 1.88%], mean ply 23.3 | 0/200 = 0.0% [0.0%, 1.88%], mean ply 23.5 (`reports/ablation_169/A4_eval.json`, 2026-05-08) | no change |
| MCTS-64 @ r=8 n=200 | **0/200 = 0.0%** [0.000%, 1.88%] | MCTS-128 ~0% pre-fine-tune | consistent |
| Eval wall | argmax 397 s + MCTS-64 674 s = 17.9 min | — | — |

**Correction (2026-05-11, post-bootstrap-argmax-checkup):** the original close-out of this entry compared the fine-tune argmax (0%) against a 22% baseline and claimed the freeze pattern "collapsed argmax sharpness". That 22% was misattributed — it belongs to **§170 P3 A1+gpool-bias** (v6w25 K-cluster + gpool-bias side-branch; sprint log L9979/L10154/L10259/L10331), not A4 canvas_realness. The actual §169 P4 A4 argmax baseline was already 0/200 (per `reports/ablation_169/A4_eval.json` and sprint log L9535 — "A4 argmax > 12% (bbox direction lives): NOT TRIGGERED (0%)"). Confirmed by re-running A4 argmax n=20 at pre-§172 commit `cedaec3`: 0/20, mean ply 23.2 — identical to HEAD. **A4 was never above 0% argmax at this radius — the fine-tune did not damage anything new, and the DEAD verdict stands strictly on the MCTS-64 axis.**

### Verdict — DEAD

```
MCTS-64 WR = 0.0000   Wilson95 = [0.0000, 0.0188]   n=200
DEAD threshold:  WR <= 0.02 AND Wilson upper < 0.04
                 0.0  ≤ 0.02 ✓    0.0188 < 0.04 ✓
```

**WITHDRAWN (2026-05-11 amend):** an earlier version of this paragraph claimed the fine-tune "collapsed argmax sharpness 22% → 0% (mean ply 48 → 23)". That comparison was based on a misattributed baseline — the 22% number belongs to §170 P3 A1+gpool-bias, NOT A4 canvas_realness. The §169 P4 A4 baseline was already 0.0% argmax at this radius (`reports/ablation_169/A4_eval.json`, sprint log L9535). Confirmed by re-running argmax n=20 at pre-§172 commit `cedaec3`: 0/20, mean ply 23.2 — identical to HEAD. **The fine-tune did not damage anything new; argmax was already at floor pre-fine-tune. The DEAD verdict rests strictly on the MCTS-64 axis.**

### Implication

The §169 P0 SPATIAL_RICH framing — "matched-MCTS collapse is a distribution-shift problem, not an architecture problem" — is **FALSIFIED** by this side-arm. Distribution-shift fine-tune over a 5% adversarial corpus, with the trunk-entry + lower trunk frozen, cannot re-tune the policy/value heads onto an MCTS-recoverable manifold; the limitation is structural to the v8 + canvas_realness + frozen-spine configuration.

Closes the §171 A4 fine-tune line as a candidate Phase 4.0 unblocker. v6w25 K-cluster (§169 A1 anchor) and v7full (§172 B2 sustained — closed without graduation per `project_172_b2_complete`) remain the two tested anchors.

### Next

**Recommendation:** proceed to §173 α multi-window K-cluster selfplay (the structural fix identified in §171 P3.4 option α; design doc at `docs/designs/encoding_alpha_multiwindow_selfplay_design.md`).

Cheaper alternative (low-prior given the argmax collapse): re-run A4 fine-tune with a different freeze recipe (fully unfrozen, or peak ≤ 1e-5) and a value-head probe before any full re-train. NOT recommended without prior evidence.

### Artefacts

- `checkpoints/sprint_171_a4/A4_finetune_p2reopen.pt` (vast)
- `reports/sprint_171_a4/A4_finetune_argmax.json`
- `reports/sprint_171_a4/A4_finetune_mcts64.json`
- `reports/sprint_171_a4/A4_finetune_eval.json` (combined + verdict)
- `logs/sprint_171_a4/finetune.log`, `logs/sprint_171_a4/eval.log`
- Investigation report: `reports/investigations/sprint_171_a4_finetune_p2reopen_2026-05-11.md`

---

