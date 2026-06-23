# §D-LOOPFIX Arm-C Re-run — In-Run Watch Sheet (DESIGN-ONLY)

> Authored by D-RERUNPREP Phase-1 B6; stride→round firing **corrected** against the
> live gate (`eval_pipeline.py:310-311`, `promotion_capable_rounds`
> `step_coordinator.py:185`) after the workflow's B6 draft mis-stated it. Cost figures
> are **estimates** (throughput assumption stated below, not a verified golong number).

**Run config:** `configs/variants/v6_live2_ls_ab.yaml` — encoding `v6_live2_ls`, `eval_interval 12500`, `--iterations 50000`
**Incumbent pinned:** `bootstrap_model_v6_live2.pt` SHA `4198d5cbd2fc0ce77ad2e3d86e32311ee40c53c926f135c52ea3749816b0a186`
**Corpus:** `data/bootstrap_corpus_v6_live2.npz` (1.70 GB — verified present on laptop + vast)
**Hardware:** vast 5080 + Ryzen 9 9900X

---

## 1 · Eval cadence (VERIFIED against the gate)

`round_idx = step // eval_interval`; an opponent fires iff `round_idx % stride == 0`
(or always at the terminal close-out, `ignore_stride=True`). With `eval_interval=12500`,
a 50k run has **4 in-run rounds** + 1 terminal:

| round_idx | step | best (s1) | sealbot/random/bootstrap_anchor (s1) | nnue (s2) | offwindow (s2) |
|--:|--:|:--:|:--:|:--:|:--:|
| 1 | 12,500 | ✅ promote | ✅ | — `1%2≠0` | — `1%2≠0` |
| 2 | 25,000 | ✅ promote | ✅ | ✅ **Obj-A read** | ✅ **Obj-A read** |
| 3 | 37,500 | ✅ promote | ✅ | — `3%2≠0` | — `3%2≠0` |
| 4 | 50,000 | ✅ promote | ✅ | ✅ **Obj-A read** | ✅ **Obj-A read** |
| **terminal** | 50,000 (final ckpt, UNLOADED) | ✅ | ✅ | ✅ (stride ignored) | ✅ (stride ignored) |

- **Promotion decisions:** 4 in-run (best stride 1 → `promotion_capable_in_run_rounds=4`) + 1 terminal = **5 decision gates** (vs the broken prior run's 1).
- **Objective-A in-run reads:** nnue + offwindow each fire at **rounds 2 and 4** = 2 in-run reads each + 1 terminal = **3 reads/opponent**. (W1's original sin was **zero** in-run reads — that is closed.)
- These in-run offwindow reads are **WATCH-ONLY**. The decisive Objective-A read is the standalone `exploit_probe.py` + counterfactual via `KClusterMCTSBot` (single-window `ModelPlayer` false-clears — §9.10/P4).

---

## 2 · Watch keys with thresholds

| Metric | Threshold / target | Justification | Watch signal |
|---|---|---|---|
| **value_accuracy_masked** (headline) | rising; golong ref | In-window move-prediction quality; load-bearing for conversion. | Monotone rise post-promotion; sudden drop ⇒ value-head drift / buffer contamination. |
| **draw_rate** (HARD-ABORT) | **≤ 0.55, 3-consec, min_step 0** | Config `hard_abort_draw_rate=0.55 / consec 3 / min_step 0`. L9: cosine-temp + jitter pairing is the draw-collapse lever (§155/§157). | Fire abort on **3 consecutive** evals ≥0.55. If it fires, run is ABORTED and **close-out does NOT run** (no terminal eval). Log firing step + 3 violating evals. |
| **components_count** | stable ~20–35 (golong ~27) | Spatial complexity. | Collapse <15 ⇒ colony-attractor sign; >50 ⇒ pathological spread. Investigate, not auto-abort. |
| **longest_line (axis_run P90)** | ref golong ~9.3 ply | Max contiguous same-color axis run. | P90 >15 **and** components <20 ⇒ likely colony capture. |
| **forced_win_conversion** | ≥ 0.75 (golong 0.75 overall / 0.83 in-window) | % of perceivable wins converted. Off-window misses are structural; in-window misses = O1 signal. | Decline >5pp ⇒ check whether off-window % rose (encoding side-effect) vs in-window misses rose (value drift). |
| **off_window_rate** | ~52% structural floor (golong 51.9%) | Multi-cluster's PRIMARY lever — off-window cells have no policy logit. | Rises >55% + low components ⇒ encoding did **not** help. Falls <50% ⇒ encoding working (window recentering / multi-cluster coverage). |

---

## 3 · Close-out lifecycle (W1) — the load-bearing sequence

At step 50,000 (`stop_step`):
1. **STOP** training loop (training-stop ≠ process-exit).
2. **DRAIN** in-flight eval games — budget = observed eval-round wall-clock × 3.0, **floored 900 s**, **hard-capped 14400 s**. WARN-never-kill.
3. **STOP pool** (halt self-play workers).
4. **TERMINAL_EVAL** on the final checkpoint — **all** opponents, stride ignored, **GPU-UNLOADED** (pool already stopped).
5. **PROMOTE** via shared `promote_anchor` (terminal result feeds the gate; `sync_inference=False`).
6. **CLOSE** (DB flush, checkpoints archived, `.provenance.json` stamp).
7. **EXIT 0**.

> **THE GATE TOKEN:** `terminal_eval_complete` MUST appear in the log **exactly once** at close-out (`step_coordinator.py:1548`, event carries `terminal=True`, `step=50000`). If absent ⇒ terminal eval was skipped or killed ⇒ the run did NOT close cleanly.
> **Terminal result is a RECORD, not a steer input** — it is emitted as the distinct `terminal_eval_complete` event (never `eval_complete`), runs outside `step()`, and on resume `resolve_anchor` loads only the `best_model.pt` resilient chain (a promoted *model*, never the terminal *record*). Covered by `test_terminal_eval_runs_full_battery_ignoring_stride` + `test_terminal_eval_skipped_on_sigint`.

---

## 4 · Vital event checklist (operator watches in order)

- [ ] **Launch:** `anchor_identity sha256=4198d5cb… pinned=4198d5cb…` — no mismatch `RuntimeError`. (Pin is the **vast/de-facto** bootstrap `4198d5cb`, re-pinned from the laptop-derived `aba28e10` after the §D-RERUNPREP host-drift finding.)
- [ ] **Launch:** `eval_schedule_capability promotion_capable_in_run_rounds=4`.
- [ ] **Round 1 (12.5k):** draw_rate logged; best n=400 fires.
- [ ] **Round 2 (25k):** bootstrap_anchor PRESENT (gate not blocked on missing-measurement); nnue + offwindow fire (Obj-A in-run).
- [ ] **Round 3 (37.5k):** value_accuracy_masked trend rising/stable.
- [ ] **Round 4 (50k):** best fires; then close-out.
- [ ] **Close-out:** `run_evaluation(ignore_stride=True)` log line; **`terminal_eval_complete` fires exactly once**.
- [ ] **Exit:** process exit code 0, no hung evaluator.
- [ ] **Checkpoint:** final promoted checkpoint archived with `.provenance.json` sidecar (step + run_id + encoding).

---

## 5 · Checkpoint banking plan

| Checkpoint | Trigger | Bank? | Where | Rationale |
|---|---|---|---|---|
| step 12.5k / 25k / 37.5k | post-round eval | bank **if promoted** that round | `checkpoints/armc_<step>_promoted.pt` | preserve any promoted incumbent for exploit_probe |
| **step 50k (final)** | post-terminal | **ALWAYS** | `checkpoints/armc_50000_final.pt` + `.provenance.json` | the Arm-C delivered checkpoint for exploit_probe + strength-vs-golong50k |
| `best_model.pt` | each promotion | final version only | local `checkpoints/` | preflight clears it to fresh-init the pinned bootstrap (W2); interim versions ephemeral |

**Off-host:** pull final checkpoint (~50 MB) + `.provenance.json` to laptop after close-out for exploit_probe / KClusterMCTSBot analysis. Replay buffer not needed post-training.

---

## 6 · Cost estimate (ESTIMATE — verify against golong wall-clock before quoting)

| Component | Calculation | Estimate |
|---|---|---|
| Throughput **assumption** | 490 steps/hr (B6 figure; **not corroborated in the sprint log** — confirm vs the golong 50k wall-clock) | — |
| Training wall-clock | 50,000 ÷ 490 | ~102 h ≈ **~4.25 days** |
| Vast 5080 rate | $0.52–0.80/hr (ssh6.vast.ai often ~$0.56) | $0.60/hr mid |
| Labour (training) | 102 h × $0.60 | **~$61** |
| With ~10% co-tenancy/idle margin | | **~$67** (range **$53–90**) |

> ⚠ The cost hinges entirely on the unverified 490 steps/hr. **Action before launch:** read the golong run's start→step-50000 wall-clock from its logs and re-derive. If real throughput differs, scale linearly.

---

## 7 · Red-team notes

1. **W2 self-deadlock:** do NOT promote-then-resume this variant — a promotion advances `best_model.pt` past the launch pin ⇒ intentional `RuntimeError` on resume (A-INCUMBENT). The pin is a single-run guard.
2. **Co-tenancy (A-COTENANCY):** GPU float non-determinism under co-tenants shifts small-n eval WR ±0.16. W1 mitigates: terminal eval UNLOADED + n=400 makes in-run small-n non-decisive. In-run offwindow reads are WATCH-ONLY; the decisive read is standalone `exploit_probe.py`.
3. **Off-window baseline:** ~52% of self-play forced-wins are off-window. The encoding (single→multi-window `v6_live2_ls`) should hold or improve this. >55% + low components ⇒ encoding failed to expand the window.
4. **Draw-rate L9:** an abort is NOT a model failure — it reflects cosine-temp + jitter × MCTS regime. If it fires, check the LR tail decays smoothly to `eta_min` (no oscillation — `project_cosine_resume_tmax_footgun`).
5. **Promotion dead-loop:** if 0 promotions and best_model.pt stays at bootstrap, confirm (a) best stride 1, (b) n=400 measuring, (c) no draw spikes killing CIs, (d) bootstrap_anchor stride 1. If all OK and still none ⇒ encoding weaker than bootstrap (multi-cluster strength cost — which is exactly what the SECONDARY strength guard is there to DETECT).
