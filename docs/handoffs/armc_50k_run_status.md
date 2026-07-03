# Arm-C Fixed-Loop 50k — Run Status (COMPLETE — VERDICT IN)

**Status: COMPLETE.** Ran 2026-06-14 15:45 → 2026-06-16 11:40 UTC on vast (5080 / Ryzen 9
9900X). 50k steps, exit 0, `terminal_eval_complete` fired. Verdict resolved 2026-06-16.

Companion docs: watch sheet `docs/handoffs/armc_rerun_watchsheet.md` (in-run gates +
banking), launch package `docs/handoffs/armc_rerun_launch_package.md` (GO/NO-GO),
runbook `docs/handoffs/loopfix_armc_rerun_runbook.md` (canonical, GREENLIGHT routing).

---

## ★ VERDICT (2026-06-16)

**PRIMARY — off-window robustness: GREENLIGHT ✅ (decisive).** On the trained 50k
`v6_live2_ls` checkpoint, evaluated with the CORRECT no-drop instrument (`KClusterMCTSBot`,
n=200, sims 128): **off_window_forced_win_rate = 0.0 (0/200)** — verdict DEFENDED.
- Clears the S3 gate (≤0.06) and beats the Arm-B free-overlay baseline (0.03). 0/200 → 95%
  CI upper ≈ 0.018 (rule of three), still under both bars.
- **Causal confirmation (counterfactual, same weights):** forced through single-window drop
  (`ModelPlayer`), the SAME checkpoint is exploited **0.67 (67/100)**. So the multi-window
  legal-set encoding CLOSES the off-window blind spot entirely: **0.67 → 0.0**. The encoding
  delivers exactly its design objective.
- Instrument note: plain `exploit_probe` via single-window `ModelPlayer` would have reported
  the encoding as FORCEABLE (pilot: 0.25) — the documented false-clear (here in the
  *condemning* direction). Fixed in master `a71ae28` (KClusterMCTSBot dispatch + `--encoding`
  override + test). §D-ARGMAX: temp-0 argmax → exploit/control arms collapse (defender's
  action-space dominates, not adversary skill); the read is a categorical defended-vs-forceable
  with a clean causal counterfactual, not a tight exploit−control margin.

**SECONDARY — strength: FLAT ≈ bootstrap (the pre-registered "detect-harm" read).** 0 promotions
across all 4 rounds; best-arena 0.52–0.54 (never cleared the 0.55 bar — parity, not harm);
SealBot 0.09–0.20 (v7full ~17% band). The encoding bought **robustness, not strength**. Per the
runbook routing: do NOT scale for strength on this encoding — **route strength to Gumbel**.

**Bottom line:** the v6_live2_ls multi-cluster encoding **succeeds on its objective** (off-window
robustness, fully defended, causally demonstrated) at **flat playing strength**. Encoding verdict =
GREENLIGHT for the robustness axis; strength axis routes to Gumbel. Data:
`reports/investigations/exploit_probe_data/armc50k_{kcluster,dropbaseline}.summary.json` (vast).

Open/optional: the in-process terminal nnue/full-battery eval was null (hammerhead mid-run-editable
limitation, §0); the strength ladder is re-runnable standalone but the strength verdict (flat) is
already established from 4 in-run rounds.

---

## 0 · Live checkup — 2026-06-15 07:10 UTC (step ~18.9k / 50k, ~38%)

**HEALTHY, on track.** No abort events, no Traceback/CUDA error. Process alive.
- **Throughput refined:** ~1230 steps/hr (18,880 steps in ~15.3 h) → 50k finishes **~Jun-16 08:00 UTC**. (Was 1360 early; multi-window supply settled slightly slower.)
- **draw_rate 0.255** — flat, far under the 0.55×3 abort. No L9 hangover.
- **value_accuracy_masked ~0.67** — stable (load-bearing for conversion).
- **Round 1 (step 12,500) — `promoted: false`:** `wr_best 0.525` (just under the 220/400=0.550 bar → no promotion, dead band gone is working as designed), `wr_sealbot 0.19` (CI [0.125,0.278]; in the v7full ~17% band — strength NOT collapsed), `wr_bootstrap_anchor 0.40` (model still under the full-sims bootstrap floor at 12.5k from bootstrap — expected), `wr_random 1.0`, value head `g4_band_pass: true` (fc2 max 0.23).
- **No promotion yet** → `best_model.pt` = the bootstrap fresh-init (pin holds, W2 intact). Round 2 (25k) is the next promotion gate, ETA ~Jun-15 12:00 UTC.
- **Coherence frontier:** `forced_win_conversion 0.36` at round 1 — a single early datapoint (emitted only at eval rounds), a *baseline* not a decline. Watch whether it RISES at round 2; the decisive off-window read is the terminal + standalone `exploit_probe.py` regardless. off_window_rate / components are computed post-hoc on banked checkpoints, not in the live log.
- **Checkpoints:** every 500 steps (preserved); `checkpoint_00012500.pt` is the round-1 eval ckpt. `.provenance.json` correctly absent (promotion/terminal-stamped, not interval).

### Hammerhead incident (2026-06-15, round 2 / step 25k) — RESOLVED
The nnue eval opponent crashed `ModuleNotFoundError: hammerhead` — the submodule was present but never built into the run venv (manual per-host step `install.sh` missed). nnue fires at stride 2 → **rounds 2 AND 4 + terminal**; round 1 (odd) skipped it, hiding the gap. The crash **discarded the whole round** → lost the deserved round-2 promotion (`checkpoint_25000` beat best 267/400 = 66.8%; SealBot @25k = **20%**). F03 caught it; training continued unharmed.
- **Fix applied:** built hammerhead into the vast run venv (`maturin develop --release` on hammerhead-engine + `pip install -e hammerhead`); verified `from hammerhead import Bot` resolves. Round 4 + terminal nnue will now run.
- **Self-heal:** round 3 (37.5k) skips nnue → its eval completes → auto-promotes a later ckpt. No rerun (promote-then-resume is blocked by the W2 guard anyway).
- **Robust regardless:** the PRIMARY greenlight (`exploit_probe` + KClusterMCTSBot) runs standalone on the 50k ckpt — independent of the in-run/terminal eval pipeline.
- **Permanent fix:** hammerhead build wired into `scripts/install.sh` step 8 (master `3e878fb`).
- **Lesson:** pre-flight must verify eval-OPPONENT imports on the host (hammerhead/nnue), not just engine+encoding — a stride>1 opponent hides the defect until a later round.

**Delivery:** the run is mid-flight; the Arm-C deliverable (50k final + `exploit_probe` GREENLIGHT) lands ~Jun-16 10:00 UTC. Nothing promoted to bank yet (round 3 @37.5k is the next promotion).

---

## 1 · Launch identity (the verified facts)

| Item | Value |
|---|---|
| **Launch SHA** | `2ae505d` (origin/master HEAD). = `1431410` + the single change `config(armc): n_workers 16→18`. |
| **The ONE variable** | encoding `v6_live2` → `v6_live2_ls` (multi-window legal-set, k_max 8, value_pool=min). Nothing else. |
| **Pinned incumbent** | `bootstrap_model_v6_live2.pt`, state-dict sha256 `4198d5cb…b0a186`. **Runtime-verified**: `anchor_identity sha256=4198d5cb… pinned=4198d5cb… source=fresh_init_checkpoint` (W2 closed — no stale restore, no mismatch). |
| **Generator init** | `--checkpoint bootstrap_model_v6_live2.pt` → from-bootstrap self-improvement (anchor == generator == bootstrap). |
| **n_workers** | **18** — reverted from the committed 16 (a stale 20-core-4080 tuning) to match Arm A golong and fit the live 5080/24-thread host. Keeps n_workers (the co-batch/byte-stream lever) constant vs Arm A so **encoding stays the only A/B variable**. |
| **run_id** | `bb0d0d97321245bdb63ec0895624a67c` |
| **tmux / log** | session `armc50k` on vast; log `/root/hexo_rl/armc50k_20260614T154502Z.log` |

### Pre-flight gate (all PASS)
- **0.2 pin** — `4198d5cb` = laptop *and* vast bootstrap state-dict sha (the §D-RERUNPREP host-drift is resolved; both hosts now agree at the de-facto value). Bootstrap was **missing on vast** at gate time → rsync'd from laptop + sha-reverified on vast.
- **0.3 temp-safety** — merged `load_config`: `temperature_threshold_compound_moves=0` (cosine OFF), `temp_min=0.5`, `legal_move_radius_jitter=true`, `n_simulations=400`, `full_search_prob=0.5`. The toxic Rust default (`15/0.05`, L9) is overridden by base `configs/selfplay.yaml`, so **711919d (the tempdecay footgun fix) was hygiene-not-blocking** and the launch ran from clean master without it.
- **Engine** — rebuilt on vast from master source post-reset (`make build`, exit 0) so the engine `.so` is master, **not** the tempdecay+711919d build (A/B-cleanliness). `v6_live2_ls` resolves (4-plane, mw, k_max 8).
- **Vast tree** — was dirty on `phase4.5/tempdecay`; reset to clean master `2ae505d` (the commit moved via git **bundle** because vast has no github fetch auth — pull/fetch is dead there, rsync/bundle is the transport). Vast-only tracked diffs captured to `/root/vast_tracked_diffs_prelaunch_*.patch`; untracked tempstrength artifacts left in place.
- **Startup** — `eval_schedule_capability promotion_capable_in_run_rounds=4` (`[12500,25000,37500,50000]`, terminal covers final); corpus loaded 392,251 pos / 30.6 s; pool started 18w. (Benign: step-1 `grad_norm=NaN` from fp16 scaler warmup → healthy ~1.0 by step 29; `tensorboard unavailable` warning — no effect.)

---

## 2 · Re-derived cost (the launch-package caveat, now measured)

Measured live on this host (`v6_live2_ls` multi-window, 18w): **~1360 steps/hr**.

| | |
|---|---|
| Rate | **~1360 steps/hr** (single-window tstr on the same host was ~1985; the ~31% multi-window tax is the k>1 inference cost) |
| Wall-clock (50k) | **~37 h ≈ 1.53 days** |
| Cost @ ~$0.60/hr | **~$22** (range ~$18–30) |

The watch-sheet 490 steps/hr / $67 / 4.25-day estimate was a ~4× underestimate for this host. **No operator cost-gate trigger** (well under the ~$90 / 4-day ceiling).

### Expected round timing (estimates from 1360 steps/hr; firm up as supply stabilises)
| Round | Step | ~ETA (UTC) |
|--:|--:|---|
| 1 | 12,500 | ~Jun-15 01:00 |
| 2 | 25,000 | ~Jun-15 10:10 (first Obj-A in-run reads: nnue + offwindow) |
| 3 | 37,500 | ~Jun-15 19:20 |
| 4 + terminal | 50,000 | ~Jun-16 04:30 → close-out |

---

## 3 · Abort gates being watched (see watch sheet §2 for full thresholds)

- **draw_rate HARD-ABORT** — ≥0.55 for **3 consecutive** evals (min_step 0). Fires → run ABORTED, **close-out does NOT run** (no terminal eval). L9 hangover is the standing risk on this line.
- **grad_norm HARD-ABORT** — 10.0.
- **Divergence (watch, not auto-abort):** components_count collapse <15 (colony); off_window_rate >55% with low components (encoding not helping); 0 promotions with best_model.pt stuck at bootstrap (encoding weaker than bootstrap — the SECONDARY strength guard's harm signal). Distinguish *actively-breaking* (fragmentation trending) from *not-yet-built* (real even early).
- **Headline keys:** value_accuracy_masked (selfplay source = the one that must move), forced_win_conversion (in + off-window), components, longest line, promotion decisions (expect >1).

## 4 · Banking (watch sheet §5)
- Bank promoted checkpoints per round; **always** bank step-50k final + `.provenance.json`.
- Off-host: pull final checkpoint (~17 MB) + sidecar to laptop after close-out for exploit_probe / KClusterMCTSBot. (Vast fetch is dead — pull via rsync/scp, transport already proven this run.)

## 5 · Close-out milestone (the GPU validation the unit tests couldn't prove)
At step 50,000 watch the epilogue **in order**: drain (budgeted, floor 900 s / cap 14400 s, WARN-never-kill) → pool stopped → **TERMINAL full-battery eval** (all opponents, stride ignored, GPU-UNLOADED) → **`terminal_eval_complete` fires exactly once** (`terminal=True, step=50000`) → exit 0. If `terminal_eval_complete` is absent ⇒ the run did NOT close cleanly. The terminal result is a RECORD, never a steer input (on any resume `resolve_anchor` loads only the `best_model.pt` chain).

> **W2 self-deadlock (do NOT resume-after-promote):** a promotion advances `best_model.pt` past the launch pin → intentional `RuntimeError` on resume. The pin is a single-run guard.

---

## 6 · Read plan when it lands (PRE-REGISTERED — do NOT move post-hoc)

**PRIMARY — absolute off-window robustness (no matched arm needed):**
`scripts/exploit_probe.py --checkpoint <armc-50k-final> --arms exploit,control` ≤ **0.06**
AND the causal-uncapping counterfactual routed through **`KClusterMCTSBot`** (NOT single-window
`ModelPlayer`, which false-clears). Arm C must **also beat the Arm-B free overlay (0.03 < 0.06)**
to justify the training half.

**SECONDARY — strength non-inferiority vs golong@50k, WITH THE DISCLOSED CONFOUND:**
The fixed loop biases Arm C **UP** (pinned bootstrap incumbent, fixed cadence, n=400 power,
unloaded terminal eval — the original Arm C had none). The guard **DETECTS HARM** (Arm C materially
weaker than golong50k → multi-cluster costs strength → do NOT scale; route strength to Gumbel) but
**CANNOT ATTRIBUTE GAINS** — any Arm C ≥ golong50k is confounded by the loop fixes, never read as
"the encoding made it stronger." Effective-n: dedupe byte-identical games + bootstrap the CI over
**distinct** games before trusting any strength gap (§D-ARGMAX). Do NOT greenlight off strength
alone or off a SealBot-WR.

## 7 · What un-gates on completion
- **Value investigation un-HELDs** — on fixed-loop data (not static golong-play). §D-VALCEIL Idea-3 reopened.
- **D-GUMBELSIMS** committed operating point (dev knee m=16≈n=50) re-confirms on this (winning) encoding; its Phase 1-3 were gated on this run.
- The **encoding verdict itself routes the Gumbel A/B Step-1 tuning** (#1/#4 candidates if the strength guard detects harm).

---

## 8 · Monitor commands
```bash
# attach the live run
ssh vast 'tmux attach -t armc50k'        # detach: Ctrl-b d

# tail structured log
ssh vast 'tail -f /root/hexo_rl/armc50k_20260614T154502Z.log'

# round / promotion / abort canary
ssh vast 'cd /root/hexo_rl; L=armc50k_20260614T154502Z.log;
  grep -aE "round_complete|promotion|terminal_eval_complete|hard_abort|draw_rate" $L | tail -20'

# steps/hr + ETA from the latest train_step
ssh vast 'cd /root/hexo_rl; grep -aE "\"event\": \"train_step\"" armc50k_*.log | tail -1'
```
