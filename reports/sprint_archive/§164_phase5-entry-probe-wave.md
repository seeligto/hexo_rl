<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §164 — Phase 5+ entry probe wave: P1 anchor / P2 perception / P3 corner-mask — 2026-05-07

### Context

§157 closed Path B (skip 40k sustained, preserve dev cycles for encoding migration).
§158-§163 landed hygiene + 5 refactors (training/loop, StepCoordinator, eval_pipeline,
selfplay/pool, mcts/policy). With master at 284b57a and the 8-plane v6/v7 trunk
about to be obsoleted, three probes were dispatched in parallel to inform
encoding-migration scope before any code is written:

* **P1** — window-anchor boundary: Bug / Principled / Aug-only?
* **P2** — asymmetric perception: deployment vulnerability when bot's perception is
  r=5 but the official site allows r=8 placements?
* **P3** — corner-mask hex-shape test: does zero-masking the 90 always-zero corner
  cells (axial parallelogram → inscribed regular hex, hex_dist ≤ 9) help, hurt, or
  neutralise on bench + 1k smoke A/B?

P1 + P2 ran on laptop (read-heavy / 200-game smoke). P3 ran on 5080 vast.ai
(bench + 1k smoke A/B + SealBot eval). All three in worktree isolation; nothing
landed on master in this wave.

Pre-flight: `make test` PASS (991 + 8 skipped). `make bench` n=3 baseline captured
to `reports/probes/baseline_bench.{txt,json}`. All 10 metrics PASS against floor;
worker pos/hr median 31,434 (target ≥ 20k).

### P1 — Principled (memory misread)

Phase 1 (code-read) conclusive in ~25 min. The memory note "Rust returns K
candidate window anchors; Python takes index 0 at training+inference boundary"
is incorrect on every hot path that drives a trained model:

* Live self-play: `worker_loop.rs:299-401` forwards all K cluster views to NN,
  min-pools value, scatter-max policy. Replay buffer push (`worker_loop.rs:649-682`)
  emits one training row per cluster per ply (not per leaf, not per index 0).
* Live inference: `selfplay/inference.py:37-108` mirrors Rust — K-batched
  forward, min-pool value, scatter-max policy.
* Eval / community-bot routes through Rust MCTS — same K aggregation.

Index-0 picks exist only in `pretrain.py:564,568` (RandomBot post-training
validation greedy bot, **Aug-only**) and `early_game_probe.py:103`
(monitoring fixture, **Aug-only**). `records.rs:48` pass-slot copy is dead
in Hex Tac Toe (no pass action). Massive-cluster anchor dedup at radius 5
keeps the newest move inside *some* window's stone planes (radius 5 ≤ window
half 9), even when its dedicated window is suppressed by a 2-3-step-older
action anchor.

Bootstrap corpus encoding (`dataset.py:45-52`) picks the **first cluster
that covers the played move**, not index 0. Principled-by-design — a
one-hot supervised target needs a single window. Same limitation as
per-cluster live self-play row push; self-consistent.

**Recommendation:** close the OPEN memory item as Principled. Optional
tidy-ups (records.rs:48 → `0.0`, pretrain.py:568 → cluster-covering-move
pick) — no behavioural change.

**Encoding-migration impact: none.** K-aggregation is invariant to plane
encoding.

Audit: `audit/probes/p1_window_anchor.md`.

### P2 — CATASTROPHIC (deployment vulnerability)

Pre-probe verification confirmed: rule = r=8 placement
(`docs/reference/hexo_package_notes.md:25-26` — `hexo` v0.2.0 default
`placement_radius=8`); our perception = r=5
(`engine/src/board/moves.rs:20` `DEFAULT_LEGAL_MOVE_RADIUS=5`,
`moves.rs:32` `CLUSTER_THRESHOLD=5`). `apply_move` accepts placements at any
empty cell — no engine modification needed to emulate r=8 from bot's POV.

Probe: three opponents × 200 games against `bootstrap_model_v7full.pt`
(§150 anchor, in_channels=8):

| Opponent | Bot WR | Opp colony ≥6 reach | Mean opp final colony in losses |
|---|---:|---:|---:|
| **`far_line`** (r=6-8 6-axis script) | **0.780** | **22.0%** | **6.0** (axis-aligned six-in-a-row) |
| `far` (r=6-8 random) | 1.000 | 0.0% | n/a |
| `control` (r≤5 random) | 1.000 | 0.0% | n/a |

In `far_line` opp-win games (n=44), mean opp colony-reach-6 ply ≈ 27 and
**42.9% of placed far stones never receive a bot response**. v7full's
SealBot baseline WR is 17.4% (n=500, §150) — i.e., a brain-dead scripted
adversary outperforms the strongest engine's empirical edge.

Mechanism: stones at hex_dist 6-8 from any bot stone form their **own
cluster** (`CLUSTER_THRESHOLD=5`) with their own 19×19 window. The policy
treats them as low-priority because there's no spatial relationship encoded
between bot windows and far windows.

**Recommendation:** encoding migration MUST include perception expansion.
Three options:

* **(a)** bump `DEFAULT_LEGAL_MOVE_RADIUS` and `CLUSTER_THRESHOLD` to 8.
  Small scope; re-opens §142/§144 fragmentation pathology. RISK.
* **(b)** hybrid: r=5 for self-play move-gen; r=8 only for inference +
  cluster partition. Per-Board override exists; needs PyO3 bindings for
  `set_legal_move_radius` + new `set_cluster_threshold`. **PREFERRED for
  short-term hotfix.**
* **(c)** 25×25 window (HALF=12). Large scope; trunk re-init; native fit
  for r=8 stones. Encoding migration must budget this anyway.

**Deployment hotfix REQUIRED.** Do NOT deploy v7full or successors to
hexo.did.science until option (b) lands and the same probe shows
`opp_winrate < 5%` with `far_line`. PyO3 bindings for
`set_legal_move_radius` + `set_cluster_threshold` are missing today.

Note: under r=8 cluster threshold, the "Massive Cluster" anchor-window
path (`state.rs:665+`) becomes the **common** case (currently rare for
span > 15). Validate before deploy.

Audit: `audit/probes/p2_asymmetric_perception.md`.
Probe code: `tests/probes/p2_far_placement_opponent.py`.
Game artifacts: `reports/probes/p2_{far,far_line,control}_{games.jsonl,summary.json}`.

### P3 — Neutral within noise (mild positive on bench + self-play; SealBot WR Δ −2.5pp NOT significant)

Worktree `probe/p3_corner_mask` (HEAD `1c9e88a`) carries 5 probe commits:
engine `CORNER_MASK_ENABLED` AtomicBool flag, `--corner-mask` bench harness
flag, A/B harness script, variant configs, 1k smoke + SealBot eval launcher.
The engine patch mirrors the v9 prior art (`3fd7ebd`) — OnceLock'd 361-cell
mask LUT applied to stone planes 0 and 8, default off, 271 cells survive
(`hex_dist((q,r),(0,0)) ≤ 9`).

The P3 subagent completed the bench A/B then exited silently before
invoking the smoke harness. Main agent restarted the smoke directly on
5080 after patching `mixing.buffer_persist: false` into both variants
(prevents OFF→ON crossover taint via shared `replay_buffer.bin`) and
deleting the pre-existing `replay_buffer.bin` from a prior 5k smoke. 90
min total wall (OFF training 30 min + OFF SealBot 15 min + ON training
30 min + ON SealBot 15 min).

Bench A/B (5080, n=3 each) PASS on both arms:

| Metric | OFF | ON | Δ% |
|---|---:|---:|---:|
| NN latency batch=1 ms | 1.54 | 1.49 | −3.47% |
| Buffer push pos/s | 971,950 | 992,686 | +2.13% |
| Buffer sample raw µs | 774.8 | 744.9 | −3.86% |
| Buffer sample augmented µs | 773.0 | 710.2 | **−8.12%** |
| Worker pos/hr | 76,424 | 91,703 | **+19.99%** |

Surface-it threshold (1-5%) grazed by aug + worker. Aug speedup
mechanistically plausible (90/361 = 25% scatter elements zeroed). Worker
+20% wants n=5 confirmation. **No regression STOP fired.**

Smoke health (last 100 games per arm):

| Metric | OFF | ON | Δ |
|---|---:|---:|---:|
| Draws (ply_cap) | 11/100 | 7/100 | −4pp |
| six-in-a-row terminations | 89/100 | 93/100 | +4 |
| Player 0 / 1 / draw | 52 / 37 / 11 | 47 / 46 / 7 | ON more balanced |
| `colony_extension_fraction` max | 0.0303 | 0.0000 | −0.030 |

ON arm marginally cleaner across the board. Within n=100 noise but
directionally consistent. Note: per-game stride5/row_max metrics are
**dashboard-only on master** per §157 follow-up #2; ply_cap fraction
substituted as a fair proxy for the §157-era stride-5 lock pathology.

SealBot WR (n=200 each, time_limit 0.5, model_sims 128 — matches §157
Gate 4):

| Arm | WR | wins | colony_wins | 95% CI |
|---|---:|---:|---:|---|
| OFF | **0.180** (36/200) | 36 | 2 | [0.127, 0.233] |
| ON | **0.155** (31/200) | 31 | 3 | [0.105, 0.205] |

Δ = −2.5pp (ON < OFF). Combined SE ≈ 0.037; **NOT statistically
significant** at α=0.05 (CIs overlap heavily). Both arms within
v7full's §150 baseline 17.4% sample noise. 1000 iterations from a
strong bootstrap is short — neither arm has materially diverged.

**Outcome:** mask is **safe to ship**. Does not clearly win in 1k
iters from v7full bootstrap; expected pattern given C6-symmetry change
requires many gradient steps to materialise as in-distribution play
strength.

**Recommendation for D3 (window shape): adopt inscribed hex** on the
§152 dihedral-symmetry argument, now reinforced by P3 bench (no
regression, mild speedups) + smoke (mild positive on self-play
health, neutral on SealBot WR). Confidence: medium-high.

Audit: `audit/probes/p3_corner_mask.md`. Bench artifacts:
`reports/probes/p3_bench_{off,on}.{txt,json}`. Smoke + SealBot
artifacts: `reports/probes/p3_smoke/{off,on}/{events.jsonl,sealbot_eval.{jsonl,log},train.log,final_ckpt.txt}`.

### Encoding-migration scope adjustments

| Dimension | Pre-probe default | Post-probe verdict |
|---|---|---|
| **D1 — Window anchor** | OPEN memory item | **No action.** P1 Principled; memory close-out only. |
| **D2 — Perception window** | r=5 status quo | **MUST expand.** P2 catastrophic. Hotfix (b) before deploy; option (c) native fit for encoding migration. |
| **D3 — Window shape** | OPEN per §152 | **Adopt inscribed hex** (P3 bench + smoke + §152 dihedral-symmetry). Bench: no regression. Smoke (1k iters, n=200 SealBot/arm): neutral within noise, mild bias toward ON on self-play health. |

### Pre-conditions for encoding migration entry

* Existing pre-conditions (§157 Path B) carry forward.
* **NEW: deployment hotfix (b) shipped + smoke-validated** (P2). Independent
  of encoding-migration entry — do not deploy to hexo.did.science before
  hotfix lands. Hotfix may merge into encoding-migration scope or land
  standalone.
* **NEW: PyO3 bindings for `set_legal_move_radius` and `set_cluster_threshold`**
  — prerequisite for hotfix (b) and any r-asymmetric flow.
* **NEW: massive-cluster path validation under r=8 cluster threshold** —
  span > 15 case becomes common; verify anchor-windowing path is sound.

### Outstanding §157 follow-ups still open

* **#2 — stride5/row_max → events.jsonl.** Still dashboard-only on master;
  P3 verdict will indicate whether the smoke gate found this blocking.
* **#4 — `sealbot_colony_bug_risk` startup-warning predicate review.**
  Out-of-scope for this wave; flag for §165 if not addressed alongside
  encoding migration.

### Verdict

§164 wave delivers three class verdicts before any encoding-migration
code is written:

* **P1 closes a memory-flagged OPEN item with no scope impact.** Encoding
  migration does not need to touch anchor selection.
* **P2 surfaces a CATASTROPHIC deployment vulnerability** that re-shapes
  encoding-migration scope on the perception axis from "consider" to
  "MUST". A standalone hotfix is required before any live deployment.
* **P3** is neutral within sampling noise on play strength (SealBot WR
  Δ = −2.5pp not significant) with mild positive bias on bench (no
  regression, +2-8% on aug + buffer ops, +20% worker pos/hr — borderline
  noise) and on self-play health (fewer draws, more decisive
  terminations, better player balance, lower colony-extension max).
  The inscribed hex shape is **safe to adopt** for encoding migration on
  the §152 dihedral-symmetry argument, now reinforced by bench + smoke
  evidence.

### Commits in §164

None on master. Probe wave is read-only / branch-only by design.
* P1: no branch (read-only verdict).
* P2: worktree branch `worktree-agent-a05ca075ad606b19e`; opponent file
  copied to `tests/probes/p2_far_placement_opponent.py` as untracked
  artifact; audit + reports artifacts in `audit/probes/` + `reports/probes/`
  (gitignored).
* P3: branch `probe/p3_corner_mask` (5 commits). Will not merge to master
  in this wave per probe-wave discipline. Bench artifacts rsync'd from
  5080 to `reports/probes/p3_bench_{off,on}.{txt,json}`. Smoke + SealBot
  artifacts in `reports/probes/p3_smoke/{off,on}/`. Variant configs were
  patched in-place on 5080 with `mixing.buffer_persist: false` to clean
  up cross-arm taint — patches NOT pushed back to laptop probe branch
  (operator decision pending on whether to keep the variant edits).

### What this sprint DOES NOT do

* Does not begin encoding migration.
* Does not ship hotfix (b) — flagged as required follow-up for operator
  decision (standalone vs bundled).
* Does not land any commits on master.
* Does not change `LEGAL_MOVE_RADIUS` or `CLUSTER_THRESHOLD` constants.
* Does not auto-launch sustained 40k run.

---

