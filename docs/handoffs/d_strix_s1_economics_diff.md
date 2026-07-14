# D-STRIX S1 — hexo-strix vs HeXO training-economics diff (READ-ONLY audit)

**Scope:** comparative economics audit only. No hexo-strix code vendored, no HeXO
code changed. hexo-strix cloned read-only to
`/tmp/claude-1000/.../scratchpad/hexo-strix` (not committed, not part of this repo).

## Target repo identification

- **URL:** `https://github.com/SootyOwl/hexo-strix` (author SootyOwl, MIT).
  Not linked from any HeXO doc, submodule, or bot-integration file — found via
  GitHub repo search (`gh search repos`, confirmed via `mcp__github__search_repositories`).
- **Commit audited:** `1b8ae4d1d2a3c1821a76485470c3314027251ccb` (branch `main`).
- **Repo age at audit time: the entire git history spans 2026-07-02 17:13–18:56
  (under 2 hours), 16 commits total.** This is almost certainly a fresh
  single-day export/mirror of a much older private project — internal code
  comments reference dated findings from 2026-05-13, 2026-05-26, 2026-06-02,
  2026-06-05 (`docs/research/2026-06-05-forward-pass-headroom/findings.md`,
  etc.) that predate the public repo by 4-8 weeks. **Commit dates cannot be
  used to infer real development/training calendar time.**
- **CRITICAL — `docs/` is gitignored** (commit `e95e451 add docs directory to
  gitignore`), along with `runs/`, `data/`, `*.pt`, `benchmark.txt`,
  `trajectories.json`, `*.sqlite`. Every research finding, eval log, and
  wall-clock/throughput number the code comments *refer to* (e.g. "2026-06-05
  vl-fidelity", "2026-06-10 human-corpus-validation", "100k+ steps" for
  `4l-128p32v`) is **absent from the audited artifacts**. The repo is
  code + config only; nothing here is a run record.
- **The task brief's "days-from-zero" quantitative claim does not appear
  anywhere in this repo.** Checked `README.md`, `hexo-a0/README.md`,
  `hexo-rs/README.md` for "day(s)", "week(s)", "hour(s) to", "wall-clock" —
  zero hits. Whatever that claim's source is, it is external to this audit's
  artifacts and is **UNVERIFIABLE** from what was cloned.

## Diff table

"Theirs" = `configs/gine-mini/4l-128p32v.toml`, self-labeled in its own header
comment as **"production-strongest, no-JK"** config (100k+ steps per its
in-file changelog) — the closest thing to a canonical flagship recipe in the
repo. "Ours" = current HeXO `configs/model.yaml` + `configs/training.yaml` +
`configs/selfplay.yaml` + `configs/variants/vast.yaml` (canonical vast 5080
production variant).

| Metric | hexo-strix (theirs) | HeXO (ours) | Ratio (ours÷theirs) | Flag | Source |
|---|---|---|---|---|---|
| Param count | **283,970** (0.284M) | **4,254,283** (4.25M) | 15.0× | **MEASURED** (both — direct model instantiation this session) — **CORRECTION 2026-07-14 (WP0.3):** theirs was mis-recorded as 222,146 (0.222M / 19.1×); actual 283,970 (0.284M / 15.0×) | `hexo-a0/src/hexo_a0/model.py` + `4l-128p32v.toml` dims; `hexo_rl/model/network.py` + `configs/model.yaml` (board 19, in_ch 8, res_blocks 12, filters 128) |
| Architecture class | GNN, GINEConv, axis-window graph, 4 msg-passing layers, per-node policy MLP + stone-pooled value MLP | CNN, ResNet-12 + SE blocks, GroupNorm, spatial policy conv+FC + GAP/max value head + 6 aux heads | — (not FLOP-comparable, see caveats) | MEASURED (code) | `hexo-a0/src/hexo_a0/model.py`; `hexo_rl/model/network.py` |
| Self-play search algo | **Gumbel-root + PUCT-interior, ALWAYS** (no alternative implemented) | PUCT + dynamic FPU + quiescence-override + Dirichlet noise **by default** (`gumbel_mcts: false`); Gumbel reserved for eval/deploy | **train≠deploy on our side; train==deploy on theirs** | MEASURED (config) | `hexo-a0/src/hexo_a0/model.py` docstring; `configs/selfplay.yaml` (`mcts.dirichlet_alpha/epsilon`, `selfplay.gumbel_mcts: false`) |
| Deploy/eval search | Gumbel SH, m=16, n=128 (production stage) | Gumbel SH, m=16, n=150, g=0 (gumbel_scale=0), greedy, no temp | m equal; n 1.17× ours | MEASURED (config) both | `4l-128p32v.toml` `[mcts]`; `docs/07_PHASE4_SPRINT_LOG.md` §2420 (`deploy_strength_eval.py`) |
| Self-play sims/move (data-gen) | 128 (production); 16–45 in earlier/ablation configs | 400 fixed (`selfplay.yaml`/`vast.yaml`); playout-cap blend n_sims_quick=100/n_sims_full=600 at full_search_prob=0.5 in some variants | ~3.1× (400 vs 128) | MEASURED (config) | `4l-128p32v.toml`; `configs/selfplay.yaml`, `configs/variants/vast.yaml` |
| Batch size (train step) | ceiling 1,536 rows; **edge_budget 1,500,000** (dynamic — graphs vary in size/curriculum stage) | fixed 256 rows (constant-shape tensor, cost-predictable) | — (not directly comparable, see localization) | MEASURED (config) | `4l-128p32v.toml` `[training]`; `configs/training.yaml` |
| Replay buffer capacity | 250,000 positions (most configs incl. 4l-128p32v); 125,000 (gine-midi) | 500,000 → scheduled to 1,000,000 (`buffer_schedule`) | 2–4× | MEASURED (config) | `4l-128p32v.toml`; `configs/selfplay.yaml`, `configs/training.yaml` |
| LR schedule | peak 5e-4 → floor 2e-5, cosine, warmup 400 steps, `total_train_steps`=60,000 **per curriculum stage** (resets each stage) | peak 2e-3 (or 1e-3 `vast.yaml`) → floor 5e-4, cosine, `total_steps`=200,000, **single monolithic run** (no stage resets — no board-size curriculum) | — | MEASURED (config) | `4l-128p32v.toml`; `configs/training.yaml`, `configs/variants/vast.yaml` |
| Loss composition | KL policy + MSE value **only** (2 heads) | KL/CE policy + value + **6 auxiliary heads**: opp_reply (0.15), value-uncertainty (0.1), ownership (0.2), threat (0.2), chain-length ×6 planes (1.0), ply-index (0.0, disabled) | — | MEASURED (code+config) | `hexo-a0/src/hexo_a0/model.py` (no aux heads present); `configs/training.yaml` weight block |
| Resign rule | none (play to `max_moves`, draw) | none (play to `max_game_moves`=150 plies, draw) | parity | MEASURED, parity | both configs |
| Opening randomization | `exploration_moves` 16–30 sampled from improved-policy visit distribution; Gumbel noise itself is the root exploration mechanism (no separate Dirichlet layer) | Dirichlet(α=0.05, ε=0.10) root noise + `random_opening_plies`=1 (0 in `vast.yaml`) + `legal_move_radius_jitter` | different mechanism class, not a scalar ratio | MEASURED (config) | `configs/gine-mini/4l-128p32v.toml` `[mcts]`; `configs/selfplay.yaml` |
| Curriculum | 5 explicit stages, win_length 4→6 **and** radius 2→8, auto-convergence gating (win-rate / champion-SPRT thresholds, `patience`) | **no** win-length/board-size curriculum — fixed 6-in-row, v6 encoding (19×19 window) from bootstrap step 0; closest analog is `legal_move_radius_schedule` 5→6→7→8 (self-play move-space widening only, same win_length throughout) | structural difference, not a ratio | MEASURED (config) | `configs/axis/curriculum.toml`, `configs/gine-mini/*.toml` `[[curriculum.stages]]`; `configs/variants/vast.yaml` `legal_move_radius_schedule` |
| Self-play throughput (games/hr, own hardware) | **not measured / never committed anywhere in the repo** (script exists, was never run — see below) | **5,522 games/hr** (RTX 5080, PUCT n_simulations=400, `b128_w16_n18` winning sweep config, §174 2026-05-12) | n/a | ours **MEASURED**; theirs **CLAIMED-ABSENT** (no number exists to even cite as claimed) | `configs/variants/vast.yaml` header comment; theirs: `scripts/bench_self_play.py` (not run — needs Rust `self_play` binary build, not CPU-cheap, skipped per task instructions) |
| Worker/position throughput (bench-harness proxy, NOT production config) | not run this session (would require building `hexo-mcts --features torch`, a full Rust+libtorch build — declined as not CPU-cheap) | 91,871 pos_gen/hr (vast 5080, `n_workers=22`, 200-sims/move bench workload); 33,565 pos_gen/hr (laptop RTX 4060) | n/a | ours MEASURED, own harness; theirs not run | `docs/rules/perf-targets.md` |
| Engine micro-bench (game-logic only) | `apply_move` 5.1µs, `clone` 490ns/771ns, `legal_moves` 21–35µs (radius-8, 6-in-row; **committed criterion table in README**) | not pulled this session (out of scope — would need a fresh `cargo bench -p hexo-engine` run) | n/a | theirs **MEASURED+COMMITTED**; ours not re-run | `hexo-rs/README.md` § Performance |
| Hardware | single ROCm-backed APU, unified memory, single iGPU shared between training and self-play (repo comments describe "~7x" throughput hit when both contend for the one device); exact chip **never named** in the repo | vast.ai rental: Ryzen 9 9900X + RTX 5080 (17.1 GB VRAM), discrete GPU | n/a | theirs **CLAIMED/INFERRED** (repo name "hexo-strix" + generic "ROCm APU" comments are consistent with an AMD Ryzen AI Max "Strix Halo"-class part, Radeon 8060S iGPU / RDNA3.5, but this is inference, not a confirmed spec in the repo); ours **MEASURED/pinned** | strix: scattered comments in `hexo_a0/config.py`, `inference_server.py`, `curriculum.py`, `sprt_daemon.py`, `sprt_watcher.py`; HeXO: `docs/rules/perf-targets.md`, CLAUDE.md |
| SealBot eval protocol | adaptive **time-limited** difficulty ladder (0.05s→2.0s/move, promote at WR≥0.7, patience=2 consecutive evals) | current (post-§D-LADDER fix) = **fixed-depth-5** minimax probe via `deploy_strength_eval.py`; an older in-loop path still uses time-limited (0.5s strong / 0.1s fast) | not directly comparable (see verdict below) | MEASURED (config) both sides | `configs/gine-mini/*.toml` `[eval.sealbot]`; `docs/07_PHASE4_SPRINT_LOG.md` §2420, `configs/eval.yaml` |
| Committed SealBot-relative strength results | **zero** — no checkpoint, no run log, no recorded WR or difficulty-level-reached anywhere in the repo | multiple committed eval handoffs with measured trajectories (e.g. one cited live read: 30k step=0.23, 60k=0.24, 120k=0.27 SealBot WR, Theil-Sen +0.044/100k) | n/a | ours MEASURED; theirs **wholly absent** | `docs/07_PHASE4_SPRINT_LOG.md` §2343 (example); `docs/handoffs/d_*` |

## Where the wall-clock goes (localization)

Four independent, stacking multipliers explain why HeXO's wall-clock-per-useful-training-step
is structurally higher than hexo-strix's, even before any hardware difference:

1. **Net size — ~19× more parameters** (4.25M vs 0.222M), confirmed by direct
   instantiation of both models this session. NN forward+backward dominates
   both projects' per-position cost; this is the single largest identified
   multiplier.

2. **Curriculum defers the expensive regime; we don't have one.** hexo-strix's
   early curriculum stages (S1: 4-in-a-row, radius 2, `max_moves`=20-80) run
   on graphs with ~180 edges and games that terminate in tens of plies — the
   `edge_budget` mechanism exists specifically because a fixed-row-count batch
   would "grind to a crawl" once boards grow (per their own README). HeXO's
   bootstrap anchor already starts at full v6 geometry (19×19 window,
   6-in-a-row, radius ~8) — there is no cheap early stage; every self-play
   game from step 0 pays close to full late-curriculum cost. This is a
   second multiplier that is *impossible to quantify precisely without their
   run logs* (which don't exist) but is qualitatively large — most of a
   curriculum run's wall-clock budget by construction sits in the cheap early
   stages before the graphs grow.

3. **Train/deploy search mismatch inflates our self-play cost specifically.**
   HeXO's self-play data generation runs PUCT+quiescence at ~400 sims/move
   (or a 100/600 quick/full blend), roughly 3× the sim count of hexo-strix's
   production self-play (Gumbel SH, 128 sims/move) — and hexo-strix's
   self-play search is the *same* lightweight algorithm it deploys with, so
   there's no separate "expensive data-gen search" tax at all. Combined with
   point 1, the per-position self-play simulation cost differential is on
   the order of `19× (net) × 3× (sims) ≈ 60×` before curriculum-stage size is
   even considered.

4. **Auxiliary-head density.** Our training step carries 6 extra loss
   heads/targets (ownership, threat, 6-plane chain-length, value-uncertainty,
   ply-index, opp-reply) that hexo-strix's bare 2-head net does not compute at
   all. This is a smaller multiplier relative to the trunk cost but is a
   deliberate, KataGo-style density-of-supervision design choice on our side
   that has no analog on theirs — pure wall-clock cost with no offsetting
   curriculum benefit.

Net effect: the pre-registered expectation ("their games/hr per FLOP much
higher; their sims/move lower") is **directionally CONFIRMED** on every axis
we can measure from static config (smaller net, same-or-fewer deploy sims,
fewer/no curriculum-deferred cost) — but cannot be confirmed as a measured
*games/hr* ratio because hexo-strix has never recorded that number.

## Caveats

- **Zero hexo-strix runtime artifacts exist.** `docs/`, `runs/`, `data/`,
  `*.pt`, `benchmark.txt`, `trajectories.json` are all gitignored. Every
  strix-side number in this report is either (a) read directly from static
  TOML config (declared *intent*, not measured *behavior*), or (b) a
  code-committed micro-benchmark (`hexo-rs/README.md` engine table), or (c)
  absent. No self-play throughput, no SealBot WR, no wall-clock-to-anything
  number exists anywhere in the cloned repo.
- **Param-count cross-check.** The task brief's pre-briefed "~0.23M params"
  for hexo-strix matches this session's direct instantiation of
  `4l-128p32v.toml` almost exactly (283,970 measured vs ~230,000 briefed;
  corrected 2026-07-14 — originally mis-recorded as 222,146),
  corroborating that config as the right reference point. The brief's
  "2.9M-param net" for HeXO does **not** match this session's direct
  instantiation of the current `configs/model.yaml` architecture (4,254,283
  measured, including 6 auxiliary heads not present in an older/smaller
  count). The 2.9M figure likely predates some of the auxiliary heads
  (chain/ply-index/value-var/ownership/threat); not reconciled further —
  would need `git blame` on `hexo_rl/model/network.py` / `configs/model.yaml`,
  out of this audit's scope.
- **FLOP/hardware normalization is not computable.** hexo-strix's exact APU is
  never named in the repo — only inferred from the "hexo-strix" project name
  and generic "ROCm APU, unified memory, single iGPU" comments (most
  consistent with an AMD Ryzen AI Max "Strix Halo" part, but unconfirmed).
  Even with a confirmed chip, GNN forward cost scales with **edge count**
  (variable across curriculum stages and even within a stage by board fill),
  while our CNN forward cost is **constant-shape** per position — the two
  architectures do not share a stable "FLOPs per position" unit at any single
  point in training, so a single per-FLOP ratio would misrepresent one side
  or the other depending on which curriculum stage it's evaluated at.
- **HeXO's cited games/hr (5,522/hr) is the PUCT training-data-generation
  regime** (400 sims), not the Gumbel m16/n150 deploy regime — presented as
  the closest same-unit, actually-measured self-play figure available on our
  side; there is no equivalent number, in any regime, on the strix side.
- **Repo git history (2 hours, 16 commits) does not reflect real development
  time** — internal comments reference dated findings 4-8 weeks older than
  the repo's first commit, meaning this is a fresh export/mirror of a
  longer-running private project. Do not read "young repo" as "young
  project."

## SealBot-anchor comparability verdict: **CANNOT compare days-to-parity**

1. hexo-strix has **zero committed SealBot (or any) eval results** — no
   checkpoint, no run log, no recorded wall-clock timestamp anywhere in the
   repository. There is nothing to anchor a "days-to-X%-vs-SealBot" claim to.
2. Even if such a number existed, the two projects' SealBot protocols differ
   in kind: hexo-strix uses an **adaptive time-limited** ladder (0.05–2.0s per
   move, promotion-gated on WR≥0.7); HeXO's current protocol (post-§D-LADDER)
   is a **fixed-depth-5** minimax probe specifically because time-limited
   SealBot depth is machine-speed-dependent — HeXO's own forensic measured
   "SealBot@0.5s reached median depth 4 / mean 4.6" on vast hardware, i.e. the
   *same* time budget does not yield the *same* search depth across machines.
   Matching wall-clock budget across an APU and a discrete RTX 5080 would
   *not* yield matched SealBot strength; only a fixed-depth match would, and
   hexo-strix's default is never fixed-depth.
3. Both projects nominally target the same upstream `Ramora0/SealBot`, which
   is the one thing that keeps a *future* comparison possible — **IF**
   hexo-strix ever commits a run log carrying (checkpoint step, wall-clock,
   SealBot config, WR or ladder-level-reached).

Given (1)-(3), this report is a **configured-recipe economics diff**
(what each project's config says it *intends* to do), not a realized-training-
outcome diff (what either project's model actually *achieved*, on any shared
clock). That is the honest ceiling of what is auditable from the current
artifacts.

## REVIEW

**Verdict: PASS-WITH-CORRECTIONS**

Independently re-verified against the clone, the current repo, and a fresh
model instantiation. Spot-checked ~9 rows/claims across the diff table and
caveats; all MEASURED numbers checked came back **exact-digit correct**:
param counts (both sides), `edge_budget`/`batch_size`, `buffer_capacity`
(250,000 and 125,000), engine micro-bench row (`apply_move` 5.1µs / `clone`
490ns/771ns / `legal_moves` 21-35µs), `games/hr` 5,522 and worker throughput
91,871/33,565 pos_gen/hr (incl. `n_workers=22`, 200-sims/move workload
detail). `.gitignore` claim (docs/runs/data/*.pt/*.sqlite/benchmark.txt/
trajectories.json all ignored) confirmed byte-for-byte. Zero committed eval
artifacts in the clone confirmed (no `runs/`, `data/`, `docs/` present on
disk despite being tracked-ignored; no `.pt`/`.sqlite`/`trajectories.json`/
`benchmark.txt` in `git ls-files`). FLOP/hardware-normalization caveat is
honest (explains why not computable rather than skipping it) — satisfies
the mandate.

**Corrections:**

1. **Commit count off by one.** Doc states "16 commits total"; `git log
   --oneline | wc -l` on the audited commit (`1b8ae4d`) returns **17**.
   Trivial, doesn't affect any downstream conclusion (the "fresh single-day
   export" finding itself is sound — see below).

2. **The "17:13–18:56, under 2 hours" repo-age claim is correct only under
   committer-date, not author-date.** Verified: `git log --format='%aI'`
   (author date) on the oldest commit gives `16:45:33+01:00`, sorted
   author-date range is `15:45:33 UTC`–`17:47:45 UTC` (~2h02m, technically
   *over* 2 hours). `git log --format='%cI'` (committer date) gives exactly
   `17:13:00+01:00` (oldest) to `18:56:21+01:00` (newest) — matching the
   doc's claim precisely. The doc doesn't disclose which timestamp field it
   used; since dependabot-merge commits carry earlier author dates than the
   "initial" commit (rebased/imported history), committer date is actually
   the more defensible choice here — but state the methodology explicitly
   next time, since author vs. committer date gave visibly different
   "spans."

3. **The param-count reconciliation hypothesis is likely wrong, and this
   matters because the review explicitly flagged the number as load-bearing
   for the 19× ratio.** The doc hedges "the 2.9M figure likely predates some
   of the auxiliary heads" but does not check the arithmetic. Verified by
   instantiating `HexTacToeNet` at the current `configs/model.yaml` dims and
   summing params by module prefix: the 6 aux heads (opp_reply, ownership,
   threat, chain, ply_index, value_var) together account for only
   **~271K params** of the 4,254,283 total — nowhere near enough to explain
   the ~1.35M gap to a hypothetical 2.9M baseline (removing them by hand
   leaves ~3.98M, not 2.9M). Sweeping `(res_blocks, filters)` combinations
   on the same class shows **`res_blocks=10, filters=112` → 2,924,883
   params** — a near-exact match to "2.9M." This strongly suggests the 2.9M
   figure (wherever it came from) reflects a **smaller trunk config**
   (fewer res blocks and/or filters), not a missing-aux-heads difference.
   Doesn't change the report's own measured number (4,254,283 for the
   *current* `configs/model.yaml`, confirmed exact-digit via independent
   instantiation) or invalidate the param ratio (19.1× as originally
   recorded; 15.0× after the 2026-07-14 count correction) *as a statement
   about the current recipe* — but the stated reconciliation guess should be
   corrected or dropped, not left as the working hypothesis.

**Unverifiable / gap:**

4. **The dispatcher-brief claim "operator's 2.9M net beats their bot at
   512 sims" is never named or caveated anywhere in this doc**, despite the
   review mandate calling it out by name and this doc being the natural
   place to apply the "small-sample caveat to ALL head-to-head claims"
   instruction. The doc's general conclusion ("zero committed SealBot
   results either side," "CANNOT compare days-to-parity") functionally
   covers it, but it's never explicitly named, and no `n` for that
   specific claim is surfaced or flagged as undocumented anywhere in the
   audit trail (repo or HeXO docs). Should have been called out by name.

   **CORRECTION 2026-07-03:** beats-claim VOID, not merely uncaveated. No
   head-to-head vs hexo-strix (or its bot) exists in this repo or ours, in
   either direction — searched, none found. The dispatcher-brief claim
   itself was unfounded, not just under-caveated. Strength ordering vs
   hexo-strix = UNKNOWN in both directions; no head-to-head exists.

## RED-TEAM

**Target: conclusion A — "iteration-efficiency gap REAL."**

**Steel-manning the opposite:** the doc's own "Net effect" paragraph already
retreats to "directionally CONFIRMED on every axis we can measure from
static config... but cannot be confirmed as a measured games/hr ratio." Push
harder: is there *any* remaining evidence hexo-strix learns faster **per
wall-clock**, or has the "days-from-zero" premise fully evaporated, leaving
"no verified gap in either direction, only a config-intent diff"?

**Re-verified this session (fresh clone reads, not re-trusting S1's own
summary):**

- Re-confirmed zero `.pt`/`.sqlite`/`runs/`/`data/` anywhere in the clone
  (`find` for `*.pt`/`*.sqlite`/`*.ckpt` → 0 hits). `tb_writer.py` exists
  (they do log to local TensorBoard), but no event file is committed and
  `runs/` is gitignored — so even their *own* operator has no externally
  shareable number without a manual export, same conclusion S1 reached.
- **The claimed "production-strongest" config's supporting evidence is a
  code comment, not a run log, and the surrounding record is MIXED, not
  uniformly positive.** `configs/gine-mini/4l-128p32v.toml:23-31`'s
  changelog reads: `gine-champion-mini (no JK): reached S3, beats sealbot
  5s` / `jk-long: collapsed at S3` / `jk: weaker than jk-long` / graft
  attempts "all failed." Two more scattered comments corroborate a
  failure-prone history: `4l-scratch.toml:12` ("failed in different ways")
  and `4l-layerscale.toml:43` ("first attempt regressed, loss +20%, SPRT
  0.29"). This is the opposite of a clean "smaller net learns faster"
  story — it's a normal, partially-failed architecture-search log, same
  genre as HeXO's own falsified-hypotheses register. S1's "zero... eval
  results anywhere in the repo" is technically correct (no WR%, no
  checkpoint, no date attached to "beats sealbot 5s") but should be phrased
  as **zero *quantified*** results — an unquantified, undated, unlogged
  claim of "beats sealbot" does exist in-repo, and S1's blanket "wholly
  absent" phrasing slightly overstates the void. Doesn't change the
  verdict (an undated, non-reproducible comment still can't anchor a
  days-to-parity or games/hr claim) but the doc should say "zero quantified"
  not "zero."
- No externally-hosted run record either (no wandb cloud project, no HF
  dataset/model card, no CI artifact) — checked `pyproject.toml`, `cli.py`,
  `curriculum.py` `wandb`-adjacent hits are all local `tb_writer`
  integration, not a shareable cloud link. There is no cheap side-channel
  that would recover a real number without asking the author directly.

**Verdict: WEAKENED, not killed.** The measured static-config multipliers
(19× params, 3× self-play sims, curriculum-deferred cost, aux-head density)
are real and do support a **plausible** iteration-efficiency gap — but
"directionally CONFIRMED" is stronger language than the evidence supports
given the record now includes at least three explicit failure/regression
data points on hexo-strix's own side, and the one success citation is an
unquantified inline comment, not a result. Correct framing: **"a
config-implied efficiency gap, unconfirmed and non-falsifiable from
either side's committed artifacts — not yet a landscape read in either
direction."**

**Cheapest re-establishing/killing observation:** don't try to reconstruct
their training run from config alone (unfalsifiable, as S1 already shows).
Instead, ask the author (near-zero cost to them, no build required on our
side) to run their own committed one-liner —
`hexo-a0 eval-sealbot --config <cfg>.toml --checkpoint <ckpt>.pt --games 100`
— against whichever checkpoint backs the "beats sealbot 5s" comment, and
share the WR% + step count + wall-clock. That single number (which their
own CLI already produces, no new tooling needed) either substantiates or
kills the entire "their recipe learns faster" premise without HeXO
building or running anything.
