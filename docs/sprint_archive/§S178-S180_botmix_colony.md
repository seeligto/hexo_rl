# §S178-S180_botmix_colony

_Relocated from `docs/07_PHASE4_SPRINT_LOG.md` (D-DOCS-DEBLOAT split, 2026-06-23). Scope: §S178/§S178a bot-mix launch + §S179 mechanism close + §S180a/§S180b colony-lever closes. Verbatim; falsified-register rows also consolidated into the sprint-log index register section._

## §S178 — v6 sustained bot-mix recipe launch (2026-05-18)

*DISCRIMINATOR: This is §S178 = Sustained Training Sprint 178. NOT to be confused with §178 (line 1697) which is the Rust engine refactor cycle 1 close. The two numbering schemes coexist permanently post-cycle-3. Always cite the branch (`phase4.5/s178_botmix`) or the prefix (§S178) for disambiguation.*

**Branch:** `phase4.5/s178_botmix` (off master `ddfa42e`); **anchor:** `checkpoints/bootstrap_model_v6.pt` (clean v6 base bootstrap; §175 anchor preserved for direct A/B vs §175 + §177).
**Design:** `docs/designs/S178_design.md` (commit `b26999b`).
**Investigation:** `reports/s178_pre_design_investigation.md` (gitignored, vast-only).
**Implementation:** 9 commits `b26999b..22597fc` on `phase4.5/s178_botmix` (design + T2 Rust split + INV26 + T3 Python wire + T1 bot corpus generator + T4/T5/T7 training-path + T6 yaml).

### Two mechanism levers vs §175/§177 colony-attractor capture

1. **SealBot-vs-anchor bot corpus pool** at `bot_batch_share=0.15`. Top-level batch slot (NOT inside selfplay decay, NOT subject to LRU eviction). Pre-generated via `make corpus.bot ANCHOR=checkpoints/bootstrap_model_v6.pt N_GAMES=700 OUT=data/bot_corpus_s178_sealbot_vs_v6.npz`. KrakenBot DROPPED (operator call, supported by Wave C BT data); bot-vs-bot games SKIPPED (no anchor-mistake signal). Bootstrap corpus UNCHANGED (human-only per §148).
2. **`ply_cap_value` split from `draw_reward`.** Rust `finalize_game` outcome branch now distinguishes terminal_reason==2 (ply-cap, → `ply_cap_value`) from winner=None,ply<max (→ `draw_reward`). Operator pre-commit `draw_value: -0.5 → -0.1`. Operator override on `ply_cap_value`: design called for `-0.8` (BCE 0.10, near-loss); operator dialed back to `-0.5` (BCE 0.25, soft-penalty) pending §178 outcome — may revisit.

### Pre-registered V178-1..8 verdicts (design §7)

| ID | Hypothesis | PASS criterion | FAIL criterion | NULL criterion |
|---|---|---|---|---|
| V178-1 | SealBot WR @ step 10K beats §177 step-10K (2%) | ≥5%, n=100 | ≤2% | 2–5% inconclusive |
| V178-2 | SealBot WR @ step 30K > 0% | ≥1 win / 100 | 0/100 | — |
| V178-3 | colony_fraction in bootstrap_anchor wins @ step 30K below §177 (~64%) | ≤50% | ≥70% | 50–70% |
| V178-4 | Non-monotonically-declining SealBot WR (no strict-decline trajectory) | observed | strict monotone decline | — |
| V178-5 | G4 value_fc2_weight_abs_max in [0.154, 0.462] by step 20K | in-band | below-band | — |
| V178-6 | Bot-pool policy-loss > 0.5 nat through step 30K (corpus dominance does NOT compete it down) | observed | bot ≈ corpus loss | — |
| V178-7 | draw_rate + ply_cap_rate both stable/down | both | either ≥5pp rise | — |
| V178-8 | v_pred(ply≥140) ∈ [-0.9, -0.7] by step 15K (ply-cap-value penalty learned) | in-band | drifts toward 0 | — |

### Risk register pointer

`docs/designs/S178_design.md` §8 — 12 rows, operator-confirmed pre-launch. Hard-abort thresholds preserved (§157/L9): grad_norm 10.0, stride5_p90 60, row_max_p90 50, colony_ext_frac_max 0.40. NEW soft-abort: `pct_at_cap_warn 0.30` (≥5pp rise without WR recovery).

### Operator overrides this sprint

- `ply_cap_value: -0.8 → -0.5` (variant yaml + training.yaml default). Design value retained in `docs/designs/S178_design.md` body for traceability; revisit if §178 outcome insufficient.
- §S178 discriminator on sprint log heading: sprint log already uses §178/§179/§180 for Rust engine refactor cycles 1/2/3 (lines 1697/1764/1921). Branch + design + verdict IDs all keep "s178"/"S178" branding.

### Pre-launch hygiene (operator, vast 5080)

1. `make bench` on vast 5080 to confirm no regression vs cycle-3-close baseline at `audit/rust-engine/cycle_3/close/04_baseline_next_cycle.txt`. Local laptop bench-gate SKIPPED (no AC; design §11). Touch points: T2 `finalize_game` + T4 `assemble_mixed_batch` in-place copyto.
2. `tmux kill-session -t s177` (if still alive on vast).
3. `data/bootstrap_corpus_v6.npz` present on vast (regen or scp; SA-E vast-only finding).
4. `rm checkpoints/replay_buffer.bin` (clean start; §177 buffer discarded).
5. `make corpus.bot ANCHOR=checkpoints/bootstrap_model_v6.pt N_GAMES=700 OUT=data/bot_corpus_s178_sealbot_vs_v6.npz` to generate bot corpus.
6. Launch: `python scripts/train.py --checkpoint checkpoints/bootstrap_model_v6.pt --variant v6_botmix_s178 --iterations 100000`.

### Forward pointers

- §179 candidate: flip `bot_corpus_refresh.enabled: true` if §178 colony_frac trajectory shows aging (currently hook present + disabled, warning-only log when triggered).
- INV26 pins ply_cap_value-distinct outcome path; INV19 byte-equivalence extends 38→39 atomically with T2.

### Pending post-launch follow-ups (for future sessions)

Pre-launch branch extension (T11/T12/T13, 2026-05-18; on `phase4.5/s178_botmix`):
- **F-fix-1** threat-target colony fix landed at **T11** (commit `1aa0c8f`): `find_winning_line` now scans all stones via fallback so the threat-head target is non-empty when `winner` is set via the `player_wins` all-stones fallback (HTT 2-moves-per-turn off-line second-move case). INV27 Rust + Python parametrized GREEN.
- **ply_cap_value revised** to `0.0` at **T12** (commit `1b47eb1`): literature-canonical per AlphaZero/KataGo (was `-0.5` = no-op vs draw_value). Variant only; training.yaml default kept at `-0.5` for back-compat.
- **T13** (this commit): sprint-log cross-pointers + this follow-up block.

Open §S179 candidates if §S178 colony reduction is modest:
- (a) `completed_q_values: false` A/B at step 30K-50K (visit-count CE policy target).
- (b) Policy Surprise Weighting / KL-weighted buffer writes (Q9; KataGo confirmed major efficiency win).
- (c) Soft policy target T=4 with 8x weight (KataGo b18c384 innovation).
- (d) `bot_batch_share` → 0.20.
- (e) `game_length_weights` neutralization (H4).

Open hygiene items (NOT in §S178 scope; defer to follow-up commits on master):
- **H1** — `bootstrap_model.pt` cleanup (rename/delete + repoint `Makefile` / `anchor.py` / `opponent_runners`; provenance per `reports/bootstrap_model_pt_provenance.md`).
- **H2** — `data/bootstrap_corpus_v6.npz` on vast (regen on vast if absent; can run before bench gate).
- **H3** — §177 vast tmux liveness (SSH check; kill if alive before bot-corpus generation).
- **H4** — `game_length_weights` colony-bias (§S179 candidate, not pre-launch).
- **H5** — `Makefile` BOOTSTRAP default → explicit error (separate hygiene commit).

Source B live cross-bot subprocess infra design deferred to §179+.
Length-discounted value target DROPPED from candidate pool per operator (terminal-reward purity principle).

---

## §S178a — Tier-1 hygiene wave (mid-§S178) — 2026-05-18

Five-item parallel hygiene wave on branch `phase4.5/tier1_hygiene` (off master
`96337c4`); §S178 sustained run on vast.ai untouched throughout (tmux `sS178`
LIVE; active anchor `bootstrap_model_v6.pt` + active bot corpus
`bot_corpus_s178_sealbot_vs_v6.npz` + active variant `v6_botmix_s178.yaml` all
untouched in every diff). Wave-driver:
`reports/tier1_hygiene_wave.md` (this aggregation report).

**A — quarantine fresh-init `bootstrap_model.pt`** (commit `c1173a8`).
Closes §S178 open hygiene **H1**. The unknown-provenance v6w25-architecture
file at `checkpoints/bootstrap_model.pt` was the silent `Makefile:11` default
when `BOOTSTRAP=` was unset; per `reports/bootstrap_model_pt_provenance.md`
§3+§5 it carries Kaiming-uniform fresh-init weights with 0/143 tensor match
vs every sibling — not a trained anchor. Repointed canonical defaults
(`Makefile:11`+`:47`, `hexo_rl/training/anchor.py`
`_BOOTSTRAP_ANCHOR_CANDIDATES`, `hexo_rl/eval/opponent_runners.py:217`
`bootstrap_anchor`) to `bootstrap_model_v6.pt`; dropped the row from
`scripts/migrations/2026_05_12_checkpoint_manifest.json`. Forensic copy
preserved at `checkpoints/archive_quarantine/bootstrap_model_random_init_v6w25.pt`
(gitignored; SHA `d00b8604…1586253`). 4 residual references outside the
contracted touch list deferred to follow-up (Q-§S178a F-A1).

**B — `tests/test_scraper.py` restore** (NO-OP, no commit).
The tracked-path test does not exist anywhere in reachable git history
(neither `7aea774` nor `913c5a0` touched a `tests/test_scraper.py`). A
gitignored local copy DOES exist on this worktree, imports
`hexo_rl.bootstrap.scraper` cleanly, and runs 19/19 PASS in venv — there
is no broken `elo_band_key` import to fix locally. The wave-driver
premise was sourced from a vast-side bench-gate report describing a
different stale gitignored file (Q-§S178a F-B1 follow-up).

**C — `scripts/verify_anchor.py` anchor verification CLI** (commit
`2740766`; 254 LOC new file). Reports `sha256` (16 hex), `format`,
`key_count`, `param_count`, `head_shapes`, `fresh_init_signature` (bool),
`verdict` (TRAINED | FRESH_INIT_SUSPECT | UNKNOWN). Heuristic per
provenance report §3: `value_fc2.weight.abs().max() / sqrt(1/fan_in)`
within `[0.8, 1.25]` → suspect. Sidecar JSON at `<ckpt>.verify.json`.
Exit codes 0 / 1 / 2. Verified end-to-end against the active anchor
(exit 0 TRAINED, ratio 2.013) and the quarantined fresh-init file
(exit 1 SUSPECT, ratio 0.999).

**D — CLAUDE.md split** (NO-OP, no commit). `wc -l CLAUDE.md` = 105 ≤
200 target; all 8 `docs/rules/*.md` topic files exist; CLAUDE.md is
index-shaped (Prime Directive + threat-probe gate + encoding registry +
rule files index + deep-dive index + MCP tools) with no fat content to
extract.

**E — variant config hygiene audit + cleanup** (commit `ba4f1e7`).
Audited 7 non-active variants (`v6_botmix_s178.yaml` SKIPPED, active
§S178). Cleaned `v6_sustained.yaml` (§175 closed-by-interrupt) and
`v7mw_sustained.yaml` (§176a experimental v7mw) — dropped 11/12
base-equal scalars each. Audit report at
`reports/variant_hygiene_audit.md` (306 LOC) covers per-variant
noise/override/extension tables. Deferred audit-only:
`m173_alpha_cold_smoke.yaml`, `smoke_radius_curriculum.yaml`,
`_sweep_template.yaml`, `v6_sustained_s177.yaml` (§S178 contrast),
`vast.yaml` (operator-designated exemplar). All 22+ removed keys
spot-checked base-equal; `tests/test_variant_configs.py` 5/5 PASS;
`#[allow` count = 29 (cycle-3 close baseline preserved). Follow-ups
Q-§S178a F-E1, F-E2.

**Verification.** Cherry-pick order A → C → E on
`phase4.5/tier1_hygiene` (B + D no-op). 5 independent REVIEW subagents
(no IMPL context, fresh per-agent) each PASS. `make test` exit 0 on
branch HEAD: 1588 passed, 21 skipped, 1 xpassed, in 155.74s. No
hot-path touch; no Rust `#[allow]` regression; active-run paths
confirmed untouched in every diff. Operator FF-merges to master
post-inspection (no tag — hygiene wave, not cycle close).

**Follow-up entries:** `docs/06_OPEN_QUESTIONS.md` — Q-§S178a row
(LOW/MED priority items absorbed during normal §S178+/§S179 work).

---

## §S179 — §S178 mechanism CLOSE: bot-mix + ply_cap split insufficient

*DISCRIMINATOR: §S179 = Sustained Training Sprint 179 (this entry). NOT §179
(line 1921) = Rust engine refactor cycle 3 close. Cite the run-id or `§S179`
prefix to disambiguate.*

**Status:** FAILED. Colony attractor reproduced. Same dead-end as §175.

**Run identity:**
- Branch at launch: master `98296f9` (post-§S178a + §S179a/§S179b/§S179c +
  F-A1 merges; `98296f9` = cadence-fix `eval_interval 5K→10K`).
- Anchor: `checkpoints/bootstrap_model_v6.pt` (SHA
  `7ab77d2cb091e3a67a0900e8c312f11fd7f9e87c8ea31cdd27102b9298372103`).
- Bot corpus: `data/bot_corpus_s178_sealbot_vs_v6.npz` (700g, static).
- Variant: `configs/variants/v6_botmix_s178.yaml` (post-cadence-fix:
  `eval_interval` 10K, SealBot stride 1 n=100; `ply_cap_value` 0.0 per §S178
  T12; `draw_value` -0.1).
- Run-id: `243e321f76504c6d908ab2f64eef8100`; tmux `sS179` on vast (5080).
- Launched 2026-05-18 17:36 UTC; SIGINT-stopped 2026-05-20 07:04 UTC at
  step **62,740** (31,370 games, ~37.4 h elapsed). Clean exit — final
  checkpoint + buffer flushed, `session_end` written. Last eval @ step 60K
  (`eval_interval` 10K); steps 60K→62.7K trained without further eval.

**Eval trajectory** (SealBot/anchor n=100, greedy-bot n=20; colony =
colony-formation wins ÷ player wins):

| Step | wr_sealbot | wr_anchor | wr_best | colony@anchor | colony@best | colony@sealbot | Elo | Promoted |
|---|---|---|---|---|---|---|---|---|
| 10K | 8% | 59% | 55% | 49% | 62% | 13% | 410 | ❌ |
| 20K | 11% | 68% | 66% | 79% | 85% | 91% | 414 | ✅ peak |
| 30K | 12% | 64% | 49% | 70% | 80% | 83% | 302 | ❌ |
| 40K | 2% | 66% | 60% | 85% | 88% | 100% | 266 | ✅ |
| 50K | 2% | 70% | 63% | 83% | 89% | 100% | 224 | ✅ |
| 60K | 4% | 75% | 45% | 77% | 93% | 100% | 224 | ❌ |

**Diagnostic.**

SealBot WR peaked 12% @ step 30K, crashed to 2–4% by 40K–60K. Peak Elo
promotion @ step 20K (Elo 414). Anchor WR climbed monotonically 59→75 over
the same span. colony@sealbot pinned 100% the last 3 rounds (every SealBot
win is a colony-formation win). Canonical anchor↑/sealbot↓ divergence — the
§155 T2 / §175 colony-capture signature.

Threat probes did not trigger the C1–C3 kill criterion through step 60K
(run ran uninterrupted to 62.7K) — the threat circuit is intact; the failure
is policy-distribution-level, not threat-representation-level. L22 confirmed
again: the sampled policy diverts into colony patterns despite correct
threat representation.

§S178 mechanism (`bot_batch_share=0.15` SealBot-vs-v6 corpus +
`ply_cap_value=0.0` + `draw_value=-0.1` + cosine-OFF + F-fix-1 threat-target
colony fix) bought ~one extra promotion vs §175 trajectory (§175 peak 17% @
15.5K; §S179 peak 12% @ 30K) but did NOT escape the attractor. The
anti-corrective force decomposition in `docs/designs/S178_design.md §3.1`
predicted 0.82:1 (DIRECT-corrective vs recent-selfplay-colony) = BORDERLINE.
Borderline lost.

**Falsified.** §S178 design hypothesis H-S178-1: "`bot_batch_share=0.15`
with SealBot-vs-v6 corpus + `ply_cap_value=0.0` + cosine-OFF is a sufficient
anti-colony lever for stable training to step ≥50K."

### Falsified Hypotheses Register addition

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §S179 (H-S178-1) | `bot_batch_share=0.15` SealBot-vs-v6 corpus + `ply_cap_value=0.0` + cosine-OFF is a sufficient anti-colony lever for stable training to step ≥50K | §S179 eval trajectory (close 2026-05-20) | SealBot WR 8→11→12→2→2→4; anchor 59→75; colony@sealbot pinned 100% from step 40K. Mechanism buys ~1 extra promotion vs §175 but does not escape the colony attractor. Borderline 0.82:1 corrective-force decomposition (S178_design §3.1) lost. |

**Archive.** `archive/s179_recipe_fail/` on vast — 8 eval-aligned/peak/final
checkpoints (`ckpt_step{10,20-peak,30,40,50,60}k.pt`, `ckpt_final_step62740.pt`,
`best_model_final.pt`) + `eval_db.sqlite` + `metadata.json` + `training_tail.log`.
Replay buffer (2.9 GB, colony-saturated) + dense intermediate checkpoints
deleted post-archive-verify (low forensic value). `best_model.pt` reflects
the last promotion (step 50K) — the step-20K peak is preserved separately
as `ckpt_step20k_peak.pt`.

**Successor.** §S180a launching: §S179 recipe + single config flip
`completed_q_values: false` (pre-registered §S179-candidate (a) — visit-count
CE policy target). Tests CQV as a colony-attractor amplifier. Single
isolated variable.

### Process patterns / Mechanism Lessons

L-numbering: L1–L17 promoted in the register table above; L18–L33 are
§-local candidates (latest L33 = §180 Wave 9). §S179 adds L34/L35.

- **L34 (anchor↑/sealbot↓ divergence = canonical colony-capture signature
  — 3rd confirming instance; promotable).** The calibration rule
  "`wr_bootstrap_anchor` drop below 50–55% = regression" is
  necessary-but-not-sufficient: it assumes the anchor is a colony-resistant
  reference. The v6 anchor SHARES the colony weakness. Rising `wr_anchor`
  with falling `wr_sealbot` is the textbook colony-capture pattern — the
  model improves at exploiting v6's colony weakness while losing the ability
  to play threat hex. Always cross-check the anchor trajectory against a
  colony-resistant opponent (SealBot). Confirming instances: §155 T2, §175,
  §S179. Calibration memo
  (`feedback_alphazero_sustained_eval_calibration`) amended accordingly.

- **L35 (§S178 mechanism — bot-mix 0.15 + ply_cap split + cosine-OFF — is an
  insufficient anti-colony lever; 1 instance).** The borderline
  anti-corrective force decomposition (0.82:1 DIRECT vs colony reinforcement)
  predicted at design time was confirmed insufficient empirically. The
  mechanism buys ~one extra promotion vs the §175 baseline but does not
  escape the attractor. Future mechanism interventions must target a ≥2:1
  corrective-force ratio OR isolate the amplification mechanism
  (CQV / `game_length_weights` / value-head pretrain bias) for direct
  nullification. Cycle 4+ candidate for 2nd-instance confirmation.

---

## §S180a — CQV-flip A/B CLOSE: CQV not the colony lever

*DISCRIMINATOR: §S180a = Sustained Training Sprint 180a (this entry). Cite
the run-id or `§S180a` prefix to disambiguate from §180 (Rust engine refactor
cycle 3).*

**Status:** FAILED. Different signature than §S179 — not colony capture,
weaker learning signal. CQV-flip RULED OUT as colony lever via single-knob
A/B vs §S179.

**Run identity:**
- Branch at launch: master `6f08042`
- Anchor: `bootstrap_model_v6.pt` (SHA `7ab77d2c…372103`)
- Variant: `configs/variants/v6_botmix_s180a_cqv_off.yaml`
- Single-knob delta vs §S179: `completed_q_values: true → false`
- Run-id: `e68e79a53793421a886611c625f9c802`
- tmux: `sS180a` on vast (5080)
- Launched 2026-05-20 07:20 UTC; SIGINT-stopped 2026-05-20 20:52 UTC at
  step **22,624** (11,312 games, ~13.5 h elapsed). Clean exit — final
  checkpoint flushed, `session_end` written. Killed after V180a-2 FAIL
  @ step 20K.

**Eval comparison (single-knob A/B):**

| @step | metric | §S179 (CQV true) | §S180a (CQV false) | delta |
|---|---|---|---|---|
| 10K | wr_sealbot | 8% | 8% | 0 |
| 10K | wr_anchor | 59% | 58% | -1 |
| 20K | wr_sealbot | 11% | 7% | -4 |
| 20K | wr_anchor | 68% | 53% | -15 |
| 20K | wr_best | 66% | 48% | -18 |

**Diagnostic.**

§S179 = colony capture (anchor↑ + sealbot↑ then crash). §S180a = not
learning (anchor↓ + sealbot↓ + wr_best <50% at step 20K, weaker than own
step-10K ckpt). Visit-count CE policy target produces weaker gradient than
CQV without escaping the colony attractor. CQV is NOT the colony amplifier
— it gave better learning, but learning the same wrong thing.

Threat probes PASS throughout both runs — circuit health independent of
colony failure mode (L22 reconfirmed).

**Falsified.** Hypothesis H-S180a-1: "`completed_q_values: false` produces
more diverse policy target, escaping colony attractor." Add to Falsified
Hypotheses Register.

### Falsified Hypotheses Register addition

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §S180a (H-S180a-1) | `completed_q_values: false` produces a more diverse policy target, escaping the colony attractor | §S180a eval trajectory (close 2026-05-20) | Visit-count CE = uniformly weaker metrics at step 20K (wr_sealbot -4pp, wr_anchor -15pp, wr_best -18pp vs §S179). Not colony capture — slower learning of the same trapped state. CQV ruled out as colony lever. |

**Archive.** `archive/s180a_cqv_off_fail/` on vast — 3 eval-aligned/final
checkpoints (`ckpt_step{10,20}k.pt`, `ckpt_final_step22624.pt`) +
`best_model_final.pt` + `eval_rounds_s180a.json` (12 `evaluation_*` events
extracted from train jsonl; `results.db` has no `eval_rounds` table) +
`metadata.json` + `training_tail.jsonl`. Replay buffer (2.9 GB) + dense
intermediate checkpoints deleted post-archive-verify.

**Successor.** §S180b — 3-knob escalation targeting direct anti-colony force:
1. `completed_q_values: true` (restore — stronger gradient confirmed)
2. `bot_batch_share: 0.15 → 0.30` (double direct anti-colony signal;
   per `docs/designs/S178_design.md §3.1`, raises DIRECT:colony ratio from
   0.82:1 to ~1.64:1)
3. `game_length_weights` neutralize (uniform 1.0/1.0/1.0 — kill colony
   upweighting in selfplay slice; Q-§S179-residual confirmed lever)

Multi-knob delta justified: §S179 + §S180a establish 2 baseline arms;
§S180b tests combined direct-force escalation. If §S180b PASS, follow-up
ablation isolates. If §S180b FAIL, surface is dead — escalate to §S181
with code-level levers (PSW or refresh hook).

### Process patterns / Mechanism Lessons

§S180a adds L36/L37.

- **L36 (single-knob A/B discipline retired when suspect-set unranked).**
  §S180a single-knob A/B isolated CQV cleanly = ruled out. But 4 remaining
  candidate levers (`bot_batch_share`, `game_length_weights`, PSW, refresh
  hook) have no quantitative ranking from §S179/§S180a data. Single-knob
  discipline on 4 unranked suspects = 4 × ~30h = 120h. Pragmatic shift:
  combine cheapest 2-3 unfired levers when all target the same mechanism.
  §S180b applies this.

- **L37 (visit-count CE = weaker gradient than CQV in colony-rich regime).**
  Empirical finding worth documenting: with bot-mix corpus + ply_cap split +
  cosine-OFF, switching from CQV to visit-count CE produced uniformly weaker
  metrics at step 20K (wr_sealbot -4pp, wr_anchor -15pp, wr_best -18pp).
  Possible mechanism: in the colony regime, the MCTS visit distribution is
  diffuse (low value-head signal), so the visit-count CE target = high-
  entropy near-uniform. CQV reweighting concentrates the target on
  high-value children = stronger learning signal despite the colony bias.
  Future variants with visit-count CE should pair with a value-head signal
  restoration mechanism.

### perf/legal-moves-cache-cap — CANDIDATE branch (merge held)

Rust-perf wave (`investigation/rust-perf-2026-05-20/`): `legal_moves_set`
pre-reserve fix. Branch `perf/legal-moves-cache-cap` HEAD `f8ff7b8` (off
master `3146144`), tag `perf-legal-moves-cache-cap-candidate`. Single-file
+26 LOC in `engine/src/board/moves.rs` — O(1) bbox+ball-area capacity
reserve before the legal-set rebuild loop, kills the hashbrown
power-of-2 rehash cascade. Laptop bench gate (8845HS, `--profile
profiling`, criterion n=800, median runs 2+3): **2.4936 ms → 1.4595 ms =
+70.9% sims/s**, uniform across n=100/400/800. perf report confirms
mechanism — `reserve_rehash` 31.4% → 1.2%, out of top-5. Independent
review `07_legal_moves_review.md` = MERGE-READY, all 8 checks pass.

**Merge HELD** — §S178/§S180b live on vast; no master push, no FF-merge
until §S178 close + vast cross-host (`9900X + 5080`) re-bench (same
3-run discard-first protocol). If vast delta within ±5pp of laptop →
FF-merge + push; if diverges >5pp → investigate L3-cache sensitivity
(8845HS 16MB vs 9900X 64MB) before merge. Residual: `legal_moves_set`
still #1 self-time (41.8%) post-fix → next perf wave targets the rebuild
insert cost itself (TLS scratch / incremental legal-set; plan Option (c)).

**CLOSED → §S182** — vast cross-host re-bench PASS (+66.4%, within ±5pp of
laptop +70.9%); FF-merged to master 2026-05-22.

---

## §S180b — 3-knob escalation CLOSE: config-level surface exhausted

*DISCRIMINATOR: §S180b = Sustained Training Sprint 180b (this entry).*

**Status:** FAILED. V180b-4 HARD FAIL @ step 50K — wr_sealbot collapsed to
**0%** (CI [0.0, 3.7]). 4th colony reproduction (§175, §S179, §S180a, §S180b).
Config-level anti-colony surface area exhausted.

**Run identity:**
- Branch at launch: master `3146144`
- Anchor: `bootstrap_model_v6.pt` (SHA `7ab77d2c…372103`)
- Variant: `configs/variants/v6_botmix_s180b_3knob_escalation.yaml`
- 3-knob delta vs §S179: `completed_q_values: true` (restored) +
  `bot_batch_share: 0.15 → 0.30` + `game_length_weights` neutralized
  (`[1.0, 0.50, 0.15] → uniform [1.0, 1.0, 1.0]`)
- Run-id: `fd9ea56e320646e5aeae11aefbe296bb`
- tmux: `sS180b` on vast (5080)
- Launched 2026-05-20 21:04 UTC; SIGINT-stopped 2026-05-21 23:00 UTC at
  step **53,890** (~26 h elapsed). Clean exit. Killed after V180b-4 FAIL
  @ step 50K.

**Eval trajectory:**

| @step | wr_sealbot | CI95 | colony@sb | wr_anchor | colony_a | wr_best | elo |
|---|---|---|---|---|---|---|---|
| 10K | 11% | [6.3, 18.6] | 7/100 | 61% | 36/100 | 52% | 422 |
| 20K | 7% | [3.4, 13.7] | 3/100 | 56% | 35/100 | 50% | 342 |
| 30K | 12% | [7.0, 19.8] | 11/100 | 61% | 40/100 | 61% | 354 |
| 40K | 19% | [12.5, 27.8] | 12/100 | 68% | 43/100 | 57% | 330 |
| 50K | **0%** | [0.0, 3.7] | 0/100 | 65% | **59/100** | 62% | 237 |

Pre-registered verdicts: V180b-1 @10K PASS; V180b-2 @20K FAIL; V180b-3 @30K
FAIL; **V180b-4 @50K FAIL**.

**Diagnostic — colony capture, masked.**

The 3-knob escalation crushed every *visible* colony metric: self-play
`colony_extension_fraction` ~0.04% of games (near-extinct), `colony@sealbot`
0–12% throughout (never the §S179 91%). Yet the policy still collapsed:
wr_sealbot 11→7→12→19→0. The 40K 19% was a transient pre-crash peak, not
recovery. At 50K the L34 capture signature fired — anchor 65% (high) /
sealbot 0% (collapsed) / `colony_a` jumped 43→59 per 100. The colony lives
in *anchor games*, a channel none of the 3 knobs touch. The model overfit
to beating the colony-prone anchor and lost all generalization to SealBot.

Threat probes PASS throughout (C1–C3, contrast 4.2–5.4, top5 40–70%,
top10 65–75%) — circuit health independent of colony collapse, L22
reconfirmed a 4th time.

**Falsified.** Hypothesis H-S180b-1: "combined config-level escalation
(CQV + 2× bot_batch_share + neutral game_length_weights) supplies enough
direct anti-colony force to escape the attractor."

### Falsified Hypotheses Register addition

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §S180b (H-S180b-1) | 3-knob config escalation supplies enough direct anti-colony force to escape the attractor | §S180b eval trajectory (close 2026-05-21) | Every visible colony metric crushed (self-play colony ~0.04%, colony@sealbot 0–12%) yet wr_sealbot still collapsed 19%→0% @50K with L34 anchor↑/sealbot↓ divergence. Capture channel is config-invisible. |

**Archive.** `archive/s180b_3knob_fail/` on vast — 6 checkpoints
(`ckpt_step{10,20,30,40,50}k.pt` + `ckpt_step53500.pt`) +
`best_model_final.pt` + `eval_rounds_s180b.jsonl` (5 `evaluation_round_complete`
events) + `metadata.json` + `training_tail.jsonl`. Replay buffer (3.1 GB) +
14 dense intermediate checkpoints deleted post-archive-verify (`checkpoints/`
4.2 GB → 569 MB).

**Successor.** §S181 — code-level levers. Config-level surface (CQV,
bot_batch_share, game_length_weights, cosine, ply_cap split) is exhausted
across §S178/§S179/§S180a/§S180b with zero escape. Next intervention must
be code-level: prioritized-sample-weighting (PSW) on bot-corpus rows, OR a
bot-corpus refresh hook that regenerates SealBot-vs-current games mid-run.

### Process patterns / Mechanism Lessons

§S180b adds L38.

- **L38 (config-level anti-colony surface is exhausted; capture channel is
  config-invisible).** §S178/§S179/§S180a/§S180b swept the full config-level
  lever set — bot-mix share (0.15, 0.30), CQV on/off, ply_cap split, cosine
  on/off, game_length_weights (biased, neutral). Every arm reproduces the
  colony attractor. §S180b is the decisive instance: it drove every *visible*
  colony metric to near-zero (self-play colony 0.04%, colony@sealbot 0–12%)
  and still collapsed via L34 anchor↑/sealbot↓ divergence. Conclusion: the
  capture operates through a channel no YAML knob reaches — diagnosis-by-
  metric is exhausted. §S181+ must use code-level levers (PSW / corpus
  refresh hook) and instrument the anchor-game colony channel directly.

---

