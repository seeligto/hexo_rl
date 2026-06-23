# §S176-S177_sustained_recipe

_Relocated from `docs/07_PHASE4_SPRINT_LOG.md` (D-DOCS-DEBLOAT split, 2026-06-23). Scope: §176 Python/PhaseA + §177 sustained recipe-attractor + Supplementary tables. Verbatim; falsified-register rows also consolidated into the sprint-log index register section._

## §176 — Python codebase refactor cycle (2026-05-13 → 2026-05-14)

**Scope:** 80-proposal audit + 6-phase execute on `refactor/python-audit` branch.

**Phases:**
- Phase 0 — master plan fixup (`c4eaa53` CLAUDE.md scattered-keys clarification) + drift annotations (`838b5ed` open-questions log)
- Phase 1a/b — HEAD-blocker fixes (B2 `sweep_harness` restore, B1 `HexTacToeNet` encoding whitelist→registry)
- Phase 2 — invariant pre-flight tests (12 INV pins under `tests/refactor_invariants/`) + 4 low-risk additions
- Phase 3a — W1 deletions (-274 LOC net)
- Phase 3b — W1 extracts + SSR fixes + small renames (24 commits)
- Phase 4 — W2 splits + extracts (25 commits; 3 bench-gated items)
- Phase 5 — W3 cross-bucket consolidation (15 commits; 4 bench-gated items)

**Outcomes:**
- 86 commits landed `c4eaa53..HEAD` (post-rebase HEAD `7233d5d`); 171 files changed, +12102/-7446 (net +4656; dominated by P39 6-module pretrain split, P70 train.py orchestrator decomposition, and INV/fixture test scaffolding)
- 75 of 80 proposals landed (5 NEEDS-WORK resolved at Phase 0; 3 deferred to W3 sub-items)
- Cross-bucket SSR debt cleared: `utils/encoding.py` + `utils/constants.py` v6/v8 entries retired; ~37 callers migrated to §172 registry; `hexo_rl/bootstrap/bots/` retired (P78a–d) — all three SSR grep targets return 0 post-merge
- 12 behavior invariants pinned as regression tests under `tests/refactor_invariants/` (all green post-merge)
- HEAD-blocker B1 (v7-family pretrain crash via whitelist) + B2 (sweep_harness broken imports after §163 deletion) fixed
- Test count: 1518 → 1574 (+56 from new fixtures + INV pins). Single pre-existing failure `test_no_stale_plane_refs` baselined and unchanged
- Bench: all hot-path edits verified within ±5% on 10-metric gate (P3, P4, P8, P22, P24); cold-path skips documented per `docs/refactor-template.md`
- `make test.py` post-merge: 1574 passed / 1 failed (pre-existing) / 17 skipped / 4 deselected / 1 xpassed
- §175 selfplay state unaffected throughout — refactor branch strictly isolated; rebase onto `phase4.5/m176a_v7mw` HEAD `838b5ed` was conflict-free (no file overlap with in-flight commits)

**Deferred to future micro-refactor cycle** (tracked as `Q-§176-residual` in `06_OPEN_QUESTIONS.md`):
- P24b/c: `HexTacToeNet.__init__` (262 LOC), `forward` (162 LOC), `aggregated_forward_K` (113 LOC) further decomposition — partial landed in Phase 5
- P70: `scripts.train::seed_everything` circular-import shim lifted inside orchestrator helper — clean candidate

Forensics: `reports/refactor_audit/00_MASTER_PLAN.md`, `reports/refactor_audit/p6_phase4.5_inflight_commits.txt`, Phase 5 reviewer verdict (in conversation history of `phase4.5/m176a_v7mw` § auditor session).

---

## §176 — Phase A — KrakenBot eval ladder validation + colony POC (2026-05-14)

Branch `phase4.5/s176_phase_a_validation`. Five-wave empirical investigation (A1–A4 parallel + B + C + D + E fresh-context review) closing **Q14 partial**, opening §176 Phase B implementation scope. §175 v6 sustained continues on vast; aborts at next eval boundary post-merge.

**Waves:**
- **A1** (`reports/s176_a1_kraken_smoke.md`) — submodule pinned `d9c5bfb`; verdict `INTEGRABLE_NOW` for MinimaxBot+RandomBot, `NEEDS_WEIGHTS_DOWNLOAD` for both MCTSBot variants (`vendor/bots/krakenbot/.gitignore:8`, no public mirror). MinimaxBot latency 222–232 ms @ time_limit∈{0.1,…,2.0}.
- **A2** (`reports/s176_a2_eval_arch.md`) — verdict `CACHING_CLEAN`. Evaluator loop stone-by-stone (evaluator.py:201-210); `_pending_move` cache already proven on SealBot+KrakenBot. Minimal-diff plan ~150-180 LOC, 0 INV pins fire.
- **A3** (`reports/s176_a3_selfplay_forensics.md`) — operator's "one large diffuse cluster" claim **REFUTED** by 21,371 §175 game records (vast run `c7e74d2842404a82bdd9f62edf740ea2`). Single-cluster fraction monotone-down 18.1% (20K) → 6.3% (50K); attractor is multi-island fragmentation. Step-change at 40K (n_components 9.61→13.77, +43%). **POC metric = `n_components` raw BFS, Cohen's d −0.822** (largest among 8 candidates). In-trainer `colony_extension_fraction` flat zero — does NOT capture §175 attractor; justifies new POC.
- **A4** (`reports/s176_a4_falsified_scan.md`) — 9 falsified rows + 11 mechanism lessons (L1–L17 subset) + 4 regressions + 4 open Qs. 15-item do-not list each empirically sourced. Surfaced 7 master-prompt gaps (pool freeze, e30/e50 pretrain, radius+cosine pairing, v6 corpus blacklist, extended smoke boundary, frozen-spine rejection, realistic plumbing budget).
- **B** (`reports/s176_b_smoke.md`) — `KrakenBotRandomBot` + `KrakenBotMCTSBot` skeleton + `scripts/tournament_validate.py` + 3 wrapper tests (PASS). 15-game smoke verdict `PROCEED-TO-C`. Flagged: `bootstrap_model.pt` is v6w25; `our_v6_*` Wave C must pin `bootstrap_model_v6.pt`.
- **C** (`reports/s176_c_tourney/summary.md`) — 1050-game round-robin / 7 bots / 50 games/pair / laptop wall 85 min. Mid-tourney critical fix: KrakenBot MinimaxBot returns `[(0,0)]` sentinel when `_generate_turns` rejects all compounds (vendor lines 184/219/325/330); naive uniform-random fallback caused 0.42 sentinel/game in mid-game (ply ≥20). Smart neighbour-2 fallback (`_smart_legal_fallback` in `hexo_rl/bots/krakenbot_bot.py`) using KrakenBot's own `_D2_OFFSETS` pool prevented 438 uniform-random degradations across full run (fb_n2=438 / fb_rand=0).
- **D** (`reports/s176_d_plan.md`) — §176 Phase B implementation plan, 6 commits ≤10 cap, ~990 LOC delta, zero bench gates (all cold paths).
- **E** (`reports/s176_e_review.md`) — fresh-context audit verdict **CLEAR** across all 5 dimensions. 7-row risk register; 3 non-blocking strengthening notes for S6.

**BT ladder (anchor=sealbot, n=50/pair):**

| Bot | Elo | CI lo | CI hi | Wins/300 | Colony>0.3 rate |
|---|---:|---:|---:|---:|---:|
| sealbot | 0 | 0 | 0 | 274 (91.3%) | 35.0% [29.6, 40.9] |
| our_v6_mcts128 | −62 | −150 | +26 | 263 (87.7%) | 33.5% [28.0, 39.4] |
| kraken_minimax_strong | −494 | −612 | −376 | 182 (60.7%) | 7.1% [4.2, 11.8] |
| kraken_minimax_fast | −499 | −618 | −381 | 181 (60.3%) | 15.5% [10.9, 21.4] |
| kraken_random | −3072 | sat. | sat. | 7 (2.3%) | 85.7% [48.7, 97.4] (n=7) |
| randombot | −3091 | sat. | sat. | 0 | n/a |
| our_v6_argmax | −3102 | sat. | sat. | 0 | n/a |

**V1–V6 verdicts** (full text + numbers in `reports/s176_c_tourney/summary.md`):

| ID | Hypothesis | Verdict | Mechanism |
|---|---|---|---|
| V1 | strongest Kraken MCTS > SealBot | `N/A_MCTSBOT_BLOCKED`; modified-V1 FAIL | strongest tested Kraken (MinimaxBot @ 1.0s) is −494 Elo vs sealbot |
| V2 | Kraken MinimaxBot @ 1.0s > MCTSBot | `N/A_MCTSBOT_BLOCKED`; side-finding: MinimaxBot @ 0.1s ≈ @ 1.0s (Δ −5 Elo within CI) | iterative-deepening saturates at depth 4 + sentinel-fallback rate inflates draws |
| V3 | MinimaxBot colony ≤ MCTSBot colony − 10pp | `N/A_MCTSBOT_BLOCKED`; modified-V3 PASS | sealbot 35.0% vs kraken_minimax_strong 7.1%, gap 27.9pp, CIs non-overlapping by 18pp |
| V4 | SealBot colony > all Kraken | FAIL (kraken_random 85.7% > sealbot 35.0%) | caveat: kraken_random only 7 wins, CI wide |
| V5 | our_v6 strictly between Random and weakest Kraken | FAIL — our_v6_mcts128 is the 2nd-strongest bot in the tourney, not a weak baseline | our_v6 −62 BT vs kraken_random −3072 / randombot −3091 |
| V6 | cross-pair colony ≠ self-pair (opponent-coupled) | PASS — sealbot colony ranges 0.000 (vs argmax) → 0.412 (vs kraken_minimax_strong), 41pp spread | colony emergence is opponent-driven, not bot-intrinsic |

**D1–D5 master-prompt decision verdicts:**

| ID | Verdict | Source |
|---|---|---|
| D1 BotProtocol caching not `get_turn` | BACKED | A2 CACHING_CLEAN + Wave C 1050-game stability |
| D2 tourney includes all Kraken variants | PARTIAL — MCTSBot blocked, defer to §177+ until weights | A1 NEEDS_WEIGHTS_DOWNLOAD |
| D3 MinimaxBot colony < MCTSBot colony | PARTIAL — modified BACKED (vs SealBot 27.9pp); MCTSBot blocked | V3 PASS |
| D4 mix 75/15/10 | NEEDS_REVISION → adjusted bot pool: sealbot 50% / our_v6 30% / kraken_strong 15% / kraken_random 5% (Elo-derived per-source weights) | Wave C BT ladder |
| D5 Source A first, Source B target | BACKED | V6 opponent-coupled colony + A4 do-not #1 mandates subprocess for B |

**New mechanism lessons (L18+ candidates):**

- **L18** — *Pretrained-bootstrap-at-MCTS-128 can match an external minimax engine.* Wave C: our_v6 bootstrap (`bootstrap_model_v6.pt`, untrained on selfplay) is 25/50 H2H vs SealBot @ 0.5s, BT delta −62 (95% CI [−150, +26], WR Wilson95 [36.3%, 63.7%], n=50 — point-estimate-parity-consistent, not strongly-asserted parity). Implication: a "weak external opponent" framing for §175-style transfer-gap diagnoses is wrong at matched MCTS perception — the degradation is internal self-play head drift, not opponent-strength regression. **Refined by L22 below:** "head drift" is sampled-policy distribution flattening into colony attractor under T=0.5, not argmax-mode regression past bootstrap.
- **L19** — *KrakenBot MinimaxBot `_generate_turns` sentinel `[(0,0)]` is the upstream strength floor, not the time_limit.* time_limit ∈ {0.1, 1.0} produces BT delta −5 Elo. The 0.42 sentinel/game rate in mid-board positions (ply ≥20) caps strength independently of search budget. Any bot wrapper consuming a third-party bot MUST validate the returned cell against the live engine board, not trust the bot's output blindly.
- **L20** — *Argmax-only proxies (MCTS-1) are structurally below RandomBot.* `our_v6_argmax` (n_sims=1, temperature=0) went 0/300 — below randombot's 0/300-with-99-draws. The argmax-handicap memory (`feedback_v6_v8_same_training_data.md`) generalises beyond cross-encoding diagnostics: argmax-only mode is a degenerate strength sensor, only useful as a relative-direction signal, never as an absolute baseline.

**New Falsified Hypotheses Register row candidates:**

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §176 Phase A | A3 — §175 selfplay terminal states are "one large diffuse cluster" | A3 §c-§d | Single-cluster fraction monotone-down 18.1%→6.3% across 20K–50K cohorts; modal pattern is multi-island fragmentation |
| §176 Phase A | V2 — KrakenBot MinimaxBot @ 1.0s > @ 0.1s | Wave C V2 | BT delta −5 Elo, head-to-head 20-30 favouring 0.1s; iterative deepening saturates at depth 4 in our off-distribution game |
| §176 Phase A | V5 — our v6 bootstrap @ MCTS-128 strictly between RandomBot and weakest Kraken | Wave C V5 | bootstrap MCTS-128 is the 2nd-strongest bot in the tourney (BT −62 vs sealbot); ~3030 Elo above RandomBot |

**Forward pointer:** §176 Phase B implementation (S1–S6 per `reports/s176_d_plan.md` §2) opens on a fresh branch (TBD). Recommended 6 commits ≤10 cap. Mix-ratio bot-pool weights (sealbot 50 / our_v6 30 / kraken_strong 15 / kraken_random 5) per S4 design doc. Source B (live cross-bot) is design-only this sprint; subprocess isolation mandatory per A4 do-not #1.

Forensics: `reports/s176_{a1,a2,a3,a4}_*.md`, `reports/s176_b_smoke{.md,/}`, `reports/s176_c_tourney/{summary.md,verdicts.txt,ratings.csv,h2h_matrix.csv,colony_table.csv,per_game.jsonl}`, `reports/s176_d_plan.md`, `reports/s176_e_review.md`. Memory: `project_176_phase_a_close.md` (to be written).

---

## §176 — Phase A Gate 1 + Gate 2 + Gate 3 close-out (2026-05-15)

Branch `phase4.5/s176_phase_a_validation`. Three-gate operator-mandated cycle: (1) fresh-context independent review of Phase A artifacts (Wave E was the implementer-adjacent fresh-context audit; Gate 1 is a second pass per L13), (2) §175 interrupt + 70K vast tourney for tide-vs-recover empirical answer, (3) step-20K checkpoint promotion to weights-only bootstrap artifact.

### Gate 1 — operator review, verdict **STRENGTHEN_ONLY**

`reports/s176_gate1_operator_review.md`. 7 dimensions audited: D1–D5 cites (12 claims PASS), Wave C BT-ladder reproducibility (3 H2H pairs reproduced from `per_game.jsonl` — sealbot vs our_v6_mcts128 25/25, sealbot vs kraken_strong 49/1, sealbot vs randombot 50/0), `_smart_legal_fallback` correctness (`hexo_rl/bots/krakenbot_bot.py:34` imports `_D2_OFFSETS` from vendor — no divergent reimplementation), Falsified Register hygiene (16 rows walked; §17 GIL daemon row explicitly cited in plan S5 mandate), L18 sufficiency (PASS with strengthening note — n=50 H2H, BT CI [-150, +26] crosses zero; lesson body should disclose CI), risk register (7 rows ≥ 5 floor; 5 non-blocking strengthening notes captured for Phase B prep).

One process-note dimension: V1–V6 pre-registration is not committed as a separate artifact (`reports/s176_c_tourney/verdicts.txt` is post-hoc results dump). Honest FAIL declarations on V1, V4, V5 are weak evidence of integrity; operator vouches pre-registration. Gate 2 verdicts (V70K-1..5) committed BEFORE tourney in `reports/s176_gate2_verdicts.txt` — establishes audit-trail pattern for Phase B.

### Gate 2 — §175 interrupt + 70K tourney on vast 5080, verdict **MIXED**

`reports/s176_gate2_tourney/{summary.md,verdicts_v70k.md,ratings.csv,h2h_matrix.csv,per_game.jsonl}` + `reports/s175_forensics/`.

§175 interrupt was a no-op: session ended cleanly at step 70176 at 2026-05-14T20:56Z (SIGINT during sealbot eval game 37/100; `shutdown_save=True` triggered, buffer persisted, final checkpoint flushed). tmux session `s175` detached, kept for state preservation. Forensics archived locally: 21 eval-DB rows, checkpoint_log.json, training-step tail (5000 lines), shutdown events.

Tourney: 5 bots × 50 games/pair × 10 pairs = 500 games, 6551.1 s wall (well under 4-hr cap). Participants: `our_v6_latest` (step 70000, MCTS-128), `our_v6_step20k` (step 20000, MCTS-128), `our_v6_bootstrap` (bootstrap_model_v6.pt, MCTS-128), `sealbot` (think_time=0.5s), `kraken_minimax_strong` (time_limit=1.0s).

**Pre-registered verdicts (V70K-1..5):**

| ID | Hypothesis | Observed | Verdict |
|---|---|---|---|
| V70K-1 | 70K vs SealBot WR ≥ 17.4% | 25/50 = 50.0%, Wilson95 [36.6, 63.4] | **PASS (greedy mode)** |
| V70K-2 | 20K stronger than 70K H2H | step20k vs latest 50/0 = 100% | **PASS strong** |
| V70K-3 | 70K improved over own bootstrap | latest vs bootstrap 50/0 = 100% | **PASS strong** |
| V70K-4 | 70K col-frac (winner-side) ≤ 65% | 100% col>0.3 rate, mean col-frac 63.3%, n_components 14.90 | **FAIL strong** |
| V70K-5 | 70K vs Kraken strong WR ≥ 60% | 49/50 = 98.0%, Wilson95 [89.5, 99.6] | **PASS strong** |

**Critical methodology divergence.** `tournament_validate.py` runs `OurModelBot` at `temperature=0.0` (greedy argmax). §175 `eval_pipeline.py` defaults to `eval_temperature=0.5` (stochastic). Two distinct play modes from the same weights produce radically different rankings. §175 trajectory was 18.0% → 4.0% (T=0.5, n=100) across 20K → 70K. This tourney shows 0.0% → 50.0% (T=0.0, n=50) across the same checkpoints. The H2H 50-0 results between checkpoints reflect 2-effective-unique-games × 25 P1/P2 repetitions inflated to Wilson95 [92.9%, 100%] — over-confident at greedy-mode determinism.

**Interpretation:**
1. V70K-2 PASS strong is robust across both modes (step20k > latest in greedy 50-0; step20k > latest in sampled 18% > 4%). **20K-as-bootstrap decision validated.** Both methodologies agree.
2. V70K-3 PASS strong contradicts L18's "regressed past own bootstrap" framing for greedy mode but is silent on sampled mode (no eval-pipeline measurement of latest_70K vs bootstrap). **L18 needs refinement** (see L21 below).
3. V70K-4 FAIL strong (100% col>0.3 rate for latest_70K wins, n_components 14.90) is the most important finding — **attractor fully captured the latest_70K argmax distribution**, sitting right at the §176 Phase B warning threshold (`n_components ≥ 15`). Argues for aggressive bot-game mixing in Source A; Section 3 weights in `s176_d_plan.md` remain sound.
4. V70K-1 PASS is mode-qualified. Greedy parity with sealbot at step 70K is real (25/25) but DOES NOT contradict the §175 sampled-eval 4% slide.

### Gate 3 — step-20K promotion, verdict **PROMOTED with n=20 caveat**

`reports/s176_gate3/smoke_eval.md` + `checkpoints/bootstrap_model_v6_step20k.{pt,json}`.

- Source: §175 `checkpoint_00020000.pt` (run `c7e74d2842404a82bdd9f62edf740ea2`), source SHA256 `540ac1cf91be38c21b8c10267d36828f34aec242d89c51bc4fd0ea6f2a8680ca`.
- Artifact: `checkpoints/bootstrap_model_v6_step20k.pt`, SHA256 `297e0ce0e48c8c9c417d923610de6ed0166c7295789ebc7bd029c90bb42bce6a`, 17.0 MB (weights-only per §34: only `model_state` + `metadata` retained; optimizer/scaler/scheduler stripped).
- Sidecar: `checkpoints/bootstrap_model_v6_step20k.json`.
- Extraction verification: tensor-equality 143/143 keys MATCH source; round-trip `load_state_dict(strict=True)` returns `<All keys matched successfully>`; `python -m hexo_rl.encoding audit` reports `v6 v6 OK` (declared==inferred).
- Smoke eval: matched §175 methodology (`eval_temperature=0.5`, per-game seed) → 1/20 = 5.0% Wilson95 [0.9%, 23.6%]; binomial P(X≤1 | n=20, p=0.18) = 0.10 — consistent with §175 18/100 anchor at α=0.10; n=20 noise dominates point estimate. Master-prompt STOP boundary (< 5%) not strictly triggered. **Promotion approved.**
- Vast parity: artifact + sidecar pushed; SHA matches both hosts.
- `bootstrap_model.pt` symlink UNTOUCHED (§175-era reproducibility preserved).

### Retained baselines — new row

| Anchor | Path | SHA256 | Source step / run | Eval WR (n=100) | Note |
|---|---|---|---|---|---|
| v6_step20k (§176 Gate 3) | `checkpoints/bootstrap_model_v6_step20k.pt` | `297e0ce0…2bce6a` | §175 step 20000 / `c7e74d…40ea2` | 18.0% [11.7, 26.7] vs SealBot | Empirical §175 sampled-eval peak; weights-only per §34 |

### New mechanism lessons (L21+)

- **L21** — *Eval temperature mode change can invert checkpoint rankings.* §175 step-20K vs SealBot at T=0.5 is 18.0% (peak across §175); same checkpoint at T=0.0 (greedy argmax) is 0.0%. §175 step-70K at T=0.5 is 4.0%; at T=0.0 it ties sealbot 25/25. The argmax-mode and sampled-mode are effectively **two different bots** from the same weights. Any cross-tooling comparison must declare temperature; defaulting to `eval_temperature=0.5` is the convention for §175-era continuity.
- **L22** — *L18 head-drift refinement.* §175 internal drift between 20K and 70K is **policy-distribution flattening into colony attractor** under T=0.5 sampling, NOT loss of argmax dominance over bootstrap. V70K-3 PASS strong (latest dominates bootstrap 50-0 in greedy) + V70K-4 FAIL strong (100% colony-spam wins for latest, n_components 14.90) + §175 eval trajectory (18% → 4% sampled) jointly pin the mechanism. L18 should read "*sampled-policy regression into colony attractor*", not "*regressed past bootstrap*".
- **L23** — *H2H 50-0 in greedy-argmax tourneys is 2 effective unique games × 25 P1/P2 repetitions.* `tournament_validate.py` with `temperature=0.0`, `dirichlet_enabled=False`, `random_opening_plies=0`, and sealbot's roughly-deterministic 0.5s response produces 2 distinct game trajectories per pair (one per opening side). Wilson95 intervals on the inflated n=50 are over-confident. For 20K-as-bootstrap-style discriminator runs, this is acceptable as a sign-test (direction); for absolute strength estimates, prefer eval_pipeline at T=0.5 to inject per-game variance.

### New Falsified Hypotheses Register rows

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §176 Phase A Gate 2 | V70K-4 — §175 step-70K winner-side col-frac ≤ 65% (attractor weakened by training past 50K) | V70K-4 strong FAIL (100% col>0.3 rate, n_components 14.90) | Attractor captured the policy. Greedy-mode wins are uniformly colony-spam patterns. |
| §176 Phase A Gate 2 | L18 strict reading — "§175 latest_70K regressed past its own bootstrap on the selfplay axis" | V70K-3 PASS strong (50-0 H2H latest dominates bootstrap in greedy argmax) | Drift is sampled-mode policy-distribution flattening, not argmax-mode regression. See L22. |
| §176 Phase A Gate 1 | Operator-prompted L18 framing requires no statistical disclaimer at n=50 | Gate 1 dim (vi) — BT 95% CI [-150, +26] crosses zero; H2H 25/25 is parity-consistent only at n=50, NOT strongly-asserted parity | L18 lesson body should disclose CI; framing already correct ("can match") |

### Phase B anchor decision

**Anchor for §176 Phase B sustained: `checkpoints/bootstrap_model_v6_step20k.pt`** (Gate 3 artifact). Validated by:
- V70K-2 PASS strong (20K dominates latest in greedy 50-0)
- §175 sampled-eval (18% vs 4% across 20K vs 70K)
- Both methodologies agree direction.

Phase B implementation opens on a fresh branch (S1–S6 per `reports/s176_d_plan.md`). Mix-ratio bot-pool weights per Section 3 unchanged. **Add eval-temperature pin to all sustained smoke prompts** (recommend T=0.5 for §175 continuity).

### Phase A close

PR #8 mergeable. §175 tmux detached but session preserved. Replay buffer (2.9 GB + 77 MB .recent.npz), 24 checkpoints from step 5000 to 70176, structlog jsonl (49 MB train + 51 MB events) all intact on vast under `c7e74d2842404a82bdd9f62edf740ea2`.

Forensics added: `reports/s176_gate1_operator_review.md`, `reports/s176_gate2_verdicts.txt`, `reports/s176_gate2_tourney/{summary.md,verdicts_v70k.md,…}`, `reports/s176_gate3/smoke_eval.md`, `reports/s175_forensics/{eval_db_rows.json,train_tail_5000.jsonl,checkpoint_log.json,…}`.

---

## §176 — Phase B pre-launch baseline (2026-05-15)

Pre-Phase-B fix wave landed on `phase4.5/s176_phase_a_validation`: six commits absorbing F01–F05 SHOULD-FIX from `reports/s176_review_findings.md` + F06 + F09 N1+N3+N4 cheap STRENGTHEN items. Phase B launch prompt committed as pre-registered done-when artifact per Gate 1 §145 Note 5.

**Forward pointer:** §176 Phase B prompt artifact at `reports/s176_phase_b_prompt.md` (pinned at commit `3994459`). A future Opus session reads that file top-to-bottom and executes Phase B S1–S6 end-to-end. Anchor for Phase B: `checkpoints/bootstrap_model_v6_step20k.pt` (SHA `297e0ce0…2bce6a`, 18.0% n=100 vs SealBot [11.7%, 26.7%]). Forensics specimen retained: `checkpoints/checkpoint_00070000.pt` (L22 attractor-capture witness; F07 deferred to Phase B S6 close-out for retention metadata sidecar OR `docs/rules/checkpoint-archive-policy.md` retention note).

Phase B scope: S1 wrapper audit + anchor n=100 re-baseline (F08), S2 dual-temperature eval ladder + Q14 close, S3 `n_components` colony POC at `pool.py game_complete` selfplay-mode emit, S4 Source A static corpus mixing design doc (lowered-expected-benefit framing per L22), S5 Source B live cross-bot games design doc (elevated to primary-fix-mechanism candidate per L22 + V6 PASS opponent-coupling), S6 close-out. Pre-registered verdicts V-PhaseB-1..9 freeze at the artifact's commit SHA per L13 + A4 do-not #9.

Phase B does NOT run a sustained training smoke; design docs queue for §177+ implementation. Pre-Phase-B fix wave commits `1d5b6b5..2014669`:

| Commit | Item |
|---|---|
| `1d5b6b5` | F01 Gate 1+2+3 close-out + L21–L23 + 3 Falsified rows (+ F06 + F09 N4 L18 inline) |
| `834f761` | F04 CLAUDE.md current phase refresh |
| `9994d5d` | F05 roadmap §175 row + §176 Phase A/B rows |
| `6c30f03` | F02 S2 eval_temperature=0.5 pin per L21 |
| `80a0205` | F03 S3 n_components threshold mode-scope qualifier |
| `2014669` | F09 N1+N3 risk register rows 8+9 |
| `3994459` | Phase B prompt artifact pre-launch baseline |

**Master merge + anchor SHA parity verification (2026-05-15).** `phase4.5/s176_phase_a_validation` merged into master via `--no-ff` (14 commits, merge SHA pinned at master push). Branch preserved local + remote per L13 traceability + bisect anchor. Anchor `bootstrap_model_v6_step20k.pt` PARITY verified across hosts: laptop SHA `297e0ce0e48c8c9c417d923610de6ed0166c7295789ebc7bd029c90bb42bce6a` (17,035,312 bytes) == vast `/workspace/hexo_rl/checkpoints/bootstrap_model_v6_step20k.pt` SHA `297e0ce0e48c8c9c417d923610de6ed0166c7295789ebc7bd029c90bb42bce6a` (17,035,312 bytes). Forensics specimen retained on vast: `/workspace/hexo_rl/checkpoints/checkpoint_00070000.pt`, SHA `1f6aa40852e57db6e3cdeac64adb879590370b8596975f2b86f9023f459224dc`, 51,131,811 bytes — L22 witness for Phase B S6 sidecar consumption.

**Phase B prompt pinned ready-to-launch (2026-05-15).** Pre-launch audit returned LAUNCHABLE_WITH_STRENGTHEN; two minor fixes folded in commit `cf14d72` (+10/-8 LOC, within 30-LOC cap): (a) Q11 naming clarification — S6 now creates a NEW `Q-§176-mechanism` Q-row carrying §174/L22 mechanism question forward, Q11 RESOLVED 2026-04-28 status + body preserved with only a one-line `See also` cross-reference appended; V-PhaseB-9 updated to match. (b) Wave A1 SHA parity re-check folded in as 30-second pre-launch insurance. Final Phase B launch artifact = `reports/s176_phase_b_prompt.md` at commit SHA `cf14d729f81f3a3f59071ad07dda3448e97c15ae`, blob SHA `6b23df987260b4affb6baa6f48efb34d24a28e2d`. A future Opus 4.7 x-high session reads that file top-to-bottom with no prior context required and executes Phase B S1–S6 end-to-end.

---

## §177-pre — Wave A1 baseline (n=100 dual-temperature SealBot eval) — 2026-05-15

Operator-driven Wave A1 ahead of S1–S6 implementation: 30-s SHA parity re-check + n=100 dual-temperature SealBot eval of the §176 Phase A anchor + variant config authoring + §177 training launch from the step-20K anchor (NOT Phase B's design-only scope; Phase B S1–S6 sequence still queued for a future session against this baseline).

**Anchor SHA parity re-check (vast 5080, post-master-pull 7d4b4fb).** PASS both:
- `bootstrap_model_v6_step20k.pt` → `297e0ce0e48c8c9c417d923610de6ed0166c7295789ebc7bd029c90bb42bce6a` (17,035,312 B) ✓ matches expected
- `checkpoint_00070000.pt` → `1f6aa40852e57db6e3cdeac64adb879590370b8596975f2b86f9023f459224dc` (51,131,811 B) ✓ matches expected

**n=100 dual-temperature SealBot baseline** (MCTS-128, `random_opening_plies=0`, `time_limit=0.5`, seed_base=42):

| Mode | n | wins | WR | Wilson 95% | mean_ply | elapsed_sec |
|---|---|---|---|---|---|---|
| greedy T=0.0   | 100 | 0  | 0.0%  | [0.0%, 3.7%]  | 77.4 | 1722 |
| sampled T=0.5  | 100 | 12 | 12.0% | [7.0%, 19.8%] | 52.7 | 1129 |

Reports: `reports/phase_b_wave_a1/{baseline_greedy.json, baseline_sampled.json, baseline_*.log}`.

**Pre-registered verdicts BL-1..BL-4 (operator prompt) — disposition.** Three of the four were NULL'd: the operator's pre-registration conflated greedy vs sampled modes for the anchor-reproduction test, but §175 step-20K's documented 18% is the **sampled** number per `eval.yaml::eval_temperature=0.5`, and L21 explicitly establishes **greedy=0%** for the same checkpoint. NULL'd per L13 + skill `investigation-probe-smoke-verdict` anti-pattern #1 (don't rewrite report to fit verdict).

| ID | User-pre-registered hypothesis | Literal verdict | Disposition |
|---|---|---|---|
| BL-1 | Greedy n=100 reproduces §175 step-20K eval (PASS: WR ∈ [11.7, 26.7]) | FAIL (0% vs 18% expected) | **NULL — basis invalidated by L21.** §175 18% is sampled mode (`eval_temperature: 0.5` pin); L21 explicitly states step-20K **greedy = 0.0%**. The 0/100 greedy result EXACTLY reproduces L21's documented greedy-mode value. Reframed: GREEDY_L21_REPRO = PASS. |
| BL-2 | Sampled diverges from greedy ≥ 6pp absolute per L21 | PASS (12pp) | **PASS** — L21 dual-mode divergence confirmed at the Phase B baseline. Locks dual-temperature gate scope for Phase B S2. |
| BL-3 | Sampled WR < greedy WR (matches §175 trajectory direction) | FAIL (sampled > greedy by 12pp) | **NULL — basis invalidated by L22.** Sampled > greedy at step-20K is the normal **pre-flattening** ordering (exploration-spread baseline). The §175 trajectory direction (sampled < greedy) emerges **post-flattening** at step-70K per V70K-3 PASS strong + §175 eval slide 18%→4%. BL-3's framing should be applied to a post-flattening checkpoint, not the pre-flattening anchor. Sampled-canonical meta-claim PASSes independently via eval.yaml `eval_temperature: 0.5` pin + L21 convention. |
| BL-4 | Colony fraction in sampled wins ≥ greedy wins by ≥ 10pp | undefined | **NULL — greedy wins=0 ⇒ divisor zero.** No greedy-side colony rate computable. Sampled-side colony measurement deferred to Phase B S3 implementation. |

**Phase B prompt S1 re-baseline verdict (canonical).** Per Section 6 V-PhaseB-5 + Section 3 S1 PASS criterion (point-estimate-in-original-CI):

- Sampled n=100 = 12.0%, point estimate **12.0% ∈ [11.7%, 26.7%]** (original §175 anchor CI).
- **V-PhaseB-5 verdict: PASS** (anchor preserved by extraction; tensor-equality + behavioral-equality both green).
- Wilson95 of 12/100 = [7.0%, 19.8%] overlaps but does not contain §175 [11.7%, 26.7%]; point-estimate criterion is what the Phase B prompt pre-registered, and it is met.
- F08 SHOULD-RE-BASELINE deferred-strengthen item now CLOSED.

**Mode comparison signature (Wave A1 forensics).**

- Greedy mean_ply 77.4 (long grinding losses, no termination via win).
- Sampled mean_ply 52.7 (12 wins terminate games earlier; sampled exploration shortens losing trajectories too).
- Sampled-greedy elapsed asymmetry: 1722 s vs 1129 s (greedy 1.5× sampled wall-time despite identical MCTS budget) — sampled's shorter games dominate.

**Variant config + training launch.** Authored `configs/variants/v6_sustained_s177.yaml` (commit `166ac7c` on master): clone of §175 `v6_sustained.yaml` with bootstrap delta only (CLI `--checkpoint` flag); L9 cosine-temp + jitter pairing preserved, `random_opening_plies=0`, `eval_interval=10000`, n_games=100, total_steps=100000. Dashboard port falls back to `monitoring.yaml` default 5001 (vast port 8080 occupied by jupyter-notebook pid 690). §177 training launched on vast 5080 in tmux `s177` from `bootstrap_model_v6_step20k.pt` along the §175 recipe — empirical zero-point for Phase B Wave C source-mixing experiments. No bot-game mixing (Source A/B arrive after Phase B S1–S6 land).

---

## §177 — v6 sustained from step-20K anchor: closed by recipe-attractor reproduction (2026-05-18)

**Branch:** `phase4.5/m176a_v7mw` (continuation), tmux session `s177` on vast 5080.
**Anchor:** `bootstrap_model_v6_step20k.pt` (SHA `297e0ce0…2bce6a`, §175 step-20K promotion).
**Launch commits:** `166ac7c` (variant `v6_sustained_s177.yaml`), `d70507a` (Wave A1 baseline doc), `072d0db` (vast checkup script).
**Reference §177-pre:** lines 1428–1470.

**Closing trajectory (vast, per §178 pre-design investigation):** SealBot WR 2→0% across step 10K→40K, mirroring §175 18→4% trajectory in different anchor. Combined with §175, falsifies the residual "anchor weight is the lever" hypothesis: same recipe + different anchors → same colony-attractor capture.

**Closed by interrupt** at the point §178 pre-design investigation gathered sufficient evidence to motivate the bot-mix + ply-cap-value mechanism intervention. No further sustained running was useful.

### L24/L25/L26 mechanism-lesson candidates (pending operator confirm — defer to L24+ register if accepted)

| ID | Candidate lesson | Evidence | Implication |
|---|---|---|---|
| L24 | **Recipe-dependent colony attractor, not anchor-dependent.** Same §175 recipe captures the colony attractor regardless of bootstrap anchor (v6.pt clean AND v6_step20k.pt continuation both ride the attractor). | §175 18→4% + §177 2→0% across two anchors; combined PASS in `reports/s178_pre_design_investigation.md` SA-A | Refines L18 (anchor-mistake signal coupling) — anchor swap is NOT a lever; recipe/objective intervention required. §178 trials this. |
| L25 | **G4 value-head band FAIL concurrent with colony capture.** §175 + §177 both showed `value_fc2_weight_abs_max` drift outside `[0.154, 0.462]` coincident with sealbot WR collapse. | §175 stages probe + §177-pre L21/L22 dual-mode | Value head flattening tracks colony entrenchment; G4 is the upstream WR-collapse predictor. |
| L26 | **Ply-cap truncation outcome = organic draw outcome silently dilutes finish-pressure on long colony-prone games.** Pre-§178 Rust path writes `outcome = draw_reward` for both winner=None && ply≥max AND winner=None && legal_count=0. Only `terminal_reason` metadata distinguishes — and value-head never sees it. | §178 investigation SA-C VC-2; §178 T2 resolves via `ply_cap_value` split | Operator pre-commit `draw_value -0.5→-0.1` alone REMOVES finish-pressure; `ply_cap_value` split (-0.5 vs -0.1 in §178 — operator dialed back from design -0.8) restores it. |

### Falsified Hypotheses Register row added

| § | Hypothesis | Evidence FAILing | Closer |
|---|---|---|---|
| §177 | Step-20K anchor escapes the colony attractor under §175 recipe | §177 2→0% across 10K→40K reproduces §175 18→4% on different anchor | §178 launch — same recipe, different mechanism (bot-mix + ply-cap split) |

---

## Supplementary tables — preserved from per-§ bodies

### §70 mode-collapse evidence (round-robin signature)

| Matchup | Score | Game length |
|---|---|---|
| ckpt_13000 vs ckpt_14000 | 100/0 P1 | exactly 25 moves, carbon-copy |
| ckpt_14000 vs ckpt_15000 | 50/0 P1 + 50 draws | 31-33 moves, carbon-copy |
| ckpt_15000 vs RandomBot | 50/0 P1 | 11-33, varied |

H(π) band 1.49–1.70 (post-collapse) vs bootstrap 2.665 — entire post-bootstrap band sits within 0.21 nats. Fixed point, not progressive collapse. Restart should select on buffer composition, not entropy rank.

### §73 Dirichlet port verification (commit `71d7e6e`)

| Site | Count post-port | §70 count |
|---|---|---|
| `apply_dirichlet_to_root` | 10 | **0** |
| `game_runner` | 30 | 30 |

10 unique noise vectors across workers. Top-1 prior: `0.540 → 0.412` post-noise (−12.8pp). Top-1 visit fraction at cm=0: 0.474 vs §70 baseline 0.65 (−17.6pp). Workers at cm=0 ply=0 span 0.33–0.55 (diverging vs §70 identical across 14 workers).

### §91 / §100.d threat-probe criterion (locked, REVISED from §85/§89)

| # | Condition | Threshold |
|---|---|---|
| C1 | contrast_mean ≥ max(0.38, 0.8 × bootstrap_contrast) | floor 0.40 (bootstrap=0.502) |
| C2 | ext_in_top5_pct ≥ 40 | direct colony-spam test |
| C3 | ext_in_top10_pct ≥ 60 | catches partial sharpness |
| C4 | abs(ext_logit_mean − bootstrap_ext_logit_mean) < 5.0 | **warning only, never gates** |

C1–C3 must all PASS for `make probe.latest` exit 0. C4 is BCE-drift / Q19 monitoring hook. Baseline `fixtures/threat_probe_baseline.json` v6 (§106 real-position regen): contrast 0.502, top5 50%, top10 65%.

### §116 torch.compile retry — three-mode comparison (PT 2.11 + Py 3.14)

| Metric | Eager | default | reduce-overhead | max-autotune-no-cudagraphs |
|---|---|---|---|---|
| Throughput batch=64 (pos/s) | 2,529 | 3,665 | **3,788** | 3,744 |
| Throughput speedup vs eager | 1.00× | 1.45× | **1.50×** | 1.48× |
| Latency batch=1 (mean ms) | 3.553 | 2.844 | **1.897** | 3.007 |
| Latency speedup vs eager | 1.00× | 1.25× | **1.87×** | 1.18× |
| Compile time | — | 11.8 s | **6.4 s** | 29.9 s |
| Graph breaks | 0 | 0 | 0 | 0 |

**§116.a landed then reverted (`1e2d82b` + `41ffad5` mode-plumbing/OptimizedModule unwrap stay; `e102a0a` flag flipped back to `false`)** — second resume at step 6002 hit futex_do_wait on 78 threads (trainer+inference dual-JIT). Re-enable preconditions documented inline.

### §118 → §121 axis-clustering causal chain

```
RecentBuffer un-augmented (67% of late-training batch)
  → absolute-position FC policy head learns axis-asymmetric features freely
  → MCTS visits concentrate on preferred E-W axis (no symmetry pressure)
  → self-play generates axis-biased trajectories
  → RecentBuffer samples reinforce bias at 67% of gradient
  → loop closes; bias grows monotonically until truncation or intervention
```

§120 closed the symmetry coverage gap (4.7/12 → 12/12 group elements per batch row) but **augmentation alone is insufficient** for relational biases (D13 heuristic preserved under rigid transformation). Two independent components in §121 — directional heuristic (rotation-equivariant, fixed by §130 permanent rotation) + clustering magnitude (rotation-invariant, architectural). §122 architectural redesign blocked on B1 D17 ablation + B2 backbone-form memo + B3 retrain cost + B4 buffer compat.

### §156 R10 within-bisection (each variant removes ONE knob from R10)

| Variant | Knob removed | n | draws | draw_rate (95% CI) | mean_ply | stride5 P50/P90 | colony_wins | wall |
|---|---|---:|---:|---|---:|---:|---:|---:|
| R10 | (full smoke regime) | 200 | 182 | 91.0% [86.2%, 94.2%] | 140 | 84/97 | 9 | 2702s |
| R11 | Dirichlet ε=0.10 → 0 | 200 | 176 | 88.0% [82.8%, 91.8%] | 139 | 76/86 | 15 | 2649s |
| **R12** | **cosine temp → fixed τ=0.5** | 200 | 10 | **5.0% [2.7%, 9.0%]** | 63 | **3/4** | 134 | 738s |
| R13 | opening_plies 1 → 4 | 200 | 170 | 85.0% [79.4%, 89.3%] | 135 | 82/100 | 15 | 2620s |
| R14 | playout cap → uniform 600 | 200 | 198 | 99.0% [96.4%, 99.7%] | 149 | 132/133 | 0 | 3576s |

R12 colony rate 67% (134/200) is the §147 v5 / §154 v9 colony attractor — mitigated by `legal_move_radius_jitter: true` + `bootstrap_floor.min_winrate: 0.45`.

### §157 5k smoke abort-signatures (5080, 1256 games, wall 3h 18m)

| Signature | End-of-run | Threshold | Status |
|---|---|---|---|
| stride-5 P90 (rolling 50 games) | 4 | 60 | ✅ |
| row max P90 (rolling 50 games) | 13 | 50 | ✅ |
| colony_ext_frac max (per-game) | 0.086 | 0.40 | ✅ |
| colony_terminal_fraction | 0.000 | — | ✅ |
| draw_rate (last 200 games) | 7.5% | 70% (WARN-only) | ✅ |
| grad_norm | 0.98–1.62 | 10.0 hard-abort | ✅ |
| NaN losses | 0 | any | ✅ |

Final-ckpt SealBot offline eval n=200: **19.0% (38/200)** — beats 17% gate, matches §150 v7full 17.4% n=500 within sample noise.

### §167 Phase B v8 variant matrix retrains (5080 + laptop, 30 epochs each)

| Arm | Final loss | NaN-skip rate | SealBot argmax n=200 |
|---|---|---|---|
| B0 (128×12, no GPool) | 3.2737 | 0% | 0/200 |
| **B1 retry (128×12 + GPool {6,10})** | **3.227** | 24% NaN-skipped (`4c7dbb5` `isfinite` guard) | 0/200 |
| B2 (96×12 + GPool, laptop) | 3.276 | 0% | 0/200 |
| B3 (128×10 + GPool {5,8}) | 3.2536 | 0% | 0/200 |
| B4 (160×12 + GPool, batch 128 fallback) | 3.2249 (~6 ep effective) | 80% NaN-skipped | 0/200 |

v7full v6-argmax baseline radius curve: r=5 6.5% / r=8 12.5% / r=10 15%. B1 across radii r=8/10/12: 0%/0%/0%. **Cross-encoding argmax-only handicap is structural** — K-cluster's inference-time multi-window pooling acts as tiny ensemble that bbox lacks. Effect vanishes under MCTS (Phase D §168).

### §169 four-way ablation matrix (post-§169a probe)

| Arm | Encoding | Pool | Loss (30 ep) | argmax @ r=8 n=200 | MCTS-128 n=200 | params (M) |
|---|---|---|---|---|---|---|
| **A1 (canonical)** | v6w25 (25×25 K-cluster) | min/max | **3.57** | **14.5%** [10.3%, 20.0%] | 25% (§169 P1 MCTS-32 n=20) | 5.29 |
| A2 | v6w25 K-cluster | PMA | 4.25 | 4.5% [2.4%, 8.3%] | 3.5% [1.7%, 7.0%] | 6.30 |
| A3 | v6w25 K-cluster | PMA + global token | 3.62 | 7.5% [4.6%, 12.0%] | 2.5% [1.1%, 5.7%] | 6.37 |
| A4 | v8 bbox + canvas_realness + PartialConv2d | KataGo head | **3.47** | **0.0%** [0.0%, 1.9%] | **0.0%** [0.0%, 1.9%] | 3.85 |

A3 learned `pool_global_gate` climbed 0.10 → 0.66 (6.6× over init) — global branch earns weight, lifts policy argmax, but doesn't fix PMA's K=1-pretrain-regime cross-cluster blindness at search time.

### §170 P4 P1 gpool-bias-policy-only (CANONICAL)

A1+gpool-bias-policy-only retrain → **22% argmax SealBot @ r=8 n=200**. Full gpool-bias on both heads (P3) is NULL on value. Earlier attribution of 22% lift to A4 fine-tune was wrong (memory `project_bootstrap_argmax_drift_check_20260511.md` documents correction); A4 was already 0% pre-fine-tune. Mechanism: gpool-bias on policy head only, applied on A1 K-cluster.

### §172 A10 close-out — high-risk hazard retirement

| # | Site | Hazard | Closure |
|---|---|---|---|
| H1 | `engine/src/game_runner/mod.rs:159` `SelfPlayRunner::new` | pyo3 default kwargs silent v6 fallback (`feature_len=8*19*19`, `policy_len=19*19+1`) | derive from `spec.state_stride()`/`spec.policy_stride()`; legacy-caller backward-compat retained |
| H2 | `engine/src/inference_bridge.rs:295` `InferenceBatcher::new` | same pattern as H1 | `encoding_spec` kwarg added |
| H3 | `engine/src/replay_buffer/sym_tables.rs:26` `N_ACTIONS=362` | v6-only consumers; v8 silently uses wrong value | audit confirmed all v6-only; v8 pinned to `spec.policy_stride()=625`; Rust unit test pins |

A10 commits: ab760ae (T1 stamp model_variant), ae97525 (T2 migrations consolidate), a133d52 (T3 DeprecationWarning), 2dc086f (T4 RegistrySpec accessors), 1262e0c (T5 retire `*_V8` const presets), 823e241 (T6 retire `config["board_size"]`), e2a73f5 (T7 cross-table consistency INV-1..6), e83e78a (T8a allowlist 881→201 hits), f7c2bc8 (T8b HIGH-RISK pyo3 fix), 47b7f17 (T9 `<auto>` config form), 1595008 (T10 model-variant backfill), 576f69d (T11 pyo3 `from_py_object` TODO).

### §172 Phase B B2 milestone curve (30K v7full sustained, n=20)

| Step | sealbot | bootstrap_anchor | best_arena (n=100) | argmax_n | elo | promoted |
|---|---|---|---|---|---|---|
| 5K  | 0.100 | 0.350 | 0.410 | 0.000 | -94.2 | F |
| 10K | 0.200 | 0.600 | 0.570 | 0.000 | +50.5 | F (CI block) |
| 15K | 0.050 | 0.650 | 0.500 | 0.000 |  -9.4 | F |
| **20K** | **0.050** | **0.650** | **0.610** | 0.000 | +34.0 | **T** (only promotion) |
| 25K | 0.050 | 0.500 | 0.560 | 0.000 | -63.2 | F (CI block) |
| 30K | 0.050 | 0.600 | 0.550 | 0.000 | -36.3 | F (CI block) |

§150 v7full anchor SealBot 17.4% n=500. B2 finished sealbot 0.050 n=20 Wilson95 [0.009, 0.236]. **REGRESSION gate did not fire** (UB 0.236 covers anchor LB 0.143). DRIFT gate cold (argmax_n 0/20 all rounds). Self-play improving vs self (best_arena 0.41 → 0.61) while sealbot stalled — **encoder-specific transfer gap, not value-drift pathology**.

### §173 bench gate (pre-α vs post-A5b v2, n=5, 90s warmup, compile OFF)

| Metric | Pre-α median | Post-A5b v2 | Δ | Target | Status |
|---|---|---|---|---|---|
| MCTS sim/s | 80,601 | 80,287 | −0.4% | ≥ 26,000 | PASS |
| NN inference pos/s | 14,278 | 14,148 | −0.9% | ≥ 8,250 | PASS |
| NN latency ms | 1.551 | 1.537 | −0.9% | ≤ 3.5 | PASS |
| Buffer push pos/s | 992,777 | 1,023,047 | +3.1% | ≥ 630,000 | PASS |
| Buffer sample raw µs | 757 | 764 | +0.9% | ≤ 1,550 | PASS |
| Buffer sample aug µs | 759 | 768 | +1.2% | ≤ 1,800 | PASS |
| GPU util % | 94.0 | 94.0 | — | ≥ 85% | PASS |
| VRAM GB | 0.105 | 0.105 | — | ≤ 6.4 | PASS |
| Worker pos/hr | 80,715 | **104,141** | **+28.9%** | ≥ 250,000 | PASS |
| Batch fill % | 99.999 | 99.976 | −0.02pp | ≥ 80% | PASS |

A5b initial −10.47% worker_pos_per_hr regression recovered to +6.01% via scalar-API + `#[inline]` (`feedback_registryspec_by_ref_in_hotpath.md` — RegistrySpec ~174-byte copy per MCTS sim).

### §173 HAZARD ledger (closed)

| HAZARD | Description | Closed by |
|---|---|---|
| H1-α | SymTables v6 unconditional → K-window rotation silent shape mismatch | A5a (`sym_tables_for()`) |
| H2-α | `rotate_aux_inplace` TOTAL_CELLS=361 silent ownership corruption for v6w25 | A5a |
| H3-α | `views[k][..TOTAL_CELLS]` truncates chain encoding for v6w25 | A5a |
| H4-α | `aggregate_policy*` BOARD_SIZE=19 — 362-vector where 626 required | A5b |
| H5-α | `sample.rs:220` pass-slot copy: latent OOB for v8 (`has_pass_slot=false`) | A4 |
| H6-α | `mod.rs:342 STATE_STRIDE` v6 constant in `collect_data` | A5a |
| H7-α | HEXB on-disk format has no encoding-name header — blocks first v6w25 persist | **CARRIED to §174** |

Python `EncodingSpec.n_cells` parity bug (used `board_size²` instead of `trunk_size²`) closed by A3.

### §174 bootstrap matrix (post-mortem)

| Bootstrap | Recipe | Final loss | SealBot MCTS-128 (random_plies=0) | Selfplay median plies @ R=8 | G4 status |
|---|---|---|---|---|---|
| **v6 (`bootstrap_model.pt`)** | 30 ep cosine 2e-3/5e-5 | — | reference | reference | PASS — §175 anchor |
| v7full | 30 ep cosine 2e-3/5e-5 | 3.1573 | 17.4% n=500 (§150) | — | PASS |
| v6w25 e30 | 30 ep cosine 1e-3/5e-5 | 3.96 nats vs uniform | **0% (0/200)** | 6 | PASS within band |
| v6w25 e50 | 50 ep cosine 1e-3/5e-5 | (lower) | 10% (10/100, artifact-suspect) | 6 | **MARGINAL FAIL** 0.489 vs band [0.154, 0.462] |
| v6w25 transfer FT | v6 trunk + Xavier policy FC + drop-restart FT | — | **0% (0/200)** | 8 | — |

Eval random_opening_plies 4 → 0 in `configs/eval.yaml:88` fully explains §168 → §174 sealbot WR drop (14.5% → 0%) — with 4 random plies model got free positional diversity masking weaknesses; with 0 SealBot's preparation lands cleanly.

### G-gate wiring status (Track 2 audit, 2026-05-13)

| Gate | Description | Wiring |
|---|---|---|
| G3 | Monotonic depth scaling | `avg_game_length` in `iteration_complete` (orchestrator.py:336); per-game `game_length` in structlog `game_complete` (pool.py:593) |
| G4 | Value-head |max| ±50% band [0.154, 0.462] around v7full 0.308 | **NEW §174 Track 2** — `_g4_value_head_band_check` runs at start of every `run_evaluation`; result persisted in `results["value_fc2_weight_abs_max"]` + `results["g4_value_head_band_pass"]`; structlog WARNING on violation; constants gate-internal (variants do not override) |
| G5 | Per-cluster variance drift ≤30% | `cluster_value_std_mean` + `cluster_policy_disagreement_mean` + `cluster_variance_sample_count` emitted in `iteration_complete` (orchestrator.py:349-351) + `train_step_summary` (orchestrator.py:404-406); drift detection is post-hoc operator computation |

`random_opening_plies` two distinct fields (selfplay vs eval paths): `selfplay.random_opening_plies` (`configs/selfplay.yaml:66` default 1, vast.yaml override 0); `eval_pipeline.eval_random_opening_plies` (`configs/eval.yaml:88` default 0, was 4 pre-§174). Pipeline build path `pipeline_setup.py:52` loads eval.yaml directly — separate from training base-config list.

### §174 escalation decision matrix

| Track 1 finding | §175 action |
|---|---|
| e30 v6w25 ≥ §150 v7full anchor on MCTS-128 sealbot n=100 | Launch sustained with e30 v6w25 |
| e30 v6w25 < §150 anchor by > 5pp absolute | Re-evaluate: retrain with different recipe OR fall back to v7full for §174 |
| e30 v6w25 within ±5pp of §150 anchor (within noise) | Launch sustained — gap is in measurement noise; α + radius curriculum are net new levers |

Track 1 returned 0% MCTS-128 across all three v6w25 bootstrap recipes → escalation to §175 v6 sustained (100K steps, n=100 SealBot eval, matched cosine LR from §174 vast.yaml, selfplay encoding v6 single-window 19×19 existing path).

---

