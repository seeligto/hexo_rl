# §176 Phase A Wave E — Fresh-context independent audit of Wave D plan

**Branch:** `phase4.5/s176_phase_a_validation` | **Date:** 2026-05-14 | **Auditor:** Wave E (no prior conversation context).
**Plan audited:** `reports/s176_d_plan.md` (545 lines).
**Method:** verified ≥10 cited claims against source files. Walked Falsified Register, master + A4 do-not lists, risk register, commit-boundary table.

---

## (i) Empirical-evidence backing — **CLEAR**

Claims sampled (12, ≥10 floor):

| # | Plan claim | Source cited | Verified | Result |
|---|---|---|---|---|
| 1 | our_v6_mcts128 BT −62 Elo vs sealbot, 25/50 H2H | `summary.md:9-19` | `ratings.csv` row our_v6_mcts128 = `-62.1`; `h2h_matrix.csv` row=our_v6_mcts128 col=sealbot = 25 | PASS |
| 2 | POC = `n_components`, Cohen's d = −0.822 | A3 §e table line 184 | A3 §e table: `n_components (all) −0.822 9.29 13.63` rank 1 | PASS |
| 3 | MCTSBot blocked on missing weights | A1 §b + `.gitignore:8` | A1 §b table: "MISSING — never committed (.pt in .gitignore:8)"; A1 §d shows `FileNotFoundError` at `mcts_bot.py:46` | PASS |
| 4 | kraken_minimax_fast 30/50 vs strong, BT delta −5 Elo | summary V2 + `h2h_matrix.csv` | h2h row kraken_minimax_fast col kraken_minimax_strong = 30; ratings: strong −493.8, fast −499.4, delta −5.6 within CI [−612,−376] | PASS |
| 5 | V3 PASS: sealbot 35.0% vs kraken_strong 7.1%, non-overlap by 18pp | summary V3 §76-83 | `verdicts.txt:38-44` sealbot 35.0% [29.6%, 40.9%]; kraken_strong 7.1% [4.2%, 11.8%]; gap 27.9pp, CIs gap 17.8pp | PASS (plan rounds to 18pp) |
| 6 | fb_n2=438 / fb_rand=0 / 1050 games | summary line 118 | `/tmp/s176_c_tourney.log`: `krakenbot_fallback_neighbor2` count = 438; `krakenbot_fallback_random_legal` = 0; per_game.jsonl = 1050 lines | PASS |
| 7 | sealbot colony spread 41pp across opponents | summary V6 + verdicts.txt:13-35 | verdicts.txt pair colonies: vs kraken_strong=0.412, vs randombot=0.024, spread 38.8pp; plan cites "0.412 − 0.000 (our_v6_argmax)" giving 41.2pp | PASS (operator framing) |
| 8 | BT MinimaxBot strong −493.8, fast −499.4, our_v6 −62.1 | `ratings.csv` | ratings.csv values exact | PASS |
| 9 | n_components mean 9.29 (20K) → 13.63 (50K) | A3 §b + §c | A3 §b table cohort row matches; A3 §c step-bin row 40-45K reads 13.77 | PASS |
| 10 | Smart neighbour-2 fallback at `krakenbot_bot.py:37-69` | A1 + Wave B | `hexo_rl/bots/krakenbot_bot.py:37-69` contains `_smart_legal_fallback` exactly | PASS |
| 11 | Q11 colony detection resolved 2026-04-28; orphan exclusion | `docs/06_OPEN_QUESTIONS.md:390` | Q11 RESOLVED 2026-04-28 verbatim, "single orphan stones are excluded" | PASS |
| 12 | §17 GIL regression 3.3× (1.52M→464K pos/hr) at sprint log line 597 | Falsified §17 | sprint log line 597 reads "3.3× GIL contention regression (1.52M→464K pos/hr)" verbatim | PASS |

**Mix-ratio per-source weights consistency (Section 3) check.** Wave C BT ladder: sealbot 0, our_v6 −62 (CI overlap with sealbot), kraken_strong −494, kraken_random −3072. Plan splits 50/30/15/5 within bot pool. Direction monotone with Elo. our_v6 30% (vs sealbot 50%) is conservative given CI overlap — defensible. kraken_random at 5% with BT −3072 is empirically diversity-only per V6; weight is non-zero but minimal. **Consistent.**

---

## (ii) Falsified Hypotheses Register re-litigation — **CLEAR**

Walked all 16 rows of register (sprint log 535-557):

| § row | Falsification | Plan re-litigation? | Verdict |
|---|---|---|---|
| §154 v9 hex-trunk | Probe ≠ MCTS-matched | S2 retains SealBot MCTS-matched gate (compliance §530). No probe-only gate. | CLEAR |
| §155 R10 super-additive | Cosine-temp sole driver | S1-S6 touch no MCTS knobs | CLEAR |
| §169 A4 PartialConv2d / canvas_realness | bbox structural | No NN architecture change. S1-S6 pure plumbing + design docs | CLEAR |
| §170 A2/A3 PMA pool | K-cluster min/max canonical | No pool architecture change | CLEAR |
| §170 P3 gpool-bias-all | policy-only load-bearing | No NN head changes | CLEAR |
| pre-§148 v6 corpus contamination | v7+ Elo-weighted canonical | Section 3 "10% human corpus (v7+ Elo-weighted per L15)". Source A bot mix does include `our_v6_mcts128` 30% — this is bot SELFPLAY-derived inference games NOT v6 corpus contamination. Distinction clean: our_v6 is a checkpoint playing live in tourney/Source A games, not the retired v6 human-corpus mixture. | CLEAR |
| §174 e50 | e30 ceiling | No pretrain recipe proposed | CLEAR |
| §174 radius compression | radius inert | No radius tuning | CLEAR |
| §174 bootstrap recipe | argmax-degeneracy / selfplay-layer | Plan does not propose bootstrap fix | CLEAR |
| pre-§73 Dirichlet missing | resolved | n/a | CLEAR |
| pre-§47 FP16 | resolved | n/a | CLEAR |
| pre-§101 C1 promoted=evaluated | resolved | n/a | CLEAR |
| §169 P0 broadcast | resolved | n/a | CLEAR |
| §131 18-plane | resolved | n/a | CLEAR |
| forced-win short-circuit | resolved | n/a | CLEAR |
| §171 A4 P2-reopen frozen-spine FT | dead bin | No fine-tune proposed | CLEAR |

§17 SealBot daemon GIL — register row at sprint log line 597. **Plan S5 explicitly mandates subprocess isolation, cites `c9f39de` and `tournament_validate.py` (single-game-at-a-time launcher template).** No in-process daemon proposal. CLEAR.

No falsified row re-litigated.

---

## (iii) Master + A4 do-not list violations — **CLEAR**

A4 15-item walkthrough (compliance check at plan §497-522):

| # | A4 item | Plan claim | Verified |
|---|---|---|---|
| 1 | No in-process daemon | S5 subprocess isolation | S5 §294-300 cites sprint log line 597; recommends `tournament_validate.py` single-game launcher. Wave B §a §43-45 also enforces "single-game-at-a-time" | CLEAR |
| 2 | No stacked knob changes | Plan touches no cosine-temp / Dirichlet / opening_plies / playout_cap | grep confirms no knob mention | CLEAR |
| 3 | Loss ≠ WR proxy | S3 uses n_components; S2 uses H2H + Elo | confirmed | CLEAR |
| 4 | No PMA pool replacement | No NN arch change | confirmed | CLEAR |
| 5 | e30 epoch ceiling | No bootstrap proposed | confirmed | CLEAR |
| 6 | No LEGAL_MOVE_RADIUS alone | No radius changes | confirmed | CLEAR |
| 7 | No cosine-temp without jitter | No cosine-temp changes | confirmed | CLEAR |
| 8 | No v6 corpus baseline | Section 3 "10% human corpus (v7+)"; our_v6 is checkpoint-bot, not corpus | CLEAR |
| 9 | No gate recalibration | S3 POC = warning-only, NOT WR gate; SealBot WR target unchanged | CLEAR |
| 10 | No smoke step extension | No smoke runs in §176 Phase B | CLEAR |
| 11 | No frozen-spine FT | No fine-tuning | CLEAR |
| 12 | No probe-only gates | S3 warning-only; SealBot MCTS-matched primary | CLEAR |
| 13 | No dev-default cold smoke | No smoke runs | CLEAR |
| 14 | No "one-line" BotProtocol | S1 budgets +300 LOC (matches L17 magnitude) | CLEAR |
| 15 | No implicit done-when | Plan IS pre-registered; Wave E (this audit) fresh-context review locks it | CLEAR |

Master prompt 10-item do-not (plan §524-532):
- (#1) GIL daemon: S5 subprocess isolation. PASS.
- (#2) Stacked knobs: none. PASS.
- (#7) Python-only POC: S3 explicit "Python-only this sprint per master prompt constraint #7". PASS.
- (#12) MCTS-matched eval: SealBot primary gate retained. PASS.
- Others not invoked.

**No do-not violations.**

Cold-path bench-gate skip verification: every S# claims "Cold path". S1 (`hexo_rl/bots/*.py`) — cold (not MCTS / replay / inference per CLAUDE.md Prime Directive). S2 (`configs/eval.yaml`, `opponent_runners.py`, `eval_pipeline.py`) — cold (eval runs per-checkpoint, not training hot-path). S3 (`hexo_rl/selfplay/pool.py` `game_complete` event) — cold (per-game terminal emit, not per-position). S4-S6 docs-only. **No bench-gate-skip violation.**

---

## (iv) Risk register completeness — **CLEAR** (with one strengthening note)

Plan Section 4 has 7 rows (≥5 floor). Cross-check against audit checklist:

| Audit-required risk | Plan row | Cited |
|---|---|---|
| Wrapper-fallback distortion (438 sentinel) | Row 1 | A1 §e + `summary.md:118` |
| Source A corpus weighting degradation | Row 2 | A4 §b L1 + L15 + ratings.csv |
| MCTSBot weights never arrive | Row 3 | A1 §b + §g |
| POC threshold moving-goalpost | Row 4 | A4 §e #9 + L12 + A3 §c |
| Source B GIL re-introduction | Row 5 | A4 §e #1 + sprint log line 597 |
| Plan author mis-budgeting LOC | Row 7 | A4 §b L17 + sprint log line 641 |
| Operator's qualitative cluster intuition | Row 6 | A3 §d + §verdict |

**Coverage gaps identified:**

1. **SealBot colony bug risk** (per `docs/rules/bot-integration.md:34`) — could distort sealbot's colony measurement at 35.0% in V3. NOT in register. The bug is documented as "colony-bug risk" upstream. V3 PASS leans on sealbot colony 35.0% > kraken_strong 7.1%; if sealbot's colony numbers are inflated by the upstream bug, V3 modified-PASS narrows. Operational impact LOW for this sprint (POC is `n_components`, not `colony_fraction`), but should be cited as a caveat.
2. **v6 → v6w25 cross-encoding drift** — Source A target is unstated re: encoding. If Source A games are stored in v6 encoding and downstream sustained training uses v6w25, encoding-header mismatch (`persist.rs` HEXB v7 per L10) blocks loads. Plan does not pin Source A target encoding. LOW risk (S4 design-only; encoding pin can be added in S4 spec) but un-anchored.
3. **KrakenBot upstream changes breaking wrappers** — vendor submodule pinned at `d9c5bfb` ("better eval"). If `vendor/bots/krakenbot` advances upstream and someone runs `git submodule update --remote`, wrappers could break silently. NOT in register.
4. **Bench-gate skip verification** — every S# claims "cold path" without per-file role grounding in the register. Verified above in (iii) — all cold. NOT a gap, but Section 4 could document the per-S# cold-path rationale.

**Verdict.** 7 rows cited and empirically anchored. Three additional caveats (SealBot colony bug, encoding pin for Source A, vendor submodule pinning) are LOW-priority strengthening notes. Plan PASSES the ≥5 floor; gaps do not block sign-off but should be addressed in S4/S5 design docs and Section 4 expansion.

**CLEAR — non-blocking strengthening notes.**

---

## (v) Commit-boundary discipline — **CLEAR**

Plan Section 6 commit table:

| # | S# | Single concern? | ≤10 cap |
|---|---|---|---|
| 1 | S1 | feat(bots) — wrappers + fallback + tests (S1 atomic) | Y |
| 2 | S2 | feat(eval) — bots + Q14 close (S2 atomic; plan flags optional 2a/2b split if Q14 grows) | Y |
| 3 | S3 | feat(selfplay) — n_components emit (S3 atomic) | Y |
| 4 | S4 | design(corpus) — Source A markdown | Y |
| 5 | S5 | design(corpus) — Source B markdown | Y |
| 6 | S6 | docs(sprint) — close-out + Q14 + lessons | Y |

Walk:

- C1 (S1): Pure implementation. Wrappers + tests + smart fallback. **Single concern.** Could arguably split into wrapper-only commit + tests-only commit but L17 ("not a one-liner") justifies bundling.
- C2 (S2): "feat(eval) + Q14 close" — Q14 close is a doc edit triggered by eval integration. **Borderline mixed concern** (implementation + open-question close). Plan acknowledges this at §481-486 by offering 2a/2b split. The Q14 close text is ~10 LOC; bundle adequate. **Acceptable.**
- C3 (S3): Pure implementation. `pool.py` emit + schema test. **Single concern.**
- C4 (S4) / C5 (S5) / C6 (S6): Doc-only. Each one design / close-out. **Single concern each.**

**Total 6 commits, headroom of 4 against the ≤10 cap.** Master prompt cap respected.

L13 origin (subagent / pre-registered pass criteria): Wave D plan IS pre-registered; Wave E (this audit) is the fresh-context review. **Complies.**

Compare to §176 refactor cycle (sprint log line 1218): 86 commits, 6 phases — sprint log notes "ranges per concern" not "single commit per concern" — i.e. discipline is single-concern-per-commit, not single-phase-per-commit. Plan's 6 commits fit single-concern discipline.

**No commit-boundary violation.**

---

## Final overall verdict — **CLEAR**

Plan ready for operator sign-off, with three OPTIONAL strengthening notes:

1. **Section 4 add row: SealBot upstream colony-bug risk** — affects V3 framing for downstream Source A weighting. Cite `docs/rules/bot-integration.md:34`. (LOW priority, NOT BLOCKING.)
2. **S4 design doc must pin Source A target encoding** — to avoid cross-encoding drift (L10, persist.rs HEXB v7 header). Cite §172 / §173 encoding registry.
3. **Section 4 add row: vendor submodule pinning** — `vendor/bots/krakenbot @ d9c5bfb`. Document the SHA pin to prevent silent breakage on `submodule update --remote`.

None of these block §176 Phase B. They are pre-S1 hygiene wires worth landing in the close-out commit (S6) or in the relevant design doc.

---

## Per-dimension verdicts

| Dim | Verdict | Notes |
|---|---|---|
| (i) Empirical evidence | CLEAR | 12/12 sampled claims pass; mix-ratio Elo-derived consistent with BT ladder |
| (ii) Falsified Register | CLEAR | All 16 rows walked; no re-litigation |
| (iii) Do-not lists | CLEAR | A4 15-item PASS; master 10-item PASS; cold-path bench skip verified per-S# |
| (iv) Risk register | CLEAR | 7 rows ≥ 5 floor; 3 strengthening notes (not blocking) |
| (v) Commit boundary | CLEAR | 6 commits ≤ 10 cap; single-concern per commit; C2 mixed-but-acceptable |

**Audit dispatch:** Wave D plan is ACCEPTABLE for operator sign-off. No re-spin required. Strengthening notes for Section 4 + S4 to land at sprint close.
