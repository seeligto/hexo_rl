# docs/archive — relocation index

Old→new map for the D-DOCS-DEBLOAT relocation (2026-06-23, docs-only, reversible).
Every relocated doc is reachable from here. Falsified-register rows from split-out
phases are consolidated into `docs/07_PHASE4_SPRINT_LOG.md` § "Falsified Hypotheses
Register" (live), with full context preserved verbatim in the archived bodies below.

---

## Sprint-log split — `docs/07_PHASE4_SPRINT_LOG.md` (6720 → 2326 lines)

The sprint log is now a lean INDEX (Parts 1–3 + register + mechanism lessons + key
config + classification tables + memory map + §178/§179/§180 refactor-cycle distilled
inline + the 4 CURRENT phases inline). Closed S-series + D-series phase bodies relocated
to `docs/sprint_archive/` (the existing per-§ archive convention; NOT a new `docs/sprint_log/`).

| old span (orig line) | new file | scope |
|---|---|---|
| §176 + §177 + Supplementary tables (1251–1729) | `docs/sprint_archive/§S176-S177_sustained_recipe.md` | §176 Python/PhaseA + §177 recipe-attractor |
| §S178/§S178a/§S179/§S180a/§S180b (2129–2599) | `docs/sprint_archive/§S178-S180_botmix_colony.md` | bot-mix launch + colony-lever closes |
| §S181 + §S181-AUDIT W1–W5 (2600–3906) | `docs/sprint_archive/§S181_structural_audit.md` | structural diagnosis + audit waves |
| §S182–§S186 (3907–4139) | `docs/sprint_archive/§S182-S186_perf_arc.md` | perf wave arc |
| §P5-CT + §P-INF (4140–4474) | `docs/sprint_archive/§P5-CT_PINF_probes.md` | compound-turn probe + inference attribution |
| §P6/§O1/§PRELONG + §D-WALLCAUSATION…§D-STRENGTHAXIS (4588–5837) | `docs/sprint_archive/§D_diagnosis_eval_arc.md` | diagnosis & eval-foundation arc |
| §D-MULTICLUSTER + §D-MULTICLUSTER-S0 (5838–5978) | `docs/sprint_archive/§D_multicluster.md` | 362-multiwindow legal-set |
| §D-VALPROBE…§D-RERUNPREP + §D-TEMPDECAY (5979–6341) | `docs/sprint_archive/§D_loopfix_tempdecay.md` | loop-fix + temp-decay |

KEPT INLINE in the index (NOT split): Parts 1–3, central falsified register + consolidated
additions, benchmark baseline/evolution, mechanism lessons, key config, §66–§174 distilled
blocks, §178/§179/§180 refactor-cycle bodies (their forensics already in `§178/§179/§180_rust_engine_audit.md`),
classification tables, memory map, and the 4 current phases (§D-TEMPSTRENGTH, §D-GUMBELSIMS,
§D-LONGRUN-READY, §D-LONGRUN-C).

---

## Handoffs — `docs/handoffs/` → `docs/archive/handoffs/`

13 tracked handoffs relocated (live 1M-Gumbel run has no written handoff; its state lives
in sprint-log §D-LONGRUN-C + `scripts/d1m_*`). Class + superseded-by:

| old | new | superseded_by |
|---|---|---|
| longrun_phase_b_mgate_runbook.md | 20260621_longrun_phase_b_mgate_runbook.md | §D-LONGRUN-C (m=16 selected, 200k run closed) |
| armc_rerun_launch_package.md | 20260612_armc_rerun_launch_package.md | armc_50k_run_status + §D-GUMBELSIMS |
| armc_rerun_watchsheet.md | 20260612_armc_rerun_watchsheet.md | armc_50k_run_status |
| loopfix_armc_rerun_runbook.md | 20260612_loopfix_armc_rerun_runbook.md | §D-RERUNPREP + §D-LOOPFIX |
| phase3_smoke_results.md | 20260612_phase3_smoke_results.md | §D-RERUNPREP green |
| phase3_finding_terminal_eval_completed.md | 20260612_phase3_finding_terminal_eval_completed.md | §D-RERUNPREP |
| tempdecay_report.md | 20260613_tempdecay_report.md | §D-TEMPDECAY + §D-TEMPSTRENGTH |
| tempdecay_probe_runbook.md | 20260612_tempdecay_probe_runbook.md | tempdecay_report |
| dexploit_handoff_20260606.md | 20260606_dexploit_handoff.md | §D-EXTLINK + §D-MULTICLUSTER |
| exploit_probe_20260606.md | 20260606_exploit_probe.md | §D-EXTLINK |
| finishing_discriminator_20260606.md | 20260606_finishing_discriminator.md | §D-ARGMAX / §D-EXTLINK |
| v7full_selfplay_baseline.md | 20260504_v7full_selfplay_baseline.md | §150 promote + §173 encoding |
| W1D_q49_dirichlet_rotation.md | 20260502_W1D_q49_dirichlet_rotation.md | §136 W1/W2 audit |

KEPT at `docs/handoffs/` (live / load-bearing): `longrun_phase_c_launch_spec.md` (invariants
govern live 1M run), `s176_phase_b_prompt.md` (CLAUDE.md "retained for follow-up").

LEFT IN PLACE — referenced by CODE or load-bearing rules (moving would orphan a reference
we cannot fix in docs-only scope): `exploit_probe_design_20260606.md` (← `scripts/exploit_probe.py`),
`compound_turn_pipeline_audit.md` (← `docs/rules/phase-4-architecture.md` + `board-representation.md`).

UNTRACKED, LEFT IN PLACE (cannot `git mv`; per operator decision): `armc_50k_run_status.md`,
`armc_fixed_loop_rerun_runbook.md`, `gumbel_ab_runbook.md`, `gumbel_vs_puct_research_prompt.md`,
`longrun_a1_vast_consolidation_runbook.md`, `longrun_a2_bootstrap_pretrain_spec.md`.

---

## Findings — `docs/perf/` → `docs/archive/findings/perf_supply_wave_2026-04/`

8 files from the closed supply-side perf wave (2026-04-22, landed). `instrumentation_notes.md`
LEFT in `docs/perf/` (← code comments in `trainer.py:386`, `inference_server.py:107`).

| old | new |
|---|---|
| docs/perf/recommendations.md | recommendations.md |
| docs/perf/supply_wave_report.md | supply_wave_report.md |
| docs/perf/static_audit.md | static_audit.md |
| docs/perf/20k_gumbel_targets_characterization.md | 20k_gumbel_targets_characterization.md |
| docs/perf/diagnostic_run_C1.md | diagnostic_run_C1.md |
| docs/perf/post_merge_validation.md | post_merge_validation.md |
| docs/perf/stream_separation_spike.md | stream_separation_spike.md |
| docs/perf/vram_headroom.md | vram_headroom.md |

---

## Repoints applied (live-doc references → new archived paths)

- `docs/06_OPEN_QUESTIONS.md:16` → `docs/archive/handoffs/20260502_W1D_q49_dirichlet_rotation.md`
- `docs/06_OPEN_QUESTIONS.md:213` → `docs/archive/findings/perf_supply_wave_2026-04/supply_wave_report.md`
- `docs/designs/dexploit_multicluster_repretrain_design.md:4` → `docs/archive/handoffs/20260606_exploit_probe.md`
- `docs/sprint_archive/§158-§158b_hygiene-wave-2026-05-06.md` (×2) → `docs/archive/findings/perf_supply_wave_2026-04/static_audit.md`

---

## `reports/` — PROPOSE-ONLY (gitignored; NOT relocated, per operator decision)

`reports/` is gitignored, so `git mv` is impossible and operator convention protects it.
~98 top-level dirs are closed-family forensics tied to shipped sprint §s (s174/s176/s178/s181/
§S180-186/q-series/encoding/perf/w-series/gpool); ~37 are live `phase4.5/gumbelprep`/longrun work
(gumbel*, thr_*, tempstrength*, p3_*, phase_b_mgate, eval, investigations, probes). This manifest
record stands as the inventory; no filesystem relocation performed. (Full per-dir classification:
Phase-1 Bucket-B manifest.)

---

## Pre-existing drift FIXED (follow-up commit)

The sprint-log §Index table (106 refs), the §179/§180 refactor-cycle cross-pointers, and the
§S181 handoff prompt pointed §66–§174 forensics at `reports/sprint_archive/` (gitignored,
**empty**); the real per-§ files live in `docs/sprint_archive/`. Repointed
`reports/sprint_archive/` → `docs/sprint_archive/` across those docs. Exception left as-is:
`docs/archive/audit/structural/01_bootstrap_corpus_bias.md` uses `reports/sprint_archive/` to
describe *checkpoint* absence (a historical audit finding, different semantic — not a doc link).

---

## Reversibility

Tracked moves: `git mv` back (handoffs, perf). Sprint-log split: restore the monolith from
git history (the pre-split content is in git; the working-tree backup used during the split was
`scratchpad/SPRINTLOG_BACKUP.md`) and `rm` the 8 new `docs/sprint_archive/§{S,P,D}_*.md` files
(untracked). Repoints + CLAUDE.md link: single-line `git` reverts.
