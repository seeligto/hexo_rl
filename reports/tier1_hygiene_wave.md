# Tier-1 hygiene wave — close-out report

**Date:** 2026-05-18
**Branch:** `phase4.5/tier1_hygiene` (off master `96337c4`)
**Scope:** 5 isolated hygiene items dispatched as a parallel wave; zero-overlap touches.
**§S178 active run:** untouched throughout (sustained tmux `sS178` LIVE on vast.ai).

## Wave summary

| Item | Scope | Commit | REVIEW |
|---|---|---|---|
| A | quarantine fresh-init `bootstrap_model.pt`; repoint canonical default to `bootstrap_model_v6.pt` | `c1173a8` | PASS |
| B | `tests/test_scraper.py` restore after `elo_band_key` deletion | NO-OP (commit none) | PASS |
| C | new `scripts/verify_anchor.py` anchor verification CLI | `2740766` | PASS |
| D | split CLAUDE.md to ≤200 lines via `docs/rules/` extraction | NO-OP (commit none) | PASS |
| E | variant config audit + cleanup of non-active variants | `ba4f1e7` | PASS |

5 REVIEWs PASS; full `make test` on branch HEAD = 1588 passed, 21 skipped, 4 deselected, 1 xpassed, exit 0 (2m 35s).

## Per-item detail

### A — `chore(anchors): quarantine fresh-init bootstrap_model.pt`

Commit `c1173a8`. Touch surface (4 files):

- `Makefile` lines 11 + 47 → `bootstrap_model_v6.pt`.
- `hexo_rl/training/anchor.py` `_BOOTSTRAP_ANCHOR_CANDIDATES` → `(v6.pt, v7full.pt)`; random-init candidate removed; `v6.pt` promoted to first.
- `hexo_rl/eval/opponent_runners.py:217` `bootstrap_anchor` default → `bootstrap_model_v6.pt`.
- `scripts/migrations/2026_05_12_checkpoint_manifest.json` — dropped the `"checkpoints/bootstrap_model.pt": "v6"` row.

Filesystem (gitignored): `checkpoints/bootstrap_model.pt` → `checkpoints/archive_quarantine/bootstrap_model_random_init_v6w25.pt`. SHA `d00b8604…1586253` matches `reports/bootstrap_model_pt_provenance.md` §6.

REVIEW verdict: PASS — scope verified (4-file diff), filesystem confirmed, active-path scan clean, targeted tests `tests/test_eval_anchor_loader.py + test_eval_opponent_runners.py + test_preflight.py` 9 passed / 1 skipped in 3.02s. Closes sprint-log §S178 open hygiene item **H1**.

### B — `tests/test_scraper.py` restore (NO-OP)

REVIEW surfaced a nuance that the IMPL agent missed:

- The tracked path `tests/test_scraper.py` was never under git on the local tree. The cited deletion commits (`7aea774`, `913c5a0`) removed `python/bootstrap/scraper.py` and `scripts/update_manifest.py`, but no tracked test file under that name.
- A **gitignored** local copy of `tests/test_scraper.py` (222 LOC, kept local per robots.txt policy per `.gitignore:58`) DOES exist on this worktree, imports `hexo_rl.bootstrap.scraper`, and runs **19/19 PASS in 0.96s** in venv — no broken `elo_band_key` reference.
- The wave-driver's premise (broken test, deleted helper) was sourced from a vast-side bench-gate report describing a different gitignored file. That vast-side discrepancy is a separate operational followup.

Wave-level verdict: NO-OP is correct — nothing to fix on this branch.

### C — `feat(scripts): add anchor verification utility`

Commit `2740766`. New file: `scripts/verify_anchor.py` (254 LOC).

CLI verifies a checkpoint and writes a `<ckpt>.verify.json` sidecar with `sha256` (16 hex), `format`, `key_count`, `param_count`, `head_shapes` (policy/value/aux), `fresh_init_signature` (bool), `verdict` (TRAINED | FRESH_INIT_SUSPECT | UNKNOWN).

Heuristic per `reports/bootstrap_model_pt_provenance.md` §3:
`value_fc2.weight.abs().max() / sqrt(1/fan_in)` within `[0.8, 1.25]` → FRESH_INIT_SUSPECT.

Exit codes 0 / 1 / 2.

Verified end-to-end:
- `bootstrap_model_v6.pt` → exit 0 TRAINED (ratio 2.013, SHA `7ab77d2c…`).
- `archive_quarantine/bootstrap_model_random_init_v6w25.pt` → exit 1 FRESH_INIT_SUSPECT (ratio 0.999, SHA `d00b8604…`).
- `/nonexistent` → exit 2 UNKNOWN.

REVIEW verdict: PASS — scope verified (single new file), heuristic spot-checked manually (`sqrt(1/256) = 0.0625` matches sidecar), no active-anchor mutation.

### D — CLAUDE.md split (NO-OP)

`wc -l CLAUDE.md` = 105 — well below the 200-line target. All 8 `docs/rules/` files exist (`background-tasks.md`, `board-representation.md`, `bot-integration.md`, `build-commands.md`, `checkpoint-archive-policy.md`, `perf-targets.md`, `phase-4-architecture.md`, `workflow.md`). CLAUDE.md is index-shaped (Prime Directive + Threat-probe + Encoding registry + Rule files index + Deep-dive index + MCP tools); no fat content to extract.

REVIEW verdict: PASS — no-op decision correct; no followup hygiene defect surfaced.

### E — `chore(configs): variant hygiene audit + cleanup`

Commit `ba4f1e7`. Touch surface (3 files):

- `configs/variants/v6_sustained.yaml` cleaned (§175 closed-by-interrupt; 11/12 base-equal scalars dropped).
- `configs/variants/v7mw_sustained.yaml` cleaned (§176a experimental v7mw; 11/12 base-equal scalars dropped).
- `reports/variant_hygiene_audit.md` (new, 306 LOC) — per-variant noise/override/extension tables.

Active variant `configs/variants/v6_botmix_s178.yaml` untouched (verified `git show ba4f1e7 -- configs/variants/v6_botmix_s178.yaml` empty).

Deferred audit-only with reasoning:

- `m173_alpha_cold_smoke.yaml` (§173 A8 design doc; pre-registered gates retained for sprint trace).
- `smoke_radius_curriculum.yaml` (§174 cohort; structural parity with vast.yaml aids sprint log).
- `_sweep_template.yaml` (docstring-pinned).
- `v6_sustained_s177.yaml` (§S178 A/B contrast variant; clean after §S178 closeout).
- `vast.yaml` (operator-designated exemplar; 17 noise keys intentional as single-file selfplay knob reference).

REVIEW verdict: PASS — active variant untouched; both cleaned variants parse; all 22+ removed keys spot-checked against base values; `tests/test_variant_configs.py` 5/5 PASS; `engine/src/` `#[allow` count = 29 (cycle-3 close baseline preserved).

## Commit plan (executed on branch)

Final ordering on `phase4.5/tier1_hygiene` (post-cherry-pick from worktrees, parent = master `96337c4`):

```
ba4f1e7 chore(configs): variant hygiene audit + cleanup        (E)
2740766 feat(scripts): add anchor verification utility         (C)
c1173a8 chore(anchors): quarantine fresh-init bootstrap_model.pt (A)
96337c4 (master) fix(selfplay): disable cosine-temp default …
```

A → C → E in commit order; B + D no-op (no commit). Aggregation commits the docs deliverables (this report + §S178a sprint log + Q-§S178a followups) on top before merge.

## Merge plan (pending operator)

1. Operator inspects this report (`reports/tier1_hygiene_wave.md`).
2. Operator runs final `make test` on `phase4.5/tier1_hygiene` HEAD (parent already ran exit 0 at branch HEAD before docs commits).
3. Operator FF-merges to master: `git checkout master && git merge --ff-only phase4.5/tier1_hygiene`.
4. Operator pushes master.
5. No tag (hygiene wave, not cycle close).
6. §S178 active run on vast untouched throughout — no anchor / corpus / variant modifications.

## Followups

Captured in `docs/06_OPEN_QUESTIONS.md` as **Q-§S178a — Tier-1 hygiene wave follow-ups**. Summary:

- **F-A1 [LOW]** — 4 residual `bootstrap_model.pt` references outside IMPL-A's strict touch list: `hexo_rl/bootstrap/pretrain_trainer.py:302`, `scripts/smoke_selfplay_gumbel.py:37`, `scripts/tournament_validate.py:598`, `Makefile` probe.bootstrap / probe.latest targets. None are silent `make train` defaults (those are closed by IMPL-A); these are output-path strings, CLI defaults, or probe-target paths.
- **F-B1 [MED]** — vast-side `tests/test_scraper.py` discrepancy: gitignored file on vast (not under git anywhere) imports the deleted `scripts.update_manifest.elo_band_key`. Local copy is healthy. Operationally: regenerate vast's gitignored test file OR clean `scripts/scrape_daily.sh:44` reference to deleted `update_manifest.py`.
- **F-E1 [LOW]** — clean `configs/variants/v6_sustained_s177.yaml` symmetrically with v6_sustained.yaml after §S178 closeout (currently the §S178 A/B contrast variant; cleanup deferred until contrast is no longer relevant).
- **F-E2 [LOW]** — fold `_sweep_template.yaml` + `m173_alpha_cold_smoke.yaml` + `smoke_radius_curriculum.yaml` cleanup into a §173/§174 retrospective sweep (deferred to preserve sprint-trace structural parity).

## Verification trail

- `make test` exit 0 on branch HEAD (1588 passed; log at `/tmp/wave_make_test.log`).
- 5 independent REVIEW subagents (no IMPL context, fresh per-agent) each returned PASS with concrete evidence (diff, scope, scope-targeted tests, active-path scan, allow-count baseline).
- No `#[allow]` regression: `rg '#\[allow' engine/src | wc -l` = 29 (cycle-3 close baseline).
- No hot-path touch (engine/, hexo_rl/selfplay/, hexo_rl/training/loop.py main flow).
- Active-run paths confirmed untouched in every diff: `configs/variants/v6_botmix_s178.yaml`, `checkpoints/bootstrap_model_v6.pt`, `data/bot_corpus_s178_sealbot_vs_v6.npz`, `logs/sS178_*`.
