<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §108 — Desktop post-W1 sustained launch `gumbel_full` (2026-04-19)

First post-W1 desktop sustained, companion to laptop `gumbel_targets` (Prompt 15). Answers Q2: does Gumbel SH add value beyond CE targets alone under identical W1 fix + Option A playout-cap repair + R1 anchor semantics.

**Config.** Host archstation (3700X + 3070). Variant `gumbel_full` (Gumbel root + completed-Q). Bootstrap `bootstrap_model.pt` (18-plane, GroupNorm(8), §93 v3b). 50k iters. Run `post_w1_desktop_gumbel_full_20260419`.

**Key decisions.** SDG rebased 4.0→2.0 (Option A removed per-game fast_prob, per-game cost up). A/B preflight skipped (analyzer not in tree, `make train.smoke` no override) — launched at 2.0 with hour-1 gate. Pre-W1 artifacts archived to `archive/prefix_desktop_20260419_154604/`; kept only `bootstrap_model.pt` + `best_model.pt`. `gumbel_full` mutex (`fast_prob: 0.0`) + regression test already landed. I1/I2 JSONL mirror landed mid-run (b35de20).

**Hour-1 telemetry.** policy_loss flat ~2.07, value_loss declining 0.548→0.530, pe_self contracting 3.82→3.13, 930 games/hr steady (vs estimated 130-180 — Option A cost was 1.16× not 2.5×), GPU 65-80%, SDG hit 2.000 exactly (trainer not starving), idle 27% = burst pattern not GPU idle, 1 pool overflow at startup with graceful fallback, 0 NaN.

**Status: RUNNING at launch.** Outcomes processed in §109+ (Q33 selfplay entropy diagnostic — `pe_self ≈ 5.35` fixed point flagged from this run's diag-20K report). See §109 for entropy verdict, §110 Q33-B for fixed-point confirmation, §111 Q33-C HALT on aug discriminator.

**Commits.**
- `config(variants): gumbel_full SDG 4.0 → 2.0 for post-Option-A launch` (299b4c0)
- `test(variants): update gumbel_full SDG pin 4.0 → 2.0` (a797abd)
- `feat(monitoring): mirror I1/I2 cluster + colony metrics to JSONL` (b35de20)
- `docs(sprint): §108 desktop gumbel_full sustained run launch`

---

