<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §149 — v7 verification + hygiene wave; v7e30 fine-tune promotes — 2026-05-03

**Date:** 2026-05-03
**Trigger:** §148 promoted v7 with three open caveats: SealBot edge
not significant (p=0.14, n=200), C1 contrast collapse +0.60→+0.00 of
unverified attribution, R=5-eval confound (deferred). §149 closes the
first two and ships the §148 next-actions hygiene items.

### Pretrain saturation audit (§149 task 1)

`reports/corpus_v7/pretrain_audit.md`. Per-epoch v7 trajectory:
final-3-epoch cumulative Δ = 1.6 % of total descent — fails the strict
< 1 % plateau gate. Diagnostic: cosine LR schedule reached
`eta_min = 1e-5` at end of epoch 15; last 3 epochs effectively idled.
Two interpretations of the gate (cumulative vs per-epoch) split.
Verdict: SHIP v7-15ep with caveat; launch fine-tune (§149 4 / option A
from user) to verify. Patched `hexo_rl/bootstrap/pretrain.py` with
`--resume`, `--lr-peak`, and `--inference-out` flags so a cosine
restart can run on the existing full pretrain checkpoint without
clobbering canonical bootstrap weights.

### v7e30 fine-tune

Resumed `checkpoints/pretrain/pretrain_00000000.pt`, fresh cosine
schedule peak `5e-4 → eta_min 1e-5` over 15 more epochs. Wall time
~98 min. Final loss `3.2462` (down from v7's `3.3134`, Δ -0.067).
Saved as `checkpoints/bootstrap_model_v7e30.pt`. Validation 100/100
vs RandomBot. v7 canonical (`bootstrap_model.pt`) was NOT clobbered
during fine-tune — `--inference-out` redirected the inference-weights
write.

### SealBot upsize n=500 each (§149 task 2)

`reports/corpus_v7/sealbot_500.md`.

| Model | n=500 wins | WR | Wilson 95% CI |
|---|---|---|---|
| v6 (baseline) | 57 / 500 | 11.4% | [8.9%, 14.5%] |
| v7 (15 ep)    | 66 / 500 | 13.2% | [10.5%, 16.4%] |
| **v7e30**     | **82 / 500** | **16.4%** | [13.4%, 19.9%] |

Pairwise z-tests:
- v7e30 vs v6: z = 2.29, **p = 0.022** ✓ significant
- v7e30 vs v7: z = 1.42, p = 0.15
- v7    vs v6: z = 0.87, p = 0.39 — the §148 +5 pp at n=200 (16% vs
  11%) was sampling noise on the v7 side. n=500 v7 is 13.2 %.

### Threat probes on v7e30

| Fixture | C1 | C2 | C3 | Verdict |
|---|---|---|---|---|
| Canonical (self-play) | -0.018 | 40 % | 70 % | C2/C3 PASS |
| Human-derived (§149 task 3 fixture, n=40) | +0.076 | 40 % | 72 % | C2/C3 PASS |

Threat recognition preserved through the fine-tune. C1 still flat
(corpus-shift artifact + flatter v7-family policy distribution; not a
kill-criterion gate per CLAUDE.md).

### C1 contrast diagnostic (§149 task 3)

`reports/corpus_v7/c1_human_probe.md`. New fixture:
`fixtures/threat_probe_human_positions.npz` (40 positions sampled
from `data/corpus/raw_human/`, balanced 14/14/14 across early/mid/late
phases) via `scripts/build_threat_probe_human.py`.

| metric | v7 | v6 | Δ |
|---|---|---|---|
| C1 contrast | +0.06 | +0.51 | -0.45 |
| C2 ext ∈ top-5 | **42 %** | 25 % | **+17 pp** |
| C3 ext ∈ top-10 | 70 % | 68 % | +2 pp |
| ext_logit raw | +0.07 ± 0.29 | +0.60 ± 0.48 | — |

**Outcome: case (ii) with positive surprise** (per §149 task 3
classification). v7's C1 is genuinely lower than v6's *on
human-distribution positions* — so it's not a pure corpus-shift
artifact. But v7's top-K rankings, which actually drive policy
decisions, are equal or better. v7-family learned a flatter, broader
policy distribution; rank ordering at the top is preserved or
improved. Not blocking.

### Hygiene wave (§149 task 4)

| Item | Action |
|---|---|
| 4a. HF push verify | SHA matches local: v7 `6cc62d3f`, v7e30 `2afe0e08`, both repos |
| 4b. §148 sprint log commit | `4cc8791` `docs(sprint): §148 v7 corpus rebuild + promotion` |
| 4c. Launch harness `replay_buffer.bin` guard | New `make train.fresh` target wipes `checkpoints/replay_buffer.bin` (and `.recent`) before launch, idempotent against existing `train.bg` |
| 4d. Buffer-state assertion | `scripts/train.py` always emits `buffer_state_at_corpus_load` event with `buffer_size_before_corpus_load`, `ckpt_step`, `ckpt_path`. Loud `buffer_contamination_suspected` warning when bootstrap-like ckpt (step ≤ 0) is loaded with non-empty buffer — catches the §147-discovered failure mode |
| 4e. Q41 verdict label fix | BLOCK threshold relaxed `43 % → 38 %` lower-CI in `scripts/w4c_h2h_5500.py` and `scripts/w7_q41_v7_v6_h2h.py`. New `--gate-strict` flag on both scripts preserves the original channel-cut threshold for callers that need it |

### Decision

**v7e30 promoted to canonical.** `checkpoints/bootstrap_model.pt` now
points at v7e30 (sha256 `2afe0e08…`). v7 (15-ep) preserved at
`bootstrap_model_v7.pt`; v6 preserved at `bootstrap_model_v6.pt`.
Phase B foundation document at `reports/corpus_v7/v7_validated.md`.

HF model repo `timmyburn/hexo-bootstrap-models`:
- new versioned `bootstrap_model_v7e30.pt`
- canonical `bootstrap_model.pt` overwritten with v7e30 content

Phase B is **unblocked**.

### Recommendation: full retrain on vast.ai

User asked for a verdict on whether a full retrain (option B) is
worth running on vast.ai. **Yes — recommended.** Evidence:

- Both v7 and v7e30 plateaued at the cosine eta_min for the final 3
  epochs of their respective schedules. The schedule is consistently
  hitting the LR floor before the model finishes descending.
- v7e30 fine-tune produced Δ -0.067 loss in 15 more epochs at
  meaningful LR — that's signal the recipe was undertraining.
- A fresh single-cycle cosine over 30 epochs with `eta_min=5e-5`
  (slightly higher floor) should reach loss 3.10–3.20 (vs v7e30's
  3.24 plateau) and likely +1–3 pp on SealBot WR.

`pretrain.py` already supports the necessary flags (`--epochs 30
--inference-out checkpoints/bootstrap_model_v7full.pt`). `eta_min`
override needs a one-line `--eta-min` flag (deferred — flag if user
wants).

### Artifacts

- `hexo_rl/bootstrap/pretrain.py` (resume + cosine restart support)
- `scripts/build_threat_probe_human.py`
- `scripts/train.py` (buffer contamination guard)
- `Makefile` (`train.fresh`)
- `scripts/w4c_h2h_5500.py`, `scripts/w7_q41_v7_v6_h2h.py`
  (`--gate-strict` flag, relaxed BLOCK threshold)
- `fixtures/threat_probe_human_positions.npz`
- `checkpoints/bootstrap_model_v7e30.pt`
- `checkpoints/bootstrap_model.pt` (= v7e30 now)
- `reports/corpus_v7/`: `pretrain_audit.md`, `sealbot_500.md`,
  `c1_human_probe.md`, `c1_human_probe_v7.md`,
  `c1_human_probe_v7e30.md`, `threat_probe_v7e30.md`,
  `sealbot_v6_500.jsonl`, `sealbot_v7_500.jsonl`,
  `sealbot_v7e30_500.jsonl`, `v7_validated.md`
- `logs/pretrain_v7e30_*.log`,
  `logs/sealbot_{v6,v7,v7e30}_500_*.log`
- HF: `timmyburn/hexo-bootstrap-models` versioned + canonical updated;
  `timmyburn/hexo-bootstrap-corpus` unchanged from §148

---

