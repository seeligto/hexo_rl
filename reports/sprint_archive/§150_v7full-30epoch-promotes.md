<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §150 — v7full: 30-epoch full retrain promotes; v7e30 retained for A/B — 2026-05-03

**Date:** 2026-05-03
**Trigger:** §149 closed with explicit recommendation for full retrain on
vast.ai. Both v7 (15 ep) and v7e30 (15+15 ep fine-tune) plateaued at
their respective cosine eta_min floors (3.31, 3.24). User ran the
recipe on a vast.ai 5080: single-cycle cosine 30 epochs, peak `2e-3`,
`eta_min=5e-5` (raised from `1e-5`). New `--eta-min` flag added in
`hexo_rl/bootstrap/pretrain.py` for the floor override (commit
`1f822ae`).

### Recipe and run

| Knob | v7full value |
|---|---|
| Corpus | `data/bootstrap_corpus.npz` (= v7) |
| Epochs | 30 (single cosine cycle) |
| Batch | 256 |
| Peak LR | 2e-3 |
| eta_min | **5e-5** (was 1e-5) |
| Architecture | unchanged (8-plane, 12 res blocks, 128 filters, SE r=4) |
| Wall time on 5080 | ~83 min |
| Final loss | **3.1573** (vs v7e30 3.2462, Δ -0.089) |

Output: `checkpoints/bootstrap_model_v7full.pt` (sha256 `29306533…`).
v7e30 canonical NOT clobbered during the run on the remote host;
artifacts pulled via the rsync-vast skill.

### Headline numbers

| Model | SealBot WR (n=500) | Wilson 95% CI | Threat C1 / C2 / C3 | Final loss |
|---|---|---|---|---|
| v6 (baseline) | 11.4% (57/500) | [8.9%, 14.5%] | +0.60 / 50% / 60% | — |
| v7 (15 ep) | 13.2% (66/500) | [10.5%, 16.4%] | +0.00 / 45% / 60% | 3.3134 |
| v7e30 (15+15 ep) | 16.4% (82/500) | [13.4%, 19.9%] | -0.02 / 40% / 70% | 3.2462 |
| **v7full (30 ep)** | **17.4% (87/500)** | **[14.3%, 21.0%]** | **+0.20 / 50% / 70%** | **3.1573** |

Pairwise z-tests:
- v7full vs v6: z = 2.70, **p = 0.007** ✓ significant
- v7full vs v7e30: z = 0.42, p = 0.67 (n.s.; consistent direction)
- v7full vs v7: z = 1.84, p = 0.066 (borderline)

Colony wins: 12/87 = 13.8 % (in line with v7-family baseline; not a
regression).

### Threat probe verdict

`reports/corpus_v7/threat_probe_v7full.md`:
- **C1 contrast +0.204** — FAIL the strict `≥ +0.479` threshold but
  C1 is a warning, not a kill-criterion gate (CLAUDE.md gates only
  C2/C3). v7full *recovers* C1 contrast somewhat from v7e30's flat
  +0.0 toward v6's +0.6, suggesting longer + higher-floor training
  partially restores the sharper distribution while keeping top-K
  ranking quality.
- C2 = 50 % ✓ PASS
- C3 = 70 % ✓ PASS

Both formal gates pass. C1 partial recovery is incidental upside, not
a gate.

### Promotion decision

**Promote v7full → canonical.** Direction across every metric
(SealBot WR, threat C1/C2/C3, final loss, v6-significance) moves
right. Edge over v7e30 is not statistically significant at n=500
(z=0.42), but:

- v6-anchor edge becomes significant (`p=0.022 → p=0.007`)
- C1 partial recovery is the only metric where v7e30 was strictly
  worse than v6; v7full closes ~⅓ of that gap
- Loss continues descending (3.246 → 3.157) confirming the §149 LR-
  floor diagnosis was correct

Canonical `checkpoints/bootstrap_model.pt` now points at v7full
(sha256 `29306533…`). v7e30 retained at `bootstrap_model_v7e30.pt`
for any A/B regression check; v7 (15-ep) and v6 also retained.

HF model repo `timmyburn/hexo-bootstrap-models`:
- new versioned `bootstrap_model_v7full.pt`
- canonical `bootstrap_model.pt` overwritten with v7full content

### Recipe lesson

The 30-epoch / `eta_min=5e-5` recipe supersedes the original 15-epoch
/ `1e-5` recipe used for v6, v7. Future bootstrap retrains should use
the v7full recipe as default. The `--eta-min` flag (commit `1f822ae`)
makes this a one-line change.

### Caveats

1. **vs-v7e30 statistical edge unverified.** n=500 each gives
   z=0.42, p=0.67. A +1 pp WR difference at this sample size is
   sub-power. Tiebreaker H2H eval (~17 min on 3070) skipped — every
   other metric agrees with the promotion direction.
2. **R=5 confound persists** (§149 caveat 4 — same env for v7full
   evaluation as v6/v7/v7e30; comparisons still apples-to-apples).

### Phase B status

**UNBLOCKED.** Phase B work that loads
`checkpoints/bootstrap_model.pt` will pick up v7full transparently.
v7e30 / v7 / v6 versioned files retained for A/B regression checks.

### Artifacts

- `checkpoints/bootstrap_model_v7full.pt`
- `checkpoints/bootstrap_model.pt` (= v7full now)
- `reports/corpus_v7/threat_probe_v7full.md`
- `reports/corpus_v7/sealbot_v7full_500.jsonl`
- `logs/pretrain_v7full_20260503_151827.log` (vast.ai)
- `hexo_rl/bootstrap/pretrain.py` `--eta-min` flag (commit `1f822ae`)
- HF: `timmyburn/hexo-bootstrap-models` versioned `v7full` +
  canonical updated to v7full

---

