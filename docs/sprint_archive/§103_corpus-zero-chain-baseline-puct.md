<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §103 — Corpus zero-chain fix + baseline_puct playout-cap pin (2026-04-17)

Two drift bugs surfaced by the post-wave-1 audits
(`reports/chain_parity_audit_2026-04-18.md` §4 and
`reports/selective_policy_audit_2026-04-18.md` §4 B2), independent fixes,
landed as separate commits. Log entry numbered §103 because §102 (the same-day
benchmark rebaseline) is already claimed; the commits retain the `§102.a` /
`§102.b` labels from the prompt that drove them.

### §103.a — Corpus chain target was zero post-§97

`batch_assembly.load_pretrained_buffer` padded corpus chain planes with
`np.zeros((T, 6, 19, 19))`. `compute_chain_loss` ran over the full batch
including corpus rows, so the chain head was pulled toward zero on the
pretrain fraction of every mixed step since §97 (2026-04-16). Silent — no
crash, no dashboard signal.

**Fix.** Compute chain planes from the stored stone planes at NPZ load via
`_compute_chain_planes(pre_states[:, 0], pre_states[:, 8])`. Route the /6
normalisation through float32 before the final f16 cast so the stored f16
bits match Rust `encode_chain_planes → f16` byte-exactly (the F2 guard only
pins the underlying int8 planes; this path pins the post-normalisation f16
values used by the self-play buffer).

**Regression.** `tests/test_corpus_chain_target.py` — two cases:

- `test_corpus_chain_planes_match_rust_byte_exact` — hand-built corpus NPZ
  round-trips through `load_pretrained_buffer → buffer.sample_batch`, matches
  `engine.compute_chain_planes` byte-exact at f16.
- `test_mixed_batch_chain_loss_uses_nonzero_corpus_targets` — 4 corpus + 4
  self-play rows → chain loss is finite and strictly positive on both
  halves, pinning that corpus targets are no longer zero.

Docstring at `trainer.py:420-427` updated to drop the stale "no
pretrain/selfplay divergence" language.

### §103.b — baseline_puct inherited selective loss

`configs/variants/baseline_puct.yaml` had no `playout_cap` override, so
post-§100 it inherited `full_search_prob: 0.25` from the base — turning a
"PRE-§67 HISTORICAL BASELINE" variant into a §100-selective run and
silently confounding any ablation using it as an unmodified control.

**Fix.** Pin `playout_cap.full_search_prob: 0.0` explicitly in
`baseline_puct.yaml`. Game-level `fast_prob` was already 0.0 via base
inheritance, so both playout caps are now OFF for this variant.

**Regression.** `tests/test_variant_configs.py::test_baseline_puct_pins_pre_100_semantics`
runs the same deep-merge path as `scripts/train.py --variant` and asserts
both `full_search_prob` and `fast_prob` resolve to 0.0.

### Commits

- `fix(training): compute corpus chain planes at NPZ load (§102.a)`
- `fix(config): pin baseline_puct full_search_prob=0.0 (§102.b)`
- `docs(sprint): §103 corpus zero-chain + baseline_puct pin`

