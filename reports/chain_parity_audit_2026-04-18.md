# Chain-plane parity audit — post-§97

Date: 2026-04-18
Auditor: parity guard audit
Scope: F2 guard (Python `_compute_chain_planes` ↔ Rust `encode_chain_planes`) after
§97 removed chain planes from the input tensor.

## §1 Guard status — Phase 1

`tests/test_chain_plane_rust_parity.py` — **PASS**, 21/21 positions byte-exact.

```
============================== 21 passed in 1.84s ==============================
```

Ancillary:

- `tests/test_chain_plane_augmentation.py` — **PASS**, 5/5. Byte-exact round-trip
  `ReplayBuffer.sample_batch(augment=True) ↔ _compute_chain_planes(transformed stones)`.

Conclusion: F2 guard is still active and meaningful. §97 did not modify either chain
implementation (`git log --oneline b06629a..HEAD -- engine/src/board/state.rs
hexo_rl/env/game_state.py` returns no commits).

## §2 Live target-path diagram (post-§97)

### Self-play training path

```
  engine/src/game_runner/worker_loop.rs:434
    encode_chain_planes(cur_view, opp_view)   ← Rust
      → records_vec (feat, chain, policy, …)
      → push_game → ReplayBuffer.chain_planes (u16 f16-bits)
      → sample_batch(..., augment=true) → apply_sym via sym_tables chain_src_lookup
      → Python 7-tuple (states, chain_planes, policies, outcomes, own, wl, ifs)
      → trainer._train_on_batch(chain_planes=…)
      → compute_chain_loss(chain_pred, chain_target)
```

### Cold-start pretrain corpus path (mixing in the RL loop)

```
  data/bootstrap_corpus.npz (18-plane, keys: states, policies, outcomes, weights)
    → hexo_rl/training/batch_assembly.py:131
        pre_chain = np.zeros((T, 6, 19, 19), dtype=np.float16)      ← ZEROS, not computed
    → pretrained_buffer.push_game(pre_states, pre_chain, …)
    → sample_batch(n_pre, True) — zeros are scatter-invariant
    → assemble_mixed_batch → trainer with n_pretrain=n_pre
    → chain_loss computed over ALL rows (trainer.py:425-427)
```

### Standalone bootstrap pretrain script (offline, not the RL loop)

```
  hexo_rl/bootstrap/dataset.py:59  /  hexo_rl/bootstrap/pretrain.py:213
    _compute_chain_planes(tensor[k, 0], tensor[k, 8])            ← Python
  → chain_planes fed as a separate collate-tuple element
```

### Test-only entry

```
  engine/src/lib.rs:659  engine.compute_chain_planes(cur, opp)   ← Rust PyO3 binding
    — only used by F2 guard
```

## §3 Gaps in byte-exact coverage

| Target-producing code                           | Implementation                | Covered? |
|-------------------------------------------------|-------------------------------|----------|
| worker_loop.rs `encode_chain_planes`            | Rust                          | F2 + aug |
| sample_batch chain-plane symmetry scatter       | Rust sym_tables               | aug-test |
| dataset.py / pretrain.py `_compute_chain_planes`| Python                        | F2       |
| load_pretrained_buffer `np.zeros` for 18ch NPZ  | N/A — not a chain computation | n/a      |

No implementation-level hole found. F2 + `test_chain_plane_augmentation.py` together
pin every live call site that produces a chain target via one of the two
implementations.

## §4 Drift flag — corpus zero-chain training-signal inconsistency

`load_pretrained_buffer` (batch_assembly.py:120-131) handles two cases:

- 24-plane legacy NPZ: extracts `pre_states[:, 18:24]` as the chain target.
- 18-plane current NPZ (the live one at `data/bootstrap_corpus.npz`, regenerated
  post-§97): pads `pre_chain = np.zeros((T, 6, 19, 19))`.

`trainer._train_on_batch` (trainer.py:420-427) computes `chain_loss` on **all** batch
rows. The docstring there states "target is deterministic from board stones, no
pretrain/selfplay divergence, so mask_aux_rows not needed" — this comment is stale
post-§97. For corpus rows the target is **zero**, not the board-deterministic
chain encoding, so chain loss pulls the chain head toward zero on the pretrain
fraction of every mixed batch.

This is NOT an F2-type parity bug (the two implementations are still byte-exact).
It is a **training-data drift** §97 introduced: two streams (self-play / corpus)
feed incompatible targets into the same aux head.

Severity assessment (first-order):

- Corpus fraction of the batch is `mixing.n_pre / batch_size`. Verify live ratio.
- Zero target on corpus rows biases chain head down uniformly → underestimates
  chain length on early positions resembling corpus density. Plausible policy-head
  side-effect if chain features are load-bearing in the trunk.
- Fast fix options (for a separate prompt):
  1. Compute chain planes from `pre_states[:, 0]` and `pre_states[:, 8]` during
     `load_pretrained_buffer` — same path `dataset.py:59` uses. One Rust-or-Python
     call per NPZ load; amortized, not per-step.
  2. Extend `mask_aux_rows` to `chain_loss` and drop the stale comment.

Recommendation: option (1). Option (2) discards the corpus aux signal entirely;
(1) preserves it and matches the standalone pretrain supervision.

**Not fixing here** per task scope. Follow-up prompt should own it.

## §5 Proposed guard additions

None. F2 + `test_chain_plane_augmentation.py` cover every live implementation-level
parity boundary. The corpus-zero-chain gap is a target-construction bug, not an
implementation-drift bug — it needs a behavioural test (e.g. "chain_loss on a
hand-built corpus batch matches chain_loss on the same batch passed through the
Rust self-play push path"), which belongs to the follow-up prompt that picks the
fix strategy.

## Deliverable summary

| Item                    | Status                                                      |
|-------------------------|-------------------------------------------------------------|
| F2 guard pass           | Yes — 21/21                                                  |
| Guard still meaningful  | Yes — both implementations remain live and byte-exact       |
| Implementation drift    | None found                                                   |
| Training-signal drift   | Flagged (§4) — out of scope for this pass                   |
| Code changes            | None                                                        |
| Test additions          | None                                                        |
