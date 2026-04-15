# feat: 18 → 24 input planes + Q13 chain features + Q19 threat pos_weight

Ships the Q13 chain-length input expansion as an atomic architectural break.
Three interlocking changes land together: 6 new per-hex-axis chain-length
planes on the NN input, a new chain-prediction auxiliary head, and the Q19
threat-head `pos_weight=59` class-imbalance fix. Pretrain v3b (regenerated
after the §93 F1 augmentation-path root-cause fix) is shipped as the new
`bootstrap_model.pt`. Tagged `v0.4.0` — minor-version bump because the NN
input tensor shape, replay buffer format, and pretrained checkpoint are all
incompatible with pre-`v0.4.0` artifacts.

---

## Linked tickets

- **Q13** (chain-length planes) — **resolved**
- **Q19** (threat-head BCE class imbalance) — **resolved**
- **Q21** (wider-window chain-aux target) — **parked** for post-baseline,
  see sprint log §92
- **Q22** (chain-plane Rust port) — **parked**, F2 parity guard pins Python
  and Rust paths together
- **Q23** (tensor assembler consolidation) — **closed** (C9.5 deleted the
  dead `TensorBuffer` second assembler)

See `docs/07_PHASE4_SPRINT_LOG.md` §92 + §93 for full design + audit trail.
Release notes: `docs/releases/v0.4.0.md`.

---

## Commits (19, `master..feat/q13-chain-planes`)

```
cd28431 docs(sprint): §93 Q13 fix-up + F1 root cause + F3 dead code + v3b (C17)
7812c0f chore(bootstrap): pretrain v3b from scratch + threat_probe_baseline v4 (C16)
5d02990 chore(naming): encode_planes_to_buffer rename + 18-plane doc cleanup (C15, W4)
55276be feat(dashboard): surface loss_chain / loss_ownership / loss_threat (C14, W3)
a899968 fix(loss): add optional legal_mask to compute_chain_loss (C13, W2)
a32a7b9 test(chain): W1 broken-four XX.X.XX + triple-axis intersection (C12)
809803a refactor(utils): consolidate hex coordinate helpers (C11, F5+F6)
9eb91e0 feat(pretrain): route augmentation through engine.apply_symmetries_batch (C10)
62c84b9 chore(selfplay): delete dead TensorBuffer and SelfPlayWorker.play_game (C9.5)
920bd55 test(guards): add F1 pretrain-aug + F2 chain-plane Rust parity guards (C9)
a88a591 refactor(engine): extract apply_symmetry_24plane kernel + PyO3 bridge (C8)
fe6a027 update gitignore
d4719d7 docs(sprint): §92 Q13 + Q13-aux + Q19 landing summary + Q21 park (C7)
27e610a chore(pretrain): pretrain v3 + regenerate threat_probe_baseline (C6)
149bb9f chore(corpus): re-export bootstrap_corpus.npz at 24 planes (C5)
7a75185 feat(corpus): compute_threat_pos_weight.py — Q19 pos_weight updater (C4)
b17d2c4 feat: atomic 18→24 plane break — Q13 inputs + Q13-aux head + Q19 pos_weight (C3)
c8f245e feat(aug): axis permutation table for 12-fold hex augmentation (Q13 C2)
5edf4e8 feat(env): _compute_chain_planes helper + 18 unit tests (Q13 C1)
```

**Diff stats:** 80 files changed, 3,724 insertions(+), 998 deletions(-).

---

## Test coverage

| Status | File | Coverage |
|---|---|---|
| New | `tests/test_chain_planes.py` (C1) | 18 `_compute_chain_planes` unit tests |
| New | `tests/test_chain_head.py` (C3) | forward shape, flag ordering, masked loss, gradient leakage, Q19 pos_weight acceptance |
| New | `tests/test_chain_plane_augmentation.py` (C3) | F4 byte-exact 4-position × 12-symmetry invariance oracle |
| New | `tests/test_chain_plane_rust_parity.py` (C9) | F2 Python↔Rust `compute_chain_planes` parity across 21 positions |
| New | `tests/test_pretrain_aug.py` (C9) | F1 pretrain-aug buffer-vs-binding parity, 3 positions × 12 syms × 4000 draws |
| New | `tests/test_coordinates.py` (C11) | 28 round-trip + parsing + hex-distance tests on new `hexo_rl/utils/coordinates.py` |
| Modified | 18 existing test files (plane constant 18 → 24, shape assertions) |  |
| Deleted | `tests/test_tensor_buffer.py`, `tests/test_tensor_buffer_parity.py`, `tests/test_fast_sims_config.py` (C9.5 — dead-code removal) |

**Pre-merge test results (2026-04-15):**

- `make test` — **746 passed, 1 deselected** (python, not slow/integration) +
  **4 passed** (rust cargo test). Runtime ~48s Python, <1s Rust.
- `_compute_chain_planes` perf: 87.1 µs/call (target <50 µs, CI budget 100 µs).
- `ReplayBuffer sample_batch` perf: 2.013 ms/batch (up from 940 µs at 18
  planes, expected: +33% state bytes + plane-aware scatter). Targets in
  `CLAUDE.md` benchmark table need rebaselining post-merge.

---

## Breaking changes

1. **Replay buffer HEXB format v2 → v3** — old buffers rejected on load with
   explicit error. Users must re-populate via self-play (no migration path).
2. **Model checkpoints incompatible** — `input_conv.weight` changes from
   `(128, 18, 3, 3)` to `(128, 24, 3, 3)`; new `chain_head` has no pre-v0.4.0
   counterpart. `strict=True` load fails. Fresh bootstrap required.
3. **Corpus NPZ** — `states` shape `(N, 18, 19, 19)` → `(N, 24, 19, 19)`
   float16. Pre-v0.4.0 corpus NPZs fail with shape mismatch.
4. **Threat probe baseline** — schema v2/v3 → v4. `fixtures/threat_probe_baseline.json`
   regenerated; absolute values not comparable across versions.

Binaries shipped in release assets — users do NOT need to re-run pretrain.

---

## Pre-merge checklist (CI gates)

- [x] `make test` green — 746 python + 4 rust passing
- [x] Smoke training run from v3b bootstrap — `checkpoint_00000020.pt` written
      as `(128, 24, 3, 3)` input_conv, 20 steps, 6 games, end-to-end clean
- [x] `make probe.bootstrap` exit 0 — v4 baseline regenerated, matches §93
      (`contrast_mean=-0.937`, consistent with untrained-threat-head noise band)
- [x] `make probe.latest` pipeline validated on 24-plane checkpoint — C1
      contrast gate +1.158 PASS; C2/C3 top-k expected FAIL on 20-step
      checkpoint (kill criterion applies at step 5000, not step 20)
- [x] `make bench` — 8/10 targets PASS against pre-Q13 floors; 2 expected
      regressions (buffer push 574k vs 630k target −8.7%, buffer sample
      augmented 1638 µs vs 1400 µs target +17%). Both are structural costs
      of the 18 → 24 plane break, not code defects. Follow-up commit
      post-merge: rebase CLAUDE.md buffer-push and sample-aug targets
      against 24-plane floors. MCTS (54,911 sim/s), NN inference
      (8,349 pos/s), NN latency (1.73 ms), GPU util (100%), worker pool
      throughput (522,425 pos/hr), batch fill (100%) all clear.
- [x] Sprint log §92 + §93 reflect final state (C17 landed)
- [x] Release notes written: `docs/releases/v0.4.0.md`
- [x] `releases/v0.4.0/manifest.json` hashes match actual artifacts (sha256
      computed from `checkpoints/bootstrap_model.pt`,
      `data/bootstrap_corpus.npz`, `fixtures/threat_probe_baseline.json`)
- [x] `scripts/install.sh` updated: `RELEASE_TAG=v0.4.0`,
      `BOOTSTRAP_MODEL_SHA256`, `BOOTSTRAP_CORPUS_SHA256`,
      `THREAT_PROBE_BASELINE_SHA256`
- [x] `README.md` plane-count reference updated 18 → 24

---

## Release artifacts (sha256)

| File | Size | sha256 |
|---|---|---|
| `checkpoints/bootstrap_model.pt` | 17,161,671 | `06271362daa257be11a7be16c87fee592fbcf04b3c3a647c0bbcd4d54bf607ab` |
| `data/bootstrap_corpus.npz` | 3,746,845,620 | `c9087b09b3db529702f3177afb450e0cc9cb3bb239758f9ec405a3031dd58790` |
| `fixtures/threat_probe_baseline.json` | 473 | `79b99f3f127fe1834e177c17af4fc83d2da91201317175d2a7f34c2c604be155` |

---

## Rollback procedure

```bash
# Revert the merge commit
git revert -m 1 <merge-sha>
git push origin master

# Restore the pre-v0.4.0 18-plane bootstrap locally
mv checkpoints/archive_18plane_pre_q13/checkpoint_*.pt checkpoints/
mv checkpoints/archive_18plane_pre_q13/replay_buffer_v2_18plane.bin checkpoints/replay_buffer.bin
cp checkpoints/bootstrap_model_18plane.pt checkpoints/bootstrap_model.pt
cp data/bootstrap_corpus_18plane.npz data/bootstrap_corpus.npz 2>/dev/null || \
  curl -L https://github.com/seeligto/hexo_rl/releases/download/v0.1-bootstrap/bootstrap_corpus.npz \
    -o data/bootstrap_corpus.npz

make rebuild          # rebuild Rust extension against reverted N_PLANES
make test             # confirm 18-plane path still green
```

Self-play checkpoints produced post-merge (24-plane) cannot be loaded by the
reverted 18-plane model — any sustained run started on v0.4.0 must be
discarded on rollback. Keep a tagged checkpoint of whatever state you want
to preserve before merging.

---

## Reviewer checklist

- [ ] `feat/q13-chain-planes` branch builds cleanly (`make rebuild`)
- [ ] `make bench` within 24-plane-adjusted targets (measured NN latency
      batch=1 = 1.73 ms, NN inference batch=64 = 8,349 pos/s, MCTS sim/s =
      54,911, worker pool = 522k pos/hr — all PASS. Buffer push 574k
      and buffer sample augmented 1638 µs FAIL against pre-Q13 targets;
      rebaseline via follow-up commit.)
- [ ] `make probe.bootstrap` returns exit 0 on a clean checkout after
      `install.sh` fetch
- [ ] Sprint log §92 + §93 reflect final state
- [ ] `docs/releases/v0.4.0.md` reviewed
- [ ] `releases/v0.4.0/manifest.json` hashes match local `sha256sum` output
- [ ] `scripts/install.sh` downloads the new bootstrap + corpus + probe
      baseline under the new `v0.4.0` tag
- [ ] `README.md` references 24-plane tensor
- [ ] No pre-§92 checkpoint paths remain on any code path the live training
      loop reaches (verified via §93 C9.5 live-path trace)
