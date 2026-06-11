<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §92 — Q13 + Q13-aux + Q19 atomic landing (2026-04-14, partially SUPERSEDED by §97)

> **Post-§97:** chain planes no longer live in the NN input tensor — they were moved to a dedicated `ReplayBuffer.chain_planes` sub-buffer. The design decisions and aux-head structure below still hold; the "18→24 input break" and per-run numbers are historical. Pretrain v3 itself was broken by an augmentation bug caught at §93 F1; v3b (§93) is the production bootstrap.

**What.** Three interlocking changes landed as a fresh-start cycle (bootstrap corpus re-export + pretrain v3 + new `bootstrap_model.pt`). Atomic because buffer layout + checkpoint shape + loss wiring cannot cross-boundary test individually. Q19's `pos_weight=59` co-landed so the threat BCE fix piggybacks on the new bootstrap.

**Motivation.** Literature review (`reports/literature_review_26_04_24/review.md`): KataGo-style Tier 2 geometric feature to accelerate tactical-threat learning; MoHex-CNN bridge planes, KataGo liberty/ladder planes, Rapfi per-axis line patterns. AZ-style Gomoku implementations stay raw-stone-only and all document the same threat-blindness failure mode. Q19: without `pos_weight`, threat-head BCE drifts globally negative at ~1.6% positive labels (§91).

**Design decisions (architectural authority — survives §97):**

1. **Chain-length semantics — post-placement.** Cell value = `1 + pos_run + neg_run` for own stones and empty cells with at least one adjacent own neighbour; 0 elsewhere and for opponent cells. Capped at 6, /6-normalised. `XX_XXX` → empty cell value = 6/6.
2. **Chain-aux target sourcing.** §92 used `chain_target = input[:, 18:24]` (slice-from-input); **§97 revised to read from the replay-buffer chain sub-buffer**. Head job remains "preserve/rediscover chain-counting through the tower".
3. **numpy-vectorised tensor assembly (no numba).** Pure Python rejected (13–33 ms budget blowout); `np.roll` rejected (wraps and violates window-edge opacity). Python helper stays in `hexo_rl/env/game_state.py::_compute_chain_planes`.
4. **`aux_chain_weight = 1.0` (not 0.10).** /6-normalised target → smooth_l1 ~0.02/cell; weight 0.10 → ~0.002 loss vs policy ~2.0 — invisible. 1.0 gives the aux head meaningful gradient share.
5. **Atomic bundle.** 56 files changed in one commit. Coverage: byte-exact augmentation-invariance test + chain-head mask tests inside the same commit.

**Downgraded expectations — not KataGo 1.65×.** That headline is from KataGo's auxiliary FUTURE-information targets (terminal ownership). Our chain target is a current-input slice (§92) / a same-window chain-plane recomputation (§97) — regularisation + intermediate supervision, not counterfactual forward information. Realistic uplift 1.1–1.3× on tactical probe convergence. Q21 parks the wider-window variant that would match KataGo's structure.

**Commit sequence (details in git log):**

| Commit | Scope |
|---|---|
| C1 | `_compute_chain_planes` Python helper + 18 unit tests. 78 µs/call at 50 stones, 165× faster than pure Python. |
| C2 | `SymTables.axis_perm` table + 10 inline tests. Axis permutation period 3 (180° identity on direction-unsigned axes). |
| C3 | **Atomic 18→24 plane break** — 56 files, 1019+/185−. Touches `game_state.to_tensor`, Rust `encode_state_to_buffer`, `SymTables` scatter split, `HexTacToeNet` + `chain_head`, `compute_chain_loss`, `Trainer._threat_pos_weight`, `pretrain.train_epoch`, dashboards, and test-layer plane-shape updates. HEXB v2 → v3 with `n_planes` header. |
| C4 | `scripts/compute_threat_pos_weight.py` — recomputes `(1-p)/p` from the buffer; falls back to §91 theoretical 59.0 when no 24-plane buffer exists. |
| C5 | Corpus re-export at 24 planes (`scripts/export_corpus_npz.py --human-only --max-positions 200000 --no-compress`). 199,470 positions, 3.6 GB. |
| C6 | Pretrain v3: 15 epochs × 779 batches, ~40 min on RTX 3070. Produces 24-plane `bootstrap_model.pt` + threat_probe_baseline v3. **Broken by F1 aug bug (see §93).** |
| C7 | §92 sprint log landing. |

**Load-bearing follow-up notes:**

- **Threat and ownership heads untrained at bootstrap.** Corpus NPZ has no per-row winning_line or ownership targets (self-play-only post §85 A1). Q19 `pos_weight=59` kicks in once self-play feeds aux targets.
- **Probe baseline policy change.** `probe_threat_logits.py --write-baseline` now always exits 0 — a bootstrap's random-init threat head cannot satisfy the absolute 0.38 contrast floor. Gate applies to post-self-play checkpoints only.
- **Checkpoint incompat (§92 onwards).** First-conv shape mismatch with any pre-§92 checkpoint. Pre-§92 archives: `bootstrap_model_18plane.pt`, `bootstrap_corpus_18plane.npz`, pre-§92 `replay_buffer.bin` (v2 HEXB) rejected at load.

**Open questions updated (see `docs/06_OPEN_QUESTIONS.md`):**

- Q13: resolved by this landing (input form); revised by §97 (aux-sub-buffer form).
- Q19: resolved by `pos_weight=59`. §91 C4 warning hook stays.
- **Q21 parked:** wider-window aux target for forward-information injection. Current chain target (§97 form) is a same-window recomputation — trunk can already see chain values in the stones. KataGo's 1.65× speedup requires future-information targets (terminal ownership); wider-window chain is the Hex analogue. Revisit after §97 baseline stabilises.

---

