# Scoping — `v6_live2` encoding (H-PLANE-MISMATCH regression fix)

**Date:** 2026-05-29. **Input:** `reports/investigations/hplane_mismatch_20260529.md`
(H-PLANE-MISMATCH CONFIRMED). **Status (updated 2026-05-29): scaffolding LANDED
+ run-blockers fixed; pretrain+smoke GPU-queued behind the control.** PROMPT 3
gates Phase 6 on this regression-class finding first.

> **UPDATE 2026-05-29 — landed.** §2 scaffolding built + engine rebuilt; a
> fresh-context review caught 6 run-blockers (chain-plane recompute hardcoding
> opp-stones at slot 4 — the ledger missed this `states[:,4]` class). All fixed
> via a registry-derived `opp_stone_slot(spec)` helper at 5 recompute sites +
> the checkpoint_loader allow-list/spec-branch; backward-compat pinned
> (opp_slot==4 for every existing encoding, 1 for v6_live2);
> `tests/test_v6_live2_wiring.py` added. v6_live2 is now launch-ready — remaining
> is the GPU pretrain (`scripts/p5a_v6_live2_pretrain.sh`) + 30k smoke, queued
> behind the control. See sprint-log §P5-CT "v6_live2 LANDED" + L65.

## 1. What `v6_live2` is + the hypothesis

`v6_live2` = v6tp **minus the dead history planes**:
`kept_plane_indices = [0, 8, 16, 17]` (4 planes = my t0, opp t0,
moves_remaining bcast, ply parity). It removes exactly the planes
(1-3/9-11) that are live in pretrain but **zeroed** in self-play.

**Sharpened mechanism (key insight).** At *inference / self-play* the v6tp model
already receives its 6 history planes as constant **zero** (Rust encoder). So
v6tp-at-inference is functionally `[0,8,16,17]` already — except its first conv
spent weights on 6 dead-zero channels and its **pretrain** learned features that
relied on (now-zero) history. The whole difference between v6tp and v6_live2 is
born in **pretrain**:

- **v6tp pretrain:** richer (uses live history) → possibly better initial
  policy/value, BUT meets a distribution cliff entering self-play (history→0) →
  those features go dead.
- **v6_live2 pretrain:** leaner (no history) → possibly weaker initial fit, BUT
  **no cliff** (pretrain == self-play, all 4 planes live in both).

The MCTS-matched smoke decides which net effect wins. **NOT a colony fix** — the
shift is invariant across collapsing/non-collapsing runs (L64); this is a
potential baseline lift.

## 2. Build scope (all NO-GPU; ~bounded)

`v6_live2` is the first **4-plane** encoding, so it trips the same
`in_channels ∈ {8,10,11}` allow-lists the hardcode ledger flagged. Touches:

| # | file:line | change |
|---|---|---|
| 1 | `engine/src/encoding/registry.toml` | new `[encodings.v6_live2]`: board_size/trunk 19, single-window, n_planes=4, plane_layout=[current_player_t0, opponent_t0, moves_remaining_bcast, ply_parity_bcast], kept_plane_indices=[0,8,16,17], n_source_planes=18, policy_logit_count=362, has_pass_slot=true, value/policy_pool=none, sym_table_id=size_19, k_max=1, schema_version=3 |
| 2 | `hexo_rl/encoding/registry.py:27` | add `"v6_live2"` to `_REGISTERED_NAMES` |
| 3 | `hexo_rl/encoding/resolvers.py:~364` | new detector branch `if in_ch == 4 and n_actions == 362: return lookup("v6_live2")` (else eval/anchor auto-detect fails on a 4-plane ckpt). Update stale error text at :381. |
| 4 | `hexo_rl/encoding/resolvers.py:130,140` | add corpus/ckpt path-map entries: `data/bootstrap_corpus_v6_live2.npz`, `checkpoints/bootstrap_model_v6_live2.pt` |
| 5 | `scripts/export_corpus_npz.py:209` | add `"v6_live2"` to `--encoding` choices; the v6-family single-window branch (~:340) already slices generically via `spec.kept_plane_indices` → 4-plane export works once past the choices guard |
| 6 | `configs/variants/v6_live2_smoke.yaml` | mirror `v6_p5a_control.yaml`; set **`encoding: v6_live2` + `in_channels: 4` explicitly** (sidesteps ledger P0-1 orchestrator in_channels=18 fallback), lever stack OFF, eval sealbot+anchor+random, total_steps 30000, eval_interval 5000 |
| 7 | `scripts/p5a_v6_live2_pretrain.sh` | mirror `p5a_v6_control_pretrain.sh` with `--encoding v6_live2` → `bootstrap_corpus_v6_live2.npz` + `bootstrap_model_v6_live2.pt` (fresh 4-plane, NOT a transfer) |

Verify with `python -m hexo_rl.encoding audit` (Rust/Python parity + invariants;
the current exit-2 is only §6 metadata WARNs on archived v6tp ckpts, orthogonal).
Round-trip test: add `v6_live2` to `tests/test_encoding_round_trip.py` and the
non-8-plane regression matrix.

**Opportunistic:** building v6_live2 is the natural moment to land ledger **P0-1**
(`orchestrator.py:286` in_channels→spec.n_planes) since v6_live2 (4) re-exposes it.

## 3. Optional cheap PRE-SCREEN (screen, NOT verdict — L2)

Before spending ~13.5hr GPU, a minutes-level probe can estimate the cliff
magnitude: take `bootstrap_model_v6tp.pt`, run its policy/value head on a matched
position set fed two ways — (a) live-history input (Python `to_tensor`, the
pretrain distribution) vs (b) zeroed-history input (the self-play distribution).
A large policy-KL / value gap ⇒ the head leans on history ⇒ the cliff is real ⇒
v6_live2 likely helps; a small gap ⇒ v6tp≈v6_live2 likely. **Argmax-level only —
cannot validate dynamic equivariance (L2); use to prioritize, not to decide.**

## 4. Gate — MCTS-matched smoke + pre-registered verdict

Run the v6_live2 30k smoke and an **MCTS-matched** eval (mcts-128, same n_games,
both T=0 and T=0.5) against the **v6tp 30k** numbers (archived
`checkpoints/v6tp_archive/`: sampled 0.23 / greedy 0.33; V_spread net-positive;
stride5_p90=4, colony_ext_frac=0.0).

| Verdict | Criterion | Action |
|---|---|---|
| **ADOPT v6_live2** | SealBot WR (both temps) ≥ v6tp within Wilson-CI overlap **AND** V_spread net-positive **AND** spam-signal clean (stride5_p90≪60, colony_ext_frac≪0.40) | adopt as the clean encoding (parsimony: 4 planes, no pretrain↔selfplay cliff) |
| **KEEP v6tp** | WR clearly below v6tp / below anchor 0.174, OR V_spread collapses, OR spam fires | pretrain head genuinely benefits from history despite self-play zeroing → close H-PLANE as "known asymmetry, net-neutral" |
| **INCONCLUSIVE** | overlapping CIs, no clear separation | n=100 both temps was underpowered → one larger matched eval (n=200) decides |

Cosine temp OFF (L9). Spam-signal primary, V_spread DEMOTED (operator note 2).

## 5. Sequencing (the control verdict informs the kept-set)

1. **GPU contention:** the single 5080 is running the CF-1 control (~12.5hr).
   v6_live2 pretrain+smoke queues **after** it (or a 2nd GPU). The no-GPU code
   scaffolding (§2) can land now so launch is one command when the GPU frees.
2. **Read the control verdict FIRST — it sets the kept-set.** Control isolates
   CF-2 (turn-phase 16/17): v6-control (8-plane) vs v6tp (10-plane).
   - Control says **CF-2 load-bearing** (v6-control < v6tp) → keep 16/17 →
     `v6_live2 = [0,8,16,17]` (this scope).
   - Control says **CF-2 null** (v6-control ≈ v6tp) → 16/17 add little → also test
     a minimal `v6_live = [0,8]` (2-plane) as a second arm; `v6_live2` still the
     safe default.

## 6. Risks

- **Thinner pretrain.** 4-plane pretrain has strictly less info than v6tp's
  (no history context for human-move prediction); pretrain loss likely higher.
  Downstream RL is what matters — the smoke measures it. (in_channels=4 vs 128
  filters: no architectural concern.)
- **More allow-list hardcodes.** in_channels=4 will surface any remaining
  `{8,10,11}` detectors beyond §2 #3 (promotion path, probe_threat_logits,
  early_game_probe). Run a 4-plane end-to-end dry smoke (a few hundred steps) to
  flush them before the 30k commit — do NOT repeat the v6tp reactive-crash slog.
- **Underpowered eval.** n=100 each temp gives wide Wilson CIs vs v6tp's; budget a
  fallback n=200 matched eval if the first pass is INCONCLUSIVE.

## 7. Cost & relation to PROMPT 3

~1hr pretrain + ~12.5hr 30k smoke on the 5080 (≈ the control's cost), queued. Per
PROMPT 3, this regression-class fix **precedes** any league spend — if v6_live2
ADOPTS, re-baseline and reassess whether anything is left for Phase 6.
