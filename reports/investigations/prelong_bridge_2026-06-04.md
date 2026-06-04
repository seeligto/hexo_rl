# §PRELONG-BRIDGE — does scatter buy enough vs-bot strength? (perception→strength CEILING)

**Date:** 2026-06-04 · **Branch:** `investigation/prelong-bridge` (off
`phase4.5/prelong_2a_centering`) · **Discipline:** standalone-eval generation →
factored-detector analysis → pre-registered routing, **training-free, frozen
weights**. · **Frozen checkpoint:** `checkpoints/v6_live2_rl/checkpoint_00030000.pt`
· **No new head, no re-pretrain, no engine/encoding change.**

> **Question (handoff §3 fork):** does fixing the off-window forced-win wall (a
> scatter ACTION-space re-pretrain) buy enough **vs-bot strength** to justify the
> scatter chapter (D-SCATTER), or bank §PRELONG-2A and reroute? This measures the
> **perception→strength CEILING** on frozen weights: the upper bound on vs-bot
> WR lift that recovering off-window dropped wins could yield.

---

## TL;DR — **BANK §PRELONG-2A. Scatter buys ~0pp vs-bot WR (both opponents). Go-long v6_live2; arm the 300k off-window-frequency tripwire.**

> Across **400 ModelPlayer-vs-bot games** (SealBot + NNUE × greedy/sampled, n=100
> each, frozen 30k, sims=128), the scatter WR-lift CEILING is **0.0pp for BOTH
> opponents** — Wilson95 upper bound **1.52pp**, well inside the ≤3pp **BANK**
> band. **Not one** non-won game (0/152 SealBot, 0/72 NNUE) contained a
> scatter-bridgeable dropped forced win. The reason is structural and robust: the
> off-window wall is a **self-play** phenomenon (HeXO-vs-HeXO spread boards). Vs
> SealBot HeXO's forced wins are rare (45/200 turns) and **0 were off-window**; vs
> the weaker NNUE, off-window forced wins do occur (5 turns) but only when HeXO is
> **dominating** (WR 0.64) — the single bridgeable drop landed in a game HeXO
> **won**, so conditioning on non-won zeroes it. The §PRELONG off-window-drop bug
> does not cost measurable vs-bot strength on the eval-time single-window path; a
> scatter ACTION-space re-pretrain cannot recover WR that was never lost there.
> Its value (if any) is the orthogonal V3 scoring residual + the self-play
> exploration prize — neither measured here, neither a vs-bot-strength argument.

---

## 0. Metric — scatter WR-lift CEILING

Per opponent (SealBot, NNUE), conditioned on **NON-WON games only**:

```
bridge = ( #non-won games with >=1 (deduped) scatter-bridgeable dropped forced win )
         ───────────────────────────────────────────────────────────────────────────  ×  0.806
                                    total_games
```

- **scatter-bridgeable dropped forced win** = a HeXO turn where a forced win was
  available, the turn was NOT won, and the binding (max-cheb) win cell is **off
  the single global ACTION window** (`to_flat == usize::MAX`) yet **inside a
  cluster window the encoder already perceives** (`get_cluster_views()` center
  within chebyshev ≤ 9). I.e. the NN sees the winning cluster but the collapsed
  single action window drops the cell → genuinely unplayable on the ModelPlayer
  path.
- **0.806** = the D1 deduped-majority **cheat-recenter recovery ceiling**
  (frozen-30k; `prelong_centering_vs_size_2026-06-04.md` §3). It is the fraction
  of off-window misses that a *perfect* re-center recovers. Scatter ≈ that perfect
  re-center → multiply by it.
- **Deduped per distinct winning LINE** within a game (§PRELONG 4.1× per-turn
  inflation: the same unconverted forced win recurs turn-after-turn).
- It is a **CEILING, not an estimate** (red-team §5).

Pre-registered routing (operator defaults; pp = bridge × 100):
**≥10pp** (either opponent) → **GREENLIGHT** D-SCATTER. **≤3pp** (both) → **BANK**
2A, go-long v6_live2, arm the 300k off-window-FREQUENCY tripwire. **3–10pp** → no
chapter commit; short leg + tripwire, re-read.

---

## 1. Path discrepancy — the bug only binds the single-window ModelPlayer path (KEY FINDING)

The §PRELONG off-window-drop is a property of the **single global ACTION window**.
There are two distinct v6_live2 inference paths, and **only one has the bug**:

| path | used by | action handling | off-window drop? |
|---|---|---|---|
| **`ModelPlayer`** (Rust `MCTSTree`, `evaluator.py:108-126`) | the **in-loop eval gate** (§5 sign-off) + §PRELONG-2A probe | builds policy over `legal_flat`, then `policy[i] if i < n_actions else 0.0` (`:113`) — a legal cell whose `to_flat==MAX` gets **prior 0 → dropped** | **YES** |
| **`KClusterMCTSBot`** (Python, `k_cluster_mcts_bot.py:117-160`) | the **standalone** `run_sealbot_eval.py` harness | scatter-maxes priors onto the legal set across **ALL K cluster windows** (`:146-155`) — any legal cell in *some* cluster window gets a non-zero prior | **NO** (already scatters) |

⇒ The bridge metric (a *reachability* recovery ceiling) is only coherent on the
**ModelPlayer** path. So this investigation **generates games with ModelPlayer**
(matches the §PRELONG-2A detector provenance + the in-loop gate the §5 sign-off
fires on). The KClusterMCTSBot path is reported as a **structural cross-check**
(§5): every scatter-bridgeable dropped win is, by construction, off the global
window but *legal + in a cluster window* — i.e. **already inside
KClusterMCTSBot's action space**. If the operator's quoted vs-bot WR comes from
the standalone (KClusterMCTSBot) harness, the off-window bug costs that number
~nothing at the reachability level, and scatter's vs-bot value there collapses to
the V3 scoring residual (orthogonal; O1/more-sims).

**The scatter chapter's deepest value is in SELF-PLAY** (the Rust `worker_loop`
single-window path that *shapes the trained model* — off-window wins never get
explored → never valued → the V3 residual). The bridge measures the **eval-time
ModelPlayer ceiling**, a *lower bound* on the self-play value, not the whole prize.

---

## 2. Method (training-free, frozen 30k weights)

**Generation** (`scripts/structural_diagnosis/prelong_bridge_gen.py`, run on vast
5080): frozen `v6_live2` 30k via `ModelPlayer` (Rust MCTS, single window),
`sims=128` (canonical eval), `c_puct=1.5`, vs **SealBot** (`think_time=0.5s`) and
**NNUE/Hammerhead** (`500ms/stone`). Both temps **{0.0 greedy, 0.5 sampled}**,
`opening_plies=2` **pinned identically across temps**, n=100 per (opponent, temp)
= **200/opponent**. Full move-list + outcome + side recorded per game.

> *opening_plies=2 vs the canonical milestone profile's 0:* guarantees game
> diversity for both opponents and matches the §PRELONG-2A / D1-oracle generation
> whose 0.806 ceiling this bridge multiplies. opening_plies>0 mildly inflates HeXO
> WR (§174) → fewer non-won games → **conservative** on the numerator. The
> off-window structural rate is opening-ply-invariant. SealBot is stochastic even
> at fixed seed (time-budget search; verified) so greedy games do not collapse.

**Detection** (`prelong_bridge_analyze.py`, CPU): **REUSES the §PRELONG-2A
factored forced-win detector** (`depth1_wins` / `depth2_wins` / `find_win_line`
imported from `prelong_2a_eval.py` — no second detector written). Per HeXO
turn-start snapshot: detect forced wins; binding = **max-cheb** cell; flags read
on the turn-start board exactly as §PRELONG-2A — `off_window` (`to_flat≥362`),
`cluster_reachable` (`get_cluster_views` center cheb≤9), `legal_at_start`,
`won_turn` (whether HeXO's actually-played turn — the visit-count argmax/sample
realized in the move-list, NOT raw policy — completed the win). Bridge + Wilson
CIs + the KCluster cross-check + routing computed per opponent.

**Detection validated** (pre-run): `to_flat` returns `usize::MAX` for cheb>9
cells (off_window=True) and a valid index for cheb≤9 (in-window) on a constructed
spread board; cluster centers drive reachability. The forced-win detector is the
already-validated §PRELONG-2A code.

---

## 3. RESULTS  (400 games, vast 5080, generation 4977s)

| opponent | games (unique) | WR | non-won | forced-win turns | off-window turns | bridgeable drops raw / deduped | non-won w/ drop | **bridge pp** [CI95] | KCluster-reaches |
|---|---|---|---|---|---|---|---|---|---|
| **SealBot** | 200 (197) | 0.24 | 152 | 45 | **0** | 0 / 0 | **0 / 200** | **0.0pp** [0.0, 1.52] | — (no drops) |
| **NNUE** | 200 (200) | 0.64 | 72 | 126 | **5** | 1 / 1 | **0 / 200** | **0.0pp** [0.0, 1.52] | **1.0** |

Per-temp (non-won-with-drop / non-won / total):

| opponent | greedy τ=0.0 | sampled τ=0.5 |
|---|---|---|
| SealBot | 0 / 79 / 100 | 0 / 73 / 100 |
| NNUE | 0 / 26 / 100 | 0 / 46 / 100 |

**Reading.** SealBot: HeXO is the weaker side (WR 0.24), reaches few forced-win
turns (45), and **none** are off-window — the single global window never dropped a
forced win vs SealBot. NNUE: HeXO is the stronger side (WR 0.64), gets many
forced-win turns (126) and **5** are off-window — but 4/5 were converted via
another (in-window) line and the **1** genuine bridgeable drop occurred in a game
HeXO **won**. So in **0** of the 224 non-won games (152 + 72) was a
scatter-reachable forced win dropped. Off-window forced wins are a *dominant-HeXO*
event; non-won games (HeXO losing/drawing) are not dominant positions, so the
conditioned numerator is empty.

---

## 4. REVIEW (harness soundness)

- **"Dropped" = argmax actually played.** `won_turn` is computed from HeXO's
  recorded move-list (the realized visit-count argmax at temp 0 / visit sample at
  temp 0.5), not the raw policy head. An off-window win is structurally unplayable
  on ModelPlayer (prior 0), so `won_turn=False` on a forced-win turn whose binding
  cell is off-window is a genuine drop.
- **"Reachable-if-scatter" = in a cluster window's cells**, not "exists on board":
  `cluster_reachable` requires a `get_cluster_views()` center within cheb ≤ 9 of
  the win cell. The 1 NNUE off-window drop that was counted passed this test (and
  was legal) — so it is genuinely scatter-reachable, not merely on-board.
- **Non-won conditioning enforced**: only `won==False` games enter the numerator.
- **Detector reuse**: forced-win logic is the §PRELONG-2A functions verbatim;
  arm-A-equivalent reproduction (off-window misses on the single-window path)
  is the same mechanism D1 measured.

---

## 5. RED-TEAM (disprove the ceiling)

- **CEILING, not estimate.** Converting one dropped win need not flip the game
  (opponent counterplay after the would-be win move; or the game was already lost
  elsewhere). The bridge is the *maximum* WR lift, achieved only if every
  recovered win flips its game — it almost certainly does not. Reported as a
  ceiling.
- **0.806 is directional.** It is the frozen-30k cheat re-center recovery. A
  scatter re-pretrain may *beat* it (the head learns to score off-window cells →
  closes part of the V3 residual) or *miss* it (re-pretrain regressions). The
  bridge inherits this as a directional, not exact, multiplier.
- **30k regime ≠ 300k sprawl.** All numbers are the noise-free local-30k spread;
  `cheb≤11` is unvalidated for the 300k Dirichlet regime (handoff §5). The
  **300k off-window-FREQUENCY tripwire** confirms the bound later — the bridge
  does not.
- **Over-count → dedup.** Recurring threats are deduped per winning LINE within a
  game; the per-game numerator is binary. Here dedup was a no-op (NNUE 1 raw drop
  turn → 1 deduped line; SealBot 0) — there is nothing to inflate. The result is
  not a dedup artifact.
- **Under-count guard (the inverse risk).** Could the metric miss real drops?
  No — `off_window_forced_turns` (SealBot 0, NNUE 5) is reported directly, so the
  near-zero bridge is traced to off-window forced wins genuinely not arising in
  vs-bot play, not to a detector that fails to flag them (the detector is the
  validated §PRELONG-2A code, and fired on self-play giving 149 misses).
- **KClusterMCTSBot cross-check.** The 1 NNUE bridgeable-drop win cell was legal +
  in a cluster window ⇒ already inside KClusterMCTSBot's scatter action space
  (`kcluster_reachable_frac_of_drops = 1.0`). So even the eval-time reachability is
  **not** lost on the standalone harness — the bug is exclusive to the
  single-window ModelPlayer path, and there are too few cases for it to matter.

---

## 6. PRE-REGISTERED ROUTING + VERDICT

| opponent | bridge | band | gate |
|---|---|---|---|
| SealBot | **0.0pp** [0.0, 1.52] | ≤3pp | BANK |
| NNUE | **0.0pp** [0.0, 1.52] | ≤3pp | BANK |

Both opponents ≤3pp (point 0.0pp, upper CI 1.52pp ≪ 3pp) → the **≤3pp BOTH** rule
fires:

> ### VERDICT: **BANK §PRELONG-2A.** Go-long on `v6_live2`; arm the 300k
> ### off-window-FREQUENCY tripwire as a go-long precondition. Do **NOT** commit
> ### the scatter chapter (D-SCATTER) on a vs-bot-strength argument.

The perception→strength bridge is empty: scatter would recover off-window dropped
wins, but in vs-bot play HeXO **does not drop scatter-reachable forced wins in
games it does not win**. The off-window wall is real in self-play (D1: 149 misses)
yet does not convert to vs-bot WR loss on the eval-time path. This matches the
handoff's closing recommendation (lean (b) bank + hold) and gives it a measured
ceiling, not just a prior.

**What this does NOT say.** It does not say scatter is worthless — only that its
vs-bot-strength justification is absent at 30k. Two orthogonal values remain
un-refuted and un-measured here: (1) the self-play **exploration** prize (the Rust
worker_loop single-window path never proposes off-window moves → the model never
learns to value them — the V3 residual's likely origin); (2) tactical
completeness as an end in itself. Greenlight D-SCATTER only on an explicit
"forced-win completeness / self-play exploration is the priority" call, per
handoff §3 — NOT on this bridge.

**Tripwire (go-long precondition, per handoff §5 + RED-TEAM §5).** All numbers are
the noise-free local-30k spread. The 300k Dirichlet regime spreads wider → both
the off-window frequency AND the chance HeXO is the dominant side in a *drawn-out*
game could rise. Arm the **300k off-window-FREQUENCY tripwire**: if, at the
long-run milestone evals, the off-window forced-win frequency among **non-won**
games climbs materially above this 30k baseline (~0/224), re-open this bridge
before relying on the BANK. The BANK conclusion itself is 30k-specific; the
*mechanism* (off-window = dominant-HeXO event, absent from non-won games) is what
the tripwire watches for drift.

**FLAG (carried from handoff §5):** the §5 SECONDARY (90% forced-win conversion)
is a **scatter + V3 JOINT** target, NOT scatter-alone — the cheat ceiling
0.806 < 0.90, the V3 residual is O1/more-sims (orthogonal). Do not gate scatter on
a bar it cannot hit alone.

---

## 7. Pointers

- **Bridge harness:** `scripts/structural_diagnosis/prelong_bridge_gen.py`,
  `prelong_bridge_analyze.py` (committed, `investigation/prelong-bridge`).
- **Data (gitignored):** `reports/investigations/prelong_bridge_data/games.jsonl`,
  `summary.json` (vast).
- **Detector reused:** `scripts/structural_diagnosis/prelong_2a_eval.py`.
- **D1 routing / 0.806 ceiling:** `prelong_centering_vs_size_2026-06-04.md`.
- **§PRELONG-2A FALLBACK:** `prelong_2a_eval_result_2026-06-04.md`; handoff
  `prelong_2a_handoff_2026-06-04.md`.
- **Frozen ckpt:** `checkpoints/v6_live2_rl/checkpoint_00030000.pt`.
