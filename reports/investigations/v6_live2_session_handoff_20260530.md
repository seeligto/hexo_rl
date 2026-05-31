# Session handoff — §P5-CT PROMPT 1/2 + v6_live2 arc (2026-05-29 → 30)

Working tree on `master` (uncommitted). Companion docs:
`reports/investigations/hplane_mismatch_20260529.md`,
`reports/investigations/v6_live2_scoping_20260529.md`,
`audit/structural/encoding_width_hardcode_ledger.md`, sprint-log §P5-CT.

This file = what was done, the reasoning, what's running, and what's left.

---

## 1. PROMPT 1 — CF-1 vs CF-2 attribution + second opponent

**Verdict DELIVERED: CF-2 (turn-phase planes 16/17) is LOAD-BEARING.**
- CF-1-only v6 control launched on vast (clean single-variable: 8-plane v6,
  same engine HEAD incl. CF-1, lever stack OFF, seed 42, matched corpus).
- In-run SealBot WR (T=0.5): **9% → 1% → 3%** at steps 5k/10k/15k, while
  self-anchor WR *rose* (0.47→0.63→0.54) — the anchor↑/sealbot↓ colony-capture
  divergence (L34). vs v6tp's held ~21% plateau.
- → The control collapses where v6tp holds ⇒ CF-1's terminal-sign fix alone
  does NOT suppress the attractor; the **turn-phase plane carries v6tp**.
  Action: **keep the v6tp turn-phase plane; CF-2 is the fix.** CF-1 + CF-4
  remain banked as correctness fixes regardless.
- Refinement: the collapse shows as the WR/anchor divergence, NOT as
  `colony_extension_fraction` (which stayed 0.0) — a strength collapse vs
  SealBot, not extension-spam.
- Control was killed at step 20k (verdict already decisive) to free the vast
  5080 for v6_live2.

**Second-opponent arm DROPPED (operator).** KrakenBot — the only non-SealBot
bundled bot — is too weak: its MinimaxBot emits illegal-move sentinels mid-game
so the wrapper falls back to a neighbor-2 heuristic (early v6tp@T0 vs Kraken
≈90% reflected the weakened opponent). Only alternative is `CommunityAPIBot`
vs a live `explore.htttx.io/bots/<name>` (needs URL + online). The trial
`run_sealbot_eval --opponent` wiring was reverted. So v6tp's ~21%/33% stays
read as distance-from-SealBot, not validated as a general plateau.

## 2. PROMPT 2 — H-PLANE-MISMATCH + hardcode ledger

**H-PLANE-MISMATCH CONFIRMED.** v6-family wire planes 1-3/9-11 (history) are
live in pretrain (corpus mean-abs ≈ live t0 ≈ 0.04, 94-99.8% of rows) but
EXACTLY ZERO in self-play (Rust encoder). 6 of 8 (v6) / 6 of 10 (v6tp) wire
planes carry full mass in pretrain and zero in RL — a transfer cliff. Matched
dump + fresh review PASS. **Critical caveat (L64):** the mismatch is INVARIANT
across §150 (no collapse) and §175/§S181 (collapse) → a constant baseline
handicap, NOT the colony differentiator. Scoped against/before Phase 6.

**Hardcode ledger** (`audit/structural/encoding_width_hardcode_ledger.md`):
P0=2, P1=6 by name-grep. **L65: the ledger MISSED the positional-slice class**
(`states[:, 4]` opp-plane index) — found only when v6_live2 (4-plane) ran.

## 3. v6_live2 — the H-PLANE fix encoding (scoped → built → running)

`kept_plane_indices = [0, 8, 16, 17]` (4 planes = my/opp t0 + turn-phase) =
v6tp minus the dead history planes ⇒ pretrain == self-play (no cliff). The
whole v6tp↔v6_live2 difference is born in pretrain (at inference v6tp already
feeds history as zeros).

**Built + de-hardcoded.** Registry entry + Python name + engine rebuilt;
resolvers detector (`in_ch==4`); export `--encoding`; configs + pretrain script.
Getting it to actually RUN took clearing blockers the scaffolding/review missed:
- chain-plane recompute hardcoded opp-stone at slot 4 (v6 pos); v6_live2 opp is
  slot 1 → fixed via registry-derived `opp_stone_slot(spec)` at 5 sites
  (batch_assembly ×3, pretrain_dataset) + checkpoint_loader allow-list/spec.
  Backward-compat pinned (opp_slot==4 for all existing encodings).
- `inference_methods.py` label allow-list (`("v6","v6tp")` → add v6_live2).
- `probe_threat_logits._probe_one` sliced v6's 8 planes via `model.encoding`
  (the geometry-family bucket) → threaded registry kept-indices.
- **Stale-state pollution** (NOT a code bug): old `best_model.pt` /
  `bootstrap_model_v6.pt` in `checkpoints/` were auto-discovered and propagated
  their encoding (v7full→v6) over v6_live2 → dashboard `_check_scattered_keys`
  crash (in_channels=4 vs n_planes=8). Fixed by isolating `checkpoints/` to only
  the v6_live2 bootstrap (old artifacts → `checkpoints/_archive_pre_v6live2/`).
  **This is the same trap on vast — clean the run dir before any launch.**

**Probes (laptop, GPU now free):**
- SealBot (T=0, mcts-128, n=100): **34% [25%,44%]** — ~2× the v7full 17.4%
  anchor. Strong bootstrap (caveat: part of the edge is the by-design cliff
  removal; the gate is the 30k smoke).
- Threat-logit (C1 contrast at bootstrap): v6_live2 **0.063**, v6tp 0.108,
  v7full 0.073 — all weak/comparable; bootstrap threat heads are undeveloped
  (the 0.599 ref baseline was an older/trained model). No red flag.

## 4. Vast swap (in progress)

Control killed → vast `checkpoints/` cleaned → 10 changed files rsync'd (incl.
the initially-missed `export_corpus_npz.py`) → engine rebuilt on vast →
v6_live2 verified → **pretrain→smoke chain launched** (tmux `v6l2`, canonical
`v6_live2_smoke`, n_workers=18). As of 07:37 UTC vast: pretrain at step −10250,
value_acc 0.77, ~45-50k steps/hr, ~12 min to pretrain-done, then the 30k smoke
auto-starts. First eval at step 5k (~2hr in).

**Known inefficiency (operator-flagged):** the laptop had already produced the
deterministic bootstrap — we could have rsync'd it (17MB) + re-exported the
corpus on vast and started the smoke immediately, skipping the ~1hr re-pretrain.
Reused the full pretrain chain wholesale instead. Lesson: reuse the artifact,
don't re-pretrain.

---

## 5. What's left (against the task)

| item | status |
|---|---|
| **PROMPT 1 verdict** (CF-2 load-bearing) | DONE — banked here; keep v6tp turn-phase plane |
| PROMPT 1 second opponent | DROPPED (operator) |
| **PROMPT 2 H-PLANE CONFIRMED** + ledger | DONE |
| **v6_live2 30k smoke = THE GATE** (ADOPT v6_live2 / KEEP v6tp) | **RUNNING on vast (~13hr)** — pending |
| **PROMPT 3 — Phase 6 scoping memo** | NOT STARTED — gated on the v6_live2 smoke + the (now-delivered) control verdict |
| Commit the v6_live2 work (code + probe fix + configs + docs) | NOT DONE — all uncommitted (local tree + rsync'd to vast) |
| Consolidate de-hardcoding (ledger P0s: orchestrator in_channels=18, generate_bot_corpus; + the reactive 4-plane fixes) | NOT DONE — reactive only; proper pass owed |
| `make probe.bootstrap` for v6_live2 without `--encoding` | minor gap (needs --encoding or fixture-encoding auto-resolve) |

**The single biggest open item: the v6_live2 30k smoke verdict** (running). It
decides ADOPT v6_live2 vs KEEP v6tp, and feeds the PROMPT 3 Phase-6 decision.
Pre-registered gate (`v6_live2_scoping_20260529.md §4`): ADOPT if SealBot WR ≥
v6tp within CI at both temps AND V_spread net-positive AND spam-signal clean;
KEEP v6tp if clearly worse / V_spread collapse / spam fires.

## 6. Lessons banked

- **L64** — a documented encoding asymmetry constant across runs is a baseline
  handicap, never a per-run failure differentiator.
- **L65** — name-grep hardcode ledgers miss positional `states[:, N]` plane
  indices; pair with a positional-slice grep / dry-run when changing plane COUNT.
- **Stale run-dir trap** — auto-discovered old checkpoints propagate their
  encoding over the configured one; clean/isolate the checkpoint dir per run.
- **Reuse the artifact** — don't re-pretrain on a new host when a deterministic
  bootstrap already exists; rsync it.
