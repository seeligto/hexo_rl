# D-K TOURNEY — cross-bot round-robin protocol (FROZEN before first game)

Status: **pre-registration / protocol freeze.** Nothing in this doc may be changed after the
first tournament game is played without a loud, dated amendment note. Execution venue = an
ephemeral vast box (RTX 5070); the E1 training box is untouched. Analysis is ours; publishing
anywhere is an operator decision, not this protocol's.

Companion artifacts:
- Adapters + fidelity: `reports/tourney/adapters_strix.md`, `reports/tourney/adapters_shrimp.md`,
  `reports/tourney/strix_g128.md`
- Box provisioning: `reports/tourney/box_provision.md`, `reports/tourney/refserver_deploy.md`,
  `scripts/tourney/box_install.sh`, `scripts/tourney/refserver_deploy.sh`
- Field machinery (gitignored local arena): `scripts/arena/run_arena.py`,
  `scripts/arena/bots/*_child.py`, `scripts/arena/configs/*.toml`, `scripts/arena/book_r5_32.json`

---

## 1. Execution model — bridge, not headless

The tournament runs bot-vs-bot **through hexo-bridge + the hexo-ref-server** (the operator's
preferred stack), NOT a bespoke headless harness. Division of labour:

- **hexo-ref-server** (on the box, `:8080`): game host + rules authority + persistence
  (Mongo `gameHistory`) + live UI. Reachable by the operator over the SSH tunnel
  `ssh -i ~/.ssh/vast_hexo -p 13523 -N -L 8080:localhost:8080 root@ssh7.vast.ai` → `http://localhost:8080`.
- **hexo-bridge**: connects each bot (a stdio child) to a ref-server game as a player.
- **`run_arena.py --rr`**: drives the pairing schedule + fixed paired-opening injection + result
  export. Serial per pairing (dispatch mandate), niced. Idempotent done-markers per
  `(pairing, opening_idx, color)` → crash-resume with no double-count.
- **Analysis (ours)**: export completed games via `/api/finished-games` and run Bradley-Terry +
  bootstrap CIs offline (§6). The ref-server's own tournament subsystem (Swiss/elim brackets, no
  opening control, no per-bot budget) is NOT used to rank — it can't control openings or budgets.

Rationale for this split: the ref-server gives persistence + live UI + the canonical HeXO rules
engine; `run_arena` retains the three things the dispatch's rigor requires that the ref-server
tournament cannot give — fixed paired openings, per-bot native budgets, and deterministic replay.

Each bot's per-bot budget is printed in every result row (it lives in the bot's config TOML).

---

## 2. Field — 7 bots

`sealbot-d6` was dropped by operator decision (2026-07-12); it was also the dominant cost driver
(~10 s/move). The field-manifest is the set of `scripts/arena/configs/*.toml` files — one row = one
bot = one config.

| # | config token | display name | family | budget | weights / source |
|---|---|---|---|---|---|
| 1 | `mantis261k` | `mantis-261k-g150` | Gumbel-SH (root Gumbel + PUCT interior) | 150 sims, g=0 argmax, t0 | `checkpoints/run2_final/checkpoint_00261500.pt` (run2 final, v6_live2_ls multi-window) |
| 2 | `mantis248k` | `mantis-248k-g150` | Gumbel-SH | 150 sims, g=0 argmax, t0 | `checkpoints/d1m_vast_latest/checkpoint_00248000.pt`, **decoded multi-window** (`--decode-as v6_live2_ls`; native is single-window v6_live2 — see §8) |
| 3 | `sealbot_d5` | `sealbot-d5` | minimax | depth 5 | SealBot (vendored) |
| 4 | `kraken_raw` | `kraken-raw` | raw-policy argmax | 0 sims, t0 | `checkpoints/external/kraken_v1.pt` |
| 5 | `kraken_mcts` | `kraken-puct-200` | **AlphaZero PUCT** (PUCT_C=1.0, KataGo FPU, Dirichlet) | 200 sims, t0 | `checkpoints/external/kraken_v1.pt` |
| 6 | `shrimp` | `shrimp-g256` | Gumbel (shrimp's own Rust MCTS, delegated) | 256 visits, t0 | `hexo-bot/models/shrimp_main7_infer.pt` (**placeholder**; real ckpt swaps in via `--checkpoint`) |
| 7 | `strix` | `strix-g128` | Gumbel-AZ (strix's own Rust `hexo_rs`, delegated) | 128 sims, m=16, t0 | `strix_checkpoint_00237000.pt` |

Search-family naming convention: `-g<N>` = Gumbel at N sims; `-puct-<N>` = AlphaZero PUCT at N
sims; `-raw` = raw-policy (0 sims); `-d<N>` = minimax depth N.

**Delegation bots (shrimp, strix)** run the competitor's OWN search code in a network-off sandbox
subprocess (trust relaxed per operator: shrimp + strix repos declared safe). Faithful by
construction. Each needs its own env provisioned on the box (§7). kraken-raw/kraken-puct/mantis run
our own code (kraken adapters are our reimplementations; mantis is our net).

---

## 3. Format

- **Full round-robin.** C(7,2) = **21 pairings**.
- **Book:** one fixed fair radius-5 opening book, 32 paired openings, versioned at
  `scripts/arena/book_r5_32.json` (seed 42, 4 plies each; 4 central stones on a radius-5 disk —
  no 6-in-a-row reachable from the opening). One book for all pairings.
- Each opening played **both colors** → 2 games/opening/pairing.
- **Games = 21 × 32 × 2 = 1344.**
- Turn clock: ref-server-enforced (45 s/move in smoke). Delegation bots get wider bridge timeouts
  (300 s) — search is heavier; this is a timeout ceiling, not a budget.

---

## 4. Determinism, dedup, escalation

Most bots are deterministic (argmax / temp-0) → a fixed opening can collapse to few distinct
trajectories (CLAUDE.md §D-ARGMAX effective-n corollary). Therefore:

- **Trajectory-hash dedup per pairing.** Report the count of DISTINCT games per pairing alongside
  the raw 64.
- **Escalation (pre-registered):** any pairing whose distinct count drops **below 24** gets the
  book escalated for that pairing only — 4-ply → deeper opening plies — applied loudly and recorded.
- The BT CI (§6) is bootstrapped over **distinct opening-pairs**, so byte-identical replays cannot
  manufacture false confidence.

---

## 5. Per-bot budget — printed every row

Every result row prints the acting bot's budget from §2 (e.g. `kraken-puct-200 @ 200 sims/t0`).
No bot is silently re-budgeted. The two mantis snapshots share the native deploy budget
(Gumbel-150, g=0, t0) — the ONLY difference between them is the checkpoint (261.5k vs 248k) and
248k's decode override (§8).

---

## 6. Ranking

- **Primary: Bradley-Terry MLE** over game results (win=1, loss=0, **draw = 0.5 to both, symmetric**).
- **CIs: bootstrap resampled at the OPENING-PAIR level** (not the game level — resampling games
  double-counts copies of a deterministic trajectory and over-narrows). Red-team must confirm the
  bootstrap resamples pairs (§ WP5).
- **Secondary: raw pairwise win-rate matrix** with per-pair CIs.
- **Headline = the BT ordering WITH CIs.** Overlapping CIs are written **exactly** as
  "indistinguishable." No point estimate is reported as a gap when its CI overlaps a neighbour's.

---

## 7. Cost estimate + calibration gate

**Measured inputs** (CPU laptop, from WP2 smoke `reports/tourney/wp2_smoke.md` — real end-to-end
per-game walls vs sealbot-d5, which is the ~0-cost side, so wall ≈ that bot's own per-game cost +
fixed bridge/ref-server overhead ≈ 4 s):

| bot | measured/est. active s/game | source |
|---|---|---|
| kraken-puct-200 | ~45 | measured (49 s vs d5 − overhead) |
| mantis-261k-g150 | ~12 | measured (16 s vs d5) |
| mantis-248k-g150 | ~8 | measured (~12 s vs d5) |
| shrimp-g256 | ~20–30 (est.) | per-move 0.73 s median / 4.5 s p90; opening phase slow |
| strix-g128 | ~15–20 (est.) | per-turn 1.14 s measured × ~13 turns |
| kraken-raw | ~1 | measured (~overhead only) |
| sealbot-d5 | ~0–10 (uncertain) | 3 ms TT-warm median is optimistic; p90 0.8 s; fresh positions cost more |

**Model:** `wall(A vs B) ≈ 4 + active(A) + active(B)`; each bot appears in 6 pairings.
Σ active ≈ 106 s → total ≈ `64 × (21×4 + 6×106)` ≈ **46 k s ≈ 13 hours** on the CPU laptop.

**Two forces move the real number, in opposite directions:**
- **GPU (box has an RTX 5070):** net bots (mantis/shrimp/strix, and kraken net eval) run far faster
  on GPU than these CPU numbers if the children are GPU-configured for the run (currently
  CPU-defaulted). This pulls the total DOWN, plausibly to ~6–9 h.
- **sealbot-d5 unknown + shrimp/strix estimated:** d5's true cost on fresh (non-TT-warm) positions,
  and shrimp/strix per-game, are not directly measured end-to-end. This is the main upward risk.

**Gate (pre-registered, honest-number discipline):** the ~13 h CPU figure is an ORDER-OF-MAGNITUDE
upper bound, not a launch number. **Before the full 1344-game burn, WP4 runs a calibration batch**
— 2–3 representative pairings on the box GPU including the worst case (`kraken-puct-200` ×
`shrimp-g256`) and a `sealbot-d5` pairing on fresh positions — to measure the real per-pairing
wall. The operator sees that measured total before authorizing the full run. No blind extrapolation
from d5 or d6 (CLAUDE.md re-validation discipline).

---

## 8. Provenance + caveats (carried verbatim into the report)

- **Competitor single-snapshot caveat (verbatim, every competitor row — strix, shrimp, kraken):**
  "single-snapshot checkpoint, no run logs, not representative of the project's ceiling."
- **mantis-248k decode:** operator chose to deploy 248k **multi-window** (`--decode-as v6_live2_ls`)
  though it trained single-window (d1m lineage, D-FORENSIC F1). This is a decode override, not its
  historical deploy; flagged so its rows are read correctly.
- **Delegation trust:** shrimp + strix run their own untrusted code in network-off sandboxes
  (operator declared both repos safe). Weights loaded `weights_only=True`; archives inspected clean.
- **Source pins (appendix):** strix delegation clone `SootyOwl/hexo-strix` @ `031d309` (WP1 net
  fidelity read `c381ffb`; checkpoint reconstructs clean under both — delegation is self-consistent
  at `031d309`). shrimp source `Cmiller132/hexo-bot` @ `ed520da`. kraken vendored (SHA in
  `reports/anchorx/krakenbot_repo_notes.md`). Book seed 42. Fill exact SHAs in the final report.

---

## 9. Pre-registered expectations (honesty scoring, NOT gates)

- `mantis-261k-g150` ≥ `mantis-248k-g150` (later checkpoint; both multi-window decode).
- `kraken-puct-200` > `kraken-raw` (search over raw policy).
- kraken-puct plausibly top-2 (strong external AlphaZero).
- (dispatch's `d6 > d5` is void — d6 dropped.)
- Any inversion = flag + investigate BEFORE publishing, not silently.

---

## 10. Amendment log

(Empty at freeze. Any change after the first game is dated and recorded here.)
