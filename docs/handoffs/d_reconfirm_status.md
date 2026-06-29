# D-RECONFIRM — status (re-measure A1 + WR lineage on the corrected multi-window instrument)

**Premise (from D-DECODE):** the deploy head ran a SINGLE-window action space mismatched to the
multi-window TRAINING legal-set. So A1's +0.165 and the whole deploy-matched WR/stall lineage were
measured on a HANDICAPPED instrument. D-RECONFIRM re-measures on the corrected multi-window head.

**Branch** `phase4.5/d-solver` @ b81a059. **Compute:** vast `/root/hexo_rl` (master checkout + d-solver
source rsynced + `maturin develop --release`; multi-window decode + `engine.TacticalSolver` both verified).
The longrun at vast `/workspace/hexo_rl` is UNTOUCHED (invariant); lineage ckpts referenced read-only.

## Instrument fix (LANDED, tested)
`scripts/eval/run_a1_solver_backup.py`:
- `--legal-set` threads `legal_set=True` into BOTH arms via `DeployHeadBot(legal_set=)` →
  `run_gumbel_on_board(legal_set=True)` = the multi-window no-drop decode the net trained under
  (`infer_batch_per_cluster` / `expand_and_backup_ls`). Original A1 ran `legal_set=False` (single-window).
- `--n-sims-full` / `--gumbel-m` override (R2 same-instrument): d1m lineage embeds **100 sims at
  95k–200k vs 150 at 272357**; fix all to 150 (the D-LADDER Gumbel@150 deploy regime).
- Regression test `tests/eval/test_a1_legal_set_flag.py` (3 pass).

## R1 — solver lift on the corrected instrument
**SMOKE (n=40, multi-window, ckpt 272357 vs SealBot-d5):** baseline 0.500 → backup_d6 **0.750**,
paired **delta +0.250, 95% CI [+0.125, +0.400]**, P(>0)=1.000, n_fired 10/40 → **LIFT_IN_WINDOW**.
→ **LIFT-HOLDS**: decoding alone does NOT recover the offense (mw baseline still 0.500); the solver lift
is search-bound → solver build justified. (CI lower +0.125 clears the ~+0.12 pre-reg threshold; mw delta
≥ original single-window +0.165.) Framing correction (red-team): this is a FRESH measurement, NOT a
"lower bound" on +0.165 — `legal_set=True` shifts the policy renormalization denominator in both arms,
which can change which games fire (n_fired).
**Caveat:** 1 in-window FALSE PROOF at n=40 (seed 20260643; off-window guard did NOT fire → in-window
SealBot-probe blemish, not off-window). Quantify rate at n=200.
**FULL (n=200, multi-window, ckpt 272357 vs SealBot-d5):** baseline 0.490 → backup_d6 **0.685**, paired
**delta +0.195, 95% CI [+0.140, +0.255]**, n_fired 43/200, P(>0)=1.000 → **LIFT_IN_WINDOW**. CI lower
+0.140 clears the ~+0.12 threshold; tighter than & consistent with the original single-window +0.165
CI[0.11,0.22]. **R1 = LIFT-HOLDS confirmed.** Multi-window baseline (0.490) ≈ original single-window
(0.470) — decoding doesn't change WR vs the in-window SealBot opponent; the lift is the solver, search-bound.

**FALSE-PROOF finding (SealBot probe, NOT the native solver):** 10 in-window false WIN proofs / 147
override-games (6.8%); only 1 off-window skip. ALL 10 have board max|coord| ∈ [11,17] — just past the
single-window boundary (window_half=9), far below the colony guard (60). NOT array OOB (11–17 is valid in
SealBot's [140][140]); it's SealBot's SINGLE-WINDOW eval mis-modeling SPREAD/multi-cluster geometry —
proving a win on a board spanning >1 window. The `window_half=9` guard checks the proven MOVE's offset, not
the BOARD's spread → misses these. Consequences: (a) these drag the +0.195 DOWN → the SOUND lift is
≥ +0.195; (b) the native `TacticalSolver` is engine-native multi-window-correct with sound WIN proofs (R3)
→ IMMUNE. **Action: route the deploy-backup through the native solver probe (the `solver_probe` DI hook),
not the interim SealBot probe; or add a board-spread guard (delegate when board spans >1 window).**

## R2 — WR lineage on the multi-window head (STALL-HOLDS vs STALL-LIFTS)
All 6 lineage ckpts located (d1m `longrun_v6_live2_ls_gumbel_m16`): vast
`/workspace/hexo_rl/checkpoints/longrun_m16/checkpoint_000{95,120,150,175,200}000.pt` +
`/workspace/hexo_rl/checkpoints/checkpoint_00272357.pt` (local dups under reports/d_decide_2026-06-24/).
Trap: `_archive_golong_kill` 95k is a DIFFERENT killed run — DO NOT USE.
**RESULT = STALL-HOLDS.** Baseline-only deploy WR vs SealBot-d5, FIXED 150 sims, n=100:

| step | single-window | multi-window |
|------|---------------|--------------|
| 120k | 0.530 | 0.575 |
| 175k | 0.570 | 0.545 |
| 272k | 0.500 | 0.530 |

Both curves FLAT ~0.50–0.58 across 120k→272k (binomial SE ≈0.05; every step-to-step and sw-vs-ls diff is
within ~1 SE → no monotonic gain). The corrected multi-window instrument does NOT unmask lineage gains the
single-window eval masked — both show the D-LADDER TRUE-STALL, net at ~PARITY with SealBot-d5 regardless of
decoding. → the stall is real in the net; the off-window defect was a deploy artifact, not the stall; the
solver (R1) is the lever, not more training. (Matches the pre-registered most-likely outcome.)

## R3 — native solver recursive-LOSS soundness (the training-z GATE) — **SETTLED: NOT_CLEARED as-is, CLEARABLE with a ~6-line guard**
Static audit = RISK_FOUND. The dynamic 15-position run ("0 false-LOSS, exit 0") was an **INVALID CLEAR**
(adv positions returned native UNKNOWN before the all-lose branch; "dropped saves" were checked with the
SAME native solver = circular; the independent brute oracle was budget-starved to UNKNOWN on EVERY
not-in-check loss → confirmed/refuted nothing). Opus/high confirmation agent then SETTLED it:

- **CODE FACT confirmed:** `ordering.rs:61-69` not-in-check branch = `threat_moves(stm) ∪ threat_moves(opp)`
  only (no legal fallback, drops quiet/open-four saves); `search.rs:142-148` = `if saw_unknown {UNKNOWN}
  else {LOSS}` with **no exhaustiveness guard**. In-check branch IS loss-complete (a quiet non-block loses
  to the standing win-in-1).
- **but current not-in-check LOSS outputs are EMPIRICALLY SOUND** (budget-adequate, properly-ordered
  full-legal brute confirms the canonical positions — single open-four, cross-fours double-open-four,
  double-four — as TRUE losses, not exhausted; the defensible control → UNKNOWN). The prior brute failed
  only on MOVE-ORDERING (30M-node blowup), not the candidate set.
- **WHY sound (prune symmetry — the load-bearing finding):** native concludes a not-in-check LOSS ONLY via
  a forcing ≤win-in-2 attack. The complete defense vs a forcing win = occupy the win cell (`winning_moves
  (opp)`) or the 5-window-creating cell (`threat_moves(opp)`) = exactly the candidate set's pre-emption
  term. A quiet developmental save provably can't beat a forcing win. The would-be-unsound class (win via a
  NON-threat fork cell) is UNREACHABLE — the prune drops the fork cell for the ATTACKER too → native
  returns UNKNOWN (`ncand=0`), never a false LOSS. **The missing guard's hole is masked by this symmetry.**
- **z_gate: NOT_CLEARED AS-IS** — soundness here is an UNENFORCED EMERGENT property, not an invariant, and
  two hazards sit on the unguarded surface: (1) `cand_cap=40` TRUNCATION — not-in-check lists
  `threat_moves(stm)` first, so a dense position (>40 stm-threats) truncates the defensive blocks → false
  LOSS (latent edge, no clean live trigger built); (2) the **DEFERRED quiet-move alpha-beta body** (Track 3's
  own planned offense body) will add quiet attacker candidates → not-in-check defender nodes in proofs →
  SILENTLY BREAK the emergent soundness and poison z. Don't wire z-LOSS on a fragile unguarded property.
- **CLEARABLE — minimal fix** at `search.rs:142-148` (capture `in_check`, `moves_len` before the loop):
  conclude LOSS only if loss-complete = `(in_check && moves_len < cand_cap) || moves_len >= legal_move_count()`,
  else UNKNOWN (conservative: sacrifices recall not precision). Recall-preserving variant: full-legal-verify
  the dropped `legal \ candidates` moves before emitting LOSS. Extend `soundness_fuzz` to NOT-IN-CHECK
  (open-four-double) positions, not just the in-check `compact_double_threat`.
- **deploy_backup_affected: NO.** A1 `SolverBackupBot` overrides on WIN only; WIN proofs are sound by
  construction (attacker plays only `threat_moves` → defender in-check at every defender node → sound
  `must_block` branch throughout). The guard touches only the LOSS backup. **Deploy/A1 ships now.**
- Artifacts on vast: `/root/hexo_rl/reports/d_reconfirm/r3_final.py` + `r3_final_result.json` (brute
  confirmation), `r3_settle.py` / `r3_construct.py` (prune-symmetry / control evidence).

## Aggregation (ALL ARMS SETTLED)
- **R1 LIFT-HOLDS** (n=200: +0.195 CI[+0.140,+0.255]; sound lift ≥ that — 10 SealBot spread-false-proofs
  drag it down) → finish Track 3 in-window-OFFENSE alpha-beta body (WIN proofs sound) → deploy root hook
  (bench-gated, ≥73k sim/s floor). Deploy-backup ships now (WIN-only) — but route it through the NATIVE
  solver probe (immune to the SealBot spread-false-proofs), not the interim SealBot probe.
- **R3 NOT_CLEARED-as-is, CLEARABLE** → land the ~6-line not-in-check LOSS soundness guard at
  `search.rs:142-148` **before wiring training-z AND as part of the quiet-move body** (the quiet-move body
  breaks the current emergent soundness). Until then, do NOT generate z-LOSS labels from the native solver.
  Current outputs are sound but the property is unenforced — the guard converts emergent→invariant.
  **Build coupling:** the Track 3 offense body and the LOSS guard must land together — the body that
  justifies R1 is the same body that breaks R3's emergent soundness.
- **R2** → pending the curve (most-likely STALL-HOLDS; the off-window defect was a deploy artifact, not the
  stall).

## Open / next
- R1-full n=200 paired delta + false-proof rate (poll biyzblpny).
- R2 sw-vs-ls WR curves (poll bsy1iu771).
- R3 confirmation verdict (agent ad1a7680) → lock the z-gate.
- Then: aggregate, update §sprint log, commit (on ask), memory.
