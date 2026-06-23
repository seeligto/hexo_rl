# D-EXPLOIT Phase 2 — in-window finishing discriminator: RESULT + VERDICT

Date: 2026-06-06. Training-free. Companion: `exploit_probe_20260606.md` (Phase 1). Harness:
`scripts/diagnosis/finishing_sims_sweep.py`. Data: `reports/investigations/
finishing_sims_sweep_vast_fixed.json`. All uncommitted (operator-gated).

---

## 0. TL;DR — VERDICT: POLICY/VALUE DEFECT → O1 reactive condition MET

On the representative fresh sample (281 vast self-play games @ ~54k, 80 unique in-window-miss
positions), a FROZEN 54.5k model re-run at **temp=0 (greedy)** converts only **~30%** of the
available in-window forced wins, and the rate is **FLAT across a 5× sims sweep** (400→2048):

| sims (temp 0) | 400 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| in-window forced-win conversion | 0.287 | 0.300 | 0.300 | 0.300 |

Pre-registered mapping: **NOT exploration** (greedy also fails — rules out temp/noise), **NOT
search budget** (5× sims doesn't help) → the finishing gap is a **policy/value defect** → **O1's
REACTIVE condition is met** (a real, observed finishing failure the run cannot self-correct).
Per the pre-registration: **re-judge O1 on a FINISHING metric** (this in-window conversion +
game length), NOT WR — still gated, still pre-registered, NOT an auto-flip.

---

## 1. What was tested

Extract the in-window forced-win positions the recorded self-play (400 sims, temperature +
Dirichlet noise) did NOT convert that turn (dedup by stone-set + win-cell set → kills recurrence
inflation), then re-run a frozen model's MCTS at **temp=0** across {400, 512, 1024, 2048} sims.
"Converts" = the mover completes the win playing the WHOLE turn greedily. Forced-win detection +
off-window flag reuse the canonical `forced_win_detector` (zero geometry literals).

## 2. Red-team that changed the result (the metric fix + the depth-2 discovery)

- **The in-window misses are ~99% depth-2 (2-stone, open-4) forced wins** — only 1 depth-1
  (immediate single-stone) position in the whole fresh set vs 426 depth-2 turns. **These ARE the
  operator's "open 4s"** (an open-4 is a 2-stones/turn forced win). The off-window misses (Phase 1)
  are a separate axis.
- **First-stone-only metric bug (caught + fixed):** the initial sweep scored a depth-2 "conversion"
  by checking only whether the model played a forced-pair FIRST stone — it never verified the model
  COMPLETES the 2-stone win. Fixed to play out the whole turn greedily and check `check_win()`.
- **Sample matters:** an early run on sparse OLDER local games (30k-era) + the buggy metric gave a
  misleading 0.83. The corrected run on the representative fresh 54k sample gives 0.30. The verdict
  rests on the corrected, representative result.

## 3. Reconciliation with the live ~0.83 per-game conversion

No contradiction: the live EMA is PER-GAME (did the game EVENTUALLY convert ≥1 in-window win,
recurrence-deduped at the game level) ≈ 0.83-0.887. This probe is PER-POSITION immediate conversion
≈ 0.30. The model **eventually** wins most games but converts only ~30% of individual open-4
opportunities in the immediate turn → finishing is **inefficient** (games run long, ~83 plies) but
eventually succeeds. The inefficiency IS the O1 finishing signal.

## 4. Interpretation

MCTS at 2048 sims SHOULD trivially find a 2-ply forced win (an open-4 is unstoppable — the opponent
does not move between the mover's two stones). That it doesn't (flat ~0.30) means the policy/value
heads do not rank/recognise the open-4 completion, and search alone (without the heads pointing
there) does not reliably find the winning first stone among ~190 legal moves. A genuine finishing
defect localised to **open-4 / 2-stone completions** — the exact structure the operator flagged.

## 5. Pre-registered NEXT (gated, not executed here)

O1 (forced-win one-hot POLICY target, built default-OFF per `project_o1_forced_win_policy_target`)
now has its **reactive condition satisfied**: a real, observed finishing failure search cannot fix.
Per the pre-registration, **re-judge O1 on a FINISHING metric** (in-window open-4 conversion +
game length), NOT the original WR gate (the wrong instrument). This is a re-judgment to AUTHOR, not
an auto-enable. Cheapest discriminator before any sustained run: a short O1-ON vs O1-OFF smoke
measuring the immediate open-4 conversion (this probe) + game length, not WR.

## 6. Red-team checklist (pre-committed)
- Same positions the operator sees? **YES** — 99% depth-2 = open-4 (2-stone) wins.
- Game length (~83 plies) driven by these? **Partly** — both the off-window misses (Phase 1, ~52%
  of forced-win turns) AND the in-window open-4 inefficiency (~70% not converted immediately) lengthen
  games. Phase 1 = an ACTION blind spot (multi-cluster); Phase 2 = a finishing/policy defect (O1).
- Metric fair? Fixed (full-turn greedy completion vs the engine's own `check_win`); depth-1 vs
  depth-2 split confirmed the set is depth-2-dominated.
