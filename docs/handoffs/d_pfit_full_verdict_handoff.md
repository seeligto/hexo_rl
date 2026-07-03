# D-PFIT — Full Verdict & Agent Handoff (2026-06-30)

Self-contained handoff. The next agent can act on this without prior conversation context.
Read this, then `docs/designs/coupled_valuez_decode_design.md` (mechanism spec) and
`docs/handoffs/l1_backup_shipping_runbook.md` (execution detail).

---

## TL;DR
The training-z "policy-z is dead, retire the GPU-week" kill was a **measurement artifact** (single-window
decode). The real lever is **coupled**: policy-z + multi-window no-drop decode recovers the bulk; a hard
~31% residual is value-bound. **Decision (user-confirmed): HOLD L2 (value-restart), SHIP L1 (decode/policy)
+ deploy-backup.** Perf body + WS1 decode fix are **built, committed, and exhaustively validated on vast.**
The real remaining lever is **WS3** (native-solver visit-injection in self-play + the L1 GPU-day smoke).

## 1. The question
Is training-z viable — can the solver, by surfacing saving moves in self-play, teach the net to find them
WITHOUT the solver at deploy? Cheapest falsifier: can the policy head **represent** the solver-proven saving moves?

## 2. Core verdict — the kill was overturned; the lever is COUPLED
Verified three ways (P1a/P1b probes + fresh REVIEW + adversarial RED-TEAM, all re-counted):
1. **Policy head already represents the saving moves** — frozen-trunk in-sample fit 32/32 (rank-0, mass 0.997);
   the *raw* head already ranks them median rank 4, 24/32 top-10. Representability is NOT the bottleneck.
2. **The Gumbel search dilutes them under the blind value.** The original "31% flip → kill" was measured through
   a SINGLE-window decode harness. Forcing the prior at the real MULTI-window deploy root flips **59–69%**
   (control 4/32 → forced 19/32 → floor 22/32; re-counted from `redteam_forced_prior_rows.json`).
3. **A hard ~31% residual (10/32) is value-bound** — even a maxed prior can't flip; the blind value's completed-Q
   rates the proven-save line ≤ the blunder (saving-child net **−0.75** vs blunder **+0.31**), robust to n150→400 /
   m16→32 sweeps (rescues 0/10).

⇒ Lever = **policy-z + multi-window no-drop decode (for the ~69%) + value-z (for the ~31% residual)**, not
policy-z alone. Corroborates D-LOCALIZE (value-target) + D-TACTICAL.

## 3. Lever-by-lever
| Lever | Verdict | Evidence |
|---|---|---|
| **L1 — decode/policy** | **GO, architecture SOUND** | Multi-window-aware fit drives decoded root prior 0.09→0.98 on 20/20 stranded traps; in-sample flip = oracle 59%. Stranding was a single-window *targeting* artifact, not head capacity. **Free training-free lift: `legal_set=True` decode 16%→22%.** Held-out static LOO fit ≈16% — but that's a memorization-regime instrument; self-play on-policy regen is the real test → GPU smoke justified, risky on generalization, use **soft** visit-injection (one-hot is collaterally destructive). |
| **Deploy-backup** | **GO, validated** | +0.165/+0.195 in-window (D-SOLVER A1, `solver_backup_bot.py`). Ships now; permanently carries the deep value-bound tail as deploy-time computation. |
| **L2 — value-restart** | **HELD** | Cheap Stage-0 gate delta = +0.000, mean_v canary fired (−0.281). BUT it tested a weak *proxy* (mate-in-1 planes 100% dead at mate-in-3..9-turn traps) → can't cleanly kill L2, but **proves L2 isn't cheaply de-riskable**: needs the expensive real solver-plane build, into a ~15%-ceiling lever with the D-FULLSPEC representational wall still live. |

## 4. Decision (user-confirmed 2026-06-30)
**Hold L2, ship L1 + deploy-backup.** Bank the low-risk bankable WR; don't sink an expensive build + GPU-week
into a ~15%-ceiling lever with zero cheap positive signal and an anti-correlation warning already firing.

## 5. Built, committed & validated
Branch `phase4.5/d-solver`, HEAD **`d9cb08b`** (pushed to `origin` AND direct to vast `/root/hexo_rl`).
- **Native perf body** (commits `f1848f4`,`206af59`,`b72cd8b`,`c79bd5b`,`d599e22`): scored α-β + mate-distance,
  729-entry pattern eval, generation-aged 2-slot TT, PVS/LMR/aspiration + killers/history + ID, net-policy
  ordering (INERT by default). `engine/src/tactics/**`. Proof core reads net NOWHERE; net-ordering reorders
  only, never concludes. **Validated on vast: `47 passed; 0 failed; 0 ignored` incl. both exhaustive full-width
  soundness sweeps (~2.4h on the 5080), 0 false proofs.** Two-tier oracle (scored vs 3-valued vs brute) +
  randomized verdict-invariance fuzz green.
- **WS1 decode fix** (commit `d9cb08b`): `DeployStrengthEvaluator` derives `legal_set` from the encoding
  `policy_pool` (`needs_no_drop_bot`) → promotion gate decodes MULTI-window for `v6_live2_ls`, bitwise-unchanged
  for single-window encodings. Closes the train↔deploy decode mismatch (free 16%→22% lift). Regression test
  `test_deploy_strength_legal_set_derived_from_encoding` green.

## 6. Repo / branch / vast state (operational)
- **Branch:** `phase4.5/d-solver` @ `d9cb08b`. Untracked durable docs/scripts in the working tree (design spec,
  runbook, `scripts/dpfit_*.py`) — uncommitted; commit on user ask.
- **Perf-body worktree** `.claude/worktrees/wf_02f2bc67-5fa-1` — merged, prunable.
- **vast** (`ssh vast`, RTX 5080 / 24 cores, root@/root): build worktree `/root/hexo_rl-dsolver-ws2` @ `d9cb08b`.
  Main tree is on `master`, DIRTY (150 files, 4 stashes) — **do not touch it.**
  - ⚠ **vast can't fetch from GitHub (auth broken).** Sync laptop→vast DIRECT: `git push vast:/root/hexo_rl <branch>`
    (push a NON-checked-out branch only; remove the worktree first if it holds the branch).
  - vast `cargo` is login-shell only: `ssh vast 'bash -lc "cd /root/hexo_rl-dsolver-ws2 && cargo ..."'`.
- **Native solver** `engine/src/pyo3/tactics.rs` exposes `#[pyclass name="TacticalSolver"]` (`PyTacticalSolver`)
  — Python-callable, already built.

## 7. What remains
### WS3 — the actual lever (GPU-day on vast)
Wire native-solver **SOFT visit-injection on PROVEN traps** (NOT one-hot — collaterally destructive) +
multi-window no-drop decode into self-play (`engine/src/game_runner/worker_loop/inner.rs` `finalize_game`
~line 1183 / move-select hook, behind the ~6-line R3 LOSS guard per D-RECONFIRM; visit-injection only, net
never guides the proof — blind net is 0/14). Warm-start `checkpoint_00200000` (clean, no input-plane change).
**Smoke (5–10k steps, ~1 GPU-day, vast):** PASS = GAME-DISJOINT held-out in-window trap-flip **≥25%** (real
margin over control 12% / single-window 16%) AND deploy off-window forced rate holds 0.0 AND threat-probe
C1–C3 PASS. KILL = held-out flip ≤16% (decode-recovers-stranded wager false → deploy-backup carries those).
Broader-goal success metric = **STANDALONE net strength, deploy-backup OFF** (Z2 discipline — never
backup-vs-SealBot), n≥200 DISTINCT games (inject opening diversity — g=0 deterministic collapses to ~2
games/pair, D-ARGMAX), distinct-game bootstrap BT-Elo, ModelPlayer false-clear co-gate on off-window.

### WS2 — Z1 deploy-backup native swap (OPTIONAL, low value, laptop, no bench)
Inject `PyTacticalSolver` as `solver_probe` in `solver_backup_bot.py` (already DI-injectable). Removes SealBot's
colony-OOB phantom-mate risk. **But SealBot backup already ships (+0.165)**, and the swap needs a PARITY check
(does the native threat-space solver prove ≥ SealBot's WINs? the 8% one-primitive ceiling is a coverage risk).
No `make bench` (Python + already-built tactics, not a hot path — the runbook's "mandatory bench" was wrong).

### L2 — HELD (do not fund now)
Revisit only via: build the REAL solver-derived `forced_loss/win_within_2` planes (not the dead cheap proxy) +
mandatory precision audit (FP≤5% on a won-bank — the mean_v warning makes this load-bearing) + re-run Stage-0
pairwise-rerank on the REAL planes with the expanded corpus (~140 new distinct-game pairs minable offline via
`scripts/dpfit_stage0_rerank.py --expand-scan`), THEN the gated GPU-week. D-FULLSPEC wall still live.

## 8. Artifacts
- Spec: `docs/designs/coupled_valuez_decode_design.md` (§9 = gate results; §8 = review corrections).
- Runbook: `docs/handoffs/l1_backup_shipping_runbook.md`.
- Cheap-gate scripts: `scripts/dpfit_{export,fit,search_mechanism,l1_mwfit_probe,stage0_rerank}.py`.
- Reports (gitignored, durable on disk): `reports/d_tactical_2026-06-26/{corpus.jsonl, dpfit_p1b_report.md,
  l1_mwfit_probe_report.md, stage0_rerank_report.md, redteam_forced_prior_rows.json}`; data `data/dpfit_traps.npz`.
- Memory: `d-pfit-verdict-coupled-lever`, `d-pfit-coupled-mechanism-spec`, `d-pfit-p1a-policy-head-already-represents`.
- Anchors: `checkpoint_00200000` (L1 warm-start), `bootstrap_model_v6_live2.pt` (L2 restart if ever).

## 9. Honest risk + operating discipline
**Biggest risk:** L1's self-play smoke lands at the memorization floor (~16% held-out) rather than the in-sample
57–59% — net internalizes the saving prior in-distribution but doesn't generalize. If so, the deploy-backup stays
the permanent mechanism and "standalone-strong net" is out of reach via this path. The program is downside-capped:
~2 days of cheap gates settled this, zero GPU-week spent on the speculative restart, bankable wins on the branch.

**Operating discipline (a thermal incident happened this session):**
- Heavy/sustained Rust, `make bench`, LTO/release, and ALL GPU → **vast only.** The laptop (Ryzen 7 8845HS +
  RTX 4060 Max-Q) hard-cuts power under sustained Rust load — it restarted mid-session from a long build agent.
- Laptop work: **serial only, `-j4` debug**, never parallel heavy agents.
- Commit / push / merge only on explicit user ask.
