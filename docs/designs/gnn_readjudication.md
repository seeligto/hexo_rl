# GNN axis-graph re-adjudication (D-L WP2, 2026-07-13)

## Purpose
Sanctioned re-proposal record for the strix **axis-graph architecture** question. Per the
CLAUDE.md re-validation discipline + the falsified-register protocol
(`docs/07_PHASE4_SPRINT_LOG.md` § "Falsified Hypotheses Register"), a register entry may only
be amended through a dated document that (a) quotes the original grounds verbatim, (b) states
the exact context the prior was killed in, (c) tests whether that context transfers to the
current objective, (d) proposes a history-preserving amendment. This is that document. It
**re-proposes only**; the run3 GNN decision itself is the operator's at the D-L convene (after
WP1 + WP3 land).

## 1. What was actually killed — and what was NOT

Two distinct strix artifacts, two distinct dispositions. Conflating them is the error this
document exists to prevent. The D-L dispatcher's shorthand "the GNN fork kill … register entry
amended from 'falsified' to 're-opened'" is imprecise: **the architecture was never in the
falsified register.** Only the kernel was.

### 1a. The CUDA kernel — FALSIFIED (register row, §D-STRIX S3, 2026-07-02). Verdict STANDS.
Verbatim (register, `docs/07_PHASE4_SPRINT_LOG.md` line 566):
> §D-STRIX S3 (2026-07-02) | Custom CUDA kernel (hexo-strix/Vladdy pattern) would speed HeXO
> forward path | §D-STRIX S3 verdict, RED-TEAM confirmed | Their kernel solves GNN
> variable-size ragged batching — a problem HeXO's dense CNN+attention does not have.
> K-cluster multi-window batching varies batch COUNT, not tensor SHAPE (fixed-geometry windows
> `torch.cat`'d on batch dim; `hexo_rl/selfplay/inference.py`) — the standard variable-N case
> cuDNN/flash already handles. If forward-throughput ever binds: torch.compile → smaller net →
> quantized eval, in that order.

Scope: a **performance** optimization (ragged-batch kernel). Grounds: HeXO's dense CNN+attention
has no ragged-batching problem to solve. This is **orthogonal** to whether the axis-graph
**representation** is a good policy inductive bias. The D-K tournament does not touch the
ragged-batching premise. **Kernel verdict unchanged; not re-opened.**

### 1b. The axis-graph representation — NEVER FALSIFIED. Banked NOTE-ONLY, restart-gated.
Verbatim (§D-STRIX portability verdicts, 2026-07-02):
> Axis-graph line-topology input: NOTE-ONLY banked representation card (restart-gated; adjacent
> to D-FULLSPEC ENTANGLED_R but frozen-trunk context does not formally falsify from-scratch).

The representation was **explicitly not falsified** — "frozen-trunk context does not formally
falsify from-scratch." It was deferred (restart-gated) and never tested. There is no
falsified-register row for the architecture; the only row is the kernel (1a).

**Consequence for the register: there is nothing to move from "falsified" to "re-opened" for the
architecture, because the architecture was never in the register.** What changes status is the
NOTE-ONLY banked card: `deferred / restart-gated` → `re-opened pending probe`.

## 2. What changed since the bank (2026-07-02 → 2026-07-13)

The D-K bridge tournament (2026-07-12, `reports/tourney/`) produced the first direct **measured**
evidence on the axis-graph net's playing strength on shared HTTT ground:

- **Deploy bar (Gumbel search, `TOURNAMENT.md`):** strix-g128 ranks **#1 of 7** at **+313 Elo**
  [+266, +364], 87.2% score — field ceiling by a wide, CI-separated margin over kraken-puct-200
  (+112) and mantis-261k-g150 (+34).
- **Raw-policy bar (argmax, no search, `argmax/ARGMAX_FINAL.md`):** strix-raw ranks **#1 of 6**
  at **+121 Elo** [+88, +159], 69.1% — vs mantis-261k-raw at **−108**, a **+229 Elo raw gap**.
  This isolates policy-head quality from search: a large part of strix's strength lives in the
  **net**, not only in Gumbel search.
- **Scale + cost:** **222,146 params** (0.22M — 19.1× smaller than our 4.25M production net),
  **33.8 ms/turn** raw argmax. A small, cheap net.

**Fidelity caveat (load-bearing — the re-open rests on the corrected number).** The raw-policy
strix arm initially scored −512 Elo / rank-last. That was a 2-stone **turn-assembly artifact**:
the child's second stone was drawn from a stale `moves_remaining=2` forward that never saw
stone-1. Fixed by a per-stone re-forward, re-verified **18/18 = 100% turn-level** against
strix's own raw deploy (`strix_argmax_verify.md`, `ARGMAX_FINAL.md`). The +121 figure is the
corrected, fidelity-verified number; the stale −512 is void.

## 3. Does the kill context transfer? (re-validation test)

Per CLAUDE.md, a prior must be re-validated against the current objective before it can gate.

- **Prior 1a (kernel):** its context = ragged-batch forward-throughput. Objective now =
  representation-as-policy-inductive-bias. **Context does NOT transfer** — the kernel kill says
  nothing about representation quality; it neither supports nor blocks the re-open. Verdict:
  kernel row stands, untouched, scoped.
- **Prior 1b (bank):** its context = frozen-trunk, from-scratch NOT falsifiable at the time, and
  no measured strength read existed ("no diagnostic ever isolated it"). Objective now = the same
  architecture question, now **with** a measured strength read (D-K). **Context transferred, and
  the blocking condition (no evidence) is now removed.** The deferral basis is now addressable by
  a cheap probe. Re-open justified.

## 4. Ruling proposed

1. **Falsified-register kernel row (§D-STRIX S3, line 566): NO CHANGE to verdict.** Add a
   one-line **scope footnote** so the row cannot be misread as an architecture kill: the row is
   CUDA-kernel-scoped (ragged-batch perf); it does NOT adjudicate the axis-graph representation.
2. **Axis-graph representation card: status advanced** `NOTE-ONLY banked / restart-gated` →
   **`RE-OPENED pending D-L WP3 probe`**. Recorded as a dated §D-STRIX addendum (nothing deleted;
   the 2026-07-02 bank text stays). The re-open asserts only that the architecture question is
   now worth a discriminating probe — **NOT** that the architecture works.

WP3 (axis-graph BC-prefit: GNN-BC vs CNN-BC, matched corpus/protocol/steps —
`docs/designs/gnn_bc_probe_design.md`) is the discriminator. Its frozen verdicts decide the
card's fate:
- **ARCH-DOMINANT** (gnn-bc ≥ cnn-bc by ≥ +100 BT Elo, CI excl 0, AND gnn-bc ≥ mantis-261k-raw)
  → representation effect real on our data → run3 primary variable re-opens (operator decision).
- **ARCH-NULL** (gnn-bc ≈ cnn-bc) → strix's edge is recipe/data/scale, not architecture → run3
  as specced (CNN + dist head); card returns to bank, "re-opened-unproven".
- **MIXED** → card becomes run4 pre-registered card #1 with this probe as its evidence base.

## 5. Register discipline
This document is the **only** sanctioned path to the re-proposal. History preserved: the kernel
verdict and the original 2026-07-02 bank text are unchanged; only a dated addendum + a scope
footnote are added. The re-open is provisional and probe-gated — it does **not** itself change
run3's primary variable. That decision is the operator's at the D-L convene, after WP1 + WP3.

## Provenance
- Register row + §D-STRIX audit: `docs/07_PHASE4_SPRINT_LOG.md` (line 566; §D-STRIX ~line 2840).
- Deploy tournament: `reports/tourney/TOURNAMENT.md`.
- Raw-policy tournament (fixed): `reports/tourney/argmax/ARGMAX_FINAL.md`; fidelity fix
  `reports/tourney/strix_argmax_verify.md`.
- Port source: `SootyOwl/hexo-strix @ 031d309` (MIT); local port
  `hexo_rl/bots/strix_v1_{graph,net,bot}.py`.
- WP3 discriminator design: `docs/designs/gnn_bc_probe_design.md` (D-L WP3, this dispatch).
