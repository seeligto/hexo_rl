# D-LOCALIZE P3 — Search-Scaling Sweep Runbook (vast 5080)

**Phase:** D-LOCALIZE P3 (deferred to vast operator run)
**Date written:** 2026-06-25
**Branch:** phase4.5/d-decide-track-b
**Harness:** `scripts/eval/gumbel_ladder.py` + `gumbel_greedy_bot.py`
**Do NOT execute on laptop** — 5080 box only; laptop 4060 MQ is thermally unsafe
for sustained multi-hour GPU+CPU concurrent load (SealBot is CPU; net is GPU).

---

## Purpose

D-LADDER Stage 1 identified the "150 sims + Gumbel SH-budget problem": m=16 at
n=150 gives a non-power-of-2 late-heavy profile (visits/candidate/phase =
2/4/10/23) vs the clean 2/4/8/16 of n=128. The OPEN question is whether
increasing n (256, 512) produces stronger play against the external minimax bar
(SealBot@depth-5) or whether the curve plateaus — i.e., whether the model's
TRUE n=150 strength is just search-budget-limited or is a genuine self-play stall.

This experiment LOCALIZES the stall: search-budget artifact vs training plateau.

---

## PREREQUISITE 0 — n-sims override (CODE CHANGE REQUIRED)

**No `--n-sims` flag exists in `gumbel_ladder.py` or `gumbel_greedy_bot.py`.**
`n_sims` is read from the checkpoint's embedded config via `extract_deploy_knobs`
(path `selfplay.playout_cap.n_sims_full`) and applied uniformly to all net bots
via `GumbelGreedyBot.__init__` (`self._n_sims = int(knobs["n_sims_full"])`).
There is no runtime override mechanism.

### Minimal code change — add `--n-sims-override` to `gumbel_ladder.py`

**File:** `scripts/eval/gumbel_ladder.py`

Change 1 — `_build_field` signature: add `n_sims_override: Optional[int] = None`
parameter. After the `knobs = extract_deploy_knobs(cfg)` line, add:

```python
if n_sims_override is not None:
    knobs = dict(knobs)          # don't mutate the original
    knobs["n_sims_full"] = n_sims_override
    print(f"[n_sims_override] n_sims_full overridden to {n_sims_override}")
```

Change 2 — `_cmd_play`: thread `args.n_sims_override` into the `_build_field` call:

```python
bots, order, enc, n_sims, _knobs = _build_field(
    args.ckpt_dir, steps, args.boot8300, args.sealbot, args.sealbot_time_limit,
    args.encoding, args.knobs_from, args.seed_base,
    sealbot_max_depth=args.sealbot_max_depth,
    n_sims_override=args.n_sims_override,   # ADD
)
```

Change 3 — CLI parser: add to the `play` sub-parser:

```python
p.add_argument("--n-sims-override", type=int, default=None,
               help="override n_sims_full from checkpoint knobs (D-LOCALIZE P3 sweep)")
```

**Scope:** 3 lines of code + 1 parser entry. No change to statistics, bot logic,
or game loop. The knobs dict patch is shallow-copy so the original embedded config
is not mutated. Verify with `--n-sims-override 128` smoke (2 games) before the
full sweep: confirm the `[knobs]` print shows `n_sims_full: 128` and
`[n_sims_override]` line appears.

---

## PREREQUISITE 1 — rsync checkpoints to vast

The 6 banked checkpoints must be present on vast at a known path. They were
originally pulled from `/workspace/hexo_rl/checkpoints` on the live vast box
during D-LADDER Stage 1 (PID 1512430, read-only). If still present:

```bash
# On vast box — verify they exist
ls -lh /workspace/hexo_rl/checkpoints/checkpoint_000{60000,120000,150000,175000,200000}.pt
ls -lh /workspace/hexo_rl/checkpoints/checkpoint_00226500.pt   # or closest 226.5k step
```

If the live run has evicted or overwritten them, rsync from the local archive.
See `docs/handoffs/longrun_a1_vast_consolidation_runbook.md` for the ssh alias.

```bash
# From laptop, rsync TO vast (if needed)
rsync -avz --progress \
  reports/d_ladder_2026-06-24/ckpts/ \
  vast:/workspace/hexo_rl/reports/d_ladder_2026-06-24/ckpts/
```

Also confirm the knobs-from ckpt (full training ckpt with embedded config —
`checkpoint_00120000.pt` was validated as the knobs source in Stage 1):

```bash
# Quick knobs check on vast
cd /workspace/hexo_rl
.venv/bin/python -c "
from scripts.eval.gumbel_greedy_bot import load_state_and_config, extract_deploy_knobs
_, cfg = load_state_and_config('reports/d_ladder_2026-06-24/ckpts/checkpoint_00120000.pt')
print(extract_deploy_knobs(cfg))
"
# Expected: {'gumbel_m': 16, 'n_sims_full': 150, ...}
```

---

## PREREQUISITE 2 — bank the Stage-1 net-vs-net games

D-LADDER Stage 1 produced 1120 games in `reports/d_ladder_2026-06-24/per_game.jsonl`
(or equivalent path). The net-vs-net pairs (15 pairs from 6 checkpoints) are
deploy-deterministic at n=150 and ARE ALREADY BANKED. For the BT ladder at each
n, we need full-field games — but the KEY signal (SealBot WR vs n) requires only
the 6 net-vs-SealBot pairs per n. Strategy: use `--sealbot-only` to play only
the 6 net-vs-SealBot pairs at each n, then merge with the banked n=150 net-vs-net
for the BT fit. This halves wall time.

HOWEVER: for the BT fit to be valid, the banked net-vs-net rows must use the
SAME label schema and be concatenated cleanly. The `--sealbot-only` flag in
`play_round_robin` already filters to `"sealbot" in (a, b)` — the resulting
jsonl is a strict subset of the full-field jsonl.

**Merge command (aggregate step):**
```bash
cat reports/d_ladder_2026-06-24/per_game.jsonl \
    reports/d_localize_p3/n128/sealbot_games.jsonl \
  > reports/d_localize_p3/n128/full_field.jsonl
```
(Repeat per n.)

---

## SH budget profile (reference)

m=16 allocator: `sims_per = remaining / (remaining_phases * candidates)`.
Phases eliminate: 16→8→4→2 candidates.

| n    | Phase1 v/c | Phase2 v/c | Phase3 v/c | Phase4 v/c | Profile shape      |
|------|-----------|-----------|-----------|-----------|---------------------|
| 128  | 2         | 4         | 8         | 16        | clean power-of-2    |
| 150  | 2.3       | 4.7       | 9.4       | 18.8      | current deploy (non-canonical) |
| 256  | 4         | 8         | 16        | 32        | clean power-of-2, 2x depth |
| 512  | 8         | 16        | 32        | 64        | clean power-of-2, 4x depth |

n=128: slightly FEWER phase-1 visits than n=150 (fixes late-heavy skew, loses
~3% budget). n=256: 1.7x training overhead if adopted (9-day→~15-day per D-LADDER
findings). n=512: 3.4x overhead (for reference only; not a realistic adoption
candidate without architecture changes).

---

## PRE-REGISTERED GATE (set BEFORE running; do not adjust post-hoc)

**Decision-relevant comparison: n=150 → n=256**

Because 128→150 is partly the phase-1-starvation fix (non-power-of-2 budget
correction), not raw search depth. The relevant adoption decision is 150→256.

### Rule:

**PLATEAU-by-150:**
  WR curve (SealBot@depth-5 win-rate) is flat from n=150 to n=256 and n=512 —
  i.e., all three 95% bootstrap CIs on WR(ckpt vs SealBot) STRADDLE the n=150
  value (no CI-clean improvement from 150→256 for ANY checkpoint in the field).
  → KEEP n=150. The stall is NOT a budget artifact. 256/512 not worth the 1.7x/3.4x
  training overhead. SH-budget non-canonicality is benign at n=150.

**CLIMBS-past-256:**
  At least ONE checkpoint shows a CI-clean WR improvement from n=150 → n=256
  (i.e., CI_low(WR@256) > CI_high(WR@150) for that checkpoint, bootstrap
  over distinct games). n=512 need not also climb.
  → ADOPT n=256 for future eval and relaunch. Flag the 1.7x overhead to the
  operator; suggest a paired training-speed tradeoff assessment.

**Note on 128 vs 150:**
  If WR(ckpt vs SealBot) at n=128 is CI-cleanly LOWER than at n=150, this
  confirms the phase-1-starvation effect is real and n=150 is a local optimum
  in the 128–256 range. Report but does NOT change the 150-vs-256 gate.
  If 128 ≈ 150 (straddles), the non-canonical late-heavy profile makes no
  measurable difference — deploy n=128 is equally valid (cleaner SH semantics).

---

## DISTINCT-GAME EFFECTIVE-N GATE

Per §D-ARGMAX: a greedy/argmax eval regime collapses to ~2 distinct games/pair
without opening diversity. Opening diversity (--opening-plies 4) is the load-
bearing fix. Gate per cell (each {ckpt, n} vs SealBot pair):

- **distinct-n ≥ 10 per pair** (hard gate; if violated, report INSUFFICIENT,
  increase --n-games-per-pair to 60 for that cell and rerun)
- **copy_multiplier ≤ 2.0** (soft; > 2 → investigate opening diversity)
- **color-balanced** (40 games = 20 as P1 + 20 as P2 per pair via gi%2)

Report `distinct_per_pair` from `aggregate.json` for every cell. The bootstrap
CI is trusted ONLY when distinct-n ≥ 10; below that, report Hessian CI with a
LOW-POWER caveat and do not gate on it.

---

## FULL COMMAND SEQUENCE

### Step 0: build / verify environment on vast

```bash
cd /workspace/hexo_rl
git status          # confirm branch = phase4.5/d-decide-track-b or clean master
source .venv/bin/activate
python -c "import engine; print('engine ok')"
python -c "from hexo_rl.bots.sealbot_bot import SealBotBot; print('sealbot ok')"
```

### Step 1: apply n-sims-override code change (see PREREQUISITE 0)

Edit `scripts/eval/gumbel_ladder.py` per the 3-change spec above.
Smoke-verify:

```bash
.venv/bin/python scripts/eval/gumbel_ladder.py play \
  --ckpt-dir reports/d_ladder_2026-06-24/ckpts \
  --steps 120000 --sealbot --sealbot-max-depth 5 --sealbot-only \
  --encoding v6_live2_ls \
  --knobs-from reports/d_ladder_2026-06-24/ckpts/checkpoint_00120000.pt \
  --n-games-per-pair 2 --opening-plies 4 \
  --n-sims-override 128 \
  --out /tmp/smoke_n128.jsonl
# Confirm: [n_sims_override] n_sims_full overridden to 128
# Confirm: [knobs] ... n_sims_full: 128 ...
```

### Step 2: n=128 — SealBot-only games (sealbot pairs only; net-vs-net banked)

```bash
mkdir -p reports/d_localize_p3/n128
.venv/bin/python scripts/eval/gumbel_ladder.py play \
  --ckpt-dir reports/d_ladder_2026-06-24/ckpts \
  --steps 60000,120000,150000,175000,200000,226500 \
  --sealbot --sealbot-max-depth 5 --sealbot-only \
  --encoding v6_live2_ls \
  --knobs-from reports/d_ladder_2026-06-24/ckpts/checkpoint_00120000.pt \
  --n-games-per-pair 40 --opening-plies 4 --seed-base 20260625 \
  --n-sims-override 128 \
  --out reports/d_localize_p3/n128/sealbot_games.jsonl
```

### Step 3: n=256 — SealBot-only games

```bash
mkdir -p reports/d_localize_p3/n256
.venv/bin/python scripts/eval/gumbel_ladder.py play \
  --ckpt-dir reports/d_ladder_2026-06-24/ckpts \
  --steps 60000,120000,150000,175000,200000,226500 \
  --sealbot --sealbot-max-depth 5 --sealbot-only \
  --encoding v6_live2_ls \
  --knobs-from reports/d_ladder_2026-06-24/ckpts/checkpoint_00120000.pt \
  --n-games-per-pair 40 --opening-plies 4 --seed-base 20260626 \
  --n-sims-override 256 \
  --out reports/d_localize_p3/n256/sealbot_games.jsonl
```

### Step 4: n=512 — SealBot-only games

```bash
mkdir -p reports/d_localize_p3/n512
.venv/bin/python scripts/eval/gumbel_ladder.py play \
  --ckpt-dir reports/d_ladder_2026-06-24/ckpts \
  --steps 60000,120000,150000,175000,200000,226500 \
  --sealbot --sealbot-max-depth 5 --sealbot-only \
  --encoding v6_live2_ls \
  --knobs-from reports/d_ladder_2026-06-24/ckpts/checkpoint_00120000.pt \
  --n-games-per-pair 40 --opening-plies 4 --seed-base 20260627 \
  --n-sims-override 512 \
  --out reports/d_localize_p3/n512/sealbot_games.jsonl
```

### Step 5: n=150 reference run (SAME seed, SAME depth-5, to compare apples-to-apples)

The banked Stage-1 SealBot games used time-limited SealBot@0.5s (pre-de-risk).
The de-risk used depth-5 but with a DIFFERENT seed. For a clean within-experiment
n=150 reference, run a fresh depth-5 set:

```bash
mkdir -p reports/d_localize_p3/n150
.venv/bin/python scripts/eval/gumbel_ladder.py play \
  --ckpt-dir reports/d_ladder_2026-06-24/ckpts \
  --steps 60000,120000,150000,175000,200000,226500 \
  --sealbot --sealbot-max-depth 5 --sealbot-only \
  --encoding v6_live2_ls \
  --knobs-from reports/d_ladder_2026-06-24/ckpts/checkpoint_00120000.pt \
  --n-games-per-pair 40 --opening-plies 4 --seed-base 20260628 \
  --out reports/d_localize_p3/n150/sealbot_games.jsonl
# No --n-sims-override (uses embedded n_sims_full=150)
```

### Step 6: aggregate per n

For each n, merge banked net-vs-net (from Stage 1) with the new SealBot games,
then fit the BT ladder:

```bash
# Adjust BANKED_GAMES path to wherever Stage-1 per_game.jsonl lives on vast
BANKED="reports/d_ladder_2026-06-24/per_game.jsonl"
ORDER="s60k,s120k,s150k,s175k,s200k,s226k,sealbot"

for N in 128 150 256 512; do
  SEAL="reports/d_localize_p3/n${N}/sealbot_games.jsonl"
  MERGED="reports/d_localize_p3/n${N}/full_field.jsonl"

  # Strip sealbot rows from banked (they used time-limited SealBot) + add new ones
  python -c "
import json, sys
banked = [json.loads(l) for l in open('${BANKED}') if l.strip()]
# Keep only net-vs-net rows (no sealbot label in either side)
net_vs_net = [g for g in banked if 'sealbot' not in (g['p1'], g['p2'])]
seal = [json.loads(l) for l in open('${SEAL}') if l.strip()]
merged = net_vs_net + seal
with open('${MERGED}','w') as f:
    for g in merged: f.write(json.dumps(g)+'\n')
print(f'n={N}: {len(net_vs_net)} net-vs-net + {len(seal)} sealbot = {len(merged)} total')
"

  .venv/bin/python scripts/eval/gumbel_ladder.py aggregate \
    --per-game "${MERGED}" \
    --anchor s120k \
    --order "${ORDER}" \
    --n-boot 2000 \
    --out "reports/d_localize_p3/n${N}/agg"

  echo "=== n=${N} ratings ==="
  cat "reports/d_localize_p3/n${N}/agg/ratings.csv"
done
```

**Anchor:** `s120k` (the promoted baseline, anchored at 0 Elo). This is the same
anchor as Stage 1 (Stage 1 used `boot8300` for the absolute ladder; here we care
only about the RELATIVE SealBot Elo shift across n — a self-anchored ladder is
sufficient). Boot8300 is optional; omit if not rsync'd to avoid a missing-ckpt error.

---

## RED-TEAM: depth-4 SealBot check

**Purpose:** Confirm the n-scaling curve is not depth-5-bar-specific. If the
plateau holds at depth-4 AND depth-5, the result is bar-independent. If the
curve shape inverts across depths, flag as instrument-sensitive (report, do not
gate on depth-4 alone).

**Run at ONE n-point only** (n=256, the decision-relevant comparison):

```bash
mkdir -p reports/d_localize_p3/redteam_d4_n256
.venv/bin/python scripts/eval/gumbel_ladder.py play \
  --ckpt-dir reports/d_ladder_2026-06-24/ckpts \
  --steps 60000,120000,150000,175000,200000,226500 \
  --sealbot --sealbot-max-depth 4 --sealbot-only \
  --encoding v6_live2_ls \
  --knobs-from reports/d_ladder_2026-06-24/ckpts/checkpoint_00120000.pt \
  --n-games-per-pair 40 --opening-plies 4 --seed-base 20260629 \
  --n-sims-override 256 \
  --out reports/d_localize_p3/redteam_d4_n256/sealbot_games.jsonl

# Merge and aggregate (net-vs-net same banked rows)
python -c "
import json
banked = [json.loads(l) for l in open('${BANKED}') if l.strip()]
net_vs_net = [g for g in banked if 'sealbot' not in (g['p1'], g['p2'])]
seal = [json.loads(l) for l in open('reports/d_localize_p3/redteam_d4_n256/sealbot_games.jsonl') if l.strip()]
with open('reports/d_localize_p3/redteam_d4_n256/full_field.jsonl','w') as f:
    for g in net_vs_net + seal: f.write(json.dumps(g)+'\n')
"

.venv/bin/python scripts/eval/gumbel_ladder.py aggregate \
  --per-game reports/d_localize_p3/redteam_d4_n256/full_field.jsonl \
  --anchor s120k \
  --order "${ORDER}" \
  --n-boot 2000 \
  --out reports/d_localize_p3/redteam_d4_n256/agg
```

**Interpretation:** Compare SealBot Elo at n=150 vs n=256 under depth-4 vs depth-5.
If both show PLATEAU → result bar-independent. If depth-4 shows CLIMB but depth-5
PLATEAU → depth-5 bar saturates the climb (model is close to SealBot@depth-5 limit
at low n; budget buys nothing because the bar is already near-saturated). In that
case PLATEAU verdict still holds for the depth-5 deployment bar.

---

## COST / WALL-TIME ESTIMATE (vast 5080 + Ryzen 9 9900X)

SealBot@depth-5 is CPU-bound (9900X), net inference is GPU-bound (5080). These
run concurrently — one process alternates between GPU inference call and CPU
SealBot call. Bottleneck is the GPU at low n; shifts toward CPU at n=512.

Per-ply timing estimate:
- Net Gumbel@n: ~0.05s (n=128), ~0.08s (n=256), ~0.15s (n=512) at 5080 speeds
- SealBot@depth-5: ~0.5–2s per move (CPU; depth-5 minimax ~468k nodes typical)
- **SealBot dominates** → effective per-game time ≈ 65 plies × ~0.5s ≈ 33s

Total games:
- 4 n-values × 6 ckpts × 40 games = **960 sealbot games**
- Depth-4 red-team at n=256: 6 × 40 = **240 games** (≈ 2h; depth-4 faster than 5)
- n=150 reference: 6 × 40 = 240 games (already counted above)

Wall estimate:
| Task | Games | Est. wall |
|------|-------|-----------|
| n=128 sealbot sweep | 240 | ~2h |
| n=150 reference | 240 | ~2h |
| n=256 sealbot sweep | 240 | ~2.5h |
| n=512 sealbot sweep | 240 | ~3h (SealBot still dominates) |
| depth-4 red-team (n=256) | 240 | ~1.5h |
| **Total** | **1200** | **~11h** |

Run each n sequentially (same GPU; no parallelism benefit within the run).
Can run n=128 + n=150 together if piping to `&` in separate processes, but
SealBot CPU contention makes this inadvisable (9900X at 100% from one process).

**Vast cost at $0.50/hr (RTX 5080 tier):** ~$5.50 for the full sweep.

---

## EXPECTED OUTCOME

D-LADDER Stage 1 verdict: **TRUE-STALL** (self-ladder flat 120k→226k; SealBot
intransitivity real but bar-independent). FINDINGS §"150 sims + Gumbel SH-budget
problem" notes the non-canonical late-heavy SH profile but treats it as benign for
the current measurement.

**Predicted outcome:** PLATEAU-by-150.

Rationale: the model's intransitivity with SealBot is a structural property (8/9
three-cycles routed through SealBot; robust at P=0.004). This is not a search-
budget artifact — doubling or quadrupling the SH budget does not change what the
network KNOWS (policy/value trained at n=150). The SH winner at n=256 will differ
on only a fraction of moves where the n=150 SH winner was wrong due to early-phase
pruning error; the policy/value calibration remains n=150-anchored. The
model⊥minimax gap is a training deficit, not a search deficit.

If the curve CLIMBS (surprise outcome): implies the n=150 search is systematically
mis-selecting moves that the network's policy/value would have gotten right with
more SH phases — i.e., phase-1 pruning at 2 visits/candidate (n=150) prunes the
true best action before phase 2. This is plausible but would require the policy
logits to be noisy enough that the gumbel+logits score at 2 visits is unreliable.
Budget the surprise appropriately.

---

## OUTPUT ARTIFACTS

```
reports/d_localize_p3/
  n128/
    sealbot_games.jsonl
    full_field.jsonl
    agg/ratings.csv
    agg/aggregate.json
  n150/
    sealbot_games.jsonl
    full_field.jsonl
    agg/ratings.csv
    agg/aggregate.json
  n256/
    sealbot_games.jsonl
    full_field.jsonl
    agg/ratings.csv
    agg/aggregate.json
  n512/
    sealbot_games.jsonl
    full_field.jsonl
    agg/ratings.csv
    agg/aggregate.json
  redteam_d4_n256/
    sealbot_games.jsonl
    full_field.jsonl
    agg/ratings.csv
    agg/aggregate.json
```

Rsync back to laptop after completion:
```bash
rsync -avz vast:/workspace/hexo_rl/reports/d_localize_p3/ \
  reports/d_localize_p3/
```

---

## REPORTING CHECKLIST

Per cell (each {n, ckpt} pair), report:
- [ ] `distinct_n` (from `distinct_per_pair` in `aggregate.json`) ≥ 10
- [ ] `copy_multiplier` ≤ 2.0
- [ ] `head_fired = True` on 100% of net moves (check stderr / game logs)
- [ ] `n_sims` confirmed = override value (check `[knobs]` print line)
- [ ] SealBot Elo per checkpoint per n (from `ratings.csv`)
- [ ] Bootstrap 95% CI per checkpoint per n (`ci_lo_boot`, `ci_hi_boot`)
- [ ] 150→256 CI comparison: does CI_low(WR@256) > CI_high(WR@150) for any ckpt?
- [ ] Depth-4 red-team: curve shape consistent with depth-5?
- [ ] Gate verdict: PLATEAU-by-150 or CLIMBS-past-256?

---

## VALIDITY GATES (pre-registered; violation → INSUFFICIENT, re-run)

1. `distinct_n ≥ 10` for every decisive pair (else bootstrap untrusted).
2. `copy_multiplier ≤ 2.0` (else opening diversity failing; investigate seed).
3. `head_fired = True` on 100% of net moves in every jsonl (no fallback-to-legal).
4. `[n_sims_override]` line present in stdout for overridden runs; absent for n=150
   reference (uses embedded 150 directly).
5. SealBot WR for at least one checkpoint in (0.1, 0.9) — if ALL checkpoints are
   near-0 or near-1 vs SealBot@depth-5, the bar is saturated and the n comparison
   is uninformative; switch to depth-3 and flag.
