#!/usr/bin/env bash
# Vast-side §177 v6 sustained training checkup (cloned from §175 variant).
# One-shot: SSH in, dump live rate, training + eval DB + G4 trajectory + mix info.
#
# Env overrides:
#   VAST_HOST   (default ssh6.vast.ai)
#   VAST_PORT   (default 13053)
#   VAST_KEY    (default ~/.ssh/vast_hexo)
#   TMUX_NAME   (default s177)
#   RUN_ID      (default 17f31b4855304ddfab2508bfd4799be1; auto-fallback to latest)
#   RUN_NAME    (default s177_v6_sustained_step20k_anchor — controls jsonl path)

set -euo pipefail

VAST_HOST="${VAST_HOST:-ssh6.vast.ai}"
VAST_PORT="${VAST_PORT:-13053}"
VAST_KEY="${VAST_KEY:-$HOME/.ssh/vast_hexo}"
TMUX_NAME="${TMUX_NAME:-s177}"
RUN_ID="${RUN_ID:-17f31b4855304ddfab2508bfd4799be1}"
RUN_NAME="${RUN_NAME:-s177_v6_sustained_step20k_anchor}"

SSH="ssh -q -i ${VAST_KEY} -p ${VAST_PORT} -o StrictHostKeyChecking=no root@${VAST_HOST}"

# Build the remote Python payload (eval DB + G4 + checkpoint_log + rate + mix).
PY_PAYLOAD=$(cat <<'PYEOF'
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/workspace/hexo_rl")
DB   = REPO / "reports/eval/results.db"
LOG  = REPO / "checkpoints/checkpoint_log.json"
CKPT_DIR = REPO / "checkpoints"

# §177 anchor — step-20K SealBot sampled-mode reading (§175 step-20K eval +
# Phase B Wave A1 re-baseline, V-PhaseB-5 PASS):
ANCHOR_LABEL     = "v6_step20k Phase B anchor"
ANCHOR_WR        = 0.180             # §175 step-20K sampled SealBot WR
ANCHOR_WR_CI     = (0.117, 0.267)    # Wilson95 n=100
G4_FLOOR, G4_CEIL = 0.154, 0.462     # value_fc2.weight |max| gate (§174 band)

# Corpus-mixing schedule from configs/variants/v6_sustained_s177.yaml +
# configs/training.yaml mixing block (initial 0.8 → min 0.1, linear decay over 200K).
MIX_INITIAL = 0.8
MIX_MIN     = 0.1
MIX_DECAY_STEPS = 200_000

def mix_weight(step: int) -> float:
    if step <= 0: return MIX_INITIAL
    if step >= MIX_DECAY_STEPS: return MIX_MIN
    return MIX_INITIAL - (MIX_INITIAL - MIX_MIN) * (step / MIX_DECAY_STEPS)

run_id   = "{RUN_ID_PLACEHOLDER}"
run_name = "{RUN_NAME_PLACEHOLDER}"
jsonl_path = REPO / "logs" / f"{run_name}.jsonl"

# ───────────────────────────────────────────────────────────────────────
# TOP-LINE — current step, rate, eta, last loss, last grad_norm
# ───────────────────────────────────────────────────────────────────────
print("=" * 90)
print(f"  TOP-LINE       run_name={run_name}")
print(f"                 run_id  ={run_id}")
print("=" * 90)

cur_step = None
last_loss = None
last_grad = None
last_lr = None
last_ts = None
first_step_ts = None
first_step = None
train_steps_in_log = 0
game_completes_in_log = 0
buffer_size = None
gpu_pct = None
gpu_temp = None
session_ended = False

if not jsonl_path.exists():
    print(f"  [WARN] {jsonl_path} not found — training may not have started")
else:
    # One pass through jsonl: collect counters, last train_step + last game_complete
    first_train_step_event = None
    last_train_step_event = None
    last_game_event = None
    last_gpu_event = None
    last_buffer_event = None
    with jsonl_path.open() as f:
        for raw in f:
            try:
                e = json.loads(raw)
            except Exception:
                continue
            ev = e.get("event", "")
            if ev == "train_step":
                train_steps_in_log += 1
                if first_train_step_event is None:
                    first_train_step_event = e
                last_train_step_event = e
            elif ev == "game_complete":
                game_completes_in_log += 1
                last_game_event = e
            elif ev == "gpu_stats":
                last_gpu_event = e
            elif ev in ("warmup", "waiting_for_games"):
                last_buffer_event = e
            elif ev == "session_end":
                session_ended = True

    if last_train_step_event is not None:
        cur_step = last_train_step_event.get("step")
        last_loss = last_train_step_event.get("total_loss")
        last_grad = last_train_step_event.get("grad_norm")
        last_lr   = last_train_step_event.get("lr")
        last_ts   = last_train_step_event.get("timestamp")
    if first_train_step_event is not None:
        first_step = first_train_step_event.get("step")
        first_step_ts = first_train_step_event.get("timestamp")
    if last_gpu_event:
        gpu_pct  = last_gpu_event.get("gpu_util_pct")
        gpu_temp = last_gpu_event.get("temp_c")
    if last_buffer_event:
        buffer_size = last_buffer_event.get("buffer")

print(f"  current step      : {cur_step}")
print(f"  total games run   : {game_completes_in_log}")
print(f"  buffer size       : {buffer_size}")
print(f"  last total_loss   : {last_loss:.4f}" if isinstance(last_loss, (int, float)) else f"  last total_loss   : {last_loss}")
print(f"  last grad_norm    : {last_grad:.4f}" if isinstance(last_grad, (int, float)) else f"  last grad_norm    : {last_grad}")
print(f"  last lr           : {last_lr:.2e}" if isinstance(last_lr, (int, float)) else f"  last lr           : {last_lr}")
print(f"  last log ts       : {last_ts}")
if gpu_pct is not None: print(f"  gpu util / temp   : {gpu_pct:.0f}% / {gpu_temp:.0f}°C")
if session_ended:       print(f"  [!] session_end event present — run has stopped")

# Throughput — overall (first→last) and recent (last 5 min)
if first_step_ts and last_ts and cur_step is not None and first_step is not None:
    ta = datetime.fromisoformat(first_step_ts.replace("Z", "+00:00"))
    tb = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
    dt_min = max(0.001, (tb - ta).total_seconds() / 60)
    ds = cur_step - first_step
    overall_steps_per_hr = (ds / dt_min) * 60
    overall_games_per_hr = (game_completes_in_log / dt_min) * 60
    print()
    print(f"  steps/hr (overall): {overall_steps_per_hr:>7.0f}   ({ds} steps in {dt_min:.1f} min)")
    print(f"  games/hr (overall): {overall_games_per_hr:>7.0f}   ({game_completes_in_log} games in {dt_min:.1f} min)")

    # Recent window — last 5 min of train_step events
    cutoff = tb.timestamp() - 5 * 60
    recent_steps = 0
    recent_games = 0
    recent_first_step = None
    recent_first_ts = None
    with jsonl_path.open() as f:
        for raw in f:
            try: e = json.loads(raw)
            except Exception: continue
            ev = e.get("event", "")
            ts = e.get("timestamp")
            if not ts: continue
            try:
                t = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            except Exception:
                continue
            if t < cutoff: continue
            if ev == "train_step":
                recent_steps += 1
                if recent_first_step is None:
                    recent_first_step = e.get("step")
                    recent_first_ts   = t
            elif ev == "game_complete":
                recent_games += 1
    if recent_first_step is not None and cur_step is not None:
        rdt_min = max(0.001, (tb.timestamp() - recent_first_ts) / 60)
        rds = cur_step - recent_first_step
        rsteps_hr = (rds / rdt_min) * 60
        rgames_hr = (recent_games / rdt_min) * 60
        print(f"  steps/hr (5-min) : {rsteps_hr:>7.0f}   ({rds} steps in {rdt_min:.1f} min)")
        print(f"  games/hr (5-min) : {rgames_hr:>7.0f}   ({recent_games} games in {rdt_min:.1f} min)")

# ───────────────────────────────────────────────────────────────────────
# TRAINING MIX — what's in the gradient at this step
# ───────────────────────────────────────────────────────────────────────
print()
print("=" * 90)
print("  TRAINING MIX  (what feeds each gradient batch)")
print("=" * 90)
mix_w = mix_weight(cur_step if cur_step is not None else 0)
print(f"  bootstrap corpus  : data/bootstrap_corpus_v6.npz  (353091 positions, v6 8-plane 19×19)")
print(f"    corpus origin   : 100% human games (no bot games, no self-play). Static; not regenerated during §177.")
print(f"  pretrained weight : {mix_w:.3f}   (schedule: init {MIX_INITIAL} → min {MIX_MIN} over {MIX_DECAY_STEPS:,} steps)")
print(f"  selfplay weight   : {1 - mix_w:.3f}   (self-play games generated by THIS run, vs itself)")
print()
print(f"  LIVE bot mixing in selfplay   : NO  (Source A/B = Phase B S4/S5 design docs, deferred to next run)")
print(f"  Sealbot games in training loop: 0   (sealbot enters only at eval; never trains the net live)")
print(f"  Sealbot at eval (every 10K)   : 100 games  (model T=0.5 sampled, MCTS-128 vs sealbot strong 0.5s)")
print(f"  Bootstrap-anchor opponent     : checkpoints/bootstrap_model_v7full.pt at eval (frozen Gate-6 floor)")
print(f"  best_checkpoint opponent      : checkpoints/best_model.pt (rotating; reset to step-20K anchor at launch)")

# ───────────────────────────────────────────────────────────────────────
# EVAL DB — SealBot trajectory
# ───────────────────────────────────────────────────────────────────────
if DB.exists():
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    cdb = con.cursor()

    if not run_id:
        row = cdb.execute(
            "SELECT run_id FROM matches "
            "WHERE run_id LIKE '________________________________' "
            "ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if row: run_id = row["run_id"]

    rows = list(cdb.execute("""
      SELECT m.eval_step, m.wins_a, m.wins_b, m.draws, m.n_games,
             m.win_rate_a, m.ci_lower, m.ci_upper, m.colony_win,
             pb.name AS opponent
      FROM matches m
      JOIN players pb ON pb.id = m.player_b_id
      WHERE m.run_id = ?
      ORDER BY m.eval_step, pb.name
    """, (run_id,)))

    if rows:
        steps = sorted({r["eval_step"] for r in rows})
        print()
        print("=" * 90)
        print(f"  SEALBOT EVAL  ({ANCHOR_LABEL}: WR {ANCHOR_WR*100:.1f}% Wilson [{ANCHOR_WR_CI[0]*100:.1f}, {ANCHOR_WR_CI[1]*100:.1f}])")
        print("=" * 90)
        print(f"  eval steps        : {steps}")
        print()
        print(f"  {'step':>7}  {'wins':>4}/{'n':<3}  {'WR':>6}  {'Wilson 95% CI':<18}  {'colony':>6}  {'col-frac':>8}  vs anchor")
        for s in steps:
            sb = [r for r in rows if r["eval_step"]==s and "SealBot" in r["opponent"]]
            if not sb: continue
            r = sb[0]
            wr = r["win_rate_a"]
            cf = (r["colony_win"] / r["wins_a"]) if r["wins_a"] else 0.0
            inside = "in CI" if ANCHOR_WR_CI[0] <= wr <= ANCHOR_WR_CI[1] else ("ABOVE" if wr > ANCHOR_WR_CI[1] else "BELOW")
            print(f"  {s:>7}  {r['wins_a']:>4}/{r['n_games']:<3}  {wr*100:>5.1f}%  [{r['ci_lower']*100:>4.1f}, {r['ci_upper']*100:>5.1f}]    {r['colony_win']:>6}    {cf*100:>5.1f}%  {inside}")

        print()
        print("=" * 90)
        print("  OTHER EVAL OPPONENTS  (best_checkpoint, bootstrap_anchor, etc.)")
        print("=" * 90)
        print(f"  {'step':>7}  opponent                                    {'WR':>6}  {'Wilson 95% CI':<18}  {'col-frac':>8}")
        for r in rows:
            if "SealBot" in r["opponent"] or "random" in r["opponent"]: continue
            wr = r["win_rate_a"]
            cf = (r["colony_win"] / r["wins_a"]) if r["wins_a"] else 0.0
            op = r["opponent"][:42]
            print(f"  {r['eval_step']:>7}  {op:<42}  {wr*100:>5.1f}%  [{r['ci_lower']*100:>4.1f}, {r['ci_upper']*100:>5.1f}]   {cf*100:>5.1f}%")
    else:
        print()
        print("=" * 90)
        print("  SEALBOT EVAL")
        print("=" * 90)
        print(f"  [no eval-pipeline rows for run_id={run_id} yet — first eval lands at step 10000]")
else:
    print(f"  [results.db not found at {DB}]")

# ───────────────────────────────────────────────────────────────────────
# G4 — value_fc2.weight |max| over checkpoints
# ───────────────────────────────────────────────────────────────────────
print()
print("=" * 90)
print(f"  G4 value-head  |value_fc2.weight|.max()   (band [{G4_FLOOR}, {G4_CEIL}], v6 step-20K anchor ≈ 0.308 ref)")
print("=" * 90)
try:
    import torch
    g4_rows = []
    for p in sorted(CKPT_DIR.glob("checkpoint_*.pt")):
        try:
            step = int(p.stem.split("_")[1])
        except ValueError:
            continue
        if step % 1000 != 0: continue
        try:
            sd = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            continue
        ms = sd.get("model_state") or sd.get("state_dict") or sd
        w = ms.get("value_fc2.weight")
        if w is None: continue
        mx = float(w.abs().max())
        verdict = "PASS" if G4_FLOOR <= mx <= G4_CEIL else "FAIL"
        g4_rows.append((step, mx, verdict))
    g4_rows = g4_rows[-12:]
    if g4_rows:
        print(f"  {'step':>7}  {'|max|':>7}  verdict")
        for step, mx, verdict in g4_rows:
            bar = "*" * int(round(mx * 100))
            print(f"  {step:>7}  {mx:>7.4f}  {verdict}  {bar}")
    else:
        print(f"  [no checkpoint_*.pt with value_fc2.weight yet — first lands at step 500 (checkpoint_interval)]")
except Exception as e:
    print(f"  (G4 skipped: {e})")

# ───────────────────────────────────────────────────────────────────────
# Recent train-step trends
# ───────────────────────────────────────────────────────────────────────
print()
print("=" * 90)
print("  RECENT TRAIN-STEP TRENDS  (last 10K steps, 1K granularity)")
print("=" * 90)
try:
    data = json.loads(LOG.read_text())
    last_step = data[-1].get("step", 0)
    print(f"  latest checkpoint_log step: {last_step}")
    print(f"  {'step':>7}  {'total':>6}  {'pol':>6}  {'val':>6}  {'pe_self':>8}")
    for s in range(max(0, last_step - 10000), last_step + 1, 1000):
        matches = [e for e in data if e.get("step") == s]
        if not matches: continue
        e = matches[-1]
        print(f"  {s:>7}  {e['loss']:>6.3f}  {e['policy_loss']:>6.3f}  {e['value_loss']:>6.4f}  {e.get('policy_entropy_selfplay', 0):>8.3f}")
except Exception as e:
    print(f"  (checkpoint_log skipped: {e})")

# ───────────────────────────────────────────────────────────────────────
# ETA to next eval
# ───────────────────────────────────────────────────────────────────────
print()
print("=" * 90)
print("  NEXT-EVAL ETA  (eval_interval=10000)")
print("=" * 90)
try:
    if 'overall_steps_per_hr' in dir():
        rate_per_min = overall_steps_per_hr / 60
        next_eval = ((cur_step // 10000) + 1) * 10000
        rem = next_eval - cur_step
        eta_min = rem / rate_per_min if rate_per_min else 0
        print(f"  rate (overall)   : {overall_steps_per_hr:.0f} steps/hr")
        print(f"  next eval step   : {next_eval}  (in {rem} steps)")
        print(f"  ETA              : {eta_min:.0f} min  (~{eta_min/60:.1f} h)")
    else:
        print(f"  [insufficient data — need ≥2 train_step events in jsonl]")
except Exception as e:
    print(f"  (ETA skipped: {e})")
PYEOF
)

# Substitute placeholders
PY_PAYLOAD="${PY_PAYLOAD//\{RUN_ID_PLACEHOLDER\}/$RUN_ID}"
PY_PAYLOAD="${PY_PAYLOAD//\{RUN_NAME_PLACEHOLDER\}/$RUN_NAME}"

# Header
echo "================================================================================"
echo "§177 v6 sustained — vast checkup    $(date -Iseconds)"
echo "  tmux=${TMUX_NAME}  run_id=${RUN_ID}  run_name=${RUN_NAME}"
echo "================================================================================"
echo

# Live status from remote in one SSH call.
$SSH "
  set -e
  echo '=== Live training (tmux ${TMUX_NAME}) ==='
  if ! tmux has-session -t ${TMUX_NAME} 2>/dev/null; then
    echo '  [ERROR] tmux session ${TMUX_NAME} not found'
  else
    last=\$(tmux capture-pane -t ${TMUX_NAME} -p -S -200 | grep -E '\"step\":' | tail -1)
    step=\$(echo \"\$last\" | grep -oE '\"step\": [0-9]+' | head -1 | grep -oE '[0-9]+')
    ts=\$(echo \"\$last\"   | grep -oE '\"timestamp\": \"[^\"]+\"' | head -1 | cut -d'\"' -f4)
    echo \"  tmux latest step : \$step\"
    echo \"  tmux log ts      : \$ts\"
  fi
  echo
  echo '=== GPU ==='
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
  echo
  /workspace/hexo_rl/.venv/bin/python <<'PYINNER'
${PY_PAYLOAD}
PYINNER
"
