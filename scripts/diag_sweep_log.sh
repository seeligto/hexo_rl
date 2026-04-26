#!/usr/bin/env bash
# Sweep diagnostic — multi-variant aware, investigation-grade.
#
# Usage:
#   scripts/diag_sweep_log.sh                # auto-discover all variant logs,
#                                            # show overview + deep-dive on most
#                                            # recently active.
#   scripts/diag_sweep_log.sh <variant>      # deep-dive on one variant by name
#                                            # (e.g. sweep_2ch).
#   scripts/diag_sweep_log.sh <logfile>      # deep-dive on a specific jsonl.
#   scripts/diag_sweep_log.sh --all          # deep-dive on every variant.
#   scripts/diag_sweep_log.sh --overview     # overview only, skip deep dive.
#
# Reports gph (games/hr), sph (steps/hr), policy-CE trajectory + slope, GPU
# util, batch fill, eval status, aborts. Numbers are computed both cumulative
# and over a recent window so you can see whether throughput is stable or
# regressing.
set -euo pipefail

SHOW_ALL=0
OVERVIEW_ONLY=0
TARGET=""
case "${1:-}" in
  --all)        SHOW_ALL=1 ;;
  --overview)   OVERVIEW_ONLY=1 ;;
  -h|--help)    sed -n '2,17p' "$0"; exit 0 ;;
  "")           ;;
  *)            TARGET="$1" ;;
esac

# ── Resolve which logs we're looking at ──────────────────────────────────────
declare -a LOGS=()
if [[ -n "$TARGET" ]]; then
  if [[ -f "$TARGET" ]]; then
    LOGS=("$TARGET")
  elif [[ -d "logs/sweep/$TARGET" ]]; then
    while IFS= read -r f; do LOGS+=("$f"); done < <(
      find "logs/sweep/$TARGET" -maxdepth 1 -name "*.jsonl" -type f \
        -printf '%T@ %p\n' 2>/dev/null | sort -rn | awk '{print $2}'
    )
  fi
  if [[ ${#LOGS[@]} -eq 0 ]]; then
    echo "ERROR: '$TARGET' is neither a file nor a logs/sweep/<variant>/ with a jsonl" >&2
    exit 1
  fi
else
  while IFS= read -r f; do LOGS+=("$f"); done < <(
    find logs/sweep -mindepth 2 -maxdepth 2 -name "*.jsonl" -type f \
      -printf '%T@ %p\n' 2>/dev/null \
      | sort -rn \
      | awk '{p=$2; split(p,a,"/"); v=a[3]; if (!seen[v]) {seen[v]=1; print p}}'
  )
  if [[ ${#LOGS[@]} -eq 0 ]]; then
    echo "No sweep logs found under logs/sweep/<variant>/*.jsonl." >&2
    echo "Has the sweep been launched?" >&2
    exit 1
  fi
fi

# ── Multi-variant overview ──────────────────────────────────────────────────
print_overview() {
  python3 - "${LOGS[@]}" <<'PYEOF'
import sys, json, math
from datetime import datetime, timezone

UNIFORM_CE = math.log(362)  # ≈ 5.892
paths = sys.argv[1:]
now = datetime.now(timezone.utc).replace(tzinfo=None)

def parse_ts(s):
    if not s: return None
    try: return datetime.fromisoformat(s.replace("Z",""))
    except Exception: return None

print(f"=== SWEEP STATUS ({len(paths)} variant(s); uniform CE = {UNIFORM_CE:.3f}) ===")
hdr = (f"  {'variant':<12} {'status':<9} {'step':>8} {'games':>5} "
       f"{'gph':>6} {'sph':>6} {'ce_now':>7} {'ce_slope':>10} {'gpu%':>5}")
print(hdr)
print("  " + "-"*(len(hdr)-2))

for p in paths:
    first_ts=last_ts=None
    max_step=0; n_games=0
    abort=None
    ce_pts=[]   # (step, ce)
    gpu_pcts=[]
    try:
        with open(p) as f:
            for line in f:
                try: d = json.loads(line)
                except Exception: continue
                ts = d.get("timestamp")
                if ts:
                    if first_ts is None: first_ts = ts
                    last_ts = ts
                ev = d.get("event")
                if ev in ("train_step","train_step_summary"):
                    s = d.get("step")
                    if isinstance(s,int) and s > max_step: max_step = s
                    ce = d.get("policy_ce", d.get("policy_loss"))
                    if isinstance(s,int) and ce is not None:
                        ce_pts.append((s, float(ce)))
                elif ev == "game_complete":
                    n_games += 1
                elif ev == "gpu_stats":
                    g = d.get("gpu_util_pct")
                    if g is not None: gpu_pcts.append(float(g))
                elif ev in ("hard_abort_grad_norm","soft_abort_ew_flat",
                            "triage_failed","evaluation_error"):
                    abort = ev
    except Exception as e:
        print(f"  read error {p}: {e}")
        continue

    t0, t1 = parse_ts(first_ts), parse_ts(last_ts)
    wall_hr = (t1 - t0).total_seconds()/3600 if (t0 and t1) else 0.0
    gph = n_games / wall_hr if wall_hr > 0 else 0.0
    sph = max_step / wall_hr if wall_hr > 0 else 0.0

    ce_pts.sort()
    ce_now = ce_pts[-1][1] if ce_pts else None
    ce_slope = None
    if len(ce_pts) >= 10:
        head = ce_pts[:max(5, len(ce_pts)//10)]
        tail = ce_pts[-max(5, len(ce_pts)//10):]
        ds = ce_pts[-1][0] - ce_pts[0][0]
        if ds > 0:
            ce_slope = (sum(c for _,c in tail)/len(tail) -
                        sum(c for _,c in head)/len(head)) / ds

    gpu_recent = sum(gpu_pcts[-12:])/min(12,len(gpu_pcts)) if gpu_pcts else None

    if abort: status = abort[:9]
    elif t1 and (now - t1).total_seconds() < 60: status = "ALIVE"
    else: status = "stopped"

    parts = p.split("/")
    variant = parts[-2] if len(parts)>=2 else "?"
    ce_now_s = f"{ce_now:.3f}" if ce_now is not None else "—"
    ce_slope_s = f"{ce_slope:+.5f}" if ce_slope is not None else "—"
    gpu_s = f"{gpu_recent:.0f}" if gpu_recent is not None else "—"
    print(f"  {variant:<12} {status:<9} {max_step:>8} {n_games:>5} "
          f"{gph:>6.0f} {sph:>6.0f} {ce_now_s:>7} {ce_slope_s:>10} {gpu_s:>5}")
PYEOF
}

if [[ ${#LOGS[@]} -gt 1 ]] || [[ "$SHOW_ALL" -eq 1 ]] || [[ "$OVERVIEW_ONLY" -eq 1 ]]; then
  print_overview
  echo
fi

[[ "$OVERVIEW_ONLY" -eq 1 ]] && exit 0

# ── Per-variant deep dive ───────────────────────────────────────────────────
deep_dive() {
  local LOG="$1"
  python3 - "$LOG" <<'PYEOF'
import sys, json, math, collections
from datetime import datetime

UNIFORM_CE = math.log(362)
path = sys.argv[1]
events = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: events.append(json.loads(line))
        except Exception: pass

variant = path.split("/")[-2] if "/" in path else path
print(f"=== {variant.upper()} DEEP DIVE  ({path}) ===")
if not events:
    print("  (no parseable events)"); sys.exit(0)

ts_first = events[0].get("timestamp")
ts_last  = events[-1].get("timestamp")

def parse_ts(s):
    try: return datetime.fromisoformat(s.replace("Z",""))
    except Exception: return None

t0 = parse_ts(ts_first) if ts_first else None
t1 = parse_ts(ts_last)  if ts_last  else None
wall_hr = (t1 - t0).total_seconds()/3600 if (t0 and t1) else 0.0

# Pull useful series
train_pts = []   # (ts, step, ce)
games_ts = []    # ts of game_complete
gpu_pts = []     # (ts, util)
batch_fills = [] # batch_fill_pct values
inf_counts = []  # inf_forward_count values

for e in events:
    ts = parse_ts(e.get("timestamp",""))
    ev = e.get("event")
    if ev in ("train_step","train_step_summary"):
        s = e.get("step")
        ce = e.get("policy_ce", e.get("policy_loss"))
        if isinstance(s,int) and ce is not None:
            train_pts.append((ts, s, float(ce)))
        bf = e.get("batch_fill_pct")
        if bf is not None: batch_fills.append(float(bf))
        ic = e.get("inf_forward_count")
        if ic is not None: inf_counts.append(int(ic))
    elif ev == "game_complete":
        games_ts.append(ts)
    elif ev == "gpu_stats":
        g = e.get("gpu_util_pct")
        if g is not None and ts is not None: gpu_pts.append((ts, float(g)))

# === Throughput ===
print(f"\nTHROUGHPUT")
print(f"  wall: {wall_hr:.2f}hr  ({ts_first[:19]} → {ts_last[:19]})")
if wall_hr > 0:
    last_step = train_pts[-1][1] if train_pts else 0
    cum_gph = len(games_ts) / wall_hr
    cum_sph = last_step / wall_hr
    print(f"  cumulative: gph={cum_gph:>6.0f}  sph={cum_sph:>6.0f}")
    # Recent-window rate (last 20% of wall time, min 5 min)
    if t0 and t1 and wall_hr > 0.1:
        window_sec = max(300, (t1 - t0).total_seconds() * 0.2)
        cutoff = t1.timestamp() - window_sec
        recent_games = sum(1 for g in games_ts if g and g.timestamp() >= cutoff)
        recent_steps = 0
        for ts, s, _ in reversed(train_pts):
            if ts and ts.timestamp() < cutoff: break
            recent_steps = max(recent_steps, s)
        # find step at cutoff (linear search backwards)
        cutoff_step = 0
        for ts, s, _ in train_pts:
            if ts and ts.timestamp() <= cutoff: cutoff_step = s
        recent_step_count = recent_steps - cutoff_step
        recent_hr = window_sec / 3600
        if recent_hr > 0:
            print(f"  recent ({recent_hr*60:.0f}min): "
                  f"gph={recent_games/recent_hr:>6.0f}  sph={recent_step_count/recent_hr:>6.0f}")

# === Policy CE trajectory ===
if train_pts:
    print(f"\nPOLICY CE  (uniform = {UNIFORM_CE:.3f})")
    print(f"  first / last:  step {train_pts[0][1]:>5d} ce={train_pts[0][2]:.4f}  →  "
          f"step {train_pts[-1][1]:>5d} ce={train_pts[-1][2]:.4f}")
    # CE checkpoints
    for tgt in (50, 100, 200, 500, 1000, 1500, 2500, 5000, 10000):
        if tgt > train_pts[-1][1]: break
        cands = [(s, c) for _, s, c in train_pts if abs(s - tgt) <= 10]
        if cands:
            s, c = min(cands, key=lambda x: abs(x[0]-tgt))
            margin = UNIFORM_CE - c
            print(f"  step ~{tgt:>5d}:  ce={c:.4f}  margin_below_uniform={margin:+.4f}")
    # Slope + projection
    if len(train_pts) >= 10:
        head = train_pts[:max(5, len(train_pts)//10)]
        tail = train_pts[-max(5, len(train_pts)//10):]
        ds = train_pts[-1][1] - train_pts[0][1]
        if ds > 0:
            slope = (sum(c for _,_,c in tail)/len(tail) -
                     sum(c for _,_,c in head)/len(head)) / ds
            print(f"  slope: {slope:+.5f} per step")
            # Project to phase 1 / 2 horizons
            for tgt in (1500, 2500, 10000):
                if tgt > train_pts[-1][1]:
                    proj = train_pts[-1][2] + slope * (tgt - train_pts[-1][1])
                    print(f"    → projected step {tgt:>5d}: ce≈{proj:.4f}")

# === GPU ===
if gpu_pts:
    utils = [u for _, u in gpu_pts]
    avg_u = sum(utils)/len(utils); max_u = max(utils)
    recent_utils = utils[-12:]
    avg_recent = sum(recent_utils)/len(recent_utils)
    print(f"\nGPU  ({len(gpu_pts)} samples)")
    print(f"  cumulative: avg={avg_u:.1f}%  max={max_u:.1f}%")
    print(f"  recent (last 12 samples): avg={avg_recent:.1f}%")

# === Inference batch saturation ===
if batch_fills:
    avg_bf = sum(batch_fills[-20:]) / min(20, len(batch_fills))
    cum_bf = sum(batch_fills) / len(batch_fills)
    print(f"\nINFERENCE BATCH FILL")
    print(f"  cumulative: {cum_bf:.1f}%   recent: {avg_bf:.1f}%   "
          f"({'saturated' if avg_bf >= 95 else 'unsaturated — workers not feeding fast enough'})")
    if inf_counts and len(inf_counts) >= 2:
        delta = inf_counts[-1] - inf_counts[max(0, len(inf_counts)-10)]
        print(f"  recent inf forwards (last 10 reports): +{delta}")

# === Eval ===
eval_prog = [e for e in events if e.get("event") == "evaluation_game_progress"]
eval_done = [e for e in events if e.get("event") in (
    "evaluation_complete","eval_complete","arena_complete")]
if eval_prog or eval_done:
    print(f"\nEVAL")
    if eval_prog:
        by_phase = collections.defaultdict(list)
        for e in eval_prog:
            by_phase[e.get("phase","?")].append(e)
        for ph, evs in by_phase.items():
            secs = [e.get("sec_per_game",0) for e in evs]
            avg = sum(secs)/max(1,len(secs))
            last = evs[-1]
            print(f"  {ph}: {last.get('game',0)}/{last.get('total_games',0)}  "
                  f"avg {avg:.1f}s/game  partial_wr={last.get('partial_winrate','?')}  "
                  f"ETA {last.get('eta_sec',0)/60:.1f}min")
    if eval_done:
        for e in eval_done[-3:]:
            slim = {k: e[k] for k in ("step","wr_best","wr_random","wr_sealbot","promoted")
                    if e.get(k) is not None}
            print(f"  done @ {e.get('timestamp','?')[:19]}: {slim}")

# === Aborts / warnings ===
abort_evs = ("hard_abort_grad_norm","soft_abort_ew_flat","triage_failed","evaluation_error")
aborts = [e for e in events if e.get("event") in abort_evs]
warnings = [e for e in events
            if e.get("level") in ("warning","warn","critical")
            and e.get("event") not in abort_evs]
if aborts:
    print(f"\nABORTS ({len(aborts)})")
    for e in aborts[:5]:
        print(f"  {e.get('timestamp','?')[:19]}  {e.get('event')}  "
              f"{e.get('msg', e.get('message',''))[:100]}")
if warnings:
    by_kind = collections.Counter(e.get("event","?") for e in warnings)
    print(f"\nWARNINGS ({len(warnings)})")
    for kind, n in by_kind.most_common(5):
        print(f"  {n:>4d}  {kind}")
PYEOF
}

if [[ "$SHOW_ALL" -eq 1 ]]; then
  for L in "${LOGS[@]}"; do
    deep_dive "$L"
    echo
  done
else
  deep_dive "${LOGS[0]}"
fi
