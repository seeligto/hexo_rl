#!/usr/bin/env bash
# d1m_status.sh — status for the D-1M Gumbel 1M run on vast.
#   default: snapshot (step, rate, ETA, health, canaries, eval table)
#   --trend: longitudinal view — coherence (forced-win conversion + colony),
#            value-divergence, and strength binned/aligned per eval window.
#
# Usage:  scripts/d1m_status.sh [--trend]
#
# Env overrides:
#   D1M_HOST  (default: vast)   D1M_REPO (default: /workspace/hexo_rl)
#   D1M_LOG   (default: logs/d1m/d1m_gumbel_m16_n150.jsonl)  path relative to repo
set -euo pipefail

HOST="${D1M_HOST:-vast}"
REPO="${D1M_REPO:-/workspace/hexo_rl}"
LOG="${D1M_LOG:-logs/d1m/d1m_gumbel_m16_n150.jsonl}"
TREND=0
for a in "$@"; do
  case "$a" in
    --trend|-T) TREND=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown arg: $a (try -h)" >&2; exit 2 ;;
  esac
done

ssh "$HOST" "cd '$REPO' && .venv/bin/python - '$LOG' '$TREND'" <<'PYEOF'
import json, sys, bisect
from datetime import datetime

log   = sys.argv[1] if len(sys.argv) > 1 else "logs/d1m/d1m_gumbel_m16_n150.jsonl"
trend = len(sys.argv) > 2 and sys.argv[2] == "1"
MILES = [(30000, "first eval"), (200000, "200k gate"), (1000000, "1M target")]

def ts(d):
    return datetime.fromisoformat((d["timestamp"] if isinstance(d, dict) else d).replace("Z", "+00:00"))

def g(d, k, nd=3):
    v = d.get(k)
    return round(v, nd) if isinstance(v, float) else ("-" if v is None else v)

def wrci(wr, ci):
    s = "-" if wr is None else ("%.3f" % wr if isinstance(wr, float) else str(wr))
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        return "%s ±%.02f" % (s, (ci[1] - ci[0]) / 2)
    if isinstance(ci, (int, float)):
        return "%s ±%.02f" % (s, ci)
    return s

try:
    raw = open(log).read().splitlines()
except FileNotFoundError:
    print("ERROR: log not found:", log); sys.exit(1)

summ, rounds, fwt = [], [], []
total_steps = None
# trend-only accumulators: value_bce + colony binned by 30k step-bucket
BIN = 30000
bce = {}     # bucket -> [sum_sp, sum_co, n]
col = {}     # bucket -> [sum_colony, n]
tsi = 0      # train_step sample counter
gci = 0      # game_complete sample counter

for ln in raw:
    s = ln.strip()
    if not s.startswith("{"):
        continue
    try:
        if '"train_step_summary"' in s:
            summ.append(json.loads(s))
        elif '"evaluation_round_complete"' in s:
            rounds.append(json.loads(s))
        elif '"forced_win_trend"' in s:
            fwt.append(json.loads(s))
        elif '"startup"' in s:
            total_steps = (json.loads(s).get("config") or {}).get("total_steps", total_steps)
        elif trend and '"event": "train_step"' in s:
            tsi += 1
            if tsi % 25 == 0:
                d = json.loads(s)
                b = d.get("step", 0) // BIN
                sp, co = d.get("value_bce_selfplay"), d.get("value_bce_corpus")
                if sp is not None and co is not None:
                    e = bce.setdefault(b, [0.0, 0.0, 0]); e[0] += sp; e[1] += co; e[2] += 1
    except Exception:
        pass

# ts -> step map (from summaries) for events lacking a step field
stimes = [ts(s) for s in summ]
ssteps = [s["step"] for s in summ]

def step_at(t):
    i = bisect.bisect_left(stimes, t)
    return ssteps[min(max(i, 0), len(ssteps) - 1)] if ssteps else 0

if trend:                                   # colony by bucket (game_complete has ts, no step)
    for ln in raw:
        s = ln.strip()
        if '"event": "game_complete"' not in s:
            continue
        gci += 1
        if gci % 10:
            continue
        try:
            d = json.loads(s)
            c = d.get("colony_extension_fraction")
            if c is None:
                continue
            b = step_at(ts(d)) // BIN
            e = col.setdefault(b, [0.0, 0]); e[0] += c; e[1] += 1
        except Exception:
            pass

# high-frequency latest values for snapshot (reverse scan)
last_ts, colony_recent = None, []
for ln in reversed(raw):
    s = ln.strip()
    if not s.startswith("{"):
        continue
    if last_ts is None and '"event": "train_step"' in s:
        try: last_ts = json.loads(s)
        except Exception: pass
    elif '"event": "game_complete"' in s and len(colony_recent) < 500:
        try:
            c = json.loads(s).get("colony_extension_fraction")
            if c is not None:
                colony_recent.append(c)
        except Exception:
            pass
    if last_ts is not None and len(colony_recent) >= 500:
        break

if total_steps:
    MILES = [m for m in MILES if m[0] < total_steps] + [(total_steps, "target")]
if not summ:
    print("no train_step_summary events yet in", log); sys.exit(0)

first, last, lt = summ[0], summ[-1], (last_ts or {})

def rate(a, b):
    dt = (ts(b) - ts(a)).total_seconds()
    return (b["step"] - a["step"]) / dt * 3600 if dt > 0 else 0.0

overall = rate(first, last)
cut = ts(last).timestamp() - 3600
recent_pts = [s for s in summ if ts(s).timestamp() >= cut]
recent = rate(recent_pts[0], last) if len(recent_pts) >= 2 else overall
eta_rate = recent if recent > 0 else overall

print("=== D-1M run ===")
print("step         : {:,}".format(last["step"]))
print("rate         : {:,.0f}/hr overall | {:,.0f}/hr recent-1h".format(overall, recent))
print("throughput   : {:,.0f} games/hr | {:,.0f} sims/s | gpu {}% | vram {} GB".format(
    last.get("games_per_hour", 0), last.get("sims_per_sec", 0), g(last, "gpu_util", 0), g(last, "vram_gb", 1)))

print("\n=== health ===")
print("draw_rate {} | x/o {}/{} | grad_norm {} | lr {} | mcts_depth {}".format(
    g(last, "draw_rate"), g(last, "x_winrate"), g(last, "o_winrate"),
    g(lt, "grad_norm"), g(lt, "lr", 5), g(last, "mcts_mean_depth", 2)))
print("policy_entropy_sp {} | avg_sigma {} | corpus_w {} | buffer {:,}".format(
    g(last, "policy_entropy_selfplay"), g(last, "avg_sigma"), g(last, "pretrained_weight"), last.get("buffer_size", 0)))

print("\n=== value-divergence canary (selfplay should track corpus) ===")
sp, co = lt.get("value_bce_selfplay"), lt.get("value_bce_corpus")
gap = "" if not (isinstance(sp, float) and isinstance(co, float)) else "  gap %+.3f" % (sp - co)
print("value_bce  selfplay {} | corpus {}{}".format(g(lt, "value_bce_selfplay"), g(lt, "value_bce_corpus"), gap))
print("value_acc  selfplay {} | masked {}    (selfplay BCE flooring while corpus drops = overfit-corpus)".format(
    g(lt, "value_accuracy_selfplay"), g(lt, "value_accuracy_masked")))

print("\n=== coherence (golong kill-gate) ===")
fw = fwt[-1] if fwt else {}
cm = (sum(colony_recent) / len(colony_recent)) if colony_recent else None
print("forced_win_conversion {} | off_window {} | colony_extension (last {} games) {}".format(
    g(fw, "forced_win_conversion"), g(fw, "off_window_forced_win_rate"),
    len(colony_recent), ("%.3f" % cm if cm is not None else "-")))
print("  read SLOPE + co-occurrence, not the absolute number: a decline that coincides with rising")
print("  colony / falling longest-line = fragmenting (golong was conv 0.89->0.66 + components 26->42).")

print("\n=== ETA (recent pace {:,.0f}/hr) ===".format(eta_rate))
for tgt, label in MILES:
    rem = tgt - last["step"]
    if rem <= 0:
        print("  {:<10s} step {:>9,}: reached".format(label, tgt)); continue
    h = rem / eta_rate if eta_rate > 0 else 0
    print("  {:<10s} step {:>9,}: {:6.1f} h ({:4.1f} d)".format(label, tgt, h, h / 24))

print("\n=== eval rounds ({}) ===".format(len(rounds)))
if rounds:
    print("  {:>7s}  {:>4s}  {:>13s}  {:>13s}  {:>6s}  {:>6s}".format("step", "prom", "wr_best", "sealbot", "random", "elo"))
    for d in rounds:
        print("  {:>7}  {:>4}  {:>13}  {:>13}  {:>6}  {:>6}".format(
            d.get("step", "?"), "yes" if d.get("promoted") else "no",
            wrci(d.get("wr_best"), d.get("ci_best")), wrci(d.get("wr_sealbot"), d.get("ci_sealbot")),
            g(d, "wr_random", 2), g(d, "elo_estimate", 0)))
else:
    print("  (none scored yet — first round at step 30,000)")

if trend:
    print("\n=== TREND: coherence trajectory (forced-win) ===")
    print("  {:>7s}  {:>11s} {:>3s}  {:>11s} {:>3s}  {:>6s}".format("~step", "conversion", "dir", "off_window", "dir", "n"))
    pc = po = None
    for d in fwt:
        st = step_at(ts(d))
        c, o = d.get("forced_win_conversion"), d.get("off_window_forced_win_rate")
        dc = "" if pc is None else ("UP" if c > pc + .005 else ("DN" if c < pc - .005 else "=="))
        do = "" if po is None else ("UP" if o > po + .005 else ("DN" if o < po - .005 else "=="))
        pc, po = c, o
        print("  {:>7}  {:>11.4f} {:>3}  {:>11.4f} {:>3}  {:>6}".format(st, c, dc, o, do, d.get("n", "-")))

    print("\n=== TREND: per-30k window (colony co-signal + value-divergence) ===")
    print("  {:>9s}  {:>8s}  {:>9s} {:>9s} {:>7s}".format("window", "colony", "bce_sp", "bce_co", "gap"))
    for b in sorted(set(list(bce) + list(col))):
        cs = ("%.4f" % (col[b][0] / col[b][1])) if b in col and col[b][1] else "-"
        if b in bce and bce[b][2]:
            spm, com = bce[b][0] / bce[b][2], bce[b][1] / bce[b][2]
            print("  {:>4}-{:<4}  {:>8}  {:>9.3f} {:>9.3f} {:>+7.3f}".format(
                b * BIN // 1000, (b + 1) * BIN // 1000, cs, spm, com, spm - com))
        else:
            print("  {:>4}-{:<4}  {:>8}  {:>9} {:>9} {:>7}".format(
                b * BIN // 1000, (b + 1) * BIN // 1000, cs, "-", "-", "-"))
PYEOF
