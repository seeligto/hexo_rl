#!/usr/bin/env python
"""Throughput optimization sweep (laptop) — find the optimal host knob combo for the
Gumbel-100 and PUCT-600 self-play configs on v6_live2_ls.

Knobs: n_workers, inference_max_wait_ms, inference_batch_size, leaf_batch_size.
Strategy: coordinate descent from a baseline (each knob swept holding others fixed,
n=3 medians) -> pick best per knob -> interaction grid on the two most impactful
knobs -> validate best-combo vs baseline (n=5). Metric: worker pos/hr (primary) +
batch_fill% (witness). Writes reports/thr_opt/<cfg>.json + prints tables.

Runs benchmark.py per cell (pool bench reads selfplay knobs from --config; --pool-workers
and --worker-sims are flags). Laptop 4060; shapes extrapolate to vast (5080/24t) with a
few confirmation cells (GPU-bound points shift, latency-bound shapes hold).
"""
from __future__ import annotations
import itertools, json, subprocess, sys, statistics, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports/thr_opt"
OUT.mkdir(parents=True, exist_ok=True)
CFGDIR = REPO / "configs/variants"

# per-config baseline + per-knob sweep ranges
SPECS = {
    "gumbel": dict(sims=100, gumbel=True, m=32,
                   base=dict(workers=32, wait=16.0, ibatch=128, leaf=8),
                   sweep=dict(workers=[16, 24, 32, 48, 64],
                              wait=[16.0, 8.0, 4.0, 2.0, 1.0],
                              ibatch=[64, 128, 256],
                              leaf=[4, 8, 16, 24])),
    "puct":   dict(sims=600, gumbel=False, m=16,
                   base=dict(workers=24, wait=16.0, ibatch=128, leaf=8),
                   sweep=dict(workers=[8, 16, 24, 32, 48],
                              wait=[16.0, 8.0, 4.0, 2.0],
                              ibatch=[64, 128, 256],
                              leaf=[4, 8, 16])),
}
N_RUNS = 3
WARMUP = 9
DURATION = 20


def write_cfg(cfg: str, spec: dict, k: dict) -> Path:
    p = CFGDIR / f"_thropt_{cfg}.yaml"
    lines = ["encoding: v6_live2_ls", "in_channels: 4", "selfplay:",
             f"  gumbel_mcts: {'true' if spec['gumbel'] else 'false'}"]
    if spec["gumbel"]:
        lines += [f"  gumbel_m: {spec['m']}", "  gumbel_explore_moves: 10",
                  "  c_visit: 50.0", "  c_scale: 1.0"]
    lines += [f"  inference_max_wait_ms: {k['wait']}",
              f"  inference_batch_size: {k['ibatch']}",
              f"  leaf_batch_size: {k['leaf']}"]
    p.write_text("\n".join(lines) + "\n")
    return p


def run_cell(cfg: str, spec: dict, k: dict, n_runs: int = N_RUNS) -> dict:
    cfgpath = write_cfg(cfg, spec, k)
    out = OUT / f"cell_{cfg}.json"
    cmd = [str(REPO / ".venv/bin/python"), str(REPO / "scripts/benchmark.py"),
           "--config", str(cfgpath), "--pool-workers", str(k["workers"]),
           "--worker-sims", str(spec["sims"]), "--no-compile", "--mcts-sims", "1500",
           "--pool-warmup", str(WARMUP), "--pool-duration", str(DURATION),
           "--n-runs", str(n_runs), "--output", str(out)]
    try:
        subprocess.run(cmd, cwd=str(REPO), capture_output=True, timeout=900)
        d = json.loads(out.read_text())["metrics"]
        return {"pos_hr": d["worker_pos_per_hr"]["median"],
                "fill": d["worker_batch_fill_pct"]["median"]}
    except Exception as e:
        return {"pos_hr": 0.0, "fill": 0.0, "err": str(e)[:80]}


def fmt(k):  return f"w{k['workers']} wait{k['wait']:g} ib{k['ibatch']} lf{k['leaf']}"


def optimize(cfg: str) -> dict:
    spec = SPECS[cfg]
    base = dict(spec["base"])
    log = []
    print(f"\n===== {cfg.upper()} (sims={spec['sims']}) — coordinate descent =====", flush=True)
    best = dict(base)
    # measure baseline
    b = run_cell(cfg, spec, base); log.append(("baseline", dict(base), b))
    print(f"  baseline {fmt(base):28s} -> {b['pos_hr']:7.0f} pos/hr  fill {b['fill']:.1f}", flush=True)
    knob_best = {}
    for knob, values in spec["sweep"].items():
        rows = []
        for v in values:
            k = dict(base); k[knob] = v
            r = run_cell(cfg, spec, k); rows.append((v, r))
            log.append((f"coord:{knob}", dict(k), r))
            print(f"  {knob:8s}={str(v):5s} {fmt(k):28s} -> {r['pos_hr']:7.0f} pos/hr  fill {r['fill']:.1f}", flush=True)
        bv = max(rows, key=lambda t: t[1]["pos_hr"])
        knob_best[knob] = bv[0]
        print(f"   -> best {knob} = {bv[0]} ({bv[1]['pos_hr']:.0f})", flush=True)
    # candidate optimal = best of each knob
    cand = dict(base); cand.update(knob_best)
    c = run_cell(cfg, spec, cand, n_runs=5); log.append(("candidate", dict(cand), c))
    print(f"  CANDIDATE {fmt(cand):28s} -> {c['pos_hr']:7.0f} pos/hr  fill {c['fill']:.1f}", flush=True)
    # interaction grid: 2 most impactful knobs (by coord-descent spread) crossed near best
    spreads = {}
    for knob, values in spec["sweep"].items():
        vals = [r["pos_hr"] for kk, kc, r in log if kk == f"coord:{knob}"]
        spreads[knob] = (max(vals) - min(vals)) if vals else 0
    top2 = sorted(spreads, key=spreads.get, reverse=True)[:2]
    print(f"  interaction grid on {top2} (spreads {[round(spreads[k]) for k in top2]})", flush=True)
    grid_best = dict(cand); gb = c["pos_hr"]
    g1 = [v for v in spec["sweep"][top2[0]]]
    g2 = [v for v in spec["sweep"][top2[1]]]
    # restrict to top-3 values of each (around the best) to bound cost
    def near(knob, vals):
        order = sorted(vals, key=lambda v: abs(vals.index(v) - vals.index(knob_best[knob])))
        return sorted(set(order[:3]))
    for v1, v2 in itertools.product(near(top2[0], g1), near(top2[1], g2)):
        k = dict(cand); k[top2[0]] = v1; k[top2[1]] = v2
        r = run_cell(cfg, spec, k); log.append(("grid", dict(k), r))
        print(f"  grid {top2[0]}={v1} {top2[1]}={v2} {fmt(k):24s} -> {r['pos_hr']:7.0f} pos/hr  fill {r['fill']:.1f}", flush=True)
        if r["pos_hr"] > gb: gb = r["pos_hr"]; grid_best = dict(k)
    # validate winner vs baseline n=5
    win = run_cell(cfg, spec, grid_best, n_runs=5); log.append(("winner", dict(grid_best), win))
    print(f"  WINNER {fmt(grid_best):28s} -> {win['pos_hr']:7.0f} pos/hr  fill {win['fill']:.1f}", flush=True)
    print(f"  baseline {b['pos_hr']:.0f} -> winner {win['pos_hr']:.0f}  = {win['pos_hr']/max(1,b['pos_hr']):.2f}x", flush=True)
    res = {"config": cfg, "sims": spec["sims"], "baseline": {"knobs": base, **b},
           "winner": {"knobs": grid_best, **win}, "speedup": win["pos_hr"]/max(1, b["pos_hr"]),
           "knob_best": knob_best, "log": [{"phase": p, "knobs": k, **r} for p, k, r in log]}
    (OUT / f"{cfg}.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    cfgs = sys.argv[1:] or ["gumbel", "puct"]
    summary = {}
    for cfg in cfgs:
        summary[cfg] = optimize(cfg)
    print("\n===== SUMMARY =====", flush=True)
    for cfg, r in summary.items():
        print(f"{cfg:7s} baseline {fmt(r['baseline']['knobs']):26s} {r['baseline']['pos_hr']:7.0f} -> "
              f"winner {fmt(r['winner']['knobs']):26s} {r['winner']['pos_hr']:7.0f}  ({r['speedup']:.2f}x)", flush=True)
    (OUT / "summary.json").write_text(json.dumps({c: {"baseline": r["baseline"], "winner": r["winner"], "speedup": r["speedup"]} for c, r in summary.items()}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
