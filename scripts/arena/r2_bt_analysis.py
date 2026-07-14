#!/usr/bin/env python3
"""R2 BC-scaling rung BT analysis (D-M R-LADDER, run3 convene ruling
AMENDMENT 1, `docs/handoffs/run3_convene_ruling_amendment_1.md`, R2).

Adapted from reports/probes/gnn_bc/build_analysis.py (untracked WP3 script;
this file is a NEW, self-contained, TRACKED port — it does not import the
untracked one). Reads the round-robin output from
scripts/arena/run_r2_scaling_tourney.py (5-bot field: gnn_bc_200k, gnn_bc_40k,
mantis261k_raw, strix_raw, sealbot_d5) and:

  1. Computes Bradley-Terry MLE Elo (win=1, draw=0.5), same iterative
     algorithm as reports/tourney/argmax/build_analysis_argmax_final.py::bt_mle
     and the WP3 build_analysis.py (kept byte-for-byte identical so ratings
     land on the same Elo scale as ARGMAX_FINAL.md / GNN_BC_VERDICT.md).
  2. Bootstraps (N=1000, seed=42) resampled at the (pairing, opening_idx)
     CELL level (a cell = the 2 color-swapped games at one opening) — the
     WP3-standard method, preserves opening-level correlation. This is the
     PRIMARY CI and is what the frozen verdict below is decided on.
  3. Effective-n (D-ARGMAX rule, CLAUDE.md `Verify the measurement unit
     before building a frame on it`): a deterministic argmax/raw-policy
     regime can replay byte-identical move sequences under different
     (pairing, opening_idx) labels, which would silently inflate n without
     adding information. Per pairing we hash each decided game's full
     ordered move list (same signature convention as
     build_analysis_argmax_final.py's "distinct trajectories" hash) and
     report BOTH the raw game count and the deduped distinct count (eff_n)
     in the ratings/pairing table.
  4. A SENSITIVITY bootstrap (N=1000, seed=42) resampled directly over the
     deduped GAME list (one game per distinct move-sequence hash, per
     pairing) rather than over cells. Both the primary (cell) CI and the
     dedup (distinct-game) CI are reported side by side, explicitly labeled.
     If they disagree qualitatively (one excludes 0, the other doesn't) this
     is flagged in the output — per the D-ARGMAX corollary, a CI that only
     "resolves" because of duplicated near-identical games is not trusted
     over one that doesn't.

## Verdict operationalization (R2 FROZEN VERDICTS, applied verbatim from the
amendment; decided on the PRIMARY cell-level CI, dedup CI reported alongside)

Amendment text (docs/handoffs/run3_convene_ruling_amendment_1.md, R2):
  "FROZEN: BC-scaled gnn >= mantis-261k-raw (CI incl.) -> ARCHITECTURE CASE
  CLOSED; recommendation hardens to run3 = GNN + dist-tail. Below mantis but
  improved vs 40k -> case remains strong (tournament covers production
  scale); proceed to R4 decision with that stated. Flat vs 40k -> BC-
  saturated; decision rests on tournament evidence alone, stated as such."

This script operationalizes each clause as follows (stated explicitly here
per the amendment's own instruction to state the operationalization):

  - "gnn_200k >= mantis (CI incl.)": let d_m = BT(gnn_bc_200k) - BT(mantis261k_raw)
    with its bootstrap 95% CI [d_m_lo, d_m_hi]. "CI incl." reads as
    CI-INCLUSIVE of a tie: gnn_200k counts as >= mantis unless the CI sits
    STRICTLY below 0 (d_m_hi < 0, i.e. significantly worse). So:
        bc_scaled_ge_mantis := NOT (d_m_hi < 0)  ==  d_m_hi >= 0
  - "improved vs 40k": let d_40 = BT(gnn_bc_200k) - BT(gnn_bc_40k) with CI
    [d_40_lo, d_40_hi]. "improved" := CI EXCLUDES 0 in the positive
    direction (d_40_lo > 0). "flat" := CI INCLUDES 0 (d_40_lo <= 0 <= d_40_hi).
    A residual case not named by the amendment's three buckets -- CI
    excludes 0 in the NEGATIVE direction (d_40_hi < 0, i.e. 200k reads worse
    than 40k) -- is reported as its own explicit REGRESSED-VS-40K case
    rather than silently folded into "flat".

Usage:
    .venv/bin/python scripts/arena/r2_bt_analysis.py \
        [--games reports/probes/gnn_bc/r2/games_raw_r2.json] \
        [--out reports/probes/gnn_bc/r2/R2_VERDICT.md]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ELO_SCALE = 400.0 / math.log(10)
N_BOOTSTRAP = 1000
BOOT_SEED = 42

GNN_200K = "gnn-bc-200k"
GNN_40K = "gnn-bc-40k"
MANTIS = "mantis-261k-raw"
STRIX = "strix-raw"
SEALBOT = "sealbot-d5"


def bt_mle(results, bots, n_iter=1000, tol=1e-8):
    """Bradley-Terry MLE (mean-0 ln-scale). Same algorithm as
    reports/tourney/argmax/build_analysis_argmax_final.py::bt_mle and the
    WP3 reports/probes/gnn_bc/build_analysis.py (kept identical so ratings
    are on a comparable Elo scale)."""
    ratings = {b: 1.0 for b in bots}
    for _ in range(n_iter):
        new = {}
        for b in bots:
            wins = sum(s for (a, bb, s) in results if a == b) + \
                   sum(1 - s for (a, bb, s) in results if bb == b)
            denom = 0.0
            for (a, bb, s) in results:
                if a == b or bb == b:
                    other = bb if a == b else a
                    denom += 1.0 / (ratings[b] + ratings[other])
            new[b] = ratings[b] if denom < 1e-12 else wins / denom
        total = sum(new.values())
        if total < 1e-12:
            break
        factor = len(bots) / total
        new = {b: v * factor for b, v in new.items()}
        max_delta = max(abs(new[b] - ratings[b]) for b in bots)
        ratings = new
        if max_delta < tol:
            break
    ln = {b: math.log(v) for b, v in ratings.items()}
    mean = sum(ln.values()) / len(bots)
    return {b: v - mean for b, v in ln.items()}


def _move_signature(game: dict) -> str:
    """Byte-identical move-sequence signature: sha256[:16] of the ordered
    flat move list, same convention as build_analysis_argmax_final.py's
    "distinct trajectories" hash."""
    moves = []
    for t in game.get("turns", []):
        for st in t.get("stones", []):
            moves.append([int(st[0]), int(st[1])])
    move_str = json.dumps(moves, separators=(",", ":"))
    return hashlib.sha256(move_str.encode()).hexdigest()[:16]


def load_games(path):
    data = json.loads(Path(path).read_text())
    rows = []
    for g in data.get("games", []):
        if g.get("error") or g.get("winner_display") is None:
            continue
        p1 = g["display_x"]  # x = P1
        p2 = g["display_o"]
        w = g["winner_display"]
        rows.append({
            "p1": p1, "p2": p2, "winner_display": w,
            "pairing": g.get("pairing", f"{min(p1,p2)}_vs_{max(p1,p2)}"),
            "opening_idx": g["opening_idx"],
            "sig": _move_signature(g),
        })
    return rows


def _score(r):
    return 1.0 if r["winner_display"] == r["p1"] else (0.0 if r["winner_display"] == r["p2"] else 0.5)


def ci(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return (float("nan"), float("nan"))
    return v[int(0.025 * n)], v[min(int(0.975 * n), n - 1)]


def bootstrap_cells(games, bots, n_bootstrap=N_BOOTSTRAP, seed=BOOT_SEED):
    """Primary bootstrap: resample at the (pairing, opening_idx) cell level."""
    cells = defaultdict(list)
    for r in games:
        cells[(r["pairing"], r["opening_idx"])].append((r["p1"], r["p2"], _score(r)))
    keys = list(cells.keys())
    rng = random.Random(seed)
    boot = defaultdict(list)
    for _ in range(n_bootstrap):
        sampled = rng.choices(keys, k=len(keys))
        br = []
        for k in sampled:
            br.extend(cells[k])
        try:
            rr = bt_mle(br, bots, n_iter=500)
            for b, v in rr.items():
                boot[b].append(v * ELO_SCALE)
        except Exception:
            pass
    return boot


def bootstrap_games(games, bots, n_bootstrap=N_BOOTSTRAP, seed=BOOT_SEED):
    """Sensitivity bootstrap: resample directly over an already-deduped game
    list (one draw per distinct game, not per cell)."""
    rows = [(r["p1"], r["p2"], _score(r)) for r in games]
    rng = random.Random(seed)
    boot = defaultdict(list)
    for _ in range(n_bootstrap):
        sampled = rng.choices(rows, k=len(rows))
        try:
            rr = bt_mle(sampled, bots, n_iter=500)
            for b, v in rr.items():
                boot[b].append(v * ELO_SCALE)
        except Exception:
            pass
    return boot


def dedupe_per_pairing(games):
    """Keep the first game seen per (pairing, move-signature). Returns the
    deduped game list plus {pairing: (raw_n, eff_n)}."""
    seen = defaultdict(set)
    deduped = []
    raw_n = defaultdict(int)
    for r in games:
        raw_n[r["pairing"]] += 1
        if r["sig"] not in seen[r["pairing"]]:
            seen[r["pairing"]].add(r["sig"])
            deduped.append(r)
    eff_n = {p: len(s) for p, s in seen.items()}
    stats = {p: (raw_n[p], eff_n.get(p, 0)) for p in raw_n}
    return deduped, stats


def delta_ci(boot, a, b):
    vals = [x - y for x, y in zip(boot[a], boot[b])] if len(boot[a]) == len(boot[b]) else None
    if vals is None:
        # bootstrap draws for different bots come from the same iteration index
        # (same resample per iter) as long as both lists are populated every
        # iter; guard anyway by truncating to the common length.
        n = min(len(boot[a]), len(boot[b]))
        vals = [boot[a][i] - boot[b][i] for i in range(n)]
    return ci(vals)


def excludes_0(lo, hi):
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", default="reports/probes/gnn_bc/r2/games_raw_r2.json")
    ap.add_argument("--out", default="reports/probes/gnn_bc/r2/R2_VERDICT.md")
    args = ap.parse_args()

    games = load_games(args.games)
    bots = sorted({r["p1"] for r in games} | {r["p2"] for r in games})
    for required in (GNN_200K, GNN_40K):
        assert required in bots, f"field missing R2 GNN arm {required!r}: {bots}"

    results = [(r["p1"], r["p2"], _score(r)) for r in games]
    bt = bt_mle(results, bots)
    bt_elo = {b: bt[b] * ELO_SCALE for b in bots}

    # ── effective-n (D-ARGMAX rule) ──
    deduped_games, pairing_stats = dedupe_per_pairing(games)

    # ── primary bootstrap: (pairing, opening_idx) cells ──
    boot_cells = bootstrap_cells(games, bots)
    bt_ci_cells = {b: ci(boot_cells[b]) for b in bots}

    # ── sensitivity bootstrap: deduped distinct games ──
    boot_dedup = bootstrap_games(deduped_games, bots)
    bt_ci_dedup = {b: ci(boot_dedup[b]) for b in bots}

    d40_lo, d40_hi = delta_ci(boot_cells, GNN_200K, GNN_40K)
    d40_lo_dd, d40_hi_dd = delta_ci(boot_dedup, GNN_200K, GNN_40K)
    delta_40 = bt_elo[GNN_200K] - bt_elo[GNN_40K]

    dm_lo, dm_hi = delta_ci(boot_cells, GNN_200K, MANTIS) if MANTIS in bots else (float("nan"), float("nan"))
    dm_lo_dd, dm_hi_dd = delta_ci(boot_dedup, GNN_200K, MANTIS) if MANTIS in bots else (float("nan"), float("nan"))
    delta_mantis = bt_elo[GNN_200K] - bt_elo.get(MANTIS, float("nan"))

    # ── frozen R2 verdict (decided on the PRIMARY cell-level CI) ──
    bc_scaled_ge_mantis = MANTIS in bots and not (dm_hi < 0)
    improved_vs_40k = d40_lo > 0
    flat_vs_40k = d40_lo <= 0 <= d40_hi
    regressed_vs_40k = d40_hi < 0  # not named by the amendment's 3 buckets; reported explicitly

    if bc_scaled_ge_mantis:
        verdict = "ARCHITECTURE CASE CLOSED"
        verdict_text = ("BC-scaled gnn >= mantis-261k-raw (CI-inclusive) -> ARCHITECTURE CASE CLOSED; "
                         "recommendation hardens to run3 = GNN + dist-tail.")
    elif improved_vs_40k:
        verdict = "CASE REMAINS STRONG (below mantis, improved vs 40k)"
        verdict_text = ("Below mantis but improved vs 40k -> case remains strong (tournament covers "
                         "production scale); proceed to R4 decision with that stated.")
    elif flat_vs_40k:
        verdict = "BC-SATURATED (flat vs 40k)"
        verdict_text = ("Flat vs 40k -> BC-saturated; decision rests on tournament evidence alone, "
                         "stated as such.")
    else:
        assert regressed_vs_40k
        verdict = "REGRESSED VS 40K (not a pre-registered bucket)"
        verdict_text = ("gnn_bc_200k reads WORSE than gnn_bc_40k with CI excluding 0 in the negative "
                         "direction. The amendment's three frozen buckets (>=mantis / improved-vs-40k / "
                         "flat-vs-40k) do not name this case. Reported as-is, not folded into 'flat' -- "
                         "escalate to the operator before proceeding to R4.")

    # dedup sensitivity disagreement flags
    dedup_disagree_40k = excludes_0(d40_lo, d40_hi) != excludes_0(d40_lo_dd, d40_hi_dd)
    dedup_disagree_mantis = (not (dm_hi < 0)) != (not (dm_hi_dd < 0)) if MANTIS in bots else False

    order = sorted(bots, key=lambda b: -bt_elo[b])
    lines = [
        "# R2 BC-scaling rung — BT verdict (D-M R-LADDER)",
        "",
        f"Games: {len(games)} decided | Field: {len(bots)} | "
        f"bootstrap N={N_BOOTSTRAP} seed={BOOT_SEED}.",
        "",
        "Source: `docs/handoffs/run3_convene_ruling_amendment_1.md` (commit b81ca86), R2. "
        "Verdict decided on the PRIMARY (pairing, opening_idx) cell-level bootstrap; the "
        "deduped-distinct-game bootstrap is reported alongside as a D-ARGMAX sensitivity check "
        "(CLAUDE.md \"Verify the measurement unit before building a frame on it\").",
        "",
        "## Bradley-Terry ratings (Elo scale, mean=0)",
        "",
        "| Rank | Bot | BT Elo | 95% CI (cell boot) | 95% CI (dedup-game boot) |",
        "|---|---|---|---|---|",
    ]
    for i, b in enumerate(order, 1):
        c_lo, c_hi = bt_ci_cells[b]
        d_lo, d_hi = bt_ci_dedup[b]
        lines.append(f"| {i} | {b} | {bt_elo[b]:+.0f} | [{c_lo:+.0f}, {c_hi:+.0f}] | [{d_lo:+.0f}, {d_hi:+.0f}] |")

    lines += [
        "",
        "## Effective-n per pairing (D-ARGMAX rule)",
        "",
        "Raw n = decided games in the pairing. eff_n = distinct byte-identical move-sequence "
        "count (sha256[:16] of the ordered flat move list, same convention as "
        "`build_analysis_argmax_final.py`'s \"distinct trajectories\" hash). raw_n > eff_n means "
        "the deterministic raw-policy regime replayed an identical trajectory under a different "
        "(pairing, opening_idx) label.",
        "",
        "| Pairing | raw n | eff_n |",
        "|---|---|---|",
    ]
    for p in sorted(pairing_stats):
        rn, en = pairing_stats[p]
        flag = "  <-- raw_n != eff_n" if rn != en else ""
        lines.append(f"| {p} | {rn} | {en}{flag} |")

    lines += [
        "",
        "## Isolating comparisons",
        "",
        f"**Δ(gnn_bc_200k − gnn_bc_40k) = {delta_40:+.0f} Elo**  "
        f"cell-boot CI [{d40_lo:+.0f}, {d40_hi:+.0f}]  |  dedup-boot CI [{d40_lo_dd:+.0f}, {d40_hi_dd:+.0f}]"
        + ("  **[DISAGREE: excludes-0 flips between cell and dedup boot]**" if dedup_disagree_40k else ""),
        "",
        f"**Δ(gnn_bc_200k − mantis-261k-raw) = {delta_mantis:+.0f} Elo**  "
        f"cell-boot CI [{dm_lo:+.0f}, {dm_hi:+.0f}]  |  dedup-boot CI [{dm_lo_dd:+.0f}, {dm_hi_dd:+.0f}]"
        + ("  **[DISAGREE: >=mantis flips between cell and dedup boot]**" if dedup_disagree_mantis else ""),
        "",
        f"- gnn_bc_200k {bt_elo[GNN_200K]:+.0f} vs gnn_bc_40k {bt_elo[GNN_40K]:+.0f} vs "
        f"mantis-261k-raw {bt_elo.get(MANTIS, float('nan')):+.0f}"
        + (f" vs strix-raw {bt_elo.get(STRIX, float('nan')):+.0f}" if STRIX in bots else "")
        + (f" vs sealbot-d5 {bt_elo.get(SEALBOT, float('nan')):+.0f}" if SEALBOT in bots else ""),
        f"- bc_scaled_ge_mantis (NOT CI-hi < 0): {bc_scaled_ge_mantis}",
        f"- improved_vs_40k (CI excl. 0, positive): {improved_vs_40k}",
        f"- flat_vs_40k (CI incl. 0): {flat_vs_40k}",
        f"- regressed_vs_40k (CI excl. 0, negative; not a pre-registered bucket): {regressed_vs_40k}",
        "",
        f"## VERDICT: **{verdict}**",
        "",
        verdict_text,
        "",
        "Operationalization is stated explicitly in this script's module docstring and above; "
        "see `docs/handoffs/run3_convene_ruling_amendment_1.md` R2 for the frozen verbatim rule "
        "this applies.",
    ]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
