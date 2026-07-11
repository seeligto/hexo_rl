"""D-F HEADSWAP — pre-registered metrics + frozen verdict over the scores tables.

Binds to scripts/headswap/RECIPE.md §"Metrics", §"PRE-REGISTERED VERDICTS".
Consumes the 4 arms' scores tables (scores_{ARM}_seed{S}.jsonl from score_all.py)
and produces reports/headswap/VERDICT.md.

Pre-registered pipeline (all frozen BEFORE any arm trains):
  1. MATCHING     greedy 1:1 positive<->negative on ply_band (single phase, so
                  ply_band is the only axis), cap <=2 negatives / source game.
                  EXACT-band + RELAXED nearest-band variants; hard gate n>=200.
  2. CANARY       logistic classifier on ply/t predicting lost-vs-safe on the
                  matched set; AUC must be <=0.55 else NO-TEST (run exact+relaxed).
  3. PRIMARY AUC  lost(=positive)-vs-safe AUC per arm scored by decoded scalar v
                  (the GATE). 65-bin arms ALSO scored by tail-mass (mechanism).
  4. PAIRED dAUC  (B-A) and (C-D) under decoded-scalar-v (gate) AND tail-mass
                  (mechanism). CI = CLUSTER BOOTSTRAP by SOURCE GAME (10k
                  resamples) over paired per-position score differences.
  5. false-pess.  fraction of SAFE negatives scored pessimistic (v <= -0.5).
  6. ECE + sign   only if a holdout-scores file is supplied; else "N/A".
  7. RED-TEAM     tail-mass rule: monotone-transform invariance + threshold sweep.
  8. VERDICT      PASS / PASS-JOINT / TRUNK-FORK, gated on decoded-scalar-v dAUC.

REGISTER GUARD (INV-D1): metrics only. lost/safe are probe LABELS from the
SealBot/solver instrument; never a training target here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FALSE_PESS_THRESHOLD = -0.5   # a "doom-happy" head must not free-score safes below this
CANARY_MAX_AUC = 0.55         # matched set is NO-TEST if ply/t predicts label above this
PASS_DELTA = 0.05             # AUC lift gate
FALSE_PESS_SLACK = 0.05       # 5pp slack on false-pessimism for PASS
MATCHED_N_GATE = 200          # hard gate on matched positives+negatives count... see note
BOOTSTRAP_RESAMPLES = 10_000
TAIL_BIN_DEFAULT = 16         # support[16] == -0.5

# The 65-bin support (linspace(-1,1,65)); tail thresholds index into it.
_SUPPORT = np.linspace(-1.0, 1.0, 65)


# ── AUC (Mann-Whitney rank statistic; ties -> midrank) ────────────────────────


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUC of `scores` predicting label==1 (positive) via the rank-sum identity.

    AUC = (R_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg), midranks for ties.
    Invariant to any strictly-monotone transform of `scores` (rank-based) — the
    red-team monotone-invariance claim rests on THIS. Undefined (nan) if either
    class is empty.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        midrank = (i + j) / 2.0 + 1.0  # 1-based midrank over the tie block
        ranks[order[i:j + 1]] = midrank
        i = j + 1
    r_pos = ranks[y == 1].sum()
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ── logistic regression (Newton-Raphson / IRLS, no sklearn) ───────────────────


def _logistic_fit(X: np.ndarray, y: np.ndarray, iters: int = 50) -> np.ndarray:
    """Fit logistic weights (with intercept column already in X) via IRLS.
    Ridge-stabilised (1e-6) so a separating feature can't blow up."""
    n, d = X.shape
    w = np.zeros(d)
    ridge = 1e-6 * np.eye(d)
    for _ in range(iters):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        wgt = np.clip(p * (1.0 - p), 1e-8, None)
        grad = X.T @ (p - y)
        H = X.T @ (X * wgt[:, None]) + ridge
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        w_new = w - step
        if np.max(np.abs(w_new - w)) < 1e-8:
            w = w_new
            break
        w = w_new
    return w


def canary_auc(features: np.ndarray, labels: Sequence[int]) -> float:
    """AUC of a logistic classifier on `features` (n, k) predicting the label.
    In-sample (a leakage upper bound — if even the in-sample fit can't beat 0.55,
    the matched set is safely balanced on ply/t)."""
    y = np.asarray(labels, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    # standardise for conditioning; prepend intercept.
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Xi = np.hstack([np.ones((len(Xs), 1)), Xs])
    w = _logistic_fit(Xi, y)
    scores = Xi @ w
    return auc(scores, np.asarray(labels, dtype=int))


# ── greedy 1:1 matching on ply_band (exact + relaxed nearest-band) ────────────


def greedy_match(
    positives: List[dict],
    negatives: List[dict],
    max_neg_per_game: int = 2,
    relaxed: bool = False,
) -> Dict[str, object]:
    """Greedy 1:1 positive<->negative match on ply_band.

    EXACT (relaxed=False): a positive can only pair a negative in the SAME
    ply_band. RELAXED (relaxed=True): if the band is exhausted, pair the nearest
    band still holding an unused negative (deep-band shortfall recovery).

    Cap: at most `max_neg_per_game` negatives consumed per source_game_id.
    Returns matched pos/neg lists (aligned), per-band coverage, drops, and the
    band-distance histogram for the relaxed variant.
    """
    # bucket negatives by band, preserve order for determinism
    by_band: Dict[int, List[dict]] = {}
    for n in negatives:
        by_band.setdefault(int(n["ply_band"]), []).append(dict(n))
    used_per_game: Dict[str, int] = {}
    all_bands = sorted(by_band.keys())

    matched_pos: List[dict] = []
    matched_neg: List[dict] = []
    band_dist: List[int] = []
    drops = {"no_band_neg": 0, "game_cap": 0}
    per_band_cov: Dict[int, Dict[str, int]] = {}

    # process positives band-major, deepest-first is irrelevant for exact but
    # helps relaxed drain scarce deep negs onto deep positives before shallow
    # positives steal them.
    pos_sorted = sorted(positives, key=lambda p: -int(p["ply_band"]))

    def _pop_from_band(band: int) -> Optional[dict]:
        pool = by_band.get(band, [])
        while pool:
            cand = pool[0]
            gid = cand["source_game_id"]
            if used_per_game.get(gid, 0) >= max_neg_per_game:
                # cap hit for this game -> discard this candidate, try next
                pool.pop(0)
                drops["game_cap"] += 1
                continue
            pool.pop(0)
            used_per_game[gid] = used_per_game.get(gid, 0) + 1
            return cand
        return None

    def _nearest_bands(target: int) -> List[int]:
        return sorted(all_bands, key=lambda b: (abs(b - target), b))

    for p in pos_sorted:
        band = int(p["ply_band"])
        per_band_cov.setdefault(band, {"pos": 0, "matched": 0})
        per_band_cov[band]["pos"] += 1
        neg = _pop_from_band(band)
        dist = 0
        if neg is None and relaxed:
            for b in _nearest_bands(band):
                if b == band:
                    continue
                neg = _pop_from_band(b)
                if neg is not None:
                    dist = abs(b - band)
                    break
        if neg is None:
            drops["no_band_neg"] += 1
            continue
        matched_pos.append(p)
        matched_neg.append(neg)
        band_dist.append(dist)
        per_band_cov[band]["matched"] += 1

    n_matched_pairs = len(matched_pos)
    return {
        "matched_pos": matched_pos,
        "matched_neg": matched_neg,
        "band_dist": band_dist,
        "n_pairs": n_matched_pairs,
        "n_matched_total": 2 * n_matched_pairs,  # positives + negatives
        "per_band_coverage": {int(k): v for k, v in sorted(per_band_cov.items())},
        "drops": drops,
        "relaxed": relaxed,
        "gate_pass": (2 * n_matched_pairs) >= MATCHED_N_GATE,
    }


# ── tail-mass re-derivation for threshold sweep ───────────────────────────────
# The scores tables carry the FROZEN tail-mass at bin 16 already. The threshold
# sweep needs P(v<=support[thr]) at other bins, which requires the per-bin
# softmax. That is not in the tidy table (only decoded v + tail@16). So the
# sweep operates on whatever per-bin tail columns are supplied; if only the
# bin-16 tail exists, the sweep reports "single-threshold (bin16) only".


# ── cluster bootstrap by source game over paired dAUC ─────────────────────────


def _paired_arrays(
    matched_pos: List[dict],
    matched_neg: List[dict],
    scores_hi: Dict[str, float],
    scores_lo: Dict[str, float],
    score_key_pos: str,
    score_key_neg: str,
) -> Optional[Dict[str, np.ndarray]]:
    """Build aligned arrays for a paired-arm dAUC bootstrap.

    scores_hi / scores_lo map position_id -> that arm's score (v or tail_mass).
    Returns arrays over the FULL matched set (pos then neg), the two arms'
    scores, labels, and the per-position source_game_id (bootstrap cluster).
    None if any matched position is missing a score in either arm.
    """
    ids: List[str] = []
    labels: List[int] = []
    games: List[str] = []
    hi: List[float] = []
    lo: List[float] = []
    for p in matched_pos:
        pid = p["position_id"]
        if pid not in scores_hi or pid not in scores_lo:
            return None
        ids.append(pid); labels.append(1); games.append(p["source_game_id"])
        hi.append(scores_hi[pid]); lo.append(scores_lo[pid])
    for n in matched_neg:
        pid = n["position_id"]
        if pid not in scores_hi or pid not in scores_lo:
            return None
        ids.append(pid); labels.append(0); games.append(n["source_game_id"])
        hi.append(scores_hi[pid]); lo.append(scores_lo[pid])
    return {
        "labels": np.asarray(labels, dtype=int),
        "games": np.asarray(games, dtype=object),
        "hi": np.asarray(hi, dtype=float),
        "lo": np.asarray(lo, dtype=float),
    }


def cluster_bootstrap_delta_auc(
    labels: np.ndarray,
    games: np.ndarray,
    hi: np.ndarray,
    lo: np.ndarray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> Dict[str, float]:
    """Cluster bootstrap by SOURCE GAME on dAUC = AUC(hi) - AUC(lo).

    Resample source games WITH replacement; rebuild the pooled positive/negative
    score vectors from the resampled games; recompute both arms' AUC on the SAME
    resample (paired); dAUC = difference. Report point estimate (full sample),
    percentile 95% CI, achieved SE (bootstrap sd), and the empirical between-arm
    score correlation r (Pearson over all matched positions).
    """
    rng = np.random.default_rng(seed)
    point = auc(hi, labels) - auc(lo, labels)

    # group row indices by source game
    unique_games = np.unique(games)
    game_to_rows: Dict[str, np.ndarray] = {
        g: np.where(games == g)[0] for g in unique_games
    }
    g_arr = list(unique_games)
    n_games = len(g_arr)

    deltas = np.empty(resamples, dtype=float)
    valid = 0
    for b in range(resamples):
        pick = rng.integers(0, n_games, size=n_games)
        rows = np.concatenate([game_to_rows[g_arr[k]] for k in pick])
        yb = labels[rows]
        if int((yb == 1).sum()) == 0 or int((yb == 0).sum()) == 0:
            continue  # degenerate resample (no contrast); skip
        d = auc(hi[rows], yb) - auc(lo[rows], yb)
        if not math.isnan(d):
            deltas[valid] = d
            valid += 1
    deltas = deltas[:valid]

    if valid == 0:
        return {
            "delta_auc": float(point), "ci_lo": float("nan"), "ci_hi": float("nan"),
            "se": float("nan"), "r": float("nan"), "n_resamples_valid": 0,
        }
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    # between-arm correlation over the matched positions (descriptive)
    if np.std(hi) == 0 or np.std(lo) == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(hi, lo)[0, 1])
    return {
        "delta_auc": float(point),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "se": float(np.std(deltas, ddof=1)) if valid > 1 else float("nan"),
        "r": r,
        "n_resamples_valid": int(valid),
    }


# ── false-pessimism, ECE, sign accuracy ───────────────────────────────────────


def false_pessimism(neg_scores: Sequence[float], threshold: float = FALSE_PESS_THRESHOLD) -> float:
    """Fraction of SAFE negatives scored pessimistic (decoded v <= threshold)."""
    s = np.asarray(neg_scores, dtype=float)
    if len(s) == 0:
        return float("nan")
    return float((s <= threshold).mean())


def ece_10bin(v: Sequence[float], y: Sequence[int]) -> float:
    """10-bin expected calibration error. v in [-1,1] mapped to p=(v+1)/2;
    label y in {0,1} = win(1)/loss(0) of the scored side. Descriptive only."""
    p = (np.asarray(v, dtype=float) + 1.0) / 2.0
    y = np.asarray(y, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, 11)
    n = len(p)
    if n == 0:
        return float("nan")
    ece = 0.0
    for k in range(10):
        lo, hi = edges[k], edges[k + 1]
        m = (p >= lo) & (p < hi) if k < 9 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def sign_accuracy(v: Sequence[float], y: Sequence[int]) -> float:
    """Decided-row sign accuracy: sign(v) matches (y==1 -> +, y==0 -> -).
    Rows with v==0 are excluded (undecided). Descriptive only."""
    v = np.asarray(v, dtype=float)
    y = np.asarray(y, dtype=int)
    decided = v != 0
    if decided.sum() == 0:
        return float("nan")
    pred_win = v[decided] > 0
    return float((pred_win == (y[decided] == 1)).mean())


# ── red-team hooks on the tail-mass rule ──────────────────────────────────────


def monotone_invariance_check(scores: Sequence[float], labels: Sequence[int]) -> Dict[str, float]:
    """AUC is rank-based -> invariant to any strictly-monotone transform. Verify
    numerically by remapping scores through a monotone map (sigmoid of a scaled
    shift) and confirming AUC is unchanged. Proves the tail-mass AUC is not an
    artifact of the specific -0.5 threshold MAPPING (only its rank order)."""
    s = np.asarray(scores, dtype=float)
    base = auc(s, labels)
    # strictly-monotone: x -> 1/(1+exp(-(3x-1))) ; strictly increasing on R
    remapped = 1.0 / (1.0 + np.exp(-(3.0 * s - 1.0)))
    remapped_auc = auc(remapped, labels)
    return {
        "base_auc": float(base),
        "remapped_auc": float(remapped_auc),
        "delta": float(abs(base - remapped_auc)),
        "invariant": bool(abs(base - remapped_auc) < 1e-9),
    }


def tail_threshold_sweep(
    per_bin_probs_pos: Optional[np.ndarray],
    per_bin_probs_neg: Optional[np.ndarray],
    bins: Sequence[int] = tuple(range(12, 21)),
) -> Dict[str, object]:
    """Recompute tail-mass AUC for thresholds at bins 12..20 (P(v<=support[bin]))
    to test if the +0.05 effect is robust to the -0.5 (bin 16) choice or
    cherry-picked. Requires per-bin softmax probabilities (pos, neg) shaped
    (n,65). If not supplied (tidy table carries only the frozen bin-16 tail),
    returns a not-available marker so the report states the limitation honestly.
    """
    if per_bin_probs_pos is None or per_bin_probs_neg is None:
        return {
            "available": False,
            "note": "per-bin softmax not in tidy scores table; sweep needs a "
                    "per-bin probs dump (score_all --emit-bin-probs). Frozen "
                    "tail is bin 16 (-0.5) only.",
        }
    curve = []
    n_pos = per_bin_probs_pos.shape[0]
    labels = np.concatenate([np.ones(n_pos, dtype=int),
                             np.zeros(per_bin_probs_neg.shape[0], dtype=int)])
    for b in bins:
        tail_pos = per_bin_probs_pos[:, : b + 1].sum(axis=1)
        tail_neg = per_bin_probs_neg[:, : b + 1].sum(axis=1)
        scores = np.concatenate([tail_pos, tail_neg])
        curve.append({"bin": int(b), "support": float(_SUPPORT[b]),
                      "auc": float(auc(scores, labels))})
    return {"available": True, "curve": curve}


# ── scores-table IO ───────────────────────────────────────────────────────────


def load_scores_table(path: str) -> Dict[str, List[dict]]:
    """Read a scores_{ARM}_seed{S}.jsonl -> {'loss': [...], 'safe': [...]} rows.
    Each row: position_id, set, ply_band, source_game_id, v, tail_mass."""
    loss: List[dict] = []
    safe: List[dict] = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        (loss if r["set"] == "loss" else safe).append(r)
    return {"loss": loss, "safe": safe}


def _pessimism_score(key: str, val: float) -> float:
    """Orient a raw score so HIGHER == MORE-LOST (the AUC positive class = loss).

    - decoded scalar `v`: a discriminating head scores losses MORE pessimistic
      (LOWER v), so the loss-detection score is NEGATED v. Otherwise a good head
      would read AUC < 0.5 and the +0.05 PASS gate would be inverted. AUC is
      shift/scale invariant, so -v vs v just flips the direction to match the
      "lost is the positive class" convention (RECIPE §Metrics; matches
      loss_tail_mass's stated 'higher -> more likely lost').
    - tail_mass P(v<=-0.5): already higher == more pessimism -> used as-is.
    """
    if key == "v":
        return -float(val)
    return float(val)


def _score_map(rows: List[dict], key: str) -> Dict[str, float]:
    """position_id -> pessimism-oriented score (v -> -v, tail_mass as-is).
    Skips rows whose raw score is None."""
    out: Dict[str, float] = {}
    for r in rows:
        val = r.get(key)
        if val is None:
            continue
        out[r["position_id"]] = _pessimism_score(key, val)
    return out


# ── pooling arm seeds (average per RECIPE) ────────────────────────────────────


def pool_arm_scores(seed_tables: List[Dict[str, List[dict]]]) -> Dict[str, List[dict]]:
    """Average a score across an arm's seeds per position_id (RECIPE: report
    per-seed + pooled; the primary AUC uses the pooled/averaged score). Uses the
    intersection of position_ids present in all seeds."""
    if len(seed_tables) == 1:
        return seed_tables[0]
    pooled: Dict[str, List[dict]] = {"loss": [], "safe": []}
    for setkey in ("loss", "safe"):
        # index each seed by position_id
        maps = []
        for t in seed_tables:
            maps.append({r["position_id"]: r for r in t[setkey]})
        common = set(maps[0])
        for m in maps[1:]:
            common &= set(m)
        for pid in common:
            base = dict(maps[0][pid])
            vs = [m[pid]["v"] for m in maps]
            base["v"] = float(np.mean(vs))
            tails = [m[pid].get("tail_mass") for m in maps]
            if all(t is not None for t in tails):
                base["tail_mass"] = float(np.mean(tails))
            else:
                base["tail_mass"] = None
            pooled[setkey].append(base)
    return pooled


# ── top-level orchestration ───────────────────────────────────────────────────


def per_arm_primary_auc(table: Dict[str, List[dict]]) -> Dict[str, float]:
    """AUC(loss vs safe) scored by decoded scalar v (gate) and — if tail_mass is
    present (65-bin arm) — by tail-mass (mechanism). Uses ALL scored rows.

    Both scores are pessimism-oriented (higher == more lost): decoded v is
    NEGATED, tail-mass used as-is, so AUC > 0.5 means the head separates the
    losing tail and the +0.05 gate is directionally coherent (see
    _pessimism_score)."""
    v = [_pessimism_score("v", r["v"]) for r in table["loss"]] + \
        [_pessimism_score("v", r["v"]) for r in table["safe"]]
    y = [1] * len(table["loss"]) + [0] * len(table["safe"])
    out = {"auc_v": auc(v, y), "n_loss": len(table["loss"]), "n_safe": len(table["safe"])}
    tails_loss = [r.get("tail_mass") for r in table["loss"]]
    tails_safe = [r.get("tail_mass") for r in table["safe"]]
    if all(t is not None for t in tails_loss) and all(t is not None for t in tails_safe) \
            and (tails_loss or tails_safe):
        tv = [float(t) for t in tails_loss] + [float(t) for t in tails_safe]
        out["auc_tail"] = auc(tv, y)
    else:
        out["auc_tail"] = None
    return out


def paired_delta(
    hi_table: Dict[str, List[dict]],
    lo_table: Dict[str, List[dict]],
    matched_pos: List[dict],
    matched_neg: List[dict],
    score_key: str,
    seed: int = 0,
) -> Optional[Dict[str, float]]:
    """dAUC(hi - lo) over the matched set for one score column ('v' or
    'tail_mass'), with cluster-bootstrap CI. None if either arm lacks that
    score (e.g. tail-mass on a scalar arm)."""
    hi_map = _score_map(hi_table["loss"] + hi_table["safe"], score_key)
    lo_map = _score_map(lo_table["loss"] + lo_table["safe"], score_key)
    arrays = _paired_arrays(matched_pos, matched_neg, hi_map, lo_map, score_key, score_key)
    if arrays is None:
        return None
    return cluster_bootstrap_delta_auc(
        arrays["labels"], arrays["games"], arrays["hi"], arrays["lo"], seed=seed,
    )


def within_arm_tail_vs_v(
    table: Dict[str, List[dict]],
    matched_pos: List[dict],
    matched_neg: List[dict],
    seed: int = 0,
) -> Optional[Dict[str, float]]:
    """Mechanism lift for a bin arm: dAUC = AUC(tail-mass) - AUC(decoded-v) on
    the SAME positions, cluster-bootstrapped by source game. Descriptive readout
    the dispatcher headlines (does the distributional tail beat the fair scalar
    comparate?). None if the arm has no tail-mass."""
    all_rows = table["loss"] + table["safe"]
    v_map = _score_map(all_rows, "v")            # pessimism-oriented (-v)
    tail_map = _score_map(all_rows, "tail_mass")  # as-is
    if not tail_map:
        return None
    arrays = _paired_arrays(matched_pos, matched_neg, tail_map, v_map, "tail_mass", "v")
    if arrays is None:
        return None
    return cluster_bootstrap_delta_auc(
        arrays["labels"], arrays["games"], arrays["hi"], arrays["lo"], seed=seed,
    )


def decide_verdict(
    d_ba: Optional[Dict[str, float]],
    d_cd: Optional[Dict[str, float]],
    fp_a: float,
    fp_b: float,
) -> Tuple[str, str]:
    """Frozen verdict logic (gate on decoded-scalar-v dAUC).

    PASS       : B>A dAUC >= +0.05, CI excl 0, AND fp_B <= fp_A + 5pp.
    PASS-JOINT : B~=A but C>D dAUC >= +0.05, CI excl 0.
    TRUNK-FORK : B~=A AND C~=D.
    Returns (verdict, rationale). B~=A means the PASS condition on (B-A) fails.
    """
    def _passes(d: Optional[Dict[str, float]]) -> bool:
        if d is None:
            return False
        return (d["delta_auc"] >= PASS_DELTA) and (d["ci_lo"] > 0.0)

    ba_pass_auc = _passes(d_ba)
    fp_ok = fp_b <= fp_a + FALSE_PESS_SLACK
    cd_pass = _passes(d_cd)

    if ba_pass_auc and fp_ok:
        return "PASS", (
            f"B>A dAUC={d_ba['delta_auc']:+.3f} CI[{d_ba['ci_lo']:+.3f},"
            f"{d_ba['ci_hi']:+.3f}] excl 0 AND false-pessimism_B={fp_b:.3f} "
            f"<= A+5pp ({fp_a + FALSE_PESS_SLACK:.3f}) -> card #1 CONFIRMED, "
            f"frozen-trunk sufficiency."
        )
    if ba_pass_auc and not fp_ok:
        return "TRUNK-FORK", (
            f"B>A dAUC passed but false-pessimism_B={fp_b:.3f} > A+5pp "
            f"({fp_a + FALSE_PESS_SLACK:.3f}) -> PASS blocked by doom-happy head; "
            f"treat B~=A. C-vs-D "
            + ("passed -> would be PASS-JOINT but B leaked pessimism; "
               if cd_pass else "did not pass -> ")
            + "ESCALATE."
        )
    if cd_pass:
        return "PASS-JOINT", (
            f"B~=A (dAUC gate not met) but C>D dAUC="
            f"{d_cd['delta_auc']:+.3f} CI[{d_cd['ci_lo']:+.3f},{d_cd['ci_hi']:+.3f}] "
            f">= +0.05 excl 0 -> head-shape needs feature adaptation; run3 trains "
            f"head jointly from start; D-FULLSPEC partly corroborated."
        )
    return "TRUNK-FORK", (
        "B~=A AND C~=D -> head shape buys nothing even with local features -> "
        "card #1 DEMOTED below card #2; D-FULLSPEC representation hypothesis "
        "PROMOTED; ESCALATE to operator before RUN3SPEC freezes."
    )


def run_metrics(
    arm_tables: Dict[str, List[Dict[str, List[dict]]]],
    holdout_tables: Optional[Dict[str, Dict[str, List[dict]]]] = None,
    seed: int = 0,
) -> Dict[str, object]:
    """arm_tables: {'A': [seed0_table, ...], 'B': [...], 'C': [...], 'D': [...]}.
    Each *_table is {'loss': rows, 'safe': rows}. Returns the full metrics dict
    ready for VERDICT.md rendering.
    """
    pooled = {arm: pool_arm_scores(tabs) for arm, tabs in arm_tables.items()}

    # matching uses the union of positives/negatives seen across arms (they score
    # the same probe set; use arm A's pooled table as the position universe).
    ref = pooled.get("A") or next(iter(pooled.values()))
    positives = ref["loss"]
    negatives = ref["safe"]

    match_exact = greedy_match(positives, negatives, relaxed=False)
    match_relaxed = greedy_match(positives, negatives, relaxed=True)

    def _canary(match: Dict[str, object]) -> float:
        mp = match["matched_pos"]; mn = match["matched_neg"]
        feats = np.asarray([[float(r["t"])] if "t" in r else [float(r["ply_band"])]
                            for r in mp + mn], dtype=float)
        # tidy scores table drops raw t; fall back to ply_band as the phase proxy
        if "t" not in (mp[0] if mp else {}):
            feats = np.asarray([[float(r["ply_band"])] for r in mp + mn], dtype=float)
        labels = [1] * len(mp) + [0] * len(mn)
        return canary_auc(feats, labels)

    canary_exact = _canary(match_exact) if match_exact["n_pairs"] else float("nan")
    canary_relaxed = _canary(match_relaxed) if match_relaxed["n_pairs"] else float("nan")

    primary = {arm: per_arm_primary_auc(pooled[arm]) for arm in pooled}
    fp = {arm: false_pessimism([r["v"] for r in pooled[arm]["safe"]]) for arm in pooled}

    # gate on the RELAXED match if exact fails the n>=200 gate, else exact.
    gate_match = match_exact if match_exact["gate_pass"] else match_relaxed
    mp, mn = gate_match["matched_pos"], gate_match["matched_neg"]

    d_ba_v = paired_delta(pooled["B"], pooled["A"], mp, mn, "v", seed) if {"A", "B"} <= set(pooled) else None
    d_cd_v = paired_delta(pooled["C"], pooled["D"], mp, mn, "v", seed) if {"C", "D"} <= set(pooled) else None
    # Literal RECIPE tail-mass paired (B-A, C-D): A/D are SCALAR arms with NO
    # tail-mass column, so this pairing is genuinely undefined (None). The
    # coherent mechanism readout is the WITHIN-bin-arm lift: does the
    # distributional tail-mass score beat the fair decoded-v comparate on the
    # SAME arm/positions? Reported as mechanism_lift below.
    d_ba_tail = paired_delta(pooled["B"], pooled["A"], mp, mn, "tail_mass", seed) if {"A", "B"} <= set(pooled) else None
    d_cd_tail = paired_delta(pooled["C"], pooled["D"], mp, mn, "tail_mass", seed) if {"C", "D"} <= set(pooled) else None

    mechanism_lift = {}
    for bin_arm in ("B", "C"):
        if bin_arm in pooled and primary[bin_arm]["auc_tail"] is not None:
            mechanism_lift[bin_arm] = within_arm_tail_vs_v(pooled[bin_arm], mp, mn, seed)

    verdict, rationale = decide_verdict(
        d_ba_v, d_cd_v, fp.get("A", float("nan")), fp.get("B", float("nan")),
    )

    # red-team on the mechanism (tail-mass) AUC of arm B, if present
    redteam = {}
    if "B" in pooled and primary["B"]["auc_tail"] is not None:
        tv = [r["tail_mass"] for r in pooled["B"]["loss"]] + [r["tail_mass"] for r in pooled["B"]["safe"]]
        ty = [1] * len(pooled["B"]["loss"]) + [0] * len(pooled["B"]["safe"])
        redteam["monotone_invariance"] = monotone_invariance_check(tv, ty)
        redteam["threshold_sweep"] = tail_threshold_sweep(None, None)  # tidy table = bin16 only

    # holdout ECE / sign accuracy (only if supplied)
    holdout = {}
    if holdout_tables:
        for arm, t in holdout_tables.items():
            v = [r["v"] for r in t["loss"]] + [r["v"] for r in t["safe"]]
            y = [0] * len(t["loss"]) + [1] * len(t["safe"])  # loss=0(win prob low), safe=1
            holdout[arm] = {"ece_10bin": ece_10bin(v, y), "sign_acc": sign_accuracy(v, y)}
    else:
        holdout = {"note": "N/A (holdout not supplied)"}

    return {
        "match_exact": {k: match_exact[k] for k in
                        ("n_pairs", "n_matched_total", "per_band_coverage", "drops", "gate_pass")},
        "match_relaxed": {k: match_relaxed[k] for k in
                          ("n_pairs", "n_matched_total", "per_band_coverage", "drops", "gate_pass")},
        "band_dist_relaxed": _band_dist_hist(match_relaxed["band_dist"]),
        "gate_match_used": "exact" if match_exact["gate_pass"] else "relaxed",
        "canary": {"exact_auc": canary_exact, "relaxed_auc": canary_relaxed,
                   "no_test_exact": bool(canary_exact > CANARY_MAX_AUC) if not math.isnan(canary_exact) else None,
                   "no_test_relaxed": bool(canary_relaxed > CANARY_MAX_AUC) if not math.isnan(canary_relaxed) else None},
        "primary_auc": primary,
        "false_pessimism": fp,
        "delta_auc": {
            "B_minus_A_v": d_ba_v, "C_minus_D_v": d_cd_v,
            "B_minus_A_tail": d_ba_tail, "C_minus_D_tail": d_cd_tail,
        },
        "mechanism_lift": mechanism_lift,  # within-bin-arm tail-mass minus decoded-v
        "redteam": redteam,
        "holdout": holdout,
        "verdict": verdict,
        "rationale": rationale,
    }


def _band_dist_hist(band_dist: List[int]) -> Dict[str, int]:
    from collections import Counter
    return {str(k): int(v) for k, v in sorted(Counter(band_dist).items())}


# ── VERDICT.md rendering ──────────────────────────────────────────────────────


def render_verdict_md(m: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append(f"# D-F HEADSWAP VERDICT: {m['verdict']}")
    lines.append("")
    lines.append(m["rationale"])
    lines.append("")
    # per-arm table
    lines.append("## Per-arm AUC (loss vs safe)")
    lines.append("")
    lines.append("| arm | AUC(decoded v) [GATE] | AUC(tail-mass) [mech] | false-pessimism | n_loss | n_safe |")
    lines.append("|-----|-----------------------|-----------------------|-----------------|--------|--------|")
    for arm in ("A", "B", "C", "D"):
        p = m["primary_auc"].get(arm)
        if p is None:
            continue
        at = "-" if p["auc_tail"] is None else f"{p['auc_tail']:.3f}"
        fp = m["false_pessimism"].get(arm, float("nan"))
        lines.append(f"| {arm} | {p['auc_v']:.3f} | {at} | {fp:.3f} | {p['n_loss']} | {p['n_safe']} |")
    lines.append("")
    # paired dAUC
    lines.append("## Paired dAUC (cluster bootstrap by source game, 10k)")
    lines.append("")
    lines.append("| contrast | score | dAUC | CI95 | SE | between-arm r |")
    lines.append("|----------|-------|------|------|----|--------------|")
    for label, key in (("B - A", "B_minus_A_v"), ("C - D", "C_minus_D_v"),
                       ("B - A", "B_minus_A_tail"), ("C - D", "C_minus_D_tail")):
        d = m["delta_auc"].get(key)
        score = "decoded v (GATE)" if key.endswith("_v") else "tail-mass (mech)"
        if d is None:
            lines.append(f"| {label} | {score} | N/A | - | - | - |")
            continue
        lines.append(
            f"| {label} | {score} | {d['delta_auc']:+.3f} | "
            f"[{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] | {d['se']:.3f} | {d['r']:.3f} |"
        )
    lines.append("")
    ml = m.get("mechanism_lift") or {}
    if ml:
        lines.append("## Mechanism lift (within bin-arm: tail-mass AUC - decoded-v AUC)")
        lines.append("")
        lines.append("Descriptive: does the distributional tail-mass readout beat the fair "
                     "scalar comparate on the SAME arm/positions? (A/D are scalar so the "
                     "literal B-A / C-D tail-mass pairing above is N/A by construction.)")
        lines.append("")
        lines.append("| bin arm | dAUC (tail - v) | CI95 | SE | r(tail,v) |")
        lines.append("|---------|-----------------|------|----|-----------|")
        for arm, d in ml.items():
            lines.append(
                f"| {arm} | {d['delta_auc']:+.3f} | [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] "
                f"| {d['se']:.3f} | {d['r']:.3f} |"
            )
        lines.append("")
    # canary + matching
    c = m["canary"]
    lines.append("## Phase+ply canary (logistic on ply/t; must be <= 0.55)")
    lines.append("")
    lines.append(f"- exact match: AUC={c['exact_auc']:.3f} "
                 f"(NO-TEST={c['no_test_exact']})")
    lines.append(f"- relaxed match: AUC={c['relaxed_auc']:.3f} "
                 f"(NO-TEST={c['no_test_relaxed']})")
    lines.append("")
    me, mr = m["match_exact"], m["match_relaxed"]
    lines.append("## Matching (greedy 1:1 on ply_band, cap 2 negs/game)")
    lines.append("")
    lines.append(f"- EXACT: {me['n_pairs']} pairs = {me['n_matched_total']} matched "
                 f"(gate n>=200: {'PASS' if me['gate_pass'] else 'FAIL'})")
    lines.append(f"- RELAXED (nearest-band): {mr['n_pairs']} pairs = {mr['n_matched_total']} "
                 f"matched (gate: {'PASS' if mr['gate_pass'] else 'FAIL'})")
    lines.append(f"- gate uses: {m['gate_match_used']} match")
    lines.append(f"- relaxed band-distance histogram: {m['band_dist_relaxed']}")
    lines.append(f"- exact drops: {me['drops']}")
    lines.append("")
    # red-team
    if m["redteam"]:
        lines.append("## Red-team (tail-mass rule)")
        lines.append("")
        mi = m["redteam"].get("monotone_invariance")
        if mi:
            lines.append(f"- monotone-transform invariance: base={mi['base_auc']:.6f} "
                         f"remapped={mi['remapped_auc']:.6f} delta={mi['delta']:.2e} "
                         f"invariant={mi['invariant']}")
        ts = m["redteam"].get("threshold_sweep", {})
        if ts.get("available"):
            lines.append("- threshold sweep (bins 12..20):")
            for row in ts["curve"]:
                lines.append(f"    bin {row['bin']} (v<={row['support']:+.3f}): AUC={row['auc']:.3f}")
        else:
            lines.append(f"- threshold sweep: {ts.get('note', 'unavailable')}")
        lines.append("")
    # holdout
    lines.append("## Holdout ECE + sign accuracy")
    lines.append("")
    if isinstance(m["holdout"], dict) and "note" in m["holdout"]:
        lines.append(f"- {m['holdout']['note']}")
    else:
        for arm, h in m["holdout"].items():
            lines.append(f"- {arm}: ECE(10-bin)={h['ece_10bin']:.4f} sign_acc={h['sign_acc']:.3f}")
    lines.append("")
    # deviations / caveats
    lines.append("## Deviations / caveats")
    lines.append("")
    gate_ok = (m["match_exact"]["gate_pass"] or m["match_relaxed"]["gate_pass"])
    lines.append(f"- matched-n gate (>=200): {'PASS' if gate_ok else 'FAIL — set is UNDER-POWERED'} "
                 f"(exact={m['match_exact']['n_matched_total']}, relaxed={m['match_relaxed']['n_matched_total']})")
    lines.append("- GATE = decoded-scalar-v dAUC (fair comparate); tail-mass is the "
                 "descriptive mechanism readout only.")
    lines.append("- CI = cluster bootstrap by SOURCE GAME (10k resamples), paired per position.")
    lines.append("- threshold sweep needs a per-bin softmax dump (score_all --emit-bin-probs); "
                 "tidy table carries only the frozen bin-16 (-0.5) tail.")
    return "\n".join(lines) + "\n"


# ── CLI ───────────────────────────────────────────────────────────────────────


def _load_arm(paths: List[str]) -> List[Dict[str, List[dict]]]:
    return [load_scores_table(p) for p in paths]


def main() -> None:
    ap = argparse.ArgumentParser(description="D-F HEADSWAP pre-registered metrics")
    for arm in ("A", "B", "C", "D"):
        ap.add_argument(f"--{arm.lower()}", nargs="+", required=True,
                        help=f"arm {arm} scores_{arm}_seed*.jsonl (one per seed)")
    ap.add_argument("--holdout", nargs="*", default=None,
                    help="arm=path holdout scores tables for ECE/sign (optional)")
    ap.add_argument("--out", default="reports/headswap/VERDICT.md")
    ap.add_argument("--json-out", default=None, help="also dump the raw metrics json")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    args = ap.parse_args()

    arm_tables = {
        "A": _load_arm(args.a), "B": _load_arm(args.b),
        "C": _load_arm(args.c), "D": _load_arm(args.d),
    }
    holdout_tables = None
    if args.holdout:
        holdout_tables = {}
        for spec in args.holdout:
            arm, p = spec.split("=", 1)
            holdout_tables[arm] = load_scores_table(p)

    m = run_metrics(arm_tables, holdout_tables, seed=args.seed)
    md = render_verdict_md(m)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(md)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(m, indent=2, default=str))
    print(md)


if __name__ == "__main__":
    main()
