"""§S181 Track A — A5 H-PRETRAIN: pretrain corpus position-level z.

§S181 T1 measured the pretrain corpus's POPULATION-level class fraction
(91.3% extension via threat-fraction). It did NOT measure per-position
value-target z bucketed by structural class. This subtask fills that
gap.

If the pretrain corpus's colony positions carry strongly colony-favouring
z, then pretrain installs the colony attractor's direction (and L47's
"loop dominates architecture" might overstate the loop's role — the
attractor would be installed at pretrain time, then maintained by the
loop). If pretrain z is symmetric, the bias must be added downstream.

Pre-registered verdict (LITERAL L13):
  PRETRAIN-COLONY-BIASED  asymmetry > +0.20 AND colony frac >= 5%
                          (consequential count)
  NEUTRAL                 asymmetry in [-0.10, +0.10]
  EXTENSION-BIASED        asymmetry < -0.20
  INCONCLUSIVE            otherwise

Outputs:
  audit/structural/track_a/A5_h_pretrain_position_z.json
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts.diagnosis.track_a.position_classifier import (
    classify_batch, MIN_STONES, MAX_MEAN_HEX_DIST, OPEN_RUN_LEN,
)

CORPUS = REPO / "data" / "bootstrap_corpus_v6.npz"
ANCHOR_VSPREAD_T3 = 0.617

Z_BUCKETS = [
    ("z == -1",        lambda z: z <= -0.9999),
    ("-1 < z <= -0.5", lambda z: (z > -0.9999) & (z <= -0.5)),
    ("-0.5 < z < 0",   lambda z: (z > -0.5)    & (z < 0.0)),
    ("z == 0",         lambda z: z == 0.0),
    ("0 < z < +0.5",   lambda z: (z > 0.0)     & (z < 0.5)),
    ("+0.5 <= z < +1", lambda z: (z >= 0.5)    & (z < 0.9999)),
    ("z == +1",        lambda z: z >= 0.9999),
]


def bucket_z(values: np.ndarray) -> dict:
    return {name: int(np.sum(fn(values))) for name, fn in Z_BUCKETS}


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    print(f"loading {CORPUS} ...")
    sha = file_sha256(CORPUS)
    d = np.load(CORPUS)
    states = d["states"]
    outcomes = d["outcomes"].astype(np.float64)
    n = len(states)
    print(f"  n_positions={n}, sha256={sha[:16]}…")
    print(f"  outcomes unique: {np.unique(outcomes)[:10]}…  mean={outcomes.mean():.4f}")

    print(f"classifying {n} positions …")
    t1 = time.time()
    cls = classify_batch(states)
    print(f"  classified in {time.time()-t1:.1f}s")

    distribution = Counter(cls)

    by_class: dict = {}
    for c in ("colony", "extension", "neither"):
        mask = (cls == c)
        n_cls = int(mask.sum())
        if n_cls == 0:
            by_class[c] = dict(n=0, frac=0.0)
            continue
        z = outcomes[mask]
        by_class[c] = dict(
            n=n_cls,
            frac=round(n_cls / n, 4),
            mean_z=round(float(z.mean()), 4),
            median_z=round(float(np.median(z)), 4),
            sd_z=round(float(z.std(ddof=0)), 4),
            z_buckets=bucket_z(z),
        )

    m_col = by_class["colony"].get("mean_z", 0.0)
    m_ext = by_class["extension"].get("mean_z", 0.0)
    asymmetry = round(m_col - m_ext, 4)
    colony_frac = by_class["colony"].get("frac", 0.0)

    # Pre-registered verdict (LITERAL L13)
    if asymmetry > 0.20 and colony_frac >= 0.05:
        verdict = "PRETRAIN-COLONY-BIASED"
    elif asymmetry < -0.20:
        verdict = "EXTENSION-BIASED"
    elif -0.10 <= asymmetry <= 0.10:
        verdict = "NEUTRAL"
    else:
        verdict = "INCONCLUSIVE"

    by_class_outcome_sign: dict = {}
    for c in ("colony", "extension", "neither"):
        mask = (cls == c)
        n_cls = int(mask.sum())
        if n_cls == 0:
            continue
        z = outcomes[mask]
        pos = int((z > 0.0).sum())
        neg = int((z < 0.0).sum())
        zero = int((z == 0.0).sum())
        by_class_outcome_sign[c] = dict(
            curr_player_wins=pos, opp_wins=neg, draws=zero,
            curr_wins_frac=round(pos / n_cls, 4),
            opp_wins_frac=round(neg / n_cls, 4),
            draws_frac=round(zero / n_cls, 4),
        )

    result = dict(
        meta=dict(
            corpus=str(CORPUS.name),
            sha256=sha,
            n_positions=n,
            anchor_vspread_t3=ANCHOR_VSPREAD_T3,
            classifier_thresholds=dict(
                min_stones=MIN_STONES,
                max_mean_hex_dist=MAX_MEAN_HEX_DIST,
                open_run_len=OPEN_RUN_LEN,
            ),
            wall_s=round(time.time() - t0, 1),
        ),
        distribution={k: int(v) for k, v in distribution.items()},
        by_class=by_class,
        by_class_outcome_sign=by_class_outcome_sign,
        asymmetry_colony_minus_extension=asymmetry,
        anchor_vspread_t3=ANCHOR_VSPREAD_T3,
        verdict=verdict,
    )

    out = REPO / "audit" / "structural" / "track_a" / "A5_h_pretrain_position_z.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}  ({result['meta']['wall_s']}s)")
    print(json.dumps({
        "distribution": result["distribution"],
        "by_class": by_class,
        "asymmetry": asymmetry,
        "verdict": verdict,
    }, indent=2))


if __name__ == "__main__":
    main()
