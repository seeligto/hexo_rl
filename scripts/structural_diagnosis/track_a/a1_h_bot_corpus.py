"""§S181 Track A — A1 H-BOT: bot corpus position-level bias.

Classifies every position in `data/bot_corpus_s178_sealbot_vs_v6.npz`
using the shared position classifier. Tabulates per-class fractions,
per-class outcome (z) distribution, and the asymmetry score against the
T3 anchor V_spread = +0.617.

Outputs:
  audit/structural/track_a/A1_h_bot_corpus_position_bias.json (sidecar)

Run:
  .venv/bin/python scripts/structural_diagnosis/track_a/a1_h_bot_corpus.py

Pre-registered verdict (L13 LITERAL):
  STRONG-CONFIRM  asymmetry > +0.30 AND colony frac >= 30%
  WEAK            asymmetry > +0.30 OR colony frac >= 30% (exclusive)
  RULED-OUT       asymmetry in [-0.10, +0.10] AND colony frac < 20%
  INCONCLUSIVE    otherwise

Corpus does NOT carry a per-position winner id (only outcome z from
current-player perspective). "Fraction by winner" therefore collapses
to "fraction by outcome sign" — reported in `by_class_outcome_sign`.
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

from scripts.structural_diagnosis.track_a.position_classifier import (
    classify_batch, MIN_STONES, MAX_MEAN_HEX_DIST, OPEN_RUN_LEN,
)

CORPUS = REPO / "data" / "bot_corpus_s178_sealbot_vs_v6.npz"
ANCHOR_VSPREAD_T3 = 0.617

# Z buckets per brief: {-1, [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1], 1}
# Bot corpus uses z in {-1, 0, +1} predominantly (game outcomes only).
Z_BUCKETS = [
    ("z == -1",       lambda z: z <= -0.9999),
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
    weights = d["weights"]
    n = len(states)
    print(f"  n_positions={n}, sha256={sha[:16]}…")
    print(f"  outcomes unique: {np.unique(outcomes)}, mean={outcomes.mean():.4f}, sd={outcomes.std():.4f}")

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

    # Asymmetry score: mean_z(colony) - mean_z(extension). Compare to T3
    # anchor V_spread = +0.617 (the value head's spread between bank
    # classes). If the corpus value-target asymmetry approaches +0.617,
    # the training data itself imprints the colony-favouring direction.
    m_col = by_class["colony"].get("mean_z", 0.0)
    m_ext = by_class["extension"].get("mean_z", 0.0)
    asymmetry = round(m_col - m_ext, 4)
    colony_frac = by_class["colony"].get("frac", 0.0)

    # Pre-registered verdict — LITERAL application (L13 guard).
    if asymmetry > 0.30 and colony_frac >= 0.30:
        verdict = "STRONG-CONFIRM"
    elif (asymmetry > 0.30) ^ (colony_frac >= 0.30):
        verdict = "WEAK"
    elif -0.10 <= asymmetry <= 0.10 and colony_frac < 0.20:
        verdict = "RULED-OUT"
    else:
        verdict = "INCONCLUSIVE"

    # Outcome-sign distribution per class (proxy for "by winner" since
    # the corpus has no per-position winner_id field; only z from
    # current-player perspective).
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

    out = REPO / "audit" / "structural" / "track_a" / "A1_h_bot_corpus_position_bias.json"
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
