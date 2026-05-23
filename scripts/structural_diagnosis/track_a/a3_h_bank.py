"""§S181 Track A — A3 H-BANK: alt bank cross-validation.

Builds an ALTERNATE 40-position value-spread bank from real bot-corpus
positions (rather than the T3 builder's synthetic spiral+run+filler
constructions). Forwards each FU-1.5 ladder checkpoint on both banks,
correlates V_spread(T3) against V_spread(alt). Tests whether V_spread
is measuring a generalizable property of the value head or a T3-bank-
specific artifact.

Pre-registered verdict (LITERAL L13):
  ROBUST        Pearson correlation r >= 0.9
  CONFOUND      r < 0.7
  INCONCLUSIVE  otherwise

Inputs:
  - `data/bot_corpus_s178_sealbot_vs_v6.npz` (alt bank source)
  - `checkpoints/bootstrap_model_v6.pt` (step 0 anchor)
  - `archive/s181_fu1_5/ckpts/checkpoint_{00002000..00020000}.pt`
    (FU-1.5 ladder)
  - V_spread(T3) values from `audit/structural/06_fu1_5_finer_ladder.md`
    §3 — hard-coded (read-only audit reproduction)

Outputs:
  - `tests/fixtures/value_spread_bank_alt.json` (alt bank fixture,
    SHA-pinned in the JSON `meta.sha256` field)
  - `audit/structural/track_a/A3_h_bank_confound.json` (sidecar)
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts.structural_diagnosis.track_a.position_classifier import (
    classify_state, MIN_STONES, MAX_MEAN_HEX_DIST, OPEN_RUN_LEN,
)
from hexo_rl.viewer.model_loader import load_model

CORPUS = REPO / "data" / "bot_corpus_s178_sealbot_vs_v6.npz"
ANCHOR = REPO / "checkpoints" / "bootstrap_model_v6.pt"
LADDER_DIR = REPO / "archive" / "s181_fu1_5" / "ckpts"
ALT_FIXTURE = REPO / "tests" / "fixtures" / "value_spread_bank_alt.json"

# V_spread(T3) from audit/structural/06_fu1_5_finer_ladder.md §3.
# (step, V_spread) pairs — anchor + 10 ladder points.
T3_LADDER: list[tuple[int, float]] = [
    (0,     +0.6173),
    (2000,  +0.1752),
    (4000,  -0.1179),
    (6000,  +0.0553),
    (8000,  +0.2172),
    (10000, +0.3901),
    (12000, +0.1669),
    (14000, +0.5226),
    (16000, +0.2209),
    (18000, +0.2083),
    (20000, +0.1075),
]

# Sampling parameters
N_PER_CLASS = 20
SEED = 20260523
MID_GAME_MIN = 12
MID_GAME_MAX = 36


def sample_alt_bank() -> dict:
    """Draw 20 colony + 20 extension positions from bot_corpus (mid-game)."""
    rng = np.random.default_rng(SEED)
    d = np.load(CORPUS)
    states = d["states"]
    n = len(states)

    # mid-game filter
    cp = (states[:, 0] > 0.5).reshape(n, -1).sum(axis=1)
    op = (states[:, 4] > 0.5).reshape(n, -1).sum(axis=1)
    n_stones = cp + op
    mid_mask = (n_stones >= MID_GAME_MIN) & (n_stones <= MID_GAME_MAX)
    candidate_idx = np.flatnonzero(mid_mask)
    print(f"  mid-game candidates: {len(candidate_idx)}/{n} "
          f"(stones in [{MID_GAME_MIN}, {MID_GAME_MAX}])")

    # classify candidates
    cls = np.empty(len(candidate_idx), dtype=object)
    for i, idx in enumerate(candidate_idx):
        cls[i] = classify_state(states[idx])

    col_pool = candidate_idx[cls == "colony"]
    ext_pool = candidate_idx[cls == "extension"]
    print(f"  pool sizes: colony={len(col_pool)}, extension={len(ext_pool)}")
    if len(col_pool) < N_PER_CLASS or len(ext_pool) < N_PER_CLASS:
        raise SystemExit(f"insufficient pool: need {N_PER_CLASS} per class")

    col_sel = rng.choice(col_pool, N_PER_CLASS, replace=False).astype(int).tolist()
    ext_sel = rng.choice(ext_pool, N_PER_CLASS, replace=False).astype(int).tolist()

    positions = []
    for k, idx in enumerate(col_sel):
        positions.append(dict(
            name=f"alt_colony_{k:02d}_corpus{int(idx):05d}",
            pos_class="colony",
            corpus_idx=int(idx),
            n_stones=int(n_stones[idx]),
            state=states[idx].astype(np.float32).tolist(),  # (8, 19, 19) nested
        ))
    for k, idx in enumerate(ext_sel):
        positions.append(dict(
            name=f"alt_ext_{k:02d}_corpus{int(idx):05d}",
            pos_class="extension",
            corpus_idx=int(idx),
            n_stones=int(n_stones[idx]),
            state=states[idx].astype(np.float32).tolist(),
        ))
    return dict(positions=positions, seed=SEED)


def bank_sha(bank: dict) -> str:
    """Reproducible SHA — same scheme as the FU-1 T3 bank: name + class +
    flattened state bytes."""
    h = hashlib.sha256()
    for spec in bank["positions"]:
        h.update(spec["name"].encode())
        h.update(spec["pos_class"].encode())
        arr = np.asarray(spec["state"], dtype=np.float32)
        h.update(arr.tobytes())
    return h.hexdigest()


def load_bank_states(bank: dict) -> tuple[np.ndarray, np.ndarray]:
    states = np.stack([np.asarray(p["state"], dtype=np.float32)
                       for p in bank["positions"]])
    classes = np.array([p["pos_class"] for p in bank["positions"]])
    return states, classes


def forward_value(net: torch.nn.Module, states: np.ndarray,
                   device: torch.device) -> np.ndarray:
    """Run the model on (B, 8, 19, 19) state tensors → (B,) value scalars."""
    x = torch.from_numpy(states).to(device)
    was_training = net.training
    net.eval()
    try:
        with torch.no_grad():
            log_policy, value, v_logit = net(x)
    finally:
        if was_training:
            net.train()
    return value.squeeze(-1).cpu().numpy()


def vspread_on_bank(net, states, classes, device) -> tuple[float, float, float]:
    values = forward_value(net, states, device)
    col = values[classes == "colony"]
    ext = values[classes == "extension"]
    mc = float(col.mean())
    me = float(ext.mean())
    return mc, me, mc - me


def main():
    t0 = time.time()
    device = torch.device("cpu")

    # Build alt bank
    print(f"sampling alt bank from {CORPUS} ...")
    bank = sample_alt_bank()
    bank_sha_hex = bank_sha(bank)
    bank["meta"] = dict(
        corpus=str(CORPUS.name),
        n_per_class=N_PER_CLASS,
        seed=SEED,
        mid_game_range=[MID_GAME_MIN, MID_GAME_MAX],
        sha256=bank_sha_hex,
        classifier_thresholds=dict(
            min_stones=MIN_STONES,
            max_mean_hex_dist=MAX_MEAN_HEX_DIST,
            open_run_len=OPEN_RUN_LEN,
        ),
    )
    ALT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    ALT_FIXTURE.write_text(json.dumps(bank, indent=2))
    print(f"  wrote {ALT_FIXTURE}  SHA={bank_sha_hex[:16]}…")

    states, classes = load_bank_states(bank)
    print(f"  alt bank: {len(states)} positions ({(classes=='colony').sum()} colony, "
          f"{(classes=='extension').sum()} ext)")

    # Iterate checkpoints
    ladder: list[dict[str, Any]] = []
    # Anchor first
    print(f"\nforwarding anchor {ANCHOR.name} ...")
    net, meta, _ = load_model(ANCHOR, device=device)
    mc, me, sp = vspread_on_bank(net, states, classes, device)
    ladder.append(dict(step=0, ckpt=ANCHOR.name,
                       mean_colony=round(mc, 4),
                       mean_ext=round(me, 4),
                       vspread_alt=round(sp, 4)))
    print(f"  alt V_spread = {sp:+.4f}")

    # FU-1.5 ladder
    for step in range(2000, 20001, 2000):
        ckpt = LADDER_DIR / f"checkpoint_{step:08d}.pt"
        if not ckpt.exists():
            print(f"  MISSING {ckpt}")
            continue
        print(f"forwarding {ckpt.name} (step {step}) ...")
        net, meta, _ = load_model(ckpt, device=device)
        mc, me, sp = vspread_on_bank(net, states, classes, device)
        ladder.append(dict(step=step, ckpt=ckpt.name,
                           mean_colony=round(mc, 4),
                           mean_ext=round(me, 4),
                           vspread_alt=round(sp, 4)))
        print(f"  alt V_spread = {sp:+.4f}")

    # Correlate against T3 ladder
    t3_dict = dict(T3_LADDER)
    pairs = [(t3_dict[r["step"]], r["vspread_alt"]) for r in ladder if r["step"] in t3_dict]
    t3_arr = np.array([p[0] for p in pairs])
    alt_arr = np.array([p[1] for p in pairs])
    n_pairs = len(pairs)
    if n_pairs >= 2:
        r_pearson = float(np.corrcoef(t3_arr, alt_arr)[0, 1])
    else:
        r_pearson = float("nan")

    # Pre-registered verdict (LITERAL)
    if not np.isfinite(r_pearson):
        verdict = "INCONCLUSIVE"
    elif r_pearson >= 0.9:
        verdict = "ROBUST"
    elif r_pearson < 0.7:
        verdict = "CONFOUND"
    else:
        verdict = "INCONCLUSIVE"

    result = dict(
        meta=dict(
            alt_bank_sha256=bank_sha_hex,
            alt_fixture_path=str(ALT_FIXTURE.relative_to(REPO)),
            corpus_sha=hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            n_checkpoints=len(ladder),
            n_correlation_pairs=n_pairs,
            wall_s=round(time.time() - t0, 1),
        ),
        ladder=ladder,
        t3_ladder=[{"step": s, "vspread_t3": v} for s, v in T3_LADDER],
        pearson_r=round(r_pearson, 4) if np.isfinite(r_pearson) else None,
        verdict=verdict,
    )

    out = REPO / "audit" / "structural" / "track_a" / "A3_h_bank_confound.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}  ({result['meta']['wall_s']}s)")
    print(json.dumps({
        "n_pairs": n_pairs,
        "pearson_r": result["pearson_r"],
        "verdict": verdict,
        "alt_ladder": [(r["step"], r["vspread_alt"]) for r in ladder],
        "t3_ladder": T3_LADDER,
    }, indent=2))


if __name__ == "__main__":
    main()
