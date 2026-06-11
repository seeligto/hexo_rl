"""§S181-AUDIT Wave 1 — C-LITE-1: v7full anchor V_spread (dual bank).

Pre-§175 era reference point. If v7full's V_spread (alt bank) ≥ +0.15
AND v7full's documented 17.4% SealBot WR proves it was sustainable, then
encoding regression v7full → v6w25 → v6 is the load-bearing change in
the §150 → §175 → §S178 trajectory.

Pre-registered verdicts (LITERAL L13):

  C-LITE-1-A  v7full alt-bank V_spread ≥ +0.15
              → encoding regression candidate confirmed
  C-LITE-1-B  v7full alt-bank V_spread in [+0.05, +0.15]
              → comparable to v6 baseline; encoding not the answer
  C-LITE-1-C  v7full alt-bank V_spread < +0.05
              → v7full's 17.4% wasn't anchor-driven; deeper issue

Encoding compatibility. v7full and v6 share board_size=19, n_planes=8,
plane_layout, kept_plane_indices, has_pass_slot — verified against
engine/src/encoding/registry.toml. Both banks' (8, 19, 19) state arrays
+ T3 Board sequences feed v7full cleanly.

Inputs:
  - checkpoints/bootstrap_model_v7full.pt
      SHA 568d8a332a41b0c6a0cfc0dd8d90ddd8153a046cd6245f517f351e77d61e8e98
  - tests/fixtures/value_spread_bank.json        (T3, SHA 9342047...)
  - tests/fixtures/value_spread_bank_alt.json    (alt, SHA a68b810f...)

Outputs:
  - audit/structural/track_c/C_LITE_1_v7full_reference.md
  - audit/structural/track_c/C_LITE_1_v7full_reference.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from hexo_rl.monitoring.value_spread_canary import (
    ALT_BANK_SHA256,
    ALT_SOFT_ABORT_THRESHOLD,
    ALT_WARN_THRESHOLD,
    BANK_SHA256,
    SOFT_ABORT_THRESHOLD,
    WARN_THRESHOLD,
    compute_value_spread_dual,
    load_alt_bank,
    load_bank,
)
from hexo_rl.viewer.model_loader import load_model

ANCHOR_V7FULL = REPO / "checkpoints" / "bootstrap_model_v7full.pt"
ANCHOR_V6 = REPO / "checkpoints" / "bootstrap_model_v6.pt"
OUT_DIR = REPO / "audit" / "structural" / "track_c"

# v6 baseline (already audited) — kept here for direct comparison in the
# JSON sidecar; values from audit/structural/05_fu1_value_spread_ladder.md
# (T3) and track_a/A3_h_bank_confound.json (alt).
V6_T3_SPREAD = 0.6173
V6_ALT_SPREAD = 0.2119


def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not ANCHOR_V7FULL.exists():
        raise SystemExit(f"v7full anchor missing: {ANCHOR_V7FULL}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    device = torch.device("cpu")

    print(f"loading {ANCHOR_V7FULL.name} ...")
    v7full_sha = sha256_of_file(ANCHOR_V7FULL)
    print(f"  SHA={v7full_sha}")

    net, meta, _dev = load_model(ANCHOR_V7FULL, device=device)
    encoding_name = (meta or {}).get("encoding_name", "?")
    print(f"  encoding_name (from metadata): {encoding_name}")

    # Force-load + verify both banks before forward (canary's load_bank /
    # load_alt_bank both SHA-pin).
    t3_bank = load_bank()
    alt_bank = load_alt_bank()
    print(f"  T3 bank SHA={t3_bank.sha[:16]}…  alt bank SHA={alt_bank.sha[:16]}…")

    print("forwarding dual canary on v7full ...")
    result = compute_value_spread_dual(
        net, device=device, t3_bank=t3_bank, alt_bank=alt_bank,
    )
    print(f"  T3  V_spread = {result.t3_spread:+.4f}  "
          f"(colony={result.t3_components['mean_colony']:+.4f}, "
          f"ext={result.t3_components['mean_ext']:+.4f})")
    print(f"  alt V_spread = {result.alt_spread:+.4f}  "
          f"(colony={result.alt_components['mean_colony']:+.4f}, "
          f"ext={result.alt_components['mean_ext']:+.4f})")
    print(f"  both_pass = {result.both_pass}")

    # ── Pre-registered verdict (alt-bank gate, LITERAL L13) ──────────────
    alt = result.alt_spread
    if alt >= 0.15:
        verdict = "C-LITE-1-A"
        verdict_desc = (
            "v7full alt-bank V_spread ≥ +0.15 — encoding regression candidate "
            "CONFIRMED; consider v7full as the real-run anchor"
        )
    elif alt >= 0.05:
        verdict = "C-LITE-1-B"
        verdict_desc = (
            "v7full alt-bank V_spread in [+0.05, +0.15] — comparable to v6 "
            "baseline; encoding NOT the answer"
        )
    else:
        verdict = "C-LITE-1-C"
        verdict_desc = (
            "v7full alt-bank V_spread < +0.05 — v7full's 17.4% was NOT "
            "anchor-driven; deeper issue elsewhere"
        )
    print(f"\nVERDICT: {verdict}")
    print(f"  {verdict_desc}")

    sidecar = dict(
        meta=dict(
            wave="§S181-AUDIT Wave 1 — Track C-LITE-1",
            anchor_path=str(ANCHOR_V7FULL.relative_to(REPO)),
            anchor_sha256=v7full_sha,
            encoding_name=encoding_name,
            t3_fixture_sha=BANK_SHA256,
            alt_fixture_sha=ALT_BANK_SHA256,
            wall_s=round(time.time() - t0, 2),
        ),
        thresholds=dict(
            t3_warn=WARN_THRESHOLD,
            t3_soft_abort=SOFT_ABORT_THRESHOLD,
            alt_warn=ALT_WARN_THRESHOLD,
            alt_soft_abort=ALT_SOFT_ABORT_THRESHOLD,
        ),
        v6_baseline=dict(
            t3_spread=V6_T3_SPREAD, alt_spread=V6_ALT_SPREAD,
        ),
        v7full=dict(
            t3_spread=result.t3_spread,
            t3_components=result.t3_components,
            alt_spread=result.alt_spread,
            alt_components=result.alt_components,
            both_pass=result.both_pass,
        ),
        verdict=verdict,
        verdict_description=verdict_desc,
    )

    json_out = OUT_DIR / "C_LITE_1_v7full_reference.json"
    json_out.write_text(json.dumps(sidecar, indent=2))
    print(f"\nwrote {json_out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
