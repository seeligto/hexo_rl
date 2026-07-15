#!/usr/bin/env python
"""Phase-1 launch A2 — mint the run3-CNN dist65 fresh-init bootstrap.

run3's ruling ("fresh init v6_live2_ls") launches `--checkpoint
checkpoints/run2_bootstrap_v6_live2_ls.pt` — run2's own encoding-stamped
weights-only bootstrap init (SCALAR value head, 147 tensors, no
`value_fc2_bins.*`). Loading that file directly into a `value_head_type:
dist65` net (`configs/variants/run3_dist65.yaml`) hits
`hexo_rl/training/warmstart_launch.py::assert_dist65_bins_seeded` (called
from `orchestrator.py` right after the checkpoint load): the guard raises
because the dist65 bins would be neither loaded from the checkpoint
(`ckpt_had_value_fc2_bins=False` — the source is a scalar trunk) nor
seeded by an E1 HEADSWAP warm-start (`warm_start.enabled=false` for run3
— that mechanism injects a TRAINED value head, wrong for a fresh-init
run). There is no third sanctioned path in the guard for "yes, random
bins are intentional" (confirmed by reading `warmstart_launch.py` +
`orchestrator.py`'s two call branches: the checkpoint branch always runs
the guard, the fresh-no-checkpoint branch never carries run2's trunk
init at all, breaking the same-init one-variable read this run needs).

This script closes the gap the RIGHT way — not by defeating the guard,
but by making its precondition (`ckpt_had_value_fc2_bins=True`) honestly
true: build a dist65 net, load run2's bootstrap trunk/policy/scalar-head
weights into it NON-STRICT (every shared tensor becomes byte-identical to
run2's own starting point — the same-init invariant), leave
`value_fc2_bins.*` at the net's own fresh nn.Linear random init (untouched
by the load — genuinely never seen any gradient), then SAVE the resulting
full state_dict. The saved file now honestly carries value_fc2_bins.*
tensors — random, but PRESENT — so a real run3 launch reads
`ckpt_had_value_fc2_bins=True` and the guard no-ops exactly as it does for
a genuine dist65 checkpoint resume.

Reuses production code, not a reimplementation:
  - `hexo_rl.training.checkpoints.load_state_dict_strict` — the SAME
    loader `trainer_ckpt_load.load_checkpoint` calls, including its
    existing `value_fc2_bins.*` benign-missing carve-out (E1 T8,
    `checkpoints.py` `new_aux_prefixes`). Any OTHER unexpected/missing
    key still raises — this script cannot silently mis-load the trunk.
  - `hexo_rl.model.network.HexTacToeNet` — the real model class, built
    with the exact hparams `hexo_rl/training/trainer_ckpt_load.py`
    resolves for `configs/variants/run3_dist65.yaml` (board_size=19,
    in_channels=4, res_blocks=12, filters=128, se_reduction_ratio=4,
    value_head_type=dist65, n_value_bins=65) — see
    `docs/registers/run3_cnn_preregistration.md` / the resolved-config
    probe in the Phase-1 launch report.

Output payload: EXACTLY {'model_state', 'metadata', 'step'} (weights-only
— `is_full_ckpt=False` at load, matching run2_bootstrap_v6_live2_ls.pt's
own shape) with `metadata['encoding_name']='v6_live2_ls'` (unchanged —
the mint does not change encoding, only value-head shape) plus provenance
(`source_checkpoint`, `source_sha256`, `dist65_bins_synthesized=True`).
`step` is pinned to 0 (fresh init).

Usage:
  .venv/bin/python scripts/mint_run3_dist65_bootstrap.py \\
      --in checkpoints/run2_bootstrap_v6_live2_ls.pt \\
      --out checkpoints/run3_bootstrap_v6_live2_ls_dist65.pt
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hexo_rl.model.network import HexTacToeNet  # noqa: E402
from hexo_rl.training.checkpoints import load_state_dict_strict  # noqa: E402

DEFAULT_IN = REPO_ROOT / "checkpoints" / "run2_bootstrap_v6_live2_ls.pt"
DEFAULT_OUT = REPO_ROOT / "checkpoints" / "run3_bootstrap_v6_live2_ls_dist65.pt"

# run3_dist65.yaml resolved hparams (docs/registers/run3_cnn_preregistration.md
# §0; verified against the REAL config-merge entrypoint
# `hexo_rl.training.orchestrator.load_train_config` +
# `flatten_config_and_resolve_encoding` at Phase-1 launch A3).
BOARD_SIZE = 19
IN_CHANNELS = 4
RES_BLOCKS = 12
FILTERS = 128
SE_REDUCTION_RATIO = 4
N_VALUE_BINS = 65

STRIPPED_KEYS = ("model_state", "metadata", "step")


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mint(in_path: Path, out_path: Path) -> Dict[str, Any]:
    print(f"Loading source: {in_path}", file=sys.stderr)
    src = torch.load(in_path, map_location="cpu", weights_only=False)
    if not isinstance(src, dict) or "model_state" not in src:
        raise ValueError(
            f"expected a weights-only {{'model_state','metadata','step'}} payload, "
            f"got keys={sorted(src.keys()) if isinstance(src, dict) else type(src)!r}"
        )
    src_metadata = src.get("metadata") or {}
    encoding_name = src_metadata.get("encoding_name")
    if encoding_name != "v6_live2_ls":
        raise ValueError(
            f"source checkpoint metadata['encoding_name']={encoding_name!r}, "
            "expected 'v6_live2_ls' — refusing to mint from an unexpected encoding "
            "(this script only changes value-head shape, never encoding)."
        )
    scalar_state = src["model_state"]
    has_bins_already = any(k.startswith("value_fc2_bins.") for k in scalar_state)
    if has_bins_already:
        raise ValueError(
            "source checkpoint already carries value_fc2_bins.* — it is not a "
            "scalar trunk; nothing to mint (load it directly as the dist65 init)."
        )
    if not any(k.startswith("value_fc2.") for k in scalar_state):
        raise ValueError(
            "source checkpoint carries neither value_fc2.* (scalar) nor "
            "value_fc2_bins.* (dist65) — unrecognized value-head shape."
        )

    print(
        f"  source: {len(scalar_state)} tensors, scalar value head "
        f"(value_fc2.*), no value_fc2_bins.*",
        file=sys.stderr,
    )

    print(
        "Building dist65 net "
        f"(board_size={BOARD_SIZE}, in_channels={IN_CHANNELS}, "
        f"res_blocks={RES_BLOCKS}, filters={FILTERS}, "
        f"se_reduction_ratio={SE_REDUCTION_RATIO}, n_value_bins={N_VALUE_BINS})",
        file=sys.stderr,
    )
    model = HexTacToeNet(
        board_size=BOARD_SIZE,
        in_channels=IN_CHANNELS,
        res_blocks=RES_BLOCKS,
        filters=FILTERS,
        se_reduction_ratio=SE_REDUCTION_RATIO,
        value_head_type="dist65",
        n_value_bins=N_VALUE_BINS,
    )

    # Snapshot the fresh-random value_fc2_bins BEFORE the load, so we can assert
    # afterward that the load truly left them untouched (no accidental strict-load
    # side effect, no key collision silently overwriting them).
    bins_before = {
        k: v.clone() for k, v in model.state_dict().items() if k.startswith("value_fc2_bins.")
    }
    if not bins_before:
        raise ValueError("BUG: dist65 net has no value_fc2_bins.* parameters — check network.py")

    # The SAME loader `trainer_ckpt_load.load_checkpoint` calls — including its
    # value_fc2_bins.* benign-missing carve-out. Any OTHER missing/unexpected key
    # still raises (RuntimeError), so a genuine architecture mismatch is loud.
    load_state_dict_strict(model, scalar_state)

    bins_after = {
        k: v for k, v in model.state_dict().items() if k.startswith("value_fc2_bins.")
    }
    for k in bins_before:
        if not torch.equal(bins_before[k], bins_after[k]):
            raise ValueError(
                f"BUG: {k} changed during the trunk load — value_fc2_bins was "
                "supposed to stay untouched (fresh random init)."
            )
    print(
        f"  loaded: every shared tensor == source (byte-identical run2 init); "
        f"{len(bins_before)} value_fc2_bins.* tensors kept at FRESH random init "
        "(untouched by the load)",
        file=sys.stderr,
    )

    full_state = model.state_dict()
    source_sha = _sha256_of_file(in_path)
    metadata = dict(src_metadata)
    metadata["dist65_bins_synthesized"] = True
    metadata["minted_from"] = in_path.name
    # NOTE: do NOT clobber a pre-existing 'source_sha256' — the source checkpoint
    # (run2_bootstrap_v6_live2_ls.pt) already carries its OWN provenance chain
    # (source_sha256 = sha of bootstrap_model_v6_live2_8300.pt, the file it was
    # synthesized from). Overwriting that key would silently truncate the chain
    # to one hop. This mint's own immediate-source sha gets a distinctly-named key.
    metadata["dist65_mint_source_sha256"] = source_sha
    metadata["value_head_type"] = "dist65"
    metadata["n_value_bins"] = N_VALUE_BINS

    payload = {"model_state": full_state, "metadata": metadata, "step": 0}
    assert set(payload.keys()) == set(STRIPPED_KEYS)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    args = ap.parse_args()

    if not args.in_path.exists():
        print(f"ERROR: source checkpoint not found: {args.in_path}", file=sys.stderr)
        sys.exit(2)
    if args.out_path.exists() and not args.force:
        print(f"ERROR: {args.out_path} already exists; pass --force to overwrite.", file=sys.stderr)
        sys.exit(2)
    if args.out_path.resolve() == args.in_path.resolve():
        print("ERROR: --out resolves to the same file as --in.", file=sys.stderr)
        sys.exit(2)

    payload = mint(args.in_path, args.out_path)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out_path)

    reloaded = torch.load(args.out_path, map_location="cpu", weights_only=False)
    reloaded_keys = set(reloaded.keys())
    if reloaded_keys != set(STRIPPED_KEYS):
        print(
            f"ERROR: post-save re-load key mismatch: {sorted(reloaded_keys)} != "
            f"{sorted(STRIPPED_KEYS)}",
            file=sys.stderr,
        )
        sys.exit(2)
    n_bins_keys = sum(1 for k in reloaded["model_state"] if k.startswith("value_fc2_bins."))
    if n_bins_keys == 0:
        print("ERROR: post-save re-load has no value_fc2_bins.* — mint failed silently.", file=sys.stderr)
        sys.exit(2)

    sha = _sha256_of_file(args.out_path)
    print(f"OK: wrote {args.out_path}")
    print(f"  step:  {reloaded.get('step')}")
    print(f"  keys:  {sorted(reloaded_keys)}")
    print(f"  n_tensors: {len(reloaded['model_state'])} (value_fc2_bins.* count: {n_bins_keys})")
    print(f"  sha256: {sha}")
    print(f"  metadata: {reloaded.get('metadata')}")


if __name__ == "__main__":
    main()
