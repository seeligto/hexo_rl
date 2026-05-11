"""Build mixed bootstrap+adversarial corpus for §171 A4 P2-reopen fine-tune.

Concatenates `data/bootstrap_corpus_v8_canvas_realness.npz` (base, 347,142 pos,
plane-8 polarity inside=1 / canvas_realness) with `data/adversarial_corpus_v8.npz`
(12,781 pos, same canvas_realness polarity by construction — see
`reports/gpool_bias/adversarial_manifest.md` §171 P0.1 transposition correction).

Rescales adversarial per-position weights so the WeightedRandomSampler at training
time draws adversarial positions with probability TARGET_FRAC (default 0.05). Bootstrap
weights stay untouched. Source-distribution invariants within each corpus preserved
because the rescale is a uniform factor applied to all adversarial weights.

Output: `data/bootstrap_v8cr_plus_adv5.npz` + sidecar via `corpus_io.save_corpus`
(stamps encoding_name=v8, sha256, n_positions, source_manifest pointer, schema_version,
plus an `extra` block with the mix recipe + per-input shas + the actual mix probability).
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

from hexo_rl.bootstrap.corpus_io import compute_npz_sha256, save_corpus


BOOTSTRAP_PATH = pathlib.Path("data/bootstrap_corpus_v8_canvas_realness.npz")
ADVERSARIAL_PATH = pathlib.Path("data/adversarial_corpus_v8.npz")
DEFAULT_OUT = pathlib.Path("data/bootstrap_v8cr_plus_adv5.npz")
DEFAULT_TARGET_FRAC = 0.05


def _load_required(path: pathlib.Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"missing corpus: {path}")
    with np.load(path, allow_pickle=True) as f:
        out = {k: f[k] for k in ("states", "policies", "outcomes", "weights")}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=pathlib.Path, default=BOOTSTRAP_PATH)
    parser.add_argument("--adversarial", type=pathlib.Path, default=ADVERSARIAL_PATH)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--target-frac", type=float, default=DEFAULT_TARGET_FRAC,
                        help="Target P(adversarial draw) under WeightedRandomSampler. "
                             "Default 0.05 (§171 A4 P2-reopen C).")
    args = parser.parse_args()

    if not (0.0 < args.target_frac < 1.0):
        raise ValueError(f"--target-frac must be in (0,1), got {args.target_frac}")

    sha_b = compute_npz_sha256(args.bootstrap)
    sha_a = compute_npz_sha256(args.adversarial)
    print(f"[bootstrap]   {args.bootstrap} sha={sha_b}")
    print(f"[adversarial] {args.adversarial} sha={sha_a}")

    b = _load_required(args.bootstrap)
    a = _load_required(args.adversarial)

    for k in ("states", "policies", "outcomes", "weights"):
        if b[k].shape[1:] != a[k].shape[1:]:
            raise ValueError(
                f"shape mismatch on '{k}': bootstrap {b[k].shape} vs adversarial {a[k].shape}"
            )
        if b[k].dtype != a[k].dtype:
            raise ValueError(
                f"dtype mismatch on '{k}': bootstrap {b[k].dtype} vs adversarial {a[k].dtype}"
            )

    sum_b = float(b["weights"].sum())
    sum_a = float(a["weights"].sum())
    if sum_a <= 0:
        raise ValueError(f"adversarial weights sum non-positive: {sum_a}")
    scale_a = (args.target_frac / (1.0 - args.target_frac)) * (sum_b / sum_a)
    adv_weights_scaled = a["weights"].astype(np.float32) * np.float32(scale_a)

    mixed_states = np.concatenate([b["states"], a["states"]], axis=0)
    mixed_policies = np.concatenate([b["policies"], a["policies"]], axis=0)
    mixed_outcomes = np.concatenate([b["outcomes"], a["outcomes"]], axis=0)
    mixed_weights = np.concatenate(
        [b["weights"].astype(np.float32), adv_weights_scaled], axis=0
    )

    sum_b_check = float(mixed_weights[: len(b["weights"])].sum())
    sum_a_check = float(mixed_weights[len(b["weights"]):].sum())
    actual_frac = sum_a_check / (sum_b_check + sum_a_check)
    print(
        f"[mix] target_frac={args.target_frac:.4f} actual_frac={actual_frac:.6f} "
        f"scale_adv={scale_a:.6f} sum_b={sum_b_check:.2f} sum_a={sum_a_check:.2f}"
    )

    extra = {
        "mix_recipe": "concat + adversarial-weight-rescale (no physical replication)",
        "target_adversarial_frac": args.target_frac,
        "actual_adversarial_frac": actual_frac,
        "adversarial_weight_scale": scale_a,
        "bootstrap_sha256": sha_b,
        "adversarial_sha256": sha_a,
        "bootstrap_path": str(args.bootstrap),
        "adversarial_path": str(args.adversarial),
        "bootstrap_n_positions": int(len(b["weights"])),
        "adversarial_n_positions": int(len(a["weights"])),
        "sprint": "§171 A4 P2-reopen C",
    }

    save_corpus(
        args.out,
        arrays={
            "states": mixed_states,
            "policies": mixed_policies,
            "outcomes": mixed_outcomes,
            "weights": mixed_weights,
        },
        encoding_name="v8",
        source_manifest="reports/gpool_bias/adversarial_manifest.md",
        extra=extra,
    )

    out_path = args.out if args.out.exists() else args.out.with_suffix(".npz")
    sha_out = compute_npz_sha256(out_path)
    print(f"[out] {out_path} sha={sha_out} n={len(mixed_outcomes)}")


if __name__ == "__main__":
    main()
