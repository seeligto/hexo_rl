#!/usr/bin/env python3
"""Anchor checkpoint verification utility.

Reports SHA + key count + params + head shapes + format detection
and applies the fresh-init Kaiming-uniform signature heuristic from
``reports/bootstrap_model_pt_provenance.md`` §3.

Heuristic (from §3 evidence table):
    value_fc2.weight |max| observed at 0.0625 = 1/sqrt(256) for fresh-init
    (Kaiming-uniform default, fan_in=256). Trained checkpoints drift far
    above this bound (e.g. 0.20167 in probe_ft.json, ratio 3.23x).
    Signature fires when |max|/bound in [0.8, 1.25] — within 20% of the
    theoretical bound, exactly the fresh-init fingerprint.

Exit codes:
    0  TRAINED               — clean load, signature did not fire
    1  FRESH_INIT_SUSPECT    — Kaiming-uniform signature fires
    2  load error / not-found / value-head not located (UNKNOWN)

Output: JSON sidecar at ``<ckpt>.verify.json`` + human-readable stdout.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

# Exit codes are part of the CLI contract; symbolic here for readability.
EXIT_TRAINED = 0
EXIT_FRESH = 1
EXIT_ERROR = 2

# Fresh-init signature window (per reports/bootstrap_model_pt_provenance.md §3):
# fresh-init value_fc2 |max| ~= 1.0 * sqrt(1/fan_in); trained drifts >>1.25x.
FRESH_RATIO_LO = 0.8
FRESH_RATIO_HI = 1.25


def _sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _detect_format(path: Path) -> str:
    """Detect pickle vs safetensors vs other via magic-byte probe."""
    with path.open("rb") as f:
        head = f.read(16)
    # safetensors: 8-byte little-endian header length, then JSON. JSON typically
    # starts with '{"' immediately at offset 8.
    if len(head) >= 10 and head[8:10] == b'{"':
        return "safetensors"
    # pickle protocol opcodes: \x80\x02, \x80\x03, \x80\x04, \x80\x05.
    if len(head) >= 2 and head[0:1] == b"\x80" and head[1] in (2, 3, 4, 5):
        return "pickle"
    # torch.save wraps with zipfile (PK\x03\x04) for PyTorch >= 1.6 default.
    if len(head) >= 4 and head[0:4] == b"PK\x03\x04":
        return "pickle"
    return "other"


def _unwrap_state_dict(obj: Any) -> Mapping[str, Any]:
    """Unwrap full-checkpoint containers down to a flat state_dict."""
    if isinstance(obj, Mapping) and "model_state" in obj and isinstance(obj["model_state"], Mapping):
        return obj["model_state"]
    if isinstance(obj, Mapping) and "state_dict" in obj and isinstance(obj["state_dict"], Mapping):
        return obj["state_dict"]
    return obj


def _trailing_out_dim(state: Mapping[str, Any], needles: tuple[str, ...]) -> int | None:
    """Return out-dim of the head's final weight tensor.

    Strategy: filter keys containing any needle and ending in '.weight'; pick the
    lexicographically last match (handles fc1/fc2 ordering; head names rarely
    use double-digit suffixes in this codebase). Return shape[0] (out features).
    """
    import torch  # local import — heavy

    candidates = []
    for k, v in state.items():
        if not k.endswith(".weight"):
            continue
        if not isinstance(v, torch.Tensor):
            continue
        kl = k.lower()
        if any(n in kl for n in needles):
            candidates.append(k)
    if not candidates:
        return None
    # Sort: prefer 'fc2' over 'fc1' over 'conv' (later layers have larger names
    # lexicographically with the canonical fc1/fc2/conv naming; fall back to
    # last alphabetical).
    candidates.sort()
    last = candidates[-1]
    return int(state[last].shape[0])


def _fresh_init_check(state: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Apply the value_fc2 Kaiming-uniform fingerprint test.

    Returns (signature_fires, details_dict). signature_fires=False with empty
    'value_fc2_weight_abs_max' indicates the value head could not be located —
    caller maps that to UNKNOWN verdict.
    """
    import torch

    # Locate the final value-head Linear weight. Try value_fc2.weight first
    # (canonical HexTacToeNet name) then the last 'value*' fc weight as fallback.
    target = state.get("value_fc2.weight")
    target_name = "value_fc2.weight"
    if target is None:
        # Fallback: search any value-head fc-like 2D weight tensor, pick highest.
        cands = [
            (k, v) for k, v in state.items()
            if k.endswith(".weight") and "value" in k.lower()
            and isinstance(v, torch.Tensor) and v.ndim == 2
        ]
        if cands:
            cands.sort()
            target_name, target = cands[-1]

    if target is None or not isinstance(target, torch.Tensor) or target.ndim < 2:
        return False, {"value_head_located": False}

    fan_in = int(target.shape[1])
    bound = math.sqrt(1.0 / max(fan_in, 1))
    amax = float(target.abs().max().item())
    ratio = amax / bound if bound > 0 else float("inf")
    fires = FRESH_RATIO_LO <= ratio <= FRESH_RATIO_HI
    return fires, {
        "value_head_located": True,
        "tensor_name": target_name,
        "fan_in": fan_in,
        "value_fc2_weight_abs_max": amax,
        "expected_kaiming_bound": bound,
        "ratio_to_bound": ratio,
        "fresh_window": [FRESH_RATIO_LO, FRESH_RATIO_HI],
    }


def _build_record(path: Path) -> tuple[dict[str, Any], int]:
    """Return (record, exit_code)."""
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": None,
        "format": None,
        "key_count": None,
        "param_count": None,
        "head_shapes": {"policy_out": None, "value_out": None, "aux_out": None},
        "fresh_init_signature": False,
        "fresh_init_details": {},
        "verdict": "UNKNOWN",
        "error": None,
    }

    if not path.exists():
        record["error"] = f"path not found: {path}"
        return record, EXIT_ERROR

    record["sha256"] = _sha256_short(path)
    record["format"] = _detect_format(path)

    try:
        import torch
    except ImportError as e:
        record["error"] = f"torch import failed: {e}"
        return record, EXIT_ERROR

    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as e:
        record["error"] = f"torch.load failed: {type(e).__name__}: {e}"
        return record, EXIT_ERROR

    state = _unwrap_state_dict(obj)
    if not isinstance(state, Mapping):
        record["error"] = f"unwrapped state is not a mapping (type={type(state).__name__})"
        return record, EXIT_ERROR

    record["key_count"] = len(state)
    record["param_count"] = sum(
        int(v.numel()) for v in state.values() if isinstance(v, torch.Tensor)
    )
    record["head_shapes"] = {
        "policy_out": _trailing_out_dim(state, ("policy",)),
        "value_out": _trailing_out_dim(state, ("value_fc", "value_head")),
        "aux_out": _trailing_out_dim(state, ("opp_reply", "aux")),
    }

    fires, details = _fresh_init_check(state)
    record["fresh_init_signature"] = fires
    record["fresh_init_details"] = details

    if not details.get("value_head_located", False):
        record["verdict"] = "UNKNOWN"
        return record, EXIT_ERROR
    if fires:
        record["verdict"] = "FRESH_INIT_SUSPECT"
        return record, EXIT_FRESH
    record["verdict"] = "TRAINED"
    return record, EXIT_TRAINED


def _format_summary(record: dict[str, Any]) -> str:
    h = record["head_shapes"]
    fid = record["fresh_init_details"]
    lines = [
        f"path:      {record['path']}",
        f"sha256:    {record['sha256']}  format={record['format']}",
        f"keys:      {record['key_count']}  params={record['param_count']}",
        f"heads:     policy={h.get('policy_out')}  value={h.get('value_out')}  aux={h.get('aux_out')}",
    ]
    if fid.get("value_head_located"):
        lines.append(
            f"fresh-sig: |max|={fid['value_fc2_weight_abs_max']:.6f}  "
            f"bound={fid['expected_kaiming_bound']:.6f}  "
            f"ratio={fid['ratio_to_bound']:.3f}  "
            f"window=[{FRESH_RATIO_LO},{FRESH_RATIO_HI}]"
        )
    else:
        lines.append("fresh-sig: value head NOT located")
    lines.append(f"verdict:   {record['verdict']}")
    if record.get("error"):
        lines.append(f"error:     {record['error']}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: verify_anchor.py <ckpt_path>", file=sys.stderr)
        return EXIT_ERROR
    path = Path(argv[1])
    record, code = _build_record(path)

    # Sidecar write (skip if input path itself missing — no sensible target).
    if path.exists():
        sidecar = path.with_suffix(path.suffix + ".verify.json")
        try:
            sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        except OSError as e:
            print(f"warning: sidecar write failed: {e}", file=sys.stderr)

    print(_format_summary(record))
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
