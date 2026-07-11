"""E1 promotion-gate CUDA isolation — the subprocess eval worker (Option A).

Runnable as ``python -m hexo_rl.eval.promotion_gate_worker`` in its OWN process (own CUDA context,
no shared ``InferenceServer`` allocator/threads), so the in-loop promotion-gate eval forwards can
NEVER deadlock the self-play inference forwards — the run2 eval-boundary livelock root
(eval-thread ⊥ self-play concurrent GPU forwards; memory ``run2-stall-watchdog``).

Contract (the bridge-sidecar pattern — NEVER stderr, which the 64KB pipe-deadlock class wedges on
a chatty eval):
  in  : ``--candidate <ckpt> --best <ckpt|-> --config <json-file> --step N --radius R|- --result F``
  out : ONE JSON line to ``--result`` F: ``{"event":"promotion_gate_result", ...EvalRoundResult...}``
        then exit 0. A crash → non-zero exit + an absent/partial sidecar the PARENT detects and
        routes to its existing ``_eval_broken`` LOUD path (promotions disabled that round).

Design: docs/designs/e1_promotion_gate_cuda_isolation.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULT_EVENT = "promotion_gate_result"


def _load_model(ckpt_path: str, device, declared_encoding=None):
    """Load an eval checkpoint, gated on the run's DECLARED encoding.

    ``declared_encoding`` (the run's ``full_config["encoding"]``) is an ASSERTION reconciled against
    the checkpoint's OWN baked stamp — a stale/foreign checkpoint whose stamp disagrees RAISES
    :class:`DeclaredEncodingMismatchError` rather than silently CROSS-DECODING into the LIVE promotion
    decision (D-EVALGATE "HOLE 3"; the ``vast-stale-checkpoint-name-collision`` risk). This mirrors the
    in-thread ``eval_pipeline._load_anchor_model`` gating so the subprocess promotion path (LIVE when
    run3 flips ``promotion_gate_subprocess_isolation``) is not a re-opened hole. ``None`` preserves the
    pre-fix shape/stamp-inference behaviour (backward compatible).
    """
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding

    model, _spec, _label = load_model_with_encoding(
        ckpt_path, device, declared_encoding=declared_encoding,
    )
    return model.to(device).eval()


def run_worker(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E1 subprocess promotion-gate eval worker")
    ap.add_argument("--candidate", required=True, help="candidate checkpoint path")
    ap.add_argument("--best", default="-", help="best/anchor checkpoint path, or '-' for none")
    ap.add_argument("--config", required=True, help="JSON file with the full eval config dict")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--best-step", default="-", help="anchor promotion step, or '-'")
    ap.add_argument("--radius", default="-", help="curriculum radius int, or '-' for registry default")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--result", required=True, help="sidecar JSONL path to write the EvalRoundResult")
    args = ap.parse_args(argv)

    import torch

    from hexo_rl.eval.eval_pipeline import EvalPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_config = json.loads(Path(args.config).read_text())
    eval_config = full_config.get("eval_pipeline", full_config)

    # D-EVALGATE HOLE 3 (subprocess path): gate BOTH candidate + best on the run's declared encoding
    # so a stale/foreign checkpoint stamp RAISES instead of silently cross-decoding into the promotion
    # decision. `best` here is the promotion-lineage best_model (run's own encoding by construction),
    # NOT the cross-encoding bootstrap_anchor (loaded separately inside run_evaluation, gated on its own
    # key). `full_config.get("encoding")` reads the run's top-level declared encoding (same access
    # pattern as step_coordinator); a config missing it → None → pre-fix shape/stamp inference.
    declared_encoding = full_config.get("encoding")
    candidate = _load_model(args.candidate, device, declared_encoding=declared_encoding)
    best = None if args.best == "-" else _load_model(args.best, device, declared_encoding=declared_encoding)
    best_step = None if args.best_step == "-" else int(args.best_step)
    radius = None if args.radius == "-" else int(args.radius)

    pipeline = EvalPipeline(eval_config, device, run_id=args.run_id)
    result = pipeline.run_evaluation(
        candidate, int(args.step), best,
        full_config=full_config, best_model_step=best_step, current_radius=radius,
    )

    # Serialize the EvalRoundResult as ONE json line (the sidecar bridge). Non-JSON values (e.g.
    # numpy scalars) are coerced via default=str so a stray type can't wedge the parent's read.
    payload = {"event": RESULT_EVENT, **dict(result)}
    Path(args.result).write_text(json.dumps(payload, default=str) + "\n")
    return 0


def run_promotion_gate_subprocess(
    *,
    candidate_ckpt: str,
    best_ckpt: str | None,
    full_config: dict,
    step: int,
    best_step: int | None,
    radius: int | None,
    work_dir: str,
    run_id: str | None = None,
    timeout_sec: float | None = None,
    python_exe: str | None = None,
) -> dict | None:
    """Parent-side: spawn the isolated eval worker + read its sidecar EvalRoundResult.

    Writes the eval config to a JSON file, launches ``python -m hexo_rl.eval.promotion_gate_worker``
    in its OWN process (own CUDA context — no shared InferenceServer state, the E1 root fix), and
    returns the parsed ``EvalRoundResult``. On ANY failure (non-zero exit, timeout, absent/partial
    sidecar) returns ``None`` — the caller routes that to its LOUD ``_eval_broken`` path (promotions
    disabled that round), matching the in-thread crash contract. Worker stdout/stderr go to a log
    file for forensics (NEVER a pipe the parent reads — the 64KB pipe-deadlock class).
    """
    import subprocess
    import sys as _sys

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    cfg_file = work / f"eval_config_{step}.json"
    result_file = work / f"promotion_gate_result_{step}.jsonl"
    log_file = work / f"promotion_gate_worker_{step}.log"
    cfg_file.write_text(json.dumps(full_config, default=str))

    cmd = [
        python_exe or _sys.executable, "-m", "hexo_rl.eval.promotion_gate_worker",
        "--candidate", candidate_ckpt,
        "--best", best_ckpt if best_ckpt else "-",
        "--config", str(cfg_file),
        "--step", str(int(step)),
        "--best-step", "-" if best_step is None else str(int(best_step)),
        "--radius", "-" if radius is None else str(int(radius)),
        "--result", str(result_file),
    ]
    if run_id:
        cmd += ["--run-id", str(run_id)]

    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    return read_result(str(result_file))


def read_result(result_path: str) -> dict | None:
    """Parent-side: read the worker's sidecar EvalRoundResult, or None if absent/malformed.

    A None return = a broken eval (crashed worker / partial write) → the caller routes to its LOUD
    ``_eval_broken`` path (promotions disabled that round), exactly like the in-thread crash path.
    """
    p = Path(result_path)
    if not p.exists():
        return None
    try:
        line = p.read_text().strip().splitlines()[0]
        payload = json.loads(line)
    except (OSError, ValueError, IndexError):
        return None
    if payload.get("event") != RESULT_EVENT:
        return None
    payload.pop("event", None)
    return payload


if __name__ == "__main__":
    sys.exit(run_worker())
