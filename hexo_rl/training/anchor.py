"""Best-model anchor management — atomic save, resilient load, quarantine.

Owns ``best_model.pt`` lifecycle: torch.save round-trip verify + .bak rotation
on save; best → .bak → bootstrap_*.pt fallback chain on load; corrupt-anchor
quarantine with timestamp suffix.

Extracted from training/loop.py per §159 refactor. No behavior change.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import structlog
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.trainer import Trainer

log = structlog.get_logger(__name__)


# Bootstrap candidates tried (in order) when no usable best_model.pt exists.
# Fresh runs anchor against the trained bootstrap, not a random fresh-init
# copy of trainer.model.
_BOOTSTRAP_ANCHOR_CANDIDATES: tuple[str, ...] = (
    "checkpoints/bootstrap_model_v6.pt",
    "checkpoints/bootstrap_model_v7full.pt",
)


def save_best_model_atomic(
    model: torch.nn.Module,
    path: Path,
    *,
    step: int | None = None,
    run_id: str | None = None,
    encoding: str | None = None,
) -> None:
    """Save ``state_dict()`` to ``path`` atomically with one-revision backup.

    Sequence:
      1. write to ``path.tmp``,
      2. verify the tmp file actually loads (catches partial writes),
      3. rotate any existing ``path`` to ``path.bak`` (clobbers an older bak),
      4. rename ``path.tmp`` → ``path``.

    A SIGKILL between (3) and (4) leaves ``.bak`` as the recovery copy;
    ``load_best_model_resilient`` knows to fall through to it.

    §D-LOOPFIX W3 — when ``step`` is supplied (the promotion path) the payload
    is wrapped with provenance (``step`` / ``run_id`` / ``promoted`` /
    ``encoding`` / ``metadata.encoding_name``) and a ``.provenance.json`` sidecar
    is written, so a promoted anchor is log- AND filename-distinguishable from
    the bootstrap. ``Trainer.load_checkpoint`` recovers ``step`` from the wrapper
    (was 0 — promoted anchors looked like bootstrap). With ``step=None`` the
    legacy bare-``state_dict`` payload is written (back-compat: callers that
    don't track provenance, e.g. tests / startup fresh-init without a run_id).
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    bak = path.with_suffix(path.suffix + ".bak")
    sd = model.state_dict()
    payload: Any
    if step is None:
        payload = sd
    else:
        payload = {
            "model_state": sd,
            "step": int(step),
            "run_id": run_id,
            "promoted": True,
            "encoding": encoding,
            "metadata": {"encoding_name": encoding} if encoding is not None else {},
        }
    torch.save(payload, tmp)
    # Round-trip verify — torch.save is not atomic on some filesystems and
    # a process kill mid-write produces exactly the truncated zip we are
    # trying to defend against.
    torch.load(tmp, map_location="cpu", weights_only=True)
    if path.exists():
        path.replace(bak)
    tmp.replace(path)
    if step is not None:
        _write_provenance_sidecar(path, step=int(step), run_id=run_id, encoding=encoding)


def _write_provenance_sidecar(
    path: Path, *, step: int, run_id: str | None, encoding: str | None,
) -> None:
    """Write ``<path>.provenance.json`` (atomic) so a promoted anchor's identity
    is greppable without loading torch — closes the W2/W3 forensic gap where a
    promoted golong anchor could only be told apart from bootstrap by a
    tensor-by-tensor compare."""
    prov = {"step": step, "run_id": run_id, "encoding": encoding, "promoted": True}
    sidecar = path.with_name(path.name + ".provenance.json")
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp.write_text(json.dumps(prov, indent=2))
    tmp.replace(sidecar)


def state_dict_sha256(state_dict: dict[str, Any]) -> str:
    """Deterministic sha256 over a model ``state_dict`` (sorted keys + raw tensor
    bytes). Stable across processes and save formats for identical weights — this
    is the tensor-identity compare the promogate W2 forensics had to do by hand,
    made cheap so a launch can pin its intended incumbent and the operator can
    reproduce the pin with ``scripts/anchor_sha256.py``.
    """
    # Canonicalise compile/DDP wrappers so the hash matches whether the weights
    # came from an UNWRAPPED model.state_dict() (resolve_anchor) or a raw stored
    # state_dict that still carries `_orig_mod.` / `module.` prefixes
    # (scripts/anchor_sha256.py via extract_model_state). Without this a
    # compiled-checkpoint pin would spuriously hard-fail the launch. Sort by the
    # CANONICAL key so the digest order is stable under (partial) wrapping.
    def _canon(key: str) -> str:
        changed = True
        while changed:  # fixpoint: handles nested/any-order wrappers (module._orig_mod.)
            changed = False
            for prefix in ("_orig_mod.", "module."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        return key

    h = hashlib.sha256()
    for canon_key, raw_key in sorted((_canon(k), k) for k in state_dict.keys()):
        h.update(canon_key.encode("utf-8"))
        value = state_dict[raw_key]
        if isinstance(value, torch.Tensor):
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(repr(value).encode("utf-8"))
    return h.hexdigest()


def checkpoint_state_sha256(path: Path) -> str:
    """sha256 of the MODEL weights STORED in a checkpoint file — the canonical,
    dtype/device-independent identity the W2 pin compares against.

    Hashes the STORED state (the same number ``scripts/anchor_sha256.py`` prints
    and ``expected_anchor_sha256`` is set from), NOT a live model whose runtime
    dtype diverges from disk. On CUDA the resolved anchor model is fp16, so
    ``state_dict_sha256(model.state_dict())`` produced a different digest than
    the fp32 on-disk pin and false-failed a CORRECT incumbent (§D-RERUNPREP F1,
    caught by the Phase-3 GPU smoke; invisible on CPU where fp16 is disabled).
    """
    from hexo_rl.training.checkpoints import extract_model_state

    raw = torch.load(path, map_location="cpu", weights_only=False)
    return state_dict_sha256(extract_model_state(raw))


def verify_launch_anchor_pin(
    *,
    eval_ext_config: dict[str, Any],
    checkpoint_path: "str | Path | None",
    trainer_step: int | None,
    run_id: str | None,
) -> None:
    """§D-RERUNPREP F1 (W2-VACUOUS) — verify the launch pin on the fresh-init path.

    ``resolve_anchor``'s existing-anchor branch only checks the pin when a
    ``best_model.pt`` is present. But the runbook's preflight ``rm best_model.pt``
    routes the intended launch through the fresh-init branch, which seeds the
    anchor from the trainer's ``--checkpoint`` and never checked the pin — so the
    W2 guard gave the real launch ZERO protection. Verify the pin against the
    ``--checkpoint`` source here, hashing the STORED weights (dtype-invariant).
    No-op when no pin is configured.
    """
    _pin = (
        eval_ext_config.get("eval_pipeline", {})
        .get("gating", {})
        .get("expected_anchor_sha256")
    )
    if _pin is None:
        return
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        _seed_sha = checkpoint_state_sha256(Path(checkpoint_path))
        log.info(
            "anchor_identity",
            path=str(checkpoint_path),
            step=trainer_step,
            run_id=run_id,
            sha256=_seed_sha,
            pinned=_pin,
            source="fresh_init_checkpoint",
        )
        if _seed_sha != _pin:
            raise RuntimeError(
                f"anchor sha256 mismatch (fresh-init seed): the --checkpoint "
                f"{checkpoint_path} resolved to {_seed_sha} but the run config "
                f"pinned {_pin}. Refusing to launch. best_model.pt was absent, so "
                f"the anchor is seeded from --checkpoint; launch with the pinned "
                f"incumbent as --checkpoint, or clear expected_anchor_sha256."
            )
    else:
        # Fail CLOSED (§D-RERUNPREP F1 review N2): a pin set + no verifiable source
        # is the very "guard that proceeds on a warning" pattern this fix exists to
        # kill — refuse rather than launch a 4-day run on an UNVERIFIED incumbent.
        raise RuntimeError(
            f"expected_anchor_sha256={_pin} is pinned but there is no readable "
            f"--checkpoint to verify the fresh-init anchor against "
            f"(checkpoint_path={checkpoint_path}). The launch incumbent would be "
            f"UNVERIFIED — refusing to launch. Pass the pinned incumbent as "
            f"--checkpoint, or clear expected_anchor_sha256."
        )


def _quarantine_corrupt(path: Path) -> Path:
    """Move a corrupt anchor aside with a unique suffix so it isn't overwritten
    by the next write. Returns the destination path for logging."""
    ts = time.strftime("%Y%m%dT%H%M%S")
    dest = path.with_suffix(path.suffix + f".corrupt-{ts}")
    path.replace(dest)
    return dest


def _try_load_anchor(
    candidate: Path,
    *,
    checkpoint_dir: str,
    device: torch.device,
    fallback_config: dict[str, Any],
    skip_encoding_mismatch: bool = False,
) -> "Optional[tuple[Trainer, Path]]":
    """Attempt to load ``candidate`` as an anchor checkpoint.

    Returns ``(loaded Trainer, candidate path)`` on success, None on any failure
    (corrupt zip, arch mismatch in the load path, unreadable file). The source
    PATH is returned so the caller can hash the STORED weights for the W2 pin —
    the live model's runtime dtype (fp16 on CUDA) diverges from disk (§D-RERUNPREP
    F1). Failures are logged but not raised — the caller decides what to fall back to.

    Exception (D-FORENSIC F1): an encoding disagreement RAISES by default —
    it is a configuration error, not corruption, and must not enter the
    quarantine/fresh-init machinery. ``skip_encoding_mismatch=True`` restores
    skip-on-mismatch for FOREIGN multi-candidate fallbacks (the hardcoded
    v6-family bootstrap list), where not matching a non-v6 variant is by
    design, not lineage rot (red-team R3b).
    """
    if not candidate.exists():
        return None
    try:
        trainer = Trainer.load_checkpoint(
            candidate,
            checkpoint_dir=checkpoint_dir,
            device=device,
            fallback_config=fallback_config,
            config_overrides={"input_channels": None, "in_channels": None},
        )
        return (trainer, candidate)
    except Exception as exc:  # broad by design: corrupt zip, arch mismatch, CUDA OOM all fall through to next candidate
        # D-FORENSIC F1: an encoding disagreement is a CONFIGURATION error,
        # NOT corruption — returning None here routes a VALID anchor into the
        # quarantine/fresh-init machinery (rename to .corrupt-*, then a
        # silent fresh-init overwrite of best_model.pt; the d1m single-window
        # cascade entered via exactly this bootstrap path). RAISE instead:
        # hard-fail the launch, matching the trainer path. Dedicated ERROR
        # event first so it can't hide among routine anchor_load_failed
        # warnings.
        if isinstance(exc, ValueError) and "Encoding version disagrees" in str(exc):
            if skip_encoding_mismatch:
                log.warning(
                    "anchor_encoding_mismatch_skipped",
                    path=str(candidate),
                    error=str(exc),
                    msg=(
                        "foreign bootstrap candidate's encoding does not "
                        "match the declared config encoding — skipping to "
                        "the next candidate (by-design mismatch, not "
                        "lineage rot)."
                    ),
                )
                return None
            log.error(
                "anchor_encoding_mismatch",
                path=str(candidate),
                error=str(exc),
                msg=(
                    "anchor candidate's encoding disagrees with the "
                    "explicitly declared config encoding — refusing to fall "
                    "through to backup/bootstrap/fresh-init. Re-stamp the "
                    "anchor or fix the variant's `encoding:`."
                ),
            )
            raise
        log.warning(
            "anchor_load_failed",
            path=str(candidate),
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return None


def load_best_model_resilient(
    best_model_path: Path,
    *,
    checkpoint_dir: str,
    device: torch.device,
    config: dict[str, Any],
) -> "Optional[tuple[Trainer, Path]]":
    """Try best_model.pt → its .bak → bootstrap candidates. None if all fail.

    Returns ``(Trainer, source_path)`` — the path identifies WHICH file supplied
    the weights so the W2 pin hashes the STORED (fp32) state rather than the live
    (fp16-on-CUDA) model (§D-RERUNPREP F1).

    On corruption of ``best_model.pt`` the file is quarantined and the next
    candidate is tried; the eventual successful candidate is then promoted
    in-place to ``best_model.pt`` by the caller (atomic save).
    """
    fallback_cfg = {k: v for k, v in config.items()
                    if k not in ("in_channels", "input_channels")}

    # 1. Live anchor.
    if best_model_path.exists():
        ref = _try_load_anchor(
            best_model_path,
            checkpoint_dir=checkpoint_dir,
            device=device,
            fallback_config=fallback_cfg,
        )
        if ref is not None:
            return ref
        quarantined = _quarantine_corrupt(best_model_path)
        log.warning(
            "anchor_quarantined",
            original=str(best_model_path),
            quarantined=str(quarantined),
            msg="best_model.pt was unreadable — moved aside, falling through to backup/bootstrap",
        )

    # 2. One-revision backup written by the previous atomic save.
    bak = best_model_path.with_suffix(best_model_path.suffix + ".bak")
    if bak.exists():
        ref = _try_load_anchor(
            bak,
            checkpoint_dir=checkpoint_dir,
            device=device,
            fallback_config=fallback_cfg,
        )
        if ref is not None:
            log.info("anchor_recovered_from_bak", path=str(bak))
            return ref

    # 3. Repo-level bootstrap candidates — FOREIGN files (v6-family by
    # construction): an encoding mismatch here is by design for any non-v6
    # variant, so skip instead of raising (red-team R3b; the raise is for
    # the same-lineage best/.bak tiers above).
    for rel in _BOOTSTRAP_ANCHOR_CANDIDATES:
        cand = Path(rel)
        ref = _try_load_anchor(
            cand,
            checkpoint_dir=checkpoint_dir,
            device=device,
            fallback_config=fallback_cfg,
            skip_encoding_mismatch=True,
        )
        if ref is not None:
            log.info("anchor_loaded_from_bootstrap", path=str(cand))
            return ref

    return None


@dataclass
class AnchorState:
    """Resolved best-model anchor + provenance.

    ``best_model`` is None only when no eval pipeline is configured (the
    caller supplies eval_pipeline=None) — this matches the pre-refactor
    invariant where best_model stayed None outside the eval branch.
    ``best_model_step`` is None when no anchor step was recoverable from
    the loaded checkpoint.
    """

    best_model: Optional[HexTacToeNet]
    best_model_step: Optional[int]
    best_model_path: Path


def resolve_anchor(
    *,
    eval_pipeline: Any,                      # EvalPipeline | None
    eval_ext_config: dict[str, Any],
    inf_model: torch.nn.Module,              # mutated when arch matches
    trainer: Trainer,
    args: Any,                               # argparse.Namespace
    config: dict[str, Any],
    device: torch.device,
    board_size: int,
    res_blocks: int,
    filters: int,
    in_channels: int,
    input_channels: Any,
    se_reduction_ratio: int,
    run_id: str | None = None,
) -> AnchorState:
    """Resolve the best-model anchor and sync ``inf_model`` to it.

    Steps:
      1. Resolve ``best_model_path`` from eval.yaml gating config.
      2. If eval_pipeline is None → return AnchorState(None, None, path).
      3. Try resilient load (best.pt → .bak → bootstrap candidates).
      4. On success: unwrap torch.compile, persist to live path if recovered
         from a fallback, sync inf_model when in_channels match (preserves
         the input_channel_index buffer when present), warn on
         trainer.step ≠ best_model_step (M2 invariant).
      5. On total failure: fresh-init from trainer.model, save atomically.

    Mutates ``inf_model`` via ``load_state_dict`` when the architecture
    matches the loaded anchor. Sweep variants intentionally leave inf_model
    on trainer.model weights (arch-mismatch logged, no sync).
    """
    best_model_path = Path(
        eval_ext_config.get("eval_pipeline", {}).get("gating", {}).get(
            "best_model_path", "checkpoints/best_model.pt"
        )
    )
    best_model: HexTacToeNet | None = None
    best_model_step: int | None = None
    if eval_pipeline is None:
        return AnchorState(None, None, best_model_path)

    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    # Resilient anchor load: tries best_model.pt → its .bak →
    # _BOOTSTRAP_ANCHOR_CANDIDATES → fresh init from trainer.model.
    # A corrupt best_model.pt is quarantined with a timestamp suffix
    # instead of being silently discarded.
    _loaded = load_best_model_resilient(
        best_model_path,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        config=config,
    )
    if _loaded is not None:
        best_ref, best_source_path = _loaded
        # Unwrap torch.compile — best_model's state_dict() is consumed
        # at multiple load_state_dict call sites below; leaving the
        # OptimizedModule wrapper would inject `_orig_mod.*` prefixes
        # into every subsequent state_dict() call.
        best_model = getattr(best_ref.model, "_orig_mod", best_ref.model)
        best_model.eval()
        best_model_step = best_ref.step
        # §D-LOOPFIX W2 — incumbent identity: ALWAYS log the resolved anchor's
        # sha256 + path + step + run_id, and HARD-FAIL the launch when it
        # disagrees with the config-pinned expectation. This is the gate the
        # silent .bak restore drove straight through — it installed golong@50k-
        # PEAK as the A/B's incumbent AND self-play generator. The resilient
        # load above may have come from best.pt, its .bak, or a bootstrap
        # candidate; the assert hashes the STORED weights at ``best_source_path``
        # (NOT the live model — its runtime fp16 cast on CUDA diverges from the
        # fp32 on-disk pin and false-failed a CORRECT incumbent, §D-RERUNPREP F1),
        # so ANY source that yields non-pinned weights is refused here.
        _anchor_sha = checkpoint_state_sha256(best_source_path)
        _pin = (
            eval_ext_config.get("eval_pipeline", {})
            .get("gating", {})
            .get("expected_anchor_sha256")
        )
        log.info(
            "anchor_identity",
            path=str(best_source_path),
            step=best_model_step,
            run_id=run_id,
            sha256=_anchor_sha,
            pinned=_pin,
        )
        if _pin is not None and _anchor_sha != _pin:
            raise RuntimeError(
                f"anchor sha256 mismatch: best_model.pt resolved to {_anchor_sha} "
                f"but the run config pinned {_pin}. Refusing to launch. Two causes:\n"
                f"  (a) WRONG INCUMBENT (W2) — a silent best_model.pt "
                f"restore-from-.bak / a stale anchor from another experiment leaked "
                f"in. Re-pin the intended incumbent into checkpoints/best_model.pt.\n"
                f"  (b) LEGITIMATE RESUME after promotions — best_model.pt has "
                f"advanced past the pinned launch incumbent (the pin is a "
                f"LAUNCH-time assertion; promotions move the anchor). This is "
                f"EXPECTED on resume: update expected_anchor_sha256 to the current "
                f"best_model.pt sha (scripts/anchor_sha256.py checkpoints/best_model.pt) "
                f"or clear it.\n"
                f"The pin cannot tell (a) from (b) by content alone — it is the "
                f"operator's per-invocation declaration of the intended incumbent."
            )
        # If best_model.pt was missing/corrupt and we recovered from a
        # bootstrap or .bak, persist the chosen anchor as the live
        # best_model.pt so subsequent runs find it directly.
        if not best_model_path.exists():  # True after quarantine: _quarantine_corrupt renames, not deletes
            save_best_model_atomic(best_model, best_model_path)
            log.info("anchor_persisted_from_fallback", path=str(best_model_path))
        # Graduation gate: self-play consumes anchor weights, not trainer.model.
        # Sync inf_model to the loaded anchor before workers start — but only
        # when architectures match. Sweep variants train a reduced-channel model
        # against an 18-channel anchor; syncing architectures is impossible and
        # wrong (sweep inf_model should start from trainer.model, not the anchor).
        _inf_base = getattr(inf_model, "_orig_mod", inf_model)
        if _inf_base.in_channels == best_model.in_channels:
            _best_sd = best_model.state_dict()
            # Anchor is always loaded without input_channels (see _try_load_anchor
            # config_overrides). If _inf_base was built with input_channels, inject
            # its own buffer so load_state_dict sees a consistent state_dict.
            _inf_idx = getattr(_inf_base, "input_channel_index", None)
            if "input_channel_index" not in _best_sd and _inf_idx is not None:
                _best_sd = dict(_best_sd)
                _best_sd["input_channel_index"] = _inf_idx.detach().clone()
            _inf_base.load_state_dict(_best_sd)
        else:
            log.info(
                "inf_model_anchor_arch_mismatch_skip_sync",
                inf_in_channels=_inf_base.in_channels,
                anchor_in_channels=best_model.in_channels,
                msg="inf_model starts from trainer.model (sweep variant)",
            )
        log.info("best_model_loaded", path=str(best_model_path), step=best_model_step)
        # M2: warn if resumed trainer.model and loaded anchor diverge on step.
        # Either side may legitimately be ahead (anchor rollback, or training
        # continued past last promotion) but a silent mismatch can produce a
        # trivially-promoted first eval that wipes a hand-picked anchor.
        if best_model_step is not None and trainer.step != best_model_step:
            log.warning(
                "resume_anchor_step_mismatch",
                trainer_step=trainer.step,
                best_model_step=best_model_step,
                msg=(
                    "trainer.model and best_model.pt were loaded from different "
                    "training steps. First eval will compare the current trainer "
                    "weights against this anchor; confirm this is intended."
                ),
            )
    else:
        # No usable anchor anywhere — last-resort fresh init from trainer.model.
        # §D-RERUNPREP F1 (W2-VACUOUS): this is the branch the runbook's preflight
        # `rm best_model.pt` actually routes the launch through — verify the pin
        # against the --checkpoint the anchor is seeded from BEFORE building it,
        # so the W2 guard protects the real launch path (not just a stale-anchor one).
        verify_launch_anchor_pin(
            eval_ext_config=eval_ext_config,
            checkpoint_path=getattr(args, "checkpoint", None),
            trainer_step=trainer.step,
            run_id=run_id,
        )
        # On a clean run this should rarely fire: the bootstrap candidates above
        # are the canonical first-run anchor. Reaching this branch means the box
        # has neither best_model.pt, its .bak, nor any bootstrap_*.pt — flag it.
        log.warning(
            "anchor_fresh_init_no_bootstrap",
            tried=list(_BOOTSTRAP_ANCHOR_CANDIDATES),
            msg="No anchor or bootstrap available — initialising best_model.pt from current trainer.model. "
                "Drop a bootstrap_model.pt (or one of _BOOTSTRAP_ANCHOR_CANDIDATES) into checkpoints/ to anchor wr_best meaningfully.",
        )
        best_model = HexTacToeNet(
            board_size=board_size, res_blocks=res_blocks, filters=filters,
            in_channels=in_channels, input_channels=input_channels,
            se_reduction_ratio=se_reduction_ratio,
        ).to(device)
        # §S181-AUDIT Wave 2 — fresh anchor adopts EMA weights when enabled
        # so this rare path stays consistent with the rest of the inference
        # dispatch surface (lifecycle.build_inference_model, eval kickoff,
        # promotion).
        best_model.load_state_dict(trainer.inference_state_dict())
        best_model.eval()
        save_best_model_atomic(best_model, best_model_path)
        best_model_step = trainer.step
        log.info("best_model_initialized", path=str(best_model_path), step=best_model_step)

    return AnchorState(best_model, best_model_step, best_model_path)
