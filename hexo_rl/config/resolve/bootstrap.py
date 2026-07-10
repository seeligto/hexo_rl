"""CONFRES P3 — single validated resolver for the training bootstrap/resume checkpoint.

One resolver for the ``--checkpoint`` / Makefile ``BOOTSTRAP`` training bootstrap: validate the
resolved path exists at LAUNCH (before ``torch.load``) so a stale path fails loudly + informatively
instead of a late, uninformative ``FileNotFoundError`` deep in checkpoint loading.

Design: docs/designs/confres_design.md §6 (P3).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable


class BootstrapNotFoundError(FileNotFoundError):
    """Resolved bootstrap/resume checkpoint path does not exist.

    Subclasses ``FileNotFoundError`` so existing ``except FileNotFoundError`` handlers still
    catch it — it just fires EARLIER (at launch, in the resolver) and names the override knob.
    """


@dataclass(frozen=True)
class ResolvedBootstrap:
    """The resolved training bootstrap. ``path is None`` ⇒ a fresh run (no checkpoint)."""

    path: str | None
    source: str  # "cli" (a checkpoint path was given) | "none" (fresh run)


def resolve_bootstrap(
    cli_checkpoint: str | None,
    *,
    exists: Callable[[str], bool] = os.path.exists,
) -> ResolvedBootstrap:
    """Resolve + validate the training bootstrap/resume checkpoint.

    ``cli_checkpoint`` is ``args.checkpoint`` — which already carries the Makefile ``BOOTSTRAP``
    default (passed through ``--checkpoint``), an operator ``BOOTSTRAP=`` / ``--checkpoint``
    override, or ``None`` for a fresh run.

    Returns ``ResolvedBootstrap(None, "none")`` for a fresh run. For a provided path, validates
    it exists and returns ``ResolvedBootstrap(path, "cli")``; raises ``BootstrapNotFoundError``
    naming the path + the ``BOOTSTRAP`` override knob if it is missing — at launch, not a late
    ``torch.load`` failure (CONFRES P3). ``exists`` is injectable for testing.
    """
    if cli_checkpoint is None:
        return ResolvedBootstrap(path=None, source="none")
    if not exists(cli_checkpoint):
        raise BootstrapNotFoundError(
            f"bootstrap/resume checkpoint {cli_checkpoint!r} does not exist. "
            f"Set BOOTSTRAP=<path> (make targets) or --checkpoint <path> to an existing file. "
            f"(CONFRES P3: validated at launch, not a late torch.load failure.)"
        )
    return ResolvedBootstrap(path=cli_checkpoint, source="cli")
