"""CONFRES S1/B2 — loud declared-vs-effective LR on a full-checkpoint resume.

On a full resume the LR is checkpoint-STATE-owned, so a declared variant ``lr:`` is silently
ignored (the v2-LR strike). This makes it LOUD. The WARN must fire on the REAL strike (the
operator declared an lr that differs from the checkpoint's baked initial lr → override ignored)
and must NOT fire spuriously on a normal resume where the only difference is the scheduler having
annealed the effective lr below the declared initial lr.
"""
from __future__ import annotations

from hexo_rl.config.resolve.lr import LrProvenance, resolve_lr_provenance


def test_declared_differs_from_baked_is_override_ignored():
    prov = resolve_lr_provenance(declared=0.001, baked=0.002, effective=0.0018)
    assert prov.override_ignored is True
    assert prov.declared == 0.001 and prov.baked == 0.002 and prov.effective == 0.0018


def test_annealed_effective_alone_does_not_warn():
    # declared == baked (0.002); the scheduler has annealed effective to 0.0018569 — NORMAL.
    # This must NOT warn (else every resume warns). The strike is declared != baked, not the anneal.
    prov = resolve_lr_provenance(declared=0.002, baked=0.002, effective=0.0018569)
    assert prov.override_ignored is False


def test_no_declared_lr_never_warns():
    prov = resolve_lr_provenance(declared=None, baked=0.002, effective=0.0018)
    assert prov.override_ignored is False


def test_no_baked_lr_cannot_compare():
    prov = resolve_lr_provenance(declared=0.001, baked=None, effective=0.001)
    assert prov.override_ignored is False


def test_provenance_carries_effective_for_emission():
    # B2: the annealed effective (state-blob) value is carried through for the batch-6 emission,
    # even when there is no override to warn about.
    prov = resolve_lr_provenance(declared=0.002, baked=0.002, effective=0.0018569)
    assert isinstance(prov, LrProvenance)
    assert prov.effective == 0.0018569


def test_equal_within_tolerance_does_not_warn():
    prov = resolve_lr_provenance(declared=0.002, baked=0.002 + 1e-15, effective=0.002)
    assert prov.override_ignored is False
