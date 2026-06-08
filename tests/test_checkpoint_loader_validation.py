"""D-EVALFOUND C1 — loader registry-by-name validation guard.

The §D-FOUNDING driver sidestepped a hardcoded-v6 loader path that crashed on the
4-plane v6_live2 net. The proper fix is a guard that validates the arch inferred from
a checkpoint matches the registry spec resolved by NAME — no silent shape-sniff
fallback. This tests the pure guard.
"""
from __future__ import annotations

import pytest

from hexo_rl.encoding import lookup
from hexo_rl.eval.checkpoint_loader import validate_arch_against_spec


def test_validate_passes_on_matching_v6_live2():
    spec = lookup("v6_live2")  # n_planes=4, policy_logit_count=362
    validate_arch_against_spec(in_channels=4, policy_logit_count=362, spec=spec)  # no raise


def test_validate_rejects_wrong_in_channels():
    spec = lookup("v6_live2")
    with pytest.raises(ValueError, match="in_channels"):
        validate_arch_against_spec(in_channels=8, policy_logit_count=362, spec=spec)


def test_validate_rejects_wrong_policy_logit_count():
    spec = lookup("v6_live2")
    with pytest.raises(ValueError, match="policy"):
        validate_arch_against_spec(in_channels=4, policy_logit_count=625, spec=spec)
