"""§174 W1 — auto-detect encoding from checkpoint metadata.

Covers ``resolve_encoding_for_eval`` (priority: override > stamped metadata >
shape-inference fallback + DeprecationWarning > friendly re-raise).

Test pattern mirrors hexo_rl/training/tests/test_checkpoint_metadata.py:
synthetic state-dicts with ``trunk.0.weight`` + ``policy_fc.weight`` for the
shape-inference path; ``save_full_checkpoint(encoding_name=...)`` for stamped
checkpoints.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

from hexo_rl.encoding import EncodingRegistryError
from hexo_rl.encoding.resolvers import (
    resolve_encoding_for_eval,
    resolve_from_checkpoint,
)
from hexo_rl.training.checkpoints import save_full_checkpoint


class _SyntheticNet(torch.nn.Module):
    """Minimal NN matching compat.py shape probes."""

    def __init__(self, n_planes: int, policy_logit_count: int) -> None:
        super().__init__()
        # trunk.0.weight matches compat._FIRST_CONV_KEYS[0]
        self.trunk = torch.nn.Sequential(
            torch.nn.Conv2d(n_planes, 4, kernel_size=3, padding=1, bias=False),
        )
        # policy_fc.weight matches compat._POLICY_FC_KEYS[0]
        self.policy_fc = torch.nn.Linear(4, policy_logit_count)

    def forward(self, x):  # pragma: no cover
        return x


def _v6_net() -> _SyntheticNet:
    # v6: n_planes=8, policy_logit_count=362 (19*19 + 1 pass slot)
    return _SyntheticNet(n_planes=8, policy_logit_count=362)


def _v8_net() -> _SyntheticNet:
    # v8: n_planes=11, policy_logit_count=625 (25*25, no pass slot)
    return _SyntheticNet(n_planes=11, policy_logit_count=625)


def _make_optim_scaler(model: torch.nn.Module):
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    return optim, scaler


def _stamped_ckpt(tmp_path: Path, encoding_name: str, model: _SyntheticNet) -> Path:
    optim, scaler = _make_optim_scaler(model)
    ckpt = tmp_path / f"checkpoint_stamped_{encoding_name}.pt"
    save_full_checkpoint(
        model, optim, scaler, scheduler=None,
        step=0, config={}, path=ckpt,
        encoding_name=encoding_name,
    )
    return ckpt


# ---- Case 1: Stamped checkpoint, no override ----------------------------

def test_resolve_stamped_no_override_silent(tmp_path: Path) -> None:
    """Stamped ckpt + no override → matching spec, no DeprecationWarning."""
    ckpt = _stamped_ckpt(tmp_path, "v6", _v6_net())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = resolve_encoding_for_eval(ckpt, encoding_override=None)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations == [], (
        f"unexpected DeprecationWarning(s) on stamped ckpt: "
        f"{[str(w.message) for w in deprecations]}"
    )
    assert spec.name == "v6"


# ---- Case 2: Stamped checkpoint + override → override wins, ckpt untouched ----

def test_resolve_override_does_not_touch_checkpoint(tmp_path: Path) -> None:
    """Override path skips ckpt read entirely.

    Pass a non-existent path: if the function honors the override correctly,
    it never touches the path so no error is raised.
    """
    bogus_path = tmp_path / "does_not_exist.pt"
    assert not bogus_path.exists()
    spec = resolve_encoding_for_eval(bogus_path, encoding_override="v6")
    assert spec.name == "v6"


# ---- Case 3: Legacy v6-shape, no override → DeprecationWarning + v6 spec ----

def test_resolve_legacy_v6_emits_deprecation(tmp_path: Path) -> None:
    """Legacy ckpt (no metadata) routes through shape-inference fallback.

    Filename includes 'v6' so compat._filename_match resolves deterministically.
    """
    model = _v6_net()
    ckpt = tmp_path / "legacy_v6_no_metadata.pt"
    torch.save({"model_state": model.state_dict()}, ckpt)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = resolve_encoding_for_eval(ckpt, encoding_override=None)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) >= 1, "expected DeprecationWarning on legacy ckpt"
    assert spec.name == "v6"


# ---- Case 4: Override vs stamped mismatch → override wins -----------------

def test_resolve_override_overrides_stamped_metadata(tmp_path: Path) -> None:
    """Override-vs-stamp mismatch is permitted; override wins.

    Stamp 'v6' but request 'v8' — function returns v8 spec, never reads ckpt.
    """
    ckpt = _stamped_ckpt(tmp_path, "v6", _v6_net())
    # Sanity: confirm stamp is v6.
    assert resolve_from_checkpoint(ckpt).name == "v6"
    # Override to v8 — should win.
    spec = resolve_encoding_for_eval(ckpt, encoding_override="v8")
    assert spec.name == "v8"


# ---- Case 5: Non-existent ckpt, no override → bubble up I/O error ---------

def test_resolve_missing_ckpt_propagates_error(tmp_path: Path) -> None:
    """No override + non-existent path → I/O-style error bubbles up.

    Function must not silently swallow torch.load's FileNotFoundError /
    RuntimeError into an EncodingRegistryError. Caller needs to see the
    underlying issue.
    """
    bogus = tmp_path / "does_not_exist.pt"
    assert not bogus.exists()
    with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
        resolve_encoding_for_eval(bogus, encoding_override=None)


# ---- Case 6: Unresolvable ckpt → friendly error names registered encodings ----

def test_resolve_unresolvable_ckpt_lists_registered_encodings(tmp_path: Path) -> None:
    """When inference fails, the wrapping error must list registered names.

    A state-dict with in_channels=99 matches no registered encoding; the
    filename has no encoding substring; torch.load succeeds so no I/O
    error masks the resolution failure. We expect EncodingRegistryError
    naming at least one registered encoding (e.g. 'v6') in its message.
    """
    # Construct a state-dict with in_channels=99 (no encoding match) and
    # policy_fc out=999 (no encoding match), filename free of encoding hints.
    bogus_sd = {
        "trunk.0.weight": torch.zeros(4, 99, 3, 3),
        "policy_fc.weight": torch.zeros(999, 4),
    }
    ckpt = tmp_path / "anonymous_unmatchable.pt"
    torch.save({"model_state": bogus_sd}, ckpt)

    with pytest.raises(EncodingRegistryError) as excinfo:
        resolve_encoding_for_eval(ckpt, encoding_override=None)
    msg = str(excinfo.value)
    # Confirm at least one registered encoding name appears in the error,
    # along with the override hint.
    assert "v6" in msg
    assert "--encoding" in msg or "encoding" in msg.lower()
