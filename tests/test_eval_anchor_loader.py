"""§172 B1 follow-up — eval_pipeline._load_anchor_model regression.

Pre-fix: ``eval_pipeline.py:49`` called
``model.load_state_dict(normalize_model_state_dict_keys(state), strict=True)``.
``normalize_model_state_dict_keys`` injects ``tower.*`` ↔ ``trunk.tower.*``
mirror aliases for backward-compat with pre-§99 checkpoints; the current
``HexTacToeNet`` declares only ``trunk.tower.*`` parameters, so the
``tower.*`` mirrors are "Unexpected key(s)" under ``strict=True``. The
bootstrap_anchor floor opponent could not load.

Post-fix: delegates to ``hexo_rl.eval.checkpoint_loader.load_model_with_encoding``,
which branches between the full normalize path (v6/v6w25/v7full) and the
lighter strip-prefix path (v8) and constructs ``HexTacToeNet`` from
shape-inferred hyperparameters.

This regression test loads each canonical anchor checkpoint that exists on
disk and asserts a forward pass succeeds with the encoding-correct input
shape. Skips encodings whose checkpoint is absent (CI-friendly).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hexo_rl.encoding import lookup as registry_lookup
from hexo_rl.eval.eval_pipeline import _load_anchor_model
from hexo_rl.model.network import HexTacToeNet


_REPO_ROOT = Path(__file__).resolve().parent.parent
_CKPT_DIR = _REPO_ROOT / "checkpoints"


# (encoding_name, ckpt_filename) for canonical anchors used by
# eval_pipeline.bootstrap_anchor across §150+ migrations.
_ANCHOR_CASES = [
    ("v7full", "bootstrap_model_v7full.pt"),
    ("v6w25", "bootstrap_model_v6w25.pt"),
    ("v6", "bootstrap_model_v6.pt"),
]


@pytest.mark.parametrize(("encoding_name", "ckpt_name"), _ANCHOR_CASES)
def test_anchor_loader_strict_load_succeeds(encoding_name: str, ckpt_name: str) -> None:
    ckpt_path = _CKPT_DIR / ckpt_name
    if not ckpt_path.exists():
        pytest.skip(f"anchor checkpoint not present: {ckpt_path}")

    spec = registry_lookup(encoding_name)
    device = torch.device("cpu")

    from tests._a2_compat import a2_load_or_skip
    model, loaded_spec, loaded_label = a2_load_or_skip(
        _load_anchor_model, ckpt_path, device,
    )

    assert isinstance(model, HexTacToeNet)
    # Encoding parity: the detected label/spec must match the registry
    # entry for this canonical anchor. Caller (run_evaluation) logs the
    # label so cross-encoding drift is observable; the test pins the
    # mapping so an inadvertent label-detection regression surfaces here.
    #
    # v7full collapses to label='v6' because checkpoint_loader's vocabulary
    # is {'v6', 'v6w25', 'v8'} and v6 + v7full are wire-format identical
    # (registry.toml: same n_planes=8, board_size=19, policy_logit_count=362).
    # Distinguishing them requires the A5 metadata stamping landing on
    # bootstrap_model_v7full.pt and an extension of the loader vocabulary —
    # both deferred per A9 §4 #5. The wire-format collapse is safe because
    # downstream eval comparisons are apples-to-apples at the tensor level.
    expected_label = "v6" if encoding_name == "v7full" else encoding_name
    assert loaded_label == expected_label, (
        f"{ckpt_name}: label mismatch (got {loaded_label!r}, "
        f"expected {expected_label!r})"
    )
    assert loaded_spec.n_planes == spec.n_planes
    assert loaded_spec.board_size == spec.board_size
    assert loaded_spec.policy_logit_count == spec.policy_logit_count

    # Encoding-correct input shape: [1, n_planes, board_size, board_size].
    # v6w25 forwards through the registry-aware path so the same call shape
    # works for K-cluster encodings at K=1.
    x = torch.zeros(1, spec.n_planes, spec.board_size, spec.board_size)
    with torch.no_grad():
        out = model(x)

    # HexTacToeNet.forward returns (policy_logits, value, ...); minimum
    # contract is policy + value heads.
    assert isinstance(out, tuple) and len(out) >= 2
    policy_logits, value = out[0], out[1]

    assert policy_logits.shape[0] == 1
    assert policy_logits.shape[-1] == spec.policy_logit_count, (
        f"{encoding_name}: policy logit count mismatch "
        f"(got {policy_logits.shape[-1]}, registry expects {spec.policy_logit_count})"
    )
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(value).all()
