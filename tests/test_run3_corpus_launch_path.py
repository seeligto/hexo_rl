"""WP0.4 — run3 launch path resolves the single-resolver + sha-pinned corpus.

Integration-marked (memory: launch-path-needs-integration-gate — the default
`not slow and not integration` suite misses launch-abort bugs). Drives the
REAL config-merge entrypoint (`hexo_rl.training.orchestrator.load_train_config`
+ `flatten_config_and_resolve_encoding`, a verbatim extraction of
`scripts/train.py::main` lines 181-289) against the REAL
`configs/variants/run3_dist65.yaml` on disk — not a reimplemented/mocked
config merge — to prove:

  1. `run3_dist65.yaml`'s `mixing.pretrained_buffer_path` (collapsed onto
     `"<auto>"` at WP0.4) resolves through the single resolver
     (`resolve_corpus_path`) to the exact launch-pinned NPZ path.
  2. That encoding has a registered launch sha pin
     (`resolve_corpus_sha_pin`) equal to the value pinned in
     `docs/registers/run3_corpus_manifest.md`.
  3. If the real corpus NPZ is present on this host (may be absent in a
     fresh clone — the file is data, not checked into git), its actual
     on-disk sha256 matches the pin — i.e. `load_pretrained_buffer`'s sha
     gate would NOT raise for a real run3 launch on this host.

Does not invoke `load_pretrained_buffer` itself against the real 2.65 GB /
610954-position corpus (the NPZ-to-ReplayBuffer materialization path is
pre-existing, unchanged by WP0.4, and separately covered by
`tests/test_corpus_chain_target.py` against synthetic corpora + by run2's
completed production training on this exact NPZ) — that would cost minutes
of wall time and several GB of RAM for no additional coverage of the WP0.4
change itself. The sha-gate LOGIC (match / mismatch / sidecar-desync) is
unit-tested against synthetic corpora in `tests/test_corpus_sha_pin_gate.py`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

_PINNED_SHA = "3813edc2fb10a7c5ab976a0293e38cbba0fd6b84e5295630f339ca421b345c97"
_EXPECTED_CORPUS_PATH_SUFFIX = "data/bootstrap_corpus_v6_live2_ls.npz"


def _args() -> argparse.Namespace:
    """Minimal Namespace covering the attrs `load_train_config` /
    `flatten_config_and_resolve_encoding` read (mirrors the pattern in
    tests/test_confres_f1_backprop.py::_args)."""
    return argparse.Namespace(
        config=None,
        variant="run3_dist65",
        iterations=None,
        no_compile=False,
    )


@pytest.mark.integration
def test_run3_variant_yaml_declares_auto_not_hardcoded_path():
    """WP0.4 scope item 2 — the launch config itself must route through the
    single resolver (`"<auto>"`), not a hardcoded literal that bypasses it.
    Checked against the RAW variant file (pre-merge) so this fails if the
    collapse is ever reverted, independent of what the path happens to
    resolve to."""
    import yaml

    variant_cfg = yaml.safe_load(Path("configs/variants/run3_dist65.yaml").read_text())
    assert variant_cfg["mixing"]["pretrained_buffer_path"] == "<auto>", (
        "run3_dist65.yaml must route mixing.pretrained_buffer_path through "
        "resolve_corpus_path (WP0.4 single-resolver collapse), not hardcode "
        "the NPZ path — a hardcoded path bypasses the launch sha-pin gate's "
        "registry lookup path (still caught by the sha check itself, but "
        "defeats the single-source-of-truth intent)."
    )


@pytest.mark.integration
def test_run3_launch_config_resolves_pinned_corpus_via_single_resolver():
    from hexo_rl.training import orchestrator as _orch

    log = __import__("structlog").get_logger()
    config, _layers = _orch.load_train_config(_args())
    combined_config, train_cfg, _mcts_cfg, registry_spec, *_ = (
        _orch.flatten_config_and_resolve_encoding(config, _args(), log)
    )

    assert registry_spec.name == "v6_live2_ls"

    mixing_cfg = train_cfg.get("mixing", config.get("mixing", {}))
    resolved_path = mixing_cfg.get("pretrained_buffer_path")
    assert resolved_path is not None, "run3_dist65 must declare mixing.pretrained_buffer_path"
    assert resolved_path != "<auto>", (
        "expand_auto_paths must have resolved the literal before load_pretrained_buffer reads it"
    )
    assert str(resolved_path).endswith(_EXPECTED_CORPUS_PATH_SUFFIX), (
        f"run3's single-resolver collapse must resolve to the launch-pinned NPZ, got {resolved_path!r}"
    )

    from hexo_rl.encoding.resolvers import resolve_corpus_sha_pin
    pin = resolve_corpus_sha_pin(registry_spec)
    assert pin == _PINNED_SHA, (
        f"v6_live2_ls sha pin drifted from docs/registers/run3_corpus_manifest.md: {pin!r}"
    )

    real_path = Path(resolved_path)
    if not real_path.exists():
        pytest.skip(
            f"{real_path} not present on this host (data artifact, not checked into git) — "
            "sha-match-vs-pin not verifiable here; verified separately per "
            "docs/registers/run3_corpus_manifest.md §4"
        )

    from hexo_rl.bootstrap.corpus_io import compute_npz_sha256
    actual_sha = compute_npz_sha256(real_path)
    assert actual_sha == _PINNED_SHA, (
        f"on-disk {real_path} sha {actual_sha[:12]}… does not match the launch pin "
        f"{_PINNED_SHA[:12]}… — load_pretrained_buffer would (correctly) refuse to launch run3 "
        "on this host; re-sync the corpus, do NOT re-export"
    )
