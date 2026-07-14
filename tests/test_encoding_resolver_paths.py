import pytest


def test_resolve_corpus_path_v6w25():
    from hexo_rl.encoding import lookup, resolve_corpus_path
    spec = lookup("v6w25")
    p = resolve_corpus_path(spec)
    assert str(p).endswith("data/bootstrap_corpus_v6w25.npz")


def test_resolve_corpus_path_v8():
    from hexo_rl.encoding import lookup, resolve_corpus_path
    spec = lookup("v8")
    p = resolve_corpus_path(spec)
    assert str(p).endswith("data/bootstrap_corpus_v8.npz")


def test_resolve_corpus_path_v6():
    from hexo_rl.encoding import lookup, resolve_corpus_path
    spec = lookup("v6")
    p = resolve_corpus_path(spec)
    assert str(p).endswith("data/bootstrap_corpus.npz")


def test_resolve_anchor_path_v6w25():
    from hexo_rl.encoding import lookup, resolve_anchor_path
    spec = lookup("v6w25")
    p = resolve_anchor_path(spec)
    assert str(p).endswith("checkpoints/bootstrap_model_v6w25.pt")


def test_resolve_anchor_path_v7full():
    from hexo_rl.encoding import lookup, resolve_anchor_path
    spec = lookup("v7full")
    p = resolve_anchor_path(spec)
    assert str(p).endswith("checkpoints/bootstrap_model_v7full.pt")


def test_resolve_corpus_path_unknown_encoding_raises():
    from hexo_rl.encoding import resolve_corpus_path, EncodingRegistryError

    class FakeSpec:
        name = "v999"

    with pytest.raises(EncodingRegistryError):
        resolve_corpus_path(FakeSpec())


def test_resolve_anchor_path_unknown_encoding_raises():
    from hexo_rl.encoding import resolve_anchor_path, EncodingRegistryError

    class FakeSpec:
        name = "v999"

    with pytest.raises(EncodingRegistryError):
        resolve_anchor_path(FakeSpec())


def test_expand_auto_paths_flat():
    """Flat form: corpus_npz / bootstrap_anchor at config top level."""
    from hexo_rl.encoding import lookup, expand_auto_paths
    spec = lookup("v6w25")
    cfg = {
        "corpus_npz": "<auto>",
        "bootstrap_anchor": "<auto>",
        "unrelated": "preserved",
    }
    expand_auto_paths(cfg, spec)
    assert cfg["corpus_npz"] == "data/bootstrap_corpus_v6w25.npz"
    assert cfg["bootstrap_anchor"] == "checkpoints/bootstrap_model_v6w25.pt"
    assert cfg["unrelated"] == "preserved"


def test_expand_auto_paths_nested():
    """Nested form: mixing.pretrained_buffer_path + eval_pipeline.opponents.bootstrap_anchor.path."""
    from hexo_rl.encoding import lookup, expand_auto_paths
    spec = lookup("v6w25")
    cfg = {
        "mixing": {"pretrained_buffer_path": "<auto>"},
        "eval_pipeline": {
            "opponents": {
                "bootstrap_anchor": {"path": "<auto>"}
            }
        },
    }
    expand_auto_paths(cfg, spec)
    assert cfg["mixing"]["pretrained_buffer_path"] == "data/bootstrap_corpus_v6w25.npz"
    assert cfg["eval_pipeline"]["opponents"]["bootstrap_anchor"]["path"] == "checkpoints/bootstrap_model_v6w25.pt"


def test_expand_auto_paths_no_auto_is_noop():
    """No <auto> markers → config dict unchanged."""
    from hexo_rl.encoding import lookup, expand_auto_paths
    spec = lookup("v6")
    cfg = {
        "corpus_npz": "data/explicit_path.npz",
        "bootstrap_anchor": "checkpoints/explicit_anchor.pt",
        "other": {"deeply": {"nested": "value"}},
    }
    import copy
    snapshot = copy.deepcopy(cfg)
    expand_auto_paths(cfg, spec)
    assert cfg == snapshot, "expand_auto_paths must not mutate non-<auto> values"


def test_expand_auto_paths_partial_replacement():
    """One <auto>, one explicit — only the <auto> expands."""
    from hexo_rl.encoding import lookup, expand_auto_paths
    spec = lookup("v8")
    cfg = {
        "corpus_npz": "<auto>",
        "bootstrap_anchor": "checkpoints/custom.pt",
    }
    expand_auto_paths(cfg, spec)
    assert cfg["corpus_npz"] == "data/bootstrap_corpus_v8.npz"
    assert cfg["bootstrap_anchor"] == "checkpoints/custom.pt"


# ---------------------------------------------------------------------------
# CG23 — drift guard: resolver dicts must cover every registered encoding
# (or be explicitly allowlisted as operator-curated not-yet-shipped).
# P62 / bucket 07. §176 P2.
# ---------------------------------------------------------------------------

# Encodings registered in registry.toml that have no corpus/anchor yet.
# These are intentional gaps (v7/v7e30 retired before resolver entries landed).
# Add entries to _CORPUS_PATHS / _ANCHOR_PATHS in hexo_rl/encoding/resolvers.py
# and remove from this allowlist when the artifacts ship.
_RESOLVER_GAPS_ALLOWLIST: frozenset[str] = frozenset({"v7", "v7e30"})


def test_corpus_paths_covers_all_registered_encodings() -> None:
    """Adding a registered encoding without a resolver corpus path
    silently breaks dataset loading. Guard against future drift.

    CG23 in bucket 07.
    """
    from hexo_rl.encoding import all_specs
    from hexo_rl.encoding.resolvers import _CORPUS_PATHS

    registered = {s.name for s in all_specs()}
    pathed = set(_CORPUS_PATHS.keys())
    missing = registered - pathed - _RESOLVER_GAPS_ALLOWLIST
    assert not missing, (
        f"Encodings registered but missing corpus paths: {missing}. "
        "Add entries to _CORPUS_PATHS in hexo_rl/encoding/resolvers.py "
        "or add to _RESOLVER_GAPS_ALLOWLIST if corpus intentionally absent."
    )


def test_anchor_paths_covers_all_registered_encodings() -> None:
    """Same drift guard for bootstrap anchor paths.

    CG23 in bucket 07.
    """
    from hexo_rl.encoding import all_specs
    from hexo_rl.encoding.resolvers import _ANCHOR_PATHS

    registered = {s.name for s in all_specs()}
    pathed = set(_ANCHOR_PATHS.keys())
    missing = registered - pathed - _RESOLVER_GAPS_ALLOWLIST
    assert not missing, (
        f"Encodings registered but missing anchor paths: {missing}. "
        "Add entries to _ANCHOR_PATHS in hexo_rl/encoding/resolvers.py "
        "or add to _RESOLVER_GAPS_ALLOWLIST if anchor intentionally absent."
    )


def test_resolve_corpus_sha_pin_v6_live2_ls():
    """WP0.4 — run3 launch corpus is sha-pinned in the registry resolver."""
    from hexo_rl.encoding import lookup
    from hexo_rl.encoding.resolvers import resolve_corpus_sha_pin
    spec = lookup("v6_live2_ls")
    assert resolve_corpus_sha_pin(spec) == (
        "3813edc2fb10a7c5ab976a0293e38cbba0fd6b84e5295630f339ca421b345c97"
    )


def test_resolve_corpus_sha_pin_unpinned_encoding_returns_none():
    """Most encodings have no launch pin registered — None, not an error."""
    from hexo_rl.encoding import lookup
    from hexo_rl.encoding.resolvers import resolve_corpus_sha_pin
    spec = lookup("v6")
    assert resolve_corpus_sha_pin(spec) is None


def test_resolver_disambiguates_v6_live2_ls_from_v6_live2():
    """§D-MULTICLUSTER-S0 / §9.10 — v6_live2_ls is shape-identical to v6_live2
    (in_ch=4, 362, 19×19); only the ckpt LABEL disambiguates. The more-specific
    "v6_live2_ls" label must resolve to the TREATMENT encoding, NOT silently to
    the CONTROL (which would false-clear both A/B axes). v6_live2 labels and
    shape-only resolution must still map to v6_live2 (no regression)."""
    from hexo_rl.model.network import HexTacToeNet
    from hexo_rl.encoding.resolvers import detect_encoding_from_state_dict as detect

    sd = HexTacToeNet(
        encoding="v6", board_size=19, in_channels=4, filters=8, res_blocks=1
    ).state_dict()

    assert detect(sd, "checkpoint_v6_live2_ls_50k.pt", strict=True).name == "v6_live2_ls"
    assert detect(sd, "checkpoint_00050000_PEAK_sb0.38.pt", strict=True).name == "v6_live2"
    assert detect(sd, "bootstrap_model_v6_live2.pt", strict=True).name == "v6_live2"
