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
