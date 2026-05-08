"""§169 A2 / A3 — K-cluster pool module unit tests.

The K-cluster pool aggregates K shared-weight cluster-window forwards into
a single (log_policy, value) output. Three implementations live in
``hexo_rl/model/pooling.py``:

  - ``MinMaxPool`` — stateless reduction (value=min over K, policy=
    scatter-max-then-renormalise over K). Mirrors the engine
    ``records::aggregate_policy`` / ``KClusterMCTSBot._aggregate_priors``
    semantics, lifted into the model so pool_type='min_max' / 'pma' /
    'pma_global' share an identical wire interface.
  - ``PMAPool`` — Set-Transformer style 1×SAB + 2 PMA seeds (value + policy)
    with attention dropout for collapse mitigation. d=128, 4 heads.
  - ``PMAGlobalPool`` (§169 A3) — PMAPool + a learned-scalar-gated global
    summary token. ``global_token=`` kwarg is required at forward; SAB sees
    K+1 tokens.

Invariants asserted here:
  (a) shape parity with min/max for K∈{1,2,4,6} — both pools produce
      (B, n_actions) log-policy + (B, 1) value at any K;
  (b) gradient sanity — backprop through PMAPool reaches every learnable
      parameter (no orphaned tensors);
  (c) PMAPool is well-defined at K=1 — single-token attention pool returns
      finite outputs, deterministic in eval mode;
  (d) state-dict round-trip — module saves/loads byte-exact and forward
      outputs match across the round-trip;
  (e) PMAGlobalPool — gate-init scalar surfaces via gate_value(); requires
      global_token; gate=0 reduces to PMAPool-like behavior on the K
      tokens; gradient reaches global_gate.
"""
from __future__ import annotations

import pytest
import torch

from hexo_rl.model.pooling import MinMaxPool, PMAGlobalPool, PMAPool, build_pool


# ── Fixtures ────────────────────────────────────────────────────────────


def _random_inputs(B: int, K: int, dim: int, n_actions: int, seed: int = 0):
    """Random per-cluster pool inputs.

    Returns:
      cluster_tokens   : (B, K, dim)        per-cluster GAP'd trunk features
      per_cluster_logp : (B, K, n_actions)  per-cluster raw policy logits
      per_cluster_value: (B, K, 1)          per-cluster value scalars
    """
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(B, K, dim, generator=g),
        torch.randn(B, K, n_actions, generator=g),
        torch.randn(B, K, 1, generator=g),
    )


# ── (a) Shape parity for K ∈ {1, 2, 4, 6} ───────────────────────────────


def test_minmax_shape_parity_across_K():
    pool = MinMaxPool()
    n_actions, dim = 626, 128
    for K in (1, 2, 4, 6):
        toks, logp_K, val_K = _random_inputs(B=3, K=K, dim=dim, n_actions=n_actions)
        log_p, val, v_logit = pool(toks, logp_K, val_K)
        _ = v_logit
        assert log_p.shape == (3, n_actions), f"K={K}: log_p {log_p.shape}"
        assert val.shape == (3, 1), f"K={K}: val {val.shape}"


def test_pma_shape_parity_across_K():
    torch.manual_seed(0)
    n_actions, dim = 626, 128
    pool = PMAPool(dim=dim, n_heads=4, n_actions=n_actions).eval()
    for K in (1, 2, 4, 6):
        toks, logp_K, val_K = _random_inputs(B=3, K=K, dim=dim, n_actions=n_actions)
        log_p, val, v_logit = pool(toks, logp_K, val_K)
        _ = v_logit
        assert log_p.shape == (3, n_actions), f"K={K}: log_p {log_p.shape}"
        assert val.shape == (3, 1), f"K={K}: val {val.shape}"


def test_pool_output_value_range():
    """Value head must be tanh-bounded; log_policy a valid log-softmax."""
    torch.manual_seed(0)
    pool = PMAPool(dim=64, n_heads=4, n_actions=50).eval()
    toks, logp_K, val_K = _random_inputs(B=2, K=3, dim=64, n_actions=50)
    log_p, val, v_logit = pool(toks, logp_K, val_K)
    assert torch.all(val.abs() <= 1.0 + 1e-6), f"value out of [-1, 1]: {val}"
    assert v_logit.shape == val.shape, "value_logit shape must match value"
    # log_softmax over a single normalised distribution sums to log(1) = 0.
    sums = log_p.exp().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
        f"log_p does not normalise: sums={sums}"


# ── (b) Gradient sanity ─────────────────────────────────────────────────


def test_pma_gradient_reaches_all_params():
    """Backprop through PMAPool must touch every learnable parameter."""
    torch.manual_seed(0)
    pool = PMAPool(dim=64, n_heads=4, n_actions=20)
    toks, logp_K, val_K = _random_inputs(B=2, K=4, dim=64, n_actions=20)
    log_p, val, _ = pool(toks, logp_K, val_K)
    # Combine both heads in the loss so gradients flow to both seeds + both MLPs.
    loss = log_p.sum() + val.sum()
    loss.backward()
    missing: list[str] = []
    for name, p in pool.named_parameters():
        if p.grad is None or torch.all(p.grad == 0):
            missing.append(name)
    assert not missing, f"parameters with no gradient: {missing}"


# ── (c) PMA at K=1 is deterministic + finite ───────────────────────────


def test_pma_k1_is_well_defined_and_deterministic():
    """K=1 path must produce finite outputs and (eval-mode) be deterministic
    across repeated calls. SAB with a single token is the layer-norm + MLP
    of that token; PMA cross-attention with one key/value is a learned
    affine pass — both must run without shape errors at K=1."""
    torch.manual_seed(0)
    pool = PMAPool(dim=32, n_heads=4, n_actions=10, attn_dropout=0.1).eval()
    toks, logp_K, val_K = _random_inputs(B=2, K=1, dim=32, n_actions=10)
    log_p1, val1, _ = pool(toks, logp_K, val_K)
    log_p2, val2, _ = pool(toks, logp_K, val_K)
    assert torch.isfinite(log_p1).all() and torch.isfinite(val1).all()
    assert torch.allclose(log_p1, log_p2), "PMAPool not deterministic in eval"
    assert torch.allclose(val1, val2), "PMAPool value not deterministic in eval"


def test_pma_invariant_to_duplicate_clusters():
    """Sanity: PMA(stack([x])) ~ PMA(stack([x, x])) up to attention scale.
    Pure permutation invariance is trivial; duplication invariance is
    a stronger sanity check that the seed reads cluster content, not order.
    Tolerance is loose because LayerNorm renormalises over the K dim
    indirectly through residual scales."""
    torch.manual_seed(0)
    pool = PMAPool(dim=32, n_heads=4, n_actions=10).eval()
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 1, 32, generator=g)
    logp = torch.randn(1, 1, 10, generator=g)
    val = torch.randn(1, 1, 1, generator=g)
    out1_lp, out1_v, _ = pool(x, logp, val)
    out2_lp, out2_v, _ = pool(x.repeat(1, 2, 1), logp.repeat(1, 2, 1), val.repeat(1, 2, 1))
    # Cross-attention with identical keys/values gives the same query response;
    # SAB is permutation-equivariant so duplication is a fixed-point input.
    assert torch.allclose(out1_lp, out2_lp, atol=1e-4), \
        f"PMA differs under cluster duplication: max |Δ|={(out1_lp-out2_lp).abs().max()}"
    assert torch.allclose(out1_v, out2_v, atol=1e-4)


# ── (d) State-dict round-trip ──────────────────────────────────────────


def test_pma_state_dict_round_trip():
    torch.manual_seed(0)
    a = PMAPool(dim=64, n_heads=4, n_actions=50).eval()
    b = PMAPool(dim=64, n_heads=4, n_actions=50).eval()
    # Pre-load: outputs must differ (different random init).
    toks, logp_K, val_K = _random_inputs(B=1, K=3, dim=64, n_actions=50)
    lp_a, _, _ = a(toks, logp_K, val_K)
    lp_b_pre, _, _ = b(toks, logp_K, val_K)
    assert not torch.allclose(lp_a, lp_b_pre), "init seeds collided"

    # Round-trip through state_dict.
    sd = a.state_dict()
    b.load_state_dict(sd)
    lp_b_post, _, _ = b(toks, logp_K, val_K)
    assert torch.allclose(lp_a, lp_b_post, atol=1e-6), \
        "state_dict round-trip changed outputs"


def test_minmax_no_state():
    """MinMaxPool must have no learnable parameters — pure reduction."""
    pool = MinMaxPool()
    assert sum(p.numel() for p in pool.parameters()) == 0, \
        "MinMaxPool unexpectedly has learnable parameters"


# ── Registry ────────────────────────────────────────────────────────────


def test_build_pool_registry():
    """build_pool dispatches on pool_type and rejects unknowns."""
    pool_mm = build_pool("min_max", dim=128, n_actions=10)
    assert isinstance(pool_mm, MinMaxPool)
    pool_pma = build_pool("pma", dim=128, n_actions=10)
    assert isinstance(pool_pma, PMAPool)
    pool_pg = build_pool("pma_global", dim=128, n_actions=10)
    assert isinstance(pool_pg, PMAGlobalPool)
    try:
        build_pool("bogus", dim=128, n_actions=10)
    except ValueError as exc:
        assert "pool_type" in str(exc).lower()
    else:
        raise AssertionError("build_pool accepted unknown pool_type")


# ── §169 A3 — PMAGlobalPool ─────────────────────────────────────────────


def _random_global_token(B: int, dim: int, seed: int = 17) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, dim, generator=g)


def test_pma_global_shape_parity_across_K():
    torch.manual_seed(0)
    n_actions, dim = 626, 64
    pool = PMAGlobalPool(dim=dim, n_heads=4, n_actions=n_actions).eval()
    for K in (1, 2, 4, 6):
        toks, logp_K, val_K = _random_inputs(B=3, K=K, dim=dim, n_actions=n_actions)
        g = _random_global_token(B=3, dim=dim)
        log_p, val, v_logit = pool(toks, logp_K, val_K, global_token=g)
        assert log_p.shape == (3, n_actions), f"K={K}: log_p {log_p.shape}"
        assert val.shape == (3, 1), f"K={K}: val {val.shape}"
        assert v_logit.shape == (3, 1)


def test_pma_global_requires_global_token():
    """Forward without global_token must raise — silent fallback would
    mask a wiring bug at the model level."""
    pool = PMAGlobalPool(dim=32, n_heads=4, n_actions=10).eval()
    toks, logp, val = _random_inputs(B=1, K=2, dim=32, n_actions=10)
    with pytest.raises(ValueError, match="global_token"):
        pool(toks, logp, val)


def test_pma_global_rejects_shape_mismatch():
    """Wrong dim or batch must surface loudly."""
    pool = PMAGlobalPool(dim=32, n_heads=4, n_actions=10).eval()
    toks, logp, val = _random_inputs(B=2, K=2, dim=32, n_actions=10)
    bad_dim = torch.zeros(2, 64)
    with pytest.raises(ValueError, match=r"\(B, dim"):
        pool(toks, logp, val, global_token=bad_dim)
    bad_batch = torch.zeros(3, 32)
    with pytest.raises(ValueError, match="batch"):
        pool(toks, logp, val, global_token=bad_batch)


def test_pma_global_gate_surfaces_init_value():
    """gate_value() exposes the learned scalar; gate_init=0.1 must report 0.1."""
    pool = PMAGlobalPool(dim=16, n_heads=2, n_actions=5, gate_init=0.1).eval()
    assert abs(pool.gate_value() - 0.1) < 1e-6
    pool2 = PMAGlobalPool(dim=16, n_heads=2, n_actions=5, gate_init=0.5).eval()
    assert abs(pool2.gate_value() - 0.5) < 1e-6


def test_pma_global_gradient_reaches_gate_and_all_params():
    """Backprop through global_gate must produce a non-zero grad — if the
    gate is dead the global branch can't earn weight."""
    torch.manual_seed(0)
    pool = PMAGlobalPool(dim=32, n_heads=4, n_actions=10, gate_init=0.1)
    toks, logp, val = _random_inputs(B=2, K=3, dim=32, n_actions=10)
    g = _random_global_token(B=2, dim=32)
    log_p, value, _ = pool(toks, logp, val, global_token=g)
    loss = log_p.sum() + value.sum()
    loss.backward()
    assert pool.global_gate.grad is not None
    assert torch.any(pool.global_gate.grad != 0), "gate has no gradient signal"
    missing = []
    for name, p in pool.named_parameters():
        if p.grad is None or torch.all(p.grad == 0):
            missing.append(name)
    assert not missing, f"params with no gradient: {missing}"


def test_pma_global_state_dict_round_trip():
    torch.manual_seed(0)
    a = PMAGlobalPool(dim=32, n_heads=4, n_actions=10).eval()
    b = PMAGlobalPool(dim=32, n_heads=4, n_actions=10).eval()
    toks, logp, val = _random_inputs(B=1, K=2, dim=32, n_actions=10)
    g = _random_global_token(B=1, dim=32)
    lp_a, _, _ = a(toks, logp, val, global_token=g)
    b.load_state_dict(a.state_dict())
    lp_b, _, _ = b(toks, logp, val, global_token=g)
    assert torch.allclose(lp_a, lp_b, atol=1e-6)


def test_pma_global_zero_gate_isolates_cluster_path():
    """When global_gate is forced to 0, the global token contributes the
    zero vector to the SAB. The output should match the PMA-only path
    (run on K cluster tokens + a zero (K+1)st token) — which is a sanity
    invariant for the gate-as-isolation knob."""
    torch.manual_seed(0)
    pool = PMAGlobalPool(dim=32, n_heads=4, n_actions=10, gate_init=0.0).eval()
    toks, logp, val = _random_inputs(B=2, K=3, dim=32, n_actions=10)
    g = _random_global_token(B=2, dim=32)
    log_p_with_g, value_with_g, _ = pool(toks, logp, val, global_token=g)
    g_zero = torch.zeros_like(g)
    log_p_zero, value_zero, _ = pool(toks, logp, val, global_token=g_zero)
    # Gate is 0 ⇒ gated*g == 0 == gated*0, so the two paths produce identical SAB inputs.
    assert torch.allclose(log_p_with_g, log_p_zero, atol=1e-6)
    assert torch.allclose(value_with_g, value_zero, atol=1e-6)


# ── MinMaxPool numerical correctness ────────────────────────────────────


def test_minmax_value_is_min_across_K():
    pool = MinMaxPool()
    toks = torch.zeros(2, 3, 8)
    logp = torch.zeros(2, 3, 5)
    val = torch.tensor([[[0.5], [-0.2], [0.9]],
                        [[1.0], [0.0], [-1.0]]], dtype=torch.float32)
    _, v_out, _ = pool(toks, logp, val)
    assert torch.allclose(v_out, torch.tensor([[-0.2], [-1.0]]))


def test_minmax_policy_is_scatter_max_in_prob_space():
    pool = MinMaxPool()
    # K=2, n_actions=3. Cluster 0 hot on action 0; cluster 1 hot on action 2.
    # Scatter-max should yield a softmax over (high, low, high).
    toks = torch.zeros(1, 2, 4)
    logp = torch.tensor([[[10.0, 0.0, 0.0],
                          [0.0, 0.0, 10.0]]])
    val = torch.zeros(1, 2, 1)
    log_p, _, _ = pool(toks, logp, val)
    probs = log_p.exp()
    # Action 1 should be ~0; actions 0 and 2 should split mass.
    assert probs[0, 1] < 1e-3, f"action 1 unexpectedly hot: {probs[0, 1]}"
    assert probs[0, 0] > 0.4 and probs[0, 2] > 0.4, \
        f"actions 0/2 not split: {probs}"
