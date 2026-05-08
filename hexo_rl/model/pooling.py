"""§169 A2 / A3 — K-cluster pool modules.

Aggregates K shared-weight cluster-window forwards into a single
``(log_policy, value)`` per board state. Three pool types share the same
``forward(cluster_tokens, per_cluster_logits, per_cluster_values, *,
global_token=None)`` interface so the model and ``KClusterMCTSBot`` can
dispatch on ``pool_type`` without shape branches.

  - ``MinMaxPool`` — stateless ``value=min(K)`` + ``policy=scatter-max(K)`` in
    prob space, mirroring the engine ``records::aggregate_policy`` semantics
    (``hexo_rl.eval.k_cluster_mcts_bot._aggregate_priors``). No learnable
    parameters; output is a deterministic function of the per-cluster head
    outputs. Cluster-token features are accepted but ignored.

  - ``PMAPool`` — Set-Transformer style pool (Lee 2019). One Self-Attention
    Block (SAB) over the K cluster tokens, then two PMA seeds — one for
    value, one for policy — each cross-attending to the SAB-output tokens.
    Each seed feeds an MLP: value → tanh-bounded scalar; policy → per-action
    logits (the spec's "scatter logits back to per-cluster spatial
    positions" — output is the per-window n_actions vector, replacing the
    bot-side scatter-max). Cluster head outputs are NOT consumed by the pool
    — PMA learns its own value/policy heads atop the trunk feature pool.

  - ``PMAGlobalPool`` (§169 A3) — PMAPool extended with a global summary
    token g. The augmented set ``S = {z_1, ..., z_K, g}`` is fed into the
    SAB block before the two PMA seeds. ``g`` is gated by a learned scalar
    (init 0.1) so the network starts effectively in the K-cluster regime
    and earns global weight only as evidence accumulates; the ``.gate_value()``
    accessor returns the scalar for training-time logging. The global token
    enters via the ``global_token=`` kwarg on ``forward``.

PMA collapse mitigation: ``attn_dropout`` (default 0.1) shows up only in
training; the §169 surfacing protocol's PMA-collapse smoke runs in eval
mode where dropout is identity. If the eval smoke shows collapse, retry
with ``attn_dropout=0.2`` per the user's hard-stop / retry policy.

A3 has an additional collapse mode — PMA collapsing onto the global token
(network ignores all K clusters, copies g). The §169 P3 surfacing protocol
catches this via the same 2-cluster fixture on the inference path: if
argmax is identical regardless of cluster content, surface STOP.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_POOL_TYPES = ("min_max", "pma", "pma_global")


def build_pool(pool_type: str, *, dim: int, n_actions: int, **kwargs) -> "ClusterPool":
    """Construct a pool by name. Raises ValueError on unknown ``pool_type``.

    Extra kwargs are forwarded to the constructor of the chosen pool. A
    registry pattern is used so adding pools is a one-line edit here.
    """
    if pool_type == "min_max":
        return MinMaxPool()
    if pool_type == "pma":
        return PMAPool(dim=dim, n_actions=n_actions, **kwargs)
    if pool_type == "pma_global":
        return PMAGlobalPool(dim=dim, n_actions=n_actions, **kwargs)
    raise ValueError(
        f"unknown pool_type={pool_type!r}; supported: {SUPPORTED_POOL_TYPES}"
    )


class ClusterPool(nn.Module):
    """Base class for K-cluster pool modules.

    Subclasses must implement ``forward(cluster_tokens, per_cluster_logits,
    per_cluster_values, *, global_token=None) -> (log_policy, value,
    value_logit)``. ``value`` is tanh-bounded; ``value_logit`` is the
    pre-tanh raw logit needed by the BCE-based value loss in
    ``hexo_rl.training.losses.compute_value_loss``. ``global_token`` is the
    A3-only optional ``(B, dim)`` global summary embedding; pools without
    a global branch (MinMaxPool, PMAPool) accept the kwarg for interface
    parity and ignore it.
    """

    def forward(
        self,
        cluster_tokens: torch.Tensor,        # (B, K, dim)
        per_cluster_logits: torch.Tensor,    # (B, K, n_actions) raw logits
        per_cluster_values: torch.Tensor,    # (B, K, 1)
        *,
        global_token: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MinMaxPool(ClusterPool):
    """Stateless reduction matching the bot-side scatter-max + value-min."""

    def forward(
        self,
        cluster_tokens: torch.Tensor,
        per_cluster_logits: torch.Tensor,
        per_cluster_values: torch.Tensor,
        *,
        global_token: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del global_token  # interface parity; min/max has no global branch.
        # ``per_cluster_values`` are post-tanh in the existing model; we treat
        # them as the value to pool and recover a pre-tanh proxy via atanh
        # (clamped to keep gradients well-behaved).
        value = per_cluster_values.min(dim=1).values             # (B, 1)
        value_logit = torch.atanh(value.clamp(-0.999999, 0.999999))

        # Policy: per-cluster softmax → max across K → renormalise → log.
        # Done in prob space so different clusters' log-prob shifts compose
        # correctly. Mirrors ``KClusterMCTSBot._aggregate_priors``.
        probs_K = F.softmax(per_cluster_logits, dim=-1)         # (B, K, A)
        max_probs = probs_K.max(dim=1).values                    # (B, A)
        max_probs = max_probs.clamp_min(1e-12)
        max_probs = max_probs / max_probs.sum(dim=-1, keepdim=True)
        log_policy = max_probs.log()                             # (B, A)
        return log_policy, value, value_logit


class _SAB(nn.Module):
    """One Set-Attention Block: pre-LN MHA + pre-LN FFN, residual on each."""

    def __init__(
        self, dim: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.ReLU(),
            nn.Linear(ff_mult * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        h = self.ln2(x)
        return x + self.ff(h)


class _PMA(nn.Module):
    """Pooling-by-Multihead-Attention with a single learnable seed.

    Lee 2019 Lemma 3 — PMA generalises any permutation-invariant pool via a
    seed query that cross-attends to the input tokens. One seed = one
    aggregated output; we use one PMA per head (value, policy).
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ln_out = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, dim) — SAB output.
        B = x.size(0)
        q = self.ln_q(self.seed.expand(B, -1, -1))               # (B, 1, dim)
        kv = self.ln_kv(x)                                        # (B, K, dim)
        a, _ = self.attn(q, kv, kv, need_weights=False)           # (B, 1, dim)
        h = q + a
        return (h + self.ff(self.ln_out(h))).squeeze(1)           # (B, dim)


class PMAPool(ClusterPool):
    """Set-Transformer pool: 1×SAB over K cluster tokens, then 2 PMA seeds.

    Args:
        dim:        cluster-token feature dim (matches trunk filter count).
        n_heads:    attention heads in SAB / PMA. Spec: 4.
        n_actions:  policy output size (cluster-window cells + pass slot).
        attn_dropout: dropout on attention weights — applies under .train()
                    only. Default 0.1; bump to 0.2 if PMA-collapse smoke fires.
    """

    def __init__(
        self,
        dim: int = 128,
        n_heads: int = 4,
        n_actions: int = 626,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_actions = int(n_actions)
        self.sab = _SAB(self.dim, n_heads, dropout=attn_dropout)
        self.value_pma = _PMA(self.dim, n_heads, dropout=attn_dropout)
        self.policy_pma = _PMA(self.dim, n_heads, dropout=attn_dropout)
        # Value head MLP — mirrors network.py's value_fc1/value_fc2 sizing
        # but consumes the PMA-aggregated single vector.
        self.value_mlp = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        # Policy head MLP — projects PMA seed output to per-action logits.
        # The spec's "scatter logits back to per-cluster spatial positions"
        # is implemented here as a direct linear over the n_actions output.
        self.policy_mlp = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.ReLU(),
            nn.Linear(4 * self.dim, self.n_actions),
        )

    def forward(
        self,
        cluster_tokens: torch.Tensor,                  # (B, K, dim)
        per_cluster_logits: Optional[torch.Tensor] = None,
        per_cluster_values: Optional[torch.Tensor] = None,
        *,
        global_token: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # PMA does NOT consume per-cluster head outputs — it builds its own
        # value/policy heads on top of the SAB-aggregated tokens. The args
        # are accepted for interface parity with MinMaxPool / PMAGlobalPool.
        del per_cluster_logits, per_cluster_values, global_token

        x = self.sab(cluster_tokens)                              # (B, K, d)
        v_pool = self.value_pma(x)                                # (B, d)
        p_pool = self.policy_pma(x)                               # (B, d)

        v_logit = self.value_mlp(v_pool)                          # (B, 1)
        value = torch.tanh(v_logit)
        p_logits = self.policy_mlp(p_pool)                        # (B, A)
        log_policy = F.log_softmax(p_logits, dim=-1)
        return log_policy, value, v_logit


class PMAGlobalPool(ClusterPool):
    """§169 A3 — Set-Transformer pool with a gated global summary token.

    The augmented set ``S = {z_1, ..., z_K, g}`` (K+1 elements) feeds the
    SAB block before the value / policy PMA seeds. ``g`` arrives via the
    ``global_token=`` kwarg; a learnable scalar (``global_gate``, init
    ``gate_init``) multiplies it before concatenation so the network starts
    in the K-cluster regime and the global branch only earns weight as
    evidence accumulates. ``gate_value()`` exposes the scalar for
    training-time logging — surfacing whether g was actually used.

    Wire shape: cluster_tokens (B, K, dim), global_token (B, dim) → SAB
    sees (B, K+1, dim). The two PMA seeds collapse to (B, dim) each,
    feeding the same value_mlp / policy_mlp shapes as PMAPool.

    Args mirror PMAPool, plus:
        gate_init: initial value of the learnable scalar gate on ``g``.
                   Default 0.1 — small enough that A3 starts close to the
                   K-cluster baseline but the gradient can grow it.
    """

    def __init__(
        self,
        dim: int = 128,
        n_heads: int = 4,
        n_actions: int = 626,
        attn_dropout: float = 0.1,
        gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_actions = int(n_actions)
        self.sab = _SAB(self.dim, n_heads, dropout=attn_dropout)
        self.value_pma = _PMA(self.dim, n_heads, dropout=attn_dropout)
        self.policy_pma = _PMA(self.dim, n_heads, dropout=attn_dropout)
        self.value_mlp = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.policy_mlp = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.ReLU(),
            nn.Linear(4 * self.dim, self.n_actions),
        )
        # Scalar gate. Stored as a (1,) parameter so state_dict load/save
        # round-trips trivially. f32 by default — autocast handles fp16 paths.
        self.global_gate = nn.Parameter(torch.tensor([gate_init], dtype=torch.float32))

    def forward(
        self,
        cluster_tokens: torch.Tensor,                  # (B, K, dim)
        per_cluster_logits: Optional[torch.Tensor] = None,
        per_cluster_values: Optional[torch.Tensor] = None,
        *,
        global_token: Optional[torch.Tensor] = None,   # (B, dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del per_cluster_logits, per_cluster_values
        if global_token is None:
            raise ValueError(
                "PMAGlobalPool requires global_token=...; got None. The model's "
                "forward path must compute the global summary token via "
                "GlobalTokenEncoder before invoking this pool."
            )
        if global_token.dim() != 2 or global_token.size(1) != self.dim:
            raise ValueError(
                f"global_token must be (B, dim={self.dim}); got shape "
                f"{tuple(global_token.shape)}"
            )
        if global_token.size(0) != cluster_tokens.size(0):
            raise ValueError(
                f"global_token batch dim {global_token.size(0)} disagrees with "
                f"cluster_tokens batch dim {cluster_tokens.size(0)}"
            )

        gated = self.global_gate.to(global_token.dtype) * global_token  # (B, dim)
        g_token = gated.unsqueeze(1)                                    # (B, 1, dim)
        # Concatenate K+1 tokens: clusters first, global last.
        x = torch.cat([cluster_tokens, g_token], dim=1)                 # (B, K+1, dim)
        x = self.sab(x)
        v_pool = self.value_pma(x)
        p_pool = self.policy_pma(x)

        v_logit = self.value_mlp(v_pool)
        value = torch.tanh(v_logit)
        p_logits = self.policy_mlp(p_pool)
        log_policy = F.log_softmax(p_logits, dim=-1)
        return log_policy, value, v_logit

    def gate_value(self) -> float:
        """Return the current scalar gate value as a Python float.

        Used by the training loop to surface whether the global branch
        earned weight or stayed near init. ``detach()`` so gradient
        graph isn't held; ``.cpu().item()`` for scalar extraction.
        """
        return float(self.global_gate.detach().cpu().item())
