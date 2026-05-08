"""§169 A2 — K-cluster pool modules.

Aggregates K shared-weight cluster-window forwards into a single
``(log_policy, value)`` per board state. Two pool types share the same
``forward(cluster_tokens, per_cluster_logits, per_cluster_values)`` interface
so the model and ``KClusterMCTSBot`` can dispatch on ``pool_type`` without
shape branches.

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

PMA collapse mitigation: ``attn_dropout`` (default 0.1) shows up only in
training; the §169 surfacing protocol's PMA-collapse smoke runs in eval
mode where dropout is identity. If the eval smoke shows collapse, retry
with ``attn_dropout=0.2`` per the user's hard-stop / retry policy.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_POOL_TYPES = ("min_max", "pma")


def build_pool(pool_type: str, *, dim: int, n_actions: int, **kwargs) -> "ClusterPool":
    """Construct a pool by name. Raises ValueError on unknown ``pool_type``.

    Extra kwargs are forwarded to the constructor of the chosen pool. A
    registry pattern is used so adding A3/P3 pools (`pma_global` etc.) is a
    one-line edit here.
    """
    if pool_type == "min_max":
        return MinMaxPool()
    if pool_type == "pma":
        return PMAPool(dim=dim, n_actions=n_actions, **kwargs)
    raise ValueError(
        f"unknown pool_type={pool_type!r}; supported: {SUPPORTED_POOL_TYPES}"
    )


class ClusterPool(nn.Module):
    """Base class for K-cluster pool modules.

    Subclasses must implement ``forward(cluster_tokens, per_cluster_logits,
    per_cluster_values) -> (log_policy, value)``.
    """

    def forward(
        self,
        cluster_tokens: torch.Tensor,        # (B, K, dim)
        per_cluster_logits: torch.Tensor,    # (B, K, n_actions) raw logits
        per_cluster_values: torch.Tensor,    # (B, K, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MinMaxPool(ClusterPool):
    """Stateless reduction matching the bot-side scatter-max + value-min."""

    def forward(
        self,
        cluster_tokens: torch.Tensor,
        per_cluster_logits: torch.Tensor,
        per_cluster_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Value: min over K (negamax-conservative; matches engine).
        value = per_cluster_values.min(dim=1).values             # (B, 1)

        # Policy: per-cluster softmax → max across K → renormalise → log.
        # Done in prob space so different clusters' log-prob shifts compose
        # correctly. Mirrors ``KClusterMCTSBot._aggregate_priors``.
        probs_K = F.softmax(per_cluster_logits, dim=-1)         # (B, K, A)
        max_probs = probs_K.max(dim=1).values                    # (B, A)
        max_probs = max_probs.clamp_min(1e-12)
        max_probs = max_probs / max_probs.sum(dim=-1, keepdim=True)
        log_policy = max_probs.log()                             # (B, A)
        return log_policy, value


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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # PMA does NOT consume per-cluster head outputs — it builds its own
        # value/policy heads on top of the SAB-aggregated tokens. The args
        # are accepted for interface parity with MinMaxPool.
        del per_cluster_logits, per_cluster_values

        x = self.sab(cluster_tokens)                              # (B, K, d)
        v_pool = self.value_pma(x)                                # (B, d)
        p_pool = self.policy_pma(x)                               # (B, d)

        v_logit = self.value_mlp(v_pool)                          # (B, 1)
        value = torch.tanh(v_logit)
        p_logits = self.policy_mlp(p_pool)                        # (B, A)
        log_policy = F.log_softmax(p_logits, dim=-1)
        return log_policy, value
