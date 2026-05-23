# §S181-AUDIT Wave 1 — Track B / V-B aggregation

**Primary verdict: V-B-E**

**Routing action**: ESCALATE — no real-run launch (V-B-E)

Sources:
- B1 (`B1_results.json`) — per-source gradient attribution
- B2 (`B2_results.json`) — buffer position-class snapshots
- B3 (`B3_trunk_drift.json`) — trunk feature drift

## Decision-tree (LITERAL L13)

| check | fires | detail |
|---|:---:|---|
| V-B-A | no | {"max_mean_source_share_in_window"=0.5628784171062502, "max_source_by_mean"="uniform_self", "mean_shares"={"pretrain"=0.09245180837025263, "recent"=0.3446697745234969, "uniform_self"=0.5628784171062502}, "per_step_max_shares"={"pretrain"=0. |
| V-B-B | no | {"shares_mean"={"pretrain"=0.09245180837025263, "recent"=0.3446697745234969, "uniform_self"=0.5628784171062502}, "band"=[0.25, 0.45]} |
| V-B-C | no | {"deadline_step"=2000, "threshold"=0.5, "fire_step"=null, "fire_frac"=null, "max_observed_frac"=0.0984, "max_observed_step"=3000, "n_snapshots"=6} |
| V-B-D | no | {"step0_inter"=23.232159, "step1000_step"=1000, "step1000_inter"=19.458044, "ratio"=0.8375478146477907, "threshold"=0.5, "delta_from_target"=0} |

## Mechanism narrative

Per the routing table above, the dominant force on the value-head discrimination collapse is the channel implied by the primary verdict. The Wave 2 lever stack pivots on this verdict — see `audit/structural/REAL_RUN_RECIPE.md` §3 conditional table and Stage 3 of the Wave 2 dispatcher.

## Track D cross-reference

Cross-reference the primary verdict with `audit/structural/track_d_pipeline_regression.md` §4 smoking-gun candidates. (Aggregator does not auto-cross; produce manually after reading the verdict + Track D ranking — see `B_track_d_xref.md`.)

