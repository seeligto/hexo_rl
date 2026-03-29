# Session Memory - March 29, 2026

## Progress Summary
- **Current Phase**: Phase 3.5 (Multi-Window Cluster Refactor) - **COMPLETED**.
- **Checklist Complete**:
  - [x] Rust core dynamic stone clustering (distance <= 8).
  - [x] Multi-window tensor generation (K tensors per state).
  - [x] ResNet-10 single-trunk batching (K clusters evaluated as one batch).
  - [x] Value min-pooling for pessimistic threat detection across colonies.
  - [x] Global Policy Mapping for infinite board coordination.
  - [x] RamoraBot bounds constraint removed.
  - [x] `requirements.txt` added and setup instructions updated.

## Benchmarks (RTX 3070)
- **MCTS Throughput**: ~263,000 sim/s (Target >= 10,000)
- **NN Inference (batch=64)**: ~52,000 pos/s (Target >= 5,000)
- **NN Latency (batch=1)**: 0.65ms (Target <= 5ms)
- **Replay Buffer (batch=256)**: 579μs (Target <= 1,000μs)
- **GPU Util**: 93%

## Architectural Decisions
- Abandoned Foveated Vision (Dual-Resolution) in favor of **Multi-Window Clustering**.
- Instead of scaling the entire board into a low-res tensor, we now identify distinct "colonies" (clusters of stones within 8 hex-distance) and evaluate each as a high-res 19x19 window.
- The worst-case value across all clusters (min-pooling) is used as the state value, ensuring the network is extremely sensitive to lethal threats in any cluster.
- Policies from different clusters are mapped back to a unified global policy for MCTS.

## Next Tasks
1. **Resume Phase 3 Bootstrap**: Run `python -m python.bootstrap.pretrain --force-regenerate` to build the new corpus in the multi-window format.
2. **Transition to Phase 4**: Implement the Elo ladder and automated tournament runner.
3. **Hyperparameter Tuning**: Evaluate if `K` clusters significantly increase VRAM during large-batch self-play.
