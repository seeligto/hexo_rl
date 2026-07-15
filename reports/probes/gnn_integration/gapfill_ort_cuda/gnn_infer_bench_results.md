# GNN inference bench — raw results

Hardware: laptop RTX 4060 Max-Q Ada Lovelace (sm_89) — NOT the authoritative 5080 verdict run; script is runnable unchanged on vast  
GPU: NVIDIA GeForce RTX 4060 Laptop GPU  
torch 2.11.0+cu130, onnxruntime 1.27.0  
Generated: 2026-07-14T16:58:53.921735+00:00

## Position provenance

- files (primary): ['/home/timmy/Work/Hexo/hexo_rl/logs/replays/games_2026-07-10.jsonl', '/home/timmy/Work/Hexo/hexo_rl/logs/replays/games_2026-07-12.jsonl', '/home/timmy/Work/Hexo/hexo_rl/logs/replays/games_2026-07-14.jsonl']
- files (fallback, June): []
- candidate pool: 1380 positions across 10 games
- sampled: 320 (requested 320)
- ply range in pool: [2, 149]
- seed: 42

## Rust builder proxy (hexo-strix axis_graph.rs)

- SKIPPED: --skip-rust or --cells

## ONNX export + parity gate

### scale=probe: OK
- parity: max|Δ|=5.484e-06 over 24 real graphs (threshold 0.0001) -> PASS

### scale=prod: OK
- parity: max|Δ|=6.557e-07 over 24 real graphs (threshold 0.0001) -> PASS

## Cell timing tables (median ms; IQR shown as ±)

### gnn_ort | scale=probe | device=cuda | precision=fp32 | backend=ort

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 128 | 111.648 | 30.689 | 1146 | 100.0 (n=2) | True |
| 256 | 214.662 | 7.061 | 1193 | 99.7 (n=3) | False |

### gnn_ort | scale=prod | device=cuda | precision=fp32 | backend=ort

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 128 | 216.360 | 31.480 | 592 | 100.0 (n=3) | False |

## Notes / warnings

- ORT-CUDA cell FAILED-SKIPPED scale=prod bs=256: Fail: [ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Expand node. Name:'/representation/convs.0/Expand' Status Message: /onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:358 void* onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*) Available memory of 842030592 is smaller than requested bytes of 1525481472

