# GNN inference bench — raw results

Hardware: laptop RTX 4060 Max-Q Ada Lovelace (sm_89) — NOT the authoritative 5080 verdict run; script is runnable unchanged on vast  
GPU: NVIDIA GeForce RTX 4060 Laptop GPU  
torch 2.11.0+cu130, onnxruntime 1.27.0  
Generated: 2026-07-14T16:08:12.940161+00:00

## Position provenance

- files (primary): ['/home/timmy/Work/Hexo/hexo_rl/logs/replays/games_2026-07-10.jsonl', '/home/timmy/Work/Hexo/hexo_rl/logs/replays/games_2026-07-12.jsonl', '/home/timmy/Work/Hexo/hexo_rl/logs/replays/games_2026-07-14.jsonl']
- files (fallback, June): []
- candidate pool: 1380 positions across 10 games
- sampled: 320 (requested 320)
- ply range in pool: [2, 149]
- seed: 42

## Graph size histogram (sampled positions, win_length=6, radius=6)

| metric | nodes | edges |
|---|---|---|
| mean | 490.0 | 2931.5 |
| p50 | 493.0 | 2924.0 |
| p90 | 729.2 | 4796.0 |
| max | 897 | 5690 |

## Python graph-builder timing (build_axis_graph_raw)

- median 14.0045 ms/position, IQR 0.5222 ms (unstable=False)

## Dense plane-encode timing (GameState.to_tensor, Python-reachable)

- median 0.0297 ms/position, IQR 0.0003 ms (unstable=False)
- Python-reachable GameState.to_tensor() dense plane-encode (K cluster views x 18ch, numpy). NOT the Rust engine::encode_state_to_buffer_channels <50us/position figure cited in the scoping doc (that path is Rust-internal, not Python-reachable per-position — cited, not measured, in this report).

## Rust builder proxy (hexo-strix axis_graph.rs)

- median 539180 ns/position (0.5392 ms/position), IQR 30382 ns (unstable=False)
- n_positions=55, n_reps=10
- mean_nodes=470.5, mean_edges=2890.7
- CAVEAT: Biased subsample: only positions where a real stone happens to sit at (0,0) are reconstructible through hexo-strix's public API (see origin_compat). Node/edge counts and ns/pos on this subset are directionally informative but not a random sample of the full graph-size distribution reported in graph_size_histogram.
- origin_compat: {'n_total': 320, 'n_compat_p1_at_origin': 55, 'n_incompat_p2_at_origin': 61, 'n_incompat_origin_empty': 204}

## ONNX export + parity gate

### scale=probe: OK
- parity: max|Δ|=5.484e-06 over 24 real graphs (threshold 0.0001) -> PASS

### scale=prod: OK
- parity: max|Δ|=6.557e-07 over 24 real graphs (threshold 0.0001) -> PASS

## Cell timing tables (median ms; IQR shown as ±)

### cnn_comparator | scale=prod_cnn_4.27M | device=cpu | precision=fp32 | backend=torch

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 16 | 41.217 | 0.601 | 388 | n/a | False |
| 32 | 84.222 | 7.781 | 380 | n/a | False |
| 64 | 208.029 | 7.234 | 308 | n/a | False |
| 128 | 681.686 | 92.024 | 188 | n/a | False |
| 256 | 1936.329 | 98.331 | 132 | n/a | False |

### cnn_comparator | scale=prod_cnn_4.27M | device=cuda | precision=fp16_autocast | backend=torch

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 16 | 4.097 | 0.272 | 3905 | 45.5 (n=2) | False |
| 32 | 6.454 | 0.225 | 4958 | 93.5 (n=2) | False |
| 64 | 13.929 | 1.019 | 4595 | 96.0 (n=2) | False |
| 128 | 35.631 | 0.216 | 3592 | 98.0 (n=2) | False |
| 256 | 99.797 | 12.373 | 2565 | 98.0 (n=2) | False |

### gnn_ort | scale=probe | device=cpu | precision=fp32 | backend=ort | intra_op_threads=1

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 16 | 621.527 | 4.751 | 26 | n/a | False |
| 32 | 1243.383 | 6.312 | 26 | n/a | False |
| 64 | 2337.185 | 25.330 | 27 | n/a | False |
| 128 | 4479.393 | 76.546 | 29 | n/a | False |
| 256 | 8782.464 | 90.103 | 29 | n/a | False |

### gnn_ort | scale=probe | device=cpu | precision=fp32 | backend=ort | intra_op_threads=2

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 16 | 383.404 | 3.668 | 42 | n/a | False |
| 32 | 754.648 | 4.293 | 42 | n/a | False |
| 64 | 1399.834 | 31.390 | 46 | n/a | False |
| 128 | 2824.637 | 16.649 | 45 | n/a | False |
| 256 | 5313.342 | 10.656 | 48 | n/a | False |

### gnn_ort | scale=probe | device=cpu | precision=fp32 | backend=ort | intra_op_threads=4

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 16 | 277.860 | 1.741 | 58 | n/a | False |
| 32 | 543.745 | 1.969 | 59 | n/a | False |
| 64 | 992.102 | 5.036 | 65 | n/a | False |
| 128 | 1996.929 | 34.533 | 64 | n/a | False |
| 256 | 3691.139 | 15.403 | 69 | n/a | False |

### gnn_ort | scale=probe | device=cpu | precision=fp32 | backend=ort | intra_op_threads=8

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 16 | 246.428 | 4.217 | 65 | n/a | False |
| 32 | 480.166 | 5.837 | 67 | n/a | False |
| 64 | 892.799 | 11.767 | 72 | n/a | False |
| 128 | 1709.275 | 26.072 | 75 | n/a | False |
| 256 | 3671.208 | 216.944 | 70 | n/a | False |

### gnn_ort | scale=probe | device=cuda | precision=fp32 | backend=ort

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 16 | 11.188 | 0.032 | 1430 | 99.0 (n=2) | False |
| 32 | 21.324 | 0.174 | 1501 | 97.5 (n=2) | False |
| 64 | 40.248 | 0.429 | 1590 | 98.5 (n=2) | False |
| 128 | 83.095 | 1.065 | 1540 | 75.5 (n=2) | False |

### gnn_ort | scale=prod | device=cpu | precision=fp32 | backend=ort | intra_op_threads=1

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 16 | 1735.313 | 19.991 | 9 | n/a | False |
| 32 | 3232.979 | 19.829 | 10 | n/a | False |
| 64 | 5476.777 | 11.899 | 12 | n/a | False |
| 128 | 13106.491 | 196.360 | 10 | n/a | False |
| 256 | 22878.419 | 96.001 | 11 | n/a | False |

### gnn_ort | scale=prod | device=cpu | precision=fp32 | backend=ort | intra_op_threads=2

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 16 | 1074.721 | 8.980 | 15 | n/a | False |
| 32 | 2048.974 | 53.456 | 16 | n/a | False |
| 64 | 3553.559 | 47.840 | 18 | n/a | False |
| 128 | 6929.462 | 69.938 | 18 | n/a | False |
| 256 | 13474.072 | 53.456 | 19 | n/a | False |

### gnn_ort | scale=prod | device=cpu | precision=fp32 | backend=ort | intra_op_threads=4

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 16 | 683.494 | 12.678 | 23 | n/a | False |
| 32 | 1314.358 | 16.106 | 24 | n/a | False |
| 64 | 2344.812 | 26.036 | 27 | n/a | False |
| 128 | 4522.388 | 21.296 | 28 | n/a | False |
| 256 | 8844.827 | 78.647 | 29 | n/a | False |

### gnn_ort | scale=prod | device=cpu | precision=fp32 | backend=ort | intra_op_threads=8

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 16 | 578.929 | 13.786 | 28 | n/a | False |
| 32 | 1111.571 | 44.104 | 29 | n/a | False |
| 64 | 2052.369 | 49.158 | 31 | n/a | False |
| 128 | 3903.337 | 101.934 | 33 | n/a | False |
| 256 | 7563.559 | 99.517 | 34 | n/a | False |

### gnn_ort | scale=prod | device=cuda | precision=fp32 | backend=ort

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 16 | 31.257 | 0.166 | 512 | 69.5 (n=2) | False |
| 32 | 60.655 | 6.566 | 528 | 99.5 (n=2) | False |
| 64 | 111.203 | 2.928 | 576 | 100.0 (n=2) | False |

### gnn_torch | scale=probe | device=cpu | precision=fp32 | backend=torch

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 16 | 83.670 | 1.327 | 191 | n/a | False |
| 32 | 180.114 | 4.334 | 178 | n/a | False |
| 64 | 340.719 | 1.946 | 188 | n/a | False |
| 128 | 652.756 | 8.277 | 196 | n/a | False |
| 256 | 1325.856 | 8.515 | 193 | n/a | False |

### gnn_torch | scale=probe | device=cuda | precision=fp16_autocast | backend=torch

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 16 | 5.659 | 0.035 | 2827 | 99.5 (n=2) | False |
| 32 | 11.212 | 0.035 | 2854 | 99.0 (n=2) | False |
| 64 | 21.464 | 0.045 | 2982 | 99.5 (n=2) | False |
| 128 | 44.073 | 0.079 | 2904 | 100.0 (n=2) | False |
| 256 | 87.216 | 0.132 | 2935 | 100.0 (n=2) | False |

### gnn_torch | scale=probe | device=cuda | precision=fp32 | backend=torch

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 16 | 6.805 | 0.029 | 2351 | 51.0 (n=2) | False |
| 32 | 13.228 | 0.051 | 2419 | 99.5 (n=2) | False |
| 64 | 24.782 | 0.044 | 2582 | 100.0 (n=2) | False |
| 128 | 50.972 | 0.085 | 2511 | 78.5 (n=2) | False |
| 256 | 98.174 | 0.075 | 2608 | 100.0 (n=2) | False |

### gnn_torch | scale=prod | device=cpu | precision=fp32 | backend=torch

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 16 | 214.654 | 4.135 | 75 | n/a | False |
| 32 | 419.113 | 11.028 | 76 | n/a | False |
| 64 | 738.379 | 5.807 | 87 | n/a | False |
| 128 | 1523.678 | 9.277 | 84 | n/a | False |
| 256 | 2992.536 | 18.556 | 86 | n/a | False |

### gnn_torch | scale=prod | device=cuda | precision=fp16_autocast | backend=torch

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 16 | 11.851 | 0.018 | 1350 | 99.5 (n=2) | False |
| 32 | 23.237 | 0.036 | 1377 | 99.5 (n=2) | False |
| 64 | 44.566 | 0.064 | 1436 | 72.0 (n=2) | False |
| 128 | 89.060 | 0.066 | 1437 | 100.0 (n=2) | False |
| 256 | 171.287 | 0.193 | 1495 | 100.0 (n=2) | False |

### gnn_torch | scale=prod | device=cuda | precision=fp32 | backend=torch

| batch | median ms | IQR ms | throughput pos/s | gpu_util % (n) | unstable |
|---|---|---|---|---|---|
| 16 | 15.247 | 0.537 | 1049 | 50.0 (n=2) | False |
| 32 | 29.918 | 0.042 | 1070 | 84.0 (n=2) | False |
| 64 | 56.228 | 0.052 | 1138 | 100.0 (n=2) | False |
| 128 | 110.648 | 0.088 | 1157 | 100.0 (n=2) | False |
| 256 | 213.253 | 0.405 | 1200 | 100.0 (n=3) | False |

## Notes / warnings

- ORT-CUDA cell FAILED-SKIPPED scale=probe bs=256: Fail: [ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Expand node. Name:'/representation/convs.0/Expand' Status Message: /onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:358 void* onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*) Available memory of 626465536 is smaller than requested bytes of 762740736

- ORT-CUDA cell FAILED-SKIPPED scale=prod bs=128: Fail: [ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Expand node. Name:'/representation/convs.0/Expand' Status Message: /onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:358 void* onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*) Available memory of 423193344 is smaller than requested bytes of 796155904

- ORT-CUDA cell FAILED-SKIPPED scale=prod bs=256: Fail: [ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Expand node. Name:'/representation/convs.0/Expand' Status Message: /onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:358 void* onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*) Available memory of 423193344 is smaller than requested bytes of 1525481472

