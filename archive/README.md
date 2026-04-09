# Archive

Historical diagnostic reports and broken run artifacts from Phase 4.0 development.
These are kept for forensics but should not clutter the live tree.

## bench_investigation_2026-04-09/

Referenced: sprint log §72. Three-run investigation (cold/hot/post-idle) confirming
a ~14% GPU clock reduction from NVIDIA `DynamicPowerManagement=3` after a full day of
workloads. Not a code regression. Led to §72 rebaseline of NN inference and worker
throughput targets in CLAUDE.md.

## checkpoints.broken-202604/

Referenced: sprint log §68. Ten checkpoints (steps 21500–24274), best_model.pt,
inference_only.pt, and replay_buffer.bin (1.4 GB) from the scheduler-poisoned run
(§67 LR scheduler bug). Kept for forensics; do not load for training.

## diagnosis_2026-04-10/

Referenced: sprint log §70; tracked as Q17 in docs/06_OPEN_QUESTIONS.md.
Diagnostic A/B/C artifacts from the Phase 4.0 overnight mode-collapse investigation.
Findings: selfplay collapse masked by pretrain buffer mix; Dirichlet noise absent from
Rust training path. No fixes proposed in this pass — findings only.

## dirichlet_port_2026-04-10/

Referenced: sprint log §73. Runtime verification that the Dirichlet port (commit
`71d7e6e`) was successful: `apply_dirichlet_to_root` now fires 10/10 times with
unique noise vectors. Superseded by §73 — Dirichlet port verified.

## eval.broken-202604/

Referenced: sprint log §68. Eval results DB containing 39 matches from the
scheduler-poisoned run (§67), all with `run_id=""`. Archived alongside
checkpoints.broken-202604/; do not use for ELO history.

## sweep_2026-04-08/

Referenced: sprint log §69. Per-run data from the 15+1 config sweep (PUCT and Gumbel
arms, 20-min windows each). Config P3 was selected as the Phase 4.0 overnight base.
Full methodology and raw results in `results.csv` and `summary.md`.

## verify_gumbel_2026-04-10/

Referenced: sprint log §71. Static audit and runtime trace verifying that
`gumbel_mcts: true` provides functionally active root noise (visit concentration
0.24 vs 0.65 for PUCT). Pre-Dirichlet-port verification; superseded by §73.
