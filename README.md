# Hex Tac Toe AlphaZero

Welcome to the Hex Tac Toe AlphaZero project. This project implements an AlphaZero-style reinforcement learning agent to play Hex Tac Toe on an infinite hexagonal grid. 

## Project Introduction

Hex Tac Toe is played on an infinite hexagonal grid. This repository contains the core Rust engine for high-performance board representation and Monte Carlo Tree Search (MCTS), bound to a Python environment where a PyTorch-based neural network is trained via self-play.

We employ a "Multi-Window Cluster-Based Approach" to handle the infinite board: the Rust core dynamically clusters active stones into distinct colonies and generates K distinct 19x19 tensors. These are evaluated as a batch by a single sliding-window ResNet, resolving Attention Hijacking while maintaining high performance.

## Setup Instructions

1. **Python Virtual Environment:**
   Ensure you have Python 3.11+ installed. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
   Install dependencies:
   ```bash
   pip install -r requirements.txt # (or as specified in the project)
   ```

2. **Rust Core:**
   The high-performance core is written in Rust. Build the Python bindings using Maturin:
   ```bash
   pip install maturin
   maturin develop --release -m native_core/Cargo.toml
   ```

3. **Git Submodules:**
   Initialize and update required submodules (for baseline bots, Ramora engine, etc.):
   ```bash
   git submodule update --init --recursive
   ```

## Roadmap

* **Phase 1: Foundation** - Basic board representation, game state, and neural network architecture.
* **Phase 2: MCTS** - Monte Carlo Tree Search implementation in Rust with PyO3 bindings.
* **Phase 3: Bootstrap** - Integration of baseline bots and community engines for initial pre-training.
* **Phase 3.5: Foveated Vision Refactor** - Implementation of the Dual-Resolution CNN to prevent attention hijacking (Current Phase).
* **Phase 4: Distributed Self-Play** - Scaling up training with distributed workers.
* **Phase 5: Evaluation** - Benchmarking against state-of-the-art bots.
