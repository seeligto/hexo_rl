"""
Multiprocess self-play worker pool.

Spawns N worker processes that each play self-play games indefinitely,
pushing positions to a shared replay buffer.  Workers send leaf states to
a shared inference queue; a GPU inference thread in the main process
batches and resolves them.

Architecture
────────────
  Main process
    ├── InferenceServer (GPU thread)  ← batches leaf requests
    ├── Worker process 0  ──┐
    ├── Worker process 1  ──┤  push leaves via mp.Queue
    ├── Worker process 2  ──┤  receive (policy, value) back
    └── Worker process 3  ──┘

  Each worker:
    loop:
      game = new_game()
      while not terminal:
        run MCTS: for each simulation:
          leaf = tree.select_leaves(1)
          policy, value = client.infer(leaf_state)
          tree.expand_and_backup(policy, value)
        move = sample(get_policy(temperature))
        game.apply(move)
      push game records to buffer (via result_queue)

Cross-process inference protocol
─────────────────────────────────
  Request:  (worker_id, request_id, state_np)   → request_queue
  Response: (worker_id, request_id, policy_np, value_float) → result_queues[worker_id]

  Each worker has its own response queue so the server can route replies
  without a global dict lock.

  Backpressure: request_queue has maxsize to prevent workers flooding the
  buffer faster than the GPU can drain it.

Usage
─────
    pool = WorkerPool(model, config, device, replay_buffer)
    pool.start()
    # ... training loop ...
    pool.stop()
"""

from __future__ import annotations

import os
import time
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.multiprocessing as mp

from python.model.network import HexTacToeNet
from python.training.replay_buffer import ReplayBuffer

# ── Constants ─────────────────────────────────────────────────────────────────

_BOARD_SIZE: int = 19
_N_ACTIONS:  int = _BOARD_SIZE * _BOARD_SIZE + 1   # 362
_STOP_SENTINEL = None  # placed on queues to signal shutdown


# ── Worker-side inference client ──────────────────────────────────────────────

class _InferenceClient:
    """Sends leaf states to the main-process InferenceServer via mp.Queue.

    Lives in the worker process.  Each request is given a monotonic ID so
    the server response can be matched back even if replies arrive out of
    order (they won't in single-GPU mode, but good hygiene).
    """

    def __init__(
        self,
        worker_id: int,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
    ) -> None:
        self._worker_id = worker_id
        self._req_q  = request_queue
        self._resp_q = response_queue
        self._next_id: int = 0

    def infer(self, state: np.ndarray):
        """Synchronously request inference for `state`.

        Returns (policy_probs, value).
        policy_probs: np.ndarray (362,) float32
        value: float
        """
        req_id = self._next_id
        self._next_id += 1
        self._req_q.put((self._worker_id, req_id, state))
        # Block until the matching response arrives.
        while True:
            resp = self._resp_q.get()
            if resp is None:
                return None  # stop sentinel
            w_id, r_id, policy, value = resp
            if r_id == req_id:
                return policy, value
            # Out-of-order (shouldn't happen in practice): put it back.
            self._resp_q.put(resp)


# ── Worker process function ───────────────────────────────────────────────────

def _worker_fn(
    worker_id: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    config: Dict[str, Any],
    stop_event: mp.Event,
) -> None:
    """Entry point for each worker process.

    Plays games continuously, using the inference client to evaluate leaves,
    and pushes completed game records via result_queue.
    """
    # Import inside worker to avoid fork-time issues.
    from native_core import Board, MCTSTree
    from python.env.game_state import GameState

    client = _InferenceClient(worker_id, request_queue, response_queue)

    mcts_cfg = config.get("mcts", config)
    sp_cfg   = config.get("selfplay", config)

    n_sims       = int(mcts_cfg.get("n_simulations", 50))
    c_puct       = float(mcts_cfg.get("c_puct", 1.5))
    temp_thresh  = int(mcts_cfg.get("temperature_threshold_ply", 30))
    board_size   = _BOARD_SIZE

    half = (board_size - 1) // 2
    tree = MCTSTree(c_puct=c_puct)

    def flat_to_coords(flat: int):
        q = flat // board_size - half
        r = flat %  board_size - half
        return q, r

    games_played = 0

    while not stop_event.is_set():
        rust_board = Board()
        state      = GameState.from_board(rust_board)
        records    = []   # (state_tensor, mcts_policy, player)

        while True:
            if rust_board.check_win() or rust_board.legal_move_count() == 0:
                break

            # ── MCTS search ──
            tree.new_game(rust_board)
            for _ in range(n_sims):
                leaves = tree.select_leaves(1)
                if not leaves:
                    break
                leaf_board = leaves[0]
                leaf_state = GameState.from_board(leaf_board)
                leaf_tensor = leaf_state.to_tensor()  # (18, 19, 19) float16

                result = client.infer(leaf_tensor)
                if result is None:
                    return  # stop signal
                policy, value = result
                tree.expand_and_backup([list(policy)], [value])

            ply = rust_board.ply
            temperature = 1.0 if ply < temp_thresh else 0.1
            mcts_policy = tree.get_policy(temperature=temperature, board_size=board_size)

            # Record position.
            state_tensor = state.to_tensor()
            policy_arr   = np.array(mcts_policy, dtype=np.float32)
            records.append((state_tensor, policy_arr, state.current_player))

            # Sample and apply move.
            legal = rust_board.legal_moves()
            if not legal:
                break

            n_acts = board_size * board_size + 1
            legal_flat = [(q + half) * board_size + (r + half) for q, r in legal]
            probs = np.array([mcts_policy[i] if i < n_acts else 0.0 for i in legal_flat],
                             dtype=np.float64)
            total = probs.sum()
            if total < 1e-9:
                probs = np.ones(len(legal)) / len(legal)
            else:
                probs /= total

            idx = np.random.choice(len(legal), p=probs)
            q, r = legal[idx]
            state = state.apply_move(rust_board, q, r)

        # Push completed game records.
        winner = rust_board.winner()
        result_queue.put((records, winner, worker_id))
        games_played += 1


# ── Main-process inference dispatcher ─────────────────────────────────────────

class _InferenceDispatcher(mp.Process):
    """Main-process thread (actually a Process-level helper) that reads from
    `request_queue`, calls InferenceServer.infer(), and routes responses back
    to the correct worker via `response_queues`.

    This decouples the response routing from the GPU thread — the GPU thread
    only does inference, never touches mp.Queue.
    """
    # Note: we do NOT inherit from Process; this runs in a regular thread
    # inside the main process.  It's a threading.Thread, not mp.Process.
    pass


# ── WorkerPool ────────────────────────────────────────────────────────────────

class WorkerPool:
    """Manages N parallel self-play worker processes.

    The pool wires together:
      - N worker processes (each playing games and pushing leaf evals)
      - An InferenceServer (GPU thread batching leaf evaluations)
      - A result collector thread (pushing game records into ReplayBuffer)

    Args:
        model:         HexTacToeNet (in main process, on GPU).
        config:        Full config dict.
        device:        GPU device.
        replay_buffer: ReplayBuffer to push completed game data into.
        n_workers:     Number of worker processes (default from config).
    """

    def __init__(
        self,
        model: HexTacToeNet,
        config: Dict[str, Any],
        device: torch.device,
        replay_buffer: ReplayBuffer,
        n_workers: Optional[int] = None,
    ) -> None:
        self.model         = model
        self.config        = config
        self.device        = device
        self.replay_buffer = replay_buffer

        sp_cfg = config.get("selfplay", config)
        self.n_workers    = n_workers or int(sp_cfg.get("n_workers", 4))
        self._batch_size  = int(sp_cfg.get("inference_batch_size", 64))
        self._queue_maxsize = self._batch_size * self.n_workers * 2  # backpressure

        # mp.Queue: workers send (worker_id, req_id, state_np)
        self._request_queue: mp.Queue = mp.Queue(maxsize=self._queue_maxsize)
        # Per-worker response queues (unbounded — server always drains fast).
        self._response_queues: List[mp.Queue] = [mp.Queue() for _ in range(self.n_workers)]
        # Workers push completed game records here.
        self._result_queue: mp.Queue = mp.Queue(maxsize=500)
        # Stop signal for workers.
        self._stop_event: mp.Event = mp.Event()

        self._workers: List[mp.Process] = []
        self._dispatch_thread: Optional[__import__("threading").Thread] = None
        self._collect_thread:  Optional[__import__("threading").Thread] = None
        self._inference_server = None  # set in start()

        # Stats.
        self.games_completed: int = 0
        self.positions_pushed: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start inference server, worker processes, and helper threads."""
        from python.selfplay.inference_server import InferenceServer
        import threading

        # 1. Start inference server (GPU thread).
        self._inference_server = InferenceServer(self.model, self.device, self.config)
        self._inference_server.start()

        # 2. Start dispatch thread: reads from request_queue, calls inference server,
        #    routes responses back to worker response queues.
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            name="inference-dispatcher",
            daemon=True,
        )
        self._dispatch_thread.start()

        # 3. Start result collector: pops completed games, pushes to replay buffer.
        self._collect_thread = threading.Thread(
            target=self._collect_loop,
            name="result-collector",
            daemon=True,
        )
        self._collect_thread.start()

        # 4. Spawn worker processes.
        for wid in range(self.n_workers):
            p = mp.Process(
                target=_worker_fn,
                args=(
                    wid,
                    self._request_queue,
                    self._response_queues[wid],
                    self._result_queue,
                    self.config,
                    self._stop_event,
                ),
                daemon=True,
                name=f"selfplay-worker-{wid}",
            )
            p.start()
            self._workers.append(p)

    def stop(self) -> None:
        """Signal all workers to stop and wait for them to exit."""
        self._stop_event.set()
        for p in self._workers:
            p.join(timeout=30.0)
            if p.is_alive():
                p.terminate()
        if self._inference_server is not None:
            self._inference_server.stop()
            self._inference_server.join(timeout=5.0)

    def load_weights(self, state_dict: dict) -> None:
        """Hot-swap model weights (called after a new best checkpoint is found)."""
        self.model.load_state_dict(state_dict)

    # ── Internal threads ──────────────────────────────────────────────────────

    def _dispatch_loop(self) -> None:
        """Read inference requests from workers, call server, route responses."""
        while not self._stop_event.is_set():
            try:
                item = self._request_queue.get(timeout=0.1)
            except Exception:
                continue
            if item is None:
                break
            worker_id, req_id, state = item
            try:
                result = self._inference_server.infer(state)
            except Exception:
                result = (np.ones(_N_ACTIONS, dtype=np.float32) / _N_ACTIONS, 0.0)
            if result is None:
                continue
            policy, value = result
            self._response_queues[worker_id].put((worker_id, req_id, policy, value))

    def _collect_loop(self) -> None:
        """Pop completed game records from result_queue → push to ReplayBuffer."""
        while not self._stop_event.is_set():
            try:
                item = self._result_queue.get(timeout=0.1)
            except Exception:
                continue
            if item is None:
                break
            records, winner, _worker_id = item
            for state_tensor, policy_arr, player in records:
                if winner is None:
                    outcome = 0.0
                else:
                    outcome = 1.0 if player == winner else -1.0
                self.replay_buffer.push(state_tensor, policy_arr, outcome)
                self.positions_pushed += 1
            self.games_completed += 1
