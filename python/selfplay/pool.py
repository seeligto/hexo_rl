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

import time
import signal
import queue
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

    def infer_batch(self, states: List[np.ndarray]):
        """Synchronously request inference for a batch of K-cluster tensors.

        Args:
            states: list of leaf tensors; each item is shape (K, 18, 19, 19).

        Returns:
            list of per-leaf outputs, where each item is
            (cluster_policies, cluster_values).
        """
        if not states:
            return []

        req_id = self._next_id
        self._next_id += 1
        try:
            self._req_q.put((self._worker_id, req_id, states))
        except (BrokenPipeError, EOFError, OSError, KeyboardInterrupt):
            return None
        # Block until the matching response arrives.
        while True:
            try:
                resp = self._resp_q.get(timeout=1.0)
            except queue.Empty:
                continue
            except (BrokenPipeError, EOFError, OSError, KeyboardInterrupt):
                return None
            if resp is None:
                return None  # stop sentinel
            w_id, r_id, outputs = resp
            if r_id == req_id:
                return outputs
            # Out-of-order (shouldn't happen in practice): put it back.
            self._resp_q.put(resp)

    def infer(self, state: np.ndarray):
        """Backwards-compatible single-state helper."""
        outputs = self.infer_batch([state])
        if outputs is None:
            return None
        return outputs[0]


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

    # Parent handles graceful shutdown. Workers ignore SIGINT so they can
    # terminate via stop_event/sentinels without noisy traceback spam.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    client = _InferenceClient(worker_id, request_queue, response_queue)

    mcts_cfg = config.get("mcts", config)
    sp_cfg   = config.get("selfplay", config)

    n_sims       = int(sp_cfg.get("n_simulations", mcts_cfg.get("n_simulations", 50)))
    leaf_batch_size = int(sp_cfg.get("leaf_batch_size", 8))
    fast_playout_prob = float(sp_cfg.get("fast_playout_prob", 0.9))
    fast_sims_min = int(sp_cfg.get("fast_sims_min", 15))
    fast_sims_max = int(sp_cfg.get("fast_sims_max", 25))
    c_puct       = float(mcts_cfg.get("c_puct", 1.5))
    temp_thresh  = int(mcts_cfg.get("temperature_threshold_ply", 30))
    board_size   = _BOARD_SIZE

    half = (board_size - 1) // 2
    tree = MCTSTree(c_puct=c_puct)

    def _merge_cluster_outputs(
        leaf_board: "Board",
        centers: List[tuple[int, int]],
        cluster_policies: np.ndarray,
        cluster_values: np.ndarray,
    ) -> tuple[list[float], float]:
        """Map per-cluster network outputs back to global board policy."""
        n_actions = board_size * board_size + 1
        global_policy = np.zeros(n_actions, dtype=np.float64)

        legal = leaf_board.legal_moves()
        for q, r in legal:
            mcts_idx = leaf_board.to_flat(q, r)
            if mcts_idx >= n_actions - 1:
                continue

            max_prob = 0.0
            for k, (cq, cr) in enumerate(centers):
                wq = q - cq + half
                wr = r - cr + half
                if 0 <= wq < board_size and 0 <= wr < board_size:
                    local_idx = wq * board_size + wr
                    p = float(cluster_policies[k, local_idx])
                    if p > max_prob:
                        max_prob = p

            global_policy[mcts_idx] = max_prob

        total = global_policy.sum()
        if total > 1e-9:
            global_policy /= total
        else:
            global_policy.fill(1.0 / n_actions)

        value = float(cluster_values.min()) if cluster_values.size else 0.0
        return global_policy.tolist(), value

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
            current_n_sims = n_sims
            if np.random.random() < fast_playout_prob:
                current_n_sims = np.random.randint(fast_sims_min, fast_sims_max + 1)

            for sim_idx in range(0, current_n_sims, leaf_batch_size):
                current_batch = min(leaf_batch_size, current_n_sims - sim_idx)
                leaves = tree.select_leaves(current_batch)
                if not leaves:
                    break

                leaf_tensors: List[np.ndarray] = []
                leaf_centers: List[List[tuple[int, int]]] = []
                for leaf_board in leaves:
                    leaf_state = GameState.from_board(leaf_board)
                    leaf_tensor, centers = leaf_state.to_tensor()  # (K, 18, 19, 19), centers
                    leaf_tensors.append(leaf_tensor)
                    leaf_centers.append(centers)

                batch_results = client.infer_batch(leaf_tensors)
                if batch_results is None:
                    return  # stop signal

                policies: List[List[float]] = []
                values: List[float] = []
                for leaf_board, centers, result in zip(leaves, leaf_centers, batch_results):
                    cluster_policies, cluster_values = result
                    policy, value = _merge_cluster_outputs(
                        leaf_board=leaf_board,
                        centers=centers,
                        cluster_policies=cluster_policies,
                        cluster_values=cluster_values,
                    )
                    policies.append(policy)
                    values.append(value)

                tree.expand_and_backup(policies, values)

            ply = rust_board.ply
            temperature = 1.0 if ply < temp_thresh else 0.1
            mcts_policy = tree.get_policy(temperature=temperature, board_size=board_size)

            # Record position.
            # to_tensor() returns (K, 18, 19, 19) tensor and list of K centers
            full_tensor, centers = state.to_tensor()
            
            # For Phase 1 / simplicity, we only store the first cluster in the buffer.
            # In Phase 2+, we will store all K clusters.
            state_tensor = full_tensor[0] # (18, 19, 19)
            
            policy_arr   = np.array(mcts_policy, dtype=np.float32)
            records.append((state_tensor, policy_arr, state.current_player))

            # Sample and apply move.
            legal = rust_board.legal_moves()
            if not legal:
                break

            n_acts = board_size * board_size + 1
            legal_flat = [rust_board.to_flat(q, r) for q, r in legal]
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
        self._dispatch_wait_s = float(sp_cfg.get("dispatch_wait_ms", 2.0)) / 1000.0
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
        self._request_queue.put(_STOP_SENTINEL)
        self._result_queue.put(_STOP_SENTINEL)
        for q in self._response_queues:
            q.put(_STOP_SENTINEL)

        if self._dispatch_thread is not None:
            self._dispatch_thread.join(timeout=5.0)
        if self._collect_thread is not None:
            self._collect_thread.join(timeout=5.0)

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
                first_item = self._request_queue.get(timeout=0.1)
            except Exception:
                continue
            if first_item is None:
                break

            pending = [first_item]
            deadline = time.monotonic() + self._dispatch_wait_s
            while len(pending) < self._batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._request_queue.get(timeout=remaining)
                except Exception:
                    break
                if item is None:
                    break
                pending.append(item)

            all_states: List[np.ndarray] = []
            req_meta: List[tuple[int, int, int]] = []  # (worker_id, req_id, n_states)
            for worker_id, req_id, state_batch in pending:
                req_meta.append((worker_id, req_id, len(state_batch)))
                all_states.extend(state_batch)

            try:
                all_outputs = self._inference_server.infer_many(all_states)
            except Exception:
                all_outputs = []
                for state in all_states:
                    k = int(state.shape[0])
                    fallback_policy = np.ones((k, _N_ACTIONS), dtype=np.float32) / _N_ACTIONS
                    fallback_value = np.zeros((k,), dtype=np.float32)
                    all_outputs.append((fallback_policy, fallback_value))

            offset = 0
            for worker_id, req_id, n_states in req_meta:
                outputs = all_outputs[offset:offset + n_states]
                offset += n_states
                self._response_queues[worker_id].put((worker_id, req_id, outputs))

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
