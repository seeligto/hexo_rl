"""StrixV1Bot — pure-policy wrapper for SootyOwl/hexo-strix GNN checkpoint.

Raw-policy forward (greedy argmax over the strix policy head; no MCTS, no
torch_geometric, no strix code imported). Mirrors the deploy path strix's own
``evaluate._model_move`` takes at ``mcts_sims=0, temperature=0`` — the
deterministic raw-policy move ``legal_moves[argmax(policy_logits)]``.

The strix GNN is a graph net over an AXIS-window line-topology graph built by
strix's OWN resolver — reimplemented from public Rust source in
``strix_v1_graph`` (NOT routed through hexo_rl encoding/window resolvers). The
forward pass is reimplemented in ``strix_v1_net`` (plain torch GINE + JK-cat).

Fidelity: byte-identical to strix's own PyG ``HeXONet`` forward on a 20-position
fixture set (max |Δ| = 0.0 on policy logits + value). See
``reports/tourney/adapters_strix.md``.

SECURITY: the .pt is UNTRUSTED — loaded with ``weights_only=True`` only. Archive
inventoried (pickle globals allowlist: OrderedDict / FloatStorage /
_rebuild_tensor_v2 only) before any load. See the report.

Native deploy budget (strix README / configs; recorded as metadata, NOT run
here — raw-policy keeps the adapter deterministic + portable like krakenbot):
Gumbel MCTS n_simulations=16, m_actions=16 (train/self-play); eval/serving
tiers use sims up to 256. This adapter plays the raw policy head.

Pair-move caching: one forward per compound turn, returns stone-1 then stone-2
on successive calls. CPU-only. Diagnostics -> JSONL sidecar, never stderr.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import structlog
import torch

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.env import GameState
from hexo_rl.bots.strix_v1_net import HeXONet
from hexo_rl.bots.strix_v1_graph import build_axis_graph_raw

log = structlog.get_logger()

# Source of the ported forward pass (record for provenance).
STRIX_SOURCE_SHA = "c381ffbeb248313a1ec177eb650d9c3c2380caa8"

# strix checkpoint game_config (from the .pt sidecar): win_length=6,
# placement_radius=6. The graph builder must use strix's OWN radius so the
# legal-move node set (and thus the policy logit ordering) matches strix.
_STRIX_WIN_LENGTH = 6
_STRIX_RADIUS = 6

# strix's declared native search budget (metadata only; raw-policy deploys here).
STRIX_NATIVE_DEPLOY = {
    "algorithm": "Gumbel AlphaZero MCTS (Danihelka 2022)",
    "n_simulations": 16,
    "m_actions": 16,
    "eval_sims_tier": 256,
    "temperature": 0.0,
    "note": "raw-policy argmax used by this adapter (deterministic, portable)",
}

_DEFAULT_CKPT = Path(__file__).parents[2] / "strix_checkpoint_00237000.pt"
_DEFAULT_DIAG = Path("reports") / "tourney" / "strix_v1_eval.jsonl"


def _smart_legal_fallback(rust_board: object, reason: str, ply: int) -> tuple[int, int]:
    """Uniform-random-legal fallback. Logs to structlog (never stderr)."""
    legal = rust_board.legal_moves()  # type: ignore[attr-defined]
    if not legal:
        raise RuntimeError("No legal moves available on board")
    log.warning("strix_v1_fallback", reason=reason, ply=ply, n_legal=len(legal))
    return random.choice(legal)


class StrixV1Bot(BotProtocol):
    """BotProtocol wrapper for the hexo-strix GNN — raw-policy argmax, CPU-only.

    Args:
        model_path: Path to strix_checkpoint_00237000.pt. None -> repo default.
        device:     Torch device string. Default "cpu". No CUDA code used.
        label:      Bot name returned by name(). Default "strix".
        seed:       Reserved for tie-break (argmax is deterministic).
        diag_path:  Append diagnostics JSON lines here. None -> default path.
                    Pass False to disable diagnostics entirely.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        label: str = "strix",
        seed: int = 0,
        diag_path: Optional[object] = None,
    ) -> None:
        path = Path(model_path) if model_path is not None else _DEFAULT_CKPT
        # SECURITY: weights_only=True — the .pt is untrusted external. Never a
        # plain torch.load. Pickle globals were inventoried before shipping this.
        ckpt = torch.load(str(path), map_location=device, weights_only=True)
        if not (isinstance(ckpt, dict) and "model_state_dict" in ckpt):
            raise ValueError("strix checkpoint missing 'model_state_dict'")
        sd = ckpt["model_state_dict"]
        mc = ckpt.get("model_config", {})

        # Pin arch from the checkpoint's embedded model_config; fall back to the
        # known 237000 shape if absent.
        self._model = HeXONet(
            in_dim=11,
            hidden=int(mc.get("hidden_dim", 128)),
            num_layers=int(mc.get("num_layers", 4)),
            policy_hidden=int(mc.get("policy_hidden", 128)),
            value_hidden=int(mc.get("value_hidden", 32)),
        )
        missing, unexpected = self._model.load_state_dict(sd, strict=False)
        # This ckpt has no train-only heads (q_head/horizon) — expect a clean load.
        if missing or unexpected:
            raise ValueError(
                f"strix state_dict load mismatch: missing={list(missing)} "
                f"unexpected={list(unexpected)}"
            )
        self._model.to(device)
        self._model.eval()

        # Graph-build flags from the checkpoint (authoritative), with the known
        # 237000 defaults (axis / threat / relative / prune).
        self._prune = bool(mc.get("prune_empty_edges", True))
        self._threat = bool(mc.get("threat_features", True))
        self._relative = bool(mc.get("relative_stone_encoding", True))
        if mc.get("graph_type", "axis") != "axis":
            raise ValueError(f"StrixV1Bot only supports graph_type=axis, got {mc.get('graph_type')}")

        gc = ckpt.get("game_config", {})
        self._win_length = int(gc.get("win_length", _STRIX_WIN_LENGTH))
        self._radius = int(gc.get("placement_radius", _STRIX_RADIUS))

        self._device = device
        self._label = label
        self._seed = seed
        self._pending_move: Optional[tuple[int, int]] = None

        if diag_path is False:
            self._diag: Optional[Path] = None
        elif diag_path is None:
            self._diag = Path(_DEFAULT_DIAG)
        else:
            self._diag = Path(diag_path)  # type: ignore[arg-type]
        if self._diag is not None:
            self._diag.parent.mkdir(parents=True, exist_ok=True)

    # ── BotProtocol ──────────────────────────────────────────────────────────

    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        """Return one legal move: the argmax of a FRESH forward on the CURRENT
        board. On a compound turn the arena child re-calls get_move per stone with
        stone-1 already placed and moves_remaining decremented, so this always
        re-forwards on the up-to-date board — exactly mirroring strix's OWN
        `_model_move`-per-ply deploy path (mcts_sims=0). No stone-2 cache: the
        stale-forward turn-assembly bug (2nd stone drawn from the mr=2 forward that
        never saw stone-1) is fixed by never caching. See
        reports/tourney/strix_argmax_verify.md."""
        # Build strix's stone map: {(q,r): player_int} with 1=P1, -1=P2.
        # Our board's get_stones() returns exactly (q, r, player) with 1/-1.
        stone_map = {(q, r): p for (q, r, p) in rust_board.get_stones()}  # type: ignore[attr-defined]

        g = build_axis_graph_raw(
            stone_map,
            state.current_player,
            state.moves_remaining,
            win_length=self._win_length,
            radius=self._radius,
            prune_empty_edges=self._prune,
            threat_features=self._threat,
            relative_stones=self._relative,
        )

        # Empty-board short-circuit: strix derives legal-move nodes from placed
        # stones, so a stone-less board (P1's very first move on a fresh eval
        # board) yields ZERO strix legal nodes — the net has no opinion. Play a
        # deterministic legal move (prefer origin) instead of a random fallback.
        if not g["legal_coords"]:
            legal = rust_board.legal_moves()  # type: ignore[attr-defined]
            if not legal:
                raise RuntimeError("No legal moves available on board")
            chosen = (0, 0) if (0, 0) in set(legal) else min(legal)
            self._write_diag(ply=state.ply, pair=[chosen], top_logit=0.0,
                             value=0.0, legal_rank=-1, fallback_used=True)
            return chosen

        n = g["num_nodes"]
        fdim = g["fdim"]
        x = torch.tensor(g["features"], dtype=torch.float32, device=self._device).reshape(n, fdim)
        E = len(g["edge_src"])
        if E:
            edge_index = torch.tensor([g["edge_src"], g["edge_dst"]], dtype=torch.int64, device=self._device)
            edge_attr = torch.tensor(g["edge_attr"], dtype=torch.float32, device=self._device).reshape(E, 5)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.int64, device=self._device)
            edge_attr = torch.zeros((0, 5), dtype=torch.float32, device=self._device)
        legal_mask = torch.tensor(g["legal_mask"], dtype=torch.bool, device=self._device)
        stone_mask = torch.tensor(g["stone_mask"], dtype=torch.bool, device=self._device)

        policy_logits, value = self._model(x, edge_index, edge_attr, legal_mask, stone_mask)

        # strix legal-node order == sorted legal_moves order == g["legal_coords"].
        # But LEGALITY is authoritative on OUR board: strix's radius-derived
        # legal set can differ from our board's radius, so filter by our board.
        legal_set = set(rust_board.legal_moves())  # type: ignore[attr-defined]
        strix_legal = g["legal_coords"]

        order = policy_logits.argsort(descending=True).tolist()

        chosen1: Optional[tuple[int, int]] = None
        legal_rank = -1
        fallback_used = False
        top_logit = float(policy_logits.max().item()) if len(policy_logits) else 0.0

        # Pick best strix-scored move that is legal on OUR board. This is the
        # single per-stone decision — the arena child re-calls get_move for the
        # second stone with stone-1 already on the board and moves_remaining
        # decremented, so THAT call re-forwards on the updated board (mr=1) and
        # picks the natural line-extension, exactly like strix's own loop.
        for rank, idx in enumerate(order):
            cand = tuple(strix_legal[idx])
            if cand in legal_set:
                chosen1 = cand
                legal_rank = rank
                break
        if chosen1 is None:
            fallback_used = True
            chosen1 = _smart_legal_fallback(rust_board, "no_legal_single", state.ply)

        self._write_diag(
            ply=state.ply,
            pair=[chosen1],
            top_logit=top_logit,
            value=float(value.item()),
            legal_rank=legal_rank,
            fallback_used=fallback_used,
        )
        return chosen1

    def reset(self) -> None:
        """No-op between games. get_move re-forwards per stone (no stone-2 cache);
        this clears the now-vestigial _pending_move slot, kept harmless."""
        self._pending_move = None

    def name(self) -> str:
        return self._label

    # ── internals ─────────────────────────────────────────────────────────────

    def _write_diag(self, ply, pair, top_logit, value, legal_rank, fallback_used) -> None:
        if self._diag is None:
            return
        rec = {
            "ply": ply,
            "pair": [list(p) for p in pair],
            "top_logit": top_logit,
            "value": value,
            "legal_rank": legal_rank,
            "fallback_used": fallback_used,
            "source_sha": STRIX_SOURCE_SHA,
        }
        try:
            with self._diag.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError as exc:
            log.warning("strix_v1_diag_write_error", exc=str(exc))
