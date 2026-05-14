"""Checkpoint validation entry (§176 P39 split from pretrain.py).

Contains:
  - validate — verifies checkpoint round-trip + plays 100 greedy games vs
    RandomBot. Skipped under v8 / pma_global / gpool_bias_active per
    encoding-specific notes; called at the tail of pretrain CLI for
    v6-family encodings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import structlog
import torch
from rich.console import Console

from engine import Board
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.env.game_state import GameState
from hexo_rl.model.network import HexTacToeNet

log = structlog.get_logger()
console = Console()


# ── Validation ────────────────────────────────────────────────────────────────

def validate(ckpt_path: Path, device: torch.device) -> None:
    """Verify checkpoint round-trip and play 100 greedy games vs RandomBot.

    Uses argmax policy (no MCTS) — suitable for a pretrained but not
    yet self-play-trained checkpoint.

    Encoding-aware (§168): reads `board_size` and `encoding` from the
    saved config and configures the Rust Board with the matching cluster
    window size + threshold + legal-move radius. v6 default at 19×19 /
    threshold 5 / r=5; v6w25 at 25×25 / threshold 8 / r=8; v8 path
    skipped (encoding != "v6"-family — no K-cluster window encoder).
    """
    from hexo_rl.bootstrap.bots.random_bot import RandomBot

    # Round-trip: load checkpoint, rebuild model, run forward pass.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert "model_state" in ckpt and "config" in ckpt, (
        f"Checkpoint missing keys; got {list(ckpt.keys())}"
    )
    cfg = ckpt["config"]
    enc_section = cfg.get("encoding")
    if isinstance(enc_section, str):
        cfg_encoding = enc_section
    elif isinstance(enc_section, dict):
        cfg_encoding = str(enc_section.get("version", "v6"))
    else:
        cfg_encoding = "v6"
    _enc_spec = _lookup_encoding(cfg_encoding)
    cfg_board_size = _enc_spec.trunk_size
    cfg_in_channels = int(cfg.get("in_channels", _enc_spec.n_planes))
    cfg_n_actions = _enc_spec.policy_logit_count
    cfg_half = (cfg_board_size - 1) // 2

    cfg_pool_type = str(cfg.get("pool_type", "min_max"))
    cfg_gpool_bias_active = bool(cfg.get("gpool_bias_active", False))
    loaded_model = HexTacToeNet(
        board_size=cfg_board_size,
        in_channels=cfg_in_channels,
        filters=int(cfg.get("filters", 128)),
        res_blocks=int(cfg.get("res_blocks", 12)),
        se_reduction_ratio=int(cfg.get("se_reduction_ratio", 4)),
        encoding=cfg_encoding,
        pool_type=cfg_pool_type,
        pool_attn_dropout=float(cfg.get("pool_attn_dropout", 0.1)),
        gpool_bias_active=cfg_gpool_bias_active,
    )
    loaded_model.load_state_dict(ckpt["model_state"])
    loaded_model.eval().to(device)

    dummy = torch.zeros(
        1, cfg_in_channels, cfg_board_size, cfg_board_size, device=device,
    )
    fwd_kwargs: Dict = {}
    if cfg_pool_type == "pma_global" or cfg_gpool_bias_active:
        # pma_global / gpool_bias_active need a global crop for the forward;
        # a zeroed canvas is a valid empty-board input. The smoke just checks
        # the wiring round-trips; full play-vs-RandomBot is skipped.
        from hexo_rl.utils.global_crop import (
            CANVAS_SIZE as _GLOBAL_CANVAS_SIZE,
            N_GLOBAL_PLANES as _N_GLOBAL_PLANES,
        )
        fwd_kwargs["global_crop"] = torch.zeros(
            1, _N_GLOBAL_PLANES, _GLOBAL_CANVAS_SIZE, _GLOBAL_CANVAS_SIZE,
            device=device,
        )
    with torch.no_grad():
        log_pol, val, v_logit = loaded_model(dummy.float(), **fwd_kwargs)
    assert log_pol.shape == (1, cfg_n_actions), \
        f"Unexpected policy shape: {log_pol.shape} (expected (1, {cfg_n_actions}))"
    log.info("checkpoint_forward_pass_ok", val=float(val[0, 0]))

    # v8 needs a different game loop (single-bbox encoder + V8ArgmaxBot
    # history tracking). Skip the play-100-greedy step under v8; the
    # round-trip + forward-pass shape check is sufficient at this layer.
    if cfg_encoding == "v8":
        log.info("validation_skipped_v8_path", reason="v8 needs V8ArgmaxBot path")
        return
    # pma_global needs a per-position global crop computed from the live board.
    # The play-100-greedy harness is K=1 cluster-based and would have to call
    # compute_global_crop_from_board per ply; defer that to the §169 A3 eval
    # script instead of doubling the wiring here.
    if cfg_pool_type == "pma_global":
        log.info(
            "validation_skipped_pma_global",
            reason="pma_global validation runs under scripts/eval_a3_pma_global.sh",
        )
        return
    # §170 P3 — gpool_bias_active also needs per-position global crops via
    # compute_global_crop_from_board. Defer to scripts/eval_gpool_bias.sh.
    if cfg_gpool_bias_active:
        log.info(
            "validation_skipped_gpool_bias_active",
            reason="gpool_bias validation runs under scripts/eval_gpool_bias.sh",
        )
        return

    # Play 100 greedy games vs RandomBot — expect high win rate after pretraining.
    # §173 A6: Board constructed via registry (Board.with_encoding_name) — no
    # triple-setter. Encoding params (radius, cluster window/threshold) come
    # from the registry entry for cfg_encoding.
    random_bot = RandomBot()
    wins = 0
    n_games = 100

    for i in range(n_games):
        board = Board.with_encoding_name(cfg_encoding)
        state = GameState.from_board(board)
        model_player = 1 if i % 2 == 0 else -1

        for _ in range(200):
            if board.check_win() or board.legal_move_count() == 0:
                break

            if board.current_player == model_player:
                tensor, centers = state.to_tensor()
                # Aug-only K-aggregation site. Live training/inference forwards ALL
                # K cluster views through the network: min-pool on value, scatter-max
                # on policy (worker_loop.rs:299-401 MCTS forward, 649-682 replay push).
                # This site picks cluster 0 ONLY because the consumer (RandomBot
                # validation) is an aug fixture, not a boundary path. See sprint §164 P1.
                aug_cluster = tensor[0]
                aug_cluster_center = centers[0]
                inp = torch.from_numpy(aug_cluster[list(_enc_spec.kept_plane_indices)]).unsqueeze(0).to(device).float()
                with torch.no_grad():
                    lp, _, _ = loaded_model(inp)
                lp_np = lp[0].cpu().numpy()
                cq, cr = aug_cluster_center
                legal = board.legal_moves()
                best_move, best_score = legal[0], -1e9
                for q, r in legal:
                    wq, wr = q - cq + cfg_half, r - cr + cfg_half
                    if 0 <= wq < cfg_board_size and 0 <= wr < cfg_board_size:
                        score = float(lp_np[wq * cfg_board_size + wr])
                        if score > best_score:
                            best_score, best_move = score, (q, r)
                q, r = best_move
            else:
                q, r = random_bot.get_move(state, board)

            state = state.apply_move(board, q, r)

        if board.winner() == model_player:
            wins += 1

    log.info("validation_complete", wins=wins, games=n_games)
    min_wins = 95
    if wins >= min_wins:
        console.print(f"[green]Validation passed: {wins}/{n_games} wins vs RandomBot[/green]")
    elif wins > 0:
        log.warning("validation_below_threshold", wins=wins, threshold=min_wins, games=n_games)
        console.print(
            f"[yellow]Validation: {wins}/{n_games} wins vs RandomBot "
            f"(below ≥{min_wins} threshold — investigate before proceeding)[/yellow]"
        )
    else:
        console.print(
            "[yellow]Validation: 0 wins vs RandomBot "
            "(expected after very brief training — checkpoint format is correct)[/yellow]"
        )
