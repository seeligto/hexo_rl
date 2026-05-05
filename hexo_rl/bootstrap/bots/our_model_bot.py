"""OurModelBot — wraps our own checkpoint + MCTS as a BotProtocol.

Used for self-evaluation and as a corpus source once the model is strong
enough to contribute useful training data.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.env import GameState
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.worker import SelfPlayWorker
from hexo_rl.training.checkpoints import normalize_model_state_dict_keys
from hexo_rl.training.trainer import Trainer
from hexo_rl.viewer.model_loader import _extract_model_state, _infer_model_hparams


class OurModelBot(BotProtocol):
    """Wraps a trained HexTacToeNet + MCTS as a BotProtocol.

    Args:
        checkpoint_path: Path to a saved model checkpoint (.pt file).
        config:          MCTS/search config dict (same schema as selfplay config).
        device:          Torch device to run inference on.
        temperature:     Sampling temperature (0 = greedy/argmax).
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: Dict[str, Any],
        device: torch.device | None = None,
        temperature: float = 0.0,
    ) -> None:
        if device is None:
            from hexo_rl.utils.device import best_device
            device = best_device()

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = normalize_model_state_dict_keys(_extract_model_state(payload))
        model_hparams = _infer_model_hparams(state_dict)
        # Fall back to config values for any dims not recoverable from weights.
        model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
        board_size = int(model_hparams.get("board_size", model_cfg.get("board_size", config.get("board_size", 19))))
        in_channels = int(model_hparams.get("in_channels", model_cfg.get("in_channels", config.get("in_channels", 8))))
        filters = int(model_hparams.get("filters", model_cfg.get("filters", config.get("filters", 128))))
        res_blocks = int(model_hparams.get("res_blocks", model_cfg.get("res_blocks", config.get("res_blocks", 12))))
        se_reduction_ratio = int(model_hparams.get("se_reduction_ratio", model_cfg.get("se_reduction_ratio", config.get("se_reduction_ratio", 4))))

        use_hex_kernel = bool(
            model_cfg.get("use_hex_kernel", config.get("use_hex_kernel", False))
        )
        net = HexTacToeNet(
            board_size=board_size,
            in_channels=in_channels,
            filters=filters,
            res_blocks=res_blocks,
            se_reduction_ratio=se_reduction_ratio,
            use_hex_kernel=use_hex_kernel,
        )
        Trainer._load_state_dict_strict(net, state_dict)
        net.to(device).eval()

        self._worker = SelfPlayWorker(model=net, config=config, device=device)
        self._temperature = temperature
        self._checkpoint_path = checkpoint_path

    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        policy = self._worker._run_mcts(
            rust_board,
            use_dirichlet=False,
            temperature=self._temperature if self._temperature > 0 else None,
        )
        legal = rust_board.legal_moves()
        if not legal:
            raise RuntimeError("OurModelBot.get_move called with no legal moves")
        q, r = self._worker._sample_action(policy, legal, rust_board)
        return q, r

    def name(self) -> str:
        import os
        base = os.path.basename(self._checkpoint_path)
        return f"our_model({base})"
