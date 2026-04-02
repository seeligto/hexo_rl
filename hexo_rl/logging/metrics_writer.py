"""
TensorBoard (and optional wandb) metrics writer.

Wraps torch.utils.tensorboard.SummaryWriter with a simple interface
used by the training loop.

Usage:
    writer = MetricsWriter(log_dir="runs/phase2")
    writer.log_step(step=10, metrics={"policy_loss": 0.4, "value_loss": 0.1})
    writer.close()
"""

from __future__ import annotations

from typing import Any, Dict


class MetricsWriter:
    """Write scalar metrics to TensorBoard (and optionally wandb).

    Args:
        log_dir:   Directory for TensorBoard event files.
        use_wandb: If True, also log to wandb (requires wandb installed).
    """

    def __init__(self, log_dir: str, use_wandb: bool = False) -> None:
        from torch.utils.tensorboard import SummaryWriter
        self._writer   = SummaryWriter(log_dir)
        self._use_wandb = use_wandb
        if use_wandb:
            import wandb
            wandb.init(project="hex-tac-toe-az", dir=log_dir)

    def log_step(self, step: int, metrics: Dict[str, Any]) -> None:
        """Log scalar metrics at `step`.

        Non-numeric values are silently ignored.
        """
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                self._writer.add_scalar(key, val, step)
        if self._use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def log_game_record(self, step: int, game_text: str) -> None:
        """Log a game record as text (for manual review in TensorBoard)."""
        self._writer.add_text("eval_game", game_text, step)

    def close(self) -> None:
        """Flush and close the writer."""
        self._writer.close()
        if self._use_wandb:
            import wandb
            wandb.finish()
