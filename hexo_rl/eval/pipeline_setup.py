"""Build the EvalPipeline from eval.yaml + per-variant overrides.

Sweep / variant configs may declare an ``eval_pipeline:`` block to override
eval cost (n_games, model_sims, opponent enables, eval_interval). This
module deep-merges that block onto the eval.yaml load before constructing
EvalPipeline; without that merge, sweep mode would pay the production
~134 min/round eval cost regardless of how short the sweep is.

Extracted from training/loop.py per §159 refactor. No behavior change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import structlog
import torch

from hexo_rl.eval.eval_pipeline import EvalPipeline

log = structlog.get_logger(__name__)


def build_eval_pipeline(
    config: dict[str, Any],
    device: torch.device,
    run_id: str,
    train_cfg: dict[str, Any],
) -> tuple[Optional[EvalPipeline], dict[str, Any], int]:
    """Load eval.yaml, apply variant override, build EvalPipeline.

    Returns ``(eval_pipeline, eval_ext_config, eval_interval)``:
      - ``eval_pipeline`` is None when ``configs/eval.yaml`` is absent or its
        ``eval_pipeline.enabled`` flag is False.
      - ``eval_ext_config`` is the merged eval.yaml content (empty dict if
        the file is absent). Caller reads ``gating.best_model_path`` from it.
      - ``eval_interval`` is the cadence to use for eval ticks. Falls back
        to ``train_cfg["eval_interval"]`` → ``config["eval_interval"]`` →
        100 when no eval pipeline is built.
    """
    from hexo_rl.utils.config import load_config as _load_config, _deep_merge

    eval_yaml_path = Path("configs/eval.yaml")
    eval_pipeline: EvalPipeline | None = None
    eval_ext_config: dict[str, Any] = {}
    eval_interval: int = int(
        train_cfg.get("eval_interval", config.get("eval_interval", 100))
    )

    if eval_yaml_path.exists():
        eval_ext_config = _load_config(str(eval_yaml_path))
        # Sweep / variant configs may declare an `eval_pipeline:` block to
        # override eval cost (n_games, model_sims, opponent enables, eval_interval).
        # Deep-merge it onto the eval.yaml load so the produced EvalPipeline
        # honours the variant. Without this, sweep mode pays the production
        # ~134 min/round eval cost regardless of how short the sweep is.
        main_ep_override = config.get("eval_pipeline", {})
        if main_ep_override:
            _deep_merge(
                eval_ext_config.setdefault("eval_pipeline", {}),
                main_ep_override,
            )
        ep_cfg = eval_ext_config.get("eval_pipeline", {})
        if ep_cfg.get("enabled", False):
            eval_pipeline = EvalPipeline(eval_ext_config, device, run_id=run_id)
            eval_interval = int(ep_cfg.get("eval_interval", 1000))
            log.info(
                "eval_pipeline_enabled",
                interval=eval_interval,
                overrides_applied=bool(main_ep_override),
            )

    return eval_pipeline, eval_ext_config, eval_interval
