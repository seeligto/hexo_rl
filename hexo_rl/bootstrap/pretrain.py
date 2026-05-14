"""
Bootstrap pretraining pipeline — Phase 4.0 architecture.

Loads human + bot corpus, applies quality/source weighting, and trains the
HexTacToeNet with Phase 4.0 losses:

    L = L_policy + L_value_BCE + aux_weight × L_opp_reply

Saves a checkpoint in the same format as Trainer.save_checkpoint() so that
scripts/train.py can resume from it seamlessly.

Usage:
    python -m python.bootstrap.pretrain [--epochs N] [--steps N] [--batch-size N]
    make pretrain          # 15 epochs

§176 P39 — split into 6 modules; this file is a re-export shim preserving
the public surface (BootstrapTrainer, AugmentedBootstrapDataset,
make_augmented_collate, _game_winner_from_replay, load_corpus,
_apply_finetune_freeze, validate, pretrain) plus the historical
module-level dir constants (RAW_HUMAN_DIR / BOT_GAMES_DIR / INJECTED_DIR)
and dependency re-exports (replay_game_to_triples, emit_event) that
external code / tests may patch via `hexo_rl.bootstrap.pretrain.X`.

Module split:
  - pretrain_dataset.py  — AugmentedBootstrapDataset + collate + winner helper
  - pretrain_legacy.py   — load_corpus (legacy raw-JSON path)
  - pretrain_trainer.py  — BootstrapTrainer
  - pretrain_freeze.py   — _apply_finetune_freeze
  - pretrain_validate.py — validate
  - pretrain_cli.py      — pretrain() main + argparse
"""

from __future__ import annotations

# ── Public surface re-exports ─────────────────────────────────────────────────
from hexo_rl.bootstrap.pretrain_dataset import (
    AugmentedBootstrapDataset,
    _game_winner_from_replay,
    make_augmented_collate,
)
from hexo_rl.bootstrap.pretrain_legacy import load_corpus
from hexo_rl.bootstrap.pretrain_trainer import BootstrapTrainer
from hexo_rl.bootstrap.pretrain_freeze import _apply_finetune_freeze
from hexo_rl.bootstrap.pretrain_validate import validate
from hexo_rl.bootstrap.pretrain_cli import pretrain

# ── Patch-surface re-exports (for tests that monkeypatch this module) ─────────
# INV23 / test_pretrain_events historically patched names on this module. We
# also re-export the underlying modules for tests that need to patch them by
# their new home (preferred post-split).
from hexo_rl.bootstrap.dataset import replay_game_to_triples  # noqa: F401
from hexo_rl.bootstrap.generate_corpus import (  # noqa: F401
    BOT_GAMES_DIR,
    INJECTED_DIR,
    RAW_HUMAN_DIR,
)
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.monitoring.events import emit_event  # noqa: F401

_V6 = _lookup_encoding("v6")
BOARD_SIZE: int = _V6.board_size
BUFFER_CHANNELS: int = _V6.n_planes


__all__ = [
    "AugmentedBootstrapDataset",
    "BootstrapTrainer",
    "BOARD_SIZE",
    "BUFFER_CHANNELS",
    "_apply_finetune_freeze",
    "_game_winner_from_replay",
    "load_corpus",
    "make_augmented_collate",
    "pretrain",
    "validate",
]


if __name__ == "__main__":
    pretrain()
