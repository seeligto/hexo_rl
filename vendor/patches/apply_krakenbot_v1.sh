#!/usr/bin/env bash
# Reproduce the KrakenBot-MCTS eval bar from a fresh clone.
#
# A fresh checkout of hexo_rl does NOT contain vendor/external/krakenbot (it is
# gitignored). This script clones upstream at the PINNED SHA, applies our two
# local edits (krakenbot_v1.patch) and builds the Cython PUCT + MCTS extensions
# so hexo_rl/bots/krakenbot_v1_mcts_bot.py can import _puct_cy / _mcts_cy.
#
# Our patch (krakenbot_v1.patch) carries two edits:
#   1. mcts/tree.py  — _expand_level2 non-square guard: only take the cached
#      nearby-set fast path when the board is the canonical square (n_cells ==
#      N_CELLS); otherwise fall through to the general path. Without it the
#      cached path returns a wrong candidate set on non-square/torus boards.
#   2. mcts_bot.py   — MCTSBot(temperature=…) kwarg threaded to select_move_pair
#      (upstream hardcodes 0.1). temperature=0 => argmax deployment reads.
#
# OPERATOR OPTION: both are arguably upstream bugs. Filing them at
# https://github.com/Ramora0/KrakenBot reveals we benchmark against KrakenBot.
# Left as an operator decision (default: do not file). If upstream merges them,
# drop this patch and bump PIN_SHA.
set -euo pipefail

REPO_URL="https://github.com/Ramora0/KrakenBot"
PIN_SHA="111783aea6211a98ed178322c8a51ce8bf16f1fe"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/krakenbot_v1.patch"
# Default dest relative to the repo root (two levels up from vendor/patches/).
DEST="${1:-$(cd "$HERE/../.." && pwd)/vendor/external/krakenbot}"

if [ ! -d "$DEST/.git" ]; then
  git clone "$REPO_URL" "$DEST"
fi
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$PIN_SHA"
# Idempotent: reset the two touched files to the pinned version, then re-apply.
git -C "$DEST" checkout -- mcts/tree.py mcts_bot.py 2>/dev/null || true
git -C "$DEST" apply --3way "$PATCH"

# Build Cython extensions in-place (PUCT + MCTS). Requires cython + a C compiler
# and the deps in $DEST/requirements.txt.
( cd "$DEST" && python setup.py build_ext --inplace && python setup_puct.py build_ext --inplace )

echo "KrakenBot v1 ready at $DEST"
echo "  pinned SHA: $PIN_SHA"
echo "  patch:      $PATCH"
