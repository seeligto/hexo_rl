#!/usr/bin/env bash
# setup.sh — run once to scaffold the full repo structure
# Usage: bash setup.sh
set -e

echo "→ Creating directory structure..."

mkdir -p docs/reference
mkdir -p native_core/src/{board,mcts,formations}
mkdir -p native_core/benches
mkdir -p python/{model,selfplay,training,bootstrap,eval,opening_book,api,logging}
mkdir -p configs
mkdir -p scripts
mkdir -p tests
mkdir -p vendor

# Touch __init__.py files
touch python/__init__.py
for dir in model selfplay training bootstrap eval opening_book api logging; do
  touch python/$dir/__init__.py
done

echo "→ Cloning community resources into vendor/..."
git clone --depth=1 https://github.com/Ramora0/SealBot vendor/bots/sealbot || \
  echo "  (already cloned or network unavailable)"
git clone --depth=1 https://github.com/Ramora0/HexTacToeBots vendor/bots/httt_collection || \
  echo "  (already cloned or network unavailable)"

echo "→ Fetching community specs into docs/reference/..."
curl -sL https://raw.githubusercontent.com/hex-tic-tac-toe/htttx-bot-api/main/definitions/bot-api-v1.yaml \
  -o docs/reference/bot-api-v1.yaml || echo "  (could not fetch bot API spec)"
git clone --depth=1 https://github.com/hex-tic-tac-toe/hexagonal-tic-tac-toe-notation \
  docs/reference/notation 2>/dev/null || echo "  (notation repo already cloned)"

echo "→ Done. Next steps:"
echo "   1. Review vendor/bots/sealbot/ README"
echo "   2. Read docs/reference/bot-api-v1.yaml"
echo "   3. Run: claude  (to start Claude Code)"
