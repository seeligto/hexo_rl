#!/usr/bin/env bash
# scrape_daily.sh — daily incremental scrape of hexo.did.science
#
# Usage:   ./scripts/scrape_daily.sh
# Cron:    0 9 * * * /path/to/scripts/scrape_daily.sh
#
# Deduplication is handled by the scraper itself: game UUIDs that already have
# a .json file in data/corpus/raw_human/ are never re-fetched.  New games is
# measured as the change in that file count.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_DIR="${REPO_ROOT}/data/corpus/raw_human"
LOG_FILE="${REPO_ROOT}/logs/scrape_history.log"
VENV="${REPO_ROOT}/.venv"

cd "${REPO_ROOT}"

# Activate virtualenv if present
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi

# Count cached UUIDs before run
mkdir -p "${CACHE_DIR}"
before=$(find "${CACHE_DIR}" -maxdepth 1 -name "*.json" | wc -l)

# Run scraper; structlog JSON goes to stderr, summary line goes to stdout
scraper_output=$(python -m hexo_rl.bootstrap.scraper --pages 5 --page-size 99 2>/dev/null)

# Count cached UUIDs after run
after=$(find "${CACHE_DIR}" -maxdepth 1 -name "*.json" | wc -l)

# Parse total unique games passing quality filter from scraper stdout
# Expected format: "Scraped N unique games."
passed=$(printf '%s' "${scraper_output}" | grep -oE 'Scraped [0-9]+' | grep -oE '[0-9]+' || echo "?")

new_games=$(( after - before ))
date_str=$(date '+%Y-%m-%d')
summary="[${date_str}] New games: ${new_games} | Total cached: ${after} | Passed filter: ${passed}"

echo "${summary}"

mkdir -p "$(dirname "${LOG_FILE}")"
echo "${summary}" >> "${LOG_FILE}"

# Update manifest.json — scans all corpus dirs (human + bot), atomic write
python "${REPO_ROOT}/scripts/update_manifest.py"
