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
scraper_output=$(python python/bootstrap/scraper.py --pages 5 --page-size 99 2>/dev/null)

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

# Write manifest.json — tiny metadata file, committed to git
MANIFEST_FILE="${REPO_ROOT}/data/corpus/manifest.json"
last_updated=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

# Derive date range from filenames: game JSONs contain a "created_at" field.
# Fall back to empty strings if no files exist or jq is unavailable.
oldest=""
newest=""
if command -v jq &>/dev/null && [[ ${after} -gt 0 ]]; then
    # Extract created_at from all cached game files, sort, take first and last.
    oldest=$(find "${CACHE_DIR}" -maxdepth 1 -name "*.json" \
        -exec jq -r '.created_at // empty' {} \; 2>/dev/null \
        | sort | head -1)
    newest=$(find "${CACHE_DIR}" -maxdepth 1 -name "*.json" \
        -exec jq -r '.created_at // empty' {} \; 2>/dev/null \
        | sort | tail -1)
fi

cat > "${MANIFEST_FILE}" <<JSON
{
  "last_updated": "${last_updated}",
  "total_games": ${after},
  "date_range": { "oldest": "${oldest}", "newest": "${newest}" },
  "filter": { "rated": true, "min_moves": 20, "reason": "six-in-a-row" }
}
JSON
