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

# --- Pass 1: standard incremental scrape ---
scraper_output=$(python -m hexo_rl.bootstrap.scraper --pages 5 --page-size 99 2>/dev/null)

after_pass1=$(find "${CACHE_DIR}" -maxdepth 1 -name "*.json" | wc -l)
passed=$(printf '%s' "${scraper_output}" | grep -oE 'Scraped [0-9]+' | grep -oE '[0-9]+' || echo "?")
new_standard=$(( after_pass1 - before ))

date_str=$(date '+%Y-%m-%d')
summary_std="[${date_str}] [standard] New games: ${new_standard} | Total cached: ${after_pass1} | Passed filter: ${passed}"
echo "${summary_std}"

mkdir -p "$(dirname "${LOG_FILE}")"
echo "${summary_std}" >> "${LOG_FILE}"

# --- Pass 2: top-player targeted scrape ---
top_output=$(python -m hexo_rl.bootstrap.scraper --pages 25 --page-size 20 --top-players-only --top-n 20 2>/dev/null)

after_pass2=$(find "${CACHE_DIR}" -maxdepth 1 -name "*.json" | wc -l)
top_passed=$(printf '%s' "${top_output}" | grep -oE 'Scraped [0-9]+' | grep -oE '[0-9]+' || echo "?")
new_top=$(( after_pass2 - after_pass1 ))

summary_top="[${date_str}] [top-player] New games: ${new_top} | Total cached: ${after_pass2} | Passed filter: ${top_passed}"
echo "${summary_top}"
echo "${summary_top}" >> "${LOG_FILE}"

# Update manifest.json — scans all corpus dirs (human + bot), atomic write
python "${REPO_ROOT}/scripts/update_manifest.py"
