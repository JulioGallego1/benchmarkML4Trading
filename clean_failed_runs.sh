#!/usr/bin/env bash
# Delete empty run directories for LSTM/PATCHTST with step=32 and H<32.
# A directory is considered failed/empty if it contains no metrics.json.
#
# Usage:
#   bash clean_failed_runs.sh [--dry-run] [RUNS_DIR]
#
# Defaults:
#   RUNS_DIR = ~/benchmarkML4Trading/runs
#   Without --dry-run the matching directories are deleted.

set -euo pipefail

DRY_RUN=0
RUNS_DIR="${HOME}/benchmarkML4Trading/runs"

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) RUNS_DIR="$arg" ;;
    esac
done

if [[ ! -d "$RUNS_DIR" ]]; then
    echo "ERROR: runs directory not found: $RUNS_DIR" >&2
    exit 1
fi

echo "Scanning: $RUNS_DIR"
[[ $DRY_RUN -eq 1 ]] && echo "(dry-run — nothing will be deleted)"
echo

count=0

for dir in "$RUNS_DIR"/*/; do
    name=$(basename "$dir")

    # Must be LSTM or PATCHTST
    if [[ ! "$name" =~ ^(LSTM|PATCHTST)_ ]]; then
        continue
    fi

    # Must contain step32
    if [[ ! "$name" =~ _step32_ ]]; then
        continue
    fi

    # Extract H value from _H<number>_ pattern
    if [[ "$name" =~ _H([0-9]+)_ ]]; then
        h="${BASH_REMATCH[1]}"
    else
        continue
    fi

    # Only target H < 32
    if [[ "$h" -ge 32 ]]; then
        continue
    fi

    # Only delete if metrics.json is missing (failed run)
    if [[ -f "${dir}metrics.json" ]]; then
        continue
    fi

    echo "  REMOVE: $name"
    if [[ $DRY_RUN -eq 0 ]]; then
        rm -rf "$dir"
    fi
    (( count++ )) || true
done

echo
if [[ $DRY_RUN -eq 1 ]]; then
    echo "Would delete $count director(ies)."
else
    echo "Deleted $count director(ies)."
fi
