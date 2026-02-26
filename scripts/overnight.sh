#!/bin/bash
# scripts/overnight.sh - macOS compatible
# Run from repo root: ./scripts/overnight.sh

set -e

# Change to repo root (parent of scripts/)
cd "$(dirname "$0")/.."

echo "=============================================="
echo "  FYP Clone Analysis"
echo "  Started: $(date)"
echo "=============================================="

mkdir -p results

echo ""
echo "Starting analysis..."
echo ""

python -m cluain batch repositories.csv \
    --output-dir ./results \
    --clone-dir ./repos \
    --years 6 \
    --fast \
    --threshold 0.95 \
    --min-size 150 \
    --workers 2 \
    2>&1 | tee results/progress.log

echo ""
echo "Generating graphs..."
python generate_graphs.py

echo ""
echo "Running statistical analysis..."
python analyse_results.py

echo ""
echo "=============================================="
echo "  DONE: $(date)"
echo "=============================================="