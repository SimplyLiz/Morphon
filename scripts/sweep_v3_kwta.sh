#!/usr/bin/env bash
# sweep_v3_kwta.sh — sweep mnist_v2 across multiple seeds, capturing the
# paired V2 (GlobalKWTA) vs V3 (LocalInhibition) comparison.
#
# Each run produces a self-describing JSON in docs/benchmark_results/v$VER/.
# Use scripts/aggregate_v3_kwta.py to summarize them after the sweep.
#
# Usage:
#   scripts/sweep_v3_kwta.sh quick     # ~75 min for 5 seeds
#   scripts/sweep_v3_kwta.sh standard  # ~5+ hours for 5 seeds
#   scripts/sweep_v3_kwta.sh fast      # quick smoke test, ~30 min for 5 seeds

set -euo pipefail

PROFILE="${1:-quick}"
# 3 seeds keeps quick-profile sweep under 2h while still giving mean+spread.
# Bump to 5 once we have a hot loop that's faster.
SEEDS=(42 43 44)

case "$PROFILE" in
    fast|quick|standard|extended) ;;
    *) echo "usage: $0 {fast|quick|standard|extended}" >&2; exit 1 ;;
esac

PROFILE_FLAG=""
if [ "$PROFILE" != "quick" ]; then
    PROFILE_FLAG="--$PROFILE"
fi

LOG_DIR="/tmp/sweep_v3_kwta_$(date +%s)"
mkdir -p "$LOG_DIR"

echo "=== sweep_v3_kwta: profile=$PROFILE, seeds=${SEEDS[*]} ==="
echo "=== logs in $LOG_DIR ==="
echo

# Build once up front
cargo build --release --example mnist_v2 2>&1 | tail -3

START=$(date +%s)
for SEED in "${SEEDS[@]}"; do
    echo "--- seed=$SEED ---"
    LOG="$LOG_DIR/seed_${SEED}.log"
    SECONDS=0
    if [ -n "$PROFILE_FLAG" ]; then
        ./target/release/examples/mnist_v2 "$PROFILE_FLAG" --seed="$SEED" >"$LOG" 2>&1
    else
        ./target/release/examples/mnist_v2 --seed="$SEED" >"$LOG" 2>&1
    fi
    ELAPSED=$SECONDS
    # Pull the saved-to path so the aggregator knows which JSON belongs to this run
    SAVED=$(grep "Saved to" "$LOG" | tail -1 | awk '{print $3}')
    V2_LINE=$(grep -E "V2 \(GlobalKWTA\)|V2:" "$LOG" | tail -1)
    V3_LINE=$(grep -E "V3 \(LocalInhib" "$LOG" | tail -1)
    echo "  elapsed=${ELAPSED}s json=$SAVED"
    echo "  $V2_LINE"
    echo "  $V3_LINE"
done
TOTAL=$(($(date +%s) - START))

echo
echo "=== sweep done in ${TOTAL}s ==="
echo "=== aggregate with: python3 scripts/aggregate_v3_kwta.py docs/benchmark_results/v4.0.0/ ==="
