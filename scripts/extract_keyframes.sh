#!/bin/bash
set -e

# Source the configuration
source "$(dirname "$0")/config.sh"

echo "=========================================================="
echo "STAGE 1: Extract Keyframes (Cut frames from Video)"
echo "=========================================================="

# Run multiple processes (shards) in parallel to speed up extraction
for i in $(seq 0 $((SHARDS-1))); do
  uv run python pipeline/extract_keyframes.py \
    --dataset-root $DATASET_ROOT/V3C1 \
    --dataset-root $DATASET_ROOT/V3C2 \
    --fps $FPS \
    --out $KF_DIR --shard-index $i --shard-count $SHARDS --limit $LIMIT_PER_SHARD &
done
wait

echo "-> Finished keyframe extraction!"
