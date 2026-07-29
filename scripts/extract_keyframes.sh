#!/bin/bash
set -e

# Catch Ctrl+C and kill background jobs to prevent zombie processes
trap 'JOBS=$(jobs -p); if [ -n "$JOBS" ]; then kill $JOBS 2>/dev/null || true; fi; exit' SIGINT SIGTERM

export DATASET_ROOT="dataset/Video_V3C"
export FPS=1
export SHARDS=12
export LIMIT_PER_SHARD=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset-root|--video-root) DATASET_ROOT="$2"; shift ;;
        --fps) FPS="$2"; shift ;;
        --shards) SHARDS="$2"; shift ;;
        --limit) LIMIT_PER_SHARD="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ "$FPS" = "0" ]; then
  export METHOD="keyframe"
else
  export METHOD="fps/$FPS"
fi

export KF_DIR="keyframes/$METHOD"

echo "=========================================================="
echo "STAGE 1: Extract Keyframes (Cut frames from Video)"
echo "=========================================================="

for i in $(seq 0 $((SHARDS-1))); do
  uv run python pipeline/extract_keyframes.py \
    --dataset-root $DATASET_ROOT/V3C1 \
    --dataset-root $DATASET_ROOT/V3C2 \
    --fps $FPS \
    --out $KF_DIR --shard-index $i --shard-count $SHARDS --limit $LIMIT_PER_SHARD &
done
wait

echo "-> Finished keyframe extraction!"
