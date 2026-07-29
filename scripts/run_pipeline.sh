#!/bin/bash
set -e

# Trap Ctrl+C (SIGINT) signal and kill all child processes
trap 'echo "Cleaning up background processes..."; kill $(jobs -p) 2>/dev/null; exit 1' SIGINT

SCRIPT_DIR="$(dirname "$0")"

echo "Starting Pipeline..."

bash "$SCRIPT_DIR/extract_keyframes.sh" \
    --video-root dataset/Video_V3C \
    --fps 1 \
    --shards 6

bash "$SCRIPT_DIR/extract_embeddings.sh" \
    --fps 1 \
    --batch-size 256 \
    --shards 2

bash "$SCRIPT_DIR/extract_metadata.sh"

bash "$SCRIPT_DIR/retrieval.sh" \
    --task-file dataset/private_round_tasks.jsonl \
    --num-tasks 700 \
    --fps 1 \
    --use-sequential "true" \
    --scene-segmenter qwen7b \
    --object-segmenter "none" \
    --agg-mode "prod" \
    --dp-mode "plus" \
    --position-mode "second" \
    --max-preds 10 \
    --clip-to-zero "true" \
    --main-query-weight "3.0" \
    --split-query "true" \
    --lm-cache "true" \
    --smoothing-window 0 \
    --smoothing-sigma 1.0 \
    --discount-factor "0.8" \
    --max-seq-gap 6000 \
    --overlap-threshold 0000 \
    --sliding-sim-threshold "0.95" \
    --num-chunks 5

echo "Pipeline fully completed!"
