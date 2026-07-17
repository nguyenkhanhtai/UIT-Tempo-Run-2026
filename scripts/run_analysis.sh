#!/bin/bash
set -e

# Move to project root directory
cd "$(dirname "$0")/.."

source scripts/config.sh

DATASET_ROOT="dataset/Video_V3C"
KF_DIR="keyframes"
INDEX_DIR="artifacts/index"
TASKS_FILE="dataset/Public_round_tasks.jsonl"
NUM_TASKS=${1:-5}

# Default model, same as run_pipeline.sh
# Check if VLMS is empty (if not set in config.sh)
if [ ${#VLMS[@]} -eq 0 ]; then
  VLMS=(
    "ViT-B-32,laion2b_s34b_b79k"
  )
fi

echo "=========================================================="
echo "Running Retrieval Analysis for top $NUM_TASKS tasks"
echo "=========================================================="

# Prepare retrieve args
RETRIEVE_ARGS=""
if [ "$USE_OCR" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-ocr"; fi
if [ "$USE_OD" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-od"; fi
if [ "$USE_CAPTIONING" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-captioning"; fi
if [ "$USE_CLUSTERING" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-clustering"; fi

if [ -n "$SMOOTHING_WINDOW" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --smoothing-window $SMOOTHING_WINDOW"; fi
if [ -n "$SMOOTHING_SIGMA" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --smoothing-sigma $SMOOTHING_SIGMA"; fi

PYTHONPATH=. uv run python pipeline/utils/analysis.py \
  --tasks $TASKS_FILE \
  --n $NUM_TASKS \
  --shards $INDEX_DIR/shards \
  --metadata $INDEX_DIR/metadata \
  --keyframes $KF_DIR \
  --vlms "${VLMS[@]}" \
  --device "cuda:0" $RETRIEVE_ARGS

echo "=========================================================="
echo "Analysis finished! Check figures/analysis/ for results."
echo "=========================================================="
