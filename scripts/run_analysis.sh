#!/bin/bash
set -e

# Move to project root directory
cd "$(dirname "$0")/.."

source "$(dirname "$0")/config.sh"

NUM_TASKS=${1:-5}

echo "=========================================================="
echo "Running Retrieval Analysis for top $NUM_TASKS tasks"
echo "=========================================================="

# Prepare retrieve args
RETRIEVE_ARGS=""
if [ "$USE_OCR" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-ocr"; fi
if [ "$USE_OD" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-od"; fi
if [ "$USE_CAPTIONING" = "true" ]; then 
    RETRIEVE_ARGS="$RETRIEVE_ARGS --use-captioning"
    if [ -n "$CAPTION_SCORING_METHOD" ]; then
        RETRIEVE_ARGS="$RETRIEVE_ARGS --caption-scoring-method $CAPTION_SCORING_METHOD"
    fi
fi
if [ "$USE_CLUSTERING" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-clustering"; fi

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
