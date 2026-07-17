#!/bin/bash
set -e

source "$(dirname "$0")/config.sh"

echo "=========================================================="
echo "STAGE 4: Retrieval (Query and generate submission.json)"
echo "=========================================================="
mkdir -p submission
NEXT_ID=$(ls -1q submission 2>/dev/null | grep -E '^[0-9]+$' | wc -l || echo 0)
NEXT_ID=$((NEXT_ID + 1))
SUB_DIR="submission/${NEXT_ID}"
mkdir -p "$SUB_DIR"
OUT_FILE="${SUB_DIR}/submission.json"

# Prepare retrieve args
RETRIEVE_ARGS=""
if [ "$USE_OCR" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-ocr"; fi
if [ "$USE_OD" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-od"; fi
if [ "$USE_CAPTIONING" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-captioning"; fi
if [ "$USE_CLUSTERING" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-clustering"; fi

if [ -n "$SMOOTHING_WINDOW" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --smoothing-window $SMOOTHING_WINDOW"; fi
if [ -n "$SMOOTHING_SIGMA" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --smoothing-sigma $SMOOTHING_SIGMA"; fi

uv run python pipeline/retrieve.py \
  --shards $INDEX_DIR/shards \
  --metadata $INDEX_DIR/metadata \
  --tasks $TASKS_FILE \
  --out $OUT_FILE \
  --vlms "${VLMS[@]}" \
  --precision $PRECISION \
  --device "cuda:0" $RETRIEVE_ARGS

# Cleanse RAM after retrieval
uv run python scripts/clean_ram.py

echo "=========================================================="
echo "Zipping submission..."
cd "$SUB_DIR"
zip -r submission.zip submission.json
cd ../..
echo "DONE! Submission file saved and zipped at: ${SUB_DIR}/submission.zip"
echo "=========================================================="
