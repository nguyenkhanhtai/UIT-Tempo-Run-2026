#!/bin/bash
set -e

source "$(dirname "$0")/config.sh"

echo "=========================================================="
echo "STAGE 3: Extract Metadata (OCR & YOLO)"
echo "=========================================================="
mkdir -p logs

if [ "$USE_OD" = "true" ]; then
  echo ">>> Extracting OD (Batch size: $OD_BATCH_SIZE)..."
  for i in $(seq 0 $((SHARDS-1))); do
    GPU_ID=$((i % NUM_GPUS))
    uv run python pipeline/extract_metadata.py \
      --task od \
      --keyframes "$KF_DIR" \
      --out "$INDEX_DIR" \
      --device "cuda:${GPU_ID}" \
      --od-engine "$OD_ENGINE" \
      --od-model "$OD_MODEL" \
      --batch-size $OD_BATCH_SIZE \
      --shard-index $i --shard-count $SHARDS --limit $LIMIT_PER_SHARD &
  done
  wait
fi

if [ "$USE_OCR" = "true" ]; then
  echo ">>> Extracting OCR (Batch size: $OCR_BATCH_SIZE)..."
  for i in $(seq 0 $((SHARDS-1))); do
    GPU_ID=$((i % NUM_GPUS))
    uv run python pipeline/extract_metadata.py \
      --task ocr \
      --keyframes "$KF_DIR" \
      --out "$INDEX_DIR" \
      --device "cuda:${GPU_ID}" \
      --ocr-engine "$OCR_ENGINE" \
      --ocr-model "$OCR_MODEL" \
      --batch-size $OCR_BATCH_SIZE \
      --shard-index $i --shard-count $SHARDS --limit $LIMIT_PER_SHARD &
  done
  wait
fi

if [ "$USE_CAPTIONING" = "true" ]; then
  echo ">>> Extracting Captions (Batch size: $OD_BATCH_SIZE)..."
  for i in $(seq 0 $((SHARDS-1))); do
    GPU_ID=$((i % NUM_GPUS))
    uv run python pipeline/extract_metadata.py \
      --task caption \
      --keyframes "$KF_DIR" \
      --out "$INDEX_DIR" \
      --device "cuda:${GPU_ID}" \
      --caption-engine "$CAPTION_ENGINE" \
      --caption-model "$CAPTION_MODEL" \
      --batch-size $OD_BATCH_SIZE \
      --shard-index $i --shard-count $SHARDS --limit $LIMIT_PER_SHARD &
  done
  wait
fi

echo "-> Finished metadata extraction!"

# Dọn dẹp RAM/VRAM
uv run python scripts/clean_ram.py
