#!/bin/bash
set -e

trap 'JOBS=$(jobs -p); if [ -n "$JOBS" ]; then kill $JOBS 2>/dev/null || true; fi; exit' SIGINT SIGTERM

export FPS=0
export SHARDS=2
export LIMIT_PER_SHARD=0

if [ "$FPS" = "0" ]; then
  export METHOD="keyframe"
else
  export METHOD="fps/$FPS"
fi

export KF_DIR="keyframes/$METHOD"
export INDEX_DIR="artifacts/index/$METHOD"

export USE_OCR="false"
export OCR_ENGINE="rapidocr"
export OCR_MODEL="rapidocr"
export OCR_BATCH_SIZE=4

export USE_OD="false"
export OD_ENGINE="yolo"
export OD_MODEL="yolo11m.pt"
export OD_BATCH_SIZE=4

export USE_CAPTIONING="false"
export CAPTION_ENGINE="florence2"
export CAPTION_MODEL="microsoft/Florence-2-large"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi

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
uv run python scripts/clean_ram.py
