#!/bin/bash
set -e

source "$(dirname "$0")/config.sh"

echo "=========================================================="
echo "STAGE 2: Extract Embeddings (Vector Encoding with VLMs)"
echo "=========================================================="

for VLM in "${VLMS[@]}"; do
  # parse model and pretrained
  MODEL="${VLM%%,*}"
  PRETRAINED="${VLM##*,}"
  if [ "$MODEL" = "$PRETRAINED" ]; then
      PRETRAINED=""
  fi
  
  echo ">>> Extracting features using $MODEL ($PRETRAINED) <<<"
  # Run multiple embedding processes in parallel, distributing across GPUs
  for i in $(seq 0 $((SHARDS-1))); do
    GPU_ID=$((i % NUM_GPUS))
    uv run python pipeline/extract_embed.py \
      --keyframes $KF_DIR --out $INDEX_DIR --device "cuda:${GPU_ID}" \
      --model "$MODEL" --pretrained "$PRETRAINED" --precision $PRECISION \
      --shard-index $i --shard-count $SHARDS --batch-size $CLIP_BATCH_SIZE --limit $LIMIT_PER_SHARD &
  done
  wait
  
  # Free memory before next model
  uv run python scripts/clean_ram.py
done

echo "-> Finished embedding!"
