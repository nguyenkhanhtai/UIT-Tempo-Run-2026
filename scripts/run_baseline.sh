#!/bin/bash
set -e

# Trap Ctrl+C (SIGINT) signal and kill all child processes
trap 'echo "Cleaning up background processes..."; kill $(jobs -p) 2>/dev/null; exit 1' SIGINT

# Move to project root directory
cd "$(dirname "$0")/.."

# Directory paths
DATASET_ROOT="dataset/Video_V3C"
KF_DIR="keyframes"
INDEX_DIR="artifacts/index"
TASKS_FILE="dataset/Public_round_tasks.jsonl"
OUT_FILE="submission.json"

# Auto-detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi
echo "Detected $NUM_GPUS GPU(s)."

# model config
MODEL="PE-Core-bigG-14-448"
PRETRAINED="meta"
PRECISION="fp16"
SHARDS=4
BATCH_SIZE=116

echo "=========================================================="
echo "STAGE 1: Extract Keyframes (Cut frames from Video)"
echo "=========================================================="
# Run multiple processes (shards) in parallel to speed up extraction
for i in $(seq 0 $((SHARDS-1))); do
  uv run python baseline/extract_keyframes.py \
    --dataset-root $DATASET_ROOT/V3C1 \
    --dataset-root $DATASET_ROOT/V3C2 \
    --out $KF_DIR --shard-index $i --shard-count $SHARDS &
done
wait
echo "-> Finished keyframe extraction!"

echo "=========================================================="
echo "STAGE 2: Extract Embeddings (Vector Encoding with $MODEL)"
echo "=========================================================="
# Run multiple embedding processes in parallel, distributing across GPUs
for i in $(seq 0 $((SHARDS-1))); do
  GPU_ID=$((i % NUM_GPUS))
  uv run python baseline/extract_embed.py \
    --keyframes $KF_DIR --out $INDEX_DIR --device "cuda:${GPU_ID}" \
    --model $MODEL --pretrained $PRETRAINED --precision $PRECISION \
    --shard-index $i --shard-count $SHARDS --batch-size $BATCH_SIZE &
done
wait
echo "-> Finished embedding!"

# Dọn dẹp RAM/VRAM sau khi hoàn thành nhúng vector
uv run python scripts/clean_ram.py

echo "=========================================================="
echo "STAGE 3: Retrieval (Query and generate submission.json)"
echo "=========================================================="
uv run python baseline/retrieve.py \
  --shards $INDEX_DIR/shards \
  --tasks $TASKS_FILE \
  --out $OUT_FILE \
  --model $MODEL --pretrained $PRETRAINED --precision $PRECISION \
  --device "cuda:0"

# Cleanse RAM after retrieval
uv run python scripts/clean_ram.py

echo "=========================================================="
echo "DONE! Submission file saved at: $OUT_FILE"
echo "=========================================================="
