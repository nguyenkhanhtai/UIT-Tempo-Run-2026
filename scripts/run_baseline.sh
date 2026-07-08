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
DEV="cuda:0"

# Lightweight model config (ViT-B-32) to save memory
MODEL="ViT-B-32"
PRETRAINED="laion2b_s34b_b79k"
PRECISION="fp16"
SHARDS=2

# echo "=========================================================="
# echo "STAGE 1: Extract Keyframes (Cut frames from Video)"
# echo "=========================================================="
# # Run multiple processes (shards) in parallel to speed up extraction
# for i in $(seq 0 $((SHARDS-1))); do
#   uv run python baseline/extract_keyframes.py \
#     --dataset-root $DATASET_ROOT/V3C1 \
#     --dataset-root $DATASET_ROOT/V3C2 \
#     --out $KF_DIR --shard-index $i --shard-count $SHARDS &
# done
# wait
# echo "-> Finished keyframe extraction!"

echo "=========================================================="
echo "STAGE 2: Extract Embeddings (Vector Encoding with $MODEL)"
echo "=========================================================="
# Run multiple embedding processes in parallel
for i in $(seq 0 $((SHARDS-1))); do
  uv run python baseline/extract_embed.py \
    --keyframes $KF_DIR --out $INDEX_DIR --device $DEV \
    --model $MODEL --pretrained $PRETRAINED --precision $PRECISION \
    --shard-index $i --shard-count $SHARDS &
done
wait
echo "-> Finished embedding!"

echo "=========================================================="
echo "STAGE 3: Retrieval (Query and generate submission.json)"
echo "=========================================================="
uv run python baseline/retrieve.py \
  --shards $INDEX_DIR/shards \
  --tasks $TASKS_FILE \
  --out $OUT_FILE \
  --model $MODEL --pretrained $PRETRAINED --precision $PRECISION \
  --device $DEV

echo "=========================================================="
echo "DONE! Submission file saved at: $OUT_FILE"
echo "=========================================================="
