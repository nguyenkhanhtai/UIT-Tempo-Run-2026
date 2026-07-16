#!/bin/bash
set -e

# Move to project root directory
cd "$(dirname "$0")/.."

DATASET_ROOT="dataset/Video_V3C"
KF_DIR="keyframes"
INDEX_DIR="artifacts/index"
TASKS_FILE="dataset/Public_round_tasks.jsonl"
NUM_TASKS=${1:-5}

# Default model, same as run_pipeline.sh
VLMS=(
  "ViT-B-32,laion2b_s34b_b79k"
)

echo "=========================================================="
echo "Running Retrieval Analysis for top $NUM_TASKS tasks"
echo "=========================================================="

PYTHONPATH=. uv run python pipeline/utils/analysis.py \
  --tasks $TASKS_FILE \
  --n $NUM_TASKS \
  --shards $INDEX_DIR/shards \
  --metadata $INDEX_DIR/metadata \
  --keyframes $KF_DIR \
  --vlms "${VLMS[@]}" \
  --device "cuda:0"

echo "=========================================================="
echo "Analysis finished! Check figures/analysis/ for results."
echo "=========================================================="
