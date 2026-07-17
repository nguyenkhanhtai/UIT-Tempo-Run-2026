#!/bin/bash
set -e

CSV_FILE=${1:-"dataset/synthetic_eval_labels.csv"}
# Lấy tên file không có đuôi để tạo file jsonl tương ứng
BASENAME=$(basename "$CSV_FILE" .csv)
JSONL_FILE="dataset/${BASENAME}_temp.jsonl"

echo "=========================================================="
echo "STAGE 1: Preparing Synthetic Evaluation Tasks"
echo "=========================================================="
# Convert CSV to JSONL so retrieve.py can read it
.venv/bin/python -c "
import csv, json
with open('$CSV_FILE', 'r') as f:
    reader = csv.DictReader(f)
    with open('$JSONL_FILE', 'w') as out:
        for row in reader:
            task = {
                'task_id': row['task_id'], 
                'description': row['description'], 
                'submission_type': 'temporal_video_retrieval', 
                'max_predictions': 10
            }
            out.write(json.dumps(task) + '\n')
"
echo "Generated $JSONL_FILE"

echo "=========================================================="
echo "STAGE 2: Running Retrieval"
echo "=========================================================="
# Source config but override the TASKS_FILE
source scripts/config.sh
export TASKS_FILE="$JSONL_FILE"

RETRIEVE_ARGS=""
if [ "$USE_OCR" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-ocr"; fi
if [ "$USE_OD" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-od"; fi
if [ "$USE_CAPTIONING" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-captioning"; fi
if [ "$USE_CLUSTERING" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-clustering"; fi
if [ "$USE_CATEGORY" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-category"; fi
if [ -n "$SMOOTHING_WINDOW" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --smoothing-window $SMOOTHING_WINDOW"; fi
if [ -n "$SMOOTHING_SIGMA" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --smoothing-sigma $SMOOTHING_SIGMA"; fi

mkdir -p submission/synthetic_eval
OUT_FILE="submission/synthetic_eval/${BASENAME}_submission.json"

.venv/bin/python pipeline/retrieve.py \
  --shards $INDEX_DIR/shards \
  --metadata $INDEX_DIR/metadata \
  --tasks $TASKS_FILE \
  --out $OUT_FILE \
  --vlms "${VLMS[@]}" \
  --precision $PRECISION \
  --device "cuda:0" $RETRIEVE_ARGS

echo "=========================================================="
echo "STAGE 3: Scoring Results"
echo "=========================================================="

.venv/bin/python pipeline/score.py --sub "$OUT_FILE" --gt "$CSV_FILE"

echo "=========================================================="
echo "EVALUATION COMPLETE"
echo "=========================================================="
