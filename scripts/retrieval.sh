#!/bin/bash
set -e

trap 'JOBS=$(jobs -p); if [ -n "$JOBS" ]; then kill $JOBS 2>/dev/null || true; fi; exit' SIGINT SIGTERM

export TASKS_FILE="dataset/private_round_tasks.jsonl"
export NUM_TASKS=700
export FPS=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --task-file) TASKS_FILE="$2"; shift ;;
        --num-tasks) NUM_TASKS="$2"; shift ;;
        --fps) FPS="$2"; shift ;;
        --use-sequential) USE_SEQUENTIAL="$2"; shift ;;
        --scene-segmenter) SCENE_SEGMENTER="$2"; shift ;;
        --object-segmenter) OBJECT_SEGMENTER="$2"; shift ;;
        --agg-mode) AGG_MODE="$2"; shift ;;
        --dp-mode) DP_MODE="$2"; shift ;;
        --position-mode) POSITION_MODE="$2"; shift ;;
        --max-preds) MAX_PREDS_PER_VIDEO="$2"; shift ;;
        --clip-to-zero) CLIP_TO_ZERO="$2"; shift ;;
        --main-query-weight) MAIN_QUERY_WEIGHT="$2"; shift ;;
        --split-query) SPLIT_QUERY="$2"; shift ;;
        --lm-cache) LM_CACHE="$2"; shift ;;
        --smoothing-window) SMOOTHING_WINDOW="$2"; shift ;;
        --smoothing-sigma) SMOOTHING_SIGMA="$2"; shift ;;
        --discount-factor) DISCOUNT_FACTOR="$2"; shift ;;
        --max-seq-gap) MAX_SEQ_GAP_MS="$2"; shift ;;
        --overlap-threshold) OVERLAP_THRESHOLD="$2"; shift ;;
        --sliding-sim-threshold) SLIDING_SIM_THRESHOLD="$2"; shift ;;
        --num-chunks) NUM_CHUNKS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ "$FPS" = "0" ]; then
  export METHOD="keyframe"
else
  export METHOD="fps/$FPS"
fi

export KF_DIR="keyframes/$METHOD"
export INDEX_DIR="artifacts/index/$METHOD"

export VLMS=(
  "PE-Core-bigG-14-448,meta"
  
  # 1. Dòng OpenCLIP Tiêu chuẩn
  # "ViT-B-32,laion2b_s34b_b79k"
  # "ViT-B-16,laion2b_s34b_b88k"
  
  # 2. Dòng OpenCLIP Lớn
  # "ViT-L-14,laion2b_s32b_b82k"
  # "ViT-H-14,laion2b_s32b_b79k"
  
  # 3. Dòng SigLIP
  # "ViT-B-16-SigLIP,webli"
  # "ViT-SO400M-14-SigLIP,webli"
  
  # 4. Dòng EVA-CLIP
  # "EVA02-B-16,merged2b_s8b_b131k"
  # "EVA02-L-14,merged2b_s4b_b131k"
)

export USE_SEQUENTIAL="true"
export SCENE_SEGMENTER="qwen7b"
export OBJECT_SEGMENTER="none"
export AGG_MODE="prod"
export DP_MODE="plus"
export POSITION_MODE="second" # options: "first", "second", "middle", "best", "median"
export MAX_PREDS_PER_VIDEO="10"
export CLIP_TO_ZERO="true"
export MAIN_QUERY_WEIGHT="3.0"
export SPLIT_QUERY="true"
export LM_CACHE="true"
export SMOOTHING_WINDOW=0
export SMOOTHING_SIGMA=1.0
export DISCOUNT_FACTOR="0.8"
export MAX_SEQ_GAP_MS=6000
export OVERLAP_THRESHOLD=0000
export SLIDING_SIM_THRESHOLD="0.95"

echo "=========================================================="
echo "Running Retrieval for all queries in $TASKS_FILE"
echo "=========================================================="

RETRIEVE_ARGS=""
if [ "$USE_SEQUENTIAL" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-sequential"; fi
if [ -n "$SCENE_SEGMENTER" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --scene-segmenter $SCENE_SEGMENTER"; fi
if [ -n "$OBJECT_SEGMENTER" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --object-segmenter $OBJECT_SEGMENTER"; fi
if [ -n "$SMOOTHING_WINDOW" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --smoothing-window $SMOOTHING_WINDOW"; fi
if [ -n "$SMOOTHING_SIGMA" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --smoothing-sigma $SMOOTHING_SIGMA"; fi

if [ -n "$OVERLAP_THRESHOLD" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --overlap-threshold $OVERLAP_THRESHOLD"; fi
if [ -n "$NUM_TASKS" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --n $NUM_TASKS"; fi

export NUM_CHUNKS=5
PYTHONPATH=. uv run python scripts/run_chunked.py \
  --tasks $TASKS_FILE \
  --out submission.json \
  --visual-shards $INDEX_DIR/visual \
  --metadata $INDEX_DIR/metadata \
  --vlms "${VLMS[@]}" \
  --device "cuda:0" $RETRIEVE_ARGS

echo "=========================================================="
echo "Retrieval finished!"
echo "=========================================================="
