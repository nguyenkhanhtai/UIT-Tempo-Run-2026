#!/bin/bash
set -e

trap 'JOBS=$(jobs -p); if [ -n "$JOBS" ]; then kill $JOBS 2>/dev/null || true; fi; exit' SIGINT SIGTERM

export TASKS_FILE="dataset/synthetic_tasks_v0.jsonl"
export GT_CSV="dataset/synthetic_eval_labels_v0.csv"
export FPS=1

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
  
  # 5. Dòng Video VLM (X-CLIP)
  # "microsoft/xclip-base-patch32,none"
  # "microsoft/xclip-large-patch14,none"
)

export USE_SEQUENTIAL="true"
export SCENE_SEGMENTER="qwen"
export MAIN_QUERY_WEIGHT="1.0"
export OBJECT_SEGMENTER="none"
export SPLIT_QUERY="true"
export LM_CACHE="false"
export USE_AUDIO="false"
export SMOOTHING_WINDOW=1
export SMOOTHING_SIGMA=1.0
export DISCOUNT_FACTOR="0.8"

echo "=========================================================="
echo "Running Validation on file: $TASKS_FILE"
echo "=========================================================="

RETRIEVE_ARGS=""
if [ "$USE_SEQUENTIAL" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-sequential"; fi
if [ -n "$SCENE_SEGMENTER" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --scene-segmenter $SCENE_SEGMENTER"; fi
if [ -n "$OBJECT_SEGMENTER" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --object-segmenter $OBJECT_SEGMENTER"; fi
if [ "$USE_AUDIO" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-audio"; fi
if [ -n "$SMOOTHING_WINDOW" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --smoothing-window $SMOOTHING_WINDOW"; fi
if [ -n "$SMOOTHING_SIGMA" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --smoothing-sigma $SMOOTHING_SIGMA"; fi

PYTHONPATH=. uv run python pipeline/retrieve.py \
  --tasks $TASKS_FILE \
  --out submission.json \
  --visual-shards $INDEX_DIR/visual \
  --audio-shards $INDEX_DIR/audio \
  --metadata $INDEX_DIR/metadata \
  --vlms "${VLMS[@]}" \
  --device "cuda:0" $RETRIEVE_ARGS

echo "=========================================================="
echo "Validation finished! Checking scores..."
echo "=========================================================="

PYTHONPATH=. uv run python pipeline/score.py --sub submission.json --gt "$GT_CSV"
