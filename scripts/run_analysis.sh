#!/bin/bash
set -e

trap 'JOBS=$(jobs -p); if [ -n "$JOBS" ]; then kill $JOBS 2>/dev/null || true; fi; exit' SIGINT SIGTERM

export FPS=1

if [ "$FPS" = "0" ]; then
  export METHOD="keyframe"
else
  export METHOD="fps/$FPS"
fi

export KF_DIR="keyframes/$METHOD"
export INDEX_DIR="artifacts/index/$METHOD"
export TASKS_FILE=${1:-"dataset/Public_round_tasks.jsonl"}
export NUM_TASKS=${2:-5}
export GT_CSV=${GT_CSV:-"dataset/synthetic_eval_labels_v0.csv"}

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
export USE_AUDIO="true"
export SCENE_SEGMENTER="qwen"
export MAIN_QUERY_WEIGHT="2.0"
export OBJECT_SEGMENTER="none"
export SPLIT_QUERY="true"
export LM_CACHE="false"
export SMOOTHING_WINDOW=1
export SMOOTHING_SIGMA=1.0
export DISCOUNT_FACTOR="0.9"

echo "=========================================================="
echo "Running Retrieval Analysis on file: $TASKS_FILE (top $NUM_TASKS tasks)"
echo "=========================================================="

RETRIEVE_ARGS=""
if [ "$USE_SEQUENTIAL" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-sequential"; fi
if [ "$USE_AUDIO" = "true" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --use-audio"; fi
if [ -n "$SCENE_SEGMENTER" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --scene-segmenter $SCENE_SEGMENTER"; fi
if [ -n "$OBJECT_SEGMENTER" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --object-segmenter $OBJECT_SEGMENTER"; fi
if [ -n "$SMOOTHING_WINDOW" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --smoothing-window $SMOOTHING_WINDOW"; fi
if [ -n "$SMOOTHING_SIGMA" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --smoothing-sigma $SMOOTHING_SIGMA"; fi
if [ -n "$GT_CSV" ]; then RETRIEVE_ARGS="$RETRIEVE_ARGS --gt $GT_CSV"; fi


echo "Cleaning previous analysis outputs: figures/analysis"
rm -rf figures/analysis
mkdir -p figures/analysis

PYTHONPATH=. uv run python pipeline/utils/analysis.py \
  --tasks $TASKS_FILE \
  --n $NUM_TASKS \
  --visual-shards $INDEX_DIR/visual \
  --audio-shards $INDEX_DIR/audio \
  --metadata $INDEX_DIR/metadata \
  --keyframes $KF_DIR \
  --vlms "${VLMS[@]}" \
  --device "cuda:0" $RETRIEVE_ARGS

echo "=========================================================="
echo "Analysis finished! Check figures/analysis/ for results."
echo "=========================================================="
