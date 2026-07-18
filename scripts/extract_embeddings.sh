#!/bin/bash
set -e

# Catch Ctrl+C and kill background jobs
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
export PRECISION="fp16"
export TEMPORAL_WINDOW=8
export CLIP_BATCH_SIZE=1024
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi

echo "=========================================================="
echo "STAGE 2: Extract Embeddings (Vector Encoding with VLMs)"
echo "=========================================================="

for VLM in "${VLMS[@]}"; do
  MODEL="${VLM%%,*}"
  PRETRAINED="${VLM##*,}"
  if [ "$MODEL" = "$PRETRAINED" ]; then
      PRETRAINED=""
  fi
  
  echo ">>> Extracting features using $MODEL ($PRETRAINED) <<<"
  for i in $(seq 0 $((SHARDS-1))); do
    GPU_ID=$((i % NUM_GPUS))
    uv run python pipeline/extract_embed.py \
      --keyframes $KF_DIR --out $INDEX_DIR --device "cuda:${GPU_ID}" \
      --model "$MODEL" --pretrained "$PRETRAINED" --precision $PRECISION \
      --temporal-window $TEMPORAL_WINDOW \
      --shard-index $i --shard-count $SHARDS --batch-size $CLIP_BATCH_SIZE --limit $LIMIT_PER_SHARD &
  done
  wait
  
  uv run python scripts/clean_ram.py
done

echo "-> Finished embedding!"
