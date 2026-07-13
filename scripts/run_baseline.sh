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
VLMS=(
  # "PE-Core-bigG-14-448,meta"
  
  # --- CÁC MÔ HÌNH VLM GỢI Ý ĐỂ ENSEMBLE (Mở comment để sử dụng) ---
  
  # 1. Dòng OpenCLIP Tiêu chuẩn (Nhẹ, cân bằng)
  "ViT-B-32,laion2b_s34b_b79k"
  "ViT-B-16,laion2b_s34b_b88k"
  
  # 2. Dòng OpenCLIP Lớn (Nặng hơn, cực kỳ chính xác cho ngữ nghĩa)
  # "ViT-L-14,laion2b_s32b_b82k"
  # "ViT-H-14,laion2b_s32b_b79k"
  
  # 3. Dòng SigLIP (Kiến trúc Sigmoid Loss, rất nhạy bén với Tiếng Anh/Văn bản)
  "ViT-B-16-SigLIP,webli"
  # "ViT-SO400M-14-SigLIP,webli"
  
  # 4. Dòng EVA-CLIP (Hiệu năng cực tốt trong các cuộc thi truy xuất)
  # "EVA02-B-16,merged2b_s8b_b131k"
  # "EVA02-L-14,merged2b_s4b_b131k"
)
PRECISION="fp16"
OCR_ENGINE="transformers"
OCR_MODEL="stepfun-ai/GOT-OCR2_0"
OD_ENGINE="yolo"
OD_MODEL="yolov8x.pt"
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
    uv run python baseline/extract_embed.py \
      --keyframes $KF_DIR --out $INDEX_DIR --device "cuda:${GPU_ID}" \
      --model "$MODEL" --pretrained "$PRETRAINED" --precision $PRECISION \
      --shard-index $i --shard-count $SHARDS --batch-size $BATCH_SIZE &
  done
  wait
  
  # Free memory before next model
  uv run python scripts/clean_ram.py
done
echo "-> Finished embedding!"


echo "=========================================================="
echo "STAGE 3: Extract Metadata (OCR & YOLO)"
echo "=========================================================="
# Run multiple metadata extraction processes in parallel
for i in $(seq 0 $((SHARDS-1))); do
  GPU_ID=$((i % NUM_GPUS))
  uv run  python baseline/extract_metadata.py \
    --keyframes "$KF_DIR" \
    --out "$INDEX_DIR" \
    --ocr-engine "$OCR_ENGINE" \
    --ocr-model "$OCR_MODEL" \
    --od-engine "$OD_ENGINE" \
    --od-model "$OD_MODEL" \
    --device "cuda:$((i % NUM_GPUS))" \
    --shard-index $i --shard-count $SHARDS &
done
wait
echo "-> Finished metadata extraction!"

# Dọn dẹp RAM/VRAM
uv run python scripts/clean_ram.py

echo "=========================================================="
echo "STAGE 4: Retrieval (Query and generate submission.json)"
echo "=========================================================="
uv run python baseline/retrieve.py \
  --shards $INDEX_DIR/shards \
  --metadata $INDEX_DIR/metadata \
  --tasks $TASKS_FILE \
  --out $OUT_FILE \
  --vlms "${VLMS[@]}" \
  --precision $PRECISION \
  --device "cuda:0"

# Cleanse RAM after retrieval
uv run python scripts/clean_ram.py

echo "=========================================================="
echo "DONE! Submission file saved at: $OUT_FILE"
echo "=========================================================="
