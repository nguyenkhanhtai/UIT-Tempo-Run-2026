#!/bin/bash

# Move to project root directory to ensure paths are correct
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Catch Ctrl+C and kill background jobs to prevent zombie processes
trap 'JOBS=$(jobs -p); if [ -n "$JOBS" ]; then kill $JOBS 2>/dev/null || true; fi; exit' SIGINT SIGTERM

# Directory paths
export DATASET_ROOT="dataset/Video_V3C"
export KF_DIR="keyframes"
export INDEX_DIR="artifacts/index"
export TASKS_FILE="dataset/Public_round_tasks.jsonl"

# Auto-detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi
export NUM_GPUS

# ======================================================================
# MODEL CONFIGURATIONS
# ======================================================================

# --- CÁC MÔ HÌNH VLM GỢI Ý ĐỂ ENSEMBLE (Mở comment để sử dụng) ---
export VLMS=(
  "PE-Core-bigG-14-448,meta"
  
  # 1. Dòng OpenCLIP Tiêu chuẩn (Nhẹ, cân bằng)
  # "ViT-B-32,laion2b_s34b_b79k"
  # "ViT-B-16,laion2b_s34b_b88k"
  
  # 2. Dòng OpenCLIP Lớn (Nặng hơn, cực kỳ chính xác cho ngữ nghĩa)
  # "ViT-L-14,laion2b_s32b_b82k"
  # "ViT-H-14,laion2b_s32b_b79k"
  
  # 3. Dòng SigLIP (Kiến trúc Sigmoid Loss, rất nhạy bén với Tiếng Anh/Văn bản)
  # "ViT-B-16-SigLIP,webli"
  # "ViT-SO400M-14-SigLIP,webli"
  
  # 4. Dòng EVA-CLIP (Hiệu năng cực tốt trong các cuộc thi truy xuất)
  # "EVA02-B-16,merged2b_s8b_b131k"
  # "EVA02-L-14,merged2b_s4b_b131k"
)
export PRECISION="fp16"

# --- CÁC MÔ HÌNH OCR GỢI Ý (Mở comment để sử dụng) ---
# 1. EasyOCR (Nhanh, nhẹ, hỗ trợ nhiều ngôn ngữ)
# export OCR_ENGINE="easyocr"
# export OCR_MODEL="easyocr"

# 2. RapidOCR (Nhanh như chớp, chạy bằng ONNX, rất nhẹ và không cần thư viện rườm rà)
export OCR_ENGINE="rapidocr"
export OCR_MODEL="rapidocr"

# 3. Florence-2 (Mô hình Vision-Language đa năng của Microsoft, siêu chính xác)
# export OCR_ENGINE="florence2"
# export OCR_MODEL="microsoft/Florence-2-large"
# HOẶC bản Base nhẹ hơn:
# export OCR_ENGINE="florence2_base"
# export OCR_MODEL="microsoft/Florence-2-base"

# --- CÁC MÔ HÌNH OBJECT DETECTION GỢI Ý (Mở comment để sử dụng) ---
export OD_ENGINE="yolo"
# Các phiên bản YOLO (yolov12n, yolov12s, yolov12m, yolo11m, yolov10m, v.v.) HOẶC florence2
export OD_MODEL="yolo11m.pt"

# --- CÁC MÔ HÌNH IMAGE CAPTIONING GỢI Ý (Mở comment để sử dụng) ---
# 1. Florence-2 (Tuỳ chọn số 1 cho Image Captioning)
export CAPTION_ENGINE="florence2"
export CAPTION_MODEL="microsoft/Florence-2-large"

# 2. BLIP (Rất nhẹ, nhanh, tốn ít VRAM, sinh câu tiếng Anh chuẩn xác)
# export CAPTION_ENGINE="blip"
# export CAPTION_MODEL="Salesforce/blip-image-captioning-base"

# ======================================================================
# TOGGLES & BATCH SETTINGS
# ======================================================================

# --- CÁC TÙY CHỌN DÙNG METADATA ĐỂ CHẤM ĐIỂM VÀ LỌC KẾT QUẢ ---
export USE_OCR="true"
export USE_OD="true"
export USE_CAPTIONING="true"
export USE_CLUSTERING="true"

# Temporal Smoothing Settings
export SMOOTHING_WINDOW=3
export SMOOTHING_SIGMA=1.0

# Batching and sharding parameters
export SHARDS=4
export CLIP_BATCH_SIZE=1024
export OCR_BATCH_SIZE=4
export OD_BATCH_SIZE=4
export LIMIT_PER_SHARD=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
