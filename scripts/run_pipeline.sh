#!/bin/bash
set -e

# Trap Ctrl+C (SIGINT) signal and kill all child processes
trap 'echo "Cleaning up background processes..."; kill $(jobs -p) 2>/dev/null; exit 1' SIGINT

SCRIPT_DIR="$(dirname "$0")"

echo "Starting Pipeline..."

# ==========================================
# BƯỚC 1: TRÍCH XUẤT KEYFRAMES TỪ VIDEO
# ==========================================
# Các tham số:
# --video-root: Thư mục chứa toàn bộ video gốc
# --fps       : Tần số trích xuất (1 frame mỗi giây)
# --shards    : Số lượng tiến trình chạy song song
bash "$SCRIPT_DIR/extract_keyframes.sh" \
    --video-root dataset/Video_V3C \
    --fps 1 \
    --shards 6

# ==========================================
# BƯỚC 2: CHẠY MÔ HÌNH VLM ĐỂ TẠO VECTOR
# ==========================================
# Các tham số:
# --fps       : Phải khớp với mức fps ở Bước 1
# --batch-size: Số lượng ảnh xử lý cùng lúc (tuỳ vào VRAM GPU)
# --shards    : Số lượng tiến trình chạy song song (do load model nặng nên để ít)
bash "$SCRIPT_DIR/extract_embeddings.sh" \
    --fps 1 \
    --batch-size 256 \
    --shards 2

# ==========================================
# BƯỚC 3: TRÍCH XUẤT METADATA PHỤ TRỢ (NẾU CÓ)
# ==========================================
bash "$SCRIPT_DIR/extract_metadata.sh"

# ==========================================
# BƯỚC 4: TRUY XUẤT KẾT QUẢ VÀ TẠO SUBMISSION
# ==========================================
# Các tham số cơ bản:
# --task-file : Đường dẫn tới file JSON chứa danh sách câu truy vấn
# --num-tasks : Số lượng câu truy vấn cần chạy
# --max-preds : Số lượng video kết quả trả về tối đa cho mỗi truy vấn
# --num-chunks: Số lượng block (chunk) chia nhỏ tiến trình để tránh tràn RAM
bash "$SCRIPT_DIR/retrieval.sh" \
    --task-file dataset/private_round_tasks.jsonl \
    --num-tasks 700 \
    --fps 1 \
    --use-sequential "true" \
    --scene-segmenter qwen7b \
    --object-segmenter "none" \
    --agg-mode "prod" \
    --dp-mode "plus" \
    --position-mode "second" \
    --max-preds 10 \
    --clip-to-zero "true" \
    --main-query-weight "3.0" \
    --split-query "true" \
    --lm-cache "true" \
    --smoothing-window 0 \
    --smoothing-sigma 1.0 \
    --discount-factor "0.8" \
    --max-seq-gap 6000 \
    --overlap-threshold 0000 \
    --sliding-sim-threshold "0.95" \
    --num-chunks 5

echo "Pipeline fully completed!"
