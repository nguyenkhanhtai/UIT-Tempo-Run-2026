#!/bin/bash
set -e

# Bẫy tín hiệu Ctrl+C (SIGINT) và diệt tất cả các tiến trình con
trap 'echo "Đang dọn dẹp các tiến trình nền..."; kill $(jobs -p) 2>/dev/null; exit 1' SIGINT

# Chuyển về thư mục gốc của project
cd "$(dirname "$0")/.."

# Đường dẫn thư mục
DATASET_ROOT="dataset/Video_V3C"
KF_DIR="keyframes"
INDEX_DIR="artifacts/index"
TASKS_FILE="dataset/Public_round_tasks.jsonl"
OUT_FILE="submission.json"
DEV="cuda:0"

# Cấu hình Model nhẹ (ViT-B-32) để tiết kiệm bộ nhớ
MODEL="ViT-B-32"
PRETRAINED="laion2b_s34b_b79k"
PRECISION="fp32"
SHARDS=2

# echo "=========================================================="
# echo "STAGE 1: Trích xuất Keyframes (Cắt ảnh từ Video)"
# echo "=========================================================="
# # Chạy song song 6 tiến trình (shards) để tăng tốc độ cắt ảnh
# for i in $(seq 0 $((SHARDS-1))); do
#   uv run python baseline/extract_keyframes.py \
#     --dataset-root $DATASET_ROOT/V3C1 \
#     --dataset-root $DATASET_ROOT/V3C2 \
#     --out $KF_DIR --shard-index $i --shard-count $SHARDS &
# done
# wait
# echo "-> Hoàn thành cắt ảnh!"

echo "=========================================================="
echo "STAGE 2: Trích xuất Embeddings (Nhúng Vector bằng $MODEL)"
echo "=========================================================="
# Chạy song song 6 tiến trình nhúng vector
for i in $(seq 0 $((SHARDS-1))); do
  uv run python baseline/extract_embed.py \
    --keyframes $KF_DIR --out $INDEX_DIR --device $DEV \
    --model $MODEL --pretrained $PRETRAINED --precision $PRECISION \
    --shard-index $i --shard-count $SHARDS &
done
wait
echo "-> Hoàn thành nhúng vector!"

echo "=========================================================="
echo "STAGE 3: Retrieval (Truy xuất và tạo file submission.json)"
echo "=========================================================="
uv run python baseline/retrieve.py \
  --shards $INDEX_DIR/shards \
  --tasks $TASKS_FILE \
  --out $OUT_FILE \
  --model $MODEL --pretrained $PRETRAINED --precision $PRECISION \
  --device $DEV

echo "=========================================================="
echo "HOÀN TẤT! File kết quả nộp bài đã được lưu tại: $OUT_FILE"
echo "=========================================================="
