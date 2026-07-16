#!/bin/bash
set -e

echo "=== 1. Chèn thêm khung hình và Re-index lại dữ liệu ==="
source scripts/config.sh

# Catch Ctrl+C and kill background jobs
trap 'kill $(jobs -p); exit' SIGINT SIGTERM

# Limit numpy/OpenBLAS threads to prevent OS limit crash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Lấy đường dẫn thư mục chứa Embeddings
MODEL_SAFE=${VLMS[0]%%,*}
MODEL_SAFE=${MODEL_SAFE//\//_}
PRE_SAFE=${VLMS[0]##*,}
PRE_SAFE=${PRE_SAFE//\//_}
EMBED_DIR="artifacts/embeddings/shards/${MODEL_SAFE}_${PRE_SAFE}"

# Chạy script chèn khung hình bằng nhiều luồng (parallel)
for i in $(seq 0 $((SHARDS-1))); do
  uv run python pipeline/fill_keyframes.py \
    --dataset-root $DATASET_ROOT/V3C1 \
    --dataset-root $DATASET_ROOT/V3C2 \
    --keyframes $KF_DIR \
    --embed-dir $EMBED_DIR \
    --shard-index $i --shard-count $SHARDS &
done
wait
echo "Re-index hoàn tất!"

echo "=== 2. Nén toàn bộ thư mục artifacts ==="
zip -q -r artifacts.zip artifacts/
echo "Nén xong: artifacts.zip"

echo "=== 3. Upload lên Google Drive ==="
# Cấu hình rclone dựa trên rclone_token.json của anh
rclone config create mydrive drive scope drive token "$(<scripts/rclone_token.json)"

# Copy file zip lên Drive
rclone copy artifacts.zip mydrive:UIT-Tempo-Run-2026/

echo "=== TẤT CẢ ĐÃ HOÀN TẤT! ==="
