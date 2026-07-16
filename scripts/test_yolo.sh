#!/bin/bash
# Lệnh bash để test mô hình YOLO (yolo11m - SOTA Object Detection)
# Có thể đổi ảnh trong test_images/yolo_test.png thành ảnh khác nếu muốn
cd "$(dirname "$0")/.."
mkdir -p logs
uv run python pipeline/utils/test.py \
    --task od \
    --engine yolo \
    --model yolo12m.pt \
    --images "test_images/yolo_*.png"
# Xóa model YOLO (file .pt) khỏi thư mục hiện tại
rm -f *.pt
