#!/bin/bash
# Lệnh bash để test mô hình YOLO (YOLOv12m - SOTA Object Detection)
# Có thể đổi ảnh trong test_images/yolo_test.png thành ảnh khác nếu muốn
cd "$(dirname "$0")/.."
mkdir -p logs
uv run python baseline/utils/inspect.py \
    --task od \
    --engine yolo \
    --model yolov12m.pt \
    --images "test_images/yolo_*.png" 2>&1 | tee logs/test_yolo.log
