#!/bin/bash
# Lệnh bash để test mô hình Image Captioning (Florence-2)
# Có thể đổi ảnh trong test_images/yolo_test.png thành ảnh khác nếu muốn
cd "$(dirname "$0")/.."
mkdir -p logs
uv run python baseline/utils/test.py \
    --task caption \
    --engine florence2 \
    --model "microsoft/Florence-2-large" \
    --images "test_images/yolo_*.png"
# Xóa cache ổ cứng
rm -rf ~/.cache/huggingface/hub/*
