#!/bin/bash
# Lệnh bash để test mô hình Image Captioning (Florence-2)
# Có thể đổi ảnh trong test_images/yolo_test.png thành ảnh khác nếu muốn
cd "$(dirname "$0")/.."
mkdir -p logs
uv run python pipeline/utils/test.py \
    --task caption \
    --engine florence2 \
    --model "microsoft/Florence-2-large" \
    --images "test_images/yolo_*.png" \
    --batch-size 2
# Xóa cache ổ cứng
rm -rf ~/.cache/huggingface/hub/*

uv run python pipeline/utils/test.py \
    --task caption \
    --engine blip \
    --model "Salesforce/blip-image-captioning-base" \
    --images "test_images/yolo_*.png" \
    --batch-size 4
rm -rf ~/.cache/huggingface/hub/*
