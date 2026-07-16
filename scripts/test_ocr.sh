#!/bin/bash
# Lệnh bash để test mô hình OCR (GOT-OCR2.0 - SOTA OCR)
# Có thể đổi ảnh trong test_images/ocr_test.png thành ảnh khác nếu muốn
cd "$(dirname "$0")/.."
mkdir -p logs


uv run python pipeline/utils/test.py \
    --task ocr \
    --engine florence2 \
    --images "test_images/ocr_*.png" \
    --batch-size 2
rm -rf ~/.cache/huggingface/hub/*

# Option 7: Florence-2-base
uv run python pipeline/utils/test.py \
    --task ocr \
    --engine florence2_base \
    --images "test_images/ocr_*.png" \
    --batch-size 4
rm -rf ~/.cache/huggingface/hub/*


# Option 9: EasyOCR
uv run python pipeline/utils/test.py \
    --task ocr \
    --engine easyocr \
    --images "test_images/ocr_*.png"
rm -rf ~/.EasyOCR/

# Option 10: RapidOCR
uv run python pipeline/utils/test.py \
    --task ocr \
    --engine rapidocr \
    --images "test_images/ocr_*.png"
