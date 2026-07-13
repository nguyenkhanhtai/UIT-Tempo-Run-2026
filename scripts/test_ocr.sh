#!/bin/bash
# Lệnh bash để test mô hình OCR (GOT-OCR2.0 - SOTA OCR)
# Có thể đổi ảnh trong test_images/ocr_test.png thành ảnh khác nếu muốn
cd "$(dirname "$0")/.."
mkdir -p logs
uv run python baseline/utils/inspect.py \
    --task ocr \
    --engine transformers \
    --model stepfun-ai/GOT-OCR2_0 \
    --images "test_images/ocr_*.png" 2>&1 | tee logs/test_ocr.log
