#!/bin/bash
# Lệnh bash để test mô hình OCR (GOT-OCR2.0 - SOTA OCR)
# Có thể đổi ảnh trong test_images/ocr_test.png thành ảnh khác nếu muốn
cd "$(dirname "$0")/.."
mkdir -p logs
# --- Option 1: GOT-OCR2.0 (Transformers) ---
# uv run python baseline/utils/test.py \
#     --task ocr \
#     --engine transformers \
#     --model stepfun-ai/GOT-OCR2_0 \
#     --images "test_images/ocr_*.png" 2>&1 | tee logs/test_ocr.log

# --- Option 2: Microsoft Florence-2 (SOTA VLM OCR, less hallucination) ---
# uv run python baseline/utils/test.py \
#     --task ocr \
#     --engine florence2 \
#     --model microsoft/Florence-2-large \
#     --images "test_images/ocr_*.png" 2>&1 | tee logs/test_ocr.log

# --- Option 3: EasyOCR (Fast, lightweight, good for simple text) ---
# uv run python baseline/utils/test.py \
#     --task ocr \
#     --engine easyocr \
#     --images "test_images/ocr_*.png" 2>&1 | tee logs/test_ocr.log

# --- Option 4: PaddleOCR (SOTA for traditional OCR and Vietnamese) ---
uv run python baseline/utils/test.py \
    --task ocr \
    --engine paddleocr \
    --images "test_images/ocr_*.png" 2>&1 | tee logs/test_ocr.log
