#!/bin/bash
# Lệnh bash để test mô hình OCR (GOT-OCR2.0 - SOTA OCR)
# Có thể đổi ảnh trong test_images/ocr_test.png thành ảnh khác nếu muốn
cd "$(dirname "$0")/.."
mkdir -p logs

# Option 6: Vintern-1B-v3.5 (Multilingual VLM, strong Vietnamese support)
# uv run python baseline/utils/test.py --task ocr --engine vintern --images "test_images/ocr_*.png" 2>&1 | tee logs/test_ocr.log
# uv run python baseline/utils/test.py --task ocr --engine florence2 --images "test_images/ocr_*.png" 2>&1 | tee logs/test_ocr.log# Option 7: Florence-2-base
# uv run python baseline/utils/test.py --task ocr --engine florence2_base --images "test_images/ocr_*.png" 2>&1 | tee logs/test_ocr_florence2_base.log

# # Option 8: PaddleOCR
# uv run python baseline/utils/test.py --task ocr --engine paddleocr --images "test_images/ocr_*.png" 2>&1 | tee logs/test_ocr_paddleocr.log

# Option 9: EasyOCR
uv run python baseline/utils/test.py --task ocr --engine easyocr --images "test_images/ocr_*.png" > logs/test_ocr_easyocr.log 2>&1
