#!/bin/bash
# Lệnh bash để test mô hình OCR (GOT-OCR2.0 - SOTA OCR)
# Có thể đổi ảnh trong test_images/ocr_test.png thành ảnh khác nếu muốn
cd "$(dirname "$0")/.."
mkdir -p logs

# Option 6: Vintern-1B-v3.5 (Multilingual VLM, strong Vietnamese support)
uv run python baseline/utils/test.py --task ocr --engine vintern --images "test_images/ocr_*.png"
# Xoá model khỏi ổ cứng ngay sau khi chạy xong để tiết kiệm dung lượng
rm -rf ~/.cache/huggingface/hub/*
uv run python baseline/utils/test.py --task ocr --engine florence2 --images "test_images/ocr_*.png" --batch-size 2
rm -rf ~/.cache/huggingface/hub/*
# Option 7: Florence-2-base
uv run python baseline/utils/test.py --task ocr --engine florence2_base --images "test_images/ocr_*.png" --batch-size 4
rm -rf ~/.cache/huggingface/hub/*

# # Option 8: PaddleOCR
uv run python baseline/utils/test.py --task ocr --engine paddleocr --images "test_images/ocr_*.png"
rm -rf ~/.paddleocr/

# Option 9: EasyOCR
uv run python baseline/utils/test.py --task ocr --engine easyocr --images "test_images/ocr_*.png"
rm -rf ~/.EasyOCR/
