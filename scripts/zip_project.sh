#!/bin/bash
# Script nén code và artifacts thành 2 file riêng biệt

CODE_ZIP="code.zip"
ARTIFACT_ZIP="artifact.zip"

# Chuyển về thư mục gốc của project
cd "$(dirname "$0")/.."

echo "Bắt đầu nén mã nguồn vào file $CODE_ZIP..."
zip -r "$CODE_ZIP" . \
  -x "dataset/*" \
  -x "keyframes/*" \
  -x "artifacts/*" \
  -x ".git/*" \
  -x ".venv/*" \
  -x "*/__pycache__/*" \
  -x "*.zip"

echo "Bắt đầu nén kết quả retrieval vào file $ARTIFACT_ZIP..."
zip -r "$ARTIFACT_ZIP" artifacts/ \
  -x "*.zip"

echo "Nén thành công! Các file được lưu tại:"
echo "- $(pwd)/$CODE_ZIP"
echo "- $(pwd)/$ARTIFACT_ZIP"
