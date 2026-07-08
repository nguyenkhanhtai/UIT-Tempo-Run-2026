#!/bin/bash
# Script nén toàn bộ project và dữ liệu retrieval (artifacts), loại trừ các file nặng như dataset, keyframes

OUTPUT_ZIP="project_backup.zip"

# Chuyển về thư mục gốc của project
cd "$(dirname "$0")/.."

echo "Bắt đầu nén project vào file $OUTPUT_ZIP..."
echo "Các thư mục bị bỏ qua: dataset, keyframes, .git, .venv, __pycache__"

zip -r "$OUTPUT_ZIP" . \
  -x "dataset/*" \
  -x "keyframes/*" \
  -x ".git/*" \
  -x ".venv/*" \
  -x "*/__pycache__/*" \
  -x "$OUTPUT_ZIP"

echo "Nén thành công! File được lưu tại: $(pwd)/$OUTPUT_ZIP"
