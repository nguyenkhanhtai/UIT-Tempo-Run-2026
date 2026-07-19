#!/bin/bash
set -e

trap 'JOBS=$(jobs -p); if [ -n "$JOBS" ]; then kill $JOBS 2>/dev/null || true; fi; exit' SIGINT SIGTERM

export FPS=1
if [ "$FPS" = "0" ]; then
  export METHOD="keyframe"
else
  export METHOD="fps/$FPS"
fi

export KF_DIR="keyframes/$METHOD"
export OUT_DIR="artifacts/index/$METHOD/audio"
export METADATA_DIR="artifacts/metadata"

echo "=========================================================="
echo "BẮT ĐẦU TRÍCH XUẤT AUDIO FEATURES CHO $METHOD"
echo "=========================================================="

PYTHONPATH=. uv run python pipeline/extract_audio_features.py \
  --kf-dir $KF_DIR \
  --metadata-dir $METADATA_DIR \
  --out-dir $OUT_DIR \
  --model "sentence-transformers/all-MiniLM-L6-v2" \
  --device "cuda:0" \
  --batch-size 1024

echo "=========================================================="
echo "HOÀN TẤT AUDIO FEATURES!"
echo "Kết quả lưu tại: $OUT_DIR/audio_features.npy"
echo "=========================================================="
