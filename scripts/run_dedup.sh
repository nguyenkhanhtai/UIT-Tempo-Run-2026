#!/bin/bash
set -e

echo "=== LỌC KHUNG HÌNH TRÙNG LẶP ==="
source scripts/config.sh

MODEL_SAFE=${VLMS[0]%%,*}
MODEL_SAFE=${MODEL_SAFE//\//_}
PRE_SAFE=${VLMS[0]##*,}
PRE_SAFE=${PRE_SAFE//\//_}
EMBED_DIR="artifacts/embeddings/shards/${MODEL_SAFE}_${PRE_SAFE}"

uv run python pipeline/dedup_keyframes.py \
    --keyframes $KF_DIR \
    --embed-dir $EMBED_DIR \
    --threshold 5.0

echo "Hoàn tất lọc trùng lặp!"
