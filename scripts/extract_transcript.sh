#!/bin/bash
set -e

trap 'JOBS=$(jobs -p); if [ -n "$JOBS" ]; then kill $JOBS 2>/dev/null || true; fi; exit' SIGINT SIGTERM

# Kích hoạt môi trường
source .venv/bin/activate

export SHARDS=4

export VIDEO_DIR="dataset/Video_V3C"
export OUT_DIR="artifacts/metadata"

# Tối ưu RAM/VRAM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================================="
echo "STAGE: Extract Audio Transcripts (Faster-Whisper)"
echo "=========================================================="
mkdir -p "$OUT_DIR"

echo ">>> Extracting Transcripts ($SHARDS Shards)..."
for i in $(seq 0 $((SHARDS-1))); do
    # Chạy mỗi shard trên một process độc lập
    # Mỗi shard chỉ cần 1 GPU worker (tổng 4 shards = 4 workers)
    python -u pipeline/extract_transcript.py \
        --video-dir "$VIDEO_DIR" \
        --out "$OUT_DIR/whisper_transcripts.json" \
        --chunk-size 200 \
        --gpu-workers 1 \
        --cpu-workers 4 \
        --shard-index $i --shard-count $SHARDS &
done

wait

echo "-> Finished audio transcript extraction!"
