#!/bin/bash
set -e

# Trap Ctrl+C (SIGINT) signal and kill all child processes
trap 'echo "Cleaning up background processes..."; kill $(jobs -p) 2>/dev/null; exit 1' SIGINT

SCRIPT_DIR="$(dirname "$0")"

echo "Starting Pipeline..."

bash "$SCRIPT_DIR/extract_keyframes.sh"
bash "$SCRIPT_DIR/extract_embeddings.sh"
bash "$SCRIPT_DIR/extract_metadata.sh"
bash "$SCRIPT_DIR/retrieval.sh"

echo "Pipeline fully completed!"
