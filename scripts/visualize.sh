#!/bin/bash
set -e

# Move to project root directory
cd "$(dirname "$0")/.."

export FPS=1

if [ "$FPS" = "0" ]; then
  export METHOD="keyframe"
else
  export METHOD="fps/$FPS"
fi

export KF_DIR="keyframes/$METHOD"

echo "Running visualization script..."
uv run python pipeline/utils/visualize.py "$@"
