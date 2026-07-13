#!/bin/bash
set -e

# Move to project root directory
cd "$(dirname "$0")/.."

echo "Running visualization script..."
uv run python baseline/utils/visualize.py "$@"
