# Video Retrieval Pipeline - Reproducibility Guide

This repository contains the complete end-to-end pipeline for our Video Retrieval system. Follow the instructions below to configure the environment, prepare the dataset, and reproduce our final submission from scratch.

## 1. Prerequisites & Environment Setup

This project uses `uv` for lightning-fast Python dependency management.

1. **Install `uv`** (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Install all dependencies**:
   Run the following command in the root directory to create an isolated virtual environment and install all packages locked in `uv.lock`.
   ```bash
   uv sync
   ```

## 2. Dataset Preparation

Before running the pipeline, ensure the dataset is placed in the correct directories:

1. **Video Dataset**: Place the raw video folders (e.g., V3C) inside `dataset/Video_V3C/`.
2. **Tasks File**: Place the evaluation queries/tasks JSONL file at `dataset/private_round_tasks.jsonl`.

## 3. Running the Full Pipeline

To reproduce the results, you only need to run a single master script. This will automatically execute keyframe extraction, embedding generation, metadata processing, and the final retrieval logic.

```bash
# Run the end-to-end pipeline
./scripts/run_pipeline.sh
```

### Pipeline Breakdown
Under the hood, `run_pipeline.sh` executes the following components sequentially:
1. **`extract_keyframes.sh`**: Samples frames from the raw videos (using FFmpeg).
2. **`extract_embeddings.sh`**: Computes visual feature embeddings using the configured models (e.g., X-CLIP, OpenCLIP).
3. **`extract_metadata.sh`**: Extracts optional multi-modal metadata (OCR/Object Detection) if enabled in the config.
4. **`retrieval.sh`**:
   - Parses the tasks file and segments complex queries using a local LLM (Qwen 2.5).
   - Computes similarity scores between text queries and video frames.
   - Executes in **fault-tolerant chunks**: Temporary chunks are saved to the root directory during execution. If the process is interrupted, re-running the script will automatically resume from the completed chunks.
   - Safely merges all chunks upon completion.

## 4. Expected Outputs

Once the pipeline finishes successfully, it will automatically clean up all temporary files and generate the final submission files inside an incremented directory (e.g., `submission/001/`).

You will find the following files:
- `submission/00x/submission.json` (The exact format required by the organizers)
- `submission/00x/submission.zip` (The zipped submission)
- `submission/00x/detailed_submission.json` (Contains detailed metadata and debugging info for each query)

## 5. Visualizing Results (Optional)

To inspect the retrieved results visually, you can start the built-in web viewer:

```bash
uv run python visualizer/app.py
```
Then, open the provided local URL (usually `http://localhost:5000`) in your web browser.
