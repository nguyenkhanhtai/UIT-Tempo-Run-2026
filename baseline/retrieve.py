"""Stage 3 — text->keyframe retrieval -> submission.json.

Loads all per-video shards into one keyframe index, embeds each task description
with CLIP text encoder, finds the most similar keyframes (chunked top-k on GPU),
dedups to distinct videos, and emits up to `max_predictions` (video_id, frame_ms) predictions
per task.

Added features (Merged):
- User's Temporal Smoothing, Dual Softmax, Causal Bonus.
- OOP model loading.
- OCR Hard Filtering.
- KMeans Clustering for Diversity.

Refactored to use modular architecture under `baseline/retrieval/`.
"""
from __future__ import annotations
import argparse, json
import torch

from retrieval.loader import load_index
from retrieval.temporal import smooth_features
from retrieval.parser import parse_queries
from retrieval.scorer import compute_similarity, aggregate_scores
from retrieval.postprocess import apply_clustering
from models.embedding.clip_model import ClipModel

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", required=True)
    p.add_argument("--metadata", default=None, help="dir containing .jsonl metadata")
    p.add_argument("--tasks", required=True, help="a round's task file, e.g. public_round_tasks.jsonl")
    p.add_argument("--out", required=True, help="submission.json path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--precision", default=None)
    p.add_argument("--top-videos", type=int, default=10)
    p.add_argument("--cand-keyframes", type=int, default=2000)
    args = p.parse_args()

    # 1. Load Data
    emb, vids, ts, metadata = load_index(args.shards, args.metadata)
    
    # 2. Temporal Smoothing
    emb_smoothed = smooth_features(emb, vids)
    
    tasks = [json.loads(l) for l in open(args.tasks)]
    print(f"[tasks] {len(tasks)}", flush=True)

    # 3. Parse Queries & Encode
    all_queries, task_mapping = parse_queries(tasks)
    
    clip = ClipModel(args.model, args.pretrained, device=args.device, precision=args.precision)
    Q = clip.encode_texts(all_queries)      # [T_all, D] fp32
    
    # Free up RAM/VRAM before loading massive indexes
    del clip
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Compute GPU Similarity (Dual Softmax)
    dev = args.device
    idx = torch.from_numpy(emb_smoothed).to(dev).float()
    Qt = torch.from_numpy(Q).to(dev).float()
    T_all, N = Qt.shape[0], idx.shape[0]
    K = min(args.cand_keyframes, N)
    
    top_idx, top_val = compute_similarity(Qt, idx, T_all, N, K, dev)

    # 5. Aggregate Scores (Base + Subs + Causal Bonus + Hard Filter)
    all_candidates = aggregate_scores(
        task_mapping, top_idx, top_val, tasks, vids, ts, emb, metadata
    )

    # 6. Postprocess (Clustering & Formatting)
    preds = []
    for task, candidates in all_candidates:
        res = apply_clustering(task, candidates, args.top_videos)
        preds.append(res)

    sub = {"predictions": preds}
    json.dump(sub, open(args.out, "w"))
    print(f"[done] wrote {args.out} ({len(preds)} tasks)", flush=True)

if __name__ == "__main__":
    main()
