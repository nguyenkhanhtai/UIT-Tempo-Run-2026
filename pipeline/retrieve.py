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

Refactored to use modular architecture under `pipeline/retrieval/`.
"""
from __future__ import annotations
import argparse, json
import torch

from retrieval.loader import load_index
from retrieval.temporal import smooth_features
from retrieval.parser import parse_queries
from retrieval.scorer import compute_similarity, aggregate_scores, precompute_metadata_bonus
from retrieval.postprocess import apply_clustering
from models.embedding.clip_model import ClipModel

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", required=True)
    p.add_argument("--metadata", default=None, help="dir containing .jsonl metadata")
    p.add_argument("--tasks", required=True, help="a round's task file, e.g. public_round_tasks.jsonl")
    p.add_argument("--out", required=True, help="submission.json path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--vlms", nargs="+", required=True, help="List of model,pretrained pairs e.g. ViT-B-32,laion2b_s34b_b79k")
    p.add_argument("--precision", default=None)
    p.add_argument("--top-videos", type=int, default=10)
    p.add_argument("--cand-keyframes", type=int, default=2000)
    
    # Toggles for metadata scoring and clustering
    p.add_argument("--use-ocr", action="store_true", help="Enable OCR scoring")
    p.add_argument("--use-od", action="store_true", help="Enable OD scoring")
    p.add_argument("--use-captioning", action="store_true", help="Enable Caption scoring")
    p.add_argument("--use-clustering", action="store_true", help="Enable KMeans clustering")
    
    args = p.parse_args()

    tasks = [json.loads(l) for l in open(args.tasks)]
    print(f"[tasks] {len(tasks)}", flush=True)

    # Set globals for toggles based on args
    import retrieval.scorer as scorer
    import retrieval.postprocess as postprocess
    scorer.USE_OCR = args.use_ocr
    scorer.USE_OD = args.use_od
    scorer.USE_CAPTIONING = args.use_captioning
    postprocess.USE_CLUSTERING = args.use_clustering

    # 3. Parse Queries & Encode
    all_queries, task_mapping = parse_queries(tasks)
    
    all_models_Q = []
    all_models_idx = []
    
    first_vids, first_ts, first_metadata, first_emb = None, None, None, None
    dev = args.device
    
    for vlm_str in args.vlms:
        parts = vlm_str.split(",")
        model_name = parts[0]
        pretrained = parts[1] if len(parts) > 1 and parts[1] else None
        
        model_safe = model_name.replace("/", "_")
        pre_safe = pretrained.replace("/", "_") if pretrained else "none"
        
        from pathlib import Path
        shard_dir = Path(args.shards) / f"{model_safe}_{pre_safe}"
        
        # 1. Load Data
        emb, vids, ts, metadata = load_index(str(shard_dir), args.metadata if first_metadata is None else None)
        
        if first_vids is None:
            first_vids = vids
            first_ts = ts
            first_metadata = metadata
            first_emb = emb
        else:
            import numpy as np
            if not np.array_equal(first_vids, vids) or not np.array_equal(first_ts, ts):
                raise ValueError(f"Shards for {vlm_str} do not match the keyframe alignment of the first model!")
                
        # 2. Temporal Smoothing
        emb_smoothed = smooth_features(emb, vids)
        all_models_idx.append(emb_smoothed)
        
        clip = ClipModel(model_name, pretrained, device=args.device, precision=args.precision)
        Q = clip.encode_texts(all_queries)      # [T_all, D] fp32
        all_models_Q.append(torch.from_numpy(Q).to(dev).float())
        
        # Free up RAM/VRAM
        del clip
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Precompute Metadata Bonus (OCR & OD) for all N frames
    B = precompute_metadata_bonus(tasks, task_mapping, first_vids, first_ts, first_metadata)
    
    # 5. Compute GPU Similarity (Ensembled Dual Softmax + Bonus)
    T_all, N = all_models_Q[0].shape[0], all_models_idx[0].shape[0]
    K = min(args.cand_keyframes, N)
    
    top_idx, top_val = compute_similarity(all_models_Q, all_models_idx, B, T_all, N, K, dev)

    # 6. Aggregate Scores (Base + Subs + Causal Bonus)
    all_candidates = aggregate_scores(
        task_mapping, top_idx, top_val, tasks, first_vids, first_ts, first_emb, first_metadata
    )

    # 7. Postprocess (Clustering & Formatting)
    preds = []
    for task, candidates in all_candidates:
        res = apply_clustering(task, candidates, args.top_videos)
        preds.append(res)

    sub = {"predictions": preds}
    json.dump(sub, open(args.out, "w"))
    print(f"[done] wrote {args.out} ({len(preds)} tasks)", flush=True)

if __name__ == "__main__":
    main()
