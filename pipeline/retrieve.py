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
from retrieval.scorer import compute_similarity, aggregate_scores
from retrieval.postprocess import apply_clustering
from models.factory import get_embedding_model



def run_retrieval(args):
    tasks = [json.loads(l) for l in open(args.tasks)]
    
    # Set globals for toggles based on args
    import retrieval.scorer as scorer
    import retrieval.postprocess as postprocess

    from pipeline.retrieval.parser import parse_queries
    
    print(f"[tasks] {len(tasks)}", flush=True)
    all_queries, task_mapping = parse_queries(tasks, args.use_sequential, args.scene_segmenter, args.object_segmenter)
    print(f"[queries] extracted {len(all_queries)} object queries from {len(tasks)} tasks", flush=True)
        
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
        emb_smoothed = smooth_features(emb, vids, smoothing_window=args.smoothing_window, sigma=args.smoothing_sigma)
        all_models_idx.append(emb_smoothed)
        
        clip = get_embedding_model(model_name, pretrained, device=args.device, precision=args.precision)
        Q = clip.encode_texts(all_queries)      # [T_all, D] fp32
        all_models_Q.append(torch.from_numpy(Q).to(dev).float())
        

        
        # Free up RAM/VRAM
        del clip
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # 5. Compute GPU Similarity (Ensembled Dual Softmax + Bonus)
    T_all, N = all_models_Q[0].shape[0], all_models_idx[0].shape[0]
    K = min(args.cand_keyframes, N)
    if args.use_sequential:
        K = N
    
    top_idx, top_val = compute_similarity(all_models_Q, all_models_idx, T_all, N, K, dev)

    # 6. Aggregate Scores (Base + Subs + Causal Bonus)
    all_candidates = aggregate_scores(
        task_mapping, top_idx, top_val, tasks, first_vids, first_ts, first_emb, first_metadata, args.use_sequential
    )

    # 7. Postprocess (Clustering & Formatting)
    preds = []
    for task, candidates in all_candidates:
        res = apply_clustering(task, candidates, args.top_videos)
        preds.append(res)

    sub = {"predictions": preds}
    
    import os, glob, shutil
    out_path = args.out
    
    if out_path == "submission.json":
        os.makedirs("submission", exist_ok=True)
        sub_dirs = glob.glob("submission/*")
        valid_dirs = [d for d in sub_dirs if os.path.basename(d).isdigit()]
        if not valid_dirs:
            next_id = 1
        else:
            valid_dirs.sort(key=lambda x: int(os.path.basename(x)))
            next_id = int(os.path.basename(valid_dirs[-1])) + 1
            
        out_dir = os.path.join("submission", f"{next_id:03d}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "submission.json")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    json.dump(sub, open(out_path, "w"))
    
    # Save parameters
    config_data = {
        "args": vars(args),
        "env_params": {
            "MAIN_QUERY_WEIGHT": os.environ.get("MAIN_QUERY_WEIGHT"),
            "SPLIT_QUERY": os.environ.get("SPLIT_QUERY"),
            "MAX_SEQ_GAP_MS": os.environ.get("MAX_SEQ_GAP_MS"),
            "DISCOUNT_FACTOR": os.environ.get("DISCOUNT_FACTOR"),
            "AGG_MODE": os.environ.get("AGG_MODE")
        }
    }
    
    if "out_dir" in locals():
        config_path = os.path.join(out_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
            
    if out_path != args.out:
        shutil.copy(out_path, args.out)
        
    if "out_dir" in locals():
        import zipfile
        zip_path = os.path.join(out_dir, "submission.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(out_path, arcname="submission.json")
            zf.write(config_path, arcname="config.json")
        print(f"[done] wrote {out_path}, {config_path}, {zip_path} and copied to {args.out} ({len(preds)} tasks)", flush=True)
    else:
        print(f"[done] wrote {out_path} ({len(preds)} tasks)", flush=True)

    # Return data for decorators/analysis if needed
    return tasks, task_mapping, first_vids, first_ts, first_metadata, all_queries, all_candidates

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

    p.add_argument("--use-sequential", action="store_true", help="Enable Sequential DP Matching with SaT")
    p.add_argument("--scene-segmenter", type=str, default="regex", help="Segmenter engine for scenes (regex, bert_srl)")
    p.add_argument("--object-segmenter", type=str, default="regex", help="Segmenter engine for objects (regex, spacy, sat, scenegraph, none)")
    p.add_argument("--smoothing-window", type=int, default=3, help="Window size for Gaussian temporal smoothing")
    p.add_argument("--smoothing-sigma", type=float, default=1.0, help="Sigma for Gaussian temporal smoothing")
    
    args = p.parse_args()
    run_retrieval(args)

if __name__ == "__main__":
    main()
