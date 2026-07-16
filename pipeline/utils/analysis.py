import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path

import pipeline.retrieve as retrieve
import pipeline.retrieval.scorer as scorer

# Global dictionary to observe values during retrieve.py execution
observed = {}

# 1. Decorate precompute_metadata_bonus to observe the total metadata bonus matrix (B)
orig_precompute = scorer.precompute_metadata_bonus
def observed_precompute(tasks, task_mapping, vids, ts, metadata):
    print("[Observer] Intercepting precompute_metadata_bonus to capture B...")
    B = orig_precompute(tasks, task_mapping, vids, ts, metadata)
    observed['B'] = B.copy()
    return B

# 2. Decorate compute_similarity to observe the final scores for ALL frames
orig_compute = scorer.compute_similarity
def observed_compute(Q_list, idx_list, B, T_all, N, K, dev):
    print(f"[Observer] Intercepting compute_similarity. Running with K={N} to capture all frames...")
    # Call the original function with K=N to get the full final score array (CLIP + B)
    top_idx, top_val = orig_compute(Q_list, idx_list, B, T_all, N, N, dev)
    
    # Store the full results
    observed['top_idx'] = top_idx
    observed['top_val'] = top_val
    
    # Return only the top K as originally requested to avoid slowing down downstream tasks
    return top_idx[:, :K], top_val[:, :K]

# Apply decorators to the namespace where they are actually used
retrieve.precompute_metadata_bonus = observed_precompute
retrieve.compute_similarity = observed_compute

def get_image_path_for_frame(vid, frame_ms, kf_dir):
    vdir = Path(kf_dir) / vid
    ts_path = vdir / "ts_ms.npy"
    if not ts_path.exists():
        return None
    try:
        ts = np.load(ts_path)
    except Exception:
        return None
    idx = np.argmin(np.abs(ts - frame_ms))
    k_filename = f"k_{idx + 1:05d}.jpg"
    img_path = vdir / k_filename
    if img_path.exists():
        return str(img_path)
    return None

import textwrap

def create_max_score_figure(task_id, query_text, category_name, max_score, meta_text, img_path, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    
    query_text_wrapped = textwrap.fill(f"Query: {query_text}", width=80)
    fig.text(0.5, 0.95, f"Task: {task_id}\n{query_text_wrapped}", fontsize=14, ha='center', va='top')
    
    if img_path and os.path.exists(img_path):
        try:
            img = Image.open(img_path)
            ax = fig.add_axes([0.1, 0.25, 0.8, 0.6])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Max {category_name} Score: {max_score:.4f}", fontsize=12)
        except Exception:
            fig.text(0.5, 0.5, "Image not found", fontsize=16, ha='center')
    else:
        fig.text(0.5, 0.5, f"Image path invalid: {img_path}", fontsize=16, ha='center')
        
    meta_text_wrapped = "\n".join([textwrap.fill(line, width=100) for line in meta_text.split("\n")])
    fig.text(0.5, 0.20, f"Metadata:\n{meta_text_wrapped}", fontsize=10, ha='center', va='top', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
             
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--shards", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--keyframes", required=True)
    p.add_argument("--vlms", nargs="+", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cand-keyframes", type=int, default=2000)
    p.add_argument("--top-videos", type=int, default=10)
    p.add_argument("--use-ocr", action="store_true")
    p.add_argument("--use-od", action="store_true")
    p.add_argument("--use-captioning", action="store_true")
    p.add_argument("--use-clustering", action="store_true")
    args = p.parse_args()

    # Cap tasks
    with open(args.tasks, 'r') as f:
        tasks_json = [json.loads(line) for line in f]
    temp_tasks_path = args.tasks + f".top{args.n}.tmp"
    with open(temp_tasks_path, 'w') as f:
        for t in tasks_json[:args.n]:
            f.write(json.dumps(t) + '\n')
            
    args.tasks = temp_tasks_path
    args.out = "submission_analysis_tmp.json"
    args.precision = None
    
    print(f"Running retrieval pipeline for top {args.n} tasks...")
    # This runs the unmodified pipeline, but our decorators will secretly capture the variables!
    ret_data = retrieve.run_retrieval(args)
    if ret_data is None:
        print("Error: retrieve.run_retrieval did not return expected data. Make sure it was refactored correctly.")
        return
    tasks_parsed, task_mapping, first_vids, first_ts, first_metadata, all_queries = ret_data

    # Cleanup temp files
    if os.path.exists(temp_tasks_path): os.remove(temp_tasks_path)
    if os.path.exists(args.out): os.remove(args.out)

    print("Pipeline finished. Generating visualizations from observed data...")
    
    top_idx = observed['top_idx']  # [T_all, N] (sorted frame indices)
    top_val = observed['top_val']  # [T_all, N] (sorted final scores)
    B_total = observed['B']        # [T_all, N] (unsorted metadata bonuses)
    
    T_all, N = top_idx.shape
    
    # We unsort top_val to get the final score for frames in their original order
    final_score_unsorted = np.zeros((T_all, N), dtype=np.float32)
    for i in range(T_all):
        final_score_unsorted[i, top_idx[i]] = top_val[i]
        
    # The pure CLIP score is Final Score - Metadata Bonus
    clip_score = final_score_unsorted - B_total
    
    os.makedirs("figures/analysis", exist_ok=True)
    
    for qi, (ti, is_main, sub_id) in enumerate(task_mapping):
        if not is_main: continue 
        
        task = tasks_parsed[ti]
        task_id = task.get("task_id", f"T_unknown_{ti}")
        query_text = all_queries[qi]
        
        task_dir = os.path.join("figures/analysis", task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes = axes.flatten()
        fig.suptitle(f"Score Distributions for Task {task_id}", fontsize=16)
        
        score_data = {
            "Total Score (CLIP + Meta)": final_score_unsorted[qi],
            "CLIP Visual Score": clip_score[qi],
            "Total Metadata Bonus": B_total[qi]
        }
        
        for idx, (name, data) in enumerate(score_data.items()):
            ax = axes[idx]
            sns.histplot(data, bins=50, kde=True, ax=ax)
            ax.set_title(f"{name}\nMean: {np.mean(data):.4f}, Var: {np.var(data):.4f}")
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            
        plt.tight_layout()
        plt.savefig(os.path.join(task_dir, "distribution.png"))
        plt.close(fig)
        
        # Max figures
        for cat_name, data in score_data.items():
            max_idx = np.argmax(data)
            max_val = data[max_idx]
            
            vid = str(first_vids[max_idx])
            ts_ms = int(first_ts[max_idx])
            
            meta_str = f"Video ID: {vid}, Frame MS: {ts_ms}\n"
            if vid in first_metadata and ts_ms in first_metadata[vid]:
                meta = first_metadata[vid][ts_ms]
                meta_str += f"OCR: {meta.get('ocr', '')}\n"
                meta_str += f"OD: {', '.join(meta.get('objects', []))}\n"
                meta_str += f"Caption: {meta.get('caption', '')}"
            else:
                meta_str += "No metadata found."
                
            img_path = get_image_path_for_frame(vid, ts_ms, args.keyframes)
            out_img = os.path.join(task_dir, cat_name.replace(" ", "_"), "max_score.png")
            create_max_score_figure(task_id, query_text, cat_name, max_val, meta_str, img_path, out_img)

    print("Analysis complete. Check figures/analysis/ directory.")

if __name__ == "__main__":
    main()
