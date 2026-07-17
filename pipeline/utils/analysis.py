import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path

import retrieve
import retrieval.scorer as scorer

# Global dictionary to observe values during retrieve.py execution
observed = {}

orig_precompute = scorer.precompute_metadata_bonus
def observed_precompute(tasks, task_mapping, vids, ts, metadata):
    print("[Observer] Intercepting precompute_metadata_bonus to isolate components in a SINGLE PASS...")
    
    # Run once with all components enabled
    B_total, B_ocr, B_od, B_cap = orig_precompute(tasks, task_mapping, vids, ts, metadata, return_components=True)
    
    # Store the individual components
    observed['ocr_bonus'] = B_ocr
    observed['od_bonus'] = B_od
    observed['caption_bonus'] = B_cap
    observed['B_total'] = B_total
    
    return B_total

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

def create_top_score_figure(task_id, query_text, category_name, max_score, meta_text, img_path, out_path, rank):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Squeeze the image vertically by reducing height from 10 to 8
    fig, ax = plt.subplots(figsize=(12, 8))
    
    wrapped_query = textwrap.fill(f"Task: {task_id} (Rank #{rank})\nInstruction/Query: {query_text}", width=90)
    
    if img_path and os.path.exists(img_path):
        try:
            img = Image.open(img_path)
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Image not found", fontsize=16, ha='center')
    else:
        ax.text(0.5, 0.5, f"Image path invalid: {img_path}", fontsize=16, ha='center')
        
    # Remove ticks and border
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Use title and xlabel for guaranteed non-overlapping layout
    ax.set_title(wrapped_query + f"\n\n[{category_name}] Score: {max_score:.4f}", fontsize=14, pad=15)
    
    # Wrap each line individually to preserve intended newlines
    lines = meta_text.strip().split('\n')
    wrapped_lines = [textwrap.fill(line, width=130) for line in lines]
    wrapped_meta = "Metadata/Content:\n" + "\n".join(wrapped_lines)
    
    ax.set_xlabel(wrapped_meta, fontsize=11, labelpad=15, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
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
    p.add_argument("--use-captioning", action="store_true", help="Enable Caption scoring")
    p.add_argument("--use-clustering", action="store_true", help="Enable KMeans clustering")
    p.add_argument("--use-category", action="store_true", help="Enable V3C Category Hard Filtering")
    p.add_argument("--smoothing-window", type=int, default=3, help="Window size for Gaussian temporal smoothing")
    p.add_argument("--smoothing-sigma", type=float, default=1.0)
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
    
    T_all, N = top_idx.shape
    
    # We unsort top_val to get the final score for frames in their original order
    final_score_unsorted = np.zeros((T_all, N), dtype=np.float32)
    for i in range(T_all):
        final_score_unsorted[i, top_idx[i]] = top_val[i]
        
    # The pure CLIP score is Final Score - Metadata Bonus
    clip_score = final_score_unsorted - observed['B_total']
    
    os.makedirs("figures/analysis", exist_ok=True)
    
    from tqdm import tqdm
    import concurrent.futures
    
    figure_tasks = []
    
    from scipy.stats import rankdata
    
    main_tasks = [(qi, info) for qi, info in enumerate(task_mapping) if info[1]]
    
    for qi, (ti, is_main, sub_id) in tqdm(main_tasks, desc="Generating per-task distributions & gathering tasks"):
        
        task = tasks_parsed[ti]
        task_id = task.get("task_id", f"T_unknown_{ti}")
        query_text = all_queries[qi]
        
        task_dir = os.path.join("figures/analysis", task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        fig.suptitle(f"Score Distributions for Task {task_id}", fontsize=16)
        
        meta_total = observed.get('ocr_bonus', np.zeros_like(final_score_unsorted))[qi] + \
                     observed.get('od_bonus', np.zeros_like(final_score_unsorted))[qi] + \
                     observed.get('caption_bonus', np.zeros_like(final_score_unsorted))[qi]
                     
        score_data = {
            "Total_Score": final_score_unsorted[qi],
            "CLIP_Visual": clip_score[qi],
            "Metadata_Total": meta_total,
            "OCR": observed.get('ocr_bonus', np.zeros_like(final_score_unsorted))[qi],
            "OD": observed.get('od_bonus', np.zeros_like(final_score_unsorted))[qi],
            "Captioning": observed.get('caption_bonus', np.zeros_like(final_score_unsorted))[qi]
        }
        
        ranks = {}
        for name, data in score_data.items():
            ranks[name] = rankdata(-data, method='min')
        
        for idx, (name, data) in enumerate(score_data.items()):
            ax = axes[idx]
            sns.histplot(data, bins=50, kde=True, ax=ax)
            ax.set_title(f"{name}\nMean: {np.mean(data):.4f}, Var: {np.var(data):.4f}")
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            
        plt.tight_layout()
        plt.savefig(os.path.join(task_dir, "distribution.png"))
        plt.close(fig)
        
        # Top 10 figures
        for cat_name, data in score_data.items():
            cat_dir = os.path.join(task_dir, cat_name)
            os.makedirs(cat_dir, exist_ok=True)
            
            # get top 10 indices
            top10_idx = np.argsort(data)[-10:][::-1]
            for rank, max_idx in enumerate(top10_idx, 1):
                max_val = data[max_idx]
                
                vid = str(first_vids[max_idx])
                ts_ms = int(first_ts[max_idx])
                
                scores_str = f"Scores -> Total: {score_data['Total_Score'][max_idx]:.3f} | CLIP: {score_data['CLIP_Visual'][max_idx]:.3f} | Meta: {score_data['Metadata_Total'][max_idx]:.3f} | OCR: {score_data['OCR'][max_idx]:.3f} | OD: {score_data['OD'][max_idx]:.3f} | Cap: {score_data['Captioning'][max_idx]:.3f}"
                ranks_str = f"Ranks -> Total: #{ranks['Total_Score'][max_idx]} | CLIP: #{ranks['CLIP_Visual'][max_idx]} | Meta: #{ranks['Metadata_Total'][max_idx]} | OCR: #{ranks['OCR'][max_idx]} | OD: #{ranks['OD'][max_idx]} | Cap: #{ranks['Captioning'][max_idx]}"
                
                meta_str = f"Video ID: {vid}, Frame MS: {ts_ms}\n"
                meta_str += scores_str + "\n"
                meta_str += ranks_str + "\n"
                    
                if vid in first_metadata and ts_ms in first_metadata[vid]:
                    meta = first_metadata[vid][ts_ms]
                    if cat_name == "OCR":
                        meta_str += f"OCR: {meta.get('ocr', '')}"
                    elif cat_name == "OD":
                        meta_str += f"OD: {', '.join(meta.get('objects', []))}"
                    elif cat_name == "Captioning":
                        meta_str += f"Caption: {meta.get('caption', '')}"
                    else: # Total or CLIP or Meta
                        meta_str += f"OCR: {meta.get('ocr', '')}\n"
                        meta_str += f"OD: {', '.join(meta.get('objects', []))}\n"
                        meta_str += f"Caption: {meta.get('caption', '')}"
                else:
                    meta_str += "No metadata found."
                    
                img_path = get_image_path_for_frame(vid, ts_ms, args.keyframes)
                out_img = os.path.join(cat_dir, f"top_{rank}.png")
                figure_tasks.append((task_id, query_text, cat_name, max_val, meta_str, img_path, out_img, rank))

    # === Global Top 10 across all tasks ===
    print("Generating Global Top 10 figures across all tasks...")
    global_dir = "figures/analysis/GLOBAL_TOP"
    os.makedirs(global_dir, exist_ok=True)
    
    global_meta_total = observed.get('ocr_bonus', np.zeros_like(final_score_unsorted)) + \
                        observed.get('od_bonus', np.zeros_like(final_score_unsorted)) + \
                        observed.get('caption_bonus', np.zeros_like(final_score_unsorted))
                        
    global_score_data = {
        "Total_Score": final_score_unsorted,
        "CLIP_Visual": clip_score,
        "Metadata_Total": global_meta_total,
        "OCR": observed.get('ocr_bonus', np.zeros_like(final_score_unsorted)),
        "OD": observed.get('od_bonus', np.zeros_like(final_score_unsorted)),
        "Captioning": observed.get('caption_bonus', np.zeros_like(final_score_unsorted))
    }
    
    global_ranks = {}
    for name, data in global_score_data.items():
        flat_data = data.flatten()
        flat_ranks = rankdata(-flat_data, method='min')
        global_ranks[name] = flat_ranks.reshape(data.shape)
    
    import csv
    csv_rows = []
    
    for cat_name, data in global_score_data.items():
        cat_dir = os.path.join(global_dir, cat_name)
        os.makedirs(cat_dir, exist_ok=True)
        
        # Flatten the matrix to find global top 10
        # data is shape [T_all, N]
        flat_data = data.flatten()
        top10_flat_idx = np.argsort(flat_data)[-10:][::-1]
        
        # Calculate average ranks for these 10 frames across all categories
        avg_ranks = {"Category_Top10": cat_name}
        for rank_cat_name, rank_matrix in global_ranks.items():
            flat_ranks = rank_matrix.flatten()
            top10_ranks = flat_ranks[top10_flat_idx]
            avg_ranks[f"AvgRank_{rank_cat_name}"] = np.mean(top10_ranks)
        csv_rows.append(avg_ranks)
        
        for rank, flat_idx in enumerate(top10_flat_idx, 1):
            qi, max_idx = np.unravel_index(flat_idx, data.shape)
            max_val = data[qi, max_idx]
            
            ti, is_main, sub_id = task_mapping[qi]
            task = tasks_parsed[ti]
            task_id = task.get("task_id", f"T_unknown_{ti}")
            query_text = all_queries[qi]
            
            vid = str(first_vids[max_idx])
            ts_ms = int(first_ts[max_idx])
            
            scores_str = f"Scores -> Total: {global_score_data['Total_Score'][qi, max_idx]:.3f} | CLIP: {global_score_data['CLIP_Visual'][qi, max_idx]:.3f} | Meta: {global_score_data['Metadata_Total'][qi, max_idx]:.3f} | OCR: {global_score_data['OCR'][qi, max_idx]:.3f} | OD: {global_score_data['OD'][qi, max_idx]:.3f} | Cap: {global_score_data['Captioning'][qi, max_idx]:.3f}"
            ranks_str = f"Global Ranks -> Total: #{global_ranks['Total_Score'][qi, max_idx]} | CLIP: #{global_ranks['CLIP_Visual'][qi, max_idx]} | Meta: #{global_ranks['Metadata_Total'][qi, max_idx]} | OCR: #{global_ranks['OCR'][qi, max_idx]} | OD: #{global_ranks['OD'][qi, max_idx]} | Cap: #{global_ranks['Captioning'][qi, max_idx]}"
            
            meta_str = f"Task: {task_id} | Video ID: {vid}, Frame MS: {ts_ms}\n"
            meta_str += scores_str + "\n"
            meta_str += ranks_str + "\n"
            
            if vid in first_metadata and ts_ms in first_metadata[vid]:
                meta = first_metadata[vid][ts_ms]
                if cat_name == "OCR":
                    meta_str += f"OCR: {meta.get('ocr', '')}"
                elif cat_name == "OD":
                    meta_str += f"OD: {', '.join(meta.get('objects', []))}"
                elif cat_name == "Captioning":
                    meta_str += f"Caption: {meta.get('caption', '')}"
                else: # Total or CLIP or Meta
                    meta_str += f"OCR: {meta.get('ocr', '')}\n"
                    meta_str += f"OD: {', '.join(meta.get('objects', []))}\n"
                    meta_str += f"Caption: {meta.get('caption', '')}"
            else:
                meta_str += "No metadata found."
                
            img_path = get_image_path_for_frame(vid, ts_ms, args.keyframes)
            out_img = os.path.join(cat_dir, f"top_{rank}.png")
            figure_tasks.append((task_id, query_text, cat_name, max_val, meta_str, img_path, out_img, rank))

    # Write CSV
    csv_path = os.path.join(global_dir, "average_ranks.csv")
    if csv_rows:
        fieldnames = ["Category_Top10"] + [f"AvgRank_{name}" for name in global_score_data.keys()]
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Exported average ranks to {csv_path}")

    print(f"Rendering {len(figure_tasks)} top-score images using multiprocessing...")
    import multiprocessing
    ctx = multiprocessing.get_context("fork")
        
    with concurrent.futures.ProcessPoolExecutor(max_workers=16, mp_context=ctx) as executor:
        futures = [executor.submit(create_top_score_figure, *args) for args in figure_tasks]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Rendering Images"):
            pass

    print("Analysis complete. Check figures/analysis/ directory.")

if __name__ == "__main__":
    main()
