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

# 2. Decorate compute_similarity to observe the final scores for ALL frames
orig_compute = scorer.compute_similarity
def observed_compute(Q_list, idx_list, T_all, N, K, dev):
    print(f"[Observer] Intercepting compute_similarity. Running with K={N} to capture all frames...")
    # Call the original function with K=N to get the full final score array
    top_idx, top_val = orig_compute(Q_list, idx_list, T_all, N, N, dev)
    
    # Store the visual pass only. Audio may use a routed ASR-query subset,
    # so overwriting this would break per-query analysis shapes.
    if 'top_idx' not in observed:
        observed['top_idx'] = top_idx
        observed['top_val'] = top_val
    
    # Return only the top K as originally requested to avoid slowing down downstream tasks
    return top_idx[:, :K], top_val[:, :K]

# Apply decorators to the namespace where they are actually used
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
    
    wrapped_query = textwrap.fill(f"Task: {task_id} (Rank #{rank}) | Instruction/Query: {query_text}", width=90)
    
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

def create_dp_sequence_figure(task_id, full_query, scenes, sequence_ms, vid, kf_dir, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    num_scenes = len(scenes)
    if num_scenes == 0:
        return
        
    fig, axes = plt.subplots(1, num_scenes, figsize=(6 * num_scenes, 6))
    if num_scenes == 1:
        axes = [axes]
        
    wrapped_query = textwrap.fill(f"Task: {task_id}\nFull Query: {full_query}", width=120)
    fig.suptitle(wrapped_query + "\n[DP Optimal Scene Sequence Match]", fontsize=16, fontweight='bold')
    
    for i in range(num_scenes):
        ax = axes[i]
        scene_text = scenes[i]
        ts_ms = sequence_ms[i] if i < len(sequence_ms) else 0
        
        img_path = get_image_path_for_frame(vid, ts_ms, kf_dir)
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Image Error", fontsize=12, ha='center')
        else:
            ax.text(0.5, 0.5, "No Image", fontsize=12, ha='center')
            
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        wrapped_scene = textwrap.fill(f"Scene {i+1}: {scene_text}", width=50)
        ax.set_title(wrapped_scene, fontsize=12, pad=10)
        ax.set_xlabel(f"Video: {vid}\nTimestamp: {ts_ms} ms", fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
        
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--visual-shards", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--keyframes", required=True)
    p.add_argument("--vlms", nargs="+", required=True)
    
    # Audio args
    p.add_argument("--use-audio", action="store_true")
    p.add_argument("--audio-shards", default="")
    p.add_argument("--audio-model", default="sentence-transformers/all-MiniLM-L6-v2")
    
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cand-keyframes", type=int, default=2000)
    p.add_argument("--top-videos", type=int, default=10)
    p.add_argument("--out", default="analysis_sub.json", help="Output file path")

    p.add_argument("--use-sequential", action="store_true", help="Enable Sequential DP Matching with SaT")
    p.add_argument("--scene-segmenter", type=str, default="regex", help="Engine to use for scene segmentation")
    p.add_argument("--object-segmenter", type=str, default="spacy", help="Engine to use for object segmentation")
    p.add_argument("--smoothing-window", type=int, default=3, help="Window size for Gaussian temporal smoothing")
    p.add_argument("--smoothing-sigma", type=float, default=1.0)
    p.add_argument("--num-expansions", type=int, default=2, help="Number of queries to expand for reranking")
    p.add_argument("--overlap-threshold", type=int, default=5000, help="Overlap threshold in ms for postprocessing")
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
    if len(ret_data) == 7:
        tasks_parsed, task_mapping, first_vids, first_ts, first_metadata, all_queries, all_candidates = ret_data
    else:
        tasks_parsed, task_mapping, first_vids, first_ts, first_metadata, all_queries = ret_data
        all_candidates = None

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
        
    os.makedirs("figures/analysis", exist_ok=True)
    
    from tqdm import tqdm
    import concurrent.futures
    
    figure_tasks = []
    
    from scipy.stats import rankdata
    
    # Group queries by task and sentence
    import collections
    task_groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for qi, (ti, sent_idx, seg_idx) in enumerate(task_mapping):
        task_groups[ti][sent_idx].append(qi)
        
    for ti, task in enumerate(tqdm(tasks_parsed, desc="Generating per-segment and aggregated distributions")):
        task_id = task.get("task_id", f"T_unknown_{ti}")
        
        # We will collect all score arrays to plot for this task
        # format: (name, query_text, score_array)
        plots_to_make = []
        
        task_total_scores = []
        
        for sent_idx, segments in task_groups[ti].items():
            sent_scores_list = []
            
            for seg_idx, qi in enumerate(segments):
                query_text = all_queries[qi]
                seg_name = f"Sent{sent_idx+1}_Seg{seg_idx+1}"
                if not os.environ.get("SPLIT_QUERY", "true").lower() == "true":
                    seg_name = "Full_Query"
                
                seg_score = final_score_unsorted[qi]
                plots_to_make.append((seg_name, query_text, seg_score))
                sent_scores_list.append(seg_score)
                
            # Compute Sentence Total (Average of segments)
            sent_total = np.mean(sent_scores_list, axis=0) if len(sent_scores_list) > 1 else sent_scores_list[0]
            sent_text = " ".join([all_queries[qi] for qi in segments])
            plots_to_make.append((f"Sent{sent_idx+1}_Total", sent_text, sent_total))
            task_total_scores.append(sent_total)
                
        # Compute Task Total (Average of all sentences, as a basic visualization of task score)
        task_total = np.mean(task_total_scores, axis=0) if len(task_total_scores) > 1 else task_total_scores[0]
        full_query_text = task.get("description", task.get("query", " ".join([all_queries[qi] for qi in sum(task_groups[ti].values(), [])])))
        plots_to_make.append(("Task_Total", full_query_text, task_total))
        
        # Precompute ranks for all categories in this task
        task_all_ranks = {}
        for seg_name, _, score_arr in plots_to_make:
            task_all_ranks[seg_name] = rankdata(-score_arr, method='min')
            
        # Write segments.txt
        task_base_dir = os.path.join("figures/analysis", task_id)
        os.makedirs(task_base_dir, exist_ok=True)
        with open(os.path.join(task_base_dir, "segments.txt"), "w", encoding="utf-8") as f:
            f.write(f"Task ID: {task_id}\n")
            f.write(f"Full Query: {full_query_text}\n")
            if "route" in task:
                f.write(f"LLM Routing Decision: {task['route']}\n")
            f.write("\n")
            
            scenes_text_list = []
            for sent_idx, segments in task_groups[ti].items():
                scene_texts = [all_queries[qi] for qi in segments]
                scene_full = " | ".join(scene_texts)
                scenes_text_list.append(scene_full)
                
                f.write(f"Scene {sent_idx+1}:\n")
                for seg_idx, qi in enumerate(segments):
                    f.write(f"  - Object {seg_idx+1}: {all_queries[qi]}\n")
                f.write("\n")
                f.write("\n")
        # Generate DP Sequence Visualization
        if all_candidates is not None and ti < len(all_candidates):
            candidates = all_candidates[ti][1]
            if candidates:
                candidates.sort(key=lambda x: x["sim"], reverse=True)
                top_cand = candidates[0]
                if "sequence_ms" in top_cand:
                    dp_out = os.path.join(task_base_dir, "DP_Best_Sequence.png")
                    create_dp_sequence_figure(
                        task_id, full_query_text, scenes_text_list, 
                        top_cand["sequence_ms"], top_cand["video_id"], 
                        args.keyframes, dp_out
                    )
            
        for seg_name, query_text, score_arr in plots_to_make:
            task_dir = os.path.join("figures/analysis", task_id, seg_name)
            os.makedirs(task_dir, exist_ok=True)
            
            fig, axes = plt.subplots(1, 1, figsize=(6, 5))
            fig.suptitle(f"Score Distributions for Task {task_id} ({seg_name})", fontsize=16)
            
            score_data = {
                "Total_Score": score_arr,
                "CLIP_Visual": score_arr,
            }
        
            for idx, (name, data) in enumerate(score_data.items()):
                sns.histplot(data, bins=50, kde=False, ax=axes)
                axes.set_title(f"{name}\nMean: {np.mean(data):.4f}, Var: {np.var(data):.4f}")
                axes.set_xlabel("Score")
                axes.set_ylabel("Count")
                
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
                    
                    scores_str = f"Scores -> Total: {score_data['Total_Score'][max_idx]:.3f} | CLIP: {score_data['CLIP_Visual'][max_idx]:.3f}"
                    
                    curr_rank = task_all_ranks[seg_name][max_idx]
                    ranks_str = f"Rank in {seg_name}: #{curr_rank}\n"
                    
                    other_ranks = []
                    for other_name, rank_arr in task_all_ranks.items():
                        if other_name != seg_name:
                            other_ranks.append(f"{other_name}: #{rank_arr[max_idx]}")
                    
                    ranks_str += "Ranks in other queries:\n"
                    for chunk_idx in range(0, len(other_ranks), 3):
                        ranks_str += " | ".join(other_ranks[chunk_idx:chunk_idx+3]) + "\n"
                    
                    meta_str = f"Video ID: {vid}, Frame MS: {ts_ms}\n"
                    meta_str += scores_str + "\n"
                    meta_str += ranks_str
                        
                    if vid in first_metadata and ts_ms in first_metadata[vid]:
                        meta = first_metadata[vid][ts_ms]
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
    
    global_score_data = {
        "Total_Score": final_score_unsorted,
        "CLIP_Visual": final_score_unsorted,
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
            
            scores_str = f"Scores -> Total: {global_score_data['Total_Score'][qi, max_idx]:.3f} | CLIP: {global_score_data['CLIP_Visual'][qi, max_idx]:.3f}"
            ranks_str = f"Global Ranks -> Total: #{global_ranks['Total_Score'][qi, max_idx]} | CLIP: #{global_ranks['CLIP_Visual'][qi, max_idx]}"
            
            meta_str = f"Task: {task_id} | Video ID: {vid}, Frame MS: {ts_ms}\n"
            meta_str += scores_str + "\n"
            meta_str += ranks_str + "\n"
            
            if vid in first_metadata and ts_ms in first_metadata[vid]:
                meta = first_metadata[vid][ts_ms]
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
