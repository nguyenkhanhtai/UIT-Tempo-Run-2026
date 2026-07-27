import numpy as np
import torch

def format_results(task, candidates, default_top_videos):
    """
    Format candidates into the final result structure and handle intervals.
    """
    max_preds = task.get("max_predictions", default_top_videos)
    
    if not candidates:
        return {"task_id": task["task_id"], "results": []}
        
    # Sort candidates descending by sim
    candidates.sort(key=lambda x: x["sim"], reverse=True)
    
    results = []
    added_intervals = {}  # video_id -> list of [start_ms, end_ms]
    
    for c in candidates:
        if len(results) >= max_preds:
            break
            
        vid = c["video_id"]
        
        # Determine sequence interval
        if "sequence_ms" in c and len(c["sequence_ms"]) > 0:
            c_start = c["sequence_ms"][0]
            c_end = c["sequence_ms"][-1]
            if c_start > c_end:
                c_start, c_end = c_end, c_start
        else:
            c_start = c["frame_ms"]
            c_end = c["frame_ms"]
            
        # Check overlap
        is_overlap = False
        if vid in added_intervals:
            for acc_start, acc_end in added_intervals[vid]:
                if max(c_start, acc_start) <= min(c_end, acc_end):
                    is_overlap = True
                    break
                    
        if not is_overlap:
            if vid not in added_intervals:
                added_intervals[vid] = []
            added_intervals[vid].append([c_start, c_end])
            
            res_dict = {
                "rank": len(results) + 1,
                "video_id": vid,
                "frame_ms": c["frame_ms"],
                "start_ms": c_start,
                "end_ms": c_end,
                "sim": float(c.get("sim", 0.0))
            }
            if "best_score_ms" in c:
                res_dict["best_score_ms"] = c["best_score_ms"]
            if "best_score_sec" in c:
                res_dict["best_score_sec"] = c["best_score_sec"]
                
            results.append(res_dict)
            
    return {"task_id": task["task_id"], "results": results}


def apply_global_exclusion(preds, overlap_threshold_ms=2000):
    """
    Global NMS: ensure that no two tasks can claim the same video segment.
    """
    # 1. Flatten all predictions across all tasks
    all_cands = []
    for t_idx, p in enumerate(preds):
        for c in p.get("results", []):
            all_cands.append((t_idx, c))
            
    # 2. Sort globally by sim (descending)
    all_cands.sort(key=lambda x: x[1].get("sim", 0.0), reverse=True)
    
    # 3. Greedy assignment
    new_results = [[] for _ in preds]
    used_intervals = {} # video_id -> list of [start, end]
    
    for t_idx, c in all_cands:
        max_preds = 10  # Standard target per task
        if len(new_results[t_idx]) >= max_preds:
            continue
            
        vid = c["video_id"]
        c_start = c["start_ms"] - overlap_threshold_ms
        c_end = c["end_ms"] + overlap_threshold_ms
        
        is_overlap = False
        if vid in used_intervals:
            for acc_start, acc_end in used_intervals[vid]:
                if max(c_start, acc_start) <= min(c_end, acc_end):
                    is_overlap = True
                    break
                    
        if not is_overlap:
            if vid not in used_intervals:
                used_intervals[vid] = []
            used_intervals[vid].append([c_start, c_end])
            
            # Re-assign rank
            c["rank"] = len(new_results[t_idx]) + 1
            new_results[t_idx].append(c)
            
    for t_idx, p in enumerate(preds):
        p["results"] = new_results[t_idx]
        
    return preds

def postprocess_pipeline(args, tasks, all_candidates_visual, first_vids, first_ts, first_emb):
    all_candidates_final = all_candidates_visual

    print("[retrieve] Formatting Results...")
    
    preds = []
    for ti, (task, candidates) in enumerate(all_candidates_final):
        res = format_results(task, candidates, args.top_videos)
        preds.append(res)
        
    print("[retrieve] Applying Global Exclusion (Global NMS)...")
    preds = apply_global_exclusion(preds, overlap_threshold_ms=args.overlap_threshold)
        
    return preds, all_candidates_final
