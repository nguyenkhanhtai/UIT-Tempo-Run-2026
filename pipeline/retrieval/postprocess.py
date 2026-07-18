"""
Purpose: Applies post-processing algorithms to diversify the final Top-K retrieval results.
"""
import numpy as np

def apply_clustering(task, candidates, default_top_videos):
    max_preds = task.get("max_predictions", default_top_videos)
    
    # 1. Chỉ giữ lại 1 frame duy nhất (có score cao nhất) cho mỗi video
    seen_vids = set()
    filtered_candidates = []
    # Đảm bảo candidates đã được sort theo sim giảm dần
    candidates = sorted(candidates, key=lambda x: x["sim"], reverse=True)
    for c in candidates:
        if c["video_id"] not in seen_vids:
            seen_vids.add(c["video_id"])
            filtered_candidates.append(c)
    
    final_results = filtered_candidates[:max_preds]
        
    results = []
    for rank, res in enumerate(final_results, 1):
        results.append({
            "rank": rank, "video_id": res["video_id"],
            "frame_ms": res["frame_ms"],
        })
    return {"task_id": task["task_id"], "results": results}
