import numpy as np
import torch

def apply_reranking(ti, task, candidates, default_top_videos, emb_lookup, first_emb, expanded_Q_embs, dev="cuda"):
    """
    Rerank the candidates using the embeddings of 5 expanded queries, then cluster.
    """
    max_preds = task.get("max_predictions", default_top_videos)
    
    # Check if we have expanded queries to rerank
    if not candidates:
        return {"task_id": task["task_id"], "results": []}
        
    filtered_candidates = []
    
    if expanded_Q_embs is not None and "expanded_queries" in task and len(task["expanded_queries"]) > 0:
        num_exp = len(task["expanded_queries"])
        # Get embeddings for the expanded queries
        # expanded_Q_embs is [T_all * num_exp, D]
        q_embs = expanded_Q_embs[ti * num_exp : (ti + 1) * num_exp]  # [num_exp, D]
        
        # Get candidate embeddings
        cand_indices = []
        valid_cands = []
        for c in candidates:
            idx = emb_lookup.get((c["video_id"], c["frame_ms"]))
            if idx is not None:
                cand_indices.append(idx)
                valid_cands.append(c)
                
        if len(valid_cands) > 0:
            cand_embs = torch.from_numpy(first_emb[cand_indices]).to(dev).float()  # [K, D]
            
            # Compute similarities: [5, D] @ [D, K] -> [5, K]
            sims = (q_embs @ cand_embs.T).cpu().numpy()
            
            # Create rankings for each of the 5 queries
            k_rrf = 60
            rrf_scores = np.zeros(len(valid_cands), dtype=np.float32)
            
            # Add original score to RRF
            for c_idx, c in enumerate(valid_cands):
                # c["sim"] is the original RRF score or cosine score.
                # Since candidates are already sorted, we can use their original rank
                orig_rank = c_idx + 1
                rrf_scores[c_idx] += 1.0 / (k_rrf + orig_rank)
                
            # Add expanded queries scores
            for q_idx in range(num_exp):
                q_sims = sims[q_idx]
                sorted_indices = np.argsort(-q_sims)
                for rank, c_idx in enumerate(sorted_indices, 1):
                    rrf_scores[c_idx] += 1.0 / (k_rrf + rank)
                    
            # Sort candidates by the new RRF score
            for c_idx in range(len(valid_cands)):
                valid_cands[c_idx]["sim"] = float(rrf_scores[c_idx])
                
            candidates = sorted(valid_cands, key=lambda x: x["sim"], reverse=True)
            
    # Flatten all candidates and their frames into a single pool
    all_frames_flat = []
    for c in candidates:
        if "all_frames" in c and len(c["all_frames"]) > 0:
            for f in c["all_frames"]:
                all_frames_flat.append({
                    "video_id": c["video_id"],
                    "frame_ms": f["frame_ms"],
                    "sim": float(f["sim"])
                })
        else:
            all_frames_flat.append({
                "video_id": c["video_id"],
                "frame_ms": c["frame_ms"],
                "sim": float(c["sim"])
            })
            
    # Sort all frames descending by sim
    all_frames_flat.sort(key=lambda x: x["sim"], reverse=True)
    
    results = []
    # Define an overlap threshold in milliseconds
    overlap_threshold = task.get("overlap_threshold", 5000)
    added_frames = {}  # video_id -> list of selected frame_ms
    
    for f in all_frames_flat:
        if len(results) >= max_preds:
            break
            
        vid = f["video_id"]
        f_ms = f["frame_ms"]
        
        # Check overlap
        is_overlap = False
        if vid in added_frames:
            for added_ms in added_frames[vid]:
                if abs(f_ms - added_ms) < overlap_threshold:
                    is_overlap = True
                    break
                    
        if not is_overlap:
            if vid not in added_frames:
                added_frames[vid] = []
            added_frames[vid].append(f_ms)
            results.append({
                "rank": len(results) + 1,
                "video_id": vid,
                "frame_ms": f_ms,
            })
            
    return {"task_id": task["task_id"], "results": results}
