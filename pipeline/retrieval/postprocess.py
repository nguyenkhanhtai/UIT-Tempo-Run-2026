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
            
    # Clustering logic: Chỉ giữ lại 1 frame duy nhất (có score cao nhất) cho mỗi video
    seen_vids = set()
    for c in candidates:
        if c["video_id"] not in seen_vids:
            seen_vids.add(c["video_id"])
            filtered_candidates.append(c)
    
    results = []
    
    for res in filtered_candidates:
        if len(results) >= max_preds:
            break
            
        if "all_frames" in res and len(res["all_frames"]) > 0:
            # Sắp xếp tất cả các frame của video này theo điểm similarity giảm dần
            sorted_frames = sorted(res["all_frames"], key=lambda x: x["sim"], reverse=True)
            # Lấy tối đa 2 frame
            top_frames = sorted_frames[:2]
            
            for f in top_frames:
                if len(results) >= max_preds:
                    break
                results.append({
                    "rank": len(results) + 1, 
                    "video_id": res["video_id"],
                    "frame_ms": f["frame_ms"],
                })
        else:
            if len(results) >= max_preds:
                break
            results.append({
                "rank": len(results) + 1, 
                "video_id": res["video_id"],
                "frame_ms": res["frame_ms"],
            })
            
    return {"task_id": task["task_id"], "results": results}
