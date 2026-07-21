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
                "end_ms": c_end
            }
            if "best_score_ms" in c:
                res_dict["best_score_ms"] = c["best_score_ms"]
            if "best_score_sec" in c:
                res_dict["best_score_sec"] = c["best_score_sec"]
                
            results.append(res_dict)
            
    return {"task_id": task["task_id"], "results": results}

def rrf_fuse(candidates1, candidates2=None, k=60):
    fused_tasks = []
    candidates2 = candidates2 or [(task, []) for task, _ in candidates1]

    for (task, cands1), (_, cands2) in zip(candidates1, candidates2):
        scores = {}
        info = {}
        route = task.get("route", {"Use_asr": False, "asr_query": None})

        # ALWAYS ENFORCE VISUAL = TRUE to avoid catastrophic recall drop
        for rank, cand in enumerate(cands1):
            vid = cand["video_id"]
            scores[vid] = scores.get(vid, 0) + 1.0 / (k + rank)
            info[vid] = cand

        if route.get("Use_asr", False) and route.get("asr_query"):
            for rank, cand in enumerate(cands2):
                vid = cand["video_id"]
                scores[vid] = scores.get(vid, 0) + 1.0 / (k + rank)
                if vid not in info:
                    info[vid] = cand

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for vid, sc in sorted_scores:
            info[vid]["sim"] = sc
        fused = [info[vid] for vid, sc in sorted_scores]
        fused_tasks.append((task, fused))
    return fused_tasks

def postprocess_pipeline(args, tasks, all_candidates_visual, all_candidates_audio, first_vids, first_ts, first_emb, expanded_Q_embs):
    if all_candidates_audio is not None:
        print("[retrieve] Fusing Visual and ASR Audio candidates...")
        all_candidates_final = rrf_fuse(all_candidates_visual, all_candidates_audio, k=60)
    else:
        all_candidates_final = all_candidates_visual

    print("[retrieve] Postprocessing and Reranking with Query Expansion...")
    
    # Pre-compute embedding lookup table for fast access
    print("[retrieve] Building embedding lookup table...")
    emb_lookup = {(v, t): i for i, (v, t) in enumerate(zip(first_vids, first_ts))}
    
    preds = []
    for ti, (task, candidates) in enumerate(all_candidates_final):
        res = apply_reranking(ti, task, candidates, args.top_videos, emb_lookup, first_emb, expanded_Q_embs, dev=args.device)
        preds.append(res)
        
    from pipeline.retrieval.reranker.od_reranker import apply_od_reranking
    preds = apply_od_reranking(tasks, preds)
        
    from pipeline.retrieval.reranker.vlm_rescorer import apply_vlm_rescoring
    preds = apply_vlm_rescoring(tasks, preds)
        
    return preds, all_candidates_final
