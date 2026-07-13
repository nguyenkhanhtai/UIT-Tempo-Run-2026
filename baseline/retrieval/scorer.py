"""
Purpose: Computes similarity scores between text queries and video frames using
Dual Softmax (Mean Centering) and aggregates base scores with Causal Bonuses.
"""
import time
import collections
import torch
import difflib
import math

def get_ocr_similarity(query, text):
    if not query or not text:
        return 0.0
    query = query.lower()
    text = text.lower()
    if query in text:
        return 1.0
        
    n = len(query)
    if len(text) <= n:
        return difflib.SequenceMatcher(None, query, text).ratio()
        
    max_ratio = 0.0
    for i in range(len(text) - n + 1):
        window = text[i:i+n]
        ratio = difflib.SequenceMatcher(None, query, window).ratio()
        if ratio > max_ratio:
            max_ratio = ratio
    return max_ratio

def precompute_metadata_bonus(tasks, task_mapping, vids, ts, metadata):
    import numpy as np
    from .parser import extract_ocr_queries, extract_object_queries
    
    T_all = len(task_mapping)
    N = len(vids)
    B = np.zeros((T_all, N), dtype=np.float32)
    
    t0 = time.time()
    
    qi_to_queries = {}
    for qi, (ti, is_main, sub_id) in enumerate(task_mapping):
        if is_main:
            desc = tasks[ti]["description"]
            ocr_q = extract_ocr_queries(desc)
            obj_q = extract_object_queries(desc)
            if ocr_q or obj_q:
                qi_to_queries[qi] = (ocr_q, obj_q)
                
    if not qi_to_queries:
        return B
        
    for r in range(N):
        v = str(vids[r])
        center = int(ts[r])
        if v in metadata and center in metadata[v]:
            meta = metadata[v][center]
            meta_ocr = meta.get("ocr", "")
            meta_objs = meta.get("objects", [])
            meta_words = set(w for obj in meta_objs for w in obj.lower().split())
            
            for qi, (ocr_query, object_query) in qi_to_queries.items():
                bonus = 0.0
                if ocr_query:
                    for q in ocr_query:
                        sim = get_ocr_similarity(q, meta_ocr)
                        bonus += 0.15 * math.exp(4 * (sim - 1.0))
                if object_query:
                    for qw in object_query:
                        if qw in meta_words:
                            bonus += 0.05
                
                if bonus > 0:
                    B[qi, r] = bonus
                    
    print(f"[retrieve] Precomputed metadata bonus for {N} frames in {time.time()-t0:.1f}s", flush=True)
    return B

def compute_similarity(Q_list, idx_list, B, T_all, N, K, dev):
    CH = 10_000
    top_val = torch.full((T_all, K), float("-inf"), device=dev, dtype=torch.float32)
    top_idx = torch.zeros((T_all, K), device=dev, dtype=torch.long)
    t0 = time.time()
    
    num_models = len(Q_list)
    
    for s in range(0, N, CH):
        e = min(s + CH, N)
        
        ensembled_sims = torch.zeros((T_all, e - s), device=dev, dtype=torch.float32)
        
        for m in range(num_models):
            Q = Q_list[m]
            idx_chunk = torch.from_numpy(idx_list[m][s:e]).to(dev).float()
            
            sims = Q @ idx_chunk.T
            # Dual Softmax (Mean Centering)
            sims = sims - sims.mean(dim=0, keepdim=True)
            
            ensembled_sims += sims
            
        ensembled_sims /= num_models
        
        # Add metadata bonus matrix chunk (B_chunk) before topk
        B_chunk = torch.from_numpy(B[:, s:e]).to(dev).float()
        ensembled_sims += B_chunk
        
        cat_v = torch.cat([top_val, ensembled_sims], 1)
        cat_i = torch.cat([top_idx, torch.arange(s, e, device=dev).expand(T_all, e - s)], 1)
        top_val, sel = cat_v.topk(K, dim=1)
        top_idx = torch.gather(cat_i, 1, sel)
        
    print(f"[retrieve] scored {N} keyframes x {T_all} total queries using {num_models} models in {time.time()-t0:.0f}s", flush=True)
    return top_idx.cpu().numpy(), top_val.float().cpu().numpy()

def aggregate_scores(task_mapping, top_idx, top_val, tasks, vids, ts, emb, metadata):
    task_results = collections.defaultdict(lambda: {'main': None, 'subs': []})
    for qi, (ti, is_main, sub_id) in enumerate(task_mapping):
        if is_main:
            task_results[ti]['main'] = (top_idx[qi], top_val[qi])
        else:
            task_results[ti]['subs'].append((sub_id, top_idx[qi], top_val[qi]))

    all_candidates = []
    
    from .parser import extract_ocr_queries, extract_object_queries
    
    for ti, task in enumerate(tasks):
        desc = task["description"]
        ocr_query = extract_ocr_queries(desc)
        object_query = extract_object_queries(desc)
        
        main_rows, main_sims = task_results[ti]['main']
        subs = task_results[ti]['subs']
        
        # 1. Base Score (which now inherently includes Meta Bonus!)
        v_main_scores = collections.defaultdict(list)
        v_main_centers = {}
        v_main_rows = {}
        
        for r, sim in zip(main_rows, main_sims):
            v = str(vids[r])
            center = int(ts[r])
                
            v_main_scores[v].append(float(sim))
            if v not in v_main_centers:
                v_main_centers[v] = center
                v_main_rows[v] = r
                
        # 2. Sub-queries max scores
        v_sub_max = collections.defaultdict(lambda: collections.defaultdict(float))
        v_sub_center = collections.defaultdict(dict)
        for sub_id, rows, sims in subs:
            for r, sim in zip(rows, sims):
                v = str(vids[r])
                if sub_id not in v_sub_max[v] or sim > v_sub_max[v][sub_id]:
                    v_sub_max[v][sub_id] = float(sim)
                    v_sub_center[v][sub_id] = int(ts[r])
                    
        # 3. Aggregate (Base + Sub Booster + Causal Bonus)
        candidates = []
        for v, m_scores in v_main_scores.items():
            main_sc = m_scores[0]
            main_bonus = sum(m_scores[1:4]) / len(m_scores[1:4]) * 0.1 if len(m_scores) > 1 else 0.0
            base_score = main_sc + main_bonus
            
            sub_bonus = 0.0
            causal_bonus = 0.0
            if v in v_sub_max:
                subs_for_v = v_sub_max[v]
                for sid, sim in subs_for_v.items():
                    decay_weight = 0.2 / sid
                    sub_bonus += sim * decay_weight
                
                if len(subs_for_v) > 1:
                    ordered = True
                    sorted_sub_ids = sorted(subs_for_v.keys())
                    for i in range(len(sorted_sub_ids) - 1):
                        id1 = sorted_sub_ids[i]
                        id2 = sorted_sub_ids[i+1]
                        if v_sub_center[v][id1] >= v_sub_center[v][id2]:
                            ordered = False
                            break
                    if ordered:
                        causal_bonus = abs(base_score) * 0.1
                        
            final_sc = base_score + sub_bonus + causal_bonus
            candidates.append({
                "video_id": v,
                "frame_ms": v_main_centers[v],
                "sim": final_sc,
                "feat": emb[v_main_rows[v]]
            })
            
        candidates.sort(key=lambda x: x["sim"], reverse=True)
        all_candidates.append((task, candidates))
        
    return all_candidates
