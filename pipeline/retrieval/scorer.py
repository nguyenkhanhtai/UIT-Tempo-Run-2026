"""
Purpose: Computes similarity scores between text queries and video frames using
Dual Softmax (Mean Centering) and aggregates base scores with Causal Bonuses.
"""
import time
import collections
import torch
import difflib
import math

# Toggles for metadata scoring
USE_OCR = True
USE_OD = True
USE_CAPTIONING = True

# --- GLOBAL VARS FOR MULTIPROCESSING ---
_G_vids = None
_G_ts = None
_G_metadata = None
_G_qi_to_queries = None
_G_sim_matrix = None
_G_query_caption_to_idx = None
_G_meta_caption_to_idx = None
_G_use_st = False

def _worker_process_chunk(start_r, end_r):
    chunk_updates = []
    frames_processed = 0
    for r in range(start_r, end_r):
        frames_processed += 1
        v = str(_G_vids[r])
        center = int(_G_ts[r])
        if v in _G_metadata and center in _G_metadata[v]:
            meta = _G_metadata[v][center]
            meta_ocr_lower = meta.get("ocr", "").lower()
            meta_objs = meta.get("objects", [])
            meta_caption = meta.get("caption", "")
            meta_words = set(w for obj in meta_objs for w in obj.lower().split())
            
            for qi, (ocr_query, object_query, caption_query) in _G_qi_to_queries.items():
                b_ocr = 0.0
                b_od = 0.0
                b_cap = 0.0
                if ocr_query and USE_OCR:
                    for q in ocr_query:
                        sim = get_ocr_similarity(q, meta_ocr_lower)
                        b_ocr += 0.15 * math.exp(4 * (sim - 1.0))
                if object_query and USE_OD:
                    for qw in object_query:
                        if qw in meta_words:
                            b_od += 0.05
                if meta_caption and USE_CAPTIONING:
                    if _G_use_st and _G_sim_matrix is not None:
                        q_idx = _G_query_caption_to_idx.get(caption_query)
                        m_idx = _G_meta_caption_to_idx.get(meta_caption)
                        if q_idx is not None and m_idx is not None:
                            cap_sim = float(_G_sim_matrix[q_idx, m_idx])
                        else:
                            cap_sim = 0.0
                    else:
                        cap_sim = get_caption_similarity(caption_query, meta_caption)
                        
                    if cap_sim > 0.4:
                        b_cap += 0.3 * cap_sim
                
                bonus = b_ocr + b_od + b_cap
                if bonus > 0:
                    chunk_updates.append((qi, r, bonus, b_ocr, b_od, b_cap))
    return frames_processed, chunk_updates

def get_ocr_similarity(query, text):
    if not query or not text:
        return 0.0
    query = query.lower()
    text = text.lower()
    if query in text:
        return 1.0
        
    try:
        from rapidfuzz import fuzz
        # partial_ratio finds the optimal substring matching and returns 0-100
        return fuzz.partial_ratio(query, text) / 100.0
    except ImportError:
        # Fallback if rapidfuzz somehow fails to load
        query_words = query.split()
        text_words = text.split()
        n_q = len(query_words)
        
        if n_q == 0 or not text_words:
            return 0.0
            
        if len(text_words) <= n_q:
            return difflib.SequenceMatcher(None, " ".join(query_words), " ".join(text_words)).ratio()
            
        max_ratio = 0.0
        matcher = difflib.SequenceMatcher(None, query)
        
        for i in range(len(text_words) - n_q + 1):
            window = " ".join(text_words[i:i+n_q])
            if window == query: return 1.0
            matcher.set_seq2(window)
            if matcher.real_quick_ratio() > max_ratio and matcher.quick_ratio() > max_ratio:
                ratio = matcher.ratio()
                if ratio > max_ratio:
                    max_ratio = ratio
                    if max_ratio > 0.95: return max_ratio
        return max_ratio

def get_caption_similarity(query, caption):
    if not query or not caption:
        return 0.0
    query = set(query.lower().split())
    caption = set(caption.lower().split())
    if not query or not caption:
        return 0.0
    # Calculate word overlap ratio
    overlap = len(query.intersection(caption))
    return overlap / len(query)

def precompute_metadata_bonus(tasks, task_mapping, vids, ts, metadata, return_components=False):
    import numpy as np
    from .parser import extract_ocr_queries, extract_object_queries
    
    T_all = len(task_mapping)
    N = len(vids)
    print(N)
    B = np.zeros((T_all, N), dtype=np.float32)
    B_ocr = np.zeros((T_all, N), dtype=np.float32) if return_components else None
    B_od = np.zeros((T_all, N), dtype=np.float32) if return_components else None
    B_cap = np.zeros((T_all, N), dtype=np.float32) if return_components else None

    
    t0 = time.time()
    
    use_st = False
    if USE_CAPTIONING:
        try:
            from sentence_transformers import SentenceTransformer, util
            print("[retrieve] Loading sentence-transformers model for caption similarity...", flush=True)
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            st_model = SentenceTransformer('all-MiniLM-L6-v2', device=device, model_kwargs={"attn_implementation": "eager"})
            use_st = True
        except ImportError:
            print("[retrieve] sentence-transformers not found, falling back to word overlap.", flush=True)
            use_st = False
    
    qi_to_queries = {}
    for qi, (ti, is_main, sub_id) in enumerate(task_mapping):
        task = tasks[ti]
        
        ocr_query = task.get("ocr", [])
        if isinstance(ocr_query, str): ocr_query = [ocr_query]
        
        object_query = task.get("objects", [])
        if isinstance(object_query, str): object_query = [object_query]
        
        # Determine caption
        caption_query = ""
        if is_main:
            caption_query = task.get("query", "")
        else:
            caption_query = ""
            for sub in task.get("sub_queries", []):
                if sub.get("sub_id") == sub_id:
                    caption_query = sub.get("text", "")
                    break
                    
        if ocr_query or object_query or caption_query:
            qi_to_queries[qi] = (ocr_query, object_query, caption_query)
            
    query_caption_to_idx = {}
    meta_caption_to_idx = {}
    sim_matrix = None

    if use_st:
        unique_caption_queries = list(set(q[2] for q in qi_to_queries.values() if q[2]))
        if unique_caption_queries:
            try:
                q_embs = st_model.encode(unique_caption_queries, convert_to_tensor=True, show_progress_bar=True)
            except Exception as e:
                print(f"[retrieve] GPU encode failed ({e}), falling back to CPU...", flush=True)
                st_model = st_model.to('cpu')
                q_embs = st_model.encode(unique_caption_queries, convert_to_tensor=True, show_progress_bar=True)
            query_caption_to_idx = {q: i for i, q in enumerate(unique_caption_queries)}
            
        unique_meta_captions = list(set(meta.get("caption", "") for meta_dict in metadata.values() for meta in meta_dict.values() if meta.get("caption")))
        if unique_meta_captions:
            print(f"[retrieve] Encoding {len(unique_meta_captions)} unique image captions with ST...", flush=True)
            try:
                m_embs = st_model.encode(unique_meta_captions, batch_size=256, convert_to_tensor=True, show_progress_bar=True)
            except Exception as e:
                print(f"[retrieve] GPU batch encode failed ({e}), falling back to CPU...", flush=True)
                st_model = st_model.to('cpu')
                m_embs = st_model.encode(unique_meta_captions, batch_size=256, convert_to_tensor=True, show_progress_bar=True)
            meta_caption_to_idx = {c: i for i, c in enumerate(unique_meta_captions)}

        # GPU MATRIX MULTIPLICATION (Chunked to save VRAM and align with batching concept)
        if unique_caption_queries and unique_meta_captions:
            sim_matrix = np.zeros((q_embs.shape[0], m_embs.shape[0]), dtype=np.float32)
            CH_sim = 10000
            for s in range(0, m_embs.shape[0], CH_sim):
                e = min(s + CH_sim, m_embs.shape[0])
                m_chunk = m_embs[s:e]
                sim_matrix[:, s:e] = util.cos_sim(q_embs, m_chunk).cpu().numpy()
        
    global _G_vids, _G_ts, _G_metadata, _G_qi_to_queries, _G_sim_matrix, _G_query_caption_to_idx, _G_meta_caption_to_idx, _G_use_st
    _G_vids = vids
    _G_ts = ts
    _G_metadata = metadata
    _G_qi_to_queries = qi_to_queries
    _G_sim_matrix = sim_matrix
    _G_query_caption_to_idx = query_caption_to_idx
    _G_meta_caption_to_idx = meta_caption_to_idx
    _G_use_st = use_st

    import concurrent.futures
    import multiprocessing
    from tqdm import tqdm

    print("[retrieve] Computing metadata scores across all frames...", flush=True)
    CHUNK_SIZE = 500
    chunks = [(s, min(s + CHUNK_SIZE, N)) for s in range(0, N, CHUNK_SIZE)]
    
    ctx = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(max_workers=16, mp_context=ctx) as executor:
        futures = [executor.submit(_worker_process_chunk, s, e) for s, e in chunks]
        with tqdm(total=N, desc="Metadata Scoring") as pbar:
            for future in concurrent.futures.as_completed(futures):
                frames_processed, updates = future.result()
                for qi, r, bonus, b_ocr, b_od, b_cap in updates:
                    B[qi, r] = bonus
                    if return_components:
                        B_ocr[qi, r] = b_ocr
                        B_od[qi, r] = b_od
                        B_cap[qi, r] = b_cap
                pbar.update(frames_processed)
    
    if use_st:
        del st_model
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
                    
    print(f"[retrieve] Precomputed metadata bonus for {N} frames in {time.time()-t0:.1f}s", flush=True)
    if return_components:
        return B, B_ocr, B_od, B_cap
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
    
    for ti, task in enumerate(tasks):
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
