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
CAPTION_SCORING_METHOD = "embedding" # "embedding" or "ngram"

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

def get_ngrams(text, n):
    words = text.lower().split()
    if len(words) < n:
        return set()
    return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))

def get_caption_similarity(query, caption):
    if not query or not caption:
        return 0.0
        
    q_uni = get_ngrams(query, 1)
    c_uni = get_ngrams(caption, 1)
    uni_overlap = len(q_uni.intersection(c_uni)) / len(q_uni) if q_uni else 0.0
    
    q_bi = get_ngrams(query, 2)
    c_bi = get_ngrams(caption, 2)
    bi_overlap = len(q_bi.intersection(c_bi)) / len(q_bi) if q_bi else 0.0
    
    if len(q_bi) == 0:
        return uni_overlap
        
    return 0.4 * uni_overlap + 0.6 * bi_overlap

def precompute_metadata_bonus(tasks, task_mapping, vids, ts, metadata, return_components=False):
    import numpy as np
    from .parser import extract_ocr_queries, extract_object_queries
    
    T_all = len(task_mapping)
    N = len(vids)
    print(N)
    B = np.zeros((T_all, N), dtype=np.float32)
    if return_components:
        B_ocr = np.zeros((T_all, N), dtype=np.float32)
        B_od = np.zeros((T_all, N), dtype=np.float32)
        B_cap = np.zeros((T_all, N), dtype=np.float32)
    
    t0 = time.time()
    
    use_st = False
    if USE_CAPTIONING and CAPTION_SCORING_METHOD == "embedding":
        try:
            from sentence_transformers import SentenceTransformer, util
            print("[retrieve] Loading sentence-transformers model for caption similarity...", flush=True)
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            st_model = SentenceTransformer('all-MiniLM-L6-v2', device=device, model_kwargs={"attn_implementation": "eager"})
            use_st = True
        except ImportError:
            print("[retrieve] sentence-transformers not found, falling back to ngram.", flush=True)
            use_st = False
    elif USE_CAPTIONING and CAPTION_SCORING_METHOD == "ngram":
        use_st = False
    
    qi_to_queries = {}
    for qi, (ti, is_main, sub_id) in enumerate(task_mapping):
        if is_main:
            desc = tasks[ti]["description"]
            ocr_q = extract_ocr_queries(desc) if USE_OCR else []
            obj_q = extract_object_queries(desc) if USE_OD else []
            caption_q = desc.lower() if USE_CAPTIONING else "" # Full query text for caption matching
            
            # Always track main queries because caption matching applies to all
            qi_to_queries[qi] = (ocr_q, obj_q, caption_q)
                
    if not qi_to_queries:
        if return_components: return B, B, B, B
        return B
        
    cap_sim_matrix_cache = {}
    
    if use_st:
        unique_caption_queries = list(set(q[2] for q in qi_to_queries.values() if q[2]))
        if unique_caption_queries:
            try:
                q_embs = st_model.encode(unique_caption_queries, convert_to_tensor=True, show_progress_bar=True)
            except Exception as e:
                print(f"[retrieve] GPU encode failed ({e}), falling back to CPU...", flush=True)
                st_model = st_model.to('cpu')
                q_embs = st_model.encode(unique_caption_queries, convert_to_tensor=True, show_progress_bar=True)
            query_emb_dict = {q: q_embs[i] for i, q in enumerate(unique_caption_queries)}
            
        unique_meta_captions = list(set(meta.get("caption", "") for meta_dict in metadata.values() for meta in meta_dict.values() if meta.get("caption")))
        if unique_meta_captions:
            print(f"[retrieve] Encoding {len(unique_meta_captions)} unique image captions with ST...", flush=True)
            try:
                m_embs = st_model.encode(unique_meta_captions, batch_size=1024, convert_to_tensor=True, show_progress_bar=True)
            except Exception as e:
                print(f"[retrieve] GPU batch encode failed ({e}), falling back to CPU...", flush=True)
                st_model = st_model.to('cpu')
                m_embs = st_model.encode(unique_meta_captions, batch_size=1024, convert_to_tensor=True, show_progress_bar=True)
            meta_emb_dict = {c: m_embs[i] for i, c in enumerate(unique_meta_captions)}
            
            print("[retrieve] Precomputing GPU Caption Dot Products...", flush=True)
            q_emb_tensor = torch.stack([query_emb_dict[q] for q in unique_caption_queries])
            sim_matrix = util.cos_sim(q_emb_tensor, m_embs).cpu().numpy()
            
            cap_sim_matrix_cache = {qi: {} for qi in qi_to_queries}
            for qi, (_, _, caption_query) in qi_to_queries.items():
                if caption_query:
                    q_idx = unique_caption_queries.index(caption_query)
                    for m_idx, meta_cap in enumerate(unique_meta_captions):
                        sim = float(sim_matrix[q_idx, m_idx])
                        if sim > 0.4:
                            cap_sim_matrix_cache[qi][meta_cap] = 0.3 * sim

    import concurrent.futures
    import math

    def process_chunk(start_r, end_r):
        for r in range(start_r, end_r):
            v = str(vids[r])
            center = int(ts[r])
            if v in metadata and center in metadata[v]:
                meta = metadata[v][center]
                meta_ocr = meta.get("ocr", "")
                meta_objs = meta.get("objects", [])
                meta_caption = meta.get("caption", "")
                meta_words = set(w for obj in meta_objs for w in obj.lower().split())
                
                for qi, (ocr_query, object_query, caption_query) in qi_to_queries.items():
                    b_ocr, b_od, b_cap = 0.0, 0.0, 0.0
                    if ocr_query and USE_OCR:
                        for q in ocr_query:
                            sim = get_ocr_similarity(q, meta_ocr)
                            b_ocr += 0.15 * math.exp(4 * (sim - 1.0))
                    if object_query and USE_OD:
                        for qw in object_query:
                            if qw in meta_words:
                                b_od += 0.05
                    if meta_caption and USE_CAPTIONING:
                        if use_st:
                            b_cap = cap_sim_matrix_cache.get(qi, {}).get(meta_caption, 0.0)
                        else:
                            cap_sim = get_caption_similarity(caption_query, meta_caption)
                            if cap_sim > 0.4:  
                                b_cap += 0.3 * cap_sim                
                            
                    bonus = b_ocr + b_od + b_cap
                    if bonus > 0:
                        B[qi, r] = bonus
                        if return_components:
                            B_ocr[qi, r] = b_ocr
                            B_od[qi, r] = b_od
                            B_cap[qi, r] = b_cap

    num_threads = min(64, (N // 2000) + 1) # Increased threads
    chunk_size = 1000 # Smaller chunks for better distribution
    num_chunks = math.ceil(N / chunk_size)
    
    print(f"[retrieve] Computing metadata bonus using {num_threads} threads...", flush=True)
    from tqdm import tqdm
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_chunks):
            start_r = i * chunk_size
            end_r = min(start_r + chunk_size, N)
            futures.append(executor.submit(process_chunk, start_r, end_r))
            
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Scoring chunks"):
            future.result()
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
            # Standard Cosine Similarity without Dual Softmax / Mean Centering
            
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
                
        # 2. Aggregate (Base Score Only)
        candidates = []
        for v, m_scores in v_main_scores.items():
            final_sc = m_scores[0]
            
            candidates.append({
                "video_id": v,
                "frame_ms": v_main_centers[v],
                "sim": final_sc,
                "feat": emb[v_main_rows[v]]
            })
            
        candidates.sort(key=lambda x: x["sim"], reverse=True)
        all_candidates.append((task, candidates))
        
    return all_candidates
