"""
Purpose: Computes similarity scores between text queries and video frames using
Dual Softmax (Mean Centering) and aggregates base scores with Causal Bonuses.
"""
import time
import collections
import torch
import numpy as np
from tqdm import tqdm

def compute_similarity(Q_list, idx_list, T_all, N, K, dev):
    CH = 10000
    t0 = time.time()
    num_models = len(Q_list)
    
    if K >= N:
        # [MEMORY OPTIMIZATION] Bypass GPU topK for full-frame scoring to avoid OOM
        all_sims = torch.zeros((T_all, N), device='cpu', dtype=torch.float32)
        for s in range(0, N, CH):
            e = min(s + CH, N)
            ensembled_sims = torch.zeros((T_all, e - s), device=dev, dtype=torch.float32)
            
            for m in range(num_models):
                Q = Q_list[m]
                idx_chunk = torch.from_numpy(idx_list[m][s:e]).to(dev).float()
                ensembled_sims += Q @ idx_chunk.T
                
            ensembled_sims /= num_models
            ensembled_sims = 0.8 * ensembled_sims
            all_sims[:, s:e] = ensembled_sims.cpu()
            
        top_val = all_sims.numpy()
        top_idx = np.tile(np.arange(N), (T_all, 1))
        print(f"[retrieve] scored {N} keyframes x {T_all} total queries using {num_models} models in {time.time()-t0:.0f}s", flush=True)
        return top_idx, top_val

    top_val = torch.full((T_all, K), float("-inf"), device=dev, dtype=torch.float32)
    top_idx = torch.zeros((T_all, K), device=dev, dtype=torch.long)
    
    num_models = len(Q_list)
    
    for s in range(0, N, CH):
        e = min(s + CH, N)
        
        # --- RAW COSINE ---
        ensembled_sims = torch.zeros((T_all, e - s), device=dev, dtype=torch.float32)
        
        for m in range(num_models):
            Q = Q_list[m]
            idx_chunk = torch.from_numpy(idx_list[m][s:e]).to(dev).float()
            
            sims = Q @ idx_chunk.T
            ensembled_sims += sims
            
        ensembled_sims /= num_models
        ensembled_sims = 0.8 * ensembled_sims
        
        cat_v = torch.cat([top_val, ensembled_sims], 1)
        cat_i = torch.cat([top_idx, torch.arange(s, e, device=dev).expand(T_all, e - s)], 1)
        top_val, sel = cat_v.topk(K, dim=1)
        top_idx = torch.gather(cat_i, 1, sel)
        
    print(f"[retrieve] scored {N} keyframes x {T_all} total queries using {num_models} models in {time.time()-t0:.0f}s", flush=True)
    return top_idx.cpu().numpy(), top_val.float().cpu().numpy()

def aggregate_scores(task_mapping, top_idx, top_val, tasks, vids, ts, emb, metadata, use_sequential=False):
    # task_results[ti][sent_idx] = [(seg_idx, rows, sims), ...]
    task_results = collections.defaultdict(lambda: collections.defaultdict(list))
    for qi, (ti, sent_idx, seg_idx) in enumerate(task_mapping):
        task_results[ti][sent_idx].append((seg_idx, top_idx[qi], top_val[qi]))

    all_candidates = []
    
    # Precompute global video frame grouping to avoid O(N) operations per task
    global_v_frames = collections.defaultdict(list)
    for r in range(len(vids)):
        v = str(vids[r])
        center = int(ts[r])
        global_v_frames[v].append((center, r))
        
    for v in global_v_frames:
        global_v_frames[v].sort(key=lambda x: x[0])
        
    import os
    agg_mode = os.environ.get("AGG_MODE", "harmonic").lower()
    dp_mode = os.environ.get("DP_MODE", "add").lower()
    clip_to_zero = os.environ.get("CLIP_TO_ZERO", "false").lower() == "true"
    
    for ti, task in enumerate(tqdm(tasks, desc="[scorer] Aggregating tasks")):
        sent_scores = {}
        N_frames = len(vids)
        for sent_idx, segments in task_results[ti].items():
            
            S_objects = np.zeros((len(segments), N_frames), dtype=np.float32)
            for i, (seg_idx, rows, sims) in enumerate(segments):
                S_objects[i, rows] = sims
                
            if agg_mode == "min":
                scene_sims = np.min(S_objects, axis=0)
            elif agg_mode == "mean":
                scene_sims = np.mean(S_objects, axis=0)
            else:
                # Harmonic Mean (requires positive numbers, so shift from [-1, 1] to [0, 2])
                shifted_S = S_objects + 1.0
                
                import os
                main_weight = float(os.environ.get("MAIN_QUERY_WEIGHT", "2.0"))
                
                num_segs = len(segments)
                W = np.ones((num_segs, 1), dtype=np.float32)
                if num_segs > 1:
                    W[0, 0] = main_weight
                    
                sum_W = np.sum(W)
                h_mean = sum_W / np.sum(W / (shifted_S + 1e-6), axis=0)
                scene_sims = h_mean - 1.0
                
            sent_scores[sent_idx] = (np.arange(N_frames), scene_sims)
            
        main_rows, main_sims = sent_scores[0]
        
        subs = []
        for sent_idx in sorted(sent_scores.keys()):
            if sent_idx > 0:
                subs.append((sent_idx, sent_scores[sent_idx][0], sent_scores[sent_idx][1]))
        
        candidates = []
        
        if use_sequential and subs:
            # === DP SEQUENTIAL MATCHING ON SENTENCES ===
            all_chunks = [(main_rows, main_sims)] + [(s[1], s[2]) for s in subs]
            M = len(all_chunks)
            
            for v, frames in global_v_frames.items():
                F = len(frames)
                if F == 0:
                    continue
                    
                S = np.zeros((M, F), dtype=np.float32)
                
                for m in range(M):
                    rows, sims = all_chunks[m]
                    # 'rows' is guaranteed to be np.arange(N), so sims[r] is the score for frame r
                    for f, (center, r) in enumerate(frames):
                        S[m, f] = sims[r]
                        
                if clip_to_zero:
                    S = np.clip(S, 0.0, 1.0)
                            
                # Convert similarities to log probabilities? NO!
                # Normalizing per video means a 1-frame video always gets probability 1.0 (log_prob = 0),
                # which will always beat a 100-frame video. We must use the raw similarities directly!
                
                if dp_mode == "prod":
                    DP = np.zeros((M, F), dtype=np.float32)
                else:
                    DP = np.full((M, F), -np.inf, dtype=np.float32)
                backptr = np.zeros((M, F), dtype=int)
                
                DP[0, :] = S[0, :]
                
                import os
                from collections import deque
                max_gap_ms = int(os.environ.get("MAX_SEQ_GAP_MS", "60000"))
                discount_factor = float(os.environ.get("DISCOUNT_FACTOR", "0.9"))
                
                for m in range(1, M):
                    q = deque()
                    for f in range(F):
                        if f > 0:
                            new_k = f - 1
                            val = DP[m-1, new_k]
                            while q and DP[m-1, q[-1]] <= val:
                                q.pop()
                            q.append(new_k)
                            
                        while q and (frames[f][0] - frames[q[0]][0] > max_gap_ms):
                            q.popleft()
                            
                        if q:
                            best_k = q[0]
                            best_val = DP[m-1, best_k]
                        else:
                            best_k = 0
                            best_val = -np.inf if dp_mode != "prod" else 0.0
                            
                        if dp_mode == "prod":
                            DP[m, f] = best_val * S[m, f]
                        else:
                            DP[m, f] = best_val + S[m, f] * (discount_factor ** m)
                        backptr[m, f] = best_k
                        
                final_scores = DP[M-1, :]
                sorted_f_indices = np.argsort(final_scores)[::-1]
                
                added_intervals = []
                frame_sims_vid = np.mean(S, axis=0)
                all_frames = [{"frame_ms": frames[i][0], "sim": float(frame_sims_vid[i])} for i in range(F)]
                
                for f_idx in sorted_f_indices:
                    final_sc = float(final_scores[f_idx])
                    if final_sc <= -1e5:
                        continue
                        
                    # Backtrack
                    curr_f = f_idx
                    sequence_f = [f_idx]
                    for m in range(M-1, 0, -1):
                        curr_f = backptr[m, curr_f]
                        sequence_f.append(curr_f)
                    
                    sequence_f.reverse()
                    sequence_ms = [frames[f][0] for f in sequence_f]
                    
                    c_start = sequence_ms[0]
                    c_end = sequence_ms[-1]
                    if c_start > c_end:
                        c_start, c_end = c_end, c_start
                        
                    is_overlap = False
                    for acc_start, acc_end in added_intervals:
                        if max(c_start, acc_start) <= min(c_end, acc_end):
                            is_overlap = True
                            break
                            
                    if not is_overlap:
                        added_intervals.append([c_start, c_end])
                        
                        middle_ms = int((c_start + c_end) / 2)
                        
                        interval_f_indices = [i for i in range(F) if c_start <= frames[i][0] <= c_end]
                        if interval_f_indices:
                            best_f_in_interval = max(interval_f_indices, key=lambda i: frame_sims_vid[i])
                            best_seq_ms = frames[best_f_in_interval][0]
                        else:
                            best_seq_ms = middle_ms

                        pos_mode = os.environ.get("POSITION_MODE", "middle").lower()
                        if pos_mode == "first":
                            chosen_frame_ms = c_start
                        elif pos_mode == "second":
                            chosen_frame_ms = sequence_ms[1] if len(sequence_ms) > 1 else sequence_ms[0]
                        elif pos_mode == "best":
                            chosen_frame_ms = best_seq_ms
                        else:
                            chosen_frame_ms = middle_ms
                            
                        closest_f = min(range(F), key=lambda x: abs(frames[x][0] - chosen_frame_ms))
                        best_r = frames[closest_f][1]
                        
                        candidates.append({
                            "video_id": v,
                            "frame_ms": chosen_frame_ms,
                            "sim": final_sc,
                            "feat": emb[best_r],
                            "sequence_ms": sequence_ms,
                            "best_score_ms": best_seq_ms,
                            "best_score_sec": best_seq_ms / 1000.0,
                            "all_frames": all_frames
                        })
                        
                        max_preds_vid = int(os.environ.get("MAX_PREDS_PER_VIDEO", "10"))
                        if len(added_intervals) >= max_preds_vid:
                            break
        else:
            # === MEAN FUSION AGGREGATION (NO CAUSAL) ===
            # We have sent_scores: sent_idx -> (rows, sims)
            # We just want to average the similarities of all sentences for each frame.
            frame_sims = collections.defaultdict(list)
            for sent_idx, (rows, sims) in sent_scores.items():
                for r, sim in zip(rows, sims):
                    frame_sims[r].append(sim)
                    
            num_sents = len(sent_scores)
            
            v_best_score = collections.defaultdict(lambda: -float('inf'))
            v_best_center = {}
            v_best_row = {}
            v_all_frames = collections.defaultdict(list)
            
            for r, sims in frame_sims.items():
                # Average score across all sentences
                avg_sim = sum(sims) / num_sents
                v = str(vids[r])
                center = int(ts[r])
                
                v_all_frames[v].append({"frame_ms": center, "sim": float(avg_sim)})
                
                if avg_sim > v_best_score[v]:
                    v_best_score[v] = float(avg_sim)
                    v_best_center[v] = center
                    v_best_row[v] = r
                    
            for v, final_sc in v_best_score.items():
                best_ms = v_best_center[v]
                candidates.append({
                    "video_id": v,
                    "frame_ms": best_ms,
                    "sim": final_sc,
                    "feat": emb[v_best_row[v]],
                    "best_score_ms": best_ms,
                    "best_score_sec": best_ms / 1000.0,
                    "all_frames": v_all_frames[v]
                })
            
        candidates.sort(key=lambda x: x["sim"], reverse=True)
        all_candidates.append((task, candidates))
        
    return all_candidates
