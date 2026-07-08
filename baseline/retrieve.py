"""Stage 3 — text->keyframe retrieval -> submission.json.

Loads all per-video shards into one keyframe index, embeds each task description
with CLIP text encoder, finds the most similar keyframes (chunked top-k on GPU),
dedups to distinct videos, and emits up to 10 (video_id, frame_ms) predictions
per task — one frame (the matched keyframe's timestamp) per distinct video.
"""
from __future__ import annotations
import argparse, glob, json, os, time
from pathlib import Path
import numpy as np
import collections
import re

def parse_queries(tasks, max_window=15):
    all_queries = []
    task_mapping = [] # (ti, is_main, sub_id)
    
    for ti, task in enumerate(tasks):
        desc = task["description"]
        all_queries.append(desc)
        task_mapping.append((ti, True, 0)) # Main query
        
        words = desc.split()
        if len(words) > max_window:
            S = max(1, max_window // 2) # stride
            sub_idx = 1
            for i in range(0, len(words), S):
                if i + max_window >= len(words):
                    chunk = words[-max_window:]
                    if len(chunk) == max_window:
                        sub_q = " ".join(chunk)
                        all_queries.append(sub_q)
                        task_mapping.append((ti, False, sub_idx))
                        sub_idx += 1
                    break
                
                chunk = words[i : i + max_window]
                if len(chunk) == max_window:
                    sub_q = " ".join(chunk)
                    all_queries.append(sub_q)
                    task_mapping.append((ti, False, sub_idx))
                    sub_idx += 1
    return all_queries, task_mapping


def load_index(shard_dir):
    embs, vids, ts = [], [], []
    files = sorted(glob.glob(os.path.join(shard_dir, "*.npz")))
    for f in files:
        d = np.load(f)
        e = d["emb"]
        if e.shape[0] == 0:
            continue
        embs.append(e)
        vid = Path(f).stem
        vids.extend([vid] * e.shape[0])
        ts.append(d["ts_ms"])
    if not embs:
        raise SystemExit(f"no shards in {shard_dir}")
    emb = np.concatenate(embs, 0)            # [N, D] fp16
    ts = np.concatenate(ts, 0).astype(np.int32)
    vids = np.array(vids)
    print(f"[index] {emb.shape[0]} keyframes from {len(files)} videos, dim={emb.shape[1]}", flush=True)
    return emb, vids, ts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", required=True)
    p.add_argument("--tasks", required=True, help="a round's task file, e.g. public_round_tasks.jsonl")
    p.add_argument("--out", required=True, help="submission.json path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--precision", default=None)
    p.add_argument("--top-videos", type=int, default=10)
    p.add_argument("--cand-keyframes", type=int, default=2000)
    p.add_argument("--max-window", type=int, default=15)
    args = p.parse_args()

    import torch
    emb, vids, ts = load_index(args.shards)
    
    print("[retrieve] Smoothing features temporally...", flush=True)
    emb_prev = np.roll(emb, 1, axis=0)
    emb_next = np.roll(emb, -1, axis=0)
    
    same_prev = (vids == np.roll(vids, 1))
    same_next = (vids == np.roll(vids, -1))
    
    smoothed = emb.astype(np.float32).copy()
    emb_prev = emb_prev.astype(np.float32)
    emb_next = emb_next.astype(np.float32)
    
    mask_mid = same_prev & same_next
    smoothed[mask_mid] = 0.6 * smoothed[mask_mid] + 0.2 * emb_prev[mask_mid] + 0.2 * emb_next[mask_mid]
    
    mask_left = same_next & ~same_prev
    smoothed[mask_left] = 0.8 * smoothed[mask_left] + 0.2 * emb_next[mask_left]
    
    mask_right = same_prev & ~same_next
    smoothed[mask_right] = 0.8 * smoothed[mask_right] + 0.2 * emb_prev[mask_right]
    
    norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
    emb = np.where(norms > 1e-9, smoothed / norms, smoothed).astype(np.float16)

    tasks = [json.loads(l) for l in open(args.tasks)]
    print(f"[tasks] {len(tasks)}", flush=True)

    from clip_model import ClipModel
    clip = ClipModel(args.model, args.pretrained, device=args.device, precision=args.precision)
    all_queries, task_mapping = parse_queries(tasks, args.max_window)
    Q = clip.encode_texts(all_queries)      # [T_all, D] fp32

    dev = args.device
    idx = torch.from_numpy(emb).to(dev).float()                    # Ép sang fp32 [N, D]
    Qt = torch.from_numpy(Q).to(dev).float()                       # Ép sang fp32 [T_all, D]
    T_all, N = Qt.shape[0], idx.shape[0]
    K = min(args.cand_keyframes, N)

    # chunked top-K keyframes per query (Giảm xuống 10k để chống OOM khi số lượng Sub-query lớn)
    CH = 10_000
    top_val = torch.full((T_all, K), float("-inf"), device=dev, dtype=torch.float32)
    top_idx = torch.zeros((T_all, K), device=dev, dtype=torch.long)
    t0 = time.time()
    for s in range(0, N, CH):
        e = min(s + CH, N)
        sims = Qt @ idx[s:e].T                                     # [T_all, chunk]
        
        # Dual Softmax (Mean Centering)
        sims = sims - sims.mean(dim=0, keepdim=True)
        
        cat_v = torch.cat([top_val, sims], 1)
        cat_i = torch.cat([top_idx, torch.arange(s, e, device=dev).expand(T_all, e - s)], 1)
        top_val, sel = cat_v.topk(K, dim=1)
        top_idx = torch.gather(cat_i, 1, sel)
    print(f"[retrieve] scored {N} keyframes x {T_all} total queries in {time.time()-t0:.0f}s", flush=True)

    top_idx = top_idx.cpu().numpy()
    top_val = top_val.float().cpu().numpy()

    task_results = collections.defaultdict(lambda: {'main': None, 'subs': []})
    for qi, (ti, is_main, sub_id) in enumerate(task_mapping):
        if is_main:
            task_results[ti]['main'] = (top_idx[qi], top_val[qi])
        else:
            task_results[ti]['subs'].append((sub_id, top_idx[qi], top_val[qi]))

    preds = []
    for ti, task in enumerate(tasks):
        main_rows, main_sims = task_results[ti]['main']
        subs = task_results[ti]['subs']
        
        # 1. Thu thập điểm của Câu Chính (Base Score)
        v_main_scores = collections.defaultdict(list)
        v_main_centers = {}
        for r, sim in zip(main_rows, main_sims):
            v = str(vids[r])
            v_main_scores[v].append(float(sim))
            if v not in v_main_centers:
                v_main_centers[v] = int(ts[r])
                
        # 2. Thu thập điểm của các Câu Phụ
        v_sub_max = collections.defaultdict(lambda: collections.defaultdict(float))
        v_sub_center = collections.defaultdict(dict)
        for sub_id, rows, sims in subs:
            for r, sim in zip(rows, sims):
                v = str(vids[r])
                if sub_id not in v_sub_max[v] or sim > v_sub_max[v][sub_id]:
                    v_sub_max[v][sub_id] = float(sim)
                    v_sub_center[v][sub_id] = int(ts[r])
                    
        # 3. Tính điểm tổng hợp (Base + Sub Booster + Causal Bonus)
        ranked_vids = []
        for v, m_scores in v_main_scores.items():
            main_sc = m_scores[0]
            main_bonus = sum(m_scores[1:4]) / len(m_scores[1:4]) * 0.1 if len(m_scores) > 1 else 0.0
            base_score = main_sc + main_bonus
            
            sub_bonus = 0.0
            causal_bonus = 0.0
            if v in v_sub_max:
                subs_for_v = v_sub_max[v]
                # Các câu phụ (sub queries) đóng vai trò booster với trọng số nghịch biến:
                # Càng về sau (sub_id càng lớn), trọng số càng nhỏ (0.2 / sub_id).
                for sid, sim in subs_for_v.items():
                    decay_weight = 0.2 / sid
                    sub_bonus += sim * decay_weight
                
                # Check causal order
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
            ranked_vids.append((final_sc, v, v_main_centers[v]))
            
        ranked_vids.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for rank, (score, v, center) in enumerate(ranked_vids[:args.top_videos], 1):
            results.append({
                "rank": rank, "video_id": v,
                "frame_ms": center,
            })
        preds.append({"task_id": task["task_id"], "results": results})

    sub = {"predictions": preds}
    json.dump(sub, open(args.out, "w"))
    print(f"[done] wrote {args.out} ({len(preds)} tasks)", flush=True)


if __name__ == "__main__":
    main()
