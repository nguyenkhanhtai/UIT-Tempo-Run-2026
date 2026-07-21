import os
import torch
from PIL import Image

def apply_od_reranking(tasks, preds):
    use_od = os.environ.get("USE_OD_RERANKING", "false").lower() == "true"
    if not use_od:
        return preds
        
    od_weight = float(os.environ.get("OD_RERANKING_WEIGHT", "0.5"))
    dp_mode = os.environ.get("DP_MODE", "sum").lower()
    print(f"[postprocess] Applying Object Detection Reranking (weight={od_weight}, dp_mode={dp_mode})...")
    
    from transformers import pipeline
    from pipeline.utils.visualize import get_keyframe_path
    
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    od_pipe = pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32", device=dev)
    
    # Global tracking across all tasks
    all_inputs = []
    task_metadata = []
    
    for ti, pred in enumerate(preds):
        task = tasks[ti]
        results = pred.get("results", [])
        if not results:
            task_metadata.append(None)
            continue
            
        segments = task.get("segments", {})
        
        # Build mapping from scene index to list of object strings
        scene_objects_map = {}
        for sent_idx_str, items in segments.items():
            sent_idx = int(sent_idx_str)
            objs = [item["text"] for item in items if item["type"] == "object"]
            if not objs:
                # Fallback to scene text if no specific objects extracted
                objs = [items[0]["text"]] if items else [task["description"]]
            
            # Truncate each object text to maximum 5 words to prevent OWL-ViT tensor length error (max 16 tokens)
            objs = [" ".join(o.split()[:5]) for o in objs]
            scene_objects_map[sent_idx] = objs
            
        c_image_indices = [] 
        
        for c in results:
            seq_ms = c.get("sequence_ms", [c.get("frame_ms")])
            c_indices = []
            for m, ms in enumerate(seq_ms):
                vid = c["video_id"]
                kf_dir = os.environ.get("KF_DIR", "keyframes/fps/1")
                path = get_keyframe_path(vid, ms, kf_dir=kf_dir)
                
                # Get objects for this scene
                objects = scene_objects_map.get(m, [])
                if not objects:
                    # Fallback to full description if mapping is missing
                    desc_objs = [" ".join(task["description"].split()[:5])]
                    objects = desc_objs
                    
                if path and os.path.exists(path):
                    c_indices.append(len(all_inputs))
                    all_inputs.append({"image": Image.open(path).convert("RGB"), "candidate_labels": objects})
                else:
                    c_indices.append(-1)
            c_image_indices.append(c_indices)
            
        task_metadata.append({
            "results": results,
            "c_image_indices": c_image_indices
        })
        
    od_results_global = []
    if all_inputs:
        batch_size = 32
        from tqdm import tqdm
        from torch.utils.data import Dataset
        
        class ImageObjectDataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
                
        dataset = ImageObjectDataset(all_inputs)
        
        for out in tqdm(od_pipe(dataset, batch_size=batch_size), total=len(dataset), desc="[od_reranker] Processing global batch"):
            # out is the prediction for a SINGLE image, which is a list of dicts (or a dict if only 1 object)
            if isinstance(out, dict):
                out = [out]
            elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
                # Just in case pipeline returns list of lists for a single image, though unlikely
                out = out[0]
            od_results_global.append(out)
            
    for ti, pred in enumerate(preds):
        meta = task_metadata[ti]
        if not meta:
            continue
            
        results = meta["results"]
        c_image_indices = meta["c_image_indices"]
        
        for c_idx, c in enumerate(results):
            indices = c_image_indices[c_idx]
            if not indices:
                continue
            
            frame_scores = []
            for idx in indices:
                if idx == -1:
                    frame_scores.append(0.0)
                    continue
                    
                frame_out = od_results_global[idx]
                if not frame_out:
                    frame_scores.append(0.0)
                    continue
                    
                label_scores = {}
                for det in frame_out:
                    lbl = det["label"]
                    sc = det["score"]
                    if lbl not in label_scores or sc > label_scores[lbl]:
                        label_scores[lbl] = sc
                
                frame_od_score = sum(label_scores.values())
                frame_scores.append(frame_od_score)
                
            # Aggregate per DP_MODE
            discount_factor = float(os.environ.get("DISCOUNT_FACTOR", "0.9"))
            if dp_mode == "prod":
                c_od_score = 1.0
                for sc in frame_scores:
                    c_od_score *= max(0.0, min(1.0, sc))
            else:
                c_od_score = sum(sc * (discount_factor ** m) for m, sc in enumerate(frame_scores))
                
            c["od_score"] = c_od_score
            # Store original rank (it's currently sorted by original sim)
            c["orig_rank"] = c.get("rank", c_idx + 1)
            
        # Sort by od_score to get od_rank
        pred["results"].sort(key=lambda x: x.get("od_score", 0), reverse=True)
        for i, c in enumerate(pred["results"]):
            c["od_rank"] = i + 1
            
        # Compute RRF score
        k = 60
        for c in pred["results"]:
            orig_rank = c.get("orig_rank", 1000)
            od_rank = c.get("od_rank", 1000)
            rrf_score = 1.0 / (k + orig_rank) + od_weight * (1.0 / (k + od_rank))
            c["sim"] = rrf_score
            
        # Sort by final RRF score
        pred["results"].sort(key=lambda x: x.get("sim", 0), reverse=True)
        for i, c in enumerate(pred["results"]):
            c["rank"] = i + 1
            
    return preds
