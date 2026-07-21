import os
import torch
from PIL import Image

def apply_od_reranking(tasks, preds):
    use_od = os.environ.get("USE_OD_RERANKING", "false").lower() == "true"
    if not use_od:
        return preds
        
    od_weight = float(os.environ.get("OD_RERANKING_WEIGHT", "0.5"))
    print(f"[postprocess] Applying Object Detection Reranking (weight={od_weight})...")
    
    from transformers import pipeline
    from pipeline.retrieval.segmentation.model import get_segmenters
    from pipeline.utils.video import get_keyframe_path
    
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    od_pipe = pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32", device=dev)
    _, obj_seg = get_segmenters(scene_engine="none", object_engine="regex")
    
    task_map = {str(t["task_id"]): t["description"] for t in tasks}
    
    for ti, pred in enumerate(preds):
        task_id = str(pred["task_id"])
        desc = task_map.get(task_id, "")
        results = pred.get("results", [])
        if not results:
            continue
            
        objects = obj_seg.segment(desc)
        if not objects:
            objects = [desc]
        
        images_to_process = []
        image_paths = []
        c_image_indices = [] 
        
        for c in results:
            seq_ms = c.get("sequence_ms", [c.get("frame_ms")])
            c_indices = []
            for ms in seq_ms:
                vid = c["video_id"]
                path = get_keyframe_path(vid, ms)
                if path and os.path.exists(path):
                    c_indices.append(len(images_to_process))
                    images_to_process.append(Image.open(path).convert("RGB"))
                    image_paths.append(path)
            c_image_indices.append(c_indices)
            
        if not images_to_process:
            continue
            
        batch_size = 32
        od_results = []
        
        for i in range(0, len(images_to_process), batch_size):
            batch_imgs = images_to_process[i:i+batch_size]
            out = od_pipe([{"image": img, "candidate_labels": objects} for img in batch_imgs])
            if isinstance(out, dict):
                out = [out]
            elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict) and "score" in out[0]:
                out = [out]
            od_results.extend(out)
            
        for c_idx, c in enumerate(results):
            indices = c_image_indices[c_idx]
            if not indices:
                continue
            
            c_od_score = 0
            for idx in indices:
                frame_out = od_results[idx]
                if not frame_out:
                    continue
                    
                label_scores = {}
                for det in frame_out:
                    lbl = det["label"]
                    sc = det["score"]
                    if lbl not in label_scores or sc > label_scores[lbl]:
                        label_scores[lbl] = sc
                
                frame_od_score = sum(label_scores.values())
                c_od_score += frame_od_score
                
            c["od_score"] = c_od_score
            c["sim"] = c.get("sim", 0) + od_weight * c_od_score
            
        pred["results"].sort(key=lambda x: x.get("sim", 0), reverse=True)
        for i, c in enumerate(pred["results"]):
            c["rank"] = i + 1
            
    return preds
