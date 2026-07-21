import os
import torch
from PIL import Image
from tqdm import tqdm

def apply_vlm_rescoring(tasks, preds):
    use_vlm = os.environ.get("USE_VLM_RESCORING", "false").lower() == "true"
    if not use_vlm:
        return preds
        
    rescore_rank = int(os.environ.get("RESCORE_RANK", "5"))
    vlm_model_id = os.environ.get("VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
    dp_mode = os.environ.get("DP_MODE", "sum").lower()
    vlm_eval_mode = os.environ.get("VLM_EVAL_MODE", "sequence").lower()
    
    print(f"[postprocess] Applying VLM Rescoring (model={vlm_model_id}, mode={vlm_eval_mode}, rescore_rank={rescore_rank}, dp_mode={dp_mode})...")
    
    from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
    from pipeline.utils.visualize import get_keyframe_path
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = LlavaNextVideoProcessor.from_pretrained(vlm_model_id)
    if "7b" in vlm_model_id.lower() or "11b" in vlm_model_id.lower() or "13b" in vlm_model_id.lower():
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            vlm_model_id, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quantization_config
        )
    else:
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            vlm_model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
    model.eval()

    yes_token_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[-1] if processor.tokenizer else 0
    yes_token_id_lower = processor.tokenizer.encode("yes", add_special_tokens=False)[-1] if processor.tokenizer else 0
    no_token_id = processor.tokenizer.encode("No", add_special_tokens=False)[-1] if processor.tokenizer else 0
    no_token_id_lower = processor.tokenizer.encode("no", add_special_tokens=False)[-1] if processor.tokenizer else 0
    
    all_inputs = []
    task_metadata = []
    
    for ti, pred in enumerate(preds):
        task = tasks[ti]
        results = pred.get("results", [])
        if not results:
            task_metadata.append(None)
            continue
            
        segments = task.get("segments", {})
        
        scene_text_map = {}
        for sent_idx_str, items in segments.items():
            sent_idx = int(sent_idx_str)
            if items and "text" in items[0]:
                scene_text_map[sent_idx] = items[0]["text"]
            else:
                scene_text_map[sent_idx] = task["description"]
                
        c_image_indices = [] 
        
        for c_idx, c in enumerate(results):
            if c_idx >= rescore_rank:
                c_image_indices.append([])
                continue
                
            seq_ms = c.get("sequence_ms", [c.get("frame_ms")])
            c_indices = []
            
            if vlm_eval_mode in ["sequence", "judge"]:
                # Collect all paths
                vid = c["video_id"]
                kf_dir = os.environ.get("KF_DIR", "keyframes/fps/1")
                paths = []
                images = []
                for ms in seq_ms:
                    path = get_keyframe_path(vid, ms, kf_dir=kf_dir)
                    if path and os.path.exists(path):
                        paths.append(path)
                        images.append(Image.open(path).convert("RGB"))
                
                if paths:
                    c_indices.append(len(all_inputs))
                    content = [{"type": "video"}]
                    
                    if vlm_eval_mode == "judge":
                        content.append({"type": "text", "text": f"Rate the semantic similarity between this video and the following description on a scale from 0 to 100. Output strictly a single integer number and nothing else.\nDescription: '{task['description']}'"})
                    else:
                        content.append({"type": "text", "text": f"Does this video exactly match this description: '{task['description']}'? Answer strictly with a single word: Yes or No."})
                    
                    messages = [{"role": "user", "content": content}]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    video_arr = np.stack([np.array(img) for img in images])
                    all_inputs.append({"text": text, "videos": video_arr})
                else:
                    c_indices.append(-1)
            else:
                for m, ms in enumerate(seq_ms):
                    vid = c["video_id"]
                    kf_dir = os.environ.get("KF_DIR", "keyframes/fps/1")
                    path = get_keyframe_path(vid, ms, kf_dir=kf_dir)
                    
                    scene_text = scene_text_map.get(m, task["description"])
                    
                    if path and os.path.exists(path):
                        c_indices.append(len(all_inputs))
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "video"},
                                    {"type": "text", "text": f"Does the image exactly match this description: '{scene_text}'? Answer strictly with a single word: Yes or No."},
                                ],
                            }
                        ]
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        video_arr = np.stack([np.array(Image.open(path).convert("RGB"))])
                        all_inputs.append({"text": text, "videos": video_arr})
                    else:
                        c_indices.append(-1)
            c_image_indices.append(c_indices)
            
        task_metadata.append({
            "results": results,
            "c_image_indices": c_image_indices
        })
        
    vlm_results_global = []
    if all_inputs:
        import re
        batch_size = 2 if vlm_eval_mode in ["sequence", "judge"] else 4
        
        class VLMRescoreDataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
                
        def collate_fn(batch):
            texts = [item["text"] for item in batch]
            videos = [item["videos"] for item in batch]
            inputs = processor(
                text=texts,
                videos=videos,
                padding=True,
                return_tensors="pt",
            )
            return inputs
            
        dataset = VLMRescoreDataset(all_inputs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="[vlm_rescorer] Processing global batch"):
                batch = {k: v.to(dev) for k, v in batch.items()}
                
                if vlm_eval_mode == "judge":
                    outputs = model.generate(**batch, max_new_tokens=5)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(batch["input_ids"], outputs)
                    ]
                    output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    
                    for text in output_texts:
                        numbers = re.findall(r'\d+', text)
                        score = float(numbers[0]) if numbers else 0.0
                        # Normalize to 0-1 scale for consistency, though not strictly needed since we just sort
                        vlm_results_global.append(score / 100.0)
                else:
                    outputs = model(**batch)
                    first_token_logits = outputs.logits[:, -1, :]
                    
                    for b_idx in range(len(batch["input_ids"])):
                        logits = first_token_logits[b_idx]
                        
                        yes_logit = max(logits[yes_token_id].item(), logits[yes_token_id_lower].item())
                        no_logit = max(logits[no_token_id].item(), logits[no_token_id_lower].item())
                        
                        if vlm_eval_mode == "sequence":
                            # Raw logit difference provides a highly granular, continuous score for sorting
                            score = float(yes_logit - no_logit)
                        else:
                            # Softmax with temperature to prevent 0/1 collapse but keep it in [0,1] for DP formula
                            score = torch.softmax(torch.tensor([yes_logit, no_logit]) / 3.0, dim=0)[0].item()
                            
                        vlm_results_global.append(score)
            
    for ti, pred in enumerate(preds):
        meta = task_metadata[ti]
        if not meta:
            continue
            
        results = meta["results"]
        c_image_indices = meta["c_image_indices"]
        
        for c_idx, c in enumerate(results):
            if c_idx >= rescore_rank:
                c["vlm_score"] = -1.0 
                continue
                
            indices = c_image_indices[c_idx]
            if not indices:
                c["vlm_score"] = 0.0
                continue
            
            frame_scores = []
            for idx in indices:
                if idx == -1:
                    frame_scores.append(0.0)
                    continue
                    
                score = vlm_results_global[idx]
                frame_scores.append(score)
                
            if vlm_eval_mode == "sequence":
                # sequence mode has exactly one score representing the whole sequence
                c_vlm_score = frame_scores[0] if frame_scores else 0.0
            else:
                discount_factor = float(os.environ.get("DISCOUNT_FACTOR", "0.9"))
                if dp_mode == "prod":
                    c_vlm_score = 1.0
                    for sc in frame_scores:
                        c_vlm_score *= max(0.0, min(1.0, sc))
                else:
                    c_vlm_score = sum(sc * (discount_factor ** m) for m, sc in enumerate(frame_scores))
                
            c["vlm_score"] = c_vlm_score
            c["orig_rank"] = c.get("rank", c_idx + 1)
            
        top_candidates = results[:rescore_rank]
        bottom_candidates = results[rescore_rank:]
        
        top_candidates.sort(key=lambda x: x.get("vlm_score", 0), reverse=True)
        
        pred["results"] = top_candidates + bottom_candidates
        
        changes = []
        for i, c in enumerate(pred["results"]):
            new_rank = i + 1
            orig_rank = c.get("orig_rank", new_rank)
            if new_rank != orig_rank and new_rank <= rescore_rank:
                changes.append(f"  - {c['video_id']}: {orig_rank} -> {new_rank} (vlm_score={c.get('vlm_score', 0):.4f})")
            c["rank"] = new_rank
            
        if changes:
            print(f"[vlm_rescorer] Rank changes for task {ti} ('{task['description'][:50]}...'):")
            for change in changes:
                print(change)
            
    # clear memory
    del model
    del processor
    torch.cuda.empty_cache()
    
    return preds
