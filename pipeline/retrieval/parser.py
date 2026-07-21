"""
Purpose: Parses user queries into main and sub-queries based on sentences,
and extracts OCR keywords enclosed in double quotes.
"""
import re
import os

def parse_queries(tasks, use_sequential=False, scene_segmenter="regex", object_segmenter="regex"):
    all_queries = []
    task_mapping = [] # stores (task_idx, sentence_idx, segment_idx)
    
    split_query = os.environ.get("SPLIT_QUERY", "true").lower() == "true"
    
    scene_seg = None
    obj_seg = None
    if use_sequential:
        from pipeline.retrieval.segmentation.model import get_segmenters
        scene_seg, obj_seg = get_segmenters(scene_engine=scene_segmenter, object_engine=object_segmenter)
        
    # Batched Scene Segmentation
    all_scenes_list = []
    if use_sequential and scene_seg is not None:
        if hasattr(scene_seg, "segment_batch"):
            all_scenes_list = scene_seg.segment_batch([task["description"] for task in tasks])
        else:
            all_scenes_list = [scene_seg.segment(task["description"]) for task in tasks]
            
    for ti, task in enumerate(tasks):
        desc = task["description"]
        
        if not split_query:
            all_queries.append(desc)
            task_mapping.append((ti, 0, 0)) # Main query only
            continue
            
        if use_sequential and scene_seg is not None:
            scenes = all_scenes_list[ti] if ti < len(all_scenes_list) else [desc]
        else:
            scenes = [desc]
            
        if not scenes:
            scenes = [desc]
            
        for sent_idx, scene in enumerate(scenes):
            if use_sequential and obj_seg is not None:
                objects = obj_seg.segment(scene)
                if scene in objects:
                    objects.remove(scene)
                objects.insert(0, scene)
            else:
                objects = [scene]
                
            if not objects:
                objects = [scene]
                
            for seg_idx, obj in enumerate(objects):
                if seg_idx > 0:
                    # Wrap isolated keywords in a prompt to align with CLIP's training distribution
                    query_str = f"A photo that has: {obj}"
                else:
                    query_str = obj
                    
                all_queries.append(query_str)
                task_mapping.append((ti, sent_idx, seg_idx))
            
    # Build per-task segment info for logging
    task_segments = {}
    for query, (ti, sent_idx, seg_idx) in zip(all_queries, task_mapping):
        if ti not in task_segments:
            task_segments[ti] = {}
        if sent_idx not in task_segments[ti]:
            task_segments[ti][sent_idx] = []
        if seg_idx == 0:  # scene-level query
            task_segments[ti][sent_idx].insert(0, {"type": "scene", "text": query})
        else:
            task_segments[ti][sent_idx].append({"type": "object", "text": query})

    if use_sequential:
        if hasattr(scene_seg, "cleanup"):
            scene_seg.cleanup()
        if hasattr(obj_seg, "cleanup"):
            obj_seg.cleanup()
            
        del scene_seg
        del obj_seg
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_queries, task_mapping, task_segments

def extract_ocr_queries(desc):
    quotes = re.findall(r'"([^"]*)"', desc)
    return [q.lower() for q in quotes]

def extract_object_queries(desc):
    """Clean the query description into a set of basic lowercase words for object matching."""
    desc = desc.lower()
    # Remove punctuation
    desc = re.sub(r'[.!?,\'"]', ' ', desc)
    words = [w.strip() for w in desc.split() if w.strip()]
    return set(words)
