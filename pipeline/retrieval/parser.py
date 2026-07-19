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
            
    for ti, task in enumerate(tasks):
        desc = task["description"]
        
        if not split_query:
            all_queries.append(desc)
            task_mapping.append((ti, 0, 0)) # Main query only
            continue
            
        if use_sequential and scene_seg is not None:
            scenes = scene_seg.segment(desc)
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
                all_queries.append(obj)
                task_mapping.append((ti, sent_idx, seg_idx))
            
    return all_queries, task_mapping

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
