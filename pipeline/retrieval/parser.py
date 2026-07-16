"""
Purpose: Parses user queries into main and sub-queries based on sentences,
and extracts OCR keywords enclosed in double quotes.
"""
import re

def parse_queries(tasks):
    all_queries = []
    task_mapping = [] # (ti, is_main, sub_id)
    
    for ti, task in enumerate(tasks):
        desc = task["description"]
        # Split by punctuation (. ! ?)
        sentences = [s.strip() for s in re.split(r'[.!?]', desc) if s.strip()]
        
        if not sentences:
            sentences = [desc]
            
        main_query = sentences[0]
        
        all_queries.append(main_query)
        task_mapping.append((ti, True, 0)) # Main query
        
        # Subsequent sentences are sub-queries
        sub_idx = 1
        for sub_q in sentences[1:]:
            all_queries.append(sub_q)
            task_mapping.append((ti, False, sub_idx))
            sub_idx += 1
            
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
