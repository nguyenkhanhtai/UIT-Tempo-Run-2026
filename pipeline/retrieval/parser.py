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
        
        main_query = desc.strip()
        all_queries.append(main_query)
        task_mapping.append((ti, True, 0)) # Main query
            
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
