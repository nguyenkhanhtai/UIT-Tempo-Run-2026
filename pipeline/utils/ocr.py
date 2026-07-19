import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi

class OCRReranker:
    def __init__(self, lang="en", use_gpu=True):
        print(f"[ocr_reranker] Initializing PaddleOCR (lang={lang}, use_gpu={use_gpu})...")
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)
        self.cache = {}

    def get_frame_path(self, video_id, frame_ms):
        base_dir = os.environ.get("KEYFRAME_DIR", "dataset/keyframes")
        v_dir = os.path.join(base_dir, str(video_id))
        if not os.path.exists(v_dir):
            return None
            
        exact_path = os.path.join(v_dir, f"{int(frame_ms)}.jpg")
        if os.path.exists(exact_path):
            return exact_path
            
        files = glob.glob(os.path.join(v_dir, "*.jpg"))
        if not files:
            return None
            
        ts_list = []
        for f in files:
            bn = os.path.basename(f).replace(".jpg", "")
            if bn.isdigit():
                ts_list.append(int(bn))
                
        if not ts_list:
            return None
            
        closest_ts = min(ts_list, key=lambda x: abs(x - int(frame_ms)))
        return os.path.join(v_dir, f"{closest_ts}.jpg")

    def extract_ocr_queries(self, text):
        import re
        # Extract quoted texts, which usually represent explicit OCR targets
        quotes = re.findall(r'"([^"]*)"', text)
        if quotes:
            # Tokenize by space for BM25
            keywords = []
            for q in quotes:
                keywords.extend(q.lower().split())
            return keywords
            
        return []

    def rerank(self, candidates, task, top_k=500, boost_score=2.0):
        query = task["description"]
        ocr_keywords = self.extract_ocr_queries(query)
        
        if not ocr_keywords:
            return candidates
            
        print(f"[ocr_reranker] Task {task['task_id']}: Boosting with BM25 for keywords: {ocr_keywords}")
        
        to_process = candidates[:top_k]
        remainder = candidates[top_k:]
        
        corpus = []
        valid_cands = []
        
        for cand in tqdm(to_process, desc="[ocr_reranker] Scanning top candidates"):
            vid = cand["video_id"]
            ts = cand["frame_ms"]
            cache_key = f"{vid}_{ts}"
            
            if cache_key in self.cache:
                texts = self.cache[cache_key]
            else:
                img_path = self.get_frame_path(vid, ts)
                if img_path is None:
                    texts = []
                else:
                    result = self.ocr.ocr(img_path, cls=True)
                    texts = []
                    if result and result[0]:
                        for line in result[0]:
                            texts.append(line[1][0].lower())
                self.cache[cache_key] = texts
                
            # Combine all texts in the frame into a single document string, then tokenize
            doc_text = " ".join(texts)
            tokenized_doc = doc_text.split()
            corpus.append(tokenized_doc)
            valid_cands.append(cand)
            
        if not corpus:
            return candidates
            
        # BM25 Scoring
        bm25 = BM25Okapi(corpus)
        doc_scores = bm25.get_scores(ocr_keywords)
        
        # Apply boost to candidates based on BM25 score
        for i, cand in enumerate(valid_cands):
            # doc_scores[i] will be > 0 if there's a match
            if doc_scores[i] > 0:
                # Add scaled BM25 score to the base similarity
                # Using min to cap the maximum boost
                cand["sim"] += min(boost_score, doc_scores[i] * 0.5)
                
        all_cands = valid_cands + remainder
        all_cands.sort(key=lambda x: x["sim"], reverse=True)
        return all_cands
