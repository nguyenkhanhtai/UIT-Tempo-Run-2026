from __future__ import annotations
import numpy as np
from typing import List
from .base_embedding import BaseEmbedding

class XClipModel(BaseEmbedding):
    def __init__(self, model_name="microsoft/xclip-base-patch32", pretrained=None, device=None, precision=None):
        import torch
        from transformers import XCLIPProcessor, XCLIPModel
        
        # We ignore pretrained as X-CLIP models in HF usually just use model_name 
        # (e.g. microsoft/xclip-base-patch32)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = XCLIPProcessor.from_pretrained(model_name)
        self.is_video_model = True
        
        dtype = torch.float16 if precision == "fp16" else torch.float32
        self.model = XCLIPModel.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        self.dim = self.model.config.projection_dim

    def encode_images(self, pil_images: list, batch_size=16) -> np.ndarray:
        torch = self.torch
        feats = []
        for i in range(0, len(pil_images), batch_size):
            batch_imgs = pil_images[i:i+batch_size]
            
            # Check if elements are already lists (snippets)
            if isinstance(batch_imgs[0], list):
                videos = batch_imgs
            else:
                videos = [[img] for img in batch_imgs]
            
            inputs = self.processor(videos=videos, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                video_features = self.model.get_video_features(**inputs)
                # L2 normalization for Cosine Similarity
                video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
                feats.append(video_features.cpu().float().numpy().astype(np.float16))
                
        return np.concatenate(feats, 0) if feats else np.zeros((0, self.dim), np.float16)

    def encode_texts(self, texts: list[str], batch_size=256) -> np.ndarray:
        torch = self.torch
        feats = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                # L2 normalization for Cosine Similarity
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                feats.append(text_features.cpu().float().numpy().astype(np.float32))
                
        return np.concatenate(feats, 0) if feats else np.zeros((0, self.dim), np.float32)
