from .base_ocr import BaseOCR
from PIL import Image
import numpy as np

class EasyOCRModel(BaseOCR):
    def __init__(self, lang_list=None, use_gpu=True, **kwargs):
        if lang_list is None:
            lang_list = ['en', 'vi']  # Default to English and Vietnamese
        from easyocr import Reader
        import logging
        logging.getLogger('easyocr').setLevel(logging.ERROR)
        
        # FIX CUDNN CRASH
        import torch
        torch.backends.cudnn.enabled = False
        
        print(f"[init] Loading EasyOCR (lang={lang_list}, use_gpu={use_gpu})...", flush=True)
        self.model = Reader(lang_list, gpu=use_gpu)

    def extract(self, imgs: list) -> list:
        if not imgs:
            return []
            
        all_results = []
        for img in imgs:
            if isinstance(img, Image.Image):
                img = np.array(img.convert('RGB'))
            
            # EasyOCR returns a list of results: [([[x,y],...], text, confidence)]
            result = self.model.readtext(img)
            
            if not result:
                all_results.append("")
                continue
                
            # Extract just the text from the result
            text = "\n".join([line[1] for line in result])
            all_results.append(text)
            
        return all_results
