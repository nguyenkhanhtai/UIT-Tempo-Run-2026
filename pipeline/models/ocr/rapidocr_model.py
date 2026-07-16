from .base_ocr import BaseOCR
import numpy as np
from PIL import Image

class RapidOCRModel(BaseOCR):
    def __init__(self, model_name=None, device='cpu', **kwargs):
        from rapidocr_onnxruntime import RapidOCR
        self.device = device
        print("[init] Loading RapidOCR (ONNX)...", flush=True)
        self.engine = RapidOCR()

    def extract(self, imgs: list, batch_size: int = 8, **kwargs) -> list:
        if not imgs:
            return []
            
        all_results = []
        for img in imgs:
            if isinstance(img, Image.Image):
                img_array = np.array(img.convert('RGB'))
            else:
                img_array = np.array(img)
                
            # Run OCR
            result, elapse = self.engine(img_array)
            
            if result:
                # result is a list of tuples: (box, text, confidence)
                texts = [res[1] for res in result if res[1].strip()]
                all_results.append(" ".join(texts))
            else:
                all_results.append("")
                
        return all_results
