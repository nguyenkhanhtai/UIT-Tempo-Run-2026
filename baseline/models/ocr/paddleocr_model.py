from .base_ocr import BaseOCR
from PIL import Image
import numpy as np

class PaddleOCRModel(BaseOCR):
    def __init__(self, lang='en', use_gpu=True, **kwargs):
        # We also enable 'vi' (Vietnamese) if multilingual is needed.
        # PaddleOCR has an 'en' model, and 'latin' or 'vi' model. We will use 'en' by default but can set to 'vi'.
        from paddleocr import PaddleOCR
        import logging
        logging.getLogger('ppocr').setLevel(logging.ERROR)
        
        print(f"[init] Loading PaddleOCR (lang={lang}, use_gpu={use_gpu})...", flush=True)
        self.model = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)

    def extract(self, imgs: list) -> list:
        if not imgs:
            return []
            
        all_results = []
        for img in imgs:
            if isinstance(img, Image.Image):
                img = np.array(img.convert('RGB'))
            
            # PaddleOCR returns a list of lines, where each line is [box, (text, score)]
            result = self.model.ocr(img, cls=True)
            
            # PaddleOCR might return None if no text is found
            if not result or not result[0]:
                all_results.append("")
                continue
                
            lines = result[0]
            # Extract just the text from the result
            text = "\n".join([line[1][0] for line in lines])
            all_results.append(text)
            
        return all_results
