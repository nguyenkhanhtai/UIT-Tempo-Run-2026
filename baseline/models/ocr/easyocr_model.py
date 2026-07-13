from .base_ocr import BaseOCR
import numpy as np

class EasyOCRModel(BaseOCR):
    def __init__(self, langs=['en', 'vi'], use_gpu=True):
        import easyocr
        self.reader = easyocr.Reader(langs, gpu=use_gpu)

    def extract(self, imgs: list) -> list:
        results = []
        for img in imgs:
            # easyocr expects numpy arrays or paths
            # Assuming imgs are numpy arrays (RGB)
            img_np = np.array(img)
            text_results = self.reader.readtext(img_np, detail=0)
            results.append(" ".join(text_results))
        return results
