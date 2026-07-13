from .base_ocr import BaseOCR
import numpy as np

class PaddleOCRModel(BaseOCR):
    def __init__(self, model_name='ch_PP-OCRv4_xx', device='cuda:0', **kwargs):
        from paddleocr import PaddleOCR
        self.device = device
        self.model_name = model_name
        use_gpu = 'cuda' in device
        print(f"[init] Loading PaddleOCR (use_gpu={use_gpu})...", flush=True)
        # We use the multilingual/chinese model by default which also supports English and numbers
        self.ocr = PaddleOCR(use_angle_cls=True, lang='vi', use_gpu=use_gpu, show_log=False)

    def extract(self, imgs: list) -> list:
        results = []
        for img in imgs:
            try:
                # PaddleOCR expects numpy array in BGR format typically, or RGB
                # We can pass numpy array directly
                img_np = np.array(img)
                # Convert RGB to BGR for PaddleOCR
                img_bgr = img_np[:, :, ::-1]
                
                res = self.ocr.ocr(img_bgr, cls=True)
                
                # result is a list of lines, where each line is [box, (text, score)]
                # we just want to extract the text
                extracted_text = []
                if res and res[0]:
                    for line in res[0]:
                        extracted_text.append(line[1][0])
                
                results.append("\n".join(extracted_text))
            except Exception as e:
                print(f"PaddleOCR error on frame: {e}")
                results.append("")
                
        return results
