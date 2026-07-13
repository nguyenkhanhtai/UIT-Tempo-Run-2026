from .base_ocr import BaseOCR
import tempfile
import os

class TransformersOCRModel(BaseOCR):
    def __init__(self, model_name='stepfun-ai/GOT-OCR2_0', device='cuda:0', **kwargs):
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.model_name = model_name
        
        print(f"[init] Loading Transformers OCR ({model_name})... This may take a while.", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            low_cpu_mem_usage=True, 
            device_map=device, 
            use_safetensors=True, 
            pad_token_id=self.tokenizer.eos_token_id
        )
        self.model = self.model.eval()

    def extract(self, imgs: list) -> list:
        results = []
        for img in imgs:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                img.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                # Use ocr_type='ocr' for general plain text extraction (specific to GOT-OCR API)
                res = self.model.chat(self.tokenizer, tmp_path, ocr_type='ocr')
                results.append(res)
            except Exception as e:
                print(f"Transformers OCR error on frame: {e}")
                results.append("")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        return results
