from .base_ocr import BaseOCR
import torch

class Florence2OCRModel(BaseOCR):
    def __init__(self, model_name='microsoft/Florence-2-large', device='cuda:0', **kwargs):
        # Mock flash_attn to prevent ImportError from transformers dynamic module check
        import sys, types, importlib.machinery
        if "flash_attn" not in sys.modules:
            m = types.ModuleType("flash_attn")
            m.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
            sys.modules["flash_attn"] = m
            
        from transformers import AutoProcessor, AutoModelForCausalLM
        self.device = device
        self.model_name = model_name
        
        print(f"[init] Loading Florence-2 OCR ({model_name})... This may take a while.", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        # Florence-2 can be loaded in float16 for speed and memory efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.float16
        ).to(device)
        self.model = self.model.eval()

    def extract(self, imgs: list) -> list:
        results = []
        task_prompt = "<OCR>"
        
        for img in imgs:
            try:
                # Ensure image is in RGB format
                if img.mode != "RGB":
                    img = img.convert("RGB")
                    
                inputs = self.processor(text=task_prompt, images=img, return_tensors="pt").to(self.device, torch.float16)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        do_sample=False,
                        num_beams=3,
                    )
                
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = self.processor.post_process_generation(
                    generated_text, 
                    task=task_prompt, 
                    image_size=(img.width, img.height)
                )
                
                # Florence-2 returns a dict with the task name as key
                extracted_text = parsed_answer.get(task_prompt, "")
                results.append(extracted_text)
                
            except Exception as e:
                print(f"Florence-2 OCR error on frame: {e}")
                results.append("")
                
        return results
