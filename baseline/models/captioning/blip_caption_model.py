from .base_caption import BaseCaption
import torch
from PIL import Image
import gc

class BlipCaptionModel(BaseCaption):
    def __init__(self, model_name=None, device='cuda:0', **kwargs):
        from ..model_shared import get_shared_model
        self.device = device
        self.model_name = model_name or 'Salesforce/blip-image-captioning-base'
        
        def load_blip():
            from transformers import BlipProcessor, BlipForConditionalGeneration
            processor = BlipProcessor.from_pretrained(self.model_name)
            model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32
            ).to(self.device)
            model.eval()
            return processor, model
            
        self.processor, self.model = get_shared_model(('blip', self.model_name, self.device), load_blip)

    def extract(self, imgs: list, batch_size: int = 8, **kwargs) -> list:
        if not imgs:
            return []
            
        all_results = []
        
        for i in range(0, len(imgs), batch_size):
            batch_imgs = imgs[i:i+batch_size]
            pil_imgs = []
            for img in batch_imgs:
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img).convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                pil_imgs.append(img)
                
            inputs = self.processor(
                images=pil_imgs,
                return_tensors="pt"
            ).to(self.device, self.model.dtype)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=3
                )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            for text in generated_text:
                all_results.append(text.strip())
                
            # Cleanup
            del inputs
            del generated_ids
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return all_results
