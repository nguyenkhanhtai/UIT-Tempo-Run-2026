from .base_od import BaseOD
import torch
from PIL import Image
import gc

class Florence2ODModel(BaseOD):
    def __init__(self, model_name=None, device='cuda:0', **kwargs):
        from ..model_shared import get_shared_model
        
        self.device = device
        self.model_name = model_name or 'microsoft/Florence-2-large'

        def load_florence():
            from transformers import AutoProcessor, AutoModelForCausalLM
            torch.backends.cudnn.enabled = False
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                attn_implementation="eager",
                trust_remote_code=True
            ).eval().to(self.device)
            return processor, model
            
        self.processor, self.model = get_shared_model(('florence2', self.model_name, self.device), load_florence)
        self.dtype = self.model.dtype

    def extract(self, imgs: list, batch_size: int = 8, **kwargs) -> list:
        if not imgs:
            return []
            
        all_results = []
        task_prompt = '<DENSE_REGION_CAPTION>'
        
        # Ensure pad token exists
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
            
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
                text=[task_prompt] * len(pil_imgs),
                images=pil_imgs,
                return_tensors="pt"
            ).to(self.device, self.dtype)
            
            # Florence-2 specific formatting
            inputs["input_ids"] = inputs["input_ids"].to(torch.int64)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"].to(torch.int64)
                
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
                use_cache=False
            )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
            
            for j, text in enumerate(generated_text):
                if self.processor.tokenizer.pad_token:
                    text = text.replace(self.processor.tokenizer.pad_token, "")
                parsed_answer = self.processor.post_process_generation(
                    text, task=task_prompt, image_size=pil_imgs[j].size
                )
                
                # parsed_answer will be a dict with key '<DENSE_REGION_CAPTION>'
                result_dict = parsed_answer.get(task_prompt, {})
                if isinstance(result_dict, dict) and 'labels' in result_dict:
                    # Extract unique labels and remove empty strings
                    labels = list(set([lbl.strip() for lbl in result_dict['labels'] if lbl.strip()]))
                    all_results.append(labels)
                else:
                    all_results.append([])
                
            # Cleanup
            del inputs
            del generated_ids
            
        gc.collect()
        return all_results
