from .base_ocr import BaseOCR
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import gc
import sys

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class VinternOCRModel(BaseOCR):
    def __init__(self, model_name=None, device='cuda:0', **kwargs):
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.model_name = model_name or '5CD-AI/Vintern-1B-v3_5'
        print(f"[init] Loading Vintern-1B ({self.model_name}) to {device}...", flush=True)
        
        # FIX CUDNN CRASH
        torch.backends.cudnn.enabled = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().to(self.device)
        
        # InternVL requires this token ID explicitly when using generic generate
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        
        # Load conversation template module
        module = self.model.__class__.__module__
        self.conv_module = sys.modules[module.rsplit('.', 1)[0] + '.conversation']

    def extract(self, imgs: list, batch_size: int = 4) -> list:
        if not imgs:
            return []
            
        all_results = []
        
        # Ensure pad token exists
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # Process in chunks to avoid OOM
        for i in range(0, len(imgs), batch_size):
            batch_imgs = imgs[i:i+batch_size]
            
            input_ids_list = []
            pixel_values_list = []
            
            for img in batch_imgs:
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                
                pixel_values = load_image(img, max_num=12)
                num_patches = pixel_values.shape[0]
                
                # Format prompt
                image_tokens = '<img>' + '<IMG_CONTEXT>' * self.model.num_image_token * num_patches + '</img>'
                query = "<image>\\nPlease transcribe the text in this image.".replace('<image>', image_tokens, 1)
                
                template = self.conv_module.get_conv_template(self.model.template)
                template.system_message = self.model.system_message
                template.append_message(template.roles[0], query)
                template.append_message(template.roles[1], None)
                full_query = template.get_prompt()
                
                # Tokenize
                inputs = self.tokenizer(full_query, return_tensors='pt')
                input_ids_list.append(inputs['input_ids'][0])
                pixel_values_list.append(pixel_values)
                
            # Pad sequences (left padding)
            from torch.nn.utils.rnn import pad_sequence
            input_ids_padded = pad_sequence([seq.flip(0) for seq in input_ids_list], batch_first=True, padding_value=self.tokenizer.pad_token_id).flip(1)
            attention_mask = (input_ids_padded != self.tokenizer.pad_token_id).long()
            pixel_values_tensor = torch.cat(pixel_values_list, dim=0).to(self.dtype).to(self.device)
            
            # Generate
            output = self.model.generate(
                pixel_values=pixel_values_tensor,
                input_ids=input_ids_padded.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=1024,
                do_sample=False,
                eos_token_id=self.tokenizer.convert_tokens_to_ids(template.sep.strip())
            )
            
            # Decode
            responses = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            for res in responses:
                all_results.append(res.split(template.sep.strip())[0].strip())
                
            # Cleanup chunk
            del pixel_values_tensor
            del input_ids_padded
            del attention_mask
            del output
            torch.cuda.empty_cache()
            
        gc.collect()
        return all_results
