import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    transform = build_transform(image_size)
    pixel_values = [transform(image)]
    return torch.stack(pixel_values)

path = 'OpenGVLab/InternVL2-8B'

model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

image1 = Image.new('RGB', (224, 224), color = 'red')
image2 = Image.new('RGB', (224, 224), color = 'blue')
pixel_values1 = dynamic_preprocess(image1).to(torch.bfloat16).cuda()
pixel_values2 = dynamic_preprocess(image2).to(torch.bfloat16).cuda()

pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
num_patches_list = [pixel_values1.shape[0], pixel_values2.shape[0]]

question = '<image>\nDescribe the first image.\n<image>\nDescribe the second image.'
print(f"Num patches list: {num_patches_list}")

generation_config = dict(max_new_tokens=1024, do_sample=False)

try:
    response = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list)
    print(f'Response: {response}')
except Exception as e:
    print(f"Chat error: {e}")
