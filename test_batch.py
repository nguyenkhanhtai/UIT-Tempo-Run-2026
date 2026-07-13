import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from baseline.models.ocr.vintern_ocr_model import load_image

device = "cuda:0"
model_name = "5CD-AI/Vintern-1B-v3_5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).eval().to(device)

model.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')

IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

imgs = [Image.new('RGB', (100, 100), color='white'), Image.new('RGB', (200, 100), color='black')]

input_ids_list = []
pixel_values_list = []

for img in imgs:
    pixel_values = load_image(img, max_num=12)
    num_patches = pixel_values.shape[0]
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
    query = "<image>\nPlease transcribe the text in this image.".replace('<image>', image_tokens, 1)
    
    import sys
    module = model.__class__.__module__
    conv_module = sys.modules[module.rsplit('.', 1)[0] + '.conversation']
    template = conv_module.get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], query)
    template.append_message(template.roles[1], None)
    full_query = template.get_prompt()
    
    inputs = tokenizer(full_query, return_tensors='pt')
    input_ids_list.append(inputs['input_ids'][0])
    pixel_values_list.append(pixel_values)

from torch.nn.utils.rnn import pad_sequence
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

input_ids_padded = pad_sequence([seq.flip(0) for seq in input_ids_list], batch_first=True, padding_value=tokenizer.pad_token_id).flip(1)
attention_mask = (input_ids_padded != tokenizer.pad_token_id).long()
pixel_values_tensor = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).to(device)

print("Running batch generate...")
output = model.generate(
    pixel_values=pixel_values_tensor,
    input_ids=input_ids_padded.to(device),
    attention_mask=attention_mask.to(device),
    max_new_tokens=100,
    do_sample=False,
    eos_token_id=tokenizer.convert_tokens_to_ids(template.sep.strip())
)

responses = tokenizer.batch_decode(output, skip_special_tokens=True)
for i, res in enumerate(responses):
    print(f"Resp {i}: {res.split(template.sep.strip())[0].strip()}")
