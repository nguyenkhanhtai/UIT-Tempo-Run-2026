import torch
import gc

_CACHE = {}

def get_florence2(model_name, device):
    key = (model_name, device)
    if key not in _CACHE:
        from transformers import AutoProcessor, AutoModelForCausalLM
        print(f"[init] Loading Shared Florence-2 ({model_name}) to {device}...", flush=True)
        torch.backends.cudnn.enabled = False
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True
        ).eval().to(device)
        _CACHE[key] = (processor, model)
    else:
        print(f"[init] Using cached Florence-2 ({model_name}) on {device}", flush=True)
        
    return _CACHE[key]
