import torch
from transformers import AutoModel, AutoTokenizer

model_id = "OpenGVLab/InternVL2-8B"
try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    print("Tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
