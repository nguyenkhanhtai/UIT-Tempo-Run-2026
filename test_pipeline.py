import torch
from transformers import pipeline
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, i):
        return self.data_list[i]

dev = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32", device=dev)

img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
inputs = [{"image": img, "candidate_labels": ["dog", "cat"]} for _ in range(5)]

ds = ListDataset(inputs)

print("Running dataset...")
for out in pipe(ds, batch_size=2):
    print(type(out), len(out) if isinstance(out, list) else "not list")
