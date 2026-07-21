import torch
import numpy as np
from PIL import Image
from transformers import LlavaNextVideoProcessor

model_id = "llava-hf/llava-next-video-7b-hf"
processor = LlavaNextVideoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": "Describe the video."},
        ],
    }
]

text = processor.apply_chat_template(messages, add_generation_prompt=True)
print("Video Prompt:", text)

video_frames = [Image.new("RGB", (224, 224), color="red"), Image.new("RGB", (224, 224), color="blue")]
# LLaVA Next Video Processor expects video as a list of frames or numpy array
import numpy as np
video_arr = np.stack([np.array(img) for img in video_frames])

inputs = processor(text=[text], videos=[video_arr], padding=True, return_tensors="pt")
print("Processor output keys:", inputs.keys())

