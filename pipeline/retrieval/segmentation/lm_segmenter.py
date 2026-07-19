"""
LM Segmenter — splits a query into scenes or objects using a local LLM.
Routing logic has been moved to pipeline/retrieval/routing/lm_router.py.
"""
import json
import os
import gc
import math
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_GLOBAL_LM = {}

def get_lm(model_id):
    if model_id not in _GLOBAL_LM:
        print(f"[lm_segmenter] Loading Local LM: {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        _GLOBAL_LM[model_id] = (model, tokenizer)
    return _GLOBAL_LM[model_id]

MODEL_MAP = {
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "qwen7b":  "Qwen/Qwen2.5-7B-Instruct",
    "qwen3b":  "Qwen/Qwen2.5-3B-Instruct",
    "qwen1.5": "Qwen/Qwen2.5-1.5B-Instruct",
    "phi3":    "microsoft/Phi-3-mini-4k-instruct",
}

PROMPTS = {
    "scene": (
        'This is a video search query. Split the query into distinct sequential scenes or events. '
        'Only return a JSON array of strings containing the exact sub-sentences/phrases from the query. '
        'Example: ["A person walks into a room", "They sit on a chair"]. '
        'Do not explain anything. Return ONLY a valid JSON array.\n\nQuery: {text}'
    ),
    "audio": (
        "This is a video search query. If the query mentions someone speaking, saying, or singing "
        "something (e.g. 'a man says hello', 'someone yells stop'), extract ONLY the exact spoken "
        'dialogue/speech. Return it as a JSON array of strings (e.g., ["hello"]). '
        "If there is no speech mentioned, return an empty JSON array []. "
        "Do not explain anything. Return ONLY a valid JSON array.\n\nQuery: {text}"
    ),
    "object": (
        'For this sentence, extract the most important noun phrases or objects. '
        'Only return a JSON array of strings containing the exact phrases from the query. '
        'Example: ["red car", "dog"]. '
        "Do not explain anything. Return ONLY a valid JSON array.\n\nQuery: {text}"
    ),
}


class LMSegmenter:
    def __init__(self, engine_name, mode="scene"):
        """
        engine_name: 'qwen', 'qwen3b', 'qwen1.5', 'phi3'
        mode: 'scene' | 'audio' | 'object'
        """
        if mode == "route":
            raise ValueError(
                "mode='route' has been moved to pipeline.retrieval.routing.LMRouter. "
                "LMSegmenter only handles scene/audio/object segmentation."
            )
        self.mode = mode
        self.model_id = MODEL_MAP.get(engine_name, "Qwen/Qwen2.5-7B-Instruct")
        self.model, self.tokenizer = get_lm(self.model_id)

        cache_flag = os.environ.get("LM_CACHE", "true")
        self.use_cache = cache_flag.lower() in {"1", "true", "yes", "on"}
        self._cache_file = "artifacts/metadata/lm_segment_cache.json"
        self._cache = {"scene": {}, "audio": {}, "object": {}}
        self._load_cache()

    # ------------------------------------------------------------------ cache
    def _load_cache(self):
        if self.use_cache and os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for key in self._cache:
                    if key in data:
                        self._cache[key] = data[key]
            except Exception:
                pass

    def _save_cache(self):
        if not self.use_cache:
            return
        os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
        with open(self._cache_file, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ cleanup
    def cleanup(self):
        global _GLOBAL_LM
        _GLOBAL_LM.pop(self.model_id, None)
        del self.model
        self.model = None
        del self.tokenizer
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ internal
    def _make_prompt(self, text: str) -> str:
        template = PROMPTS.get(self.mode, PROMPTS["object"])
        return template.format(text=text)

    def _parse(self, response: str):
        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end > 0:
                result = json.loads(response[start:end])
                if isinstance(result, list) and all(isinstance(x, str) for x in result):
                    cleaned = [x.strip() for x in result if x.strip()]
                    return cleaned if cleaned else None
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ public API
    def segment(self, text: str) -> list:
        """Segment a single query; returns a list of sub-strings."""
        if self.use_cache and text in self._cache[self.mode]:
            return self._cache[self.mode][text]

        prompt = self._make_prompt(text)
        messages = [
            {"role": "system", "content": "You are a helpful assistant that strictly outputs valid JSON. Do not use markdown blocks."},
            {"role": "user",   "content": prompt},
        ]
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)

        for attempt in range(3):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.3 + attempt * 0.2,
                    do_sample=(attempt > 0),
                )
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            result = self._parse(response)
            if result is not None:
                if self.use_cache:
                    self._cache[self.mode][text] = result
                    self._save_cache()
                return result

        print(f"[lm_segmenter] WARNING: Could not parse response, returning original text.")
        return [text]

    def segment_batch(self, texts: list, batch_size: int = 16) -> list:
        """Segment a list of queries in batches. Returns list of lists."""
        results = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if self.use_cache and text in self._cache[self.mode]:
                results[i] = self._cache[self.mode][text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if not uncached_texts:
            return results

        prompts = [self._make_prompt(t) for t in uncached_texts]
        messages_batch = [
            [
                {"role": "system", "content": "You are a helpful assistant that strictly outputs valid JSON. Do not use markdown blocks."},
                {"role": "user",   "content": p},
            ]
            for p in prompts
        ]

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        text_inputs = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]

        num_batches = math.ceil(len(text_inputs) / batch_size)
        all_responses = []

        print(f"[lm_segmenter] Running batched inference for {len(text_inputs)} queries in {num_batches} batches...")
        for b in tqdm(range(num_batches), desc="[lm_segmenter] Segmenting"):
            batch = text_inputs[b * batch_size : (b + 1) * batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.01,
                    do_sample=False,
                )
            responses = self.tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            all_responses.extend(responses)

        for i, response in enumerate(all_responses):
            result = self._parse(response)
            if result is None:
                result = [uncached_texts[i]]  # fallback: original text
            if self.use_cache:
                self._cache[self.mode][uncached_texts[i]] = result
            results[uncached_indices[i]] = result

        self._save_cache()
        return results
