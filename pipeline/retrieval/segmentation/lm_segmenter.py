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
    "qwen":     "Qwen/Qwen2.5-7B-Instruct",
    "qwen7b":   "Qwen/Qwen2.5-7B-Instruct",
    "qwen3b":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen1.5":  "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3-8b": "Qwen/Qwen3-8B-Instruct",
    "phi3":     "microsoft/Phi-3-mini-4k-instruct",
}

PROMPTS = {
    "scene": (
        'You are an expert video analyst. Your task is to:\n'
        '1. Extract distinct, sequential scenes or events from a video search query and arrange them strictly in chronological order.\n'
        '2. Identify which specific scene contains the target frame requested by the user (often indicated by phrases like "identify the moment, at which exact moment, ... and the like").\n'
        '3. CRITICAL: You must extract the scenes by taking the exact words and phrases directly from the original query as much as possible. DO NOT hallucinate, invent, or add any details that are not explicitly present in the query.\n\n'
        'Output ONLY a valid JSON object with this exact structure, without any markdown tags:\n'
        '{{\n'
        '  "scenes": ["scene 1", "scene 2", ...],\n'
        '  "target_scene_index": integer (0-based index of the scene containing the target frame, or null if not explicitly requested)\n'
        '}}\n\n'
        'Examples:\n'
        'Query: A man eats an apple after picking it from the tree.\n'
        'Output: {{"scenes": ["A man picks an apple from the tree", "A man eats an apple"], "target_scene_index": null}}\n\n'
        'Query: A boy in a green shirt rides a bicycle down a dirt path, stops to pick up a stick, and throws it into a river; identify the moment the stick hits the water.\n'
        'Output: {{"scenes": ["A boy in a green shirt rides a bicycle down a dirt path", "The boy stops to pick up a stick", "The boy throws the stick into a river", "The stick hits the water"], "target_scene_index": 3}}\n\n'
        'Query: Identify the moment the scene transitions from a wide shot of a bustling city street at night with neon signs to a quiet indoor cafe where a woman sits alone drinking coffee.\n'
        'Output: {{"scenes": ["A wide shot of a bustling city street at night with neon signs", "The scene transitions to a quiet indoor cafe where a woman sits alone drinking coffee"], "target_scene_index": 1}}\n\n'
        'Query: A mechanic in blue overalls opens the hood of a red car, inspects the engine, and then reaches for a wrench on the nearby table. After fixing a loose bolt, he closes the hood and wipes his hands on a towel.\n'
        'Output: {{"scenes": ["A mechanic in blue overalls opens the hood of a red car and inspects the engine", "He reaches for a wrench on the nearby table", "He fixes a loose bolt", "He closes the hood and wipes his hands on a towel"], "target_scene_index": null}}\n\n'
        'Query: {text}'
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
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > 0:
                result = json.loads(response[start:end])
                
                # Check for new structured format
                if isinstance(result, dict) and "scenes" in result:
                    scenes = result.get("scenes", [])
                    idx = result.get("target_scene_index", None)
                    cleaned = [x.strip() for x in scenes if x.strip()]
                    if cleaned:
                        return {"scenes": cleaned, "target_scene_index": idx}
                        
                # Backward compatibility for old cache (list of strings)
                elif isinstance(result, list) and all(isinstance(x, str) for x in result):
                    cleaned = [x.strip() for x in result if x.strip()]
                    if cleaned:
                        return {"scenes": cleaned, "target_scene_index": None}
                        
            # Also try to parse as array if it outputted an array directly
            start_arr = response.find("[")
            end_arr = response.rfind("]") + 1
            if start_arr != -1 and end_arr > 0:
                result = json.loads(response[start_arr:end_arr])
                if isinstance(result, list) and all(isinstance(x, str) for x in result):
                    cleaned = [x.strip() for x in result if x.strip()]
                    if cleaned:
                        return {"scenes": cleaned, "target_scene_index": None}
                        
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

    def segment_batch(self, texts: list, batch_size: int = 8) -> list:
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

        print(f"[lm_segmenter] Running batched inference for {len(text_inputs)} queries...")
        all_responses = []
        b_idx = 0
        pbar = tqdm(total=len(text_inputs), desc="[lm_segmenter] Segmenting")
        
        while b_idx < len(text_inputs):
            batch = text_inputs[b_idx : b_idx + batch_size]
            try:
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
                
                b_idx += len(batch)
                pbar.update(len(batch))
                
                del inputs, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"\n[lm_segmenter] Error during generation: {e}")
                if 'inputs' in locals():
                    del inputs
                if 'outputs' in locals():
                    del outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    print(f"[lm_segmenter] Halving batch size to {batch_size} and retrying...")
                else:
                    print(f"[lm_segmenter] Batch size is already 1, cannot reduce further. Failing batch.")
                    responses = [""] * len(batch)
                    all_responses.extend(responses)
                    b_idx += len(batch)
                    pbar.update(len(batch))
                    
        pbar.close()

        for i, response in enumerate(all_responses):
            result = self._parse(response)
            if result is None:
                result = [uncached_texts[i]]  # fallback: original text
            if self.use_cache:
                self._cache[self.mode][uncached_texts[i]] = result
            results[uncached_indices[i]] = result

        self._save_cache()
        return results
