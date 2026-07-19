"""
LM Router — decides which modalities to activate for a given query.
Completely separate from the LMSegmenter so each can evolve independently.
"""
import json
import os
import gc
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_GLOBAL_LM = {}

def _get_lm(model_id):
    if model_id not in _GLOBAL_LM:
        print(f"[lm_router] Loading model: {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        _GLOBAL_LM[model_id] = (model, tokenizer)
    return _GLOBAL_LM[model_id]

ROUTE_PROMPT = """You are a video search routing agent. Analyze the following query and decide if ASR (audio transcript) or OCR (on-screen text) search is needed.

Rules:
1. Visual search is always ON.
2. Enable ASR ONLY if the query specifically seeks spoken words, dialogue, narration, lyrics, or speeches. Non-speech sounds (like "dog barking") do not use ASR.
3. Enable OCR ONLY if the query explicitly mentions written text, signs, logos, or captions visible on the screen.
4. If ASR is needed, extract ONLY the exact spoken phrases to be searched as the 'asr_query'.
5. If OCR is needed, extract ONLY the exact written phrases/words to be searched as the 'ocr_query'.

Query: "{text}"

Return ONLY a valid JSON object with exactly these fields: "use_asr" (boolean), "asr_query" (string or null), "use_ocr" (boolean), "ocr_query" (string or null), and "reason" (string). 

Examples:
Query: "A man running in the park with his dog"
{{"use_asr": false, "asr_query": null, "use_ocr": false, "ocr_query": null, "reason": "Visual scene, no speech or written text mentioned."}}

Query: "Find the moment where the president says we will not surrender"
{{"use_asr": true, "asr_query": "we will not surrender", "use_ocr": false, "ocr_query": null, "reason": "Specifically asks for a spoken statement."}}

Query: "Show me a video of a store with a red sign that says OPEN 24 HOURS"
{{"use_asr": false, "asr_query": null, "use_ocr": true, "ocr_query": "OPEN 24 HOURS", "reason": "The words 'OPEN 24 HOURS' are written on a visual sign."}}

Query: "Someone singing let it go while holding a book titled Frozen"
{{"use_asr": true, "asr_query": "let it go", "use_ocr": true, "ocr_query": "Frozen", "reason": "Requires both searching the audio transcript for singing and on-screen text for the book title."}}
"""

MODEL_MAP = {
    "qwen":   "Qwen/Qwen2.5-7B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen1.5": "Qwen/Qwen2.5-1.5B-Instruct",
    "phi3":   "microsoft/Phi-3-mini-4k-instruct",
}

class LMRouter:
    """
    Batched LLM Router — given a list of query strings, returns a list of
    route dicts like {"Use_asr": True/False, "asr_query": str|None}.
    """

    def __init__(self, engine_name="qwen"):
        self.model_id = MODEL_MAP.get(engine_name, "Qwen/Qwen2.5-7B-Instruct")
        self.model, self.tokenizer = _get_lm(self.model_id)
        cache_flag = os.environ.get("LM_CACHE", "true")
        self.use_cache = cache_flag.lower() in {"1", "true", "yes", "on"}
        self._cache = {}

        if (self.use_cache):
            print("[Router] Use caching model")
        self._cache_file = "artifacts/metadata/lm_route_cache.json"
        self._load_cache()

    # ------------------------------------------------------------------ cache
    def _load_cache(self):
        if self.use_cache and os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self):
        if not self.use_cache:
            return
        os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
        with open(self._cache_file, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ route
    def route(self, text: str) -> dict:
        """Route a single query. Returns e.g. {"Use_asr": False, "asr_query": None}."""
        if self.use_cache and text in self._cache:
            return self._normalize_route(self._cache[text])

        prompt = ROUTE_PROMPT.format(text=text)
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
                    max_new_tokens=64,
                    temperature=0.3 + attempt * 0.2,
                    do_sample=(attempt > 0),
                )
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            result = self._parse(response)
            if result is not None:
                if self.use_cache:
                    self._cache[text] = result
                    self._save_cache()
                return result

        print(f"[lm_router] WARNING: Could not parse route for query, defaulting to Use_asr=False, Use_ocr=False.")
        return {"Use_asr": False, "asr_query": None, "Use_ocr": False, "ocr_query": None}

    def route_batch(self, texts: list, batch_size: int = 4) -> list:
        """Route a batch of queries. Returns list of normalized route dicts."""
        results = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if self.use_cache and text in self._cache:
                results[i] = self._normalize_route(self._cache[text])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if not uncached_texts:
            return results

        prompts = [ROUTE_PROMPT.format(text=t) for t in uncached_texts]
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

        import math
        num_batches = math.ceil(len(text_inputs) / batch_size)
        all_responses = []

        print(f"[lm_router] Routing {len(text_inputs)} queries in {num_batches} batches...")
        for b in tqdm(range(num_batches), desc="[lm_router] Routing"):
            batch = text_inputs[b * batch_size : (b + 1) * batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0.01,
                    do_sample=False,
                )
            responses = self.tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            all_responses.extend(responses)

        for i, response in enumerate(all_responses):
            print(response)
            result = self._parse(response)
            if result is None:
                result = {"Use_asr": False, "asr_query": None, "Use_ocr": False, "ocr_query": None}
            if self.use_cache:
                self._cache[uncached_texts[i]] = result
            results[uncached_indices[i]] = result

        self._save_cache()
        return results

    # ------------------------------------------------------------------ helpers
    def _to_bool(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return False

    def _normalize_route(self, result):
        if not isinstance(result, dict):
            return None

        use_asr_value = result.get("Use_asr", result.get("use_asr", result.get("audio", False)))
        use_asr = self._to_bool(use_asr_value)
        asr_query = result.get("asr_query", result.get("ASR_query"))

        if isinstance(asr_query, str):
            asr_query = asr_query.strip()
            if asr_query.lower() in {"", "null", "none", "nil", "n/a"}:
                asr_query = None
        else:
            asr_query = None

        if use_asr and asr_query is None:
            use_asr = False

        use_ocr_value = result.get("Use_ocr", result.get("use_ocr", result.get("ocr", False)))
        use_ocr = self._to_bool(use_ocr_value)
        ocr_query = result.get("ocr_query", result.get("OCR_query"))

        if isinstance(ocr_query, str):
            ocr_query = ocr_query.strip()
            if ocr_query.lower() in {"", "null", "none", "nil", "n/a"}:
                ocr_query = None
        else:
            ocr_query = None

        if use_ocr and ocr_query is None:
            use_ocr = False

        return {"Use_asr": use_asr, "asr_query": asr_query, "Use_ocr": use_ocr, "ocr_query": ocr_query}

    def _parse(self, response: str):
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > 0:
                return self._normalize_route(json.loads(response[start:end]))
        except Exception:
            pass
        return None

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
