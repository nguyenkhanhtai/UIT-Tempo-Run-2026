"""
Query Expander — generates N synonymous queries for a given query to improve recall in reranking.
"""
import json
import os
import gc
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pipeline.retrieval.routing.lm_router import _get_lm, MODEL_MAP, _GLOBAL_LM

QE_PROMPT = """You are an expert query expansion agent for a video search engine. 
Given a user search query, generate exactly 5 synonymous or semantically similar queries that search for the exact same core video event.

CRITICAL INSTRUCTIONS:
1. Pay strict attention to descriptive words (colors, sizes, emotions, specific adjectives).
2. Preserve important keywords, fixed phrases, and specific entities. Do not lose the core meaning.
3. Use different vocabulary, sentence structures, or phrasing to improve search recall, but keep the critical constraints from the original query.
4. Do not hallucinate new details not present in the original query.

Query: "{text}"

Return ONLY a valid JSON array of exactly 5 strings. Do not explain anything.

Example:
Query: "A man running in the park with his dog"
[
  "A male jogging outdoors alongside his pet dog in a park",
  "A guy and his dog running through a grassy park area",
  "Footage of a man taking his dog for a run in nature",
  "A person sprinting with a dog in a public park",
  "A man exercising with his canine companion in the park"
]
"""

class QueryExpander:
    """
    Batched LLM Query Expander — given a list of query strings, returns a list of
    lists, where each inner list contains N synonymous queries.
    """

    def __init__(self, engine_name="qwen", num_expansions=0):
        self.model_id = MODEL_MAP.get(engine_name, "Qwen/Qwen2.5-7B-Instruct")
        self.model, self.tokenizer = _get_lm(self.model_id)
        
        cache_flag = os.environ.get("LM_CACHE", "true")
        self.use_cache = cache_flag.lower() in {"1", "true", "yes", "on"}
        self.num_expansions = num_expansions
        
        self._cache = {}
        self._cache_file = "artifacts/metadata/qe_cache.json"
        
        print(f"[qe_expander] Use {num_expansions} queries")
        if self.use_cache:
            print("[qe_expander] Use caching model")
        self._load_cache()

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

    def expand_batch(self, texts: list, batch_size: int = 16) -> list:
        """Expand a batch of queries. Returns list of lists of strings."""
        results = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if self.use_cache and text in self._cache:
                results[i] = self._normalize_result(self._cache[text], text)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if not uncached_texts:
            return results

        prompts = [QE_PROMPT.format(text=t) for t in uncached_texts]
        messages_batch = [
            [
                {"role": "system", "content": "You are a helpful assistant that strictly outputs valid JSON arrays. Do not use markdown blocks."},
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

        print(f"[qe_expander] Expanding {len(text_inputs)} queries in {num_batches} batches...")
        for b in tqdm(range(num_batches), desc="[qe_expander] Expanding"):
            batch = text_inputs[b * batch_size : (b + 1) * batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False,
                )
            responses = self.tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            all_responses.extend(responses)

        for i, response in enumerate(all_responses):
            result = self._parse(response)
            norm_res = self._normalize_result(result, uncached_texts[i])
            if self.use_cache:
                self._cache[uncached_texts[i]] = norm_res
            results[uncached_indices[i]] = norm_res

        self._save_cache()
        return results

    def _normalize_result(self, result, original_text):
        if isinstance(result, list) and all(isinstance(x, str) for x in result):
            cleaned = [x.strip() for x in result if x.strip()]
            if cleaned:
                # pad or truncate to exact num_expansions
                while len(cleaned) < self.num_expansions:
                    cleaned.append(original_text)
                return cleaned[:self.num_expansions]
        return [original_text] * self.num_expansions

    def _parse(self, response: str):
        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end > 0:
                return json.loads(response[start:end])
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
