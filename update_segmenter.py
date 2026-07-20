import re

with open("pipeline/retrieval/segmentation/lm_segmenter.py", "r") as f:
    content = f.read()

batch_func = """
    def segment_batch(self, texts):
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text in self.cache[self.mode]:
                results.append(self.cache[self.mode][text])
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
                
        if not uncached_texts:
            return results
            
        prompts = []
        for text in uncached_texts:
            if self.mode == "scene":
                p = f"This is a query for finding a moment in a video. Find the points where you think scene transitions might occur and split the following text into distinct scenes. Only return a JSON array of strings (e.g., [\\"scene 1\\", \\"scene 2\\"]) containing the list of scenes. Do not explain anything.\\n\\nQuery: {text}"
            elif self.mode == "audio":
                p = f"This is a video search query. If the query mentions someone speaking, saying, or singing something (e.g. 'a man says hello', 'someone yells stop'), extract ONLY the exact spoken dialogue/speech. Return it as a JSON array of strings (e.g., [\\"hello\\"]). If there is no speech mentioned, return an empty JSON array []. Do not explain anything.\\n\\nQuery: {text}"
            elif self.mode == "route":
                p = f"This is a video search query. Decide if it requires searching by Visual content (actions, objects, scenes), Audio content (someone speaking or specific sounds), or both. Return ONLY a valid JSON object with boolean values. Example: {{\\"visual\\": true, \\"audio\\": false}} or {{\\"visual\\": true, \\"audio\\": true}} or {{\\"visual\\": false, \\"audio\\": true}}. Do not explain anything.\\n\\nQuery: {text}"
            else:
                p = f"For this sentence, extract the important phrases (noun phrases/actions) within it. Only return a JSON array of strings (e.g., [\\"phrase 1\\", \\"phrase 2\\"]) containing the list of extracted phrases. Do not explain anything.\\n\\nQuery: {text}"
            prompts.append(p)
            
        messages_batch = [[
            {"role": "system", "content": "You are a helpful assistant that strictly outputs valid JSON. Do not use markdown blocks."},
            {"role": "user", "content": p}
        ] for p in prompts]
        
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        text_inputs = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_batch]
        inputs = self.tokenizer(text_inputs, return_tensors="pt", padding=True).to(self.model.device)
        
        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.01,
                do_sample=False
            )
            
        responses = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        import json
        for i, response in enumerate(responses):
            try:
                start = response.find('{') if self.mode == "route" else response.find('[')
                end = (response.rfind('}') + 1) if self.mode == "route" else (response.rfind(']') + 1)
                
                if start != -1 and end != 0:
                    json_str = response[start:end]
                    result = json.loads(json_str)
                    
                    if self.mode == "route":
                        if isinstance(result, dict) and "visual" in result and "audio" in result:
                            self.cache[self.mode][uncached_texts[i]] = result
                            results[uncached_indices[i]] = result
                    else:
                        if isinstance(result, list) and all(isinstance(x, str) for x in result):
                            cleaned = [x.strip() for x in result if x.strip()]
                            if cleaned:
                                self.cache[self.mode][uncached_texts[i]] = cleaned
                                results[uncached_indices[i]] = cleaned
            except Exception:
                pass
                
            # Default fallback if parsing fails
            if results[uncached_indices[i]] is None:
                if self.mode == "route":
                    results[uncached_indices[i]] = {"visual": True, "audio": True}
                else:
                    results[uncached_indices[i]] = []
                    
        self._save_cache()
        return results
"""

content = content + "\n" + batch_func
with open("pipeline/retrieval/segmentation/lm_segmenter.py", "w") as f:
    f.write(content)
print("Updated lm_segmenter.py")
