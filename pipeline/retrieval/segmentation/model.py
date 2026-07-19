import os
import re

class BaseSegmenter:
    def segment(self, text):
        raise NotImplementedError

class SceneSegmenter(BaseSegmenter):
    def __init__(self):
        self.split_pattern = re.compile(r'[.!?;]|\b(?:then|while|after|before|followed by|the scene shifts to|the scene shifts|scene shifts|the scene cuts to|cuts to|next|later)\b', flags=re.IGNORECASE)
        print("[segmenter] Using Regex Scene Segmenter.")

    def segment(self, text):
        chunks = self.split_pattern.split(text)
        cleaned_chunks = [c.strip() for c in chunks if c.strip() and len(c.strip()) > 3]
        return cleaned_chunks if cleaned_chunks else [text]

class ObjectSegmenter(BaseSegmenter):
    def __init__(self):
        self.split_pattern = re.compile(r',|\b(?:and|with|and then)\b', flags=re.IGNORECASE)
        print("[segmenter] Using Regex Object Segmenter.")

    def segment(self, text):
        chunks = self.split_pattern.split(text)
        cleaned_chunks = [c.strip() for c in chunks if c.strip() and len(c.strip()) > 2]
        return cleaned_chunks if cleaned_chunks else [text]

class SaTSegmenter(BaseSegmenter):
    def __init__(self, model_name="sat-3l-sm"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from wtpsplit import SaT
            print(f"[segmenter] Loading SaT model '{self.model_name}' for semantic chunking...")
            self.model = SaT(self.model_name)
        except ImportError:
            print("[segmenter] wtpsplit not installed. Text segmenter will be disabled.")
            self.model = None

    def segment(self, text):
        if self.model is None:
            return [text]
            
        try:
            chunks = self.model.split(text, threshold=0.01)
            return [c.strip() for c in chunks if c.strip()]
        except Exception as e:
            print(f"[segmenter] Error splitting text: {e}")
            return [text]

class SpacySegmenter(BaseSegmenter):
    def __init__(self):
        self.nlp = None
        self._load_model()
        
    def _load_model(self):
        try:
            import spacy
            print("[segmenter] Loading spaCy model 'en_core_web_sm' for syntactic chunking...")
            # Disable NER and Lemmatizer to speed up parsing
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
        except ImportError:
            print("[segmenter] spaCy is not installed. Run: uv pip install spacy && uv run python -m spacy download en_core_web_sm")
        except OSError:
            print("[segmenter] spaCy model not found. Run: uv run python -m spacy download en_core_web_sm")
            
    def segment(self, text):
        if self.nlp is None:
            return [text]
            
        doc = self.nlp(text)
        
        # A simple syntactic chunker: split on CCONJ (and, but, or) and SCONJ (while, before, after)
        # or punctuation to form sub-clauses
        chunks = []
        current_chunk = []
        
        for token in doc:
            if token.pos_ in ("CCONJ", "SCONJ", "PUNCT"):
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                # Optionally keep the conjunction/punctuation as part of the next chunk
                # if we wanted to, but usually we discard them for pure semantic matching
            else:
                current_chunk.append(token.text)
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        cleaned_chunks = [c.strip() for c in chunks if c.strip() and len(c.strip()) > 3]
        return cleaned_chunks if cleaned_chunks else [text]

class SceneGraphSegmenter(BaseSegmenter):
    def __init__(self):
        self.parser = None
        self._load_model()
        
    def _load_model(self):
        try:
            import sng_parser
            self.parser = sng_parser
            print("[segmenter] Loading sng_parser for Scene Graph object extraction...")
        except ImportError:
            print("[segmenter] sng_parser is not installed. Run: uv pip install sng_parser")
            
    def segment(self, text):
        if self.parser is None:
            return [text]
            
        try:
            graph = self.parser.parse(text)
            chunks = []
            for entity in graph['entities']:
                span = entity.get('span', '')
                if span:
                    chunks.append(span)
            
            cleaned_chunks = [c.strip() for c in chunks if c.strip() and len(c.strip()) > 2]
            return cleaned_chunks if cleaned_chunks else [text]
        except Exception as e:
            print(f"[segmenter] Error parsing Scene Graph: {e}")
            return [text]

class BertSRLSegmenter(BaseSegmenter):
    def __init__(self):
        print("[segmenter] Khởi tạo BERT_SRL Segmenter (Dùng HuggingFace transformers)...")
        try:
            from transformers import pipeline
            self.pipe = pipeline("token-classification", model="vblagoje/bert-english-uncased-finetuned-pos", aggregation_strategy="simple")
        except Exception as e:
            print(f"[segmenter] Lỗi tải transformers: {e}. Vui lòng chạy: uv pip install transformers")
            self.pipe = None

    def segment(self, text):
        if self.pipe is None:
            import re
            chunks = re.split(r'\b(?:while|and then|before|after)\b', text, flags=re.IGNORECASE)
            cleaned = [c.strip() for c in chunks if c.strip()]
            return cleaned if cleaned else [text]
            
        try:
            preds = self.pipe(text)
            chunks = []
            last_idx = 0
            for p in preds:
                if p['entity_group'] in ['CCONJ', 'SCONJ'] and p['start'] > last_idx:
                    chunk = text[last_idx:p['start']].strip()
                    if len(chunk) > 3:
                        chunks.append(chunk)
                    last_idx = p['start']
            
            final_chunk = text[last_idx:].strip()
            if len(final_chunk) > 3:
                chunks.append(final_chunk)
                
            return chunks if chunks else [text]
        except Exception as e:
            print(f"[segmenter] Lỗi khi chạy BertSRLSegmenter: {e}")
            return [text]

class SpacySceneSegmenter(BaseSegmenter):
    def __init__(self):
        print("[segmenter] Khởi tạo Spacy Scene Segmenter (Dependency Parsing)...")
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            print("[segmenter] Đang tải mô hình spacy en_core_web_sm...")
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            import spacy
            self.nlp = spacy.load("en_core_web_sm")

    def segment(self, text):
        doc = self.nlp(text)
        scenes = []
        
        # Bước 1: Tìm các Hạt nhân (Clause Heads) đại diện cho các hành động/sự kiện độc lập.
        # ROOT: Động từ chính, advcl: Mệnh đề trạng ngữ (while/when...), conj: Mệnh đề nối (and...), 
        # relcl: Mệnh đề quan hệ (who/which...), ccomp/xcomp: Mệnh đề bổ ngữ
        clause_dep = {"ROOT", "advcl", "conj", "relcl", "ccomp", "xcomp"}
        clause_heads = set()
        
        for token in doc:
            if token.dep_ in clause_dep:
                # Đảm bảo nó là một động từ (hoặc ROOT có thể là tính từ/danh từ)
                if token.pos_ == "VERB" or token.dep_ == "ROOT":
                    clause_heads.add(token.i)
                    
        # Nếu không tìm thấy cấu trúc sự kiện nào, trả về câu gốc
        if not clause_heads:
            return [text]
            
        # Bước 2: Gán mỗi từ trong câu vào Hạt nhân gần nhất trên cây phụ thuộc (Nearest Ancestor)
        groups = {h: [] for h in clause_heads}
        for token in doc:
            curr = token
            assigned_head = None
            while curr is not None:
                if curr.i in clause_heads:
                    assigned_head = curr.i
                    break
                if curr.head == curr:  # Đã lên tới ROOT
                    break
                curr = curr.head
                
            if assigned_head is not None:
                groups[assigned_head].append(token)
                
        # Bước 3: Tái tạo văn bản cho từng Scene (đảm bảo giữ nguyên trật tự từ)
        import re
        valid_scenes = []
        for head_idx in sorted(groups.keys(), key=lambda k: groups[k][0].i if groups[k] else -1):
            tokens = sorted(groups[head_idx], key=lambda t: t.i)
            # Dùng text_with_ws để ghép lại thành câu hoàn chỉnh
            clause_text = "".join(t.text_with_ws for t in tokens).strip()
            # Dọn dẹp khoảng trắng thừa và dấu câu
            clause_text = re.sub(r'\s+', ' ', clause_text)
            clause_text = clause_text.strip(" ,.;")
            if len(clause_text) > 3:
                valid_scenes.append(clause_text)
                
        return valid_scenes if valid_scenes else [text]

def get_segmenters(scene_engine="regex", object_engine="regex"):
    scene_engine = scene_engine.lower() if scene_engine else "regex"
    object_engine = object_engine.lower() if object_engine else "regex"
    
    if scene_engine == "bert_srl":
        scene_seg = BertSRLSegmenter()
    elif scene_engine == "spacy":
        scene_seg = SpacySceneSegmenter()
    else:
        scene_seg = SceneSegmenter()
    
    if object_engine == "none" or object_engine == "":
        return scene_seg, None
        
    if object_engine == "spacy":
        obj_seg = SpacySegmenter()
    elif object_engine == "sat":
        obj_seg = SaTSegmenter()
    elif object_engine == "scenegraph":
        obj_seg = SceneGraphSegmenter()
    else:
        obj_seg = ObjectSegmenter()
        
    return scene_seg, obj_seg
