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

def get_segmenters():
    engine = os.environ.get("SEGMENTER_ENGINE", "regex").lower()
    scene_seg = SceneSegmenter()
    
    if engine == "spacy":
        obj_seg = SpacySegmenter()
    elif engine == "sat":
        obj_seg = SaTSegmenter()
    else:
        obj_seg = ObjectSegmenter()
        
    return scene_seg, obj_seg
