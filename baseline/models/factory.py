import importlib

# Registry of available OCR models: mapping model_name -> (module_path, class_name)
OCR_MODELS = {
    "easyocr": ("models.ocr.easyocr_model", "EasyOCRModel"),
    "transformers": ("models.ocr.transformers_ocr_model", "TransformersOCRModel"),
}

# Registry of available Object Detection models
OD_MODELS = {
    "yolo": ("models.object_detection.yolo_model", "YOLOModel"),
}

def get_ocr_model(model_name: str, **kwargs):
    """
    Factory method to instantiate an OCR model dynamically.
    Avoids long if-else chains and lazy-loads heavy dependencies (like PyTorch/Transformers).
    """
    if model_name not in OCR_MODELS:
        raise ValueError(f"Unknown OCR model '{model_name}'. Available: {list(OCR_MODELS.keys())}")
        
    module_path, class_name = OCR_MODELS[model_name]
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class(**kwargs)

def get_od_model(model_name: str, **kwargs):
    """
    Factory method to instantiate an Object Detection model dynamically.
    """
    if model_name not in OD_MODELS:
        raise ValueError(f"Unknown Object Detection model '{model_name}'. Available: {list(OD_MODELS.keys())}")
        
    module_path, class_name = OD_MODELS[model_name]
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class(**kwargs)
