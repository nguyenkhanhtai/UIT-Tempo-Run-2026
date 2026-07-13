import importlib

# Registry of available OCR models: mapping model_name -> (module_path, class_name)
OCR_MODELS = {
    "easyocr": ("models.ocr.easyocr_model", "EasyOCRModel"),
    "transformers": ("models.ocr.transformers_ocr_model", "TransformersOCRModel"),
    "florence2": ("models.ocr.florence2_ocr_model", "Florence2OCRModel"),
    "paddleocr": ("models.ocr.paddleocr_model", "PaddleOCRModel"),
}

# Registry of available Object Detection models
OD_MODELS = {
    "yolo": ("models.object_detection.yolo_model", "YOLOModel"),
}

def get_ocr_model(engine: str, **kwargs):
    """
    Factory method to instantiate an OCR model dynamically.
    Avoids long if-else chains and lazy-loads heavy dependencies (like PyTorch/Transformers).
    """
    if engine not in OCR_MODELS:
        raise ValueError(f"Unknown OCR engine '{engine}'. Available: {list(OCR_MODELS.keys())}")
        
    module_path, class_name = OCR_MODELS[engine]
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class(**kwargs)

def get_od_model(engine: str, **kwargs):
    """
    Factory method to instantiate an Object Detection model dynamically.
    """
    if engine not in OD_MODELS:
        raise ValueError(f"Unknown Object Detection engine '{engine}'. Available: {list(OD_MODELS.keys())}")
        
    module_path, class_name = OD_MODELS[engine]
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class(**kwargs)
