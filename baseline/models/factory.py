from .ocr.vintern_ocr_model import VinternOCRModel
from .ocr.florence2_ocr_model import Florence2OCRModel

def get_ocr_model(engine: str, **kwargs):
    engine = engine.lower()
    if engine == 'vintern':
        return VinternOCRModel(**kwargs)
    elif engine == 'florence2':
        return Florence2OCRModel(**kwargs)
    elif engine == 'florence2_base':
        # Reuse Florence2OCRModel but point to the base version
        return Florence2OCRModel(model_name='microsoft/Florence-2-base', **{k: v for k, v in kwargs.items() if k != 'model_name'})
    elif engine == 'paddleocr':
        from .ocr.paddleocr_model import PaddleOCRModel
        return PaddleOCRModel(**kwargs)
    elif engine == 'easyocr':
        from .ocr.easyocr_model import EasyOCRModel
        return EasyOCRModel(**kwargs)
    else:
        raise ValueError(f"Unknown OCR engine: {engine}. Available: vintern, florence2, florence2_base, paddleocr, easyocr")

def get_od_model(engine: str, **kwargs):
    engine = engine.lower()
    if engine == 'yolo':
        from .object_detection.yolo_model import YOLOModel
        return YOLOModel(**kwargs)
    else:
        raise ValueError(f"Unknown OD engine: {engine}. Available: yolo")
