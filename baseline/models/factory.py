from .ocr.florence2_ocr_model import Florence2OCRModel

def get_ocr_model(engine: str, **kwargs):
    engine = engine.lower()
    if engine == 'florence2':
        return Florence2OCRModel(**kwargs)
    elif engine == 'florence2_base':
        # Reuse Florence2OCRModel but point to the base version
        return Florence2OCRModel(model_name='microsoft/Florence-2-base', **{k: v for k, v in kwargs.items() if k != 'model_name'})
    elif engine == 'easyocr':
        from .ocr.easyocr_model import EasyOCRModel
        return EasyOCRModel(**kwargs)
    elif engine == 'rapidocr':
        from .ocr.rapidocr_model import RapidOCRModel
        return RapidOCRModel(**kwargs)
    else:
        raise ValueError(f"Unknown OCR engine: {engine}. Available: florence2, florence2_base, easyocr, rapidocr")

def get_od_model(engine: str, **kwargs):
    engine = engine.lower()
    if engine == 'yolo':
        from .object_detection.yolo_model import YOLOModel
        return YOLOModel(**kwargs)
    elif engine == 'florence2':
        from .object_detection.florence2_od_model import Florence2ODModel
        return Florence2ODModel(**kwargs)
    elif engine == 'florence2_base':
        from .object_detection.florence2_od_model import Florence2ODModel
        return Florence2ODModel(model_name='microsoft/Florence-2-base', **{k: v for k, v in kwargs.items() if k != 'model_name'})
    else:
        raise ValueError(f"Unknown OD engine: {engine}. Available: yolo, florence2, florence2_base")

def get_caption_model(engine: str, **kwargs):
    engine = engine.lower()
    if engine == 'florence2':
        from .captioning.florence2_caption_model import Florence2CaptionModel
        return Florence2CaptionModel(**kwargs)
    elif engine == 'florence2_base':
        from .captioning.florence2_caption_model import Florence2CaptionModel
        return Florence2CaptionModel(model_name='microsoft/Florence-2-base', **{k: v for k, v in kwargs.items() if k != 'model_name'})
    elif engine == 'blip':
        from .captioning.blip_caption_model import BlipCaptionModel
        return BlipCaptionModel(**kwargs)
    else:
        raise ValueError(f"Unknown Caption engine: {engine}. Available: florence2, florence2_base, blip")
