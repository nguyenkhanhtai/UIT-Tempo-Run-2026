from .base_od import BaseOD

class YOLOModel(BaseOD):
    def __init__(self, model_name="yolov8x.pt", device="cuda:0", **kwargs):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        if 'cuda' in device:
            self.model.to(device)

    def extract(self, imgs: list, batch_size: int = 32, **kwargs) -> list:
        results = []
        # YOLO can process a batch of images
        preds = self.model(imgs, verbose=False, batch=batch_size)
        for p in preds:
            classes = [p.names[int(c)] for c in p.boxes.cls]
            results.append(classes)
        return results
