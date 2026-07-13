from .base_od import BaseOD

class YOLOModel(BaseOD):
    def __init__(self, model_name="yolov8n.pt", device="cuda"):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        if 'cuda' in device:
            self.model.to(device)

    def extract(self, imgs: list) -> list:
        results = []
        # YOLO can process a batch of images
        preds = self.model(imgs, verbose=False)
        for p in preds:
            classes = [p.names[int(c)] for c in p.boxes.cls]
            results.append(classes)
        return results
