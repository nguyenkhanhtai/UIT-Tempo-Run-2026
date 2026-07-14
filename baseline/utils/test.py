import argparse
import sys
import os
import glob
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.factory import get_ocr_model, get_od_model, get_caption_model

def main():
    parser = argparse.ArgumentParser(description="Test OCR and YOLO models locally")
    parser.add_argument("--task", choices=["ocr", "od", "caption"], required=True, help="Task type (ocr, od, or caption)")
    parser.add_argument("--engine", required=True, help="Engine (e.g. transformers, yolo, easyocr)")
    parser.add_argument("--model", default=None, help="Model name or weights path")
    parser.add_argument("--images", nargs='+', required=True, help="Path(s) or glob pattern(s) to input images")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    # Expand globs and gather all valid image paths
    image_paths = []
    for pattern in args.images:
        for path in glob.glob(pattern):
            if os.path.isfile(path):
                image_paths.append(path)
                
    if not image_paths:
        print("Error: No images found matching the provided paths/patterns.")
        sys.exit(1)
        
    imgs = []
    valid_paths = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(img)
            valid_paths.append(p)
        except Exception as e:
            print(f"Failed to open image '{p}': {e}")
            
    if not imgs:
        print("Error: No valid images could be loaded.")
        sys.exit(1)
    
    if args.task == "ocr":
        print(f"[INIT] Loading OCR ({args.engine}: {args.model})...")
        ocr = get_ocr_model(engine=args.engine, model_name=args.model, device=args.device, use_gpu='cuda' in args.device)
        print(f"[RUNNING] Extracting OCR features for {len(imgs)} images...")
        results = ocr.extract(imgs)
        
        # Free RAM/VRAM immediately after extraction
        del ocr
        import gc; gc.collect()
        import torch; torch.cuda.empty_cache()
        
        print("\n=== OCR RESULTS ===")
        for path, res in zip(valid_paths, results):
            print(f"\n--- {os.path.basename(path)} ---")
            print(res)
        print("\n===================")
    elif args.task == "od":
        print(f"[INIT] Loading OD ({args.engine}: {args.model})...")
        od = get_od_model(engine=args.engine, model_name=args.model, device=args.device)
        print(f"[RUNNING] Extracting Objects for {len(imgs)} images...")
        results = od.extract(imgs)
        
        # Free RAM/VRAM immediately after extraction
        del od
        import gc; gc.collect()
        import torch; torch.cuda.empty_cache()
        
        print("\n=== OBJECT DETECTION RESULTS ===")
        for path, res in zip(valid_paths, results):
            print(f"\n--- {os.path.basename(path)} ---")
            if res:
                from collections import Counter
                counts = Counter(res)
                for k, v in counts.items():
                    print(f" - {k}: {v}")
            else:
                print("No objects found.")
        print("\n================================")
    elif args.task == "caption":
        print(f"[INIT] Loading Caption ({args.engine}: {args.model})...")
        caption_model = get_caption_model(engine=args.engine, model_name=args.model, device=args.device)
        print(f"[RUNNING] Extracting Captions for {len(imgs)} images...")
        results = caption_model.extract(imgs)
        
        # Free RAM/VRAM immediately after extraction
        del caption_model
        import gc; gc.collect()
        import torch; torch.cuda.empty_cache()
        
        print("\n=== CAPTIONING RESULTS ===")
        for path, res in zip(valid_paths, results):
            print(f"\n--- {os.path.basename(path)} ---")
            print(res)
        print("\n==========================")

if __name__ == "__main__":
    main()
