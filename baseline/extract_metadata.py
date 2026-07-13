"""Stage 2 — metadata extraction (OCR & Objects) to .jsonl."""
import argparse
import glob
import json
import os
import time
from pathlib import Path
from PIL import Image

def list_keyframe_dirs(keyframes_dir: str) -> list[Path]:
    return sorted([p for p in Path(keyframes_dir).iterdir() if p.is_dir()])

def load_frames(vdir: Path):
    imgs = []
    ts = []
    for f in sorted(glob.glob(os.path.join(vdir, "*.jpg"))):
        try:
            im = Image.open(f).convert("RGB")
            # Force load so we can catch truncation errors
            im.load()
            imgs.append(im)
            ts.append(int(Path(f).stem))
        except Exception:
            continue
    return imgs, ts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", required=True, help="dir produced by extract_keyframes.py")
    ap.add_argument("--out", required=True, help="dir to save metadata")
    ap.add_argument("--ocr-engine", default="easyocr", help="OCR engine to use")
    ap.add_argument("--ocr-model", default=None, help="OCR model checkpoint (if applicable)")
    ap.add_argument("--od-engine", default="yolo", help="OD engine to use")
    ap.add_argument("--od-model", default="yolov8x.pt", help="OD model checkpoint")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    meta_dir = Path(args.out) / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    fail_log = Path(args.out) / f"failed_metadata_shard{args.shard_index}.txt"

    vdirs = list_keyframe_dirs(args.keyframes)
    if not vdirs:
        raise SystemExit(f"no keyframes in {args.keyframes} — run extract_keyframes.py first")
    
    mine = [d for i, d in enumerate(vdirs) if i % args.shard_count == args.shard_index]
    if args.limit:
        mine = mine[:args.limit]
    
    print(f"[shard {args.shard_index}/{args.shard_count}] {len(mine)}/{len(vdirs)} videos", flush=True)

    # Initialize models via factory/imports
    print(f"[init] Loading OCR ({args.ocr_engine}: {args.ocr_model}) and OD ({args.od_engine}: {args.od_model}) models...", flush=True)
    
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from baseline.models.factory import get_ocr_model, get_od_model
    ocr = get_ocr_model(args.ocr_engine, model_name=args.ocr_model, device=args.device, use_gpu='cuda' in args.device)
    od = get_od_model(args.od_engine, model_name=args.od_model, device=args.device)

    t0 = time.time(); done = nframes = failed = 0
    
    for vdir in mine:
        vid = vdir.name
        out_jsonl = meta_dir / f"{vid}.jsonl"
        if out_jsonl.exists():
            done += 1
            continue
            
        try:
            imgs, ts = load_frames(vdir)
            if not imgs:
                continue
                
            nframes += len(imgs)
            
            # Extract metadata
            ocr_results = ocr.extract(imgs)
            od_results = od.extract(imgs)
            
            with open(out_jsonl, 'w', encoding='utf-8') as f:
                for t, ocr_text, objects in zip(ts, ocr_results, od_results):
                    data = {
                        "ts_ms": t,
                        "ocr": ocr_text.lower(),
                        "objects": [obj.lower() for obj in objects]
                    }
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    
            done += 1
            if done % 10 == 0:
                print(f"  done {done}/{len(mine)} videos in {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            failed += 1
            with open(fail_log, "a") as fl:
                fl.write(f"{vid}\t{str(e)}\n")
            print(f"  ERROR {vid}: {e}")
            
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print(f"Finished {done} videos ({nframes} frames), {failed} errors in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
