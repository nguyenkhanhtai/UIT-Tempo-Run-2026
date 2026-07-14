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
    import numpy as np
    files = sorted(glob.glob(os.path.join(vdir, "*.jpg")))
    ts_path = os.path.join(vdir, "ts_ms.npy")
    if not os.path.exists(ts_path):
        return [], []
    ts_array = np.load(ts_path)
    n = min(len(files), len(ts_array))
    
    imgs = []
    ts = []
    for i in range(n):
        try:
            f = files[i]
            im = Image.open(f).convert("RGB")
            # Force load so we can catch truncation errors
            im.load()
            imgs.append(im)
            ts.append(int(ts_array[i]))
        except Exception as e:
            print(f"[ERROR] Failed to load frame {f}: {e}", flush=True)
            continue
    return imgs, ts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", required=True, help="dir produced by extract_keyframes.py")
    ap.add_argument("--out", required=True, help="dir to save metadata")
    ap.add_argument("--task", choices=["ocr", "od", "caption", "all"], default="all")
    ap.add_argument("--ocr-engine", default="easyocr", help="OCR engine to use")
    ap.add_argument("--ocr-model", default=None, help="OCR model checkpoint (if applicable)")
    ap.add_argument("--od-engine", default="yolo", help="OD engine to use")
    ap.add_argument("--od-model", default="yolov8x.pt", help="OD model checkpoint")
    ap.add_argument("--caption-engine", default="florence2", help="Caption engine to use")
    ap.add_argument("--caption-model", default="microsoft/Florence-2-large", help="Caption model checkpoint")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.task == "ocr":
        meta_dir = Path(args.out) / "metadata" / "ocr"
    elif args.task == "od":
        meta_dir = Path(args.out) / "metadata" / "od"
    elif args.task == "caption":
        meta_dir = Path(args.out) / "metadata" / "caption"
    else:
        meta_dir = Path(args.out) / "metadata"
        
    meta_dir.mkdir(parents=True, exist_ok=True)
    fail_log = Path(args.out) / f"failed_metadata_{args.task}_shard{args.shard_index}.txt"

    vdirs = list_keyframe_dirs(args.keyframes)
    if not vdirs:
        raise SystemExit(f"no keyframes in {args.keyframes} — run extract_keyframes.py first")
    
    mine = [d for i, d in enumerate(vdirs) if i % args.shard_count == args.shard_index]
    if args.limit:
        mine = mine[:args.limit]
    
    print(f"[shard {args.shard_index}/{args.shard_count}] Task: {args.task}, {len(mine)}/{len(vdirs)} videos", flush=True)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from baseline.models.factory import get_ocr_model, get_od_model, get_caption_model

    ocr = None
    od = None
    caption_model = None
    if args.task in ["ocr", "all"]:
        print(f"[init] Loading OCR ({args.ocr_engine}: {args.ocr_model})...", flush=True)
        ocr = get_ocr_model(args.ocr_engine, model_name=args.ocr_model, device=args.device, use_gpu='cuda' in args.device)
    if args.task in ["od", "all"]:
        print(f"[init] Loading OD ({args.od_engine}: {args.od_model})...", flush=True)
        od = get_od_model(args.od_engine, model_name=args.od_model, device=args.device)
    if args.task in ["caption", "all"]:
        print(f"[init] Loading Captioning ({args.caption_engine}: {args.caption_model})...", flush=True)
        caption_model = get_caption_model(args.caption_engine, model_name=args.caption_model, device=args.device)

    t0 = time.time(); done = nframes = failed = 0
    
    from collections import defaultdict
    frames_expected = {}
    frames_processed = defaultdict(int)
    pending_results = defaultdict(lambda: {"ts": [], "ocr": [], "od": [], "cap": []})
    global_batch = [] # list of (vid, t, img)
    
    def process_global_batch(batch):
        nonlocal done, failed
        if not batch: return
        vids = [b[0] for b in batch]
        ts = [b[1] for b in batch]
        imgs = [b[2] for b in batch]
        
        ocr_res = ocr.extract(imgs, batch_size=args.batch_size) if ocr else [None]*len(imgs)
        od_res = od.extract(imgs, batch_size=args.batch_size) if od else [None]*len(imgs)
        cap_res = caption_model.extract(imgs, batch_size=args.batch_size) if caption_model else [None]*len(imgs)
        
        for i in range(len(batch)):
            vid = vids[i]
            pending_results[vid]["ts"].append(ts[i])
            pending_results[vid]["ocr"].append(ocr_res[i])
            pending_results[vid]["od"].append(od_res[i])
            pending_results[vid]["cap"].append(cap_res[i])
            frames_processed[vid] += 1
            
            # If video is fully processed, write to disk
            if frames_processed[vid] == frames_expected[vid]:
                try:
                    out_jsonl = meta_dir / f"{vid}.jsonl"
                    log_lines = []
                    with open(out_jsonl, 'w', encoding='utf-8') as f:
                        for j in range(frames_expected[vid]):
                            t_ms = pending_results[vid]["ts"][j]
                            data = {"ts_ms": t_ms, "shard_index": args.shard_index}
                            
                            o = pending_results[vid]["ocr"][j]
                            if o is not None:
                                data["ocr"] = o.lower()
                                if data["ocr"].strip():
                                    log_lines.append(f"[{vid}] {t_ms}ms | OCR: {data['ocr']}")
                                    
                            d = pending_results[vid]["od"][j]
                            if d is not None:
                                data["objects"] = [obj.lower() for obj in d]
                                if data["objects"]:
                                    log_lines.append(f"[{vid}] {t_ms}ms | OD: {data['objects']}")
                                    
                            c = pending_results[vid]["cap"][j]
                            if c is not None:
                                data["caption"] = c.lower()
                                if data["caption"].strip():
                                    log_lines.append(f"[{vid}] {t_ms}ms | Caption: {data['caption']}")
                                    
                            f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    
                    if log_lines:
                        os.makedirs("logs", exist_ok=True)
                        with open(f"logs/results_{args.task}_shard_{args.shard_index}.log", "a", encoding="utf-8") as lf:
                            lf.write("\n".join(log_lines) + "\n")
                            
                    done += 1
                    if done % 10 == 0:
                        info_str = f"Task: {args.task}"
                        if args.task in ["ocr", "all"]:
                            info_str += f" | OCR: {args.ocr_engine}({args.ocr_model})"
                        if args.task in ["od", "all"]:
                            info_str += f" | OD: {args.od_engine}({args.od_model})"
                        if args.task in ["caption", "all"]:
                            info_str += f" | Cap: {args.caption_engine}({args.caption_model})"
                        elapsed = time.time() - t0
                        eta_seconds = (elapsed / done) * (len(mine) - done)
                        eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m {int(eta_seconds % 60)}s" if eta_seconds >= 3600 else f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                        print(f"  [{info_str}] done {done}/{len(mine)} videos in {elapsed:.0f}s | ETA: {eta_str}", flush=True)
                except Exception as e:
                    failed += 1
                    with open(fail_log, "a") as fl:
                        fl.write(f"{vid}\t{str(e)}\n")
                    print(f"  ERROR saving {vid}: {e}")
                
                # Cleanup state for this vid
                del pending_results[vid]
                del frames_expected[vid]
                del frames_processed[vid]

        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            frames_expected[vid] = len(imgs)
            
            for i in range(len(imgs)):
                global_batch.append((vid, ts[i], imgs[i]))
                if len(global_batch) >= args.batch_size:
                    process_global_batch(global_batch)
                    global_batch = []
                    
        except Exception as e:
            failed += 1
            with open(fail_log, "a") as fl:
                fl.write(f"{vid}\t{str(e)}\n")
            print(f"  ERROR {vid}: {e}")
            
    # Process any remaining frames in buffer
    if global_batch:
        process_global_batch(global_batch)
        global_batch = []
            
    print(f"Finished {done} videos ({nframes} frames), {failed} errors in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
