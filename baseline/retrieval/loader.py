"""
Purpose: Loads precomputed .npz video feature shards and .jsonl metadata
for the retrieval pipeline.
"""
import glob
import os
import json
from pathlib import Path
import numpy as np

def load_index(shard_dir, meta_dir=None):
    embs, vids, ts = [], [], []
    metadata = {}
    
    if meta_dir and os.path.exists(meta_dir):
        print(f"[index] Loading metadata from {meta_dir}...", flush=True)
        
        def load_jsonl_files(directory):
            if not os.path.exists(directory):
                return
            for f in glob.glob(os.path.join(directory, "*.jsonl")):
                vid = Path(f).stem
                if vid not in metadata:
                    metadata[vid] = {}
                with open(f, 'r', encoding='utf-8') as jf:
                    for line in jf:
                        try:
                            data = json.loads(line)
                            t = data["ts_ms"]
                            if t not in metadata[vid]:
                                metadata[vid][t] = {}
                            # Update dictionary so we can merge OCR and OD smoothly
                            metadata[vid][t].update(data)
                        except Exception:
                            continue
                            
        # Load legacy unified metadata (if any)
        load_jsonl_files(meta_dir)
        # Load separated OCR and OD metadata
        load_jsonl_files(os.path.join(meta_dir, "ocr"))
        load_jsonl_files(os.path.join(meta_dir, "od"))

    files = sorted(glob.glob(os.path.join(shard_dir, "*.npz")))
    for f in files:
        d = np.load(f)
        e = d["emb"]
        if e.shape[0] == 0:
            continue
        embs.append(e)
        vid = Path(f).stem
        vids.extend([vid] * e.shape[0])
        ts.append(d["ts_ms"])
    if not embs:
        raise SystemExit(f"no shards in {shard_dir}")
    emb = np.concatenate(embs, 0)            # [N, D] fp16
    ts = np.concatenate(ts, 0).astype(np.int32)
    vids = np.array(vids)
    print(f"[index] {emb.shape[0]} keyframes from {len(files)} videos, dim={emb.shape[1]}", flush=True)
    return emb, vids, ts, metadata
