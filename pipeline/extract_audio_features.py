"""
Builds audio_features.npy aligning frame_ms from a visual index 
with Whisper transcript JSON files.
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

def load_transcripts(metadata_dir):
    """Load all whisper_transcripts*.json into a unified dict."""
    transcripts = {}
    files = glob.glob(os.path.join(metadata_dir, "whisper_transcripts*.json"))
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
                # merge
                transcripts.update(data)
        except Exception as e:
            print(f"[extract_audio] Failed to load {f}: {e}")
    return transcripts

def map_frames_to_text(vids, ts, transcripts):
    """
    Given arrays of vids and ts (in ms), return a list of texts.
    If frame_ms falls within a speech segment [start, end], assign its text.
    Otherwise, return "".
    """
    texts = []
    
    # Process video by video for faster mapping
    current_vid = None
    vid_segments = []
    
    for i in range(len(vids)):
        vid = vids[i]
        t = ts[i] / 1000.0  # convert to seconds
        
        if vid != current_vid:
            current_vid = vid
            vid_clean = vid.replace("v3c1_", "").replace("v3c2_", "")
            vid_segments = transcripts.get(vid_clean, [])
            
        mapped_text = ""
        # Find the segment covering 't'
        # Segments look like: {"start": 0.5, "end": 2.3, "text": " hello"}
        for seg in vid_segments:
            if seg["start"] <= t <= seg["end"]:
                mapped_text = seg["text"].strip()
                break
                
        texts.append(mapped_text)
        
    return texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kf-dir", required=True, help="Path to keyframes directory (e.g., keyframes/fps/1)")
    ap.add_argument("--metadata-dir", default="artifacts/metadata", help="Path to folder containing whisper_transcripts*.json")
    ap.add_argument("--out-dir", required=True, help="Directory to save audio_features.npy")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="HuggingFace text embedding model")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    print(f"[extract_audio] Loading visual keyframes timeline from {args.kf_dir}...")
    vids = []
    ts = []
    
    # Load vids and ts from ts_ms.npy just like extract_embed.py
    import glob
    vdirs = sorted(glob.glob(os.path.join(args.kf_dir, "*", "ts_ms.npy")))
    for np_file in vdirs:
        vid = os.path.basename(os.path.dirname(np_file))
        timestamps = np.load(np_file)
        vids.extend([vid] * len(timestamps))
        ts.extend(timestamps.tolist())
        
    print(f"[extract_audio] Found {len(vids)} frames across {len(vdirs)} videos.")
    
    print(f"[extract_audio] Loading Whisper transcripts from {args.metadata_dir}...")
    transcripts = load_transcripts(args.metadata_dir)
    print(f"[extract_audio] Loaded transcripts for {len(transcripts)} videos.")
    
    print("[extract_audio] Mapping frame_ms to speech segments...")
    t0 = time.time()
    texts = map_frames_to_text(vids, ts, transcripts)
    print(f"[extract_audio] Mapped {len(texts)} frames in {time.time()-t0:.2f}s")
    
    # Calculate coverage
    empty_cnt = sum(1 for t in texts if not t)
    print(f"[extract_audio] {len(texts) - empty_cnt}/{len(texts)} frames have speech ({((len(texts)-empty_cnt)/len(texts))*100:.1f}%)")
    
    # Load embedding model
    print(f"[extract_audio] Loading text embedding model {args.model}...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("[extract_audio] Please run: uv pip install sentence-transformers")
        
    model = SentenceTransformer(args.model, device=args.device)
    
    print(f"[extract_audio] Encoding texts (Batch Size: {args.batch_size})...")
    t1 = time.time()
    
    # Encode
    # Normalize embeddings for cosine similarity via dot product later
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
    
    print(f"[extract_audio] Encoded in {time.time()-t1:.2f}s. Shape: {embeddings.shape}")
    
    # Save
    model_safe = args.model.replace("/", "_")
    audio_shard_dir = os.path.join(args.out_dir, model_safe)
    os.makedirs(audio_shard_dir, exist_ok=True)
    
    print(f"[extract_audio] Saving .npz files per video to {audio_shard_dir}...")
    ts = np.array(ts, dtype=np.int32)
    current_vid = None
    start_idx = 0
    for i in range(len(vids)):
        if vids[i] != current_vid:
            if current_vid is not None:
                out_npz = os.path.join(audio_shard_dir, f"{current_vid}.npz")
                np.savez(out_npz, emb=embeddings[start_idx:i].astype(np.float16), ts_ms=ts[start_idx:i].astype(np.int32))
            current_vid = vids[i]
            start_idx = i
            
    # save the last video
    if current_vid is not None:
        out_npz = os.path.join(audio_shard_dir, f"{current_vid}.npz")
        np.savez(out_npz, emb=embeddings[start_idx:].astype(np.float16), ts_ms=ts[start_idx:].astype(np.int32))
        
    print(f"[extract_audio] Saved audio embeddings to {audio_shard_dir}")

if __name__ == "__main__":
    main()
